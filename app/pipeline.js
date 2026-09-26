// Anima text-to-image pipeline: download-once assets -> WebGPU models -> sampling -> VAE.

import { GPU } from "./gpu/device.js";
import { SafeTensors } from "./weights.js";
import { cachedFile, requestPersistence } from "./store.js";
import { AnimaTokenizer } from "./prompt.js";
import { Qwen3 } from "./models/qwen3.js";
import { AnimaDiT } from "./models/dit.js";
import { VAEDecoder } from "./models/vae.js";
import { TorchGenerator } from "./rng.js";
import { sample, SCHEDULERS } from "./samplers.js";
import { linearRegistry, loadLora, attachLoras, destroyLora } from "./lora.js";

// ComfyUI Wan21 latent -> RGB preview projection
const RGB_FACTORS = [
  [-0.1299, -0.1692, 0.2932], [0.0671, 0.0406, 0.0442], [0.3568, 0.2548, 0.1747], [0.0372, 0.2344, 0.142],
  [0.0313, 0.0189, -0.0328], [0.0296, -0.0956, -0.0665], [-0.3477, -0.4059, -0.2925], [0.0166, 0.1902, 0.1975],
  [-0.0412, 0.0267, -0.1364], [-0.1293, 0.074, 0.1636], [0.068, 0.3019, 0.1128], [0.0032, 0.0581, 0.0639],
  [-0.1251, 0.0927, 0.1699], [0.006, -0.0633, 0.0005], [0.3477, 0.2275, 0.295], [0.1984, 0.0913, 0.1861],
];
const RGB_BIAS = [-0.1835, -0.0868, -0.336];
const LATENT_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921];
const LATENT_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916];

function latentPreview(latent, h, w) {
  const img = new Uint8ClampedArray(h * w * 4);
  const hw = h * w;
  for (let i = 0; i < hw; i++) {
    for (let c = 0; c < 3; c++) {
      let v = RGB_BIAS[c];
      for (let k = 0; k < 16; k++) v += (latent[k * hw + i] * LATENT_STD[k] + LATENT_MEAN[k]) * RGB_FACTORS[k][c];
      img[i * 4 + c] = ((v + 1) / 2) * 255;
    }
    img[i * 4 + 3] = 255;
  }
  return { data: img, width: w, height: h };
}

export async function fetchManifest(baseUrl) {
  const res = await fetch(new URL("manifest.json", baseUrl), { cache: "no-cache" });
  if (!res.ok) throw new Error(`could not load ${new URL("manifest.json", baseUrl)} (${res.status}). Build the assets with tools/build_assets.py or pass ?models=<url>.`);
  const m = await res.json();
  if (m.version !== 2) throw new Error("models/manifest.json is from an older build; rerun tools/build_assets.py.");
  return m;
}

// Resolve a user selection against manifest v2 into the three files to load.
export function resolveFiles(manifest, { model, dit, te }) {
  const m = manifest.models.find((x) => x.id === model);
  if (!m) throw new Error(`unknown model ${model}`);
  if (!m.dit[dit]) throw new Error(`${model} has no ${dit} build`);
  if (!manifest.te[te]) throw new Error(`no ${te} text encoder build`);
  return { model: m, files: { dit: m.dit[dit], te: manifest.te[te], vae: manifest.vae } };
}

export class AnimaPipeline {
  // gpuOptions: passed to GPU.create (e.g. { profile: true } for tools/profile.html)
  constructor(gpuOptions = {}) {
    this.gpuOptions = gpuOptions;
    this.gpu = null;
    this.loaded = {}; // component -> path of the file currently on the GPU
    this.model = null;
    this.ctxCache = new Map();
    this.loraSpec = [];
    this.loraCache = new Map(); // key -> uploaded LoRA for the current model build
  }

  isLoaded(files) {
    return ["dit", "te", "vae"].every((k) => this.loaded[k] === files[k].path);
  }

  // Downloads (first time only) and loads the selected files; components that are already
  // on the GPU with the same file are kept.
  // token: optional Hugging Face access token, sent only when the files come from huggingface.co
  async load(baseUrl, manifest, selection, { onStatus = () => {}, signal, token = "" } = {}) {
    await requestPersistence();
    this.gpu = this.gpu || (await GPU.create(this.gpuOptions));
    const gpu = this.gpu;
    const headers = token && new URL(baseUrl).hostname === "huggingface.co" ? { Authorization: `Bearer ${token}` } : {};
    const fetchJson = async (p) => (await fetch(new URL(p, baseUrl), { headers })).json();
    this.tokenizer = this.tokenizer || (await AnimaTokenizer.load(fetchJson));
    const { model, files: want } = resolveFiles(manifest, selection);

    const parts = ["te", "vae", "dit"].filter((k) => this.loaded[k] !== want[k].path);
    const total = parts.reduce((a, k) => a + want[k].size, 0);
    const got = Object.fromEntries(parts.map((k) => [k, 0]));
    const files = {};
    for (const k of parts) {
      const f = want[k];
      files[k] = await cachedFile(new URL(f.path, baseUrl).href, f.path, f.size, (done) => {
        got[k] = done;
        onStatus({ phase: "download", file: f.path, done: parts.reduce((a, p) => a + got[p], 0), total });
      }, signal, headers);
    }

    for (const k of parts) this.unloadPart(k);
    // cached prompt encodings depend on the text encoder and the adapter (inside the DiT file)
    if (parts.includes("te") || parts.includes("dit")) this.clearContexts();
    const labels = { te: "text encoder", dit: "diffusion model", vae: "VAE" };
    for (const k of parts) {
      onStatus({ phase: "load", what: labels[k], frac: 0 });
      const progress = (f) => onStatus({ phase: "load", what: labels[k], frac: f });
      if (k === "te") this.te = await Qwen3.load(gpu, await SafeTensors.open(files.te), progress);
      if (k === "dit") this.dit = await AnimaDiT.load(gpu, await SafeTensors.open(files.dit, "model.diffusion_model."), progress);
      if (k === "vae") this.vae = await VAEDecoder.load(gpu, await SafeTensors.open(files.vae));
      this.loaded[k] = want[k].path;
    }
    // LoRA matrices are laid out for a specific model build (A is rotated for quantized layers)
    if (parts.includes("te") || parts.includes("dit")) {
      for (const L of this.loraCache.values()) destroyLora(L);
      this.loraCache.clear();
      if (this.loraSpec.length) {
        onStatus({ phase: "load", what: "LoRAs", frac: 1 });
        await this.applyLoras();
      }
    }
    await gpu.sync();
    this.model = model;
    this.selection = { ...selection };
    onStatus({ phase: "ready" });
  }

  // list: [{ key, blob, strength }] (blob: File/Blob of the .safetensors). Returns a report per
  // LoRA: { key, matched, total, unsupported } so the UI can flag files that don't fit.
  async setLoras(list) {
    this.loraSpec = list;
    return this.applyLoras();
  }

  async applyLoras() {
    if (!this.dit || !this.te) return [];
    const regs = { dit: linearRegistry(this.dit), te: linearRegistry(this.te) };
    const active = [];
    const report = [];
    for (const spec of this.loraSpec) {
      let L = this.loraCache.get(spec.key);
      if (!L) {
        L = await loadLora(this.gpu, spec.blob, regs);
        this.loraCache.set(spec.key, L);
      }
      active.push({ lora: L, strength: spec.strength });
      report.push({ key: spec.key, matched: L.matched, total: L.total, unsupported: L.unsupported });
    }
    // drop files that are no longer used
    for (const [k, L] of this.loraCache) {
      if (!this.loraSpec.some((s) => s.key === k)) { destroyLora(L); this.loraCache.delete(k); }
    }
    attachLoras(regs, active);
    this.clearContexts(); // the adapter and cross-attention K/V depend on the LoRAs
    return report;
  }

  unloadPart(k) {
    // weights are plain GPU buffers; destroy eagerly so a switch doesn't briefly need memory for both
    const destroy = (o) => {
      if (!o || typeof o !== "object") return;
      if (o instanceof GPUBuffer) { o.destroy(); return; }
      for (const v of Array.isArray(o) ? o : Object.values(o)) if (v && typeof v === "object" && v !== this.gpu && !(v instanceof SafeTensors)) destroy(v);
    };
    if (this[k]) destroy(this[k]);
    this[k] = null;
    delete this.loaded[k];
  }

  clearContexts() {
    for (const t of this.ctxCache.values()) t.release();
    this.ctxCache.clear();
  }

  unload() {
    for (const k of ["te", "dit", "vae"]) this.unloadPart(k);
    for (const L of this.loraCache.values()) destroyLora(L);
    this.loraCache.clear();
    this.clearContexts();
    this.gpu?.pool.trim();
    this.model = null;
  }

  // Prompt -> conditioning ready for the DiT (adapter output + per-block cross-attention K/V).
  // Only the most recent prompt is kept: the cached K/V are ~235 MB.
  async encode(text) {
    if (this.ctxCache.has(text)) return this.ctxCache.get(text);
    this.clearContexts();
    const { qwenIds, t5Ids, t5Weights } = this.tokenizer.encode(text);
    const hidden = await this.te.encode(qwenIds);
    const ctx = await this.dit.adapt(hidden, t5Ids, t5Weights);
    hidden.release();
    const cond = this.dit.prepareContext(ctx);
    this.ctxCache.set(text, cond);
    return cond;
  }

  // Anima Turbo is distilled for CFG 1: one DiT pass per step, no negative prompt.
  async generate(opts) {
    const { prompt, width, height, sampler, seed, shift = 3, scheduler = "simple", onProgress = () => {}, onPreview, signal } = opts;
    const gpu = this.gpu;
    const h = height / 8, w = width / 8;
    const t0 = performance.now();
    onProgress({ phase: "encode" });
    const cond = await this.encode(prompt);

    const sigmas = (SCHEDULERS[scheduler] || SCHEDULERS.simple)(opts.steps, shift);
    const steps = sigmas.length - 1; // the beta scheduler can merge duplicate timesteps
    const noise = new TorchGenerator(seed).randn(16 * h * w);
    let x = noise.map((v) => v * sigmas[0]);
    const tSample = performance.now();

    const check = () => { if (signal?.aborted) throw new DOMException("Generation cancelled", "AbortError"); };
    const denoise = async (xin, sigma, i) => {
      const tc = this.dit.timeCond(sigma);
      const v = await this.dit.forward(xin, h, w, cond, tc, async (b) => {
        check();
        onProgress({ phase: "sample", step: i, steps, frac: (i + (b + 1) / 28) / steps });
        if (b % 7 === 6) await gpu.sync(); // keep the queue short so cancel/progress stay responsive
      });
      tc.release();
      const den = new Float32Array(v.length);
      for (let j = 0; j < v.length; j++) den[j] = xin[j] - sigma * v[j];
      return den;
    };

    let image, tVae;
    try {
      x = await sample(sampler, denoise, x, sigmas, {
        seed, shift,
        onStep: (i, xi, den) => { if (onPreview) onPreview(latentPreview(den, h, w), i); },
      });
      check();
      tVae = performance.now();
      onProgress({ phase: "decode", frac: 0 });
      image = await this.vae.decode(x, h, w, (f) => onProgress({ phase: "decode", frac: f }));
    } finally {
      // free pooled activations (the VAE's full-resolution tensors dominate) so idle VRAM is
      // just the weights; also runs after a cancel
      await gpu.sync().catch(() => {});
      gpu.pool.trim();
    }
    const t1 = performance.now();
    return {
      image,
      latent: x,
      timings: { encode: tSample - t0, sample: tVae - tSample, decode: t1 - tVae, total: t1 - t0, perStep: (tVae - tSample) / steps },
    };
  }
}
