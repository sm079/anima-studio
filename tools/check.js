import { AnimaPipeline, fetchManifest } from "../app/pipeline.js";
import { simpleSigmas } from "../app/samplers.js";

const q = new URLSearchParams(location.search);
const base = new URL(q.get("models") || "../models/", location.href);
const dumpDir = new URL(q.get("dump") || "../out/dump_int8/", location.href);
const only = (q.get("only") || "te,ctx,dit,vae").split(",");
const logEl = document.getElementById("log");
const log = (s, cls) => {
  const span = document.createElement("span");
  if (cls) span.className = cls;
  span.textContent = s + "\n";
  logEl.append(span);
  console.log(s);
};
window.__checkDone = false;
window.__checkResults = {};

async function loadDump(name) {
  const r = await fetch(new URL(name + ".bin", dumpDir));
  return new Float32Array(await r.arrayBuffer());
}

function compare(name, got, ref, tol) {
  let maxAbs = 0, num = 0, den = 0;
  for (let i = 0; i < ref.length; i++) {
    const d = got[i] - ref[i];
    maxAbs = Math.max(maxAbs, Math.abs(d));
    num += d * d;
    den += ref[i] * ref[i];
  }
  const rel = Math.sqrt(num / Math.max(den, 1e-30));
  const ok = rel < tol && Number.isFinite(rel) && got.length >= ref.length;
  window.__checkResults[name] = { rel, maxAbs, ok };
  log(`${ok ? "PASS" : "FAIL"} ${name}: rel L2 ${rel.toExponential(3)}  max|d| ${maxAbs.toExponential(3)}  (n=${ref.length}, tol ${tol})`, ok ? "ok" : "bad");
  return ok;
}

try {
  const meta = await (await fetch(new URL("index.json", dumpDir))).json();
  const manifest = await fetchManifest(base);
  const selection = { model: q.get("model") || meta.model || "turbo-v1.1", dit: q.get("dit") || meta.dit || "int8", te: q.get("te") || meta.te || "int8", backend: q.get("backend") || "webgpu", precision: q.get("precision") || undefined };
  log(`selection ${selection.model} dit ${selection.dit} te ${selection.te} backend ${selection.backend}`);
  const pipe = new AnimaPipeline();
  let lastPct = -1;
  await pipe.load(base, manifest, selection, {
    onStatus: (s) => {
      if (s.phase === "download") {
        const pct = Math.floor((100 * s.done) / s.total);
        if (pct !== lastPct) { lastPct = pct; if (pct % 10 === 0) log(`download ${pct}%`); }
      } else if (s.phase === "load" && s.frac === 0) log(`loading ${s.what}`);
    },
  });
  const gpu = pipe.gpu;
  log(`adapter: ${gpu.info.vendor || ""} ${gpu.info.architecture || ""} ${gpu.info.description || ""}; maxBinding ${(gpu.maxBinding / 2 ** 20) | 0} MiB`);

  if (meta.loras?.length) {
    const specs = await Promise.all(meta.loras.map(async (l) => ({ key: l.path, strength: l.strength, blob: await (await fetch(new URL("../" + l.path, dumpDir))).blob() })));
    const report = await pipe.setLoras(specs);
    for (const r of report) log(`lora ${r.key}: ${r.matched}/${r.total} modules applied${r.unsupported ? `, ${r.unsupported} unsupported tensors` : ""}`);
  }

  const tok = pipe.tokenizer.encode(meta.prompt);
  log(`tokens qwen ${JSON.stringify(tok.qwenIds) === JSON.stringify(meta.qwen_ids) ? "match" : "MISMATCH"}, t5 ${JSON.stringify(tok.t5Ids) === JSON.stringify(meta.t5_ids) ? "match" : "MISMATCH"}`);

  let t = performance.now();
  const hidden = await pipe.te.encode(meta.qwen_ids);
  const hv = await gpu.read(hidden);
  log(`text encoder ${(performance.now() - t).toFixed(0)} ms`);
  if (only.includes("te")) compare("te_hidden", hv, await loadDump("te_hidden"), 2e-3);

  t = performance.now();
  const ctx = await pipe.dit.adapt(hidden, meta.t5_ids, null);
  log(`adapter ${(performance.now() - t).toFixed(0)} ms`);
  if (only.includes("ctx")) compare("ctx", await gpu.read(ctx), await loadDump("ctx"), 2e-3);

  const h = meta.height / 8, w = meta.width / 8;
  if (only.includes("dit") && pipe.nn) {
    // WebNN backend (?backend=webnn): same step from the reference context; fp16 math, so a
    // looser tolerance than the fp32 WebGPU engine
    const nn = pipe.nn;
    log(`webnn: compute ${nn.ct}, int8 weights ${nn.caps.int8 ? "in graph" : "expanded"}, constant tensors ${nn.linears()[0].tensors ? "yes" : "no"}`);
    const cond = { ctx: await loadDump("ctx"), Lk: 512 };
    const noise = await loadDump("noise");
    const sigma = simpleSigmas(meta.steps, 3)[0];
    const x = noise.map((v) => v * sigma);
    for (let rep = 0; rep < 3; rep++) {
      t = performance.now();
      const v = await nn.forward(x, h, w, cond, sigma, rep === 0 ? (f) => { if (f === 1) log(`graph built in ${(performance.now() - t).toFixed(0)} ms`); } : null);
      log(`webnn dit forward ${meta.width}x${meta.height}: ${(performance.now() - t).toFixed(0)} ms${rep === 0 ? " (includes graph build + compile)" : ""}`);
      if (rep === 0) compare("denoised0", x.map((xi, i) => xi - sigma * v[i]), await loadDump("denoised0"), Number(q.get("tol")) || (nn.ct === "float16" ? 3e-2 : 5e-3));
    }
  } else if (only.includes("dit")) {
    const refCond = pipe.dit.prepareContext(gpu.fromArray(await loadDump("ctx"), [512, 1024]));
    const noise = await loadDump("noise");
    const sigma = simpleSigmas(meta.steps, 3)[0];
    const x = noise.map((v) => v * sigma);
    for (let rep = 0; rep < 2; rep++) {
      t = performance.now();
      const tc = pipe.dit.timeCond(sigma);
      const refBlock0 = rep === 0 ? await loadDump("dit_block0") : null;
      const v = await pipe.dit.forward(x, h, w, refCond, tc, async (b, xs) => {
        if (b === 0 && refBlock0) compare("dit_block0", await gpu.read(xs), refBlock0, 5e-3);
      });
      tc.release();
      log(`dit forward ${meta.width}x${meta.height}: ${(performance.now() - t).toFixed(0)} ms${rep === 0 ? " (includes shader compile)" : ""}`);
      if (rep === 0) {
        const den = x.map((xi, i) => xi - sigma * v[i]);
        compare("denoised0", den, await loadDump("denoised0"), 5e-3);
      }
    }
    refCond.release();
  }

  if (only.includes("full")) {
    // the whole sampling loop through the real pipeline (euler, same seed, same CFG) vs the reference
    const res = await pipe.generate({ prompt: meta.prompt, negative: meta.negative ?? "", cfg: meta.cfg ?? 1, width: meta.width, height: meta.height, steps: meta.steps, sampler: "euler", seed: meta.seed });
    log(`full generate ${(res.timings.total / 1000).toFixed(1)} s (${(res.timings.perStep / 1000).toFixed(2)} s/step)`);
    compare("latent_final", res.latent, await loadDump("latent_final"), 2e-2);
  }

  if (only.includes("vae")) {
    const lat = await loadDump("latent_final");
    t = performance.now();
    const img = await pipe.vae.decode(lat, h, w);
    log(`vae decode ${(performance.now() - t).toFixed(0)} ms`);
    const ref = await loadDump("image"); // [3, H, W] in [0,1]
    const got = new Float32Array(ref.length);
    const HW = img.width * img.height;
    for (let i = 0; i < HW; i++) for (let c = 0; c < 3; c++) got[c * HW + i] = img.data[i * 4 + c] / 255;
    compare("image", got, ref, 1e-2);
    const cv = document.getElementById("c");
    cv.width = img.width; cv.height = img.height;
    cv.getContext("2d").putImageData(new ImageData(img.data, img.width, img.height), 0, 0);
  }
  log("done");
} catch (e) {
  log("ERROR " + (e.stack || e), "bad");
  window.__checkError = String(e.stack || e);
}
window.__checkDone = true;
