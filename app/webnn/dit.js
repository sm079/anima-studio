// Optional WebNN backend for the diffusion transformer (the per-step hot path, ~95% of the time).
//
// The whole DiT step (timestep embedding, adaLN, 28 blocks, final layer) is one static WebNN graph
// per (resolution, prompt length, LoRA set): inputs are the patchified latent, the timestep
// embedding, the prompt context and the rope tables; the output is the velocity. The browser maps
// the graph onto the platform ML stack (e.g. Windows ML / DirectML), which can use fp16 tensor
// cores that the hand-written WebGPU kernels can't reach. The text encoder, LLM adapter and VAE
// keep running on WebGPU.
//
// Numerics: linears and attention run in `ct` (fp16 when supported), the residual stream, norms and
// adaLN stay fp32. ConvRot-quantized weights stay rotated: the graph rotates the activations with
// the same 256-point Hadamard (a [256, 256] matmul). Quantized weights are kept int8 and
// dequantized in the graph when the backend supports it (w4 codes are decoded to their int8 grid
// on load); otherwise they are expanded to `ct` on load (more memory, same math).
//
// LoRAs: side paths y += s * B(A x) are graph constants; their strengths are a graph input, so
// moving a slider doesn't recompile, adding/removing a LoRA does.

import { bf16ToF32 } from "../weights.js";
import { hadamardRows } from "../lora.js";
import { ditRopeAngles } from "../models/common.js";
import { patchify, unpatchify, timestepEmbedding } from "../models/dit.js";
import { createWebNNContext, capabilities } from "./support.js";

const D = 2048;
const HEADS = 16;
const HD = 128;
const ATTN_BUDGET = 256 * 2 ** 20; // bytes per attention score chunk

// MLOperand.shape/dataType were methods in older Chromium builds and are attributes now
const shapeOf = (o) => (typeof o.shape === "function" ? o.shape() : o.shape);
const typeOf = (o) => (typeof o.dataType === "function" ? o.dataType() : o.dataType);

// ------------------------------------------------------------------ weight preparation (CPU)

let toHalf;
if (typeof Float16Array !== "undefined") {
  toHalf = (f32) => new Uint8Array(new Float16Array(f32).buffer);
} else {
  const fb = new Float32Array(1), ub = new Uint32Array(fb.buffer);
  const bits = (v) => {
    fb[0] = v;
    const x = ub[0];
    const sign = (x >>> 16) & 0x8000;
    const e = ((x >>> 23) & 0xff) - 112;
    let m = x & 0x7fffff;
    if (e >= 31) return sign | 0x7c00 | (((x >>> 23) & 0xff) === 0xff && m ? 0x200 : 0);
    if (e <= 0) {
      if (e < -10) return sign;
      m |= 0x800000;
      const shift = 14 - e;
      let h = m >> shift;
      const rem = m & ((1 << shift) - 1), half = 1 << (shift - 1);
      if (rem > half || (rem === half && h & 1)) h++;
      return sign | h;
    }
    let h = (e << 10) | (m >> 13);
    const rem = m & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && h & 1)) h++;
    return sign | h;
  };
  toHalf = (f32) => {
    const u = new Uint16Array(f32.length);
    for (let i = 0; i < f32.length; i++) u[i] = bits(f32[i]);
    return new Uint8Array(u.buffer);
  };
}

// typed data for a constant of type t from float values
const floatData = (f32, t) => (t === "float16" ? toHalf(f32) : f32);

function fp8e4m3(b) {
  const e = (b >> 3) & 15, m = b & 7;
  const v = e === 0 ? m * 0.001953125 : (1 + m * 0.125) * 2 ** (e - 7);
  return b & 128 ? -v : v;
}

// WGSL round(): half to even
function roundEven(x) {
  const r = Math.round(x);
  return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

// One linear as stored in the file: { quant, N, K, q?: Int8Array [N,K], scale?: Float32Array [N],
// f?: Float32Array [N,K], bias? }
async function readLinear(st, base) {
  const w = st.info(base + "weight");
  const bias = st.has(base + "bias") ? await st.f32(base + "bias") : null;
  const N = w.shape[0];
  if (st.has(base + "comfy_quant")) {
    const meta = JSON.parse(new TextDecoder().decode(await st.bytes(base + "comfy_quant")));
    if (!meta.convrot) throw new Error(`${base}: only ConvRot-quantized linears are supported`);
    let scale = await st.f32(base + (meta.format === "asym_w4a8_int8" ? "weight_s_channel" : "weight_scale"));
    if (scale.length === 1) scale = new Float32Array(N).fill(scale[0]);
    if (meta.format === "int8_tensorwise") {
      const u8 = await st.bytes(base + "weight");
      return { quant: true, N, K: w.shape[1], q: new Int8Array(u8.buffer, u8.byteOffset, u8.byteLength), scale, bias };
    }
    if (meta.format === "asym_w4a8_int8") {
      // decode the 4-bit codes onto the int8 grid exactly like the GEMM tile loader
      const K = w.shape[1] * 2;
      const codes = await st.bytes(base + "weight");
      const srel = await st.bytes(base + "weight_s_rel");
      const cb = await st.f32(base + "weight_codebook");
      const q = new Int8Array(N * K);
      for (let g = 0; g < (N * K) / 16; g++) {
        const s = fp8e4m3(srel[g]);
        const lut = cb.map((c) => Math.max(-127, Math.min(127, roundEven(c * s))));
        for (let j = 0; j < 16; j++) {
          const e = g * 16 + j;
          const byte = codes[e >> 1];
          q[e] = lut[e & 1 ? byte >> 4 : byte & 15];
        }
      }
      return { quant: true, N, K, q, scale, bias };
    }
    throw new Error(`unsupported quant format ${meta.format}`);
  }
  const K = w.shape.slice(1).reduce((a, b) => a * b, 1);
  if (w.dtype !== "BF16") throw new Error(`expected bf16 weight for ${base}, got ${w.dtype}`);
  return { quant: false, N, K, f: bf16ToF32(await st.bytes(base + "weight")), bias };
}

// ------------------------------------------------------------------ linear descriptors

// A linear (or several stacked along N, e.g. q/k/v) of the WebNN graph. Registered with the LoRA
// machinery (linearRegistry) like the WebGPU weights, so LoRAs attach to it the same way.
async function linearDesc(st, bases) {
  const infos = bases.map((b) => st.info(b + "weight"));
  const quant = bases.map((b) => st.has(b + "comfy_quant"));
  let K = infos[0].shape.slice(1).reduce((a, b) => a * b, 1);
  if (quant[0]) {
    const meta = JSON.parse(new TextDecoder().decode(await st.bytes(bases[0] + "comfy_quant")));
    if (meta.format === "asym_w4a8_int8") K *= 2; // two 4-bit codes per byte
  }
  // kind only matters for the LoRA code: quantized ("i8"/"w4") layers take rotated inputs
  const desc = { webnn: true, kind: quant[0] ? "i8" : "bf16", N: 0, K, bases, lora: [] };
  const parts = [];
  bases.forEach((b, i) => {
    const n = infos[i].shape[0];
    parts.push({ name: b.replace(/\.$/, ""), off: desc.N, n });
    desc.N += n;
  });
  if (bases.length > 1) desc.parts = parts;
  else desc.name = parts[0].name;
  return desc;
}

// q/k/v can share one matmul only when they take the same (rotated or plain) input
function canStack(st, bases) {
  const q = bases.map((b) => st.has(b + "comfy_quant"));
  return q.every((x) => x === q[0]);
}

// ------------------------------------------------------------------ the backend

const moduleIds = new WeakMap();
let nextModuleId = 1;
const moduleId = (m) => moduleIds.get(m) || (moduleIds.set(m, nextModuleId), nextModuleId++);

export class WebNNDiT {
  // st: SafeTensors of the DiT file (prefix "model.diffusion_model.")
  // precision: "float16" (default, falls back to float32 when unsupported) | "float32"
  static async load(st, { precision = "float16", onProgress } = {}) {
    const m = new WebNNDiT();
    m.st = st;
    m.context = await createWebNNContext();
    m.caps = capabilities(m.context, precision);
    m.ct = m.caps.ct;
    const vec = (name) => st.f32(name);
    const L = (base) => linearDesc(st, [base]);
    const mod = async (p, n) => [await L(p + n + ".1."), await L(p + n + ".2.")];

    m.xEmbed = await L("x_embedder.proj.1.");
    m.t1 = await L("t_embedder.1.linear_1.");
    m.t2 = await L("t_embedder.1.linear_2.");
    m.tNorm = await vec("t_embedding_norm.weight");
    m.blocks = [];
    for (let i = 0; i < 28; i++) {
      const p = `blocks.${i}.`;
      const sp = p + "self_attn.", cp = p + "cross_attn.";
      const qkvBases = ["q_proj.", "k_proj.", "v_proj."].map((s) => sp + s);
      const stacked = canStack(st, qkvBases);
      m.blocks.push({
        modSelf: await mod(p, "adaln_modulation_self_attn"),
        modCross: await mod(p, "adaln_modulation_cross_attn"),
        modMlp: await mod(p, "adaln_modulation_mlp"),
        self: {
          qkv: stacked ? await linearDesc(st, qkvBases) : null,
          q: stacked ? null : await L(qkvBases[0]), k: stacked ? null : await L(qkvBases[1]), v: stacked ? null : await L(qkvBases[2]),
          o: await L(sp + "output_proj."),
          qn: await vec(sp + "q_norm.weight"),
          kn: await vec(sp + "k_norm.weight"),
        },
        cross: {
          q: await L(cp + "q_proj."), k: await L(cp + "k_proj."), v: await L(cp + "v_proj."), o: await L(cp + "output_proj."),
          qn: await vec(cp + "q_norm.weight"),
          kn: await vec(cp + "k_norm.weight"),
        },
        l1: await L(p + "mlp.layer1."),
        l2: await L(p + "mlp.layer2."),
      });
    }
    m.finalMod = await mod("final_layer.", "adaln_modulation");
    m.finalLinear = await L("final_layer.linear.");

    // Keep the prepared weights on the device (MLTensor constants) when the browser supports it,
    // so recompiling for another resolution or LoRA set doesn't re-read and re-convert 2+ GB.
    if (typeof m.context.createConstantTensor === "function") {
      const all = m.linears();
      try {
        for (let i = 0; i < all.length; i++) {
          const d = all[i];
          const w = await m.prepare(d);
          d.tensors = {};
          for (const [k, v] of Object.entries(w)) if (v) d.tensors[k] = await m.context.createConstantTensor(v.desc, v.data);
          onProgress?.((i + 1) / all.length);
        }
      } catch (e) {
        console.warn("WebNN constant tensors unavailable, weights will be read at graph build:", e.message);
        for (const d of all) { for (const t of Object.values(d.tensors || {})) t.destroy?.(); delete d.tensors; }
      }
    }
    return m;
  }

  linears() {
    const out = [this.xEmbed, this.t1, this.t2, ...this.finalMod, this.finalLinear];
    for (const b of this.blocks) {
      out.push(...b.modSelf, ...b.modCross, ...b.modMlp, b.l1, b.l2, b.cross.q, b.cross.k, b.cross.v, b.cross.o, b.self.o);
      out.push(...(b.self.qkv ? [b.self.qkv] : [b.self.q, b.self.k, b.self.v]));
    }
    return out;
  }

  // Graph-ready weight data for a descriptor: { w, scale?, zp?, bias? }, each { desc, data }.
  // w is [K, N] (transposed so the graph does x @ w), int8 + per-column scale or float `ct`.
  async prepare(d) {
    const ct = this.ct;
    const src = [];
    for (const b of d.bases) src.push(await readLinear(this.st, b));
    const { K } = src[0];
    const N = d.N;
    const int8 = this.caps.int8 && src.every((s) => s.quant);
    const w = int8 ? new Int8Array(K * N) : new Float32Array(K * N);
    const scale = int8 ? new Float32Array(N) : null;
    let off = 0;
    for (const s of src) {
      const vals = s.q || s.f;
      for (let n = 0; n < s.N; n++) {
        const r = n * K, c = off + n;
        const sc = s.quant && !int8 ? s.scale[n] : 1;
        for (let k = 0; k < K; k++) w[k * N + c] = vals[r + k] * sc;
        if (int8) scale[c] = s.scale[n];
      }
      off += s.N;
    }
    const out = {
      w: int8 ? { desc: { dataType: "int8", shape: [K, N] }, data: w } : { desc: { dataType: ct, shape: [K, N] }, data: floatData(w, ct) },
    };
    if (int8) {
      out.scale = { desc: { dataType: ct, shape: [1, N] }, data: floatData(scale, ct) };
      out.zp = { desc: { dataType: "int8", shape: [1, N] }, data: new Int8Array(N) };
    }
    if (src.some((s) => s.bias)) {
      const bias = new Float32Array(N);
      off = 0;
      for (const s of src) { if (s.bias) bias.set(s.bias, off); off += s.N; }
      out.bias = { desc: { dataType: ct, shape: [N] }, data: floatData(bias, ct) };
    }
    return out;
  }

  // ---------------------------------------------------------------- graph construction

  loraEntries() {
    const list = [];
    for (const d of this.linears()) for (const e of d.lora || []) list.push({ d, e });
    return list;
  }

  graphKey(Hp, Wp, Lk) {
    const lora = this.loraEntries().map(({ d, e }) => `${d.name || d.parts[0].name}:${e.off}:${moduleId(e.module)}`).join(",");
    return `${Hp}x${Wp}|${Lk}|${this.ct}|${lora}`;
  }

  async build(Hp, Wp, Lk, onProgress) {
    const b = new MLGraphBuilder(this.context);
    const ct = this.ct;
    const L = Hp * Wp;
    const f32 = (arr, shape) => b.constant({ dataType: "float32", shape }, arr);
    const scalars = new Map();
    const scalar = (v, t = "float32") => {
      const key = t + v;
      if (!scalars.has(key)) scalars.set(key, b.constant({ dataType: t, shape: [1] }, floatData(new Float32Array([v]), t)));
      return scalars.get(key);
    };
    const cast = (x, t) => (typeOf(x) === t ? x : b.cast(x, t));
    const silu = (x) => b.mul(x, b.sigmoid(x));

    const input = (name, shape) => b.input(name, { dataType: "float32", shape });
    const patches = input("patches", [L, 68]);
    const temb = input("temb", [1, D]);
    const ctx = input("ctx", [Lk, 1024]);
    const cos = input("cos", [L, 1, 64]);
    const sin = input("sin", [L, 1, 64]);
    const entries = this.loraEntries();
    const loraIn = entries.length ? input("lora", [entries.length]) : null;
    const loraIdx = new Map(entries.map(({ e }, i) => [e, i]));

    let had = null;
    const hadamard = () => {
      if (!had) {
        const h = new Float32Array(256 * 256);
        for (let i = 0; i < 256; i++) h[i * 256 + i] = 1;
        hadamardRows(h, 256, 256); // row i = e_i H, i.e. the matrix itself
        had = b.constant({ dataType: ct, shape: [256, 256] }, floatData(h, ct));
      }
      return had;
    };

    const constOf = (t) => (t.desc ? b.constant(t.desc, t.data) : b.constant(t));
    const weights = async (d) => {
      const w = d.tensors || (await this.prepare(d));
      let W = constOf(w.w);
      if (w.scale) W = b.dequantizeLinear(W, constOf(w.scale), constOf(w.zp));
      return { W, bias: w.bias ? constOf(w.bias) : null };
    };

    // y = x @ W^T (+ LoRAs) (+ bias) (act), in ct
    const lin = async (x, d, act) => {
      let xin = cast(x, ct);
      const [M, K] = shapeOf(xin);
      if (d.kind !== "bf16") xin = b.reshape(b.matmul(b.reshape(xin, [(M * K) / 256, 256]), hadamard()), [M, K]);
      const { W, bias } = await weights(d);
      let y = b.matmul(xin, W);
      for (const e of d.lora || []) {
        const At = new Float32Array(K * e.r); // A [r, K] -> [K, r]
        for (let r = 0; r < e.r; r++) for (let k = 0; k < K; k++) At[k * e.r + r] = e.Adata[r * K + k];
        const Bt = new Float32Array(e.r * e.n); // B [n, r] -> [r, n]
        for (let n = 0; n < e.n; n++) for (let r = 0; r < e.r; r++) Bt[r * e.n + n] = e.Bdata[n * e.r + r];
        let t = b.matmul(xin, b.constant({ dataType: ct, shape: [K, e.r] }, floatData(At, ct)));
        t = b.matmul(t, b.constant({ dataType: ct, shape: [e.r, e.n] }, floatData(Bt, ct)));
        t = b.mul(t, cast(b.slice(loraIn, [loraIdx.get(e)], [1]), ct));
        if (e.n !== d.N) t = b.pad(t, [0, e.off], [0, d.N - e.off - e.n], { mode: "constant", value: 0 });
        y = b.add(y, t);
      }
      if (bias) y = b.add(y, bias);
      if (act === "gelu") y = b.gelu(y);
      if (act === "silu") y = silu(y);
      return y;
    };
    const lin32 = async (x, d, act) => cast(await lin(x, d, act), "float32");

    // RMSNorm over the last axis (fp32), times weight
    const rms = (x, weight) => {
      const s = shapeOf(x);
      const ms = b.reduceMean(b.mul(x, x), { axes: [s.length - 1], keepDimensions: true });
      return b.mul(b.div(x, b.sqrt(b.add(ms, scalar(1e-6)))), f32(weight, [weight.length]));
    };
    const slice2 = (x, c0, n) => b.slice(x, [0, c0], [shapeOf(x)[0], n]);
    // LayerNorm (no affine) + adaLN: LN(x) * (1 + scale) + shift, mod = [shift | scale | gate]
    const lnMod = (x, mod) => b.add(b.mul(b.layerNormalization(x, { axes: [1], epsilon: 1e-6 }), b.add(slice2(mod, D, D), scalar(1))), slice2(mod, 0, D));
    const gated = (x, mod, y) => b.add(x, b.mul(slice2(mod, 2 * D, D), y));
    // split-half rope on [L, H, 128]
    const rope = (x) => {
      const x1 = b.slice(x, [0, 0, 0], [L, HEADS, HD / 2]);
      const x2 = b.slice(x, [0, 0, HD / 2], [L, HEADS, HD / 2]);
      return b.concat([b.sub(b.mul(x1, cos), b.mul(x2, sin)), b.add(b.mul(x2, cos), b.mul(x1, sin))], 2);
    };
    // q [Lq, H, HD] (fp32), k [Lk, H, HD] (fp32), v [Lk, H, HD] -> [Lq, D] in ct
    const attention = (q, k, v, Lq, Lk_) => {
      const qs = b.transpose(cast(b.mul(q, scalar(1 / Math.sqrt(HD))), ct), { permutation: [1, 0, 2] });
      const ks = b.transpose(cast(k, ct), { permutation: [1, 2, 0] });
      const vs = b.transpose(cast(v, ct), { permutation: [1, 0, 2] });
      const bytes = ct === "float16" ? 2 : 4;
      let hc = HEADS;
      while (hc > 1 && hc * Lq * Lk_ * bytes > ATTN_BUDGET) hc /= 2;
      const outs = [];
      for (let h0 = 0; h0 < HEADS; h0 += hc) {
        const s = b.matmul(b.slice(qs, [h0, 0, 0], [hc, Lq, HD]), b.slice(ks, [h0, 0, 0], [hc, HD, Lk_]));
        const p = b.softmax(s, 2);
        outs.push(b.matmul(p, b.slice(vs, [h0, 0, 0], [hc, Lk_, HD])));
      }
      const o = outs.length > 1 ? b.concat(outs, 0) : outs[0];
      return b.reshape(b.transpose(o, { permutation: [1, 0, 2] }), [Lq, D]);
    };
    const heads = (x, n) => b.reshape(x, [n, HEADS, HD]);

    // timestep conditioning
    const tl = await lin32(await lin(temb, this.t1, "silu"), this.t2); // adaln_lora [1, 3D]
    const semb = silu(rms(temb, this.tNorm));
    const mod = async ([w1, w2], n) => b.add(b.slice(tl, [0, 0], [1, n]), await lin32(await lin(semb, w1), w2));

    let x = await lin32(patches, this.xEmbed);
    for (let i = 0; i < 28; i++) {
      const blk = this.blocks[i];
      const mSelf = await mod(blk.modSelf, 3 * D);
      const mCross = await mod(blk.modCross, 3 * D);
      const mMlp = await mod(blk.modMlp, 3 * D);

      // self-attention
      let n = lnMod(x, mSelf);
      let q, k, v;
      if (blk.self.qkv) {
        const qkv = await lin(n, blk.self.qkv);
        q = slice2(qkv, 0, D); k = slice2(qkv, D, D); v = slice2(qkv, 2 * D, D);
      } else {
        q = await lin(n, blk.self.q); k = await lin(n, blk.self.k); v = await lin(n, blk.self.v);
      }
      q = rope(rms(heads(cast(q, "float32"), L), blk.self.qn));
      k = rope(rms(heads(cast(k, "float32"), L), blk.self.kn));
      let a = attention(q, k, heads(v, L), L, L);
      x = gated(x, mSelf, await lin32(a, blk.self.o));

      // cross-attention (keys/values from the prompt context)
      n = lnMod(x, mCross);
      q = rms(heads(await lin32(n, blk.cross.q), L), blk.cross.qn);
      k = rms(heads(await lin32(ctx, blk.cross.k), Lk), blk.cross.kn);
      v = heads(await lin(ctx, blk.cross.v), Lk);
      a = attention(q, k, v, L, Lk);
      x = gated(x, mCross, await lin32(a, blk.cross.o));

      // MLP
      n = lnMod(x, mMlp);
      x = gated(x, mMlp, await lin32(await lin(n, blk.l1, "gelu"), blk.l2));
      onProgress?.((i + 1) / 29);
    }
    const fm = await mod(this.finalMod, 2 * D);
    const out = await lin32(lnMod(x, fm), this.finalLinear); // [L, 64]
    const graph = await b.build({ out });
    onProgress?.(1);
    return graph;
  }

  // ---------------------------------------------------------------- execution

  async tensor(shape, usage) {
    return this.context.createTensor({ dataType: "float32", shape, ...usage });
  }

  // (Re)compiles the graph for this resolution / prompt length / LoRA set when needed.
  async ensureGraph(Hp, Wp, Lk, onProgress) {
    const key = this.graphKey(Hp, Wp, Lk);
    if (this.run?.key === key) return;
    this.releaseGraph();
    const L = Hp * Wp;
    const graph = await this.build(Hp, Wp, Lk, onProgress);
    const W = { writable: true };
    const nLora = this.loraEntries().length;
    const run = {
      key, graph, L, Lk,
      inputs: {
        patches: await this.tensor([L, 68], W),
        temb: await this.tensor([1, D], W),
        ctx: await this.tensor([Lk, 1024], W),
        cos: await this.tensor([L, 1, 64], W),
        sin: await this.tensor([L, 1, 64], W),
      },
      out: await this.tensor([L, 64], { readable: true }),
      ctxOf: null,
    };
    if (nLora) run.inputs.lora = await this.tensor([nLora], W);
    const angles = ditRopeAngles(Hp, Wp);
    this.context.writeTensor(run.inputs.cos, angles.map(Math.cos));
    this.context.writeTensor(run.inputs.sin, angles.map(Math.sin));
    this.run = run;
  }

  releaseGraph() {
    if (!this.run) return;
    for (const t of Object.values(this.run.inputs)) t.destroy();
    this.run.out.destroy();
    this.run.graph.destroy();
    this.run = null;
  }

  // latent: Float32Array [16, h, w]; cond: { ctx: Float32Array [Lk, 1024], Lk }.
  // onCompile(frac): progress while a graph is being (re)built. Returns the velocity [16, h, w].
  async forward(latent, h, w, cond, sigma, onCompile) {
    await this.ensureGraph(h / 2, w / 2, cond.Lk, onCompile);
    const { context: c, run } = this;
    c.writeTensor(run.inputs.patches, patchify(latent, h, w));
    c.writeTensor(run.inputs.temb, timestepEmbedding(sigma));
    if (run.ctxOf !== cond) {
      c.writeTensor(run.inputs.ctx, cond.ctx);
      run.ctxOf = cond;
    }
    if (run.inputs.lora) c.writeTensor(run.inputs.lora, new Float32Array(this.loraEntries().map(({ e }) => e.scale)));
    const outputs = { out: run.out };
    c.dispatch(run.graph, run.inputs, outputs);
    const o = new Float32Array(await c.readTensor(run.out));
    return unpatchify(o, h, w);
  }

  dispose() {
    this.releaseGraph();
    for (const d of this.linears()) for (const t of Object.values(d.tensors || {})) t.destroy?.();
    this.context.destroy?.();
  }
}

