// Tensor ops built on the WGSL kernels. Activations are f32 Tensors (row-major).

import { grid } from "./device.js";
import * as K from "./kernels.js";

const mmParams = (o) => [
  ["u32", o.M], ["u32", o.N], ["u32", o.K], ["f32", o.alpha ?? 1],
  ["u32", o.lda ?? o.K], ["u32", o.aBatch ?? 0], ["u32", o.aOff ?? 0], ["u32", o.ldb ?? o.K],
  ["u32", o.bBatch ?? 0], ["u32", o.bDiv ?? 1], ["u32", o.bOff ?? 0], ["u32", o.ldc ?? o.N],
  ["u32", o.cBatch ?? 0], ["u32", o.cOff ?? 0], ["u32", o.gOff ?? 0], ["u32", o.cin ?? 0],
  ["u32", o.ch ?? 0], ["u32", o.cw ?? 0], ["u32", o.up ? 1 : 0], ["u32", 0],
];

function runMatmul(gpu, { a = "rows", A, W, C, bias, act = "none", resid = false, gate, add, batch = 1, name, ...o }) {
  const b = W.kind;
  // big 128x128 tiles unless the problem is too small to fill them
  const R = o.M >= 256 && o.N >= 128 ? 8 : 4;
  const T = 16 * R;
  // vec4 loads need whole 8-wide k chunks and 4-aligned offsets/strides
  const al = (...xs) => xs.every((x) => (x ?? 0) % 4 === 0);
  const k8 = o.K % 8 === 0;
  const vecA = k8 && (a === "conv3" ? o.cin % 8 === 0 : al(o.lda ?? o.K, o.aOff, o.aBatch));
  const vecB = k8 && (b === "bf16" || b === "i8" || (b === "f32" && al(o.ldb ?? o.K, o.bOff, o.bBatch)));
  const code = K.matmulShader({ a, b, bias: !!bias, act, resid, gate: !!gate, R, vecA, vecB, add: !!add });
  const bufs = [A, W.buf, C];
  if (b === "i8" || b === "w4") bufs.push(W.scale);
  if (b === "w4") bufs.push(W.srel, W.codebook);
  if (bias) bufs.push(bias);
  if (gate) bufs.push(gate);
  if (add) bufs.push(add);
  const meta = { name: name || (a === "conv3" ? `conv3x3.${b}` : `gemm.${b}`), flops: 2 * o.M * o.N * o.K * batch };
  gpu.dispatch(code, bufs, mmParams(o), [Math.ceil(o.N / T), Math.ceil(o.M / T), batch], meta);
}

export const needsRotation = (W) => W.kind === "i8" || W.kind === "w4";

// ConvRot input rotation (Hadamard over each group of 256 features).
export function rotate(gpu, x) {
  const y = gpu.empty(x.shape);
  const groups = x.size / 256;
  const [nx, ny] = grid(groups);
  gpu.dispatch(K.hadamardShader(), [x, y], [["u32", groups], ["u32", nx]], [nx, ny], { name: "hadamard" });
  return y;
}

// y = x @ W^T (+ bias) (act). `xr` is the pre-rotated input for quantized weights; when
// omitted it is computed here. opts.out + opts.resid: out += gate[gOff+n] * y (in place).
export function linear(gpu, x, W, opts = {}) {
  const M = x.size / W.K;
  let input = x;
  let tmp = null;
  if (needsRotation(W)) {
    input = opts.xr || (tmp = rotate(gpu, x));
  }
  // LoRAs: delta = sum_i scale_i * B_i (A_i x), added inside the epilogue before act/gate.
  // A is pre-rotated for quantized weights, so it takes the same (rotated) input.
  const delta = W.lora?.length ? loraDelta(gpu, input, W, M) : null;
  const out = opts.out || gpu.empty([M, W.N]);
  runMatmul(gpu, {
    A: input, W, C: out, M, N: W.N, K: W.K, add: delta,
    bias: opts.bias || W.bias, act: opts.act, resid: !!opts.resid, gate: opts.gate, gOff: opts.gOff, name: opts.name,
  });
  if (tmp) tmp.release();
  delta?.release();
  return out;
}

// [M, W.N] low-rank update. Entries: { A: [r, K] f32, B: [n, r] f32, r, scale, off, n } where
// off/n select the output columns (the slices of a stacked q/k/v weight).
function loraDelta(gpu, x, W, M) {
  const D = gpu.empty([M, W.N]);
  const isFull = (e) => e.off === 0 && e.n === W.N;
  const entries = [...W.lora].sort((a, b) => isFull(b) - isFull(a)); // a full-width entry initializes D
  const full = entries.some(isFull);
  if (!full) zero(gpu, D);
  let first = full;
  for (const e of entries) {
    const t = gpu.empty([M, e.r]);
    runMatmul(gpu, { A: x, W: { kind: "f32", buf: e.A }, C: t, M, N: e.r, K: W.K, name: "lora" });
    // the first full-width entry stores; everything else accumulates
    const store = first && isFull(e);
    runMatmul(gpu, { A: t, W: { kind: "f32", buf: e.B }, C: D, M, N: e.n, K: e.r, ldc: W.N, cOff: e.off, alpha: e.scale, resid: !store, name: "lora" });
    if (store) first = false;
    t.release();
  }
  return D;
}

export function zero(gpu, t) {
  const n = t.size;
  const [nx, ny] = grid(Math.ceil(n / 256));
  gpu.dispatch(K.zeroShader(), [t], [["u32", n], ["u32", nx]], [nx, ny], { name: "zero" });
}

// 3x3 conv (pad 1) on an NHWC image [h*w, cin] -> [H*W, cout]; up=true fuses a 2x nearest
// upsample of the input (output is then 2h x 2w).
export function conv3x3(gpu, x, W, h, w, up = false, into = null) {
  const H = up ? h * 2 : h;
  const Wd = up ? w * 2 : w;
  const cin = W.K / 9;
  const out = into || gpu.empty([H * Wd, W.N]);
  runMatmul(gpu, { a: "conv3", A: x, W, C: out, M: H * Wd, N: W.N, K: W.K, cin, ch: H, cw: Wd, up, bias: W.bias, resid: !!into });
  return out;
}

// out += conv3x3(x)  (residual connection fused into the epilogue)
export function conv3x3Into(gpu, x, W, h, w, out) {
  return conv3x3(gpu, x, W, h, w, false, out);
}

export function rmsnorm(gpu, x, weight, cols, eps = 1e-6, out) {
  const rows = x.size / cols;
  const y = out || gpu.empty(x.shape);
  const [nx, ny] = grid(rows);
  gpu.dispatch(K.rmsnormShader(0, false), [x, weight, y], [["u32", rows], ["u32", cols], ["u32", nx], ["f32", eps]], [nx, ny], { name: "rmsnorm" });
  return y;
}

// Wan VAE RMS_norm over channels (+ SiLU).
export function channelNorm(gpu, x, gamma, cols, silu) {
  const rows = x.size / cols;
  const y = gpu.empty(x.shape);
  const [nx, ny] = grid(rows);
  gpu.dispatch(K.rmsnormShader(1, silu), [x, gamma, y], [["u32", rows], ["u32", cols], ["u32", nx], ["f32", 0]], [nx, ny], { name: "vae.channelnorm" });
  return y;
}

// rotate: output is already ConvRot-rotated (input for quantized linears via { xr })
export function layernormMod(gpu, x, mod, shiftOff, scaleOff, cols, rotate = false) {
  const rows = x.size / cols;
  const y = gpu.empty(x.shape);
  const [nx, ny] = grid(rows);
  gpu.dispatch(K.layernormModShader(rotate), [x, mod, y],
    [["u32", rows], ["u32", cols], ["u32", shiftOff], ["u32", scaleOff], ["u32", nx], ["f32", 1e-6]], [nx, ny],
    { name: rotate ? "layernorm_mod+rot" : "layernorm_mod" });
  return y;
}

// In-place per-head RMSNorm + RoPE on the q and k parts of a fused QKV buffer [L, 3*H*D].
export function qkNormRope(gpu, qkv, qn, kn, cs, L, H, D) {
  const [nx, ny] = grid(L * 2 * H);
  gpu.dispatch(K.qkNormRopeShader(D), [qkv, qn, kn, cs.buf],
    [["u32", L], ["u32", H], ["u32", 3 * H * D], ["u32", cs.sinOff], ["u32", nx], ["f32", 1e-6]], [nx, ny],
    { name: "qk_norm_rope" });
}

// In-place rope on x viewed as [L, H, D] inside rows of `rowStride` starting at colOff.
export function rope(gpu, x, cs, L, H, D, rowStride = H * D, colOff = 0) {
  const n = L * H * (D / 2);
  const [nx, ny] = grid(Math.ceil(n / 64));
  gpu.dispatch(K.ropeShader(), [x, cs.buf], [["u32", L], ["u32", H], ["u32", D], ["u32", cs.sinOff], ["u32", rowStride], ["u32", colOff]], [nx, ny], { name: "rope" });
}

export function elementwise(gpu, op, a, b, out) {
  const y = out || gpu.empty(a.shape);
  const n = y.size;
  const [nx, ny] = grid(Math.ceil(n / 256));
  gpu.dispatch(K.elementwiseShader(op), b ? [a, b, y] : [a, y], [["u32", n], ["u32", nx]], [nx, ny], { name: "elementwise." + op });
  return y;
}

// Multi-head attention without masking except optional causal.
//  q: [Lq, ldq] with head h at column qOff + h*D; k, v: [Lk, ldk]/[Lk, ldv] with kv head
//  h/group at kOff/vOff + (h/group)*D. Output [Lq, H*D].
// Scores are materialized per chunk of (heads x query rows) to bound memory.
export function attention(gpu, { q, k, v, Lq, Lk, H, D, ldq, ldk, ldv, qOff = 0, kOff = 0, vOff = 0, group = 1, causal = false, out }) {
  const o = out || gpu.empty([Lq, H * D]);
  // fused kernel: no score matrix in memory (needs vec4-aligned rows/offsets)
  const aligned = [ldq, ldk, ldv, qOff, kOff, vOff].every((x) => x % 4 === 0);
  if (!causal && group === 1 && (D === 64 || D === 128) && aligned) {
    gpu.dispatch(K.flashAttentionShader(D), [q, k, v, o],
      [["u32", Lq], ["u32", Lk], ["u32", ldq], ["u32", ldk], ["u32", ldv], ["u32", H * D],
        ["u32", qOff], ["u32", kOff], ["u32", vOff], ["u32", 0], ["f32", 1 / Math.sqrt(D)]],
      [Math.ceil(Lq / 64), H], { name: "attn.flash", flops: 4 * Lq * Lk * D * H });
    return o;
  }
  const budget = Math.min(gpu.maxBinding, 256 * 2 ** 20) / 4; // floats per score chunk
  let rows = Math.min(Lq, Math.max(64, Math.floor(budget / Lk / 64) * 64));
  let heads = Math.max(1, Math.min(H, Math.floor(budget / (rows * Lk))));
  const scale = 1 / Math.sqrt(D);
  if (group > 1) heads = Math.max(group, heads - (heads % group));
  if (causal && rows < Lq) throw new Error("causal attention must fit in one row chunk");
  for (let h0 = 0; h0 < H; h0 += heads) {
    const nh = Math.min(heads, H - h0);
    for (let r0 = 0; r0 < Lq; r0 += rows) {
      const nr = Math.min(rows, Lq - r0);
      const S = gpu.empty([nh, nr, Lk]);
      // S = Q K^T  (batch over heads; kv head = (h0+z)/group)
      runMatmul(gpu, {
        A: q, W: { kind: "f32", buf: k.buf }, C: S, batch: nh, M: nr, N: Lk, K: D,
        lda: ldq, aBatch: D, aOff: r0 * ldq + qOff + h0 * D,
        ldb: ldk, bBatch: D, bDiv: group, bOff: kOff + Math.floor(h0 / group) * D,
        ldc: Lk, cBatch: nr * Lk, name: "attn.qk",
      });
      const [nx, ny] = grid(nh * nr);
      gpu.dispatch(K.softmaxShader(), [S], [["u32", nh * nr], ["u32", Lk], ["u32", causal ? 1 : 0], ["u32", nr], ["u32", nx], ["f32", scale]], [nx, ny], { name: "attn.softmax" });
      // O = P V
      runMatmul(gpu, {
        A: S, W: { kind: "f32t", buf: v.buf }, C: o, batch: nh, M: nr, N: D, K: Lk,
        lda: Lk, aBatch: nr * Lk, aOff: 0,
        ldb: ldv, bBatch: D, bDiv: group, bOff: vOff + Math.floor(h0 / group) * D,
        ldc: H * D, cBatch: D, cOff: r0 * H * D + h0 * D, name: "attn.pv",
      });
      S.release();
    }
  }
  return o;
}
