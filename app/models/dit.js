// Anima diffusion transformer: Cosmos-Predict2 MiniTrainDIT (2B) + the LLM adapter that maps
// Qwen3 hidden states onto T5-token queries.

import * as ops from "../gpu/ops.js";
import { concatLinears } from "../weights.js";
import { ropeTable, ditRopeTable } from "./common.js";

const D = 2048;
const HEADS = 16;
const HD = 128;
const CTX_LEN = 512;

async function loadAttn(gpu, st, p, names) {
  return {
    q: await st.linear(gpu, p + "q_proj."),
    k: await st.linear(gpu, p + "k_proj."),
    v: await st.linear(gpu, p + "v_proj."),
    o: await st.linear(gpu, p + names.o + "."),
    qn: await st.vector(gpu, p + "q_norm.weight"),
    kn: await st.vector(gpu, p + "k_norm.weight"),
  };
}

function project(gpu, x, W, xr, name) {
  return ops.linear(gpu, x, W, { xr, name });
}

function maybeRotate(gpu, x, ...Ws) {
  return Ws.some(ops.needsRotation) ? ops.rotate(gpu, x) : null;
}

// LayerNorm + adaLN for inputs that feed `Ws`: when every consumer is a ConvRot-quantized linear
// the rotation is fused into the norm kernel and the result is passed on as { xr }.
function normFor(gpu, x, mod, Ws) {
  const rotated = Ws.every(ops.needsRotation);
  const n = ops.layernormMod(gpu, x, mod, 0, D, D, rotated);
  return { n, xr: rotated ? n : undefined };
}

export class AnimaDiT {
  static async load(gpu, st, onProgress) {
    const m = new AnimaDiT(gpu, st);
    // LLM adapter
    const A = { blocks: [] };
    for (let i = 0; i < 6; i++) {
      const p = `llm_adapter.blocks.${i}.`;
      A.blocks.push({
        nSelf: await st.vector(gpu, p + "norm_self_attn.weight"),
        nCross: await st.vector(gpu, p + "norm_cross_attn.weight"),
        nMlp: await st.vector(gpu, p + "norm_mlp.weight"),
        self: await loadAttn(gpu, st, p + "self_attn.", { o: "o_proj" }),
        cross: await loadAttn(gpu, st, p + "cross_attn.", { o: "o_proj" }),
        mlp0: await st.linear(gpu, p + "mlp.0."),
        mlp2: await st.linear(gpu, p + "mlp.2."),
      });
    }
    A.outProj = await st.linear(gpu, "llm_adapter.out_proj.");
    A.norm = await st.vector(gpu, "llm_adapter.norm.weight");
    m.adapter = A;

    m.xEmbed = await st.linear(gpu, "x_embedder.proj.1.");
    m.t1 = await st.linear(gpu, "t_embedder.1.linear_1.");
    m.t2 = await st.linear(gpu, "t_embedder.1.linear_2.");
    m.tNorm = await st.vector(gpu, "t_embedding_norm.weight");
    m.blocks = [];
    for (let i = 0; i < 28; i++) {
      const p = `blocks.${i}.`;
      const mod = async (n) => [await st.linear(gpu, p + n + ".1."), await st.linear(gpu, p + n + ".2.")];
      const self = await loadAttn(gpu, st, p + "self_attn.", { o: "output_proj" });
      // one [3*D, D] GEMM instead of three (null if the formats can't be stacked)
      self.qkv = concatLinears(gpu, [self.q, self.k, self.v]);
      if (self.qkv) { delete self.q; delete self.k; delete self.v; }
      m.blocks.push({
        modSelf: await mod("adaln_modulation_self_attn"),
        modCross: await mod("adaln_modulation_cross_attn"),
        modMlp: await mod("adaln_modulation_mlp"),
        self,
        cross: await loadAttn(gpu, st, p + "cross_attn.", { o: "output_proj" }),
        l1: await st.linear(gpu, p + "mlp.layer1."),
        l2: await st.linear(gpu, p + "mlp.layer2."),
      });
      onProgress?.((i + 1) / 28);
    }
    m.finalMod = [await st.linear(gpu, "final_layer.adaln_modulation.1."), await st.linear(gpu, "final_layer.adaln_modulation.2.")];
    m.finalLinear = await st.linear(gpu, "final_layer.linear.");
    return m;
  }

  constructor(gpu, st) {
    this.gpu = gpu;
    this.st = st;
  }

  // ------------------------------------------------------------------ LLM adapter

  // hidden: Tensor [Ls, 1024] from Qwen3; t5Ids/t5Weights: arrays. Returns Tensor [512, 1024].
  async adapt(hidden, t5Ids, t5Weights) {
    const gpu = this.gpu;
    const A = this.adapter;
    const Lt = t5Ids.length;
    const Ls = hidden.shape[0];
    let x = gpu.fromArray(await this.st.rows("llm_adapter.embed.weight", t5Ids), [Lt, 1024]);
    const csT = ropeTable(gpu, Lt, 64, 1e4);
    const csS = ropeTable(gpu, Ls, 64, 1e4);

    const attn = (w, xq, ctx, Lq, Lk, csQ, csK) => {
      const q = ops.linear(gpu, xq, w.q);
      const k = ops.linear(gpu, ctx, w.k);
      const v = ops.linear(gpu, ctx, w.v);
      const qn = ops.rmsnorm(gpu, q, w.qn, 64);
      const kn = ops.rmsnorm(gpu, k, w.kn, 64);
      q.release(); k.release();
      ops.rope(gpu, qn, csQ, Lq, 16, 64);
      ops.rope(gpu, kn, csK, Lk, 16, 64);
      const a = ops.attention(gpu, { q: qn, k: kn, v, Lq, Lk, H: 16, D: 64, ldq: 1024, ldk: 1024, ldv: 1024 });
      qn.release(); kn.release(); v.release();
      ops.linear(gpu, a, w.o, { out: x, resid: true });
      a.release();
    };

    for (const b of A.blocks) {
      let n = ops.rmsnorm(gpu, x, b.nSelf, 1024);
      attn(b.self, n, n, Lt, Lt, csT, csT);
      n.release();
      n = ops.rmsnorm(gpu, x, b.nCross, 1024);
      attn(b.cross, n, hidden, Lt, Ls, csT, csS);
      n.release();
      n = ops.rmsnorm(gpu, x, b.nMlp, 1024);
      const h = ops.linear(gpu, n, b.mlp0, { act: "gelu" });
      n.release();
      ops.linear(gpu, h, b.mlp2, { out: x, resid: true });
      h.release();
    }
    csT.release(); csS.release();
    const o = ops.linear(gpu, x, A.outProj);
    x.release();
    const on = ops.rmsnorm(gpu, o, A.norm, 1024);
    o.release();

    // apply prompt weights and zero-pad to 512 tokens (on CPU: tiny)
    const vals = await gpu.read(on);
    on.release();
    const ctx = new Float32Array(Math.max(CTX_LEN, Lt) * 1024);
    for (let i = 0; i < Lt; i++) {
      const w = t5Weights ? t5Weights[i] : 1;
      for (let j = 0; j < 1024; j++) ctx[i * 1024 + j] = vals[i * 1024 + j] * w;
    }
    return gpu.fromArray(ctx, [ctx.length / 1024, 1024]);
  }

  // ------------------------------------------------------------------ timestep conditioning

  // Everything that depends only on sigma.
  timeCond(sigma) {
    const gpu = this.gpu;
    const half = D / 2;
    const temb = new Float32Array(D);
    for (let i = 0; i < half; i++) {
      const f = Math.fround(Math.exp(Math.fround((-Math.log(10000) * i) / half)));
      const a = Math.fround(sigma * f);
      temb[i] = Math.cos(a);
      temb[half + i] = Math.sin(a);
    }
    const t = gpu.fromArray(temb, [1, D]);
    const h = ops.linear(gpu, t, this.t1, { act: "silu" });
    const lora = ops.linear(gpu, h, this.t2); // adaln_lora [1, 6144]
    h.release();
    const emb = ops.rmsnorm(gpu, t, this.tNorm, D);
    t.release();
    const semb = ops.elementwise(gpu, "silu", emb);
    emb.release();
    const mod = ([w1, w2], n) => {
      const out = gpu.empty([1, n]);
      ops.elementwise(gpu, "copy", lora, null, out); // copies the first n entries
      const hh = ops.linear(gpu, semb, w1);
      ops.linear(gpu, hh, w2, { out, resid: true });
      hh.release();
      return out;
    };
    const blocks = this.blocks.map((b) => ({ self: mod(b.modSelf, 3 * D), cross: mod(b.modCross, 3 * D), mlp: mod(b.modMlp, 3 * D) }));
    const final = mod(this.finalMod, 2 * D);
    semb.release();
    lora.release();
    return {
      blocks, final,
      release() {
        for (const b of blocks) { b.self.release(); b.cross.release(); b.mlp.release(); }
        final.release();
      },
    };
  }

  // ------------------------------------------------------------------ forward

  // Cross-attention keys/values depend only on the prompt, not on the latent or the step:
  // compute them once per prompt for all blocks (k already q/k-normed). ~235 MB for 28 blocks.
  // Takes ownership of ctx (Tensor [512, 1024]).
  prepareContext(ctx) {
    const gpu = this.gpu;
    const cr = maybeRotate(gpu, ctx, ...this.blocks.flatMap((b) => [b.cross.k, b.cross.v]));
    const kv = this.blocks.map((b) => {
      const k = project(gpu, ctx, b.cross.k, cr, "dit.cross.kv");
      const kn = ops.rmsnorm(gpu, k, b.cross.kn, HD);
      k.release();
      return { k: kn, v: project(gpu, ctx, b.cross.v, cr, "dit.cross.kv") };
    });
    cr?.release();
    return {
      ctx, kv, Lk: ctx.shape[0],
      release() {
        ctx.release();
        for (const e of kv) { e.k.release(); e.v.release(); }
      },
    };
  }

  // latent: Float32Array [16, h, w]; cond: prepareContext(...); tc: timeCond(sigma).
  // Returns the flow velocity as Float32Array [16, h, w].
  async forward(latent, h, w, cond, tc, onBlock) {
    const gpu = this.gpu;
    const Hp = h / 2, Wp = w / 2, L = Hp * Wp;
    // patchify "c (h m) (w n) -> (h w) (c m n)" with the zero padding-mask channel as c = 16
    const patches = new Float32Array(L * 68);
    for (let y = 0; y < Hp; y++) for (let x = 0; x < Wp; x++) {
      const o = (y * Wp + x) * 68;
      for (let c = 0; c < 16; c++) for (let m = 0; m < 2; m++) for (let n = 0; n < 2; n++) {
        patches[o + c * 4 + m * 2 + n] = latent[c * h * w + (2 * y + m) * w + (2 * x + n)];
      }
    }
    const xin = gpu.fromArray(patches, [L, 68]);
    const x = ops.linear(gpu, xin, this.xEmbed);
    xin.release();
    const cs = ditRopeTable(gpu, Hp, Wp);

    for (let i = 0; i < 28; i++) {
      const b = this.blocks[i];
      const m = tc.blocks[i];

      // self-attention
      let a;
      if (b.self.qkv) {
        const { n, xr } = normFor(gpu, x, m.self, [b.self.qkv]);
        const qkv = ops.linear(gpu, n, b.self.qkv, { xr, name: "dit.self.qkv" }); // [L, 3D]
        n.release();
        ops.qkNormRope(gpu, qkv, b.self.qn, b.self.kn, cs, L, HEADS, HD);
        a = ops.attention(gpu, { q: qkv, k: qkv, v: qkv, Lq: L, Lk: L, H: HEADS, D: HD, ldq: 3 * D, ldk: 3 * D, ldv: 3 * D, qOff: 0, kOff: D, vOff: 2 * D });
        qkv.release();
      } else {
        const n = ops.layernormMod(gpu, x, m.self, 0, D, D);
        const nr = maybeRotate(gpu, n, b.self.q, b.self.k, b.self.v);
        const q = project(gpu, n, b.self.q, nr, "dit.self.qkv");
        const k = project(gpu, n, b.self.k, nr, "dit.self.qkv");
        const v = project(gpu, n, b.self.v, nr, "dit.self.qkv");
        nr?.release(); n.release();
        const qn = ops.rmsnorm(gpu, q, b.self.qn, HD);
        const kn = ops.rmsnorm(gpu, k, b.self.kn, HD);
        q.release(); k.release();
        ops.rope(gpu, qn, cs, L, HEADS, HD);
        ops.rope(gpu, kn, cs, L, HEADS, HD);
        a = ops.attention(gpu, { q: qn, k: kn, v, Lq: L, Lk: L, H: HEADS, D: HD, ldq: D, ldk: D, ldv: D });
        qn.release(); kn.release(); v.release();
      }
      ops.linear(gpu, a, b.self.o, { out: x, resid: true, gate: m.self, gOff: 2 * D, name: "dit.self.out" });
      a.release();

      // cross-attention against the per-prompt cached keys/values
      {
        const { n, xr } = normFor(gpu, x, m.cross, [b.cross.q]);
        const q = ops.linear(gpu, n, b.cross.q, { xr, name: "dit.cross.q" });
        n.release();
        const qn = ops.rmsnorm(gpu, q, b.cross.qn, HD);
        q.release();
        const { k, v } = cond.kv[i];
        a = ops.attention(gpu, { q: qn, k, v, Lq: L, Lk: cond.Lk, H: HEADS, D: HD, ldq: D, ldk: D, ldv: D });
        qn.release();
      }
      ops.linear(gpu, a, b.cross.o, { out: x, resid: true, gate: m.cross, gOff: 2 * D, name: "dit.cross.out" });
      a.release();

      // MLP
      const { n, xr } = normFor(gpu, x, m.mlp, [b.l1]);
      const hdn = ops.linear(gpu, n, b.l1, { xr, act: "gelu", name: "dit.mlp1" });
      n.release();
      ops.linear(gpu, hdn, b.l2, { out: x, resid: true, gate: m.mlp, gOff: 2 * D, name: "dit.mlp2" });
      hdn.release();

      gpu.flush();
      if (onBlock) await onBlock(i, x);
    }
    cs.release();

    const n = ops.layernormMod(gpu, x, tc.final, 0, D, D);
    x.release();
    const out = ops.linear(gpu, n, this.finalLinear); // [L, 64] = (p1 p2 c)
    n.release();
    const o = await gpu.read(out);
    out.release();
    // unpatchify "(h w) (p1 p2 c) -> c (h p1) (w p2)"
    const v = new Float32Array(16 * h * w);
    for (let y = 0; y < Hp; y++) for (let xx = 0; xx < Wp; xx++) {
      const base = (y * Wp + xx) * 64;
      for (let p1 = 0; p1 < 2; p1++) for (let p2 = 0; p2 < 2; p2++) for (let c = 0; c < 16; c++) {
        v[c * h * w + (2 * y + p1) * w + (2 * xx + p2)] = o[base + p1 * 32 + p2 * 16 + c];
      }
    }
    return v;
  }
}
