// Qwen3-0.6B text encoder (final hidden state after the last norm), as used by Anima.

import * as ops from "../gpu/ops.js";
import { ropeTable } from "./common.js";

const CFG = { layers: 28, hidden: 1024, heads: 16, kvHeads: 8, headDim: 128, theta: 1e6 };

export class Qwen3 {
  static async load(gpu, st, onProgress) {
    const m = new Qwen3(gpu, st);
    const L = [];
    for (let i = 0; i < CFG.layers; i++) {
      const p = `model.layers.${i}.`;
      L.push({
        ln1: await st.vector(gpu, p + "input_layernorm.weight"),
        ln2: await st.vector(gpu, p + "post_attention_layernorm.weight"),
        qn: await st.vector(gpu, p + "self_attn.q_norm.weight"),
        kn: await st.vector(gpu, p + "self_attn.k_norm.weight"),
        q: await st.linear(gpu, p + "self_attn.q_proj."),
        k: await st.linear(gpu, p + "self_attn.k_proj."),
        v: await st.linear(gpu, p + "self_attn.v_proj."),
        o: await st.linear(gpu, p + "self_attn.o_proj."),
        gate: await st.linear(gpu, p + "mlp.gate_proj."),
        up: await st.linear(gpu, p + "mlp.up_proj."),
        down: await st.linear(gpu, p + "mlp.down_proj."),
      });
      onProgress?.((i + 1) / CFG.layers);
    }
    m.layers = L;
    m.norm = await st.vector(gpu, "model.norm.weight");
    return m;
  }

  constructor(gpu, st) {
    this.gpu = gpu;
    this.st = st; // embedding rows are gathered from the file on demand
  }

  // ids -> Tensor [L, 1024]
  async encode(ids) {
    const gpu = this.gpu;
    const { hidden: D, heads: H, kvHeads: KH, headDim: HD } = CFG;
    const Lt = ids.length;
    let x = gpu.fromArray(await this.st.rows("model.embed_tokens.weight", ids), [Lt, D]);
    const cs = ropeTable(gpu, Lt, HD, CFG.theta);
    for (const l of this.layers) {
      const h = ops.rmsnorm(gpu, x, l.ln1, D);
      const hr = ops.needsRotation(l.q) ? ops.rotate(gpu, h) : null;
      const q = ops.linear(gpu, h, l.q, { xr: hr });
      const k = ops.linear(gpu, h, l.k, { xr: hr });
      const v = ops.linear(gpu, h, l.v, { xr: hr });
      hr?.release();
      h.release();
      const qn = ops.rmsnorm(gpu, q, l.qn, HD);
      const kn = ops.rmsnorm(gpu, k, l.kn, HD);
      q.release();
      k.release();
      ops.rope(gpu, qn, cs, Lt, H, HD);
      ops.rope(gpu, kn, cs, Lt, KH, HD);
      const a = ops.attention(gpu, { q: qn, k: kn, v, Lq: Lt, Lk: Lt, H, D: HD, ldq: H * HD, ldk: KH * HD, ldv: KH * HD, group: H / KH, causal: true });
      qn.release(); kn.release(); v.release();
      ops.linear(gpu, a, l.o, { out: x, resid: true });
      a.release();
      const h2 = ops.rmsnorm(gpu, x, l.ln2, D);
      const h2r = ops.needsRotation(l.gate) ? ops.rotate(gpu, h2) : null;
      const g = ops.linear(gpu, h2, l.gate, { xr: h2r });
      const u = ops.linear(gpu, h2, l.up, { xr: h2r });
      h2r?.release();
      h2.release();
      const gu = ops.elementwise(gpu, "silu_mul", g, u);
      g.release(); u.release();
      ops.linear(gpu, gu, l.down, { out: x, resid: true });
      gu.release();
    }
    cs.release();
    const out = ops.rmsnorm(gpu, x, this.norm, D);
    x.release();
    return out;
  }
}
