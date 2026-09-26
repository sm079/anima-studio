// Qwen-Image (Wan 2.1) VAE decoder for single images, NHWC on the GPU.
// The asset builder already collapsed the causal 3D convs to 2D (see tools/build_assets.py).

import * as ops from "../gpu/ops.js";

const LATENT_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921];
const LATENT_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916];

export class VAEDecoder {
  static async load(gpu, st) {
    const v = new VAEDecoder(gpu);
    const res = async (p) => ({
      n1: await st.vector(gpu, p + ".residual.0.gamma"),
      c1: await st.linear(gpu, p + ".residual.2."),
      n2: await st.vector(gpu, p + ".residual.3.gamma"),
      c2: await st.linear(gpu, p + ".residual.6."),
      sc: st.has(p + ".shortcut.weight") ? await st.linear(gpu, p + ".shortcut.") : null,
    });
    v.conv2 = await st.linear(gpu, "conv2.");
    v.conv1 = await st.linear(gpu, "decoder.conv1.");
    v.mid0 = await res("decoder.middle.0");
    v.attn = {
      norm: await st.vector(gpu, "decoder.middle.1.norm.gamma"),
      qkv: await st.linear(gpu, "decoder.middle.1.to_qkv."),
      proj: await st.linear(gpu, "decoder.middle.1.proj."),
    };
    v.mid2 = await res("decoder.middle.2");
    v.ups = [];
    for (let i = 0; i < 15; i++) {
      const p = `decoder.upsamples.${i}`;
      if (st.has(p + ".resample.1.weight")) v.ups.push({ up: await st.linear(gpu, p + ".resample.1.") });
      else v.ups.push({ res: await res(p) });
    }
    v.headNorm = await st.vector(gpu, "decoder.head.0.gamma");
    v.head = await st.linear(gpu, "decoder.head.2.");
    return v;
  }

  constructor(gpu) {
    this.gpu = gpu;
  }

  resblock(x, r, h, w) {
    const gpu = this.gpu;
    const cin = x.shape[1];
    let t = ops.channelNorm(gpu, x, r.n1, cin, true);
    const a = ops.conv3x3(gpu, t, r.c1, h, w);
    t.release();
    t = ops.channelNorm(gpu, a, r.n2, r.c1.N, true);
    a.release();
    const out = r.sc ? ops.linear(gpu, x, r.sc) : x;
    ops.conv3x3Into(gpu, t, r.c2, h, w, out);
    t.release();
    if (out !== x) x.release();
    return out;
  }

  // latent: Float32Array [16, h, w] in the model's normalized space -> RGBA Uint8ClampedArray
  async decode(latent, h, w, onStage) {
    const gpu = this.gpu;
    const hw = h * w;
    const z = new Float32Array(hw * 16);
    for (let c = 0; c < 16; c++) for (let i = 0; i < hw; i++) z[i * 16 + c] = latent[c * hw + i] * LATENT_STD[c] + LATENT_MEAN[c];
    let x = gpu.fromArray(z, [hw, 16]);
    let t = ops.linear(gpu, x, this.conv2);
    x.release();
    x = ops.conv3x3(gpu, t, this.conv1, h, w);
    t.release();
    x = this.resblock(x, this.mid0, h, w);

    // single-head spatial attention
    const C = 384;
    const n = ops.channelNorm(gpu, x, this.attn.norm, C, false);
    const qkv = ops.linear(gpu, n, this.attn.qkv);
    n.release();
    const a = ops.attention(gpu, { q: qkv, k: qkv, v: qkv, Lq: hw, Lk: hw, H: 1, D: C, ldq: 3 * C, ldk: 3 * C, ldv: 3 * C, qOff: 0, kOff: C, vOff: 2 * C });
    qkv.release();
    ops.linear(gpu, a, this.attn.proj, { out: x, resid: true });
    a.release();
    x = this.resblock(x, this.mid2, h, w);
    await gpu.sync();
    onStage?.(0);

    let H = h, W = w;
    for (let i = 0; i < this.ups.length; i++) {
      const u = this.ups[i];
      if (u.up) {
        const y = ops.conv3x3(gpu, x, u.up, H, W, true);
        x.release();
        x = y;
        H *= 2; W *= 2;
      } else {
        x = this.resblock(x, u.res, H, W);
      }
      await gpu.sync();
      onStage?.((i + 1) / this.ups.length);
    }
    t = ops.channelNorm(gpu, x, this.headNorm, x.shape[1], true);
    x.release();
    const rgb = ops.conv3x3(gpu, t, this.head, H, W);
    t.release();
    const px = await gpu.read(rgb);
    rgb.release();
    const img = new Uint8ClampedArray(H * W * 4);
    for (let i = 0; i < H * W; i++) {
      for (let c = 0; c < 3; c++) {
        const v = (Math.min(1, Math.max(-1, px[i * 3 + c])) + 1) / 2;
        img[i * 4 + c] = Math.round(v * 255);
      }
      img[i * 4 + 3] = 255;
    }
    return { data: img, width: W, height: H };
  }
}
