"""PyTorch reference implementation of the Anima pipeline, reading the built web assets.

It exists to (1) validate the architecture port end-to-end by producing an image, and
(2) dump intermediate tensors that the WebGPU engine is checked against (tools/check.html).

  python tools/reference.py --model turbo-v1.1 --dit int8 --te int8 --prompt "..." --seed 42 --out ref.png [--dump dumps/]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
LATENT_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916]


class Weights:
    """Loads a web asset file; quantized linears are dequantized to fp32 on access."""

    def __init__(self, path: str, prefix: str = ""):
        self.f = safe_open(path, "pt", device="cpu")
        self.keys = set(self.f.keys())
        self.prefix = prefix

    def __call__(self, name: str) -> torch.Tensor:
        k = self.prefix + name
        base = k[: -len("weight")] if k.endswith("weight") else None
        if base is not None and base + "comfy_quant" in self.keys:
            t = {s: self.f.get_tensor(base + s).to(DEV) for s in ("weight", "weight_scale", "weight_s_rel", "weight_s_channel", "weight_codebook", "comfy_quant") if base + s in self.keys}
            t["comfy_quant"] = t["comfy_quant"].cpu()
            return quant.dequantize(t)
        return self.f.get_tensor(k).to(DEV).float()

    def has(self, name: str) -> bool:
        return self.prefix + name in self.keys


class LoraMerged:
    """Wraps Weights and merges LoRAs into matching linear weights: W += s * alpha/r * B @ A.
    Same key conventions as app/lora.js (PEFT lora_A/B or kohya lora_down/up + alpha)."""

    SUFFIXES = [(".lora_A.weight", "A"), (".lora_B.weight", "B"), (".lora_down.weight", "A"), (".lora_up.weight", "B"), (".alpha", "alpha")]
    PREFIXES = ["lora_unet_", "model.diffusion_model.", "diffusion_model.", "lora_te_", "text_encoder."]

    def __init__(self, base: Weights, loras):
        self.base = base
        self.keys = base.keys
        self.prefix = base.prefix
        self.deltas = {}
        for path, strength in loras:
            f = safe_open(path, "pt", device="cpu")
            groups = {}
            for k in f.keys():
                for suf, role in self.SUFFIXES:
                    if k.endswith(suf):
                        groups.setdefault(k[: -len(suf)], {})[role] = k
            for module, ks in groups.items():
                name = module
                for p in self.PREFIXES:
                    if name.startswith(p):
                        name = name[len(p):]
                        break
                A = f.get_tensor(ks["A"]).float().to(DEV)
                B = f.get_tensor(ks["B"]).float().to(DEV)
                scale = strength * (f.get_tensor(ks["alpha"]).float().item() / A.shape[0] if "alpha" in ks else 1.0)
                key = name.replace(".", "_")
                self.deltas.setdefault(key, []).append((scale, B, A))  # low-rank; merged on access

    def __call__(self, name):
        w = self.base(name)
        if name.endswith(".weight"):
            for scale, B, A in self.deltas.get(name[: -len(".weight")].replace(".", "_"), []):
                w = w + scale * (B @ A)
        return w

    def has(self, name):
        return self.base.has(name)


def rms_norm(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope_cos_sin(positions, head_dim, theta):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=DEV).float() / head_dim))
    f = positions.float()[:, None] * inv[None]
    emb = torch.cat((f, f), -1)
    return emb.cos(), emb.sin()


def attention(q, k, v, causal=False):
    # q [H, Lq, D], k/v [H, Lk, D]
    return F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=causal)[0]


# ----------------------------------------------------------------------------- text encoder

def qwen3(W: Weights, ids: list[int]) -> torch.Tensor:
    x = W("model.embed_tokens.weight")[torch.tensor(ids, device=DEV)]
    L = x.shape[0]
    cos, sin = rope_cos_sin(torch.arange(L, device=DEV), 128, 1e6)
    for i in range(28):
        p = f"model.layers.{i}."
        h = rms_norm(x, W(p + "input_layernorm.weight"))
        q = (h @ W(p + "self_attn.q_proj.weight").T).view(L, 16, 128).transpose(0, 1)
        k = (h @ W(p + "self_attn.k_proj.weight").T).view(L, 8, 128).transpose(0, 1)
        v = (h @ W(p + "self_attn.v_proj.weight").T).view(L, 8, 128).transpose(0, 1)
        q = rms_norm(q, W(p + "self_attn.q_norm.weight"))
        k = rms_norm(k, W(p + "self_attn.k_norm.weight"))
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        k = k.repeat_interleave(2, 0)
        v = v.repeat_interleave(2, 0)
        a = attention(q, k, v, causal=True).transpose(0, 1).reshape(L, 2048)
        x = x + a @ W(p + "self_attn.o_proj.weight").T
        h = rms_norm(x, W(p + "post_attention_layernorm.weight"))
        g = h @ W(p + "mlp.gate_proj.weight").T
        u = h @ W(p + "mlp.up_proj.weight").T
        x = x + (F.silu(g) * u) @ W(p + "mlp.down_proj.weight").T
    return rms_norm(x, W("model.norm.weight"))


def llm_adapter(W: Weights, src: torch.Tensor, t5_ids: list[int], t5_weights: list[float] | None = None) -> torch.Tensor:
    p0 = "llm_adapter."
    x = W(p0 + "embed.weight")[torch.tensor(t5_ids, device=DEV)]
    Lt, Ls = x.shape[0], src.shape[0]
    cos_t, sin_t = rope_cos_sin(torch.arange(Lt, device=DEV), 64, 1e4)
    cos_s, sin_s = rope_cos_sin(torch.arange(Ls, device=DEV), 64, 1e4)

    def attn(p, xq, ctx, cq, sq, ck, sk):
        Lq, Lk = xq.shape[0], ctx.shape[0]
        q = rms_norm((xq @ W(p + "q_proj.weight").T).view(Lq, 16, 64), W(p + "q_norm.weight")).transpose(0, 1)
        k = rms_norm((ctx @ W(p + "k_proj.weight").T).view(Lk, 16, 64), W(p + "k_norm.weight")).transpose(0, 1)
        v = (ctx @ W(p + "v_proj.weight").T).view(Lk, 16, 64).transpose(0, 1)
        q = q * cq + rotate_half(q) * sq
        k = k * ck + rotate_half(k) * sk
        return attention(q, k, v).transpose(0, 1).reshape(Lq, 1024) @ W(p + "o_proj.weight").T

    for i in range(6):
        p = f"{p0}blocks.{i}."
        x = x + attn(p + "self_attn.", rms_norm(x, W(p + "norm_self_attn.weight")), rms_norm(x, W(p + "norm_self_attn.weight")), cos_t, sin_t, cos_t, sin_t)
        x = x + attn(p + "cross_attn.", rms_norm(x, W(p + "norm_cross_attn.weight")), src, cos_t, sin_t, cos_s, sin_s)
        h = rms_norm(x, W(p + "norm_mlp.weight"))
        h = F.gelu(h @ W(p + "mlp.0.weight").T + W(p + "mlp.0.bias"))
        x = x + h @ W(p + "mlp.2.weight").T + W(p + "mlp.2.bias")
    out = rms_norm(x @ W(p0 + "out_proj.weight").T + W(p0 + "out_proj.bias"), W(p0 + "norm.weight"))
    if t5_weights is not None:
        out = out * torch.tensor(t5_weights, device=DEV)[:, None]
    if out.shape[0] < 512:
        out = F.pad(out, (0, 0, 0, 512 - out.shape[0]))
    return out


# ----------------------------------------------------------------------------- DiT

def dit_rope(H, W_):
    """VideoRopePosition3DEmb for T=1: 64 frequency slots per token (22 t, 21 h, 21 w)."""
    head_dim = 128
    dim_h = head_dim // 6 * 2
    dim_t = head_dim - 2 * dim_h
    sp = torch.arange(0, dim_h, 2, device=DEV)[: dim_h // 2].float() / dim_h
    tp = torch.arange(0, dim_t, 2, device=DEV)[: dim_t // 2].float() / dim_t
    h_theta = 10000.0 * 4.0 ** (dim_h / (dim_h - 2))
    t_theta = 10000.0 * 1.0
    hf = 1.0 / h_theta**sp
    tf = 1.0 / t_theta**tp
    hh, ww = torch.meshgrid(torch.arange(H, device=DEV).float(), torch.arange(W_, device=DEV).float(), indexing="ij")
    ang = torch.cat([torch.zeros(H * W_, tf.numel(), device=DEV) * tf, hh.reshape(-1, 1) * hf, ww.reshape(-1, 1) * hf], dim=1)  # [L, 64]
    return ang.cos(), ang.sin()


def layer_norm(x, eps=1e-6):
    return F.layer_norm(x, (x.shape[-1],), eps=eps)


def dit_forward(W: Weights, x: torch.Tensor, sigma: float, ctx: torch.Tensor, dump=None) -> torch.Tensor:
    """x: latent [16, h, w] (normalized latent space). Returns velocity [16, h, w]."""
    C, h, w = x.shape
    H, Wp = h // 2, w // 2
    L = H * Wp
    xin = torch.cat([x, torch.zeros(1, h, w, device=DEV)], 0)  # padding-mask channel
    patches = xin.view(17, H, 2, Wp, 2).permute(1, 3, 0, 2, 4).reshape(L, 68)  # (c m n)
    x = patches @ W("x_embedder.proj.1.weight").T

    half = 1024
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=DEV).float() / half)
    temb = sigma * freqs
    temb = torch.cat([temb.cos(), temb.sin()])[None]  # [1, 2048]
    adaln_lora = F.silu(temb @ W("t_embedder.1.linear_1.weight").T) @ W("t_embedder.1.linear_2.weight").T  # [1, 6144]
    emb = rms_norm(temb, W("t_embedding_norm.weight"))
    if dump is not None:
        dump["dit_x_embed"] = x
        dump["dit_emb"] = emb
        dump["dit_adaln_lora"] = adaln_lora

    cos, sin = dit_rope(H, Wp)
    cos = torch.cat([cos, cos], -1)
    sin = torch.cat([sin, sin], -1)

    def mod(p):
        m = F.silu(emb) @ W(p + ".1.weight").T @ W(p + ".2.weight").T + adaln_lora
        return m.chunk(3, -1)

    def attn(p, xq, c, rope):
        Lq, Lk = xq.shape[0], c.shape[0]
        q = rms_norm((xq @ W(p + "q_proj.weight").T).view(Lq, 16, 128), W(p + "q_norm.weight")).transpose(0, 1)
        k = rms_norm((c @ W(p + "k_proj.weight").T).view(Lk, 16, 128), W(p + "k_norm.weight")).transpose(0, 1)
        v = (c @ W(p + "v_proj.weight").T).view(Lk, 16, 128).transpose(0, 1)
        if rope:
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        return attention(q, k, v).transpose(0, 1).reshape(Lq, 2048) @ W(p + "output_proj.weight").T

    for i in range(28):
        p = f"blocks.{i}."
        sh, sc, g = mod(p + "adaln_modulation_self_attn")
        n = layer_norm(x) * (1 + sc) + sh
        x = x + g * attn(p + "self_attn.", n, n, True)
        sh, sc, g = mod(p + "adaln_modulation_cross_attn")
        n = layer_norm(x) * (1 + sc) + sh
        x = x + g * attn(p + "cross_attn.", n, ctx, False)
        sh, sc, g = mod(p + "adaln_modulation_mlp")
        n = layer_norm(x) * (1 + sc) + sh
        x = x + g * (F.gelu(n @ W(p + "mlp.layer1.weight").T) @ W(p + "mlp.layer2.weight").T)
        if dump is not None and i == 0:
            dump["dit_block0"] = x

    m = F.silu(emb) @ W("final_layer.adaln_modulation.1.weight").T @ W("final_layer.adaln_modulation.2.weight").T + adaln_lora[:, :4096]
    sh, sc = m.chunk(2, -1)
    x = (layer_norm(x) * (1 + sc) + sh) @ W("final_layer.linear.weight").T  # [L, 64] = (p1 p2 c)
    return x.view(H, Wp, 2, 2, 16).permute(4, 0, 2, 1, 3).reshape(16, h, w)


# ----------------------------------------------------------------------------- VAE decoder

def vae_decode(W: Weights, z: torch.Tensor) -> torch.Tensor:
    """z: [16, h, w] raw (de-normalized) latent -> image [3, 8h, 8w] in [0, 1]."""

    def conv(x, p, pad=True):
        w = W(p + ".weight")
        b = W(p + ".bias")
        if w.dim() == 2:
            return torch.einsum("chw,oc->ohw", x, w) + b[:, None, None]
        w = w.permute(0, 3, 1, 2)  # OHWI -> OIHW
        return F.conv2d(x[None], w, b, padding=1 if pad else 0)[0]

    def norm(x, p):
        return F.normalize(x, dim=0) * math.sqrt(x.shape[0]) * W(p + ".gamma")[:, None, None]

    def res(x, p):
        hdn = conv(F.silu(norm(x, p + ".residual.0")), p + ".residual.2")
        hdn = conv(F.silu(norm(hdn, p + ".residual.3")), p + ".residual.6")
        sc = conv(x, p + ".shortcut") if W.has(p + ".shortcut.weight") else x
        return hdn + sc

    def attn_block(x, p):
        c, hh, ww = x.shape
        n = norm(x, p + ".norm")
        qkv = conv(n, p + ".to_qkv")
        q, k, v = qkv.view(3, c, hh * ww).transpose(1, 2)  # [HW, c]
        a = attention(q[None], k[None], v[None])[0].T.reshape(c, hh, ww)
        return conv(a, p + ".proj") + x

    x = conv(z, "conv2")
    x = conv(x, "decoder.conv1")
    x = res(x, "decoder.middle.0")
    x = attn_block(x, "decoder.middle.1")
    x = res(x, "decoder.middle.2")
    for i in range(15):
        p = f"decoder.upsamples.{i}"
        if W.has(p + ".resample.1.weight"):
            x = x.repeat_interleave(2, 1).repeat_interleave(2, 2)
            x = conv(x, p + ".resample.1")
        else:
            x = res(x, p)
    x = conv(F.silu(norm(x, "decoder.head.0")), "decoder.head.2")
    return ((x.clamp(-1, 1) + 1) / 2).clamp(0, 1)


# ----------------------------------------------------------------------------- sampling

def sigmas_simple(steps: int, shift: float) -> list[float]:
    table = [shift * t / (1 + (shift - 1) * t) for t in (np.arange(1, 1001) / 1000.0)]
    ss = 1000 / steps
    return [float(np.float32(table[-(1 + int(x * ss))])) for x in range(steps)] + [0.0]


def torch_cpu_noise(seed: int, shape) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32)


def tokenize(model_dir: str, text: str):
    from tokenizers import Tokenizer

    qt = Tokenizer.from_file(os.path.join(model_dir, "tokenizers/qwen/tokenizer.json"))
    tt = Tokenizer.from_file(os.path.join(model_dir, "tokenizers/t5/tokenizer.json"))
    q = qt.encode(text, add_special_tokens=False).ids or [151643]
    t = tt.encode(text, add_special_tokens=True).ids
    return q, t


def open_weights(models_dir: str, model: str, dit: str, te: str):
    manifest = json.load(open(os.path.join(models_dir, "manifest.json")))
    m = next(x for x in manifest["models"] if x["id"] == model)
    return (
        m,
        Weights(os.path.join(models_dir, manifest["te"][te]["path"])),
        Weights(os.path.join(models_dir, m["dit"][dit]["path"]), "model.diffusion_model."),
        Weights(os.path.join(models_dir, manifest["vae"]["path"])),
    )


@torch.no_grad()
def generate(models_dir, Wte, Wd, Wv, prompt, negative="", width=768, height=768, steps=8, cfg=1.0, seed=42, dump=None, log=print):
    """Deterministic euler sampling. Returns (image [3,H,W] in [0,1], ctx, final latent)."""
    def encode(text):
        q, t = tokenize(models_dir, text)
        hs = qwen3(Wte, q)
        return hs, llm_adapter(Wd, hs, t), q, t

    hs, ctx, qids, tids = encode(prompt)
    ctx_neg = encode(negative)[1] if cfg != 1.0 else None
    if dump is not None:
        dump.update(te_hidden=hs, ctx=ctx, qwen_ids=qids, t5_ids=tids)

    h, w = height // 8, width // 8
    x = torch_cpu_noise(seed, (1, 16, 1, h, w))[0, :, 0].to(DEV)
    if dump is not None:
        dump["noise"] = x.clone()
    sig = sigmas_simple(steps, 3.0)
    x = x * sig[0]
    for i in range(steps):
        s = sig[i]
        v = dit_forward(Wd, x, s, ctx, dump if i == 0 else None)
        if ctx_neg is not None:
            vn = dit_forward(Wd, x, s, ctx_neg)
            v = vn + (v - vn) * cfg
        den = x - v * s
        if dump is not None and i == 0:
            dump["denoised0"] = den
        x = den if sig[i + 1] == 0 else x + (x - den) / s * (sig[i + 1] - s)
        log(f"step {i + 1}/{steps} sigma {s:.4f}")

    mean = torch.tensor(LATENT_MEAN, device=DEV)[:, None, None]
    std = torch.tensor(LATENT_STD, device=DEV)[:, None, None]
    img = vae_decode(Wv, x * std + mean)
    if dump is not None:
        dump["latent_final"] = x
        dump["image"] = img
    return img, ctx, x


def save_png(img, path):
    from PIL import Image

    Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
    ap.add_argument("--model", default="turbo-v1.1")
    ap.add_argument("--dit", default="int8", choices=["bf16", "int8", "w4a8"])
    ap.add_argument("--te", default="int8", choices=["bf16", "int8", "w4a8"])
    ap.add_argument("--prompt", default="masterpiece, best quality, score_7, safe, 1girl, solo, silver hair, blue eyes, school uniform, cherry blossoms, smile, looking at viewer")
    ap.add_argument("--negative", default="worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ref.png")
    ap.add_argument("--dump", default=None, help="directory to write intermediate tensors (.bin f32 + index.json)")
    ap.add_argument("--lora", action="append", default=[], help="path:strength (repeatable); merged into the DiT/adapter weights")
    args = ap.parse_args()

    m, Wte, Wd, Wv = open_weights(args.models, args.model, args.dit, args.te)
    loras = [(p.rsplit(":", 1)[0], float(p.rsplit(":", 1)[1])) for p in args.lora]
    if loras:
        Wd = LoraMerged(Wd, loras)
    steps = args.steps or m["defaults"]["steps"]
    cfg = args.cfg if args.cfg is not None else m["defaults"]["cfg"]
    dump = {} if args.dump else None
    img, _, _ = generate(args.models, Wte, Wd, Wv, args.prompt, args.negative, args.width, args.height, steps, cfg, args.seed, dump)
    save_png(img, args.out)
    print("saved", args.out)
    if dump is not None:
        os.makedirs(args.dump, exist_ok=True)
        index = {}
        qids, tids = dump.pop("qwen_ids"), dump.pop("t5_ids")
        for k, t in dump.items():
            a = t.detach().float().cpu().contiguous().numpy()
            a.tofile(os.path.join(args.dump, k + ".bin"))
            index[k] = list(a.shape)
        meta = {"prompt": args.prompt, "negative": args.negative, "seed": args.seed, "width": args.width, "height": args.height, "steps": steps, "cfg": cfg,
                "model": args.model, "dit": args.dit, "te": args.te, "qwen_ids": qids, "t5_ids": tids, "tensors": index,
                "loras": [{"path": os.path.relpath(p, os.path.dirname(os.path.abspath(args.dump))).replace("\\", "/"), "strength": s} for p, s in loras]}
        json.dump(meta, open(os.path.join(args.dump, "index.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
