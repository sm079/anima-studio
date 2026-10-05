"""One-time offline build of the web assets served to the browser.

Downloads the official Anima split files, quantizes and re-lays out every weight so the
browser does nothing but download-once + run. The diffusion model and the text encoder are
built in several precisions so users can trade quality for download size / VRAM:

  models/
    manifest.json                               components, sizes, sampler defaults
    anima-<variant>.<bf16|int8|w4a8>.safetensors  DiT + LLM adapter (ComfyUI comfy_quant format)
    qwen3-0.6b.<bf16|int8|w4a8>.safetensors      text encoder
    qwen-image-vae-decoder.safetensors          decoder only, causal 3D convs collapsed to 2D, OHWI layout
    tokenizers/{qwen,t5}/tokenizer.json

Precisions:
  bf16  original weights
  int8  block linears int8 + ConvRot (per-channel scales)
  w4a8  "mixed": block linears W4A8 + ConvRot, first and last block int8
        (as in the reference *_w4a8_mixed checkpoints)
  Quantized text encoders keep their last layer bf16 (the most sensitive one) and store the
  embedding table as per-row int8.

Every released version can be built (see VARIANTS). Turbo models are distilled for CFG 1 and
8-12 steps; Base and Aesthetic use classifier-free guidance with a negative prompt (two DiT
passes per step) and more steps. Each model's sampler defaults go into the manifest.

Usage:
  python tools/build_assets.py                                    # turbo v1.1, every precision
  python tools/build_assets.py --variants all                     # every version
  python tools/build_assets.py --variants turbo-v1.1 turbo-v1.0 --dit int8 w4a8 --te int8
  python tools/measure_quality.py                                 # then add quality scores to the manifest
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant  # noqa: E402

REPO = "circlestone-labs/Anima"
TOKENIZER_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"

# Released versions, in the order the app lists them. family: turbo (distilled, CFG 1) |
# aesthetic | base (CFG with a negative prompt). Defaults follow the model card and its example
# workflow.
VARIANTS = {
    "turbo-v1.1": {"file": "anima-turbo-v1.1.safetensors", "label": "Anima Turbo v1.1", "family": "turbo", "steps": 8, "cfg": 1.0, "sampler": "er_sde", "scheduler": "beta"},
    "turbo-v1.0": {"file": "anima-turbo-v1.0.safetensors", "label": "Anima Turbo v1.0", "family": "turbo", "steps": 8, "cfg": 1.0, "sampler": "er_sde", "scheduler": "beta"},
    "aesthetic-v1.1": {"file": "anima-aesthetic-v1.1.safetensors", "label": "Anima Aesthetic v1.1", "family": "aesthetic", "steps": 30, "cfg": 4.0, "sampler": "er_sde", "scheduler": "simple"},
    "aesthetic-v1.0": {"file": "anima-aesthetic-v1.0.safetensors", "label": "Anima Aesthetic v1.0", "family": "aesthetic", "steps": 30, "cfg": 4.0, "sampler": "er_sde", "scheduler": "simple"},
    "aesthetic-v1.0b": {"file": "anima-aesthetic-v1.0b.safetensors", "label": "Anima Aesthetic v1.0b", "family": "aesthetic", "steps": 30, "cfg": 4.0, "sampler": "er_sde", "scheduler": "simple"},
    "base-v1.0": {"file": "anima-base-v1.0.safetensors", "label": "Anima Base v1.0", "family": "base", "steps": 30, "cfg": 4.5, "sampler": "er_sde", "scheduler": "simple"},
}

DIT_PREFIX = "model.diffusion_model."

# DiT block linears that get quantized. Everything else (adaLN, embedders, final layer,
# LLM adapter, norms) stays bf16: small, and the most precision-sensitive parts.
DIT_QUANT_RE = re.compile(r"^" + re.escape(DIT_PREFIX) + r"blocks\.(\d+)\.(self_attn\.(q|k|v|output)_proj|cross_attn\.(q|k|v|output)_proj|mlp\.layer[12])\.weight$")
TE_QUANT_RE = re.compile(r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")


def fetch(repo: str, path: str, src_dir: str | None) -> str:
    if src_dir:
        local = os.path.join(src_dir, os.path.basename(path))
        if os.path.exists(local):
            return local
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path)


def fmt_size(n: int) -> str:
    return f"{n / 2**30:.2f} GiB" if n > 2**30 else f"{n / 2**20:.1f} MiB"


# Base v1.0 is saved with the training script's "net." prefix instead of ComfyUI's
SRC_PREFIXES = ("net.",)


def build_dit(src: str, out: str, mode: str, device: str, w4_attn: bool = True) -> None:
    with safe_open(src, "pt") as f:
        src_keys = list(f.keys())
    # source key -> key in the web asset (always DIT_PREFIX + ...)
    rename = {}
    for k in src_keys:
        p = next((p for p in SRC_PREFIXES if k.startswith(p)), None)
        rename[k] = DIT_PREFIX + k[len(p):] if p else k
    if mode == "bf16" and all(rename[k] == k for k in src_keys):
        shutil.copyfile(src, out)
        return
    tensors: dict[str, torch.Tensor] = {}
    n_blocks = 0
    with safe_open(src, "pt") as f:
        for k in src_keys:
            m = re.match(re.escape(DIT_PREFIX) + r"blocks\.(\d+)\.", rename[k])
            if m:
                n_blocks = max(n_blocks, int(m.group(1)) + 1)
        for i, sk in enumerate(src_keys):
            t = f.get_tensor(sk)
            k = rename[sk]
            if mode == "bf16":
                tensors[k] = t
                continue
            m = DIT_QUANT_RE.search(k)
            if m and quant.can_convrot(t):
                block = int(m.group(1))
                base = k[: -len("weight")]
                # "mixed" w4a8: first and last block stay int8, as in the reference mixed checkpoints
                use_w4 = mode == "w4a8" and 0 < block < n_blocks - 1 and (w4_attn or ".mlp." in k)
                q = (quant.quantize_w4a8 if use_w4 else quant.quantize_int8_convrot)(t.to(device))
                for suffix, v in q.items():
                    tensors[base + suffix] = v.cpu()
            else:
                tensors[k] = t.to(torch.bfloat16) if t.is_floating_point() else t
            if i % 50 == 0:
                print(f"  dit {mode}: {i}/{len(src_keys)}", flush=True)
    save_file(tensors, out, metadata={"format": "pt", "anima_web": "dit", "quant": mode})


def build_te(src: str, out: str, mode: str, device: str) -> None:
    if mode == "bf16":
        shutil.copyfile(src, out)
        return
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(src, "pt") as f:
        n_layers = 1 + max(int(m.group(1)) for k in f.keys() if (m := re.search(r"layers\.(\d+)\.", k)))
        for k in f.keys():
            t = f.get_tensor(k)
            base = k[: -len("weight")]
            layer = int(m.group(1)) if (m := re.search(r"layers\.(\d+)\.", k)) else -1
            if layer == n_layers - 1:
                # the last layer feeds the output norm directly; quantizing it causes most of
                # the text-conditioning error (12% -> 3% hidden-state error for int8), ~30 MB bf16
                tensors[k] = t.to(torch.bfloat16)
                continue
            m = TE_QUANT_RE.search(k)
            if m and quant.can_convrot(t):
                use_w4 = mode == "w4a8" and layer > 0
                q = (quant.quantize_w4a8 if use_w4 else quant.quantize_int8_convrot)(t.to(device))
            elif k.endswith("embed_tokens.weight"):
                q = quant.quantize_int8_rows(t.to(device))
            else:
                tensors[k] = t.to(torch.bfloat16)
                continue
            for suffix, v in q.items():
                tensors[base + suffix] = v.cpu()
    save_file(tensors, out, metadata={"format": "pt", "anima_web": "te", "quant": mode})


def build_vae_decoder(src: str, out: str) -> None:
    """Keep the decoder only. For a single image (T=1, no feature cache) every CausalConv3d
    sees zero causal padding, so only its last temporal kernel slice contributes, and the
    upsample3d time_conv is never run. Conv weights are stored OHWI so the NHWC implicit-GEMM
    kernel reads K = (ky*kw + kx)*Cin + ci contiguously."""
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(src, "pt") as f:
        for k in f.keys():
            if not (k.startswith("decoder.") or k.startswith("conv2.")) or ".time_conv." in k:
                continue
            t = f.get_tensor(k).float()
            if k.endswith(".weight") and t.dim() == 5:
                t = t[:, :, -1]  # [O, I, kh, kw]
            if k.endswith(".weight") and t.dim() == 4:
                t = t.permute(0, 2, 3, 1).contiguous()  # OHWI
                if t.shape[1] == 1 and t.shape[2] == 1:
                    t = t.reshape(t.shape[0], t.shape[3])
            if k.endswith(".gamma"):
                t = t.reshape(-1)
            tensors[k] = t.to(torch.bfloat16).contiguous()
    save_file(tensors, out, metadata={"format": "pt", "anima_web": "vae_decoder"})


QUANTS = ["bf16", "int8", "w4a8"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["turbo-v1.1"], choices=[*VARIANTS, "all"])
    ap.add_argument("--dit", nargs="+", default=QUANTS, choices=QUANTS, help="DiT precisions to build")
    ap.add_argument("--te", nargs="+", default=QUANTS, choices=QUANTS, help="text encoder precisions to build")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
    ap.add_argument("--src", default=None, help="directory holding already-downloaded split files")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--w4-attn", choices=["w4", "int8"], default="w4",
                    help="attention linears in the w4a8 DiT: w4 (reference mixed scheme, 1.54 GiB) or int8 "
                         "(4-bit MLPs only, 1.86 GiB; fewer prompt reinterpretations)")
    args = ap.parse_args()
    if "all" in args.variants:
        args.variants = list(VARIANTS)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    if manifest.get("version") != 2:
        manifest = {"version": 2, "models": [], "te": {}}

    def need(path: str) -> bool:
        return args.force or not os.path.exists(path)

    def entry(name: str) -> dict:
        return {"path": name, "size": os.path.getsize(os.path.join(out_dir, name))}

    te_src = None
    for mode in args.te:
        name = f"qwen3-0.6b.{mode}.safetensors"
        if need(os.path.join(out_dir, name)):
            te_src = te_src or fetch(REPO, "split_files/text_encoders/qwen_3_06b_base.safetensors", args.src)
            print(f"text encoder {mode} ...")
            build_te(te_src, os.path.join(out_dir, name), mode, args.device)
        manifest["te"][mode] = {**manifest["te"].get(mode, {}), **entry(name)}
        print(f"  {name}: {fmt_size(manifest['te'][mode]['size'])}")

    vae_name = "qwen-image-vae-decoder.safetensors"
    if need(os.path.join(out_dir, vae_name)):
        print("vae decoder ...")
        build_vae_decoder(fetch(REPO, "split_files/vae/qwen_image_vae.safetensors", args.src), os.path.join(out_dir, vae_name))
    manifest["vae"] = entry(vae_name)

    for sub, folder in (("qwen", "tokenizer"), ("t5", "t5_tokenizer")):
        d = os.path.join(out_dir, "tokenizers", sub)
        os.makedirs(d, exist_ok=True)
        for fn in ("tokenizer.json", "tokenizer_config.json"):
            if need(os.path.join(d, fn)):
                src = os.path.join(args.src, "tok", sub, fn) if args.src else None
                if not (src and os.path.exists(src)):
                    from huggingface_hub import hf_hub_download

                    src = hf_hub_download(TOKENIZER_REPO, f"{folder}/{fn}")
                shutil.copyfile(src, os.path.join(d, fn))
    manifest["tokenizers"] = {"qwen": "tokenizers/qwen/tokenizer.json", "t5": "tokenizers/t5/tokenizer.json"}

    for variant in args.variants:
        v = VARIANTS[variant]
        model = next((m for m in manifest["models"] if m["id"] == variant), None)
        if model is None:
            model = {"id": variant, "dit": {}}
            manifest["models"].append(model)
        model.update({"label": v["label"], "family": v["family"], "defaults": {"steps": v["steps"], "cfg": v["cfg"], "sampler": v["sampler"], "scheduler": v["scheduler"], "shift": 3.0}})
        src = None
        for mode in args.dit:
            name = f"anima-{variant}.{mode}.safetensors"
            path = os.path.join(out_dir, name)
            if need(path):
                src = src or fetch(REPO, f"split_files/diffusion_models/{v['file']}", args.src)
                print(f"dit {variant} {mode} ...")
                build_dit(src, path, mode, args.device, w4_attn=args.w4_attn == "w4")
            model["dit"][mode] = {**model["dit"].get(mode, {}), **entry(name)}
            print(f"  {name}: {fmt_size(os.path.getsize(path))}")

    order = list(VARIANTS)
    manifest["models"].sort(key=lambda m: order.index(m["id"]) if m["id"] in order else len(order))
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print("wrote", manifest_path)


if __name__ == "__main__":
    main()
