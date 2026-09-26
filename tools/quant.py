"""Offline weight quantizers producing ComfyUI / comfy_kitchen compatible tensors.

Two formats, matching the reference checkpoints:
  * int8_tensorwise + convrot  (per-output-channel int8 of the Hadamard-rotated weight)
  * asym_w4a8_int8 + convrot   (4-bit Lloyd-Max codebook, fp8 group scales, int8 grid)

The math is ported from comfy_kitchen/backends/eager (quantization.py, w4a8_int8.py), and the
tensors use ComfyUI's comfy_quant layout.
"""

from __future__ import annotations

import json
import math

import torch

CONVROT_GROUP = 256
W4_GROUP = 16

# Lloyd-Max-optimal 16 levels for a group-normalized Gaussian (comfy_kitchen _FIXED_LUT).
FIXED_LUT = (
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671,
)
_ALS_ITERS = 2
_GATE_KURTOSIS = -0.1


def hadamard(size: int, device, dtype=torch.float32) -> torch.Tensor:
    """Normalized regular (symmetric) Hadamard matrix, size must be a power of 4."""
    if size < 4 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"size must be a power of 4, got {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / math.sqrt(size)


def rotate(weight: torch.Tensor, group: int = CONVROT_GROUP) -> torch.Tensor:
    """W_rot = W @ H^T per contiguous group of `group` input features (H is symmetric)."""
    n, k = weight.shape
    h = hadamard(group, weight.device, torch.float32)
    return (weight.float().reshape(n, k // group, group) @ h.T).reshape(n, k)


def can_convrot(weight: torch.Tensor) -> bool:
    return weight.dim() == 2 and weight.shape[1] % CONVROT_GROUP == 0


def quant_meta(fmt: str, **extra) -> torch.Tensor:
    meta = {"format": fmt, **extra}
    return torch.tensor(list(json.dumps(meta).encode()), dtype=torch.uint8)


# ----------------------------------------------------------------------------- int8

def quantize_int8_convrot(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    rot = rotate(weight)
    scale = (rot.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
    q = (rot / scale).round().clamp(-128, 127).to(torch.int8)
    return {
        "weight": q.contiguous(),
        "weight_scale": scale.float().contiguous(),
        "comfy_quant": quant_meta("int8_tensorwise", convrot=True, convrot_groupsize=CONVROT_GROUP),
    }


def quantize_int8_rows(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-row int8 without rotation, for embedding tables (rows are gathered, never matmul'd)."""
    w = weight.float()
    scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
    return {
        "weight": (w / scale).round().clamp(-128, 127).to(torch.int8).contiguous(),
        "weight_scale": scale.contiguous(),
        "comfy_quant": quant_meta("int8_tensorwise"),
    }


def dequantize_int8_convrot(t: dict[str, torch.Tensor]) -> torch.Tensor:
    w = t["weight"].float() * t["weight_scale"].float()
    return rotate(w)  # H is symmetric and orthogonal: rotating twice is the identity


# ----------------------------------------------------------------------------- w4a8

def _assign_codes(normalized: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    last = codebook.numel() - 1
    pos = torch.searchsorted(codebook, normalized.contiguous())
    lo = (pos - 1).clamp(0, last)
    hi = pos.clamp(0, last)
    dlo = (normalized - codebook[lo]).abs()
    dhi = (normalized - codebook[hi]).abs()
    return torch.where(dhi < dlo, hi, lo)


def _fit_codebook(normalized: torch.Tensor, levels=16, iterations=25, sample_size=300000) -> torch.Tensor:
    samples = normalized.flatten()
    if samples.numel() > sample_size:
        g = torch.Generator(device=samples.device).manual_seed(0)
        samples = samples[torch.randint(0, samples.numel(), (sample_size,), device=samples.device, generator=g)]
    samples = samples.float()
    cb = torch.quantile(samples, torch.linspace(0, 1, levels, device=samples.device))
    for _ in range(iterations):
        a = (samples.unsqueeze(-1) - cb).abs().argmin(-1)
        upd = cb.clone()
        for i in range(levels):
            sel = a == i
            if sel.any():
                upd[i] = samples[sel].mean()
        cb = upd
    return cb.contiguous()


def _codebook_for(normalized: torch.Tensor) -> torch.Tensor:
    x = normalized.flatten().float()
    if x.numel() > (1 << 19):
        g = torch.Generator(device=x.device).manual_seed(0)
        x = x[torch.randint(0, x.numel(), (1 << 19,), device=x.device, generator=g)]
    kurt = ((x - x.mean()) / (x.std() + 1e-9)).pow(4).mean() - 3.0
    if kurt.item() <= _GATE_KURTOSIS:
        return torch.tensor(FIXED_LUT, device=normalized.device, dtype=torch.float32)
    return _fit_codebook(normalized)


def _grid_levels(codebook: torch.Tensor, s_rel: torch.Tensor) -> torch.Tensor:
    """int8 value each codebook level decodes to, per group: round(clamp(level * s_rel))."""
    return (codebook.view(1, 1, -1) * s_rel.float().unsqueeze(-1)).round().clamp(-127, 127)


def quantize_w4a8(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    n, k = weight.shape
    groups = k // W4_GROUP
    rot = rotate(weight)
    gw = rot.view(n, groups, W4_GROUP)

    amax = gw.abs().amax(dim=-1, keepdim=True)
    gscale = amax.clamp(min=1e-8)
    codebook = _codebook_for(gw / gscale)
    q = _assign_codes(gw / gscale, codebook)
    for _ in range(_ALS_ITERS):
        lv = codebook[q]
        gscale = ((gw * lv).sum(-1, keepdim=True) / (lv * lv).sum(-1, keepdim=True).clamp(min=1e-8)).clamp(min=1e-8)
        q = _assign_codes(gw / gscale, codebook)
    shifted = codebook[q] * gscale

    s_channel = (shifted.abs().amax(dim=(1, 2)) / 127.0).clamp(min=1e-8)
    s_rel = (gscale.squeeze(-1) / s_channel.unsqueeze(1)).float().to(torch.float8_e4m3fn)

    # Final assignment against the levels the kernels will actually decode (fp8 scale, int8 grid).
    levels = _grid_levels(codebook, s_rel)  # [n, groups, 16], sorted
    target = (gw / s_channel.view(-1, 1, 1)).reshape(n * groups, W4_GROUP).contiguous()
    lvf = levels.reshape(n * groups, 16).contiguous()
    pos = torch.searchsorted(lvf, target)
    lo = (pos - 1).clamp(0, 15)
    hi = pos.clamp(0, 15)
    dlo = (target - torch.gather(lvf, 1, lo)).abs()
    dhi = (target - torch.gather(lvf, 1, hi)).abs()
    codes = torch.where(dhi < dlo, hi, lo).to(torch.int32).view(n, k)

    packed = ((codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)).to(torch.uint8).view(torch.int8)
    return {
        "weight": packed.contiguous(),
        "weight_s_rel": s_rel.contiguous(),
        "weight_s_channel": s_channel.float().contiguous(),
        "weight_codebook": codebook.float().contiguous(),
        "comfy_quant": quant_meta("asym_w4a8_int8", group_size=W4_GROUP, convrot=True, convrot_groupsize=CONVROT_GROUP),
    }


def dequantize_w4a8(t: dict[str, torch.Tensor]) -> torch.Tensor:
    packed = t["weight"].view(torch.uint8).to(torch.int32)
    n, half = packed.shape
    k = half * 2
    codes = torch.empty(n, k, dtype=torch.int32, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = (packed >> 4) & 0xF
    vals = t["weight_codebook"].float()[codes].view(n, k // W4_GROUP, W4_GROUP)
    int8 = (vals * t["weight_s_rel"].float().unsqueeze(-1)).round().clamp(-127, 127)
    w = int8.view(n, k) * t["weight_s_channel"].float().view(n, 1)
    return rotate(w)


def dequantize(t: dict[str, torch.Tensor]) -> torch.Tensor:
    meta = json.loads(bytes(t["comfy_quant"].tolist()).decode())
    if meta["format"] == "int8_tensorwise":
        if not meta.get("convrot"):
            return t["weight"].float() * t["weight_scale"].float()
        return dequantize_int8_convrot(t)
    return dequantize_w4a8(t)
