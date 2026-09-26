"""Measure how closely each quantized precision reproduces the bf16 model and store the
numbers in models/manifest.json.

For a few fixed prompts/seeds (deterministic euler sampling) every DiT/TE precision is compared
with the all-bf16 output:
  * image PSNR (dB) vs bf16  -- per DiT precision (bf16 TE) and per TE precision (bf16 DiT)
  * text-conditioning relative L2 error vs the bf16 text encoder

  python tools/measure_quality.py [--model turbo-v1.1] [--out out/quality]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R  # noqa: E402

PROMPTS = [
    ("masterpiece, best quality, score_7, safe, 1girl, solo, silver hair, blue eyes, school uniform, cherry blossoms, smile, looking at viewer", 42),
    ("masterpiece, best quality, score_7, safe, 1boy, knight, full armor, holding sword, castle, sunset, dramatic lighting, wide shot", 7),
    ("best quality, safe, no humans, scenery, cozy cafe interior, rain on window, warm lighting, plants, bookshelf, detailed background", 1234),
]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
    ap.add_argument("--model", default="turbo-v1.1")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--out", default=None, help="directory for comparison PNGs")
    args = ap.parse_args()

    manifest_path = os.path.join(args.models, "manifest.json")
    manifest = json.load(open(manifest_path))
    model = next(m for m in manifest["models"] if m["id"] == args.model)
    steps, cfg = model["defaults"]["steps"], model["defaults"]["cfg"]
    dits = [q for q in ("bf16", "int8", "w4a8") if q in model["dit"]]
    tes = [q for q in ("bf16", "int8", "w4a8") if q in manifest["te"]]
    assert "bf16" in dits and "bf16" in tes, "build the bf16 DiT and TE first (they are the reference)"
    configs = [("bf16", "bf16")] + [(d, "bf16") for d in dits if d != "bf16"] + [("bf16", t) for t in tes if t != "bf16"]
    if "w4a8" in dits and "w4a8" in tes:
        configs.append(("w4a8", "w4a8"))
    if "int8" in dits and "int8" in tes:
        configs.append(("int8", "int8"))

    results = {c: {"psnr": [], "ctx": []} for c in configs}
    refs = {}
    for pi, (prompt, seed) in enumerate(PROMPTS):
        for dit, te in configs:
            _, Wte, Wd, Wv = R.open_weights(args.models, args.model, dit, te)
            img, ctx, _ = R.generate(args.models, Wte, Wd, Wv, prompt, "", args.size, args.size, steps, cfg, seed, log=lambda *_: None)
            if (dit, te) == ("bf16", "bf16"):
                refs[pi] = (img, ctx)
            ref_img, ref_ctx = refs[pi]
            results[(dit, te)]["psnr"].append(psnr(img, ref_img))
            results[(dit, te)]["ctx"].append(((ctx - ref_ctx).norm() / ref_ctx.norm()).item())
            if args.out:
                os.makedirs(args.out, exist_ok=True)
                R.save_png(img, os.path.join(args.out, f"p{pi}_dit-{dit}_te-{te}.png"))
            print(f"prompt {pi} dit {dit:5} te {te:5}  PSNR {results[(dit, te)]['psnr'][-1]:6.2f} dB  ctx err {results[(dit, te)]['ctx'][-1]:.2e}", flush=True)

    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    table = []
    for (dit, te), r in results.items():
        if (dit, te) == ("bf16", "bf16"):
            continue
        table.append({"dit": dit, "te": te, "psnr": round(mean(r["psnr"]), 2), "ctx_err": float(f"{mean(r['ctx']):.3g}")})
    for dit in dits:
        if dit != "bf16":
            model["dit"][dit]["psnr"] = round(mean(results[(dit, "bf16")]["psnr"]), 2)
    for te in tes:
        if te != "bf16":
            manifest["te"][te]["psnr"] = round(mean(results[("bf16", te)]["psnr"]), 2)
            manifest["te"][te]["ctx_err"] = float(f"{mean(results[('bf16', te)]['ctx']):.3g}")
    model["fidelity"] = {"reference": "bf16 DiT + bf16 TE", "size": args.size, "steps": steps, "prompts": len(PROMPTS), "table": table}
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
