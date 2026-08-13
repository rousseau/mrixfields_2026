#!/usr/bin/env python3
"""Capacité de représentation MedVAE par PATCHES, à plusieurs résolutions.

Répond à : « les poids MedVAE peuvent-ils fonctionner à la résolution des
données du challenge ? »

Contexte (mesuré, pas supposé) :
  - À 2mm (96x112x96), MedVAE pré-entraîné et notre fine-tuné L1 reconstruisent
    tous deux avec un flou visible (SSIM ~0.89-0.90) : la recette de loss n'est
    donc PAS la cause dominante (le pré-entraîné a pourtant été entraîné avec
    LPIPS+PatchGAN et fait même un peu moins bien ici).
  - Le plein-volume à 1mm est IMPOSSIBLE : la mémoire croît en ~x3 quand les
    voxels doublent (4.7GB @96x112x96, 13.8GB @128^3, 46.1GB @160^3, OOM à
    192x224x192) — signature du coût quadratique de l'attention au goulot.
    D'où ce script, entièrement par patches.

Le patch 64^3 est celui de la recette officielle MedVAE 3D
(medvae/utils/loaders.py::load_mri_3d_finetune → RandSpatialCrop 64^3), donc le
régime pour lequel ces poids ont réellement été entraînés.

Usage:
    PYTHONPATH=src python src/vae3d/eval_representation_multires.py
    PYTHONPATH=src python src/vae3d/eval_representation_multires.py --resolutions 2 1
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_env
from common.io import center_crop_or_pad_np, load_nifti_volume
from common.metrics import compute_nrmse, compute_ssim

MODALITY = "T1W"
SUBJECTS = ["0006", "0007", "0009"]
FIELDS = ["0.1T", "3T", "7T"]

# spacing_mm -> volume_size (FOV ~192mm, divisible par 4)
RESOLUTION_TABLE = {
    2.0: (96, 112, 96),
    1.0: (192, 224, 192),
    0.5: (384, 448, 384),
}

FINETUNED_CKPT = "outputs/medvae/runs/medvae_finetune_all/weights/model_best.pth"
OUT_CSV = Path("results/benchmark_vae/metrics/medvae_representation_multires.csv")


# ---------------------------------------------------------------------------
# Reconstruction par patches (fenêtre glissante + blending Hann)
# ---------------------------------------------------------------------------


def _positions(size: int, patch: int, stride: int) -> List[int]:
    """Positions de départ couvrant [0, size), bord final inclus."""
    if size <= patch:
        return [0]
    pos = list(range(0, size - patch + 1, stride))
    if pos[-1] != size - patch:
        pos.append(size - patch)
    return pos


def _hann3d(patch: Tuple[int, int, int], device) -> torch.Tensor:
    ws = [torch.hann_window(p, periodic=False, device=device).clamp_min(1e-3) for p in patch]
    return ws[0][:, None, None] * ws[1][None, :, None] * ws[2][None, None, :]


@torch.no_grad()
def patched_reconstruct(
    inner, x: torch.Tensor, patch: Tuple[int, int, int], stride: Tuple[int, int, int],
    batch_size: int = 8, use_amp: bool = True,
) -> torch.Tensor:
    """Encode+decode par patches avec recouvrement, fusionnés par fenêtre de Hann.

    `inner` : l'AutoencoderKL sous-jacent (encode() -> distribution).
    x : (1, 1, H, W, D) dans [-1, 1].
    """
    device = x.device
    _, _, H, W, D = x.shape
    ph, pw, pd = patch
    acc = torch.zeros(1, 1, H, W, D, device=device, dtype=torch.float32)
    wacc = torch.zeros(1, 1, H, W, D, device=device, dtype=torch.float32)
    blend = _hann3d(patch, device)

    coords = [(i, j, k)
              for i in _positions(H, ph, stride[0])
              for j in _positions(W, pw, stride[1])
              for k in _positions(D, pd, stride[2])]

    for b0 in range(0, len(coords), batch_size):
        chunk = coords[b0 : b0 + batch_size]
        batch = torch.cat([x[:, :, i:i+ph, j:j+pw, k:k+pd] for (i, j, k) in chunk], dim=0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(use_amp and device.type == "cuda")):
            post = inner.encode(batch)
            z = post.mode() if hasattr(post, "mode") else post.sample()
            rec = inner.decode(z)
        rec = rec.float()
        for n, (i, j, k) in enumerate(chunk):
            acc[:, :, i:i+ph, j:j+pw, k:k+pd] += rec[n : n + 1] * blend
            wacc[:, :, i:i+ph, j:j+pw, k:k+pd] += blend

    return acc / wacc.clamp_min(1e-6)


# ---------------------------------------------------------------------------


def _build_model(device, finetuned: bool):
    from medvae import MVAE

    model = MVAE(model_name="medvae_4_1_3d", modality="mri").to(device)
    if finetuned:
        ckpt = torch.load(FINETUNED_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return getattr(model, "model", model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolutions", type=float, nargs="+", default=[2.0, 1.0, 0.5])
    ap.add_argument("--patch", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output", default=str(OUT_CSV))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(load_env("local")["data_root"])
    patch = tuple(args.patch)
    stride = tuple(max(1, int(p * (1.0 - args.overlap))) for p in patch)
    out_csv = Path(args.output)

    print(f"Patch {patch}  stride {stride}  (overlap {args.overlap})")
    variants = {"pretrained": _build_model(device, False)}
    if Path(FINETUNED_CKPT).exists():
        variants["finetuned_l1"] = _build_model(device, True)

    rows = []
    for spacing in args.resolutions:
        volume_size = RESOLUTION_TABLE[spacing]
        tag = f"{spacing:g}mm_{'x'.join(map(str, volume_size))}"
        print(f"\n{'=' * 78}\nRésolution {tag}\n{'=' * 78}", flush=True)

        for sid in SUBJECTS:
            for field in FIELDS:
                p = data_root / "Training_prospective" / MODALITY / field / f"P_{MODALITY}_{field}_{sid}.nii.gz"
                if not p.exists():
                    continue
                vol, _ = load_nifti_volume(
                    p, target_spacing=(spacing,) * 3, volume_size=None,
                    normalize=True, lo_pct=0.5, hi_pct=99.5,
                )
                vol = center_crop_or_pad_np(vol, volume_size)
                x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
                gt = np.clip((vol + 1) / 2, 0, 1)
                fg = gt > 0.025

                line = f"  {sid} {field:5s} |"
                for name, inner in variants.items():
                    t0 = time.time()
                    rec = patched_reconstruct(inner, x, patch, stride, args.batch_size)
                    rec_np = np.clip((rec[0, 0].cpu().numpy() + 1) / 2, 0, 1)
                    nrmse = float(compute_nrmse(rec_np, gt, mask=fg))
                    ssim = float(compute_ssim(rec_np, gt))
                    rows.append({
                        "resolution": tag, "spacing_mm": spacing, "variant": name,
                        "subject": sid, "field": field,
                        "nrmse": round(nrmse, 5), "ssim": round(ssim, 5),
                        "seconds": round(time.time() - t0, 1),
                    })
                    line += f"  {name}: nRMSE={nrmse:.4f} SSIM={ssim:.4f} ({time.time()-t0:.0f}s) |"
                    del rec
                    torch.cuda.empty_cache()
                print(line, flush=True)

                out_csv.parent.mkdir(parents=True, exist_ok=True)
                with open(out_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

    print(f"\n{'=' * 78}\nRÉSUMÉ\n{'=' * 78}")
    for tag in sorted({r["resolution"] for r in rows}, key=lambda t: -float(t.split("mm")[0])):
        print(f"\n--- {tag} ---")
        for name in variants:
            sub = [r for r in rows if r["resolution"] == tag and r["variant"] == name]
            if sub:
                print(f"  {name:14s} : nRMSE={np.mean([r['nrmse'] for r in sub]):.4f}  "
                      f"SSIM={np.mean([r['ssim'] for r in sub]):.4f}")
        for field in FIELDS:
            parts = []
            for name in variants:
                sub = [r for r in rows if r["resolution"] == tag and r["variant"] == name and r["field"] == field]
                if sub:
                    parts.append(f"{name} SSIM={np.mean([r['ssim'] for r in sub]):.4f}")
            if parts:
                print(f"    {field:5s} : " + "  |  ".join(parts))
    print(f"\nCSV : {out_csv}")


if __name__ == "__main__":
    main()
