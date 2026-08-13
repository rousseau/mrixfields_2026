#!/usr/bin/env python3
"""Capacité de représentation du backbone INR (SIREN) à plusieurs résolutions.

Pendant INR de `src/vae3d/eval_representation_multires.py` (MedVAE).

Intérêt spécifique de l'INR ici : il n'a PAS la contrainte qui bloque MedVAE.
MedVAE possède une attention « vanilla » au goulot, donc quadratique en nombre
de voxels latents — mesuré : 4.7GB à 96x112x96, 13.8GB à 128^3, 46.1GB à 160^3,
puis OOM (>121GB) à 192x224x192 ; il faut donc découper en patches, ce qui
dégrade la reconstruction (perte du contexte global). L'INR, lui, mappe des
coordonnées normalisées [-1,1]^3 vers une intensité : changer de résolution ne
fait que densifier l'échantillonnage du MÊME cube, sans aucun terme quadratique
ni découpage. C'est exactement la propriété d'invariance à la résolution
revendiquée par NOIR/MedFuncta — ce script la met à l'épreuve sur nos données.

Attention à l'interprétation : le backbone a été méta-entraîné à 96x112x96@2mm.
Évaluer à 1mm/0.5mm teste sa capacité à RESTITUER des détails plus fins que
ceux vus à l'entraînement — il ne peut pas inventer de l'information absente de
`z` (4096 dims), donc une dégradation en montant en résolution est attendue et
informative : elle borne ce que la représentation encode réellement.

Usage:
    PYTHONPATH=src python src/cfm/eval_inr_multires.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_yaml_with_include, load_env, resolve_paths
from common.io import center_crop_or_pad_np, load_nifti_volume
from common.metrics import compute_nrmse, compute_ssim
from cfm.arch_inr import load_inr_backbone
from cfm.inr_backbone import fit_new_volume, make_coord_grid

MODALITY = "T1W"
SUBJECTS = ["0006", "0007", "0009"]
FIELDS = ["0.1T", "3T", "7T"]

RESOLUTION_TABLE = {
    2.0: (96, 112, 96),
    1.0: (192, 224, 192),
    0.5: (384, 448, 384),
}

OUT_CSV = Path("results/mmfm/comparison_20260801_final/inr_representation_multires.csv")


@torch.no_grad()
def _decode_chunked(backbone, coords, z, chunk: int = 1_000_000) -> torch.Tensor:
    """Décode φ(x; θ, M_ψ(z)) par blocs de coordonnées (une grille 0.5mm fait
    ~66M points : impossible en une passe)."""
    outs = []
    for i in range(0, coords.shape[0], chunk):
        sub = coords[i : i + chunk].unsqueeze(0)
        outs.append(backbone.decode(sub, z).squeeze(0).squeeze(-1))
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolutions", type=float, nargs="+", default=[2.0, 1.0, 0.5])
    ap.add_argument("--fit-points", type=int, default=131072,
                    help="Points échantillonnés pour le fitting de z (Algorithme 2)")
    ap.add_argument("--output", default=str(OUT_CSV))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = resolve_paths(load_yaml_with_include("configs/mmfm/inr.yaml"), load_env("local"))
    print("Chargement du backbone INR (production)...")
    backbone = load_inr_backbone(cfg, device)
    data_root = Path(cfg["data"]["data_root"])
    out_csv = Path(args.output)

    rows = []
    for spacing in args.resolutions:
        volume_size = RESOLUTION_TABLE[spacing]
        tag = f"{spacing:g}mm_{'x'.join(map(str, volume_size))}"
        n_vox = int(np.prod(volume_size))
        print(f"\n{'=' * 78}\nRésolution {tag}  ({n_vox / 1e6:.1f}M voxels)\n{'=' * 78}", flush=True)
        coords = make_coord_grid(volume_size, device=device)

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

                t0 = time.time()
                z = fit_new_volume(
                    backbone, x, coords,
                    num_steps=backbone.cfg.inner_steps_eval,
                    lr=backbone.cfg.inner_lr,
                    num_points=min(args.fit_points, n_vox),
                )
                rec = _decode_chunked(backbone, coords, z).reshape(volume_size)
                dt = time.time() - t0

                gt = np.clip((vol + 1) / 2, 0, 1)
                rec_np = np.clip((rec.float().cpu().numpy() + 1) / 2, 0, 1)
                fg = gt > 0.025
                nrmse = float(compute_nrmse(rec_np, gt, mask=fg))
                ssim = float(compute_ssim(rec_np, gt))
                rows.append({
                    "resolution": tag, "spacing_mm": spacing, "variant": "inr",
                    "subject": sid, "field": field,
                    "nrmse": round(nrmse, 5), "ssim": round(ssim, 5), "seconds": round(dt, 1),
                })
                print(f"  {sid} {field:5s} | INR nRMSE={nrmse:.4f} SSIM={ssim:.4f} ({dt:.0f}s)", flush=True)

                del rec, z, x
                torch.cuda.empty_cache()

                out_csv.parent.mkdir(parents=True, exist_ok=True)
                with open(out_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

        del coords
        torch.cuda.empty_cache()

    print(f"\n{'=' * 78}\nRÉSUMÉ INR\n{'=' * 78}")
    for tag in sorted({r["resolution"] for r in rows}, key=lambda t: -float(t.split("mm")[0])):
        sub = [r for r in rows if r["resolution"] == tag]
        print(f"  {tag:22s} : nRMSE={np.mean([r['nrmse'] for r in sub]):.4f}  "
              f"SSIM={np.mean([r['ssim'] for r in sub]):.4f}")
        for field in FIELDS:
            fs = [r for r in sub if r["field"] == field]
            if fs:
                print(f"      {field:5s} : SSIM={np.mean([r['ssim'] for r in fs]):.4f}")
    print(f"\nCSV : {out_csv}")


if __name__ == "__main__":
    main()
