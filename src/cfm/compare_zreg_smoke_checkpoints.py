#!/usr/bin/env python3
"""Phase 2 sweep verification: for each z_reg_weight smoke checkpoint, fit
z live (Algorithm 2) on a handful of same-field/cross-field volume pairs and
compare interpolation smoothness — confirms the penalty is actually moving
the measured roughness, not just passing the pass/fail smoke checks. See
/home/rousseau/.claude/plans/temporal-plotting-engelbart.md.

Usage:
    PYTHONPATH=src python src/cfm/compare_zreg_smoke_checkpoints.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_env
from common.io import SPLIT_MAP
from cfm.diagnose_inr_latent_smoothness import _weighted_l1, VOLUME_SIZE
from cfm.evaluate_inr_reconstruction import _load_volume
from cfm.inr_backbone import INRBackbone, INRBackboneConfig, fit_new_volume, make_coord_grid

MODALITY = "T1W"
FIELDS_FOR_TEST = ["0.1T", "3T", "7T"]  # spread across the field axis
N_PER_FIELD = 3
N_INTERP_STEPS = 8

CHECKPOINTS = {
    "w=0 (control)": "outputs/mmfm/_smoke_inr_zreg/w0/weights/model_final.pth",
    "w=1e-8": "outputs/mmfm/_smoke_inr_zreg/w1e-8/weights/model_final.pth",
    "w=1e-7": "outputs/mmfm/_smoke_inr_zreg/w1e-7/weights/model_final.pth",
}

BACKBONE_KWARGS = dict(
    latent_dim=4096, hidden_dim=256, num_hidden_layers=6,
    hyper_hidden_dim=512, inner_steps_train=10, inner_steps_eval=20,
)


@torch.no_grad()
def _interp_smoothness(backbone, coords_full, z0, z1, n_steps=N_INTERP_STEPS) -> float:
    alphas = np.linspace(0.0, 1.0, n_steps)
    diffs, prev = [], None
    for a in alphas:
        z = z0 * (1.0 - a) + z1 * float(a)
        recon = backbone.decode(coords_full.unsqueeze(0), z)
        if prev is not None:
            diffs.append(_weighted_l1(recon, prev))
        prev = recon
    return float(np.std(diffs))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = load_env("local")
    data_root = Path(env["data_root"])
    coords_full = make_coord_grid(VOLUME_SIZE, device=device)

    print("Chargement des volumes de test...")
    per_field = {}
    for f in FIELDS_FOR_TEST:
        src_dir = data_root / SPLIT_MAP["retro_train"] / MODALITY / f
        files = sorted(src_dir.glob("*.nii.gz"))[:N_PER_FIELD]
        vols = []
        for p in files:
            vol = _load_volume(p)
            vols.append(torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device))
        per_field[f] = vols
        print(f"  {f}: {len(vols)} volumes")

    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n=== {name} ({ckpt_path}) ===")
        cfg = INRBackboneConfig(**BACKBONE_KWARGS)
        backbone = INRBackbone(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        backbone.load_state_dict(state["model"])
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        t0 = time.time()
        z_per_field = {}
        for f in FIELDS_FOR_TEST:
            zs = []
            for vol in per_field[f]:
                z = fit_new_volume(backbone, vol, coords_full, num_steps=cfg.inner_steps_eval, lr=cfg.inner_lr)
                zs.append(z)
            z_per_field[f] = zs
        print(f"  Fitting z : {time.time() - t0:.1f}s")

        same_field_scores = []
        for f in FIELDS_FOR_TEST:
            zs = z_per_field[f]
            for i in range(len(zs)):
                for j in range(i + 1, len(zs)):
                    same_field_scores.append(_interp_smoothness(backbone, coords_full, zs[i], zs[j]))

        cross_field_scores = []
        for i, f0 in enumerate(FIELDS_FOR_TEST):
            for f1 in FIELDS_FOR_TEST[i + 1:]:
                for z0 in z_per_field[f0]:
                    for z1 in z_per_field[f1]:
                        cross_field_scores.append(_interp_smoothness(backbone, coords_full, z0, z1))

        sf_mean = float(np.mean(same_field_scores))
        cf_mean = float(np.mean(cross_field_scores))
        ratio = cf_mean / sf_mean if sf_mean > 0 else float("nan")
        print(f"  same_field  mean std_step = {sf_mean:.6f}  (n={len(same_field_scores)})")
        print(f"  cross_field mean std_step = {cf_mean:.6f}  (n={len(cross_field_scores)})")
        print(f"  ratio cross/same = {ratio:.3f}")

        del backbone
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
