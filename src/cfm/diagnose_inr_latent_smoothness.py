#!/usr/bin/env python3
"""Phase 1 diagnostic (z-regularization plan) — local smoothness of decode(z)
for the INR backbone vs MedVAE, along the interpolation paths that actually
matter for the flow model: same-field (control) vs cross-field, same
modality (the axis the flow's velocity field must traverse).

No training — uses frozen production checkpoints and already-cached latents
(no re-fitting). See /home/rousseau/.claude/plans/temporal-plotting-engelbart.md
and project memory (project_mmfm_inr_variant.md) for full context: this
checks whether decode(z) is measurably rougher for the INR than for MedVAE
BEFORE committing to implementing/testing a smoothness regularizer (which
requires a full ~6.3h backbone->precompute->flow retrain to validate).

Two metrics, both using the foreground-weighted output distance (plain
MSE/L1 is dominated by shared background — see inr_backbone.py::_weighted_mse
docstring for why this matters, a real bug this exact pitfall caused once
already in this codebase):

  1. Interpolation smoothness: linearly interpolate between two cached `z`
     in `n_steps`, decode each step, measure the std of consecutive decoded
     diffs (lower = smoother). Compared for same-field pairs (control) vs
     cross-field pairs (the flow's actual task). Adapted from
     src/vae3d/evaluate_latent_regularity.py::interpolation_smoothness, but
     starting from already-cached z (no re-encoding) and comparing the two
     pairing conditions.

  2. Local Lipschitz: perturb z by isotropic noise scaled relative to that
     latent's own typical inter-sample distance (INR and MedVAE differ in
     raw scale by ~50000x, so an absolute epsilon would be meaningless),
     decode original vs perturbed, report ||delta_output|| / ||delta_z||.

Usage:
    PYTHONPATH=src python src/cfm/diagnose_inr_latent_smoothness.py \\
        --n-pairs 20 --modality T1W \\
        --output results/mmfm/comparison_20260801_final/inr_latent_smoothness.csv

    # Re-check a smoke-test checkpoint during the z_reg_weight sweep:
    PYTHONPATH=src python src/cfm/diagnose_inr_latent_smoothness.py \\
        --inr-checkpoint outputs/mmfm/_smoke_inr_zreg/weights/model_final.pth \\
        --n-pairs 10 --output /tmp/smoke_zreg_1e-3.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_yaml_with_include, load_env, resolve_paths
from cfm.arch_inr import load_inr_backbone
from cfm.inr_backbone import make_coord_grid, sample_points
from cfm.mmfm_vectorized import LatentVectorizer
from models.vae_loader import load_vae

VOLUME_SIZE = (96, 112, 96)
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
FG_WEIGHT = 5.0
BG_THRESHOLD = -0.9
N_INR_POINTS = 32768  # sparse coordinate subset per pair (fixed across interpolation steps)


def _weighted_l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Foreground-upweighted mean |pred-target| — same weighting convention
    as inr_backbone.py::_weighted_mse, applied here to L1 to match
    evaluate_latent_regularity.py's interpolation_smoothness metric."""
    with torch.no_grad():
        fg_mask = (target > BG_THRESHOLD).float()
        weight = 1.0 + (FG_WEIGHT - 1.0) * fg_mask
        weight = weight / weight.mean()
        return float(((pred - target).abs() * weight).mean().item())


def _list_field_latents(cache_dir: Path, modality: str, field: str, n: int, seed: int) -> List[Path]:
    d = cache_dir / modality / field
    paths = sorted(d.glob("*.pt"))
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:n]


def _sample_pairs(per_field: dict, n_pairs: int, cross_field: bool, seed: int) -> List[Tuple[Path, Path]]:
    rng = random.Random(seed)
    fields = [f for f in FIELDS if per_field.get(f)]
    pairs = []
    for _ in range(n_pairs):
        if cross_field:
            f0, f1 = rng.sample(fields, 2)
        else:
            f0 = f1 = rng.choice(fields)
        p0 = rng.choice(per_field[f0])
        p1 = rng.choice(per_field[f1])
        if p0 == p1 and len(per_field[f0]) > 1:
            p1 = rng.choice([p for p in per_field[f0] if p != p0])
        pairs.append((p0, p1))
    return pairs


@torch.no_grad()
def _interp_smoothness_inr(backbone, coords_sub, z0, z1, n_steps=8) -> float:
    alphas = np.linspace(0.0, 1.0, n_steps)
    diffs, prev = [], None
    for a in alphas:
        z = z0 * (1.0 - a) + z1 * float(a)
        recon = backbone.decode(coords_sub.unsqueeze(0), z.unsqueeze(0))
        if prev is not None:
            diffs.append(_weighted_l1(recon, prev))
        prev = recon
    return float(np.std(diffs))


@torch.no_grad()
def _interp_smoothness_medvae(vae, vectorizer, z0_flat, z1_flat, n_steps=8) -> float:
    alphas = np.linspace(0.0, 1.0, n_steps)
    diffs, prev = [], None
    for a in alphas:
        z_flat = z0_flat * (1.0 - a) + z1_flat * float(a)
        z = vectorizer.unflatten(z_flat.unsqueeze(0))
        recon = vae.decode(z)
        if prev is not None:
            diffs.append(_weighted_l1(recon, prev))
        prev = recon
    return float(np.std(diffs))


@torch.no_grad()
def _local_lipschitz_inr(backbone, coords_sub, z0, ref_scale, n_samples=8) -> List[float]:
    ratios = []
    for _ in range(n_samples):
        noise = torch.randn_like(z0) * ref_scale * 0.05
        z_pert = z0 + noise
        r0 = backbone.decode(coords_sub.unsqueeze(0), z0.unsqueeze(0))
        r1 = backbone.decode(coords_sub.unsqueeze(0), z_pert.unsqueeze(0))
        dz = noise.norm().item()
        if dz > 1e-12:
            ratios.append(_weighted_l1(r1, r0) / dz)
    return ratios


@torch.no_grad()
def _local_lipschitz_medvae(vae, vectorizer, z0_flat, ref_scale, n_samples=8) -> List[float]:
    ratios = []
    for _ in range(n_samples):
        noise = torch.randn_like(z0_flat) * ref_scale * 0.05
        z_pert_flat = z0_flat + noise
        r0 = vae.decode(vectorizer.unflatten(z0_flat.unsqueeze(0)))
        r1 = vae.decode(vectorizer.unflatten(z_pert_flat.unsqueeze(0)))
        dz = noise.norm().item()
        if dz > 1e-12:
            ratios.append(_weighted_l1(r1, r0) / dz)
    return ratios


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inr-config", default="configs/mmfm/inr.yaml")
    ap.add_argument("--inr-checkpoint", default=None, help="Override inr_backbone.checkpoint in the config")
    ap.add_argument("--vectorized-config", default="configs/mmfm/vectorized.yaml")
    ap.add_argument("--modality", default="T1W")
    ap.add_argument("--n-pairs", type=int, default=20)
    ap.add_argument("--n-per-field", type=int, default=40, help="How many cached z per field to draw pairs from")
    ap.add_argument("--n-interp-steps", type=int, default=8)
    ap.add_argument("--n-lipschitz-samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="results/mmfm/comparison_20260801_final/inr_latent_smoothness.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Chargement du backbone INR...")
    inr_cfg = load_yaml_with_include(args.inr_config)
    inr_cfg = resolve_paths(inr_cfg, load_env("local"))
    if args.inr_checkpoint:
        inr_cfg["inr_backbone"]["checkpoint"] = args.inr_checkpoint
    backbone = load_inr_backbone(inr_cfg, device)
    coords_full = make_coord_grid(VOLUME_SIZE, device=device)
    coords_sub, _ = sample_points(coords_full, coords_full.unsqueeze(0), N_INR_POINTS)
    inr_cache_dir = Path(inr_cfg["data"]["latent_cache_dir"])

    print("Chargement de MedVAE (référence)...")
    vec_cfg = load_yaml_with_include(args.vectorized_config)
    vec_cfg = resolve_paths(vec_cfg, load_env("local"))
    vae = load_vae(vec_cfg, device)
    vae.eval()
    # vae.latent_shape is inferred from a fixed dummy input size unrelated to
    # this project's actual VOLUME_SIZE (96,112,96) and gives the wrong shape
    # (4096 vs the real 16128) — read the real shape from the production flow
    # checkpoint's arch_meta instead (written by mmfm_core.py's ArchAdapter,
    # see arch_vector.py::arch_meta_dict), the same value the flow model was
    # actually trained/cached with.
    _vec_ckpt = torch.load("outputs/mmfm/vectorized/weights/model_final.pth", map_location="cpu", weights_only=False)
    latent_shape = tuple(_vec_ckpt["arch_meta"]["latent_shape"])
    vectorizer = LatentVectorizer(latent_shape)
    medvae_cache_dir = Path(vec_cfg["data"]["latent_cache_dir"])

    # ---- gather cached latents per field ------------------------------------
    inr_per_field, medvae_per_field = {}, {}
    for f in FIELDS:
        inr_per_field[f] = _list_field_latents(inr_cache_dir, args.modality, f, args.n_per_field, args.seed)
        medvae_per_field[f] = _list_field_latents(medvae_cache_dir, args.modality, f, args.n_per_field, args.seed)
        print(f"  {f}: INR n={len(inr_per_field[f])}  MedVAE n={len(medvae_per_field[f])}")

    rows = []
    for cross in (False, True):
        cond = "cross_field" if cross else "same_field"
        inr_pairs = _sample_pairs(inr_per_field, args.n_pairs, cross, args.seed)
        medvae_pairs = _sample_pairs(medvae_per_field, args.n_pairs, cross, args.seed)

        print(f"\n[{cond}] Lissage d'interpolation ({args.n_interp_steps} pas)...")
        inr_scores, medvae_scores = [], []
        for p0, p1 in inr_pairs:
            z0 = torch.load(p0, map_location=device).float()
            z1 = torch.load(p1, map_location=device).float()
            inr_scores.append(_interp_smoothness_inr(backbone, coords_sub, z0, z1, args.n_interp_steps))
        for p0, p1 in medvae_pairs:
            z0 = torch.load(p0, map_location=device).float().flatten()
            z1 = torch.load(p1, map_location=device).float().flatten()
            medvae_scores.append(_interp_smoothness_medvae(vae, vectorizer, z0, z1, args.n_interp_steps))

        inr_mean, medvae_mean = float(np.mean(inr_scores)), float(np.mean(medvae_scores))
        print(f"  INR    mean std_step = {inr_mean:.6f}  (n={len(inr_scores)})")
        print(f"  MedVAE mean std_step = {medvae_mean:.6f}  (n={len(medvae_scores)})")
        rows.append({"metric": "interp_smoothness_std_step", "condition": cond,
                      "model": "inr", "value": inr_mean, "n": len(inr_scores)})
        rows.append({"metric": "interp_smoothness_std_step", "condition": cond,
                      "model": "medvae", "value": medvae_mean, "n": len(medvae_scores)})

    print("\n[isotropic] Lipschitz local (bruit ~5% de la norme de z)...")
    all_inr_paths = [p for f in FIELDS for p in inr_per_field[f]][: args.n_pairs]
    all_medvae_paths = [p for f in FIELDS for p in medvae_per_field[f]][: args.n_pairs]

    inr_lip = []
    for p in all_inr_paths:
        z0 = torch.load(p, map_location=device).float()
        inr_lip.extend(_local_lipschitz_inr(backbone, coords_sub, z0, z0.norm(), args.n_lipschitz_samples))
    medvae_lip = []
    for p in all_medvae_paths:
        z0 = torch.load(p, map_location=device).float().flatten()
        medvae_lip.extend(_local_lipschitz_medvae(vae, vectorizer, z0, z0.norm(), args.n_lipschitz_samples))

    inr_lip_med, inr_lip_std = float(np.median(inr_lip)), float(np.std(inr_lip))
    medvae_lip_med, medvae_lip_std = float(np.median(medvae_lip)), float(np.std(medvae_lip))
    print(f"  INR    Lipschitz median={inr_lip_med:.6f}  std={inr_lip_std:.6f}  (n={len(inr_lip)})")
    print(f"  MedVAE Lipschitz median={medvae_lip_med:.6f}  std={medvae_lip_std:.6f}  (n={len(medvae_lip)})")
    rows.append({"metric": "lipschitz_median", "condition": "isotropic", "model": "inr",
                 "value": inr_lip_med, "n": len(inr_lip)})
    rows.append({"metric": "lipschitz_std", "condition": "isotropic", "model": "inr",
                 "value": inr_lip_std, "n": len(inr_lip)})
    rows.append({"metric": "lipschitz_median", "condition": "isotropic", "model": "medvae",
                 "value": medvae_lip_med, "n": len(medvae_lip)})
    rows.append({"metric": "lipschitz_std", "condition": "isotropic", "model": "medvae",
                 "value": medvae_lip_std, "n": len(medvae_lip)})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "condition", "model", "value", "n"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[INFO] Sauvegardé : {out}")

    # ---- headline read: is INR's cross-field/same-field ratio worse than MedVAE's? ----
    def _ratio(model):
        cf = next(r["value"] for r in rows if r["metric"] == "interp_smoothness_std_step"
                  and r["condition"] == "cross_field" and r["model"] == model)
        sf = next(r["value"] for r in rows if r["metric"] == "interp_smoothness_std_step"
                  and r["condition"] == "same_field" and r["model"] == model)
        return cf / sf if sf > 0 else float("nan")

    inr_ratio, medvae_ratio = _ratio("inr"), _ratio("medvae")
    print("\n─── Lecture (porte de décision) ─────────────────────────────")
    print(f"  INR    cross_field/same_field ratio = {inr_ratio:.3f}")
    print(f"  MedVAE cross_field/same_field ratio = {medvae_ratio:.3f}")
    print(f"  INR isotropic Lipschitz std/median   = {inr_lip_std / inr_lip_med if inr_lip_med else float('nan'):.3f}")
    print(f"  MedVAE isotropic Lipschitz std/median = {medvae_lip_std / medvae_lip_med if medvae_lip_med else float('nan'):.3f}")
    print("────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
