#!/usr/bin/env python3
"""Latent regularity evaluation for VAE models.

Measures the smoothness of the latent space via:
  1. Linear interpolation smoothness:  encode a pair of volumes, interpolate
     their latents in 8 steps, decode each step, compute pixel-level
     deviation from monotonic change (lower = smoother).
  2. Local Lipschitz estimate:  randomly perturb input patches, measure
     ||Δz|| / ||Δx|| (should be bounded for a well-trained VAE).
  3. Modality gap:  average ||z_T1W - z_T2W|| for paired prospective subjects
     (same subject, same field, different modality) — a proxy for how much
     modality information is retained vs discarded.

Results are printed as a table and saved to a CSV.

Usage:
    python src/vae3d/evaluate_latent_regularity.py \\
        --vae-type aekl \\
        --checkpoint outputs/vae3d/runs/vae3d_T1W/weights/model_best.pth \\
        --config configs/vae3d_multimodal.yaml \\
        --env local \\
        --output results/benchmark_vae/analysis/latent_regularity_aekl.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
import torch.nn.functional as F

MODALITIES = ["T1W", "T2W", "T2FLAIR"]
FIELDS     = ["0.1T", "1.5T", "3T", "5T", "7T"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_norm(path: Path, patch_size=(128, 128, 128)) -> torch.Tensor:
    """Load a NIfTI, normalize to [-1, 1], return (1,1,H',W',D') central patch."""
    import nibabel as nib
    vol = nib.load(str(path)).get_fdata(dtype=np.float32)
    lo, hi = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
    if hi > lo:
        vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0) * 2.0 - 1.0
    else:
        vol = np.zeros_like(vol)
    ph, pw, pd = patch_size
    H, W, D = vol.shape
    sh, sw, sd = max(0,(H-ph)//2), max(0,(W-pw)//2), max(0,(D-pd)//2)
    patch = vol[sh:sh+ph, sw:sw+pw, sd:sd+pd]
    # pad if needed
    if patch.shape != (ph, pw, pd):
        pp = [(0, max(0, ph - patch.shape[0])),
              (0, max(0, pw - patch.shape[1])),
              (0, max(0, pd - patch.shape[2]))]
        patch = np.pad(patch, pp)
    x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
    return x


# ---------------------------------------------------------------------------
# Metric 1: interpolation smoothness
# ---------------------------------------------------------------------------

@torch.no_grad()
def interpolation_smoothness(vae, x0: torch.Tensor, x1: torch.Tensor,
                              n_steps: int = 8) -> float:
    """Average decoded L1 diff between consecutive interpolation steps.

    Smooth VAE: decoded images change monotonically, consecutive diffs similar.
    Returns the std of consecutive L1 differences (lower = smoother).
    """
    z0 = vae.encode(x0)
    z1 = vae.encode(x1)
    alphas = np.linspace(0.0, 1.0, n_steps)
    diffs = []
    prev_recon = None
    for a in alphas:
        z = z0 * (1.0 - a) + z1 * float(a)
        recon = vae.decode(z)
        if prev_recon is not None:
            diff = (recon - prev_recon).abs().mean().item()
            diffs.append(diff)
        prev_recon = recon
    return float(np.std(diffs))  # std of step sizes — lower = smoother


# ---------------------------------------------------------------------------
# Metric 2: local Lipschitz constant estimate
# ---------------------------------------------------------------------------

@torch.no_grad()
def local_lipschitz(vae, x: torch.Tensor, n_samples: int = 10,
                    eps: float = 0.05) -> float:
    """Estimate ||Δz||₂ / ||Δx||₂ for random perturbations of x."""
    z0 = vae.encode(x)
    z0_vec = vae.to_vector(z0)
    ratios = []
    for _ in range(n_samples):
        noise = torch.randn_like(x) * eps
        x_pert = (x + noise).clamp(-1.0, 1.0)
        z_pert = vae.encode(x_pert)
        z_pert_vec = vae.to_vector(z_pert)
        dz = (z_pert_vec - z0_vec).norm().item()
        dx = noise.norm().item()
        if dx > 1e-9:
            ratios.append(dz / dx)
    return float(np.median(ratios)) if ratios else float("nan")


# ---------------------------------------------------------------------------
# Metric 3: modality gap (paired prospective only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def modality_gap(vae, data_root: Path, patch_size=(128, 128, 128),
                 device="cpu") -> dict:
    """Compute average latent L2 distance between modality pairs per subject.

    Expects prospective structure:
      data_root/Training_prospective/<MOD>/<FIELD>/<subject>.nii.gz
    where subject IDs match across modalities.
    """
    from collections import defaultdict

    split_dir = data_root / "Training_prospective"
    # Map subject_id -> { mod -> { field -> path } }
    subjects: dict = defaultdict(lambda: defaultdict(dict))
    for mod in MODALITIES:
        for field in FIELDS:
            d = split_dir / mod / field
            if not d.exists():
                continue
            for f in sorted(d.glob("*.nii.gz")):
                # subject ID: strip modality and field prefix, e.g. P_T1W_3T_0006 -> 0006
                parts = f.stem.split("_")
                subj_id = parts[-1] if parts else f.stem
                subjects[subj_id][mod][field] = f

    gaps = {}
    mod_pairs = [("T1W", "T2W"), ("T1W", "T2FLAIR"), ("T2W", "T2FLAIR")]
    for (mA, mB) in mod_pairs:
        dists = []
        for subj, mod_dict in subjects.items():
            if mA not in mod_dict or mB not in mod_dict:
                continue
            common_fields = set(mod_dict[mA]) & set(mod_dict[mB])
            for field in common_fields:
                try:
                    xA = load_norm(mod_dict[mA][field], patch_size).to(device)
                    xB = load_norm(mod_dict[mB][field], patch_size).to(device)
                    zA = vae.to_vector(vae.encode(xA)).squeeze(0)
                    zB = vae.to_vector(vae.encode(xB)).squeeze(0)
                    dists.append((zA - zB).norm().item())
                except Exception as e:
                    print(f"  [WARN] {subj} {mA}/{mB} {field}: {e}")
        if dists:
            gaps[f"{mA}_vs_{mB}"] = float(np.mean(dists))
    return gaps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Latent regularity evaluation")
    parser.add_argument("--vae-type", default="aekl")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--env", default="local")
    parser.add_argument("--output", default="results/benchmark_vae/analysis/latent_regularity.csv")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128],
                        metavar=("H", "W", "D"))
    parser.add_argument("--n-interp-pairs", type=int, default=5,
                        help="Number of volume pairs for interpolation test")
    parser.add_argument("--n-lipschitz", type=int, default=10,
                        help="Number of perturbation samples per volume")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── Load VAE ──────────────────────────────────────────────────────────────
    from models.vae_loader import load_vae
    from common.config import load_env, resolve_paths

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        env_cfg = load_env(args.env)
        cfg = resolve_paths(cfg, env_cfg)
    else:
        cfg = {"vae": {"vae_type": args.vae_type, "frozen": True}}
    if args.checkpoint:
        cfg.setdefault("vae", {})["checkpoint"] = args.checkpoint

    vae = load_vae(cfg, device=device)
    vae.eval()
    print(f"[INFO] VAE: {args.vae_type}  vector_dim={vae.vector_dim}")

    # ── Data root ─────────────────────────────────────────────────────────────
    if args.data_root:
        data_root = Path(args.data_root)
    elif "data" in cfg and "data_root" in cfg["data"]:
        data_root = Path(cfg["data"]["data_root"])
    else:
        data_root = Path.home() / "Data" / "MRIxFields_20260414"

    patch_size = tuple(args.patch_size)

    # ── Collect a handful of volumes ─────────────────────────────────────────
    pros_dir = data_root / "Training_prospective"
    all_files = []
    for mod in MODALITIES:
        for field in FIELDS:
            d = pros_dir / mod / field
            if d.exists():
                all_files.extend(sorted(d.glob("*.nii.gz"))[:1])
    if not all_files:
        # fallback to retrospective
        retro_dir = data_root / "Training_retrospective"
        for mod in MODALITIES:
            for field in FIELDS:
                d = retro_dir / mod / field
                if d.exists():
                    all_files.extend(sorted(d.glob("*.nii.gz"))[:1])
    if not all_files:
        print("[ERROR] No volumes found.")
        sys.exit(1)

    # ── Metric 1: interpolation smoothness ───────────────────────────────────
    print("\n[1/3] Interpolation smoothness …")
    n_pairs = min(args.n_interp_pairs, len(all_files) // 2)
    smooth_scores = []
    for i in range(n_pairs):
        try:
            x0 = load_norm(all_files[i * 2], patch_size).to(device)
            x1 = load_norm(all_files[i * 2 + 1], patch_size).to(device)
            s = interpolation_smoothness(vae, x0, x1)
            smooth_scores.append(s)
            print(f"  pair {i}: std_step={s:.5f}")
        except Exception as e:
            print(f"  pair {i}: [WARN] {e}")
    mean_smooth = float(np.mean(smooth_scores)) if smooth_scores else float("nan")
    print(f"  mean std_step = {mean_smooth:.5f}  (lower = smoother)")

    # ── Metric 2: local Lipschitz ─────────────────────────────────────────────
    print("\n[2/3] Local Lipschitz estimate …")
    lip_vals = []
    for path in all_files[:5]:
        try:
            x = load_norm(path, patch_size).to(device)
            lip = local_lipschitz(vae, x, n_samples=args.n_lipschitz)
            lip_vals.append(lip)
            print(f"  {path.name}: Lip={lip:.4f}")
        except Exception as e:
            print(f"  {path.name}: [WARN] {e}")
    mean_lip = float(np.mean(lip_vals)) if lip_vals else float("nan")
    print(f"  mean Lipschitz = {mean_lip:.4f}")

    # ── Metric 3: modality gap ────────────────────────────────────────────────
    print("\n[3/3] Modality gap (prospective subjects) …")
    gaps = modality_gap(vae, data_root, patch_size=patch_size, device=device)
    for k, v in gaps.items():
        print(f"  {k}: mean ||Δz|| = {v:.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {"metric": "interp_smoothness_mean_std_step", "value": mean_smooth,
         "vae_type": args.vae_type, "checkpoint": args.checkpoint or ""},
        {"metric": "lipschitz_median", "value": mean_lip,
         "vae_type": args.vae_type, "checkpoint": args.checkpoint or ""},
    ]
    for k, v in gaps.items():
        rows.append({"metric": f"modality_gap_{k}", "value": v,
                     "vae_type": args.vae_type, "checkpoint": args.checkpoint or ""})

    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["vae_type", "checkpoint", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[INFO] Saved: {out}")

    # Summary table
    print("\n─── Summary ───────────────────────────────")
    for r in rows:
        print(f"  {r['metric']:<40s}  {r['value']:.5f}")
    print("────────────────────────────────────────────")


if __name__ == "__main__":
    main()
