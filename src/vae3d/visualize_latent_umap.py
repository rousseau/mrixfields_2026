#!/usr/bin/env python3
"""UMAP visualization of VAE latent spaces.

Encodes a set of prospective (and optionally validation) NIfTI volumes,
collects z vectors (spatial VAEs: flattened via to_vector()), fits UMAP
in 2D, and saves a scatter plot color-coded by modality and field.

Usage:
    python src/vae3d/visualize_latent_umap.py \\
        --vae-type aekl \\
        --checkpoint outputs/vae3d/runs/vae3d_T1W/weights/model_best.pth \\
        --config configs/vae3d_multimodal.yaml \\
        --env local \\
        --output results/qc/umap_aekl_multimodal.png \\
        [--n-per-combo 1]   # volumes per (modality, field) combo

    python src/vae3d/visualize_latent_umap.py \\
        --vae-type medvae \\
        --output results/qc/umap_medvae_frozen.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── sys.path so that `import models.vae_loader` etc. works without install ──
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch

MODALITIES = ["T1W", "T2W", "T2FLAIR"]
FIELDS     = ["0.1T", "1.5T", "3T", "5T", "7T"]

# Distinct colours / markers for UMAP scatter
MOD_COLORS  = {"T1W": "#e41a1c", "T2W": "#377eb8", "T2FLAIR": "#4daf4a"}
FIELD_MARKERS = {"0.1T": "o", "1.5T": "s", "3T": "D", "5T": "^", "7T": "P"}


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_volumes(data_root: Path, split: str, n_per_combo: int = 1):
    """Yield (path, modality, field) for each available combination.

    Searches both `Training_prospective/<MOD>/<FIELD>/*.nii.gz`
    and `Training_retrospective/<MOD>/<FIELD>/*.nii.gz`.
    """
    entries = []
    split_dir = data_root / split
    if not split_dir.exists():
        return entries
    for mod in MODALITIES:
        for field in FIELDS:
            field_dir = split_dir / mod / field
            if not field_dir.exists():
                continue
            files = sorted(field_dir.glob("*.nii.gz"))[:n_per_combo]
            for f in files:
                entries.append((f, mod, field))
    return entries


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_volume(vae, path: Path, patch_size=(128, 128, 128), device="cpu"):
    """Return a 1-D numpy latent vector for a single volume."""
    import nibabel as nib

    img = nib.load(str(path))
    vol = img.get_fdata(dtype=np.float32)
    lo = np.percentile(vol, 0.5)
    hi = np.percentile(vol, 99.5)
    if hi > lo:
        vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0) * 2.0 - 1.0
    else:
        vol = np.zeros_like(vol)

    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W,D)

    # Crop / pad to patch_size for a single central patch (cheap, no sliding window)
    ph, pw, pd = patch_size
    H, W, D = vol.shape
    sh = max(0, (H - ph) // 2)
    sw = max(0, (W - pw) // 2)
    sd = max(0, (D - pd) // 2)
    patch = x[:, :, sh:sh+ph, sw:sw+pw, sd:sd+pd]

    # Zero-pad if smaller than patch_size
    if patch.shape[2:] != (ph, pw, pd):
        pad_h = ph - patch.shape[2]
        pad_w = pw - patch.shape[3]
        pad_d = pd - patch.shape[4]
        import torch.nn.functional as F
        patch = F.pad(patch, (0, pad_d, 0, pad_w, 0, pad_h))

    z = vae.encode(patch)                    # spatial or vector
    z_vec = vae.to_vector(z)                 # (1, D_flat)
    return z_vec.squeeze(0).cpu().float().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UMAP of VAE latent space")
    parser.add_argument("--vae-type", default="aekl",
                        help="VAE type: aekl | medvae | medvae_finetune | pythae_vae | …")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to VAE checkpoint (.pth)")
    parser.add_argument("--config", default=None,
                        help="Path to YAML config used during training")
    parser.add_argument("--env", default="local",
                        help="Environment: local | remote")
    parser.add_argument("--output", default="results/qc/umap_latent.png",
                        help="Output PNG path")
    parser.add_argument("--data-root", default=None,
                        help="Override data root directory")
    parser.add_argument("--split", default="Training_prospective",
                        choices=["Training_prospective", "Training_retrospective"],
                        help="Data split to encode")
    parser.add_argument("--n-per-combo", type=int, default=2,
                        help="Number of volumes per (modality, field) combo")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128],
                        metavar=("H", "W", "D"))
    parser.add_argument("--device", default=None,
                        help="cuda | cpu (default: auto-detect)")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
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
        # Minimal config for frozen medvae
        cfg = {"vae": {"vae_type": args.vae_type, "frozen": True}}

    if args.checkpoint:
        cfg.setdefault("vae", {})["checkpoint"] = args.checkpoint

    vae = load_vae(cfg, device=device)
    vae.eval()
    print(f"[INFO] VAE loaded: {args.vae_type}  vector_dim={vae.vector_dim}")

    # ── Data root ─────────────────────────────────────────────────────────────
    if args.data_root:
        data_root = Path(args.data_root)
    elif "data" in cfg and "data_root" in cfg["data"]:
        data_root = Path(cfg["data"]["data_root"])
    else:
        data_root = Path.home() / "Data" / "MRIxFields_20260414"
    print(f"[INFO] Data root: {data_root}")

    # ── Collect volumes ───────────────────────────────────────────────────────
    entries = collect_volumes(data_root, args.split, args.n_per_combo)
    if not entries:
        print(f"[ERROR] No volumes found in {data_root / args.split}")
        sys.exit(1)
    print(f"[INFO] Found {len(entries)} volumes to encode")

    # ── Encode ───────────────────────────────────────────────────────────────
    patch_size = tuple(args.patch_size)
    vectors, mods, fields = [], [], []
    for i, (path, mod, field) in enumerate(entries):
        try:
            z = encode_volume(vae, path, patch_size=patch_size, device=device)
            vectors.append(z)
            mods.append(mod)
            fields.append(field)
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(entries)}] encoded")
        except Exception as e:
            print(f"  [WARN] Skipping {path.name}: {e}")

    if len(vectors) < 3:
        print("[ERROR] Too few vectors for UMAP (need >= 3)")
        sys.exit(1)

    X = np.stack(vectors, axis=0)  # (N, D)
    print(f"[INFO] Latent matrix: {X.shape}")

    # ── UMAP ─────────────────────────────────────────────────────────────────
    try:
        import umap
    except ImportError:
        print("[ERROR] umap-learn not installed. Run: pip install umap-learn")
        sys.exit(1)

    print(f"[INFO] Fitting UMAP (n_neighbors={args.umap_n_neighbors}, "
          f"min_dist={args.umap_min_dist}) …")
    reducer = umap.UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        random_state=42,
        low_memory=True,
    )
    embedding = reducer.fit_transform(X)  # (N, 2)
    print("[INFO] UMAP done.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: colour = modality, marker = field
    ax = axes[0]
    ax.set_title("Colour = modality  |  Marker = field")
    for (mod, field), (color, marker) in [
        ((m, fi), (MOD_COLORS.get(m, "gray"), FIELD_MARKERS.get(fi, "o")))
        for m, fi in zip(mods, fields)
    ]:
        pass  # iterate below

    for i, (mod, field) in enumerate(zip(mods, fields)):
        color  = MOD_COLORS.get(mod, "gray")
        marker = FIELD_MARKERS.get(field, "o")
        ax.scatter(embedding[i, 0], embedding[i, 1],
                   c=color, marker=marker, s=60, alpha=0.8, linewidths=0)

    # Legend modality
    mod_handles = [
        mlines.Line2D([], [], color=c, marker="o", linestyle="None",
                      markersize=8, label=m)
        for m, c in MOD_COLORS.items()
    ]
    field_handles = [
        mlines.Line2D([], [], color="gray", marker=mk, linestyle="None",
                      markersize=8, label=fi)
        for fi, mk in FIELD_MARKERS.items()
    ]
    ax.legend(handles=mod_handles + field_handles, fontsize=8, ncol=2)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Right: colour = field
    FIELD_COLORS = {
        "0.1T": "#e41a1c", "1.5T": "#ff7f00", "3T": "#a6d854",
        "5T": "#377eb8",   "7T": "#984ea3",
    }
    ax2 = axes[1]
    ax2.set_title("Colour = field  |  Marker = modality")
    MOD_MARKERS = {"T1W": "o", "T2W": "s", "T2FLAIR": "D"}
    for i, (mod, field) in enumerate(zip(mods, fields)):
        color  = FIELD_COLORS.get(field, "gray")
        marker = MOD_MARKERS.get(mod, "o")
        ax2.scatter(embedding[i, 0], embedding[i, 1],
                    c=color, marker=marker, s=60, alpha=0.8, linewidths=0)

    field_handles2 = [
        mlines.Line2D([], [], color=c, marker="o", linestyle="None",
                      markersize=8, label=fi)
        for fi, c in FIELD_COLORS.items()
    ]
    mod_handles2 = [
        mlines.Line2D([], [], color="gray", marker=mk, linestyle="None",
                      markersize=8, label=m)
        for m, mk in MOD_MARKERS.items()
    ]
    ax2.legend(handles=field_handles2 + mod_handles2, fontsize=8, ncol=2)
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")

    vae_label = args.vae_type
    if args.checkpoint:
        vae_label += f"  [{Path(args.checkpoint).name}]"
    fig.suptitle(f"Latent UMAP — {vae_label}\n{args.split}  n={len(vectors)}", fontsize=11)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[INFO] Saved: {out}")

    # Also save raw embedding as CSV alongside the figure
    csv_path = out.with_suffix(".csv")
    import csv
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "modality", "field", "umap1", "umap2"])
        for (path, mod, field), (u1, u2) in zip(entries[:len(vectors)],
                                                  embedding):
            writer.writerow([str(path), mod, field, f"{u1:.6f}", f"{u2:.6f}"])
    print(f"[INFO] Embedding CSV: {csv_path}")


if __name__ == "__main__":
    main()
