#!/usr/bin/env python3
"""Extract VAE latent representations and save as NIfTI.

For each input NIfTI volume:
  - Encodes via the chosen VAE (patch-based via MRIxFieldsVAE.extract_latent_nifti)
  - Spatial VAEs  → 4D NIfTI  (H', W', D', C)
  - Vector VAEs   → 1D NIfTI  (D_lat,)

Supports batch processing of entire modality × field subsets.

Usage — single volume:
    python src/vae3d/extract_latent_nifti.py \\
        --vae-type aekl \\
        --checkpoint outputs/vae3d/runs/vae3d_T1W/weights/model_best.pth \\
        --config configs/vae3d_multimodal.yaml \\
        --env local \\
        --input /path/to/subject.nii.gz \\
        --output outputs/latents/subject_z.nii.gz

Usage — batch (modality × field from prospective split):
    python src/vae3d/extract_latent_nifti.py \\
        --vae-type aekl \\
        --checkpoint outputs/vae3d/runs/vae3d_T1W/weights/model_best.pth \\
        --config configs/vae3d_multimodal.yaml \\
        --env local \\
        --batch \\
        --split Training_prospective \\
        --output-dir outputs/latents/aekl_prospective
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import nibabel as nib
import numpy as np
import torch

MODALITIES = ["T1W", "T2W", "T2FLAIR"]
FIELDS     = ["0.1T", "1.5T", "3T", "5T", "7T"]


# ---------------------------------------------------------------------------
# Core extraction (single volume)
# ---------------------------------------------------------------------------

def extract_one(vae, path: Path, output_path: Path, patch_size=(128, 128, 128),
                device="cpu") -> nib.Nifti1Image:
    """Encode a volume and save latent as NIfTI.

    Uses the `extract_latent_nifti` method from MRIxFieldsVAE (vae_base.py).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # MRIxFieldsVAE.extract_latent_nifti already handles loading + normalisation
    nifti = vae.extract_latent_nifti(
        volume=path,
        output_path=output_path,
        patch_size=patch_size,
        device=device,
        compress=True,
    )
    return nifti


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def extract_batch(vae, data_root: Path, split: str, output_dir: Path,
                  patch_size=(128, 128, 128), device="cpu",
                  modalities=None, fields=None):
    """Encode all volumes in a split and mirror the directory structure.

    Output structure:
        output_dir/<modality>/<field>/<subject>_z.nii.gz
    """
    mods = modalities or MODALITIES
    flds = fields or FIELDS
    split_dir = data_root / split
    if not split_dir.exists():
        print(f"[ERROR] Split directory not found: {split_dir}")
        sys.exit(1)

    total = 0
    skipped = 0
    for mod in mods:
        for field in flds:
            src_dir = split_dir / mod / field
            if not src_dir.exists():
                continue
            dst_dir = output_dir / mod / field
            dst_dir.mkdir(parents=True, exist_ok=True)
            files = sorted(src_dir.glob("*.nii.gz"))
            for path in files:
                out_name = path.stem.replace(".nii", "") + "_z.nii.gz"
                out_path = dst_dir / out_name
                if out_path.exists():
                    print(f"  [SKIP] {out_path.name} already exists")
                    skipped += 1
                    continue
                try:
                    extract_one(vae, path, out_path, patch_size=patch_size,
                                device=device)
                    total += 1
                    print(f"  [{mod}/{field}] {path.name} -> {out_path.name}")
                except Exception as e:
                    print(f"  [WARN] {path.name}: {e}")
                    skipped += 1

    print(f"\n[INFO] Done: {total} extracted, {skipped} skipped/failed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract VAE latents as NIfTI")
    parser.add_argument("--vae-type", default="aekl")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--env", default="local")
    parser.add_argument("--device", default=None)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128],
                        metavar=("H", "W", "D"))

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", default=None,
                      help="Single input NIfTI path")
    mode.add_argument("--batch", action="store_true",
                      help="Process an entire split")

    # Single mode
    parser.add_argument("--output", default=None,
                        help="Output NIfTI path (single mode)")

    # Batch mode
    parser.add_argument("--split", default="Training_prospective",
                        help="Data split (batch mode)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (batch mode)")
    parser.add_argument("--modalities", nargs="+", default=None,
                        help="Subset of modalities (default: all)")
    parser.add_argument("--fields", nargs="+", default=None,
                        help="Subset of fields (default: all)")
    parser.add_argument("--data-root", default=None)

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
    print(f"[INFO] VAE: {args.vae_type}  "
          f"format={vae.latent_format}  vector_dim={vae.vector_dim}")

    patch_size = tuple(args.patch_size)

    # ── Single mode ──────────────────────────────────────────────────────────
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[ERROR] Input not found: {input_path}")
            sys.exit(1)
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = input_path.parent / (input_path.stem.replace(".nii", "") + "_z.nii.gz")

        print(f"[INFO] Extracting latent: {input_path.name} -> {out_path}")
        nifti = extract_one(vae, input_path, out_path,
                            patch_size=patch_size, device=device)
        print(f"[INFO] Latent shape: {nifti.get_fdata().shape}  saved: {out_path}")
        return

    # ── Batch mode ───────────────────────────────────────────────────────────
    if args.data_root:
        data_root = Path(args.data_root)
    elif "data" in cfg and "data_root" in cfg["data"]:
        data_root = Path(cfg["data"]["data_root"])
    else:
        data_root = Path.home() / "Data" / "MRIxFields_20260414"

    if not args.output_dir:
        out_dir_name = f"latents_{args.vae_type}_{args.split.lower().replace('training_', '')}"
        output_dir = REPO_ROOT / "outputs" / out_dir_name
    else:
        output_dir = Path(args.output_dir)

    print(f"[INFO] Batch extraction: {data_root / args.split} -> {output_dir}")
    extract_batch(
        vae, data_root, args.split, output_dir,
        patch_size=patch_size, device=device,
        modalities=args.modalities, fields=args.fields,
    )


if __name__ == "__main__":
    main()
