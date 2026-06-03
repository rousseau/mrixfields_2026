#!/usr/bin/env python3
"""QC figure script — reconstruction quality for all VAEs on prospective data.

For each (subject, modality) pair, generates a figure showing the original volumes
at all 5 field strengths and the reconstruction from each VAE side by side.

Layout per figure:
  rows = VAEs (AEKL, Pythae_VAE, Pythae_VQVAE, Pythae_RHVAE, MedVAE_frozen,
               MedVAE_finetuned, NV_Generate)
  cols = fields (0.1T, 1.5T, 3T, 5T, 7T) × 2 (original | recon)
  Top header row shows originals only.

Output: results/qc/qc_{modality}_{subject}.png

Usage:
    PYTHONPATH=src python src/vae3d/qc_all_vaes.py [--device cuda] [--modalities T1W]
"""

from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common.io import DOMAINS, MODALITIES
from models.vae_loader import load_vae
from utils.patched_vae import PatchedVAE

# Re-use registry from benchmark_vae
from vae3d.benchmark_vae import (
    VAE_REGISTRY,
    PATCH_SIZE,
    PATCH_OVERLAP,
    RHVAE_VOLUME_SIZE,
    PROSPECTIVE_SUBJECTS,
    load_prospective_volume,
    crop_or_pad,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_DEFAULT = "/home/rousseau/Data/MRIxFields_20260414"


def get_axial_slice(vol: np.ndarray) -> np.ndarray:
    """Return central axial slice (axis 2)."""
    mid = vol.shape[2] // 2
    return vol[:, :, mid]


def reconstruct_volume(
    vae: nn.Module,
    vol: np.ndarray,
    device: torch.device,
    is_rhvae: bool,
) -> np.ndarray:
    """Encode + decode a volume; return reconstruction clipped to [0,1]."""
    if is_rhvae:
        vol_crop = crop_or_pad(vol, RHVAE_VOLUME_SIZE)
        x = torch.from_numpy(vol_crop).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            z = vae.encode(x)
            xhat = vae.decode(z).squeeze().cpu().numpy()
    else:
        vae_w = PatchedVAE(vae, patch_size=PATCH_SIZE, overlap=PATCH_OVERLAP).to(device)
        x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            result = vae_w.forward(x, encode_only=False, batch_size=1)
        xhat = result["reconstruction"].squeeze().cpu().numpy()
    return np.clip(xhat, 0.0, 1.0)


def make_qc_figure(
    subject: str,
    modality: str,
    originals: Dict[str, Optional[np.ndarray]],
    reconstructions: Dict[str, Dict[str, Optional[np.ndarray]]],
    output_path: Path,
) -> None:
    """Generate and save a QC figure."""
    vae_names = list(reconstructions.keys())
    fields = list(DOMAINS)

    n_rows = 1 + len(vae_names)
    n_cols = len(fields)

    fig_w = max(18, n_cols * 2.2 + 3)
    fig_h = max(12, n_rows * 2.2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f"VAE Reconstruction QC — {modality} sub-{subject}", fontsize=13, y=0.98)

    row_labels = ["Original"] + [v.replace("_", " ") for v in vae_names]

    def _show(ax: plt.Axes, vol: Optional[np.ndarray], title: str = "") -> None:
        if vol is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="grey")
            ax.set_facecolor("#f0f0f0")
        else:
            sl = get_axial_slice(vol)
            ax.imshow(np.rot90(sl), cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
        if title:
            ax.set_title(title, fontsize=7, pad=2)
        ax.axis("off")

    # Row 0 — originals
    for j, field in enumerate(fields):
        _show(axes[0, j], originals.get(field), title=field)

    # Rows 1+ — reconstructions
    for i, vae_name in enumerate(vae_names):
        row = i + 1
        recon_dict = reconstructions[vae_name]
        for j, field in enumerate(fields):
            _show(axes[row, j], recon_dict.get(field))

    # Add row labels using fig.text (robust)
    for row_idx, label in enumerate(row_labels):
        y_pos = 1.0 - (row_idx + 0.5) / n_rows
        fig.text(0.01, y_pos, label, fontsize=8, rotation=0,
                 va="center", ha="left", fontweight="bold")

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.02, top=0.92,
                        wspace=0.05, hspace=0.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def run_qc(
    modalities: List[str],
    subjects: List[str],
    data_root: Path,
    output_dir: Path,
    device: torch.device,
    vae_names: Optional[List[str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for modality in modalities:
        for subject in subjects:
            print(f"\n{'='*70}")
            print(f"  QC: {modality} sub-{subject}")
            print(f"{'='*70}")

            # Load originals
            originals: Dict[str, Optional[np.ndarray]] = {}
            for field in DOMAINS:
                originals[field] = load_prospective_volume(data_root, modality, field, subject)

            # Compute reconstructions per VAE
            reconstructions: Dict[str, Dict[str, Optional[np.ndarray]]] = {}

            for vae_name, vae_cfg, _partial, _epoch in VAE_REGISTRY:
                if vae_names and vae_name not in vae_names:
                    continue
                print(f"  VAE: {vae_name}")
                try:
                    vae = load_vae(vae_cfg, device)
                except Exception as e:
                    print(f"    ✗ Load failed: {e}")
                    reconstructions[vae_name] = {f: None for f in DOMAINS}
                    continue

                is_rhvae = (vae.latent_format == "vector")
                vae.eval()
                recon_dict: Dict[str, Optional[np.ndarray]] = {}

                for field in DOMAINS:
                    vol = originals.get(field)
                    if vol is None:
                        recon_dict[field] = None
                        continue
                    try:
                        recon = reconstruct_volume(vae, vol, device, is_rhvae)
                        recon_dict[field] = recon
                    except Exception as e:
                        print(f"    ✗ {field}: {e}")
                        recon_dict[field] = None

                reconstructions[vae_name] = recon_dict

                del vae
                torch.cuda.empty_cache()

            out_path = output_dir / f"qc_{modality}_{subject}.png"
            make_qc_figure(subject, modality, originals, reconstructions, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate QC figures for all VAEs on prospective data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES))
    parser.add_argument("--subjects", nargs="+", default=PROSPECTIVE_SUBJECTS)
    parser.add_argument("--output-dir", default="results/qc")
    parser.add_argument("--device", default=None)
    parser.add_argument("--vae", nargs="+", default=None,
                        help="Run only these VAEs")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    run_qc(
        modalities=args.modalities,
        subjects=args.subjects,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        device=device,
        vae_names=args.vae,
    )


if __name__ == "__main__":
    main()
