#!/usr/bin/env python3
"""QC figure script — reconstruction quality comparison for all VAEs.

Generates 9 figures (3 modalities × 3 subjects) showing side-by-side
reconstructions across all 5 field strengths for every VAE architecture.

Layout per figure:
  rows  = Original + 7 VAEs  (8 rows)
  cols  = 5 fields (0.1T, 1.5T, 3T, 5T, 7T)

Style: dark background, colored row labels, SSIM overlay on reconstructions.

Output: results/benchmark_vae/visuals/comparison_{modality}_{subject}.png

Usage:
    PYTHONPATH=src python src/vae3d/qc_all_vaes.py [--device cuda] [--modalities T1W]
"""

from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter

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

# ─── Constants ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_DEFAULT = "/home/rousseau/Data/MRIxFields_20260414"

BG_COLOR = "#08080f"
NA_COLOR = "#18182a"

ROW_COLORS = {
    "Original": "#88ccff",
    "AEKL": "#ffcc66",
    "Pythae_VAE": "#88ff88",
    "Pythae_VQVAE": "#ff9999",
    "Pythae_RHVAE": "#ff66ff",
    "MedVAE_frozen": "#ff9944",
    "MedVAE_finetuned": "#cccc66",
    "NV_Generate": "#66ccff",
}


def _ssim_2d(im1: np.ndarray, im2: np.ndarray) -> float:
    c1, c2 = 0.01**2, 0.03**2
    m1 = gaussian_filter(im1.astype(float), 1.5)
    m2 = gaussian_filter(im2.astype(float), 1.5)
    s1 = gaussian_filter(im1**2, 1.5) - m1**2
    s2 = gaussian_filter(im2**2, 1.5) - m2**2
    s12 = gaussian_filter(im1 * im2, 1.5) - m1 * m2
    return float(
        np.mean(
            ((2 * m1 * m2 + c1) * (2 * s12 + c2))
            / ((m1**2 + m2**2 + c1) * (s1 + s2 + c2) + 1e-8)
        )
    )


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
            result = vae_w.forward(x, encode_only=False, batch_size=8)
        xhat = result["reconstruction"].squeeze().cpu().numpy()
    return np.clip(xhat, 0.0, 1.0)


def make_comparison_figure(
    subject: str,
    modality: str,
    originals: Dict[str, Optional[np.ndarray]],
    reconstructions: Dict[str, Dict[str, Optional[np.ndarray]]],
    output_path: Path,
) -> None:
    """Generate and save a dark-styled comparison figure with SSIM overlay."""
    vae_names = list(reconstructions.keys())
    fields = list(DOMAINS)
    row_names = ["Original"] + vae_names
    n_rows = len(row_names)
    n_cols = len(fields)

    cell_w, cell_h = 2.5, 2.5
    lmargin = 1.4
    tmargin = 0.7
    fig_w = lmargin + cell_w * n_cols + 0.3
    fig_h = tmargin + cell_h * n_rows + 0.2

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG_COLOR)
    fig.suptitle(
        f"VAE Comparison  —  {modality}  sub-{subject}",
        fontsize=12,
        fontweight="bold",
        color="white",
        y=0.99,
    )

    left_n = lmargin / fig_w
    right_n = 1.0 - 0.3 / fig_w
    top_n = 1.0 - tmargin / fig_h
    bottom_n = 0.2 / fig_h

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.04, wspace=0.03,
        left=left_n, right=right_n,
        top=top_n, bottom=bottom_n,
    )

    for row_idx, row_name in enumerate(row_names):
        row_color = ROW_COLORS.get(row_name, "white")

        for col_idx, field in enumerate(fields):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.set_facecolor("#111120")
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                vol = originals.get(field)
            else:
                vol = reconstructions[row_name].get(field)

            if vol is not None:
                sl = get_axial_slice(vol)
                ax.imshow(
                    sl.T,
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                    aspect="equal",
                    origin="lower",
                    interpolation="bilinear",
                )

                # SSIM overlay for reconstruction rows
                if row_idx > 0 and originals.get(field) is not None:
                    try:
                        orig_sl = get_axial_slice(originals[field])
                        ssim_v = _ssim_2d(orig_sl, sl)
                        ax.text(
                            0.03, 0.97,
                            f"SSIM {ssim_v:.3f}",
                            ha="left", va="top",
                            fontsize=7,
                            color="yellow",
                            transform=ax.transAxes,
                            bbox=dict(
                                facecolor="#000000",
                                alpha=0.55,
                                pad=1.5,
                                edgecolor="none",
                            ),
                        )
                    except Exception:
                        pass
            else:
                ax.set_facecolor(NA_COLOR)
                ax.text(
                    0.5, 0.5,
                    "N/A",
                    ha="center", va="center",
                    fontsize=8.5,
                    color="#5555aa",
                    transform=ax.transAxes,
                )

            # Column headers
            if row_idx == 0:
                ax.set_title(
                    field,
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                    pad=5,
                )

            # Row labels
            if col_idx == 0:
                ax.set_ylabel(
                    row_name,
                    fontsize=9.5,
                    color=row_color,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=8,
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(output_path),
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
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

            # Load originals (full volume)
            originals: Dict[str, Optional[np.ndarray]] = {}
            for field in DOMAINS:
                vol = load_prospective_volume(data_root, modality, field, subject)
                originals[field] = vol

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

            out_path = output_dir / f"comparison_{modality}_{subject}.png"
            make_comparison_figure(subject, modality, originals, reconstructions, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate VAE comparison figures on prospective data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES))
    parser.add_argument("--subjects", nargs="+", default=PROSPECTIVE_SUBJECTS)
    parser.add_argument(
        "--output-dir", default="results/benchmark_vae/visuals"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--vae", nargs="+", default=None, help="Run only these VAEs")
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
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
