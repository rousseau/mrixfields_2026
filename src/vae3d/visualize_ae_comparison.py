#!/usr/bin/env python3
"""
Analyse qualitative comparative des 4 autoencodeurs VAE 3D.

Génère 3 figures PNG (une par modalité T1W / T2W / T2FLAIR) avec :
  Lignes  : Original | AEKL | VQ-VAE | MedVAE frozen | MedVAE ft
  Colonnes: 0.1T | 1.5T | 3T | 5T | 7T  — vue axiale centrale

Sujet prospectif unique (même anatomie sur tous les champs magnétiques).

Usage :
  cd PROJECT_ROOT
  python src/vae3d/visualize_ae_comparison.py --subject-idx 0
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "vae3d"))

import argparse
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter, uniform_filter1d
from scipy.ndimage import zoom as scipy_zoom

from vae3d.benchmark_vae import (
    FIELDS_IDX,
    MODALITIES_IDX,
    VQVAECompatWrapper,
    load_aekl,
    load_medvae,
    load_vqvae,
)

# ─── Constantes ──────────────────────────────────────────────────────────────
MODALITIES = ["T1W", "T2W", "T2FLAIR"]
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
AE_NAMES = ["AEKL", "VQ-VAE", "MedVAE frozen", "MedVAE ft"]

# Taille du patch : multiple de 16 (VQ-VAE 16× downsample) et de 4 (AEKL/MedVAE)
# (256, 256, 80) → axial = patch[:, :, 40] de forme (256, 256) — aspect carré
PATCH_XY = 256
PATCH_Z = 80

# Couleurs
BG_COLOR = "#08080f"
NA_COLOR = "#18182a"

# Etiquette de style par ligne
ROW_COLORS = {
    "Original": "#88ccff",
    "AEKL": "#ffcc66",
    "VQ-VAE": "#88ff88",
    "MedVAE frozen": "#ff9999",
    "MedVAE ft": "#ff66ff",
}


# ─── Utilitaires données ──────────────────────────────────────────────────────


def _normalize(vol: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    p_lo, p_hi = np.percentile(vol, lo), np.percentile(vol, hi)
    if p_hi <= p_lo:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol - p_lo) / (p_hi - p_lo), 0.0, 1.0).astype(np.float32)


def load_and_patch(
    path: Path,
    patch_xy: int = PATCH_XY,
    patch_z: int = PATCH_Z,
):
    """
    Charge un volume NIfTI, normalise, extrait un patch (patch_xy, patch_xy, patch_z).

    Centrage :
      - x, y  : centre géométrique du volume (coupes sagittale et coronale)
      - z      : tranche axiale la plus brillante (maximum d'intensité moyenne)
                 → approximativement au niveau des ventricules latéraux

    Retourne : (patch_3d,  z_local)  avec  z_local ≈ patch_z // 2
    """
    img = nib.load(str(path))
    vol = _normalize(img.get_fdata(dtype=np.float32))
    H, W, D = vol.shape

    # Centre en x, y
    x0 = max((H - patch_xy) // 2, 0)
    y0 = max((W - patch_xy) // 2, 0)
    x0 = min(x0, max(H - patch_xy, 0))
    y0 = min(y0, max(W - patch_xy, 0))

    # Tranche axiale la plus brillante (sur la région x,y du patch)
    roi = vol[x0 : x0 + patch_xy, y0 : y0 + patch_xy, :]
    z_means = roi.mean(axis=(0, 1))
    z_smooth = uniform_filter1d(z_means, size=20)
    z_best = int(np.argmax(z_smooth))

    # Centre en z autour de z_best
    z0 = max(0, z_best - patch_z // 2)
    z0 = min(z0, max(D - patch_z, 0))

    patch = vol[x0 : x0 + patch_xy, y0 : y0 + patch_xy, z0 : z0 + patch_z]

    # Padding si nécessaire
    pad = [
        (0, max(0, patch_xy - patch.shape[0])),
        (0, max(0, patch_xy - patch.shape[1])),
        (0, max(0, patch_z - patch.shape[2])),
    ]
    if any(p[1] > 0 for p in pad):
        patch = np.pad(patch, pad, mode="reflect")

    z_local = z_best - z0  # ≈ patch_z // 2
    return patch.astype(np.float32), z_local


# ─── Inférence modèle ────────────────────────────────────────────────────────


def reconstruct(
    model: nn.Module, patch: np.ndarray, device: torch.device
) -> np.ndarray:
    """Encode → decode un seul patch 3D. Retourne le volume reconstruit."""
    x = torch.from_numpy(patch)[None, None].to(device)  # (1, 1, H, W, D)
    with torch.no_grad():
        z = model.encode(x)
        if isinstance(z, tuple):
            z = z[0]  # prendre la moyenne pour un VAE distribution
        xr = model.decode(z)
    out = xr.squeeze().cpu().float().numpy()

    # Recadrage si la taille de sortie diffère (ne devrait pas arriver avec des
    # patches multiples de 16, mais sécurité)
    if out.shape != patch.shape:
        factors = tuple(t / s for t, s in zip(patch.shape, out.shape))
        out = scipy_zoom(out, factors, order=1)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ─── Métriques ───────────────────────────────────────────────────────────────


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


# ─── Construction de la figure ───────────────────────────────────────────────


def make_figure(
    modality: str,
    slices_dict: dict,  # (row_name, field) → 2D array ou None
    ssim_dict: dict,  # (ae_name, field)  → float ou None
    subject_id: str = "",
) -> plt.Figure:
    """
    Construit la figure comparative pour une modalité donnée.

    Disposition :
      Lignes  : Original + 4 AE   (5 lignes)
      Colonnes: 5 champs           (5 colonnes)
    """
    row_names = ["Original"] + AE_NAMES
    n_rows = len(row_names)  # 5
    n_cols = len(FIELDS)  # 5

    cell_w = 2.5  # inches
    cell_h = 2.5
    lmargin = 1.4  # espace pour les étiquettes de lignes
    tmargin = 0.7  # titre

    fig_w = lmargin + cell_w * n_cols + 0.3
    fig_h = tmargin + cell_h * n_rows + 0.2

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG_COLOR)

    subj_label = f"  (sujet {subject_id})" if subject_id else ""
    fig.suptitle(
        f"Comparaison qualitative des autoencodeurs  —  {modality}{subj_label}",
        fontsize=12,
        fontweight="bold",
        color="white",
        y=0.99,
    )

    # Marges normalisées
    left_n = lmargin / fig_w
    right_n = 1.0 - 0.3 / fig_w
    top_n = 1.0 - tmargin / fig_h
    bottom_n = 0.2 / fig_h

    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        hspace=0.04,
        wspace=0.03,
        left=left_n,
        right=right_n,
        top=top_n,
        bottom=bottom_n,
    )

    for row_idx, row_name in enumerate(row_names):
        row_color = ROW_COLORS.get(row_name, "white")

        for col_idx, field in enumerate(FIELDS):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.set_facecolor("#111120")
            sl = slices_dict.get((row_name, field))

            if sl is not None:
                # Affichage de la tranche (transposée pour orientation correcte)
                ax.imshow(
                    sl.T,
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                    aspect="equal",
                    origin="lower",
                    interpolation="bilinear",
                )
                # Annotation SSIM (lignes de reconstruction uniquement)
                if row_idx > 0:
                    ssim_v = ssim_dict.get((row_name, field))
                    if ssim_v is not None:
                        ax.text(
                            0.03,
                            0.97,
                            f"SSIM {ssim_v:.3f}",
                            ha="left",
                            va="top",
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
            else:
                # Modèle non applicable pour cette modalité
                ax.set_facecolor(NA_COLOR)
                ax.text(
                    0.5,
                    0.5,
                    "N/A\n(T1W only)",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="#5555aa",
                    transform=ax.transAxes,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            # En-têtes de colonnes (champs magnétiques)
            if row_idx == 0:
                ax.set_title(
                    field,
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                    pad=5,
                )

            # Étiquettes de lignes (noms des modèles)
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

    return fig


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Figures qualitatives comparatives 4 AE × 3 modalités × 5 champs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        default="/home/rousseau/Data/MRIxFields_20260414",
    )
    parser.add_argument(
        "--split",
        default="Training_prospective",
        choices=["Training_prospective", "Training_retrospective"],
        help="Sous-dossier de données. 'Training_prospective' = même sujet sur tous les champs.",
    )
    parser.add_argument(
        "--subject-idx",
        type=int,
        default=0,
        help="Index du sujet dans la liste triée (0 = premier, ex. P_T1W_0.1T_0006).",
    )
    parser.add_argument(
        "--aekl-ckpt",
        default="outputs/vae3d/runs/vae3d_T1W_jeanzay/weights/model_final.pth",
    )
    parser.add_argument(
        "--vqvae-ckpt",
        default="outputs/vqvae3d/runs/vqvae_final/weights/model_best.pth",
    )
    parser.add_argument(
        "--medvae-finetuned-ckpt",
        default="outputs/medvae/runs/medvae_finetune_all/weights/model_final.pth",
    )
    parser.add_argument("--medvae-model-name", default="medvae_4_1_3d")
    parser.add_argument("--output-dir", default="results/qc/ae_comparison")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Résolution des figures PNG.",
    )
    parser.add_argument(
        "--patch-xy",
        type=int,
        default=256,
        help="Taille du patch dans les dimensions x et y (doit être multiple de 16).",
    )
    parser.add_argument(
        "--patch-z",
        type=int,
        default=80,
        help="Épaisseur du patch dans la dimension axiale (doit être multiple de 16).",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device : {device}")
    print(f"Split  : {args.split}  (subject_idx={args.subject_idx})")
    print(f"Patch  : ({args.patch_xy}, {args.patch_xy}, {args.patch_z})\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    # ── Chargement des modèles (une seule fois) ───────────────────────────────
    print("📦 Chargement des modèles…")

    aekl = load_aekl(args.aekl_ckpt, device)
    if aekl:
        aekl.eval()

    vqvae_base = load_vqvae(args.vqvae_ckpt, device)

    medvae_frz = load_medvae(args.medvae_model_name, device, checkpoint_path=None)
    if medvae_frz:
        medvae_frz.eval()

    ckpt_ft = Path(args.medvae_finetuned_ckpt)
    medvae_ft = load_medvae(
        args.medvae_model_name,
        device,
        checkpoint_path=ckpt_ft if ckpt_ft.exists() else None,
    )
    if medvae_ft:
        medvae_ft.eval()

    loaded = [
        n
        for n, m in [
            ("AEKL", aekl),
            ("VQ-VAE", vqvae_base),
            ("MedVAE frozen", medvae_frz),
            ("MedVAE ft", medvae_ft),
        ]
        if m is not None
    ]
    print(f"✓ Modèles chargés : {loaded}\n")

    # ── Boucle sur les modalités ──────────────────────────────────────────────
    for modality in MODALITIES:
        print(f"\n{'═' * 60}")
        print(f"  Modalité : {modality}")
        print(f"{'═' * 60}")

        slices_dict: dict = {}
        ssim_dict: dict = {}

        # Récupère l'ID du premier sujet pour le titre
        sample_dir = data_root / args.split / modality / FIELDS[0]
        sample_files = sorted(sample_dir.glob("*.nii.gz"))
        subject_id = ""
        if sample_files:
            fname = sample_files[args.subject_idx % len(sample_files)].stem
            # Ex : P_T1W_0.1T_0006 → subject ID = 0006
            subject_id = fname.split("_")[-1]

        for field in FIELDS:
            print(f"\n  ── {field} ──")

            vol_dir = data_root / args.split / modality / field
            files = sorted(vol_dir.glob("*.nii.gz"))

            if not files:
                print(f"    ⚠ Aucun fichier trouvé — cellule vide")
                for row in ["Original"] + AE_NAMES:
                    slices_dict[(row, field)] = None
                continue

            path = files[args.subject_idx % len(files)]
            print(f"    Fichier : {path.name}")

            # Extraction du patch
            patch, z_loc = load_and_patch(path, args.patch_xy, args.patch_z)
            orig_axial = patch[:, :, z_loc]
            slices_dict[("Original", field)] = orig_axial
            print(f"    Patch   : {patch.shape}  z_loc={z_loc}")

            # ── AEKL ─────────────────────────────────────────────────────
            if modality == "T1W" and aekl is not None:
                try:
                    rec = reconstruct(aekl, patch, device)
                    sl = rec[:, :, z_loc]
                    slices_dict[("AEKL", field)] = sl
                    ssim_dict[("AEKL", field)] = _ssim_2d(orig_axial, sl)
                    print(f"    AEKL        SSIM={ssim_dict[('AEKL', field)]:.4f}")
                except Exception as e:
                    print(f"    AEKL error : {e}")
                    slices_dict[("AEKL", field)] = None
            else:
                slices_dict[("AEKL", field)] = None  # N/A pour T2W / T2FLAIR

            # ── VQ-VAE ───────────────────────────────────────────────────
            if vqvae_base is not None:
                mod_idx = MODALITIES_IDX.get(modality, 0)
                field_idx = FIELDS_IDX.get(field, 0)
                vqvae = VQVAECompatWrapper(
                    vqvae_base, mod_idx=mod_idx, field_idx=field_idx
                ).to(device)
                try:
                    rec = reconstruct(vqvae, patch, device)
                    sl = rec[:, :, z_loc]
                    slices_dict[("VQ-VAE", field)] = sl
                    ssim_dict[("VQ-VAE", field)] = _ssim_2d(orig_axial, sl)
                    print(f"    VQ-VAE      SSIM={ssim_dict[('VQ-VAE', field)]:.4f}")
                except Exception as e:
                    print(f"    VQ-VAE error : {e}")
                    slices_dict[("VQ-VAE", field)] = None

            # ── MedVAE frozen ─────────────────────────────────────────────
            if medvae_frz is not None:
                try:
                    rec = reconstruct(medvae_frz, patch, device)
                    sl = rec[:, :, z_loc]
                    slices_dict[("MedVAE frozen", field)] = sl
                    ssim_dict[("MedVAE frozen", field)] = _ssim_2d(orig_axial, sl)
                    print(
                        f"    MedVAE frz  SSIM={ssim_dict[('MedVAE frozen', field)]:.4f}"
                    )
                except Exception as e:
                    print(f"    MedVAE frozen error : {e}")
                    slices_dict[("MedVAE frozen", field)] = None

            # ── MedVAE ft ────────────────────────────────────────────────
            if medvae_ft is not None:
                try:
                    rec = reconstruct(medvae_ft, patch, device)
                    sl = rec[:, :, z_loc]
                    slices_dict[("MedVAE ft", field)] = sl
                    ssim_dict[("MedVAE ft", field)] = _ssim_2d(orig_axial, sl)
                    print(f"    MedVAE ft   SSIM={ssim_dict[('MedVAE ft', field)]:.4f}")
                except Exception as e:
                    print(f"    MedVAE ft error : {e}")
                    slices_dict[("MedVAE ft", field)] = None

        # ── Génération et sauvegarde de la figure ─────────────────────────
        print(f"\n  Génération de la figure…")
        fig = make_figure(modality, slices_dict, ssim_dict, subject_id=subject_id)

        out_path = out_dir / f"ae_comparison_{modality}.png"
        fig.savefig(
            str(out_path),
            dpi=args.dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        print(f"  ✓ Figure sauvegardée : {out_path}")

    print(f"\n✅ Terminé. 3 figures dans : {out_dir}/")


if __name__ == "__main__":
    main()
