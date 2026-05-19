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
AE_NAMES = ["AEKL", "VQ-VAE", "MedVAE 4×1", "MedVAE 4×1 ft", "MedVAE 8×1"]

# Champ de référence pour le calcul de z_best (robuste pour tous les AE)
REF_FIELD = "3T"

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
    "MedVAE 4×1": "#ff9999",
    "MedVAE 4×1 ft": "#ff66ff",
    "MedVAE 8×1": "#ff9944",
}


# ─── Utilitaires données ──────────────────────────────────────────────────────


def _normalize_model(vol: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    """
    Normalisation [-1, 1] pour l'entrée des modèles.
    Identique à benchmark_vae._normalize — les modèles ont été entraînés sur ce range.
    """
    p_lo, p_hi = np.percentile(vol, lo), np.percentile(vol, hi)
    if p_hi <= p_lo:
        return np.zeros_like(vol, dtype=np.float32)
    clipped = np.clip((vol - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    return (clipped * 2.0 - 1.0).astype(np.float32)  # [-1, 1]


def _model_to_display(arr: np.ndarray) -> np.ndarray:
    """Convertit la sortie du modèle [-1, 1] → [0, 1] pour l'affichage matplotlib."""
    return np.clip((arr + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def _find_z_best(
    vol_raw: np.ndarray, x0: int, y0: int, patch_xy: int, size_smooth: int = 20
) -> int:
    """
    Trouve l'indice de la tranche axiale la plus brillante dans la région [x0,y0]
    du volume brut (non normalisé). Utilisé comme référence inter-champs.
    """
    roi = vol_raw[x0 : x0 + patch_xy, y0 : y0 + patch_xy, :].astype(np.float32)
    # Normalisation locale pour ne pas être biaisé par les intensités absolues
    lo, hi = np.percentile(roi, 0.5), np.percentile(roi, 99.5)
    roi_n = np.clip((roi - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    z_means = roi_n.mean(axis=(0, 1))
    z_smooth = uniform_filter1d(z_means, size=size_smooth)
    return int(np.argmax(z_smooth))


def load_and_patch(
    path: Path,
    patch_xy: int = PATCH_XY,
    patch_z: int = PATCH_Z,
    z_best_ref: int = -1,
):
    """
    Charge un volume NIfTI et extrait un patch (patch_xy, patch_xy, patch_z).

    Centrage :
      - x, y  : centre géométrique du volume
      - z      : si z_best_ref >= 0, utilise cette position de référence
                 (indispensable pour données recalées, même sujet différents champs)
                 sinon calcule la tranche la plus brillante

    Retourne :
      patch_model   : array [-1, 1]  → entrée modèles
      patch_display : array [0, 1]   → affichage de l'original
      z_best        : indice de la tranche de référence dans le volume complet
      x0, y0        : coin supérieur gauche du patch en x, y
    """
    img = nib.load(str(path))
    vol_raw = img.get_fdata(dtype=np.float32)
    H, W, D = vol_raw.shape

    # Centre en x, y
    x0 = min(max((H - patch_xy) // 2, 0), max(H - patch_xy, 0))
    y0 = min(max((W - patch_xy) // 2, 0), max(W - patch_xy, 0))

    # Tranche axiale de référence
    if z_best_ref < 0:
        z_best = _find_z_best(vol_raw, x0, y0, patch_xy)
    else:
        z_best = int(np.clip(z_best_ref, patch_z // 2, D - patch_z // 2 - 1))

    # Centre en z autour de z_best
    z0 = int(np.clip(z_best - patch_z // 2, 0, max(D - patch_z, 0)))
    z_local = z_best - z0  # index dans le patch (≈ patch_z // 2)

    # Extraction du patch
    def _extract(v):
        p = v[x0 : x0 + patch_xy, y0 : y0 + patch_xy, z0 : z0 + patch_z]
        pad = [
            (0, max(0, patch_xy - p.shape[0])),
            (0, max(0, patch_xy - p.shape[1])),
            (0, max(0, patch_z - p.shape[2])),
        ]
        return np.pad(p, pad, mode="reflect") if any(q[1] > 0 for q in pad) else p

    patch_raw = _extract(vol_raw)
    patch_model = _normalize_model(patch_raw)  # [-1, 1]
    patch_display = _model_to_display(patch_model)  # [0, 1]

    return (
        patch_model.astype(np.float32),
        patch_display.astype(np.float32),
        z_best,
        x0,
        y0,
    )


# ─── Inférence modèle ────────────────────────────────────────────────────────


def reconstruct(
    model: nn.Module, patch_model: np.ndarray, device: torch.device
) -> np.ndarray:
    """
    Encode → decode un seul patch 3D.

    Args:
      patch_model : array [-1, 1] (normalisation modèle)
    Returns:
      array [0, 1] prêt pour l'affichage matplotlib
    """
    x = torch.from_numpy(patch_model)[None, None].to(device)  # (1, 1, H, W, D)
    with torch.no_grad():
        z = model.encode(x)
        if isinstance(z, tuple):
            z = z[0]  # prendre la moyenne pour un VAE distribution
        xr = model.decode(z)
    out = xr.squeeze().cpu().float().numpy()

    # Recadrage si la taille de sortie diffère
    if out.shape != patch_model.shape:
        factors = tuple(t / s for t, s in zip(patch_model.shape, out.shape))
        out = scipy_zoom(out, factors, order=1)

    # Sortie du modèle en [-1,1] → [0,1] pour l'affichage
    return _model_to_display(out)


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
        help="Taille du patch dans les dimensions x et y (doit être multiple de 16). "
        "Pour VQ-VAE, utiliser 112 (entraîné sur 112x128x80).",
    )
    parser.add_argument(
        "--patch-z",
        type=int,
        default=80,
        help="Épaisseur du patch dans la dimension axiale (doit être multiple de 16). "
        "Pour VQ-VAE, utiliser 80 (entraîné sur 112x128x80).",
    )
    parser.add_argument(
        "--vqvae-training-patch",
        action="store_true",
        help="Forcer la taille de patch d'entraînement VQ-VAE (112,128) "
        "pour une comparaison équitable.",
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

    # ── Chargement des modèles (une seule fois) ───────────────────────────────────
    print("📦 Chargement des modèles…")

    aekl = load_aekl(args.aekl_ckpt, device)
    if aekl:
        aekl.eval()

    vqvae_base = load_vqvae(args.vqvae_ckpt, device)

    # --- Vérification patch size pour VQ-VAE ---
    VQ_TRAINING_PATCH = (112, 128, 80)
    if args.vqvae_training_patch:
        patch_xy, patch_z = VQ_TRAINING_PATCH[0], VQ_TRAINING_PATCH[2]
        print(f"⚠ VQ-VAE patch override → ({patch_xy}, {patch_xy}, {patch_z})")
    else:
        patch_xy, patch_z = args.patch_xy, args.patch_z
        if vqvae_base is not None:
            if (patch_xy, patch_z) != (VQ_TRAINING_PATCH[0], VQ_TRAINING_PATCH[2]):
                print(
                    f"⚠ VQ-VAE patch mismatch: evaluation ({patch_xy},{patch_z}) vs training {VQ_TRAINING_PATCH}"
                )
                print(
                    f"   → Ajouter --vqvae-training-patch pour une comparaison équitable"
                )

    # MedVAE 4×1 frozen (poids HuggingFace)
    medvae_4x1 = load_medvae("medvae_4_1_3d", device, checkpoint_path=None)
    if medvae_4x1:
        medvae_4x1.eval()

    # MedVAE 4×1 fine-tuné
    ckpt_ft = Path(args.medvae_finetuned_ckpt)
    medvae_4x1_ft = load_medvae(
        "medvae_4_1_3d",
        device,
        checkpoint_path=ckpt_ft if ckpt_ft.exists() else None,
    )
    if medvae_4x1_ft:
        medvae_4x1_ft.eval()

    # MedVAE 8×1 frozen (poids HuggingFace — compression 8× par dim, 512× total)
    medvae_8x1 = load_medvae("medvae_8_1_3d", device, checkpoint_path=None)
    if medvae_8x1:
        medvae_8x1.eval()

    loaded = [
        n
        for n, m in [
            ("AEKL", aekl),
            ("VQ-VAE", vqvae_base),
            ("MedVAE 4×1", medvae_4x1),
            ("MedVAE 4×1 ft", medvae_4x1_ft),
            ("MedVAE 8×1", medvae_8x1),
        ]
        if m is not None
    ]
    print(f"✓ Modèles chargés : {loaded}\n")

    # ── Boucle sur les modalités ──────────────────────────────────────────────────
    BAR = "\u2550" * 60  # ═════════════════════════════════════════════════════
    for modality in MODALITIES:
        print(f"\n{BAR}")
        print(f"  Modalité : {modality}")
        print(f"{BAR}")

        slices_dict: dict = {}
        ssim_dict: dict = {}

        # ── Étape 1 : calculer z_best depuis le champ de référence (3T) ─────────
        # Pour les données prospectives recalées, tous les volumes ont le même
        # espace ; on utilise z_best du volume 3T pour assurer la même coupe
        # anatomique dans toutes les colonnes (résoudre le bug 7T z_best=197).
        ref_path = (
            (
                data_root
                / args.split
                / modality
                / REF_FIELD
                / sorted(
                    (data_root / args.split / modality / REF_FIELD).glob("*.nii.gz")
                )[args.subject_idx % 3]
            )
            if (data_root / args.split / modality / REF_FIELD).exists()
            else None
        )

        z_best_ref = -1  # défaut : calculé par volume
        x0_ref = y0_ref = -1

        if ref_path and ref_path.exists():
            import nibabel as _nib

            _img = _nib.load(str(ref_path))
            _vol = _img.get_fdata(dtype=np.float32)
            _H, _W, _ = _vol.shape
            _x0 = min(max((_H - patch_xy) // 2, 0), max(_H - patch_xy, 0))
            _y0 = min(max((_W - patch_xy) // 2, 0), max(_W - patch_xy, 0))
            z_best_ref = _find_z_best(_vol, _x0, _y0, patch_xy)
            x0_ref, y0_ref = _x0, _y0
            print(
                f"  Coupe de référence ({REF_FIELD}) : z_best={z_best_ref}  x0={x0_ref} y0={y0_ref}"
            )

        # Récupère l'ID du premier sujet pour le titre
        sample_dir = data_root / args.split / modality / FIELDS[0]
        sample_files = sorted(sample_dir.glob("*.nii.gz"))
        subject_id = ""
        if sample_files:
            fname = sample_files[args.subject_idx % len(sample_files)].stem
            subject_id = fname.split("_")[-1]

        # ── Étape 2 : traiter chaque champ ──────────────────────────────────────
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

            # Extraction du patch (z_best_ref fixé depuis la référence 3T)
            patch_model, patch_display, z_best, x0, y0 = load_and_patch(
                path, patch_xy, patch_z, z_best_ref=z_best_ref
            )
            z_loc = z_best - max(0, z_best - patch_z // 2)
            # z_loc dans le patch = z_best_ref - z0, clamped dans [0, patch_z-1]
            z_loc = int(np.clip(z_loc, 0, patch_z - 1))
            orig_axial = patch_display[:, :, z_loc]
            slices_dict[("Original", field)] = orig_axial
            print(f"    Patch   : {patch_model.shape}  z_best={z_best}  z_loc={z_loc}")

            def _run(model, key, patch_m=patch_model):
                try:
                    rec = reconstruct(model, patch_m, device)
                    sl = rec[:, :, z_loc]
                    slices_dict[(key, field)] = sl
                    ssim_dict[(key, field)] = _ssim_2d(orig_axial, sl)
                    print(f"    {key:16s} SSIM={ssim_dict[(key, field)]:.4f}")
                except Exception as e:
                    print(f"    {key} error : {e}")
                    import traceback

                    traceback.print_exc()
                    slices_dict[(key, field)] = None

            # AEKL — T1W uniquement
            if modality == "T1W" and aekl is not None:
                _run(aekl, "AEKL")
            else:
                slices_dict[("AEKL", field)] = None

            # VQ-VAE
            if vqvae_base is not None:
                mod_idx = MODALITIES_IDX.get(modality, 0)
                field_idx = FIELDS_IDX.get(field, 0)
                vqvae = VQVAECompatWrapper(
                    vqvae_base, mod_idx=mod_idx, field_idx=field_idx
                ).to(device)
                _run(vqvae, "VQ-VAE")

            # MedVAE 4×1 frozen
            if medvae_4x1 is not None:
                _run(medvae_4x1, "MedVAE 4×1")

            # MedVAE 4×1 fine-tuné
            if medvae_4x1_ft is not None:
                _run(medvae_4x1_ft, "MedVAE 4×1 ft")

            # MedVAE 8×1 frozen
            if medvae_8x1 is not None:
                _run(medvae_8x1, "MedVAE 8×1")

        # ── Génération et sauvegarde de la figure ─────────────────────────────
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
