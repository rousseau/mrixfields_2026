"""
Helpers communs pour la visualisation multi-sujet / multi-champ MRIxFields.

Partagés entre :
  - visualize_stargan2d.py
  - visualize_cfm3d.py
  - (et futurs scripts de figure)
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing as nib_proc
import numpy as np


DEFAULT_DATA_DIR = Path("/home/rousseau/Data/MRIxFields_20260414")


def strip_nifti_ext(name: str) -> str:
    """Retire l'extension .nii.gz ou .nii d'un nom de fichier."""
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def subject_id_from_name(name: str) -> str:
    """
    Extrait le subject_id d'un nom officiel P_{MOD}_{FIELD}_{ID}.nii.gz.
    Gère correctement les champs contenant un point (ex: 0.1T).
    """
    return strip_nifti_ext(name).split("_")[-1]


def load_mid_slice(nifti_path: Path, axis: int = 2) -> np.ndarray:
    """Charge un volume NIfTI et retourne la coupe centrale selon l'axe donné."""
    img = nib.load(str(nifti_path))
    data = img.get_fdata(dtype=np.float32)
    idx = data.shape[axis] // 2
    return np.take(data, idx, axis=axis)


def load_mid_slice_aligned(nifti_path: Path, ref_path: Path, axis: int = 2) -> np.ndarray:
    """Charge un volume NIfTI et le recale sur la grille physique (shape+affine)
    d'un volume de référence avant d'extraire la coupe centrale.

    Nécessaire quand `nifti_path` (ex: prédiction à résolution/FOV réduite,
    comme un crop 128x128x80 @ 1mm) n'a pas le même spacing/shape que
    `ref_path` (ex: ground truth pleine résolution 364x436x364 @ 0.5mm) :
    sans ce recalage, une simple coupe centrale brute compare deux grilles
    de tailles physiques différentes et la figure apparaît désalignée /
    mal mise à l'échelle (zoom apparent).
    """
    img = nib.load(str(nifti_path))
    ref_img = nib.load(str(ref_path))
    if img.shape[:3] != ref_img.shape[:3] or not np.allclose(
        img.affine, ref_img.affine, atol=1e-3
    ):
        img = nib_proc.resample_from_to(img, ref_img, order=1, mode="constant", cval=0.0)
    data = img.get_fdata(dtype=np.float32)
    idx = data.shape[axis] // 2
    return np.take(data, idx, axis=axis)


def normalize(
    arr: np.ndarray,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
) -> np.ndarray:
    """Normalisation robuste [0, 1] par percentile."""
    lo, hi = np.percentile(arr, lo_pct), np.percentile(arr, hi_pct)
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def find_subject_file(
    directory: Path,
    subject_id: str,
    suffix: str = ".nii.gz",
) -> Path | None:
    """
    Trouve un fichier NIfTI correspondant à un subject_id dans un répertoire.
    Accepte aussi bien `P_T1W_7T_0006.nii.gz` que
    `P_T1W_0.1T_0006_T1W_7T_mmfm_unet.nii.gz`.
    """
    if not directory.exists():
        return None
    escaped_suffix = re.escape(suffix)
    pattern = re.compile(rf".*_{re.escape(subject_id)}(_.*|){escaped_suffix}$")
    for f in sorted(directory.glob(f"*{suffix}")):
        if pattern.match(f.name):
            return f
    return None


def list_subjects(
    modality: str,
    source_field: str,
    data_dir: Path | str | None = None,
) -> list[str]:
    """Retourne les IDs de sujets disponibles dans Training_prospective."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    src_dir = data_dir / "Training_prospective" / modality / source_field
    if not src_dir.exists():
        raise FileNotFoundError(f"Répertoire source introuvable : {src_dir}")
    return [subject_id_from_name(f.name) for f in sorted(src_dir.glob("*.nii.gz"))]


def make_multi_field_figure(
    subjects: list[str],
    modality: str,
    source_field: str,
    target_fields: list[str],
    pred_dir: Path,
    out_path: Path,
    data_dir: Path | str | None = None,
    axis: int = 2,
    title: str | None = None,
) -> None:
    """
    Génère une figure grid [sujets × champs] : source + (pred + GT) par cible.

    Args:
        subjects: liste de subject_id (ex: ["0006", "0007", "0009"]).
        modality: T1W / T2W / T2FLAIR.
        source_field: champ source (ex: "0.1T").
        target_fields: champs cibles (ex: ["1.5T", "3T", "5T", "7T"]).
        pred_dir: répertoire contenant les sous-dossiers `<src>_to_<tgt>/`.
        out_path: chemin de sauvegarde de la figure PNG.
        data_dir: racine des données (défaut: DEFAULT_DATA_DIR).
        axis: axe de coupe (0=sagittal, 1=coronal, 2=axial).
        title: titre global (défaut: auto).
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    col_labels = [f"{source_field}\n(input)"]
    for tgt in target_fields:
        col_labels.append(f"{tgt}\n(pred)")
        col_labels.append(f"{tgt}\n(GT)")
    n_cols = len(col_labels)
    n_rows = len(subjects)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.2 * n_cols, 2.5 * n_rows),
        squeeze=False,
    )

    for row_idx, subject_id in enumerate(subjects):
        # ---- Source --------------------------------------------------------
        src_dir = data_dir / "Training_prospective" / modality / source_field
        src_file = find_subject_file(src_dir, subject_id)
        if src_file is not None:
            sl = normalize(load_mid_slice(src_file, axis))
        else:
            sl = np.zeros((64, 64))
        axes[row_idx][0].imshow(sl.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[row_idx][0].set_title(col_labels[0] if row_idx == 0 else "", fontsize=7)
        axes[row_idx][0].set_ylabel(f"Sujet {subject_id}", fontsize=7)
        axes[row_idx][0].axis("off")

        # ---- Pred + GT pour chaque champ cible -----------------------------
        for tgt_idx, tgt_field in enumerate(target_fields):
            col_pred = 1 + 2 * tgt_idx
            col_gt = col_pred + 1

            # Ground truth chargé en premier : sert de référence de recalage
            # physique pour la prédiction, qui peut avoir un spacing/FOV
            # différent (voir load_mid_slice_aligned).
            gt_dir = data_dir / "Training_prospective" / modality / tgt_field
            gt_file = find_subject_file(gt_dir, subject_id)
            if gt_file is not None:
                sl_gt = normalize(load_mid_slice(gt_file, axis))
            else:
                sl_gt = np.zeros((64, 64))
                print(f"  [ATTENTION] GT introuvable : {gt_dir} / sujet {subject_id}")

            # Prédiction — recalée sur la grille du GT pour éviter tout
            # problème d'échelle si la prédiction est à une résolution/FOV
            # différente (ex: sortie patch-based à 1mm vs GT à 0.5mm).
            tgt_pred_dir = pred_dir / f"{source_field}_to_{tgt_field}"
            pred_file = find_subject_file(tgt_pred_dir, subject_id)
            if pred_file is not None:
                if gt_file is not None:
                    sl_pred = normalize(load_mid_slice_aligned(pred_file, gt_file, axis))
                else:
                    sl_pred = normalize(load_mid_slice(pred_file, axis))
            else:
                sl_pred = np.zeros((64, 64))
                print(
                    f"  [ATTENTION] Prédiction introuvable : "
                    f"{tgt_pred_dir} / sujet {subject_id}"
                )

            for col_idx, sl in ((col_pred, sl_pred), (col_gt, sl_gt)):
                axes[row_idx][col_idx].imshow(
                    sl.T, cmap="gray", origin="lower", vmin=0, vmax=1
                )
                if row_idx == 0:
                    axes[row_idx][col_idx].set_title(col_labels[col_idx], fontsize=7)
                axes[row_idx][col_idx].axis("off")

    if title is None:
        title = (
            f"{modality} : {source_field} → {', '.join(target_fields)}\n"
            f"(Training_prospective, coupe centrale axe {axis})"
        )
    fig.suptitle(title, fontsize=9, y=1.01)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {out_path}")
