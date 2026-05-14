#!/usr/bin/env python3
"""
Visualisation — StarGAN v2 2D — MRIxFields 2026

Génère une figure de comparaison multi-sujet / multi-champ :
  - Lignes  : sujets (Training_prospective)
  - Colonnes: 0.1T input | (pred + GT) × 4 champs cibles

Usage :
    python src/visualization/visualize.py --method stargan2d --modality T1W \\
        --checkpoint outputs/stargan2d/runs/task3_any_to_any_T1W/weights/model_final.pth

    # Depuis prédictions déjà calculées (sans re-inférer)
    python src/visualization/visualize.py --method stargan2d --modality T1W \\
        --pred-dir outputs/stargan2d/predictions_ckpt/T1W/checkpoint_150000
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Chemins par défaut (machine locale) — surchargés via --env
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/rousseau/Data/MRIxFields_20260414")
RESULTS_DIR = Path("/home/rousseau/Exp/mrixfields_2026/results")
PROJECT_DIR = Path("/home/rousseau/Exp/mrixfields_2026")

TARGET_FIELDS = ["1.5T", "3T", "5T", "7T"]
SOURCE_FIELD = "0.1T"

# StarGAN (challenge baseline)
PYTHON = "/home/rousseau/miniforge3/bin/python"
CHALLENGE_DIR = Path("/home/rousseau/Code/MRIxFields2026/Baseline")


def _setup_paths(env_arg: str | None) -> None:
    """Charge l'env YAML et met à jour les chemins globaux."""
    global DATA_DIR, RESULTS_DIR, PROJECT_DIR, PYTHON, CHALLENGE_DIR
    if env_arg is None:
        return
    env_path = env_arg if env_arg.endswith(".yaml") else f"configs/env/{env_arg}.yaml"
    if not os.path.isabs(env_path):
        candidate = PROJECT_DIR / env_path
        env_path = str(candidate) if candidate.exists() else env_path
    with open(env_path) as f:
        raw = yaml.safe_load(f)
    env = {k: Path(os.path.expandvars(str(v))) for k, v in raw.items()}
    DATA_DIR = env["data_root"]
    RESULTS_DIR = env["project_root"] / "results"
    PROJECT_DIR = env["project_root"]
    PYTHON = str(env.get("python", Path("python3")))
    CHALLENGE_DIR = env.get("challenge_dir", CHALLENGE_DIR)


# ---------------------------------------------------------------------------
# Helpers partagés
# ---------------------------------------------------------------------------


def _load_mid_slice(nifti_path: Path, axis: int = 2) -> np.ndarray:
    img = nib.load(str(nifti_path))
    data = img.get_fdata(dtype=np.float32)
    return np.take(data, data.shape[axis] // 2, axis=axis)


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _find_subject_file(directory: Path, subject_id: str) -> Path | None:
    if not directory.exists():
        return None
    pattern = re.compile(rf".*_{subject_id}\.nii\.gz$")
    for f in sorted(directory.glob("*.nii.gz")):
        if pattern.match(f.name):
            return f
    return None


def _list_subjects(modality: str) -> list:
    src_dir = DATA_DIR / "Training_prospective" / modality / SOURCE_FIELD
    if not src_dir.exists():
        raise FileNotFoundError(f"Répertoire source introuvable : {src_dir}")
    subjects = []
    for f in sorted(src_dir.glob("*.nii.gz")):
        parts = f.stem.replace(".nii", "").split("_")
        if parts:
            subjects.append(parts[-1])
    return subjects


def _slices_from_dir(pred_dir: Path, subjects: list, axis: int) -> dict:
    """Construit un pred_slices dict depuis un répertoire de .nii.gz.

    pred_dir doit contenir des sous-dossiers 0.1T_to_<field>/.
    """
    result: dict = {}
    for sid in subjects:
        result[sid] = {}
        for tgt in TARGET_FIELDS:
            tgt_dir = pred_dir / f"{SOURCE_FIELD}_to_{tgt}"
            nii_file = _find_subject_file(tgt_dir, sid)
            if nii_file is not None:
                result[sid][tgt] = _load_mid_slice(nii_file, axis)
            else:
                print(f"  [ATTENTION] Prédiction manquante : {tgt_dir} / {sid}")
    return result


# ---------------------------------------------------------------------------
# Inférence StarGAN — coupe centrale uniquement
# ---------------------------------------------------------------------------


def _infer_stargan_middle_slices(
    checkpoint: Path,
    config: Path,
    modality: str,
    subjects: list,
    axis: int,
) -> dict:
    """Lance inference.py du challenge sur les sujets, sauvegarde les .nii.gz,
    puis charge uniquement la coupe centrale de chacun dans un dict.
    """
    pred_dir = (
        PROJECT_DIR
        / "outputs"
        / "stargan2d"
        / "predictions_ckpt"
        / modality
        / checkpoint.stem
    )

    env_os = {
        **os.environ,
        "OUTPUT_DIR": str(PROJECT_DIR / "outputs" / "stargan2d" / "runs"),
        "DATA_DIR": str(DATA_DIR),
        "PREPROCESSED_DIR": str(PROJECT_DIR / "outputs" / "stargan2d" / "preprocessed"),
    }

    for tgt in TARGET_FIELDS:
        out_dir = pred_dir / f"{SOURCE_FIELD}_to_{tgt}"
        out_dir.mkdir(parents=True, exist_ok=True)
        already = all(_find_subject_file(out_dir, sid) is not None for sid in subjects)
        if already:
            print(f"  StarGAN {SOURCE_FIELD}→{tgt} déjà présent, skip.")
            continue
        print(f"  StarGAN {SOURCE_FIELD}→{tgt} …", flush=True)
        cmd = [
            PYTHON,
            str(CHALLENGE_DIR / "scripts" / "inference.py"),
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--input_dir",
            str(DATA_DIR / "Training_prospective" / modality / SOURCE_FIELD),
            "--output_dir",
            str(out_dir),
            "--target_field",
            tgt,
        ]
        res = subprocess.run(cmd, env=env_os, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  [ERREUR] StarGAN {tgt}:\n{res.stderr[-1000:]}")
        else:
            print(f"  OK → {out_dir}")

    return _slices_from_dir(pred_dir, subjects, axis)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(
    method_label: str,
    modality: str,
    subjects: list,
    axis: int,
    out_path: Path,
    pred_slices: dict,
) -> None:
    """Génère la grille de comparaison depuis un dict de coupes pré-calculées.

    pred_slices : {subject_id: {target_field: np.ndarray}}
    Colonnes : 0.1T input | (pred + GT) × 4 champs cibles
    """
    col_labels = [f"{SOURCE_FIELD}\n(input)"]
    for tgt in TARGET_FIELDS:
        col_labels.append(f"{tgt}\n(pred)")
        col_labels.append(f"{tgt}\n(GT)")
    n_cols = len(col_labels)
    n_rows = len(subjects)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.2 * n_cols, 2.5 * n_rows),
        squeeze=False,
    )

    src_base = DATA_DIR / "Training_prospective" / modality

    for row_idx, sid in enumerate(subjects):
        # ---- Source 0.1T ------------------------------------------------
        src_file = _find_subject_file(src_base / SOURCE_FIELD, sid)
        sl_src = (
            _normalize(_load_mid_slice(src_file, axis))
            if src_file
            else np.zeros((64, 64))
        )
        ax = axes[row_idx][0]
        ax.imshow(sl_src.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.set_title(col_labels[0] if row_idx == 0 else "", fontsize=7)
        ax.set_ylabel(f"Sujet {sid}", fontsize=7)
        ax.axis("off")

        # ---- Pred + GT par champ cible ----------------------------------
        for tgt_idx, tgt_field in enumerate(TARGET_FIELDS):
            col_pred = 1 + 2 * tgt_idx
            col_gt = col_pred + 1

            arr = pred_slices.get(sid, {}).get(tgt_field)
            if arr is not None:
                sl_pred = _normalize(arr)
            else:
                sl_pred = np.zeros((64, 64))
                print(f"  [ATTENTION] Prédiction manquante : {sid}/{tgt_field}")

            gt_file = _find_subject_file(src_base / tgt_field, sid)
            sl_gt = (
                _normalize(_load_mid_slice(gt_file, axis))
                if gt_file
                else np.zeros((64, 64))
            )

            for col_idx, sl in ((col_pred, sl_pred), (col_gt, sl_gt)):
                ax = axes[row_idx][col_idx]
                ax.imshow(sl.T, cmap="gray", origin="lower", vmin=0, vmax=1)
                if row_idx == 0:
                    ax.set_title(col_labels[col_idx], fontsize=7)
                ax.axis("off")

    fig.suptitle(
        f"{method_label} — {modality} : {SOURCE_FIELD} → champs cibles\n"
        f"(Training_prospective, coupe centrale axe {axis})",
        fontsize=9,
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_METHOD_LABELS = {
    "stargan2d": "StarGAN v2 2D",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualisation unifiée MRIxFields 2026",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method", required=True, choices=["stargan2d"], help="Méthode à visualiser"
    )
    parser.add_argument("--modality", default="T1W", choices=["T1W", "T2W", "T2FLAIR"])
    parser.add_argument(
        "--axis",
        type=int,
        default=2,
        help="Axe de coupe (0=sagittal, 1=coronal, 2=axial)",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="IDs de sujets (défaut : tous dans Training_prospective)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint .pth — déclenche l'inférence à la volée",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config YAML StarGAN (optionnel, déduit depuis le challenge dir)",
    )
    parser.add_argument(
        "--pred-dir",
        default=None,
        help="Répertoire de prédictions existantes (skip l'inférence)",
    )
    parser.add_argument("--out", default=None, help="Chemin de sauvegarde de la figure")
    parser.add_argument(
        "--env", default=None, help="Env config : 'local', 'jeanzay', ou chemin .yaml"
    )
    args = parser.parse_args()

    _setup_paths(args.env)

    if args.subjects is None:
        args.subjects = _list_subjects(args.modality)
    print(f"Méthode  : {_METHOD_LABELS[args.method]}")
    print(f"Modalité : {args.modality}")
    print(f"Sujets   : {args.subjects}")

    # ---- Construction de pred_slices ------------------------------------
    if args.pred_dir is not None:
        pred_slices = _slices_from_dir(Path(args.pred_dir), args.subjects, args.axis)
        step_tag = Path(args.pred_dir).name

    elif args.checkpoint is not None:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            print(f"[ERREUR] Checkpoint introuvable : {ckpt}", file=sys.stderr)
            sys.exit(1)
        step_tag = ckpt.stem
        config = (
            Path(args.config)
            if args.config
            else (
                CHALLENGE_DIR
                / "configs"
                / "task3"
                / "stargan"
                / f"any_to_any_{args.modality}.yaml"
            )
        )
        if not config.exists():
            print(f"[ERREUR] Config introuvable : {config}", file=sys.stderr)
            sys.exit(1)
        pred_slices = _infer_stargan_middle_slices(
            ckpt,
            config,
            args.modality,
            args.subjects,
            args.axis,
        )
    else:
        print("[ERREUR] Fournir --checkpoint ou --pred-dir.", file=sys.stderr)
        sys.exit(1)

    # ---- Figure ---------------------------------------------------------
    if args.out is None:
        args.out = str(
            RESULTS_DIR / "qc" / f"{args.method}_{args.modality.lower()}_{step_tag}.png"
        )
    make_figure(
        method_label=_METHOD_LABELS[args.method],
        modality=args.modality,
        subjects=args.subjects,
        axis=args.axis,
        out_path=Path(args.out),
        pred_slices=pred_slices,
    )


if __name__ == "__main__":
    main()
