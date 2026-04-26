#!/usr/bin/env python3
"""
Visualisation des résultats OT-CFM 2D — MRIxFields 2026

Génère une figure de comparaison multi-sujet / multi-champ pour une modalité :
  - Lignes : sujets (Training_prospective)
  - Colonnes : 0.1T input | 1.5T pred | 1.5T GT | 3T pred | 3T GT | 5T pred | 5T GT | 7T pred | 7T GT

Utilisation :
    python src/visualize_cfm2d.py --checkpoint outputs/cfm2d/runs/cfm2d_T1W/weights/checkpoint_5000.pth \\
                                   --config configs/cfm_T1W.yaml

Prérequis :
    - Si --checkpoint fourni : l'inférence est lancée à la volée sur les sujets prospectifs
    - Sinon : les prédictions doivent être dans $EXP_DIR/cfm2d/predictions/<modality>/0.1T_to_<field>/
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
# Chemins — valeurs par défaut (machine locale), surchargées via --env
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/rousseau/Data/MRIxFields_20260414")
EXP_DIR = Path("/home/rousseau/Exp/mrixfields_2026/outputs/cfm2d")
RESULTS_DIR = Path("/home/rousseau/Exp/mrixfields_2026/results")

TARGET_FIELDS = ["1.5T", "3T", "5T", "7T"]
SOURCE_FIELD = "0.1T"

PYTHON = "/home/rousseau/miniforge3/bin/python"
TRAIN_SCRIPT = Path("/home/rousseau/Exp/mrixfields_2026/src/train_cfm2d.py")


def _setup_paths(env_arg: str | None) -> None:
    """Charge l'env YAML et met à jour les chemins globaux."""
    global DATA_DIR, EXP_DIR, RESULTS_DIR, PYTHON, TRAIN_SCRIPT
    if env_arg is None:
        return
    env_path = env_arg if env_arg.endswith(".yaml") else f"configs/env/{env_arg}.yaml"
    if not os.path.isabs(env_path):
        if os.path.exists(env_path):
            env_path = os.path.abspath(env_path)
        else:
            project_root = Path(__file__).parent.parent
            candidate = project_root / env_path
            if candidate.exists():
                env_path = str(candidate)
    with open(env_path) as f:
        raw = yaml.safe_load(f)
    env = {k: Path(os.path.expandvars(str(v))) for k, v in raw.items()}
    output_root = env["output_root"]
    project_root = env["project_root"]
    DATA_DIR = env["data_root"]
    EXP_DIR = output_root / "cfm2d"
    RESULTS_DIR = project_root / "results"
    PYTHON = str(env.get("python", Path("python3")))
    TRAIN_SCRIPT = project_root / "src" / "train_cfm2d.py"


# ---------------------------------------------------------------------------
# Inférence à la volée depuis un checkpoint
# ---------------------------------------------------------------------------

def run_inference_for_checkpoint(
    checkpoint: Path,
    config: Path,
    modality: str,
    subjects: list,
    pred_dir: Path,
    env_arg: str | None = None,
) -> None:
    """Lance train_cfm2d.py --mode infer pour chaque champ cible."""
    input_dir = DATA_DIR / "Training_prospective" / modality / SOURCE_FIELD

    for tgt in TARGET_FIELDS:
        out_dir = pred_dir / f"{SOURCE_FIELD}_to_{tgt}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Skip si déjà calculé (tous les sujets présents)
        already = all(
            _find_subject_file(out_dir, sid) is not None for sid in subjects
        )
        if already:
            print(f"  Inférence {SOURCE_FIELD}→{tgt} déjà présente, skip.")
            continue

        print(f"  Inférence {SOURCE_FIELD}→{tgt} …")
        cmd = [
            PYTHON,
            str(TRAIN_SCRIPT),
            "--mode", "infer",
            "--config", str(config),
            "--checkpoint", str(checkpoint),
            "--input_dir", str(input_dir),
            "--output_dir", str(out_dir),
            "--source_field", SOURCE_FIELD,
            "--target_field", tgt,
        ]
        if env_arg:
            cmd += ["--env", env_arg]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [ERREUR] inférence {tgt} :\n{result.stderr[-1000:]}")
        else:
            print(f"  OK → {out_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_mid_slice(nifti_path: Path, axis: int = 2) -> np.ndarray:
    """Charge un volume NIfTI et retourne la coupe centrale selon l'axe."""
    img = nib.load(str(nifti_path))
    data = img.get_fdata(dtype=np.float32)
    idx = data.shape[axis] // 2
    return np.take(data, idx, axis=axis)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalisation robuste [0, 1] par percentile 1–99."""
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _find_subject_file(directory: Path, subject_id: str) -> Path | None:
    """Trouve le NIfTI d'un sujet dans un répertoire."""
    if not directory.exists():
        return None
    pattern = re.compile(rf".*_{subject_id}\.nii\.gz$")
    for f in sorted(directory.glob("*.nii.gz")):
        if pattern.match(f.name):
            return f
    return None


def _list_subjects(modality: str) -> list:
    """Retourne les IDs de sujets disponibles dans Training_prospective."""
    src_dir = DATA_DIR / "Training_prospective" / modality / SOURCE_FIELD
    if not src_dir.exists():
        raise FileNotFoundError(f"Répertoire source introuvable : {src_dir}")
    subjects = []
    for f in sorted(src_dir.glob("*.nii.gz")):
        parts = f.stem.replace(".nii", "").split("_")
        if parts:
            subjects.append(parts[-1])
    return subjects


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(
    modality: str,
    subjects: list,
    axis: int,
    out_path: Path,
    pred_dir: Path | None = None,
) -> None:
    """Génère la grille de comparaison.

    pred_dir : répertoire contenant les sous-dossiers <src>_to_<tgt>/.
    Si None, utilise EXP_DIR/predictions/<modality>/.
    """
    if pred_dir is None:
        pred_dir = EXP_DIR / "predictions" / modality

    col_labels = [f"{SOURCE_FIELD}\n(input)"]
    for tgt in TARGET_FIELDS:
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
        # Source (0.1T input)
        src_dir = DATA_DIR / "Training_prospective" / modality / SOURCE_FIELD
        src_file = _find_subject_file(src_dir, subject_id)
        sl_src = _normalize(_load_mid_slice(src_file, axis)) if src_file else np.zeros((64, 64))
        axes[row_idx][0].imshow(sl_src.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[row_idx][0].set_title(col_labels[0] if row_idx == 0 else "", fontsize=7)
        axes[row_idx][0].set_ylabel(f"Sujet {subject_id}", fontsize=7)
        axes[row_idx][0].axis("off")

        for tgt_idx, tgt_field in enumerate(TARGET_FIELDS):
            col_pred = 1 + 2 * tgt_idx
            col_gt = col_pred + 1

            # Prédiction
            tgt_pred_dir = pred_dir / f"{SOURCE_FIELD}_to_{tgt_field}"
            pred_file = _find_subject_file(tgt_pred_dir, subject_id)
            if pred_file is not None:
                sl_pred = _normalize(_load_mid_slice(pred_file, axis))
            else:
                sl_pred = np.zeros((64, 64))
                print(f"  [ATTENTION] Prédiction manquante : {tgt_pred_dir} / {subject_id}")

            # Ground truth
            gt_dir = DATA_DIR / "Training_prospective" / modality / tgt_field
            gt_file = _find_subject_file(gt_dir, subject_id)
            sl_gt = _normalize(_load_mid_slice(gt_file, axis)) if gt_file else np.zeros((64, 64))

            for col_idx, sl in ((col_pred, sl_pred), (col_gt, sl_gt)):
                axes[row_idx][col_idx].imshow(sl.T, cmap="gray", origin="lower", vmin=0, vmax=1)
                if row_idx == 0:
                    axes[row_idx][col_idx].set_title(col_labels[col_idx], fontsize=7)
                axes[row_idx][col_idx].axis("off")

    fig.suptitle(
        f"OT-CFM 2D — {modality} : 0.1T → champs cibles\n"
        f"(Training_prospective, coupe centrale axe {axis})",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualisation OT-CFM 2D")
    parser.add_argument("--modality", default="T1W", choices=["T1W", "T2W", "T2FLAIR"])
    parser.add_argument("--axis", type=int, default=2,
                        help="Axe de coupe (0=sagittal, 1=coronal, 2=axial)")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="IDs de sujets (défaut : tous dans Training_prospective)")
    parser.add_argument("--out", default=None, help="Chemin de sauvegarde de la figure")
    parser.add_argument("--checkpoint", default=None,
                        help="Chemin vers un checkpoint (.pth) — lance l'inférence à la volée")
    parser.add_argument("--config", default=None,
                        help="Config YAML à utiliser avec --checkpoint")
    parser.add_argument(
        "--env", default=None,
        help="Env config : 'local', 'jeanzay', ou chemin vers configs/env/*.yaml",
    )
    args = parser.parse_args()

    # Résoudre les chemins selon l'environnement
    _setup_paths(args.env)

    if args.subjects is None:
        args.subjects = _list_subjects(args.modality)
    print(f"Sujets : {args.subjects}")

    pred_dir = None

    if args.checkpoint is not None:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            print(f"[ERREUR] Checkpoint introuvable : {ckpt}", file=sys.stderr)
            sys.exit(1)

        step_tag = ckpt.stem  # ex: "checkpoint_5000" ou "model_final"

        # Config : argument ou auto-détecté depuis project_root
        if args.config:
            config = Path(args.config)
        else:
            project_root = TRAIN_SCRIPT.parent.parent
            config = project_root / "configs" / f"cfm_{args.modality}.yaml"
        if not config.exists():
            print(f"[ERREUR] Config introuvable : {config}", file=sys.stderr)
            sys.exit(1)

        pred_dir = EXP_DIR / "predictions_ckpt" / args.modality / step_tag
        run_inference_for_checkpoint(
            ckpt, config, args.modality, args.subjects, pred_dir, env_arg=args.env
        )

        if args.out is None:
            args.out = str(RESULTS_DIR / f"cfm2d_{args.modality.lower()}_{step_tag}.png")

    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / f"cfm2d_{args.modality.lower()}.png"
    )

    make_figure(
        modality=args.modality,
        subjects=args.subjects,
        axis=args.axis,
        out_path=out_path,
        pred_dir=pred_dir,
    )


if __name__ == "__main__":
    main()
