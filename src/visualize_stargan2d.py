"""
Visualisation des résultats StarGAN v2 2D — MRIxFields 2026

Génère une figure de comparaison multi-sujet / multi-champ pour la modalité T1W :
  - Lignes : sujets (Training_prospective, ex: 0006, 0007, 0009)
  - Colonnes : 0.1T input | 1.5T pred | 1.5T GT | 3T pred | 3T GT | 5T pred | 5T GT | 7T pred | 7T GT

Utilisation :
    /home/rousseau/miniforge3/bin/python visualize_stargan2d.py [--modality T1W] [--axis 2] [--subjects 0006 0007 0009]
    /home/rousseau/miniforge3/bin/python visualize_stargan2d.py --checkpoint /path/to/checkpoint_10000.pth

Prérequis :
    - Si --checkpoint fourni : l'inférence est lancée à la volée sur les 3 sujets
    - Sinon : les prédictions doivent être dans $EXP_DIR/predictions/T1W/0.1T_to_<field>/
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/rousseau/Data/MRIxFields_20260414")
EXP_DIR = Path("/home/rousseau/Exp/mrixfields_2026/outputs/stargan2d")
RESULTS_DIR = Path("/home/rousseau/Exp/mrixfields_2026/results")

TARGET_FIELDS = ["1.5T", "3T", "5T", "7T"]
SOURCE_FIELD = "0.1T"

PYTHON = "/home/rousseau/miniforge3/bin/python"
CHALLENGE_DIR = Path("/home/rousseau/Code/MRIxFields2026/Baseline")


# ---------------------------------------------------------------------------
# Inférence à la volée depuis un checkpoint intermédiaire
# ---------------------------------------------------------------------------

def run_inference_for_checkpoint(
    checkpoint: Path,
    config: Path,
    modality: str,
    subjects: list[str],
    pred_dir: Path,
) -> None:
    """Lance inference.py pour chaque champ cible sur les sujets donnés."""
    input_dir = DATA_DIR / "Training_prospective" / modality / SOURCE_FIELD
    for tgt in TARGET_FIELDS:
        out_dir = pred_dir / f"{SOURCE_FIELD}_to_{tgt}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Vérifie si déjà fait (tous les sujets présents)
        already = all(
            find_subject_file(out_dir, sid) is not None for sid in subjects
        )
        if already:
            print(f"  Inférence {SOURCE_FIELD}→{tgt} déjà présente, skip.")
            continue
        print(f"  Inférence {SOURCE_FIELD}→{tgt} …")
        env = {
            **__import__("os").environ,
            "OUTPUT_DIR": str(EXP_DIR / "runs"),
            "DATA_DIR": str(DATA_DIR),
            "PREPROCESSED_DIR": str(EXP_DIR / "preprocessed"),
        }
        cmd = [
            PYTHON,
            str(CHALLENGE_DIR / "scripts" / "inference.py"),
            "--config", str(config),
            "--checkpoint", str(checkpoint),
            "--input_dir", str(input_dir),
            "--output_dir", str(out_dir),
            "--target_field", tgt,
        ]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [ERREUR] inference.py pour {tgt}:\n{result.stderr[-1000:]}")
        else:
            print(f"  OK → {out_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_mid_slice(nifti_path: Path, axis: int = 2) -> np.ndarray:
    """Charge un volume NIfTI et retourne la coupe centrale selon l'axe donné."""
    img = nib.load(str(nifti_path))
    data = img.get_fdata(dtype=np.float32)
    idx = data.shape[axis] // 2
    sl = np.take(data, idx, axis=axis)
    return sl


def normalize(arr: np.ndarray) -> np.ndarray:
    """Normalisation [0, 1] robuste (percentile 1–99)."""
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def find_subject_file(directory: Path, subject_id: str) -> Path | None:
    """Trouve le fichier NIfTI correspondant à un sujet dans un répertoire."""
    if not directory.exists():
        return None
    pattern = re.compile(rf".*_{subject_id}\.nii\.gz$")
    for f in sorted(directory.glob("*.nii.gz")):
        if pattern.match(f.name):
            return f
    return None


def list_subjects(modality: str) -> list[str]:
    """Retourne les IDs de sujets disponibles dans Training_prospective."""
    src_dir = DATA_DIR / "Training_prospective" / modality / SOURCE_FIELD
    if not src_dir.exists():
        raise FileNotFoundError(f"Répertoire source introuvable : {src_dir}")
    subjects = []
    for f in sorted(src_dir.glob("*.nii.gz")):
        # Format : P_T1W_0.1T_XXXX.nii.gz
        parts = f.stem.replace(".nii", "").split("_")
        if parts:
            subjects.append(parts[-1])
    return subjects


# ---------------------------------------------------------------------------
# Génération de la figure
# ---------------------------------------------------------------------------

def make_figure(
    modality: str,
    subjects: list[str],
    axis: int,
    out_path: Path,
    pred_dir: Path | None = None,
) -> None:
    """pred_dir : répertoire contenant les sous-dossiers <src>_to_<tgt>/.
    Si None, utilise le répertoire par défaut EXP_DIR/predictions/<modality>/."""
    if pred_dir is None:
        pred_dir = EXP_DIR / "predictions" / modality
    # Colonnes : source + (pred + GT) × N_targets
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
        # ---- Source (0.1T input) ----------------------------------------
        src_dir = DATA_DIR / "Training_prospective" / modality / SOURCE_FIELD
        src_file = find_subject_file(src_dir, subject_id)
        if src_file is not None:
            sl = normalize(load_mid_slice(src_file, axis))
        else:
            sl = np.zeros((64, 64))
        axes[row_idx][0].imshow(sl.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[row_idx][0].set_title(col_labels[0] if row_idx == 0 else "", fontsize=7)
        axes[row_idx][0].set_ylabel(f"Sujet {subject_id}", fontsize=7)
        axes[row_idx][0].axis("off")

        # ---- Pred + GT pour chaque champ cible --------------------------
        for tgt_idx, tgt_field in enumerate(TARGET_FIELDS):
            col_pred = 1 + 2 * tgt_idx
            col_gt = col_pred + 1

            # Prédiction
            tgt_pred_dir = pred_dir / f"{SOURCE_FIELD}_to_{tgt_field}"
            pred_file = find_subject_file(tgt_pred_dir, subject_id)
            if pred_file is not None:
                sl_pred = normalize(load_mid_slice(pred_file, axis))
            else:
                sl_pred = np.zeros((64, 64))
                print(f"  [ATTENTION] Prédiction introuvable : {tgt_pred_dir} / sujet {subject_id}")

            # Ground truth
            gt_dir = DATA_DIR / "Training_prospective" / modality / tgt_field
            gt_file = find_subject_file(gt_dir, subject_id)
            if gt_file is not None:
                sl_gt = normalize(load_mid_slice(gt_file, axis))
            else:
                sl_gt = np.zeros((64, 64))
                print(f"  [ATTENTION] GT introuvable : {gt_dir} / sujet {subject_id}")

            for col_idx, sl in ((col_pred, sl_pred), (col_gt, sl_gt)):
                axes[row_idx][col_idx].imshow(sl.T, cmap="gray", origin="lower", vmin=0, vmax=1)
                if row_idx == 0:
                    axes[row_idx][col_idx].set_title(col_labels[col_idx], fontsize=7)
                axes[row_idx][col_idx].axis("off")

    fig.suptitle(
        f"StarGAN v2 2D — {modality} : 0.1T → champs cibles\n"
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

def main():
    parser = argparse.ArgumentParser(description="Visualisation StarGAN v2 2D")
    parser.add_argument("--modality", default="T1W", choices=["T1W", "T2W", "T2FLAIR"])
    parser.add_argument("--axis", type=int, default=2,
                        help="Axe de coupe (0=sagittal, 1=coronal, 2=axial)")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="IDs de sujets (défaut : tous dans Training_prospective)")
    parser.add_argument("--out", default=None,
                        help="Chemin de sauvegarde de la figure")
    parser.add_argument("--checkpoint", default=None,
                        help="Chemin vers un checkpoint intermédiaire (.pth) "
                             "— déclenche l'inférence à la volée")
    parser.add_argument("--config", default=None,
                        help="Config YAML à utiliser avec --checkpoint "
                             "(défaut : configs/task3/stargan/any_to_any_<modality>.yaml)")
    args = parser.parse_args()

    if args.subjects is None:
        args.subjects = list_subjects(args.modality)
    print(f"Sujets : {args.subjects}")

    pred_dir = None  # utilise le défaut (predictions/<modality>/)

    if args.checkpoint is not None:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            print(f"[ERREUR] Checkpoint introuvable : {ckpt}", file=sys.stderr)
            sys.exit(1)

        # Nom du step pour nommer les sous-dossiers et la figure
        step_tag = ckpt.stem  # ex: "checkpoint_10000" ou "model_final"

        config = Path(args.config) if args.config else (
            CHALLENGE_DIR / "configs" / "task3" / "stargan" / f"any_to_any_{args.modality}.yaml"
        )
        if not config.exists():
            print(f"[ERREUR] Config introuvable : {config}", file=sys.stderr)
            sys.exit(1)

        # Prédictions stockées dans un sous-dossier par checkpoint
        pred_dir = EXP_DIR / "predictions_ckpt" / args.modality / step_tag
        run_inference_for_checkpoint(ckpt, config, args.modality, args.subjects, pred_dir)

        if args.out is None:
            args.out = str(RESULTS_DIR / f"stargan2d_{args.modality.lower()}_{step_tag}.png")

    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / f"stargan2d_{args.modality.lower()}.png"
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
