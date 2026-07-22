"""
Visualisation des résultats StarGAN v2 2D — MRIxFields 2026

Génère une figure de comparaison multi-sujet / multi-champ pour la modalité T1W :
  - Lignes : sujets (Training_prospective, ex: 0006, 0007, 0009)
  - Colonnes : 0.1T input | 1.5T pred | 1.5T GT | 3T pred | 3T GT | ...

Utilisation :
    python src/visualization/visualize_stargan2d.py
    python src/visualization/visualize_stargan2d.py --checkpoint /path/to/checkpoint_10000.pth

Prérequis :
    - Si --checkpoint fourni : l'inférence est lancée à la volée sur les 3 sujets
    - Sinon : les prédictions doivent être dans $EXP_DIR/predictions/<modality>/<src>_to_<tgt>/
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Make src/ importable when running from any directory
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

from visualization.common import (
    DEFAULT_DATA_DIR,
    find_subject_file,
    list_subjects,
    make_multi_field_figure,
)

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
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
    input_dir = DEFAULT_DATA_DIR / "Training_prospective" / modality / SOURCE_FIELD
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
            "DATA_DIR": str(DEFAULT_DATA_DIR),
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
# CLI
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
        args.subjects = list_subjects(args.modality, SOURCE_FIELD)
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

    if pred_dir is None:
        pred_dir = EXP_DIR / "predictions" / args.modality

    make_multi_field_figure(
        subjects=args.subjects,
        modality=args.modality,
        source_field=SOURCE_FIELD,
        target_fields=TARGET_FIELDS,
        pred_dir=pred_dir,
        out_path=out_path,
        axis=args.axis,
        title=(
            f"StarGAN v2 2D — {args.modality} : {SOURCE_FIELD} → "
            f"{', '.join(TARGET_FIELDS)}\n"
            f"(Training_prospective, coupe centrale axe {args.axis})"
        ),
    )


if __name__ == "__main__":
    main()
