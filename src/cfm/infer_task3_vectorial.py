"""
Inférence Task 3 complète (any-to-any) pour MMFM vectoriel.

Génère les prédictions pour toutes les paires source→cible sur les sujets
Training_prospective, dans la structure attendue par src/evaluation/evaluate.py :
    outputs/predictions/<method>/task3/<modality>/<src>_to_<tgt>/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cfm.train_mmfm_3d import infer

DOMAINS = ["0.1T", "1.5T", "3T", "5T", "7T"]


def main():
    parser = argparse.ArgumentParser(description="Inférence Task 3 MMFM vectoriel")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--modality", default="T1W")
    parser.add_argument("--method", default="mmfm",
                        help="Nom de la méthode pour le répertoire de sortie")
    parser.add_argument("--env", default="local")
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    out_root = Path("outputs/predictions") / args.method / "task3" / args.modality

    for src in DOMAINS:
        for tgt in DOMAINS:
            if src == tgt:
                continue
            out_dir = out_root / f"{src}_to_{tgt}"
            if out_dir.exists() and any(out_dir.glob("*.nii*")):
                print(f"Skip {src}→{tgt} (déjà présent)")
                continue
            print(f"\n=== {src} → {tgt} ===")
            infer(
                cfg_path=args.config,
                checkpoint=args.checkpoint,
                output_dir=str(out_dir),
                source_field=src,
                source_modality=args.modality,
                target_field=tgt,
                target_modality=args.modality,
                env_path=args.env,
                input_dir=str(
                    Path("/home/rousseau/Data/MRIxFields_20260414")
                    / "Training_prospective" / args.modality / src
                ),
                input_volume=None,
                n_steps=args.n_steps,
                use_ema=not args.no_ema,
            )

    print(f"\nToutes les prédictions sont dans : {out_root}")


if __name__ == "__main__":
    main()
