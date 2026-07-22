"""
Visualisation des résultats CFM / MMFM 3D — MRIxFields 2026

Génère une figure de comparaison multi-sujet / multi-champ :
  - Lignes : sujets (Training_prospective, ex: 0006, 0007, 0009)
  - Colonnes : <src> input | <tgt1> pred | <tgt1> GT | <tgt2> pred | <tgt2> GT | ...

Le script lance l'inférence CFM/MMFM pour chaque champ cible, sauvegarde les
prédictions sous `<pred_dir>/<src>_to_<tgt>/`, puis appelle la figure grid
partagée avec visualize_stargan2d.py.

Utilisation (MMFM-UNet V2) :
    python src/visualization/visualize_cfm3d.py \
        --config configs/mmfm3d_unet_v2_medvae_multimodal.yaml \
        --checkpoint outputs/cfm3d/runs/mmfm3d_unet_v2_medvae_multimodal/weights/checkpoint_20000.pth \
        --modality T1W \
        --source-field 0.1T \
        --target-fields 1.5T 3T 5T 7T
"""

import argparse
import sys
from pathlib import Path

import yaml

# Make src/ importable when running from any directory
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

# Make scripts/ importable (pipeline d'inférence full-résolution MMFM-UNet)
_SCRIPTS_DIR = _SRC.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from visualization.common import (
    DEFAULT_DATA_DIR,
    list_subjects,
    make_multi_field_figure,
    subject_id_from_name,
)

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("/home/rousseau/Exp/mrixfields_2026/results")
DEFAULT_PRED_DIR = Path("/home/rousseau/Exp/mrixfields_2026/outputs/cfm3d/predictions_viz")

TARGET_FIELDS = ["1.5T", "3T", "5T", "7T"]
SOURCE_FIELD = "0.1T"
SOURCE_MODALITY = "T1W"


# ---------------------------------------------------------------------------
# Inférence multi-champ
# ---------------------------------------------------------------------------


def _dispatch_infer(method: str):
    """Retourne la fonction infer adaptée à la méthode du YAML."""
    if method == "cfm3d":
        from cfm.train_cfm_3d import infer as _infer
        return _infer
    if method in ("mmfm3d", "mmfm", "mmfm3d_vectorized", "mmfm3d_vectorized_v1"):
        from cfm.train_mmfm_3d import infer as _infer
        return _infer
    if method in ("mmfm3d_unet_v2", "mmfm3d_unet"):
        # Pipeline pleine résolution (patches glissants + blending), qui
        # remplace l'ancien infer() bas-résolution (crop centré unique
        # 128x128x80 @ 1mm) : ce dernier produisait un mismatch d'échelle/FOV
        # avec le GT pleine résolution, aussi bien en visualisation qu'en
        # évaluation quantitative. Utilisé pour v1 (dont le fine-tune) et v2.
        from infer_mmfm_unet_v2_batch import infer as _infer
        return _infer
    raise ValueError(f"Méthode non supportée pour la visualisation : {method}")


def run_inference_for_targets(
    cfg_path: Path,
    checkpoint: Path,
    modality: str,
    source_field: str,
    target_fields: list[str],
    subjects: list[str],
    pred_dir: Path,
    n_steps: int | None,
    use_ema: bool,
    env_path: str | None,
) -> None:
    """Lance l'inférence pour chaque champ cible sur les sujets sélectionnés."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    method = cfg.get("method", "cfm3d")
    infer_fn = _dispatch_infer(method)

    input_dir = DEFAULT_DATA_DIR / "Training_prospective" / modality / source_field
    available = {subject_id_from_name(f.name) for f in input_dir.glob("*.nii.gz")}
    missing = [sid for sid in subjects if sid not in available]
    if missing:
        raise FileNotFoundError(
            f"Sujets introuvables dans {input_dir}: {missing}"
        )

    for tgt in target_fields:
        out_dir = pred_dir / f"{source_field}_to_{tgt}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Inférence {source_field}@{modality} → {tgt}@{modality} …")

        if method == "cfm3d":
            infer_fn(
                cfg_path=str(cfg_path),
                checkpoint=str(checkpoint),
                input_dir=str(input_dir),
                output_dir=str(out_dir),
                source_domain=source_field,
                target_domain=tgt,
                env_path=env_path,
                n_steps=n_steps,
                use_ema=use_ema,
            )
        else:
            infer_fn(
                cfg_path=str(cfg_path),
                checkpoint=str(checkpoint),
                output_dir=str(out_dir),
                source_field=source_field,
                source_modality=modality,
                target_field=tgt,
                target_modality=modality,
                env_path=env_path,
                input_dir=str(input_dir),
                input_volume=None,
                n_steps=n_steps,
                use_ema=use_ema,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Visualisation CFM / MMFM 3D multi-champ"
    )
    parser.add_argument("--config", required=True,
                        help="Chemin vers le YAML de configuration")
    parser.add_argument("--checkpoint", required=True,
                        help="Chemin vers le checkpoint (.pth)")
    parser.add_argument("--modality", default=SOURCE_MODALITY,
                        choices=["T1W", "T2W", "T2FLAIR"])
    parser.add_argument("--source-field", default=SOURCE_FIELD,
                        help="Champ source (ex: 0.1T)")
    parser.add_argument("--target-fields", nargs="+", default=TARGET_FIELDS,
                        help="Champs cibles (ex: 1.5T 3T 5T 7T)")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="IDs de sujets (défaut : tous dans Training_prospective)")
    parser.add_argument("--axis", type=int, default=2,
                        help="Axe de coupe (0=sagittal, 1=coronal, 2=axial)")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="Nombre de pas Euler (défaut: config inference.n_steps)")
    parser.add_argument("--pred-dir", default=None,
                        help="Répertoire de sauvegarde des prédictions")
    parser.add_argument("--out", default=None,
                        help="Chemin de sauvegarde de la figure")
    parser.add_argument("--env", default="local",
                        help="Environnement de résolution des chemins (local/remote/chemin)")
    parser.add_argument("--no-ema", action="store_true",
                        help="Ignorer les poids EMA")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    checkpoint = Path(args.checkpoint)
    if not cfg_path.exists():
        print(f"[ERREUR] Config introuvable : {cfg_path}", file=sys.stderr)
        sys.exit(1)
    if not checkpoint.exists():
        print(f"[ERREUR] Checkpoint introuvable : {checkpoint}", file=sys.stderr)
        sys.exit(1)

    if args.subjects is None:
        args.subjects = list_subjects(args.modality, args.source_field)
    print(f"Sujets : {args.subjects}")

    # Nom du run pour structurer les prédictions
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    task_name = cfg.get("task_name", cfg_path.stem)
    step_tag = checkpoint.stem

    pred_dir = Path(args.pred_dir) if args.pred_dir else (
        DEFAULT_PRED_DIR / task_name / step_tag
    )

    run_inference_for_targets(
        cfg_path=cfg_path,
        checkpoint=checkpoint,
        modality=args.modality,
        source_field=args.source_field,
        target_fields=args.target_fields,
        subjects=args.subjects,
        pred_dir=pred_dir,
        n_steps=args.n_steps,
        use_ema=not args.no_ema,
        env_path=args.env,
    )

    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / "cfm" / f"{task_name}_{step_tag}_{args.modality.lower()}_multi_field.png"
    )

    make_multi_field_figure(
        subjects=args.subjects,
        modality=args.modality,
        source_field=args.source_field,
        target_fields=args.target_fields,
        pred_dir=pred_dir,
        out_path=out_path,
        axis=args.axis,
        title=(
            f"{task_name} — {args.modality} : {args.source_field} → "
            f"{', '.join(args.target_fields)}\n"
            f"(Training_prospective, coupe centrale axe {args.axis})"
        ),
    )


if __name__ == "__main__":
    main()
