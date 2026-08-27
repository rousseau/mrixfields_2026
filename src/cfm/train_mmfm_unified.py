#!/usr/bin/env python3
"""Unified MMFM 3D — single CLI entry point for both architectures.

Replaces train_mmfm_3d.py (vectorized MLP) and train_mmfm_unet_3d.py (spatial
UNet): same dataloader, same multi-marginal sampler, same OT-CFM coupling,
same losses/regularizers, same checkpoint schema, same evaluation contract —
the ONLY thing that differs between `--method mmfm3d_vectorized` and
`--method mmfm3d_unet` is the model architecture itself (see mmfm_core.py's
module docstring and arch_vector.py / arch_unet.py).

All experiment-behavior knobs (adjacent_only, identity_prob,
accumulation_steps, num_targets_per_step, warmup_steps, lambda_*,
use_latent_cache, latent_cache_root/dir) live in the YAML config's
`train:`/`data:` blocks, not as CLI flags — see mmfm_core.py's `train()`.
Only paths/invocation stay on the CLI.

Usage:
  # Entraînement (vectorisé)
  PYTHONPATH=src python src/cfm/train_mmfm_unified.py \\
      --method mmfm3d_vectorized \\
      --config configs/mmfm/vectorized.yaml --env local

  # Entraînement (UNet), reprise poids-seuls
  PYTHONPATH=src python src/cfm/train_mmfm_unified.py \\
      --method mmfm3d_unet \\
      --config configs/mmfm/unet.yaml --env local \\
      --resume outputs/mmfm/unet/weights/model_final.pth --resume_weights_only

  # Inférence
  PYTHONPATH=src python src/cfm/train_mmfm_unified.py --mode infer \\
      --method mmfm3d_unet \\
      --config configs/mmfm/unet.yaml --env local \\
      --checkpoint outputs/mmfm/unet/weights/model_final.pth \\
      --input_volume /path/to/sub_T1W_0.1T_0001.nii.gz \\
      --output_dir outputs/predictions/mmfm_unified/ \\
      --source_field 0.1T --source_modality T1W \\
      --target_field 7T   --target_modality T1W
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cfm.mmfm_core import train, infer


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MMFM 3D unifié — any-to-any multimodal, vectorisé ou UNet spatial"
    )
    p.add_argument("--mode", default="train", choices=["train", "infer"])
    p.add_argument(
        "--method", required=True,
        choices=["mmfm3d_vectorized", "mmfm3d_unet", "mmfm3d_inr", "mmfm3d_synthetic"],
        help="Architecture : MLP vectoriel, UNet 3D spatial (MONAI), latent INR "
             "(SIREN+hypernetwork), ou harnais synthetique a reponse connue",
    )
    p.add_argument("--config", required=True, help="Chemin vers le YAML de configuration")
    p.add_argument("--env", default=None, help="Env YAML (local / remote / chemin)")
    # Train-only (paths/invocation — experiment knobs live in the YAML config)
    p.add_argument("--resume", default=None, help="[train] Reprendre depuis ce checkpoint")
    p.add_argument(
        "--resume_weights_only", action="store_true",
        help="[train] Avec --resume : ne charger que les poids modèle+EMA "
             "(pas optimizer/scheduler/scaler, start_iter=0) — pour une "
             "nouvelle phase de fine-tuning indépendante.",
    )
    p.add_argument(
        "--field_norm_stats", default=None,
        help="[train/infer] Chemin JSON produit par compute_field_norm_stats.py "
             "(normalisation fixe par champ au lieu de percentile par volume)",
    )
    # Infer-only
    p.add_argument("--checkpoint", default=None, help="[infer] Chemin vers le checkpoint (.pth)")
    p.add_argument("--input_dir", default=None, help="[infer] Répertoire de volumes .nii.gz source")
    p.add_argument("--input_volume", default=None, help="[infer] Volume .nii.gz unique source")
    p.add_argument("--output_dir", default=None, help="[infer] Répertoire de sortie des prédictions")
    p.add_argument("--source_field", default=None, help="[infer] Champ magnétique source (ex. '0.1T')")
    p.add_argument("--source_modality", default=None, help="[infer] Modalité source (ex. 'T1W')")
    p.add_argument("--target_field", default=None, help="[infer] Champ magnétique cible (ex. '7T')")
    p.add_argument("--target_modality", default=None, help="[infer] Modalité cible (ex. 'T1W')")
    p.add_argument("--n_steps", type=int, default=None, help="[infer] Nombre de pas Euler (défaut: 20)")
    p.add_argument("--no_ema", action="store_true", help="[infer] Ignorer les poids EMA")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.mode == "train":
        train(
            cfg_path=args.config,
            method=args.method,
            env_path=args.env,
            resume=args.resume,
            resume_weights_only=args.resume_weights_only,
            field_norm_stats_path=args.field_norm_stats,
        )
    else:
        missing = [
            name for name, val in [
                ("--checkpoint", args.checkpoint),
                ("--output_dir", args.output_dir),
                ("--source_field", args.source_field),
                ("--source_modality", args.source_modality),
                ("--target_field", args.target_field),
                ("--target_modality", args.target_modality),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(
                f"Mode infer : arguments manquants : {', '.join(missing)}\n"
                "Fournir aussi --input_dir ou --input_volume."
            )
        if args.input_dir is None and args.input_volume is None:
            raise ValueError("Mode infer : fournir --input_dir ou --input_volume.")
        infer(
            cfg_path=args.config,
            method=args.method,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            source_field=args.source_field,
            source_modality=args.source_modality,
            target_field=args.target_field,
            target_modality=args.target_modality,
            env_path=args.env,
            input_dir=args.input_dir,
            input_volume=args.input_volume,
            n_steps=args.n_steps,
            use_ema=not args.no_ema,
            field_norm_stats_path=args.field_norm_stats,
        )
