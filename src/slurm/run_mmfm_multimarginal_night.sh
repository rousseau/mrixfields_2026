#!/usr/bin/env bash
# Lance pré-encodage + entraînement MMFM-UNet multi-marginal pour une nuit.
# Usage: bash src/slurm/run_mmfm_multimarginal_night.sh [config] [env]

set -e

CONFIG="${1:-configs/mmfm3d_multimarginal_medvae_run1.yaml}"
ENV="${2:-local}"

echo "[$(date)] Démarrage pré-encodage : config=$CONFIG env=$ENV"
PYTHONPATH=src python src/cfm/precompute_latents.py --config "$CONFIG" --env "$ENV"

echo "[$(date)] Pré-encodage terminé. Démarrage entraînement."
PYTHONPATH=src python src/cfm/train_mmfm_unet_3d.py --config "$CONFIG" --env "$ENV"

echo "[$(date)] Entraînement terminé."
