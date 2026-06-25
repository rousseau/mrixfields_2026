#!/bin/bash
# Wrapper shell appelant l'inférence batch Python pour UNet V2.
# Génère toutes les prédictions Task 1/2/3 (180 prédictions).

set -e

CHECKPOINT="outputs/cfm3d/runs/mmfm3d_unet_v2_medvae_multimodal/weights/checkpoint_20000.pth"
CONFIG="configs/mmfm3d_unet_v2_medvae_multimodal.yaml"
OUTPUT_BASE="results/mmfm/visuals/mmfm_unet_v2_all_tasks"

mkdir -p "$OUTPUT_BASE"

PYTHONPATH=src python scripts/infer_mmfm_unet_v2_batch.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_BASE" \
    --env local \
    --n_steps 50

echo "Inférence terminée. Prédictions dans : $OUTPUT_BASE"
