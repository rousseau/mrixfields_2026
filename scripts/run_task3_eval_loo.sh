#!/bin/bash
# Task 3, INFERENCE seule pour un repli leave-one-out du fine-tuning supervise
# sur pro_train -- ecrit dans un dossier de predictions PARTAGE entre les 3
# replis (chacun n'y depose QUE son sujet tenu a l'ecart), car evaluate.py
# exige les 3 sujets prospectifs presents (PROSPECTIVE_SUBJECTS, cf.
# src/evaluation/evaluate.py) et ne sait pas noter un sous-ensemble seul.
# L'evaluation officielle se lance une seule fois, apres les 3 replis, via
# scripts/eval_task3_loo_aggregate.sh.
#
# Usage :
#   bash scripts/run_task3_eval_loo.sh <excl_subject> <checkpoint.pth> [geometry] [shared_output_dir]
# Exemple (un repli) :
#   bash scripts/run_task3_eval_loo.sh 0006 \
#       outputs/mmfm/finetune_loo_excl0006/weights/checkpoint_1050.pth \
#       cc outputs/mmfm/finetune_loo_aggregate/predictions
set -euo pipefail

EXCL=${1:?sujet tenu a l ecart manquant, ex. 0006}
CKPT=${2:?checkpoint manquant}
GEOMETRY=${3:-cc}
SHARED_OUT=${4:-outputs/mmfm/finetune_loo_aggregate/predictions}
FIELD_NORM_STATS=${5:-configs/mmfm/field_norm_stats.json}

case "$GEOMETRY" in
  8win) CROP_FLAG=() ;;
  cc)   CROP_FLAG=(--center_crop_only) ;;
  *)    echo "GEOMETRY doit valoir '8win' ou 'cc', pas '$GEOMETRY'"; exit 1 ;;
esac

CONFIG="configs/mmfm/vectorized_finetune_loo_excl${EXCL}.yaml"
[ -f "$CONFIG" ] || { echo "config absente : $CONFIG"; exit 1; }
[ -f "$CKPT" ] || { echo "checkpoint absent : $CKPT"; exit 1; }
[ -f "$FIELD_NORM_STATS" ] || { echo "field_norm_stats absent : $FIELD_NORM_STATS"; exit 1; }

cd "$(dirname "$0")/.."
export PYTHONPATH=src

echo "=== Inférence LOO — sujet tenu à l'écart : $EXCL ==="
echo "config     : $CONFIG"
echo "checkpoint : $CKPT"
echo "géométrie  : $GEOMETRY"
echo "sortie (partagée entre replis) : $SHARED_OUT"

python src/cfm/infer_mmfm_unified.py \
    --config "$CONFIG" --checkpoint "$CKPT" \
    --output_dir "$SHARED_OUT" \
    --split Training_prospective \
    --modalities T1W T2W T2FLAIR \
    --subjects "$EXCL" \
    --field_norm_stats "$FIELD_NORM_STATS" \
    "${CROP_FLAG[@]}" \
    --skip_existing
