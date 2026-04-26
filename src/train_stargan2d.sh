#!/bin/bash
# ---------------------------------------------------------------------------
# Entraînement StarGAN v2 2D (baseline)  — MRIxFields 2026
#
# Utilisation :
#   bash train_stargan2d.sh [MODALITY] [MODE]
#
#   MODALITY : T1W | T2W | T2FLAIR | all  (défaut : all)
#   MODE     : retro_scratch | pro_pretrained  (défaut : retro_scratch)
#
# Étapes :
#   retro_scratch   → pré-entraînement non-supervisé sur Training_retrospective
#   pro_pretrained  → finetuning supervisé sur Training_prospective
#                     (nécessite d'avoir lancé retro_scratch au préalable)
#
# Le checkpoint final est sauvegardé dans :
#   $EXP_DIR/runs/<task_name>/stargan_v2/<MODE>/weights/model_final.pth
# ---------------------------------------------------------------------------

set -e

MODALITY=${1:-all}
MODE=${2:-retro_scratch}

CHALLENGE_DIR=/home/rousseau/Code/MRIxFields2026/Baseline
CONFIGS_DIR=${CHALLENGE_DIR}/configs/task3/stargan
EXP_DIR=/home/rousseau/Exp/mrixfields_2026/outputs/stargan2d
PYTHON=/home/rousseau/miniforge3/bin/python

# Surcharge OUTPUT_DIR : load_env() ne l'écrase pas si déjà défini
export OUTPUT_DIR="${EXP_DIR}/runs"
export DATA_DIR=/home/rousseau/Data/MRIxFields_20260414
export PREPROCESSED_DIR="${EXP_DIR}/preprocessed"

mkdir -p "${OUTPUT_DIR}"

run_training() {
    local config=$1
    echo ""
    echo "======================================================================"
    echo " Entraînement  : $(basename ${config})"
    echo " Mode          : ${MODE}"
    echo " Output        : ${OUTPUT_DIR}"
    echo "======================================================================"
    "${PYTHON}" "${CHALLENGE_DIR}/scripts/train.py" \
            --config "${config}" \
            --mode "${MODE}"
}

# Se placer dans le répertoire Baseline (chemins relatifs dans les configs)
cd "${CHALLENGE_DIR}"

case "${MODALITY}" in
    T1W)
        run_training "${CONFIGS_DIR}/any_to_any_T1W.yaml"
        ;;
    T2W)
        run_training "${CONFIGS_DIR}/any_to_any_T2W.yaml"
        ;;
    T2FLAIR)
        run_training "${CONFIGS_DIR}/any_to_any_T2FLAIR.yaml"
        ;;
    all)
        run_training "${CONFIGS_DIR}/any_to_any_T1W.yaml"
        run_training "${CONFIGS_DIR}/any_to_any_T2W.yaml"
        run_training "${CONFIGS_DIR}/any_to_any_T2FLAIR.yaml"
        ;;
    *)
        echo "ERREUR : modalité inconnue '${MODALITY}'. Valeurs possibles : T1W, T2W, T2FLAIR, all"
        exit 1
        ;;
esac

echo ""
echo "Entraînement terminé. Checkpoints dans : ${OUTPUT_DIR}"
