#!/bin/bash
# ---------------------------------------------------------------------------
# Inférence StarGAN v2 2D (baseline) — MRIxFields 2026
#
# Utilisation :
#   bash run_inference_stargan2d.sh [MODALITY] [SOURCE_FIELD]
#
#   MODALITY     : T1W | T2W | T2FLAIR | all  (défaut : T1W)
#   SOURCE_FIELD : 0.1T | 1.5T | 3T | 5T | 7T | all  (défaut : 0.1T)
#
# Traduit les volumes du split Training_prospective depuis SOURCE_FIELD vers
# tous les autres champs cibles.  Les prédictions sont sauvegardées dans :
#   $EXP_DIR/predictions/<modality>/<source_field>_to_<target_field>/
#
# Prérequis : avoir lancé train_stargan2d.sh (mode retro_scratch au minimum).
# ---------------------------------------------------------------------------

set -e

MODALITY=${1:-T1W}
SOURCE_FIELD=${2:-0.1T}

CHALLENGE_DIR=/home/rousseau/Code/MRIxFields2026/Baseline
EXP_DIR=/home/rousseau/Exp/mrixfields_2026/outputs/stargan2d
DATA_DIR=/home/rousseau/Data/MRIxFields_20260414
PYTHON=/home/rousseau/miniforge3/bin/python

export OUTPUT_DIR="${EXP_DIR}/runs"
export DATA_DIR
export PREPROCESSED_DIR="${CHALLENGE_DIR}/preprocessed"

ALL_FIELDS=(0.1T 1.5T 3T 5T 7T)
ALL_MODALITIES=(T1W T2W T2FLAIR)

# Détermine le chemin vers le checkpoint final pour une modalité donnée
get_checkpoint() {
    local mod=$1
    local task_name="task3_any_to_any_${mod}"
    local ckpt="${OUTPUT_DIR}/${task_name}/stargan_v2/retro_scratch/weights/model_final.pth"
    echo "${ckpt}"
}

# Détermine le chemin vers la config pour une modalité
get_config() {
    local mod=$1
    echo "${CHALLENGE_DIR}/configs/task3/stargan/any_to_any_${mod}.yaml"
}

run_inference_one() {
    local mod=$1
    local src=$2
    local tgt=$3

    local config
    config=$(get_config "${mod}")
    local ckpt
    ckpt=$(get_checkpoint "${mod}")

    if [ ! -f "${ckpt}" ]; then
        echo "AVERTISSEMENT : checkpoint introuvable pour ${mod} : ${ckpt}"
        echo "  → Lancez d'abord : bash train_stargan2d.sh ${mod} retro_scratch"
        return
    fi

    local input_dir="${DATA_DIR}/Training_prospective/${mod}/${src}"
    local output_dir="${EXP_DIR}/predictions/${mod}/${src}_to_${tgt}"

    if [ ! -d "${input_dir}" ]; then
        echo "AVERTISSEMENT : répertoire source introuvable : ${input_dir}"
        return
    fi

    echo ""
    echo "----------------------------------------------------------------------"
    echo " Inférence  : ${mod}  |  ${src} → ${tgt}"
    echo " Input      : ${input_dir}"
    echo " Output     : ${output_dir}"
    echo "----------------------------------------------------------------------"

    "${PYTHON}" "${CHALLENGE_DIR}/scripts/inference.py" \
            --config "${config}" \
            --checkpoint "${ckpt}" \
            --input_dir "${input_dir}" \
            --output_dir "${output_dir}" \
            --target_field "${tgt}"
}

run_all_targets() {
    local mod=$1
    local src=$2
    for tgt in "${ALL_FIELDS[@]}"; do
        if [ "${tgt}" != "${src}" ]; then
            run_inference_one "${mod}" "${src}" "${tgt}"
        fi
    done
}

# Expansion des arguments "all"
if [ "${MODALITY}" = "all" ]; then
    MODALITIES=("${ALL_MODALITIES[@]}")
else
    MODALITIES=("${MODALITY}")
fi

if [ "${SOURCE_FIELD}" = "all" ]; then
    SOURCES=("${ALL_FIELDS[@]}")
else
    SOURCES=("${SOURCE_FIELD}")
fi

cd "${CHALLENGE_DIR}"

for mod in "${MODALITIES[@]}"; do
    for src in "${SOURCES[@]}"; do
        run_all_targets "${mod}" "${src}"
    done
done

echo ""
echo "Inférence terminée. Résultats dans : ${EXP_DIR}/predictions/"
