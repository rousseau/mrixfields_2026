#!/bin/bash
# ---------------------------------------------------------------------------
# Surveillance des checkpoints OT-CFM 2D et génération de figures de suivi
#
# Se lance en arrière-plan (dans screen) pendant l'entraînement.
# Dès qu'un nouveau checkpoint_*.pth apparaît, lance l'inférence sur les
# sujets prospectifs et génère une figure dans results/.
#
# Utilisation :
#   bash src/watch_and_viz_cfm2d.sh [MODALITY] [CHECK_INTERVAL_SECONDS]
# ---------------------------------------------------------------------------

set -e

MODALITY=${1:-T1W}
CHECK_INTERVAL=${2:-600}
ENV=${3:-local}   # 'local' | 'jeanzay' | chemin vers configs/env/*.yaml

EXP_DIR=/home/rousseau/Exp/mrixfields_2026
CFM_OUTPUT_DIR=${EXP_DIR}/outputs/cfm2d
RESULTS_DIR=${EXP_DIR}/results
PYTHON=/home/rousseau/miniforge3/bin/python
VIZ_SCRIPT=${EXP_DIR}/src/visualize_cfm2d.py
CONFIG=${EXP_DIR}/configs/cfm_${MODALITY}.yaml

# Chemin des checkpoints (selon la config cfm_<MODALITY>.yaml)
WEIGHTS_DIR="${CFM_OUTPUT_DIR}/runs/cfm2d_${MODALITY}/weights"
LOG="${CFM_OUTPUT_DIR}/watch_viz_${MODALITY}.log"

declare -A DONE

echo "======================================================" | tee -a "${LOG}"
echo " watch_and_viz_cfm2d.sh démarré : ${MODALITY}"        | tee -a "${LOG}"
echo " Surveillance : ${WEIGHTS_DIR}"                        | tee -a "${LOG}"
echo " Intervalle   : ${CHECK_INTERVAL}s"                    | tee -a "${LOG}"
echo " Log          : ${LOG}"                                 | tee -a "${LOG}"
echo "======================================================" | tee -a "${LOG}"

while true; do
    if [ ! -d "${WEIGHTS_DIR}" ]; then
        echo "[$(date '+%H:%M:%S')] Dossier weights pas encore créé, attente..." | tee -a "${LOG}"
        sleep "${CHECK_INTERVAL}"
        continue
    fi

    # Parcourir les checkpoints intermédiaires
    for ckpt in "${WEIGHTS_DIR}"/checkpoint_*.pth; do
        [ -f "${ckpt}" ] || continue
        step_tag=$(basename "${ckpt}" .pth)

        [ -n "${DONE[${step_tag}]}" ] && continue

        fig="${RESULTS_DIR}/cfm2d_${MODALITY,,}_${step_tag}.png"
        echo "" | tee -a "${LOG}"
        echo "[$(date '+%H:%M:%S')] Nouveau checkpoint : ${step_tag}" | tee -a "${LOG}"
        echo "  → Inférence + visualisation…" | tee -a "${LOG}"

        "${PYTHON}" "${VIZ_SCRIPT}" \
            --modality "${MODALITY}" \
            --checkpoint "${ckpt}" \
            --config "${CONFIG}" \
            --env "${ENV}" \
            --out "${fig}" \
            2>&1 | tee -a "${LOG}"

        DONE["${step_tag}"]=1
        echo "[$(date '+%H:%M:%S')] Figure : ${fig}" | tee -a "${LOG}"
    done

    # Checkpoint final
    final_ckpt="${WEIGHTS_DIR}/model_final.pth"
    if [ -f "${final_ckpt}" ] && [ -z "${DONE[model_final]}" ]; then
        fig="${RESULTS_DIR}/cfm2d_${MODALITY,,}_model_final.png"
        echo "" | tee -a "${LOG}"
        echo "[$(date '+%H:%M:%S')] Checkpoint final détecté !" | tee -a "${LOG}"

        "${PYTHON}" "${VIZ_SCRIPT}" \
            --modality "${MODALITY}" \
            --checkpoint "${final_ckpt}" \
            --config "${CONFIG}" \
            --env "${ENV}" \
            --out "${fig}" \
            2>&1 | tee -a "${LOG}"

        DONE["model_final"]=1
        echo "[$(date '+%H:%M:%S')] Figure finale : ${fig}" | tee -a "${LOG}"
        echo "[$(date '+%H:%M:%S')] Surveillance terminée." | tee -a "${LOG}"
        break
    fi

    sleep "${CHECK_INTERVAL}"
done
