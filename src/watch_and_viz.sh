#!/bin/bash
# ---------------------------------------------------------------------------
# Surveillance des checkpoints StarGAN v2 et génération de figures de suivi
#
# Se lance en arrière-plan (dans screen) pendant l'entraînement.
# Dès qu'un nouveau checkpoint_*.pth apparaît, lance l'inférence sur les
# sujets prospectifs et génère une figure de comparaison dans results/.
#
# Utilisation :
#   bash src/watch_and_viz.sh [MODALITY] [CHECK_INTERVAL_SECONDS]
#
#   MODALITY         : T1W | T2W | T2FLAIR  (défaut : T1W)
#   CHECK_INTERVAL   : intervalle de vérification en secondes  (défaut : 600 = 10 min)
# ---------------------------------------------------------------------------

set -e

MODALITY=${1:-T1W}
CHECK_INTERVAL=${2:-600}

EXP_DIR=/home/rousseau/Exp/mrixfields_2026/outputs/stargan2d
RESULTS_DIR=/home/rousseau/Exp/mrixfields_2026/results
CHALLENGE_DIR=/home/rousseau/Code/MRIxFields2026/Baseline
PYTHON=/home/rousseau/miniforge3/bin/python
VIZ_SCRIPT=/home/rousseau/Exp/mrixfields_2026/src/visualize_stargan2d.py

WEIGHTS_DIR="${EXP_DIR}/runs/task3_any_to_any_${MODALITY}/stargan_v2/retro_scratch/weights"
CONFIG="${CHALLENGE_DIR}/configs/task3/stargan/any_to_any_${MODALITY}.yaml"
LOG="${EXP_DIR}/watch_viz_${MODALITY}.log"

# Ensemble des checkpoints déjà visualisés (évite les doublons)
declare -A DONE

echo "======================================================================" | tee -a "${LOG}"
echo " watch_and_viz.sh démarré : ${MODALITY}" | tee -a "${LOG}"
echo " Surveillance : ${WEIGHTS_DIR}" | tee -a "${LOG}"
echo " Intervalle   : ${CHECK_INTERVAL}s" | tee -a "${LOG}"
echo " Log          : ${LOG}" | tee -a "${LOG}"
echo "======================================================================" | tee -a "${LOG}"

while true; do
    # Attendre que le dossier weights existe (l'entraînement ne l'a pas encore créé)
    if [ ! -d "${WEIGHTS_DIR}" ]; then
        echo "[$(date '+%H:%M:%S')] Dossier weights pas encore créé, attente..." | tee -a "${LOG}"
        sleep "${CHECK_INTERVAL}"
        continue
    fi

    # Parcourir tous les checkpoints intermédiaires (checkpoint_NNNNN.pth)
    for ckpt in "${WEIGHTS_DIR}"/checkpoint_*.pth; do
        [ -f "${ckpt}" ] || continue
        step_tag=$(basename "${ckpt}" .pth)  # ex: checkpoint_10000

        # Déjà traité ?
        if [ -n "${DONE[${step_tag}]}" ]; then
            continue
        fi

        # Fichier de figure correspondant
        fig="${RESULTS_DIR}/stargan2d_${MODALITY,,}_${step_tag}.png"

        echo "" | tee -a "${LOG}"
        echo "[$(date '+%H:%M:%S')] Nouveau checkpoint : ${step_tag}" | tee -a "${LOG}"
        echo "  → Inférence + visualisation…" | tee -a "${LOG}"

        "${PYTHON}" "${VIZ_SCRIPT}" \
            --modality "${MODALITY}" \
            --checkpoint "${ckpt}" \
            --config "${CONFIG}" \
            --out "${fig}" \
            2>&1 | tee -a "${LOG}"

        DONE["${step_tag}"]=1
        echo "[$(date '+%H:%M:%S')] Figure sauvegardée : ${fig}" | tee -a "${LOG}"
    done

    # Générer une figure depuis model_final.pth si présent et pas encore fait
    final_ckpt="${WEIGHTS_DIR}/model_final.pth"
    if [ -f "${final_ckpt}" ] && [ -z "${DONE[model_final]}" ]; then
        fig="${RESULTS_DIR}/stargan2d_${MODALITY,,}_model_final.png"
        echo "" | tee -a "${LOG}"
        echo "[$(date '+%H:%M:%S')] model_final.pth détecté — visualisation finale" | tee -a "${LOG}"
        "${PYTHON}" "${VIZ_SCRIPT}" \
            --modality "${MODALITY}" \
            --checkpoint "${final_ckpt}" \
            --config "${CONFIG}" \
            --out "${fig}" \
            2>&1 | tee -a "${LOG}"
        DONE[model_final]=1
        echo "[$(date '+%H:%M:%S')] Entraînement terminé. Fin de la surveillance." | tee -a "${LOG}"
        break
    fi

    sleep "${CHECK_INTERVAL}"
done
