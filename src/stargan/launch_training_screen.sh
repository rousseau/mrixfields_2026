#!/bin/bash
# ---------------------------------------------------------------------------
# Lance l'entraînement StarGAN v2 + surveillance dans des sessions screen
#
# Utilisation :
#   bash src/launch_training_screen.sh [MODALITY] [CHECK_INTERVAL_SECONDS]
#
#   MODALITY         : T1W | T2W | T2FLAIR  (défaut : T1W)
#   CHECK_INTERVAL   : intervalle de vérif des checkpoints (défaut : 600 = 10 min)
#
# Sessions créées :
#   stargan_train_<MOD>  — entraînement
#   stargan_viz_<MOD>    — surveillance + visualisation périodique
#
# Pour suivre en direct :
#   screen -r stargan_train_T1W
#   screen -r stargan_viz_T1W
#   (Ctrl+A D pour détacher)
#
# Pour arrêter l'entraînement :
#   screen -r stargan_train_T1W  puis  Ctrl+C
# ---------------------------------------------------------------------------

set -e

MODALITY=${1:-T1W}
CHECK_INTERVAL=${2:-600}

EXP_DIR=/home/rousseau/Exp/mrixfields_2026/outputs/stargan2d
CHALLENGE_DIR=/home/rousseau/Code/MRIxFields2026/Baseline
PYTHON=/home/rousseau/miniforge3/bin/python
SCRIPTS_DIR=/home/rousseau/Exp/mrixfields_2026/src

TRAIN_SESSION="stargan_train_${MODALITY}"
VIZ_SESSION="stargan_viz_${MODALITY}"
TRAIN_LOG="${EXP_DIR}/train_${MODALITY}_screen.log"

# ---------------------------------------------------------------------------
# Vérifications préalables
# ---------------------------------------------------------------------------

echo "======================================================================="
echo " Lancement entraînement StarGAN v2 2D — ${MODALITY}"
echo "======================================================================="

# Preprocessing : vérifie que les slices sont présentes
PREPROCESSED="${EXP_DIR}/preprocessed/retro_train/${MODALITY}"
if [ ! -d "${PREPROCESSED}" ]; then
    echo ""
    echo "[ERREUR] Données pré-traitées introuvables : ${PREPROCESSED}"
    echo "         Lancez d'abord le preprocessing :"
    echo ""
    echo "  cd ${CHALLENGE_DIR} && \\"
    echo "  PREPROCESSED_DIR=${EXP_DIR}/preprocessed \\"
    echo "  DATA_DIR=/home/rousseau/Data/MRIxFields_20260414 \\"
    echo "  ${PYTHON} scripts/preprocess.py extract-slices \\"
    echo "    --splits retro_train --modalities ${MODALITY} \\"
    echo "    --output_dir ${EXP_DIR}/preprocessed"
    exit 1
fi
N_NPZ=$(find "${PREPROCESSED}" -name "*.npz" | wc -l)
echo " Preprocessing  : OK (${N_NPZ} slices .npz)"

# Config
CONFIG="${CHALLENGE_DIR}/configs/task3/stargan/any_to_any_${MODALITY}.yaml"
if [ ! -f "${CONFIG}" ]; then
    echo "[ERREUR] Config introuvable : ${CONFIG}"
    exit 1
fi
echo " Config         : ${CONFIG}"

# Vérifier que screen est installé
if ! command -v screen &>/dev/null; then
    echo "[ERREUR] 'screen' n'est pas installé. Installez-le avec : sudo apt install screen"
    exit 1
fi

# Tuer les sessions existantes du même nom si elles existent
screen -ls "${TRAIN_SESSION}" &>/dev/null && screen -S "${TRAIN_SESSION}" -X quit 2>/dev/null || true
screen -ls "${VIZ_SESSION}"   &>/dev/null && screen -S "${VIZ_SESSION}"   -X quit 2>/dev/null || true

echo " Sessions       : ${TRAIN_SESSION} / ${VIZ_SESSION}"
echo " Log train      : ${TRAIN_LOG}"
echo ""

# ---------------------------------------------------------------------------
# Lancement de l'entraînement dans screen
# ---------------------------------------------------------------------------

TRAIN_CMD="cd ${CHALLENGE_DIR} && \
export OUTPUT_DIR=${EXP_DIR}/runs && \
export DATA_DIR=/home/rousseau/Data/MRIxFields_20260414 && \
export PREPROCESSED_DIR=${EXP_DIR}/preprocessed && \
${PYTHON} scripts/train.py \
  --config ${CONFIG} \
  --mode retro_scratch \
  2>&1 | tee ${TRAIN_LOG}; \
echo ''; echo '[ENTRAÎNEMENT TERMINÉ]'; exec bash"

screen -dmS "${TRAIN_SESSION}" bash -c "${TRAIN_CMD}"
echo " ✓ Session entraînement démarrée : screen -r ${TRAIN_SESSION}"

# ---------------------------------------------------------------------------
# Lancement de la surveillance + visualisation dans screen
# ---------------------------------------------------------------------------

# Petit délai pour laisser l'entraînement démarrer
VIZ_CMD="sleep 30 && \
bash ${SCRIPTS_DIR}/watch_and_viz.sh ${MODALITY} ${CHECK_INTERVAL}; \
exec bash"

screen -dmS "${VIZ_SESSION}" bash -c "${VIZ_CMD}"
echo " ✓ Session visualisation démarrée : screen -r ${VIZ_SESSION}"

# ---------------------------------------------------------------------------
# Résumé
# ---------------------------------------------------------------------------

echo ""
echo "======================================================================="
echo " Sessions screen actives :"
echo "   Entraînement  →  screen -r ${TRAIN_SESSION}"
echo "   Visualisation →  screen -r ${VIZ_SESSION}"
echo ""
echo " Figures générées dans :"
echo "   /home/rousseau/Exp/mrixfields_2026/results/"
echo "   (une figure par checkpoint intermédiaire, tous les 10 000 steps)"
echo ""
echo " Détacher d'une session : Ctrl+A puis D"
echo " Arrêter l'entraînement : screen -r ${TRAIN_SESSION}  puis  Ctrl+C"
echo "======================================================================="
