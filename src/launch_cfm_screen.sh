#!/bin/bash
# ---------------------------------------------------------------------------
# Lance l'entraînement OT-CFM 2D + surveillance dans des sessions screen
#
# Utilisation :
#   bash src/launch_cfm_screen.sh [MODALITY] [CHECK_INTERVAL_SECONDS]
#
#   MODALITY         : T1W | T2W | T2FLAIR  (défaut : T1W)
#   CHECK_INTERVAL   : intervalle de vérif des checkpoints (défaut : 600 = 10 min)
#
# Sessions créées :
#   cfm_train_<MOD>  — entraînement
#   cfm_viz_<MOD>    — surveillance + visualisation périodique
#
# Pour suivre en direct :
#   screen -r cfm_train_T1W
#   screen -r cfm_viz_T1W
#   (Ctrl+A D pour détacher)
# ---------------------------------------------------------------------------

set -e

MODALITY=${1:-T1W}
CHECK_INTERVAL=${2:-600}
ENV=${3:-local}   # 'local' | 'jeanzay' | chemin vers configs/env/*.yaml

EXP_DIR=/home/rousseau/Exp/mrixfields_2026
CFM_OUTPUT_DIR=${EXP_DIR}/outputs/cfm2d
PYTHON=/home/rousseau/miniforge3/bin/python
SCRIPTS_DIR=${EXP_DIR}/src

TRAIN_SESSION="cfm_train_${MODALITY}"
VIZ_SESSION="cfm_viz_${MODALITY}"
CONFIG="${EXP_DIR}/configs/cfm_${MODALITY}.yaml"
TRAIN_LOG="${CFM_OUTPUT_DIR}/train_${MODALITY}_screen.log"

# ---------------------------------------------------------------------------
# Vérifications
# ---------------------------------------------------------------------------

echo "======================================================================="
echo " Lancement entraînement OT-CFM 2D — ${MODALITY}"
echo "======================================================================="

if [ ! -f "${CONFIG}" ]; then
    echo "[ERREUR] Config introuvable : ${CONFIG}"
    exit 1
fi
echo " Config         : ${CONFIG}"

# Vérifier les données preprocessées (réutilise celles de StarGAN)
PREPROCESSED="/home/rousseau/Exp/mrixfields_2026/outputs/stargan2d/preprocessed/retro_train/${MODALITY}"
if [ ! -d "${PREPROCESSED}" ]; then
    echo ""
    echo "[ERREUR] Données pré-traitées introuvables : ${PREPROCESSED}"
    echo "         Ces données sont partagées avec StarGAN."
    echo "         Lancez d'abord le preprocessing StarGAN pour ${MODALITY}."
    exit 1
fi
N_NPZ=$(find "${PREPROCESSED}" -name "*.npz" | wc -l)
echo " Preprocessing  : OK (${N_NPZ} .npz dans ${PREPROCESSED})"

if ! command -v screen &>/dev/null; then
    echo "[ERREUR] 'screen' non installé."
    exit 1
fi

mkdir -p "${CFM_OUTPUT_DIR}"

# Tuer les sessions existantes du même nom
screen -ls "${TRAIN_SESSION}" &>/dev/null && screen -S "${TRAIN_SESSION}" -X quit 2>/dev/null || true
screen -ls "${VIZ_SESSION}"   &>/dev/null && screen -S "${VIZ_SESSION}"   -X quit 2>/dev/null || true

echo " Sessions       : ${TRAIN_SESSION} / ${VIZ_SESSION}"
echo " Log train      : ${TRAIN_LOG}"
echo ""

# ---------------------------------------------------------------------------
# Session entraînement
# ---------------------------------------------------------------------------

TRAIN_CMD="${PYTHON} ${SCRIPTS_DIR}/train_cfm2d.py \
  --mode train \
  --config ${CONFIG} \
  --env ${ENV} \
  2>&1 | tee ${TRAIN_LOG}; \
echo ''; echo '[ENTRAÎNEMENT CFM TERMINÉ]'; exec bash"

screen -dmS "${TRAIN_SESSION}" bash -c "${TRAIN_CMD}"
echo " ✓ Session entraînement démarrée : screen -r ${TRAIN_SESSION}"

# ---------------------------------------------------------------------------
# Session visualisation (délai 60s pour laisser le premier checkpoint arriver)
# ---------------------------------------------------------------------------

VIZ_CMD="sleep 60 && \
bash ${SCRIPTS_DIR}/watch_and_viz_cfm2d.sh ${MODALITY} ${CHECK_INTERVAL} ${ENV}; \
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
echo "   (une figure par checkpoint, toutes les 5 000 itérations)"
echo ""
echo " Log en direct :"
echo "   tail -f ${TRAIN_LOG}"
echo ""
echo " Détacher d'une session : Ctrl+A puis D"
echo " Arrêter l'entraînement : screen -r ${TRAIN_SESSION}  puis  Ctrl+C"
echo "======================================================================="
