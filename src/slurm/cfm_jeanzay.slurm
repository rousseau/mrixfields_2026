#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Script SLURM — Entraînement OT-CFM 2D sur Jean Zay (H100 SXM5 80GB)
#
# Partition gpu_p6 : H100 SXM5 80GB, 4 GPUs/nœud
# QOS : qos_gpu_h100-t4 (100h max)
#
# Durée estimée sur H100 (×10 vs GB10) :
#   100k iters, batch=8, 512×512 : ~6-10h
#
# Lancement :
#   sbatch src/slurm/cfm_jeanzay.slurm T1W
#   sbatch src/slurm/cfm_jeanzay.slurm T2W
#   sbatch src/slurm/cfm_jeanzay.slurm T2FLAIR
#
# Prérequis (une seule fois via setup_jeanzay.sh) :
#   $WORK/MRIX/mrixfields_2026/   ← ce repo (git clone)
#   $WORK/MRIX/challenge/         ← code officiel du challenge (git clone)
#   $WORK/MRIX/mrixfields_2026/outputs/stargan2d/preprocessed/  ← .npz
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=cfm2d
#SBATCH -A crp@h100
#SBATCH -C h100

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1

#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=20:00:00

#SBATCH --output=logs/cfm2d_%x_%j.out
#SBATCH --error=logs/cfm2d_%x_%j.err

# Signal 120s avant la fin pour auto-requeue
#SBATCH --signal=USR1@120

set -e

MODALITY="${1:-T1W}"
CODE_DIR="$WORK/MRIX/mrixfields_2026"
mkdir -p "$CODE_DIR/logs"

echo "======================================================================="
echo " Job ID    : $SLURM_JOB_ID"
echo " Nœud      : $SLURM_JOB_NODELIST"
echo " Modalité  : $MODALITY"
echo " Code      : $CODE_DIR"
echo "======================================================================="

# ─── Modules Jean Zay ────────────────────────────────────────────────────────
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

# ─── Dépendances utilisateur ($WORK/.local, installées une seule fois) ───────
export PYTHONUSERBASE="$WORK/.local"
export PATH="$WORK/.local/bin:$PATH"

_install_if_missing() {
    python3 -c "import $1" 2>/dev/null || pip install --user --quiet "$2"
}
_install_if_missing torchcfm  torchcfm
_install_if_missing ot        pot
_install_if_missing torchdiffeq torchdiffeq
_install_if_missing nibabel   nibabel

# Package challenge (mrixfields) — installé depuis le clone local
CHALLENGE_DIR="$WORK/MRIX/challenge"
if [ -d "$CHALLENGE_DIR" ]; then
    python3 -c "import mrixfields" 2>/dev/null || \
        pip install --user --quiet "$CHALLENGE_DIR"
else
    echo "[ERREUR] $CHALLENGE_DIR introuvable. Lancez setup_jeanzay.sh d'abord."
    exit 1
fi

# ─── Cache PyTorch (pas d'internet sur les nœuds de calcul) ──────────────────
export TORCH_HOME="$WORK/.cache/torch"

# ─── Gestion auto-requeue ─────────────────────────────────────────────────────
_requeue_handler() {
    echo "[$(date)] Timeout imminent — requeue du job ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap '_requeue_handler' USR1

# ─── Vérification des données préprocessées ──────────────────────────────────
PREPROCESSED="$WORK/MRIX/mrixfields_2026/outputs/stargan2d/preprocessed/retro_train/${MODALITY}"
if [ ! -d "$PREPROCESSED" ]; then
    echo "[ERREUR] Données préprocessées introuvables : $PREPROCESSED"
    echo "         Transférez les .npz depuis la machine locale (voir setup_jeanzay.sh)"
    exit 1
fi
N_NPZ=$(find "$PREPROCESSED" -name "*.npz" | wc -l)
echo "Preprocessing  : OK ($N_NPZ fichiers .npz)"

echo "GPU            : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ─── Lancement ───────────────────────────────────────────────────────────────
cd "$CODE_DIR"

CONFIG="configs/cfm_${MODALITY}.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "[ERREUR] Config introuvable : $CONFIG"
    exit 1
fi

# Reprendre depuis le dernier checkpoint si disponible
WEIGHTS_DIR="$WORK/MRIX/mrixfields_2026/outputs/cfm2d/runs/cfm2d_${MODALITY}/weights"
RESUME_CKPT=""
if [ -d "$WEIGHTS_DIR" ]; then
    LAST_CKPT=$(ls "$WEIGHTS_DIR"/checkpoint_*.pth 2>/dev/null | sort -V | tail -1)
    if [ -n "$LAST_CKPT" ]; then
        echo "Reprise depuis  : $LAST_CKPT"
        RESUME_CKPT="$LAST_CKPT"
    fi
fi

# srun : alloue correctement GPU/CPUs dans le contexte SLURM
# Le & + wait permet au trap USR1 d'intercepter le signal de requeue
srun python3 -u src/train_cfm2d.py \
    --config "$CONFIG" \
    --env jeanzay \
    ${RESUME_CKPT:+--resume "$RESUME_CKPT"} &

wait $!
echo "Entraînement terminé (exit: $?)"
