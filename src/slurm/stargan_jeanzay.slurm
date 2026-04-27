#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Script SLURM — Entraînement StarGAN v2 2D sur Jean Zay (H100 SXM5 80GB)
#
# Partition gpu_p6 : H100 SXM5 80GB, 4 GPUs/nœud
# QOS : qos_gpu_h100-t4 (100h max)
#
# Durée estimée sur H100 (200k iters, 512×512, batch=8) :
#   retro_scratch : ~10-16h par modalité
#   pro_pretrained : ~2-4h par modalité
#
# Lancement :
#   sbatch src/slurm/stargan_jeanzay.slurm T1W retro_scratch
#   sbatch src/slurm/stargan_jeanzay.slurm T2W retro_scratch
#   sbatch src/slurm/stargan_jeanzay.slurm T2FLAIR retro_scratch
#
#   # Finetuning (après retro_scratch) :
#   sbatch src/slurm/stargan_jeanzay.slurm T1W pro_pretrained
#
# Prérequis (une seule fois via setup_jeanzay.sh) :
#   $WORK/MRIX/mrixfields_2026/   ← ce repo (git clone)
#   $WORK/MRIX/challenge/         ← code officiel du challenge (git clone)
#   $WORK/MRIX/data/              ← données NIfTI (rsync depuis machine locale)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=stargan2d
#SBATCH -A crp@h100
#SBATCH -C h100

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1

#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=20:00:00

#SBATCH --output=logs/stargan2d_%x_%j.out
#SBATCH --error=logs/stargan2d_%x_%j.err

# Signal 120s avant la fin pour auto-requeue
#SBATCH --signal=USR1@120

set -e

MODALITY="${1:-T1W}"
MODE="${2:-retro_scratch}"
CODE_DIR="$WORK/MRIX/mrixfields_2026"
mkdir -p "$CODE_DIR/logs"

echo "======================================================================="
echo " Job ID    : $SLURM_JOB_ID"
echo " Nœud      : $SLURM_JOB_NODELIST"
echo " Modalité  : $MODALITY"
echo " Mode      : $MODE"
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
_install_if_missing nibabel nibabel

# Package challenge (mrixfields) — installé depuis Baseline/ du clone local
CHALLENGE_BASELINE="$WORK/MRIX/challenge/Baseline"
if [ -f "$CHALLENGE_BASELINE/setup.py" ]; then
    python3 -c "import mrixfields" 2>/dev/null || \
        pip install --user --quiet --no-deps "$CHALLENGE_BASELINE"
else
    echo "[ERREUR] $CHALLENGE_BASELINE/setup.py introuvable."
    exit 1
fi

# ─── Cache PyTorch ────────────────────────────────────────────────────────────
export TORCH_HOME="$WORK/.cache/torch"

# ─── Gestion auto-requeue ─────────────────────────────────────────────────────
_requeue_handler() {
    echo "[$(date)] Timeout imminent — requeue du job ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap '_requeue_handler' USR1

echo "GPU : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ─── Vérification des données ────────────────────────────────────────────────
DATA_PATH="$WORK/MRIX/data/Training_retrospective/${MODALITY}"
if [ ! -d "$DATA_PATH" ]; then
    echo "[ERREUR] Données introuvables : $DATA_PATH"
    echo "         Transférez les NIfTI depuis la machine locale (voir setup_jeanzay.sh)"
    exit 1
fi

# ─── Lancement ───────────────────────────────────────────────────────────────
cd "$CODE_DIR"

# Passer les chemins Jean Zay à train_stargan2d.sh via variables d'environnement
export CHALLENGE_DIR="$WORK/MRIX/challenge/Baseline"
export EXP_DIR="$WORK/MRIX/mrixfields_2026/outputs/stargan2d"
export DATA_DIR="$WORK/MRIX/data"
export PYTHON="python3"

# srun : alloue correctement GPU/CPUs dans le contexte SLURM
# Le & + wait permet au trap USR1 d'intercepter le signal de requeue
srun bash src/train_stargan2d.sh "${MODALITY}" "${MODE}" &

wait $!
echo "Entraînement terminé (exit: $?)"
