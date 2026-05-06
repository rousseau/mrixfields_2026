#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Script SLURM — Entraînement OT-CFM 2D sur Jean Zay (H100 SXM5 80GB)
#
# Partition gpu_p6 : H100 SXM5 80GB, 4 GPUs/nœud
# QOS : qos_gpu_h100-t4 (100h max)
#
# Durée estimée sur H100 (×10 vs GB10) :
#   100k iters, batch=16, 512×512 : ~6-12h selon data loader / contention GPU
#
# Lancement :
#   sbatch src/slurm/cfm_jeanzay.slurm T1W
#   sbatch src/slurm/cfm_jeanzay.slurm T1W configs/cfm_T1W_h100_b16.yaml
#   sbatch src/slurm/cfm_jeanzay.slurm T2W
#   sbatch src/slurm/cfm_jeanzay.slurm T2FLAIR
#
# Prérequis (une seule fois via setup_jeanzay.sh) :
#   $WORK/MRIX/mrixfields_2026/   ← ce repo (git clone)
#   $WORK/MRIX/challenge/         ← code officiel du challenge (git clone)
#   $WORK/MRIX/data/              ← données NIfTI (rsync depuis machine locale)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=cfm2d
#SBATCH -A crp@h100
#SBATCH -C h100

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1

#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=100:00:00

#SBATCH --output=logs/cfm2d_%x_%j.out
#SBATCH --error=logs/cfm2d_%x_%j.err

# Signal 120s avant la fin pour auto-requeue
#SBATCH --signal=USR1@120

set -e

MODALITY="${1:-T1W}"
CONFIG_ARG="${2:-}"
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

# Package challenge (mrixfields) — installé depuis Baseline/ du clone local
CHALLENGE_BASELINE="$WORK/MRIX/challenge/Baseline"
if [ -f "$CHALLENGE_BASELINE/setup.py" ]; then
    python3 -c "import mrixfields" 2>/dev/null || \
        pip install --user --quiet --no-deps "$CHALLENGE_BASELINE"
else
    echo "[ERREUR] $CHALLENGE_BASELINE/setup.py introuvable."
    echo "         Vérifiez que $WORK/MRIX/challenge/ contient bien le repo du challenge."
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

# ─── Vérification des données NIfTI ─────────────────────────────────────────
DATA_PATH="$WORK/MRIX/data/Training_retrospective/${MODALITY}"
if [ ! -d "$DATA_PATH" ]; then
    echo "[ERREUR] Données NIfTI introuvables : $DATA_PATH"
    echo "         Transférez les données depuis la machine locale (voir setup_jeanzay.sh)"
    exit 1
fi
N_NIFTI=$(find "$DATA_PATH" -name "*.nii.gz" | wc -l)
echo "Données NIfTI  : OK ($N_NIFTI volumes dans $DATA_PATH)"

echo "GPU            : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ─── Lancement ───────────────────────────────────────────────────────────────
cd "$CODE_DIR"

if [ -n "$CONFIG_ARG" ]; then
    CONFIG="$CONFIG_ARG"
elif [ -f "configs/cfm_${MODALITY}_h100_b16.yaml" ]; then
    CONFIG="configs/cfm_${MODALITY}_h100_b16.yaml"
else
    CONFIG="configs/cfm_${MODALITY}.yaml"
fi

if [ ! -f "$CONFIG" ]; then
    echo "[ERREUR] Config introuvable : $CONFIG"
    exit 1
fi

echo "Config          : $CONFIG"

OUTPUT_SUBDIR=$(python3 - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(cfg["data"]["output_subdir"])
PY
)

TASK_NAME=$(python3 - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(cfg.get("task_name", "cfm2d"))
PY
)
echo "Run name        : $TASK_NAME"
echo "Output subdir   : $OUTPUT_SUBDIR"

# Reprendre depuis le dernier checkpoint si disponible
WEIGHTS_DIR="$WORK/MRIX/mrixfields_2026/outputs/${OUTPUT_SUBDIR}/weights"
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
