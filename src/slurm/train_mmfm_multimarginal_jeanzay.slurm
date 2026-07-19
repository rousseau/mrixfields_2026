#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Script SLURM — Entraînement MMFM-UNet multi-marginal sur Jean Zay (4×H100)
#
# Continue l'entraînement depuis le checkpoint du Run 1 (effectué localement sur
# DGX GB10). Pré-encode le cache latent si nécessaire, puis lance torchrun DDP.
#
# Lancement :
#   sbatch src/slurm/train_mmfm_multimarginal_jeanzay.slurm \
#       configs/mmfm3d_multimarginal_medvae_run2.yaml \
#       outputs/cfm3d/runs/mmfm3d_multimarginal_medvae_run1/weights/model_final.pth
#
# Prérequis (via sync_to_jeanzay.sh) :
#   - Code synchronisé dans $WORK/MRIX/mrixfields_2026/
#   - Checkpoint Run 1 transféré :
#       outputs/cfm3d/runs/mmfm3d_multimarginal_medvae_run1/weights/model_final.pth
#   - Poids MedVAE fine-tuné transféré :
#       outputs/medvae/runs/medvae_finetune_all/weights/model_best.pth
#   - Données NIfTI dans $WORK/MRIX/data/Training_retrospective/
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=mmfm_mm_run2
#SBATCH -A crp@h100
#SBATCH -C h100

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=20:00:00

#SBATCH --output=logs/mmfm_mm_run2_%x_%j.out
#SBATCH --error=logs/mmfm_mm_run2_%x_%j.err

#SBATCH --signal=USR1@120

set -e

CONFIG="${1:-configs/mmfm3d_multimarginal_medvae_run2.yaml}"
RESUME_CKPT="${2:-outputs/cfm3d/runs/mmfm3d_multimarginal_medvae_run1/weights/model_final.pth}"

CODE_DIR="$WORK/MRIX/mrixfields_2026"
mkdir -p "$CODE_DIR/logs"

echo "======================================================================="
echo " Job ID      : $SLURM_JOB_ID"
echo " Nœud        : $SLURM_JOB_NODELIST"
echo " Config      : $CONFIG"
echo " Resume ckpt : $RESUME_CKPT"
echo " Code        : $CODE_DIR"
echo "======================================================================="

# ─── Modules Jean Zay ────────────────────────────────────────────────────────
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

# ─── Dépendances utilisateur ─────────────────────────────────────────────────
export PYTHONUSERBASE="$WORK/.local"
export PATH="$WORK/.local/bin:$PATH"

cd "$CODE_DIR"

# ─── Vérifications ───────────────────────────────────────────────────────────
if [ ! -f "$CONFIG" ]; then
    echo "[ERREUR] Config introuvable : $CONFIG"
    exit 1
fi

if [ ! -f "$RESUME_CKPT" ]; then
    echo "[ERREUR] Checkpoint Run 1 introuvable : $RESUME_CKPT"
    echo "         Lancez d'abord : bash src/slurm/sync_to_jeanzay.sh"
    exit 1
fi

echo "GPU           : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ─── Vérification du cache latent ────────────────────────────────────────────
CACHE_DIR=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
print(c.get('data', {}).get('latent_cache_dir', ''))
" 2>/dev/null || echo "")

if [ -z "$CACHE_DIR" ]; then
    echo "[ERREUR] data.latent_cache_dir non défini dans la config."
    exit 1
fi

INDEX_FILE="$CACHE_DIR/index.json"
if [ ! -f "$INDEX_FILE" ]; then
    echo "[INFO] Cache latent absent. Pré-encodage sur 1 GPU avant DDP..."
    PYTHONPATH=src python src/cfm/precompute_latents.py \
        --config "$CONFIG" \
        --env remote
    echo "[INFO] Pré-encodage terminé."
else
    N_CACHED=$(find "$CACHE_DIR" -name "*.pt" | wc -l)
    echo "[INFO] Cache latent présent : $N_CACHED fichiers dans $CACHE_DIR"
fi

# ─── Vérification des données ────────────────────────────────────────────────
DATA_ROOT=$(python3 -c "
import yaml
with open('configs/env/remote.yaml') as f:
    e = yaml.safe_load(f)
import os
print(os.path.expandvars(e.get('data_root', '\$WORK/MRIX/data')))
" 2>/dev/null || echo "$WORK/MRIX/data")

if [ -d "$DATA_ROOT/Training_retrospective" ]; then
    N_NIFTI=$(find "$DATA_ROOT/Training_retrospective" -name "*.nii.gz" | wc -l)
    echo "Données NIfTI  : OK ($N_NIFTI volumes dans Training_retrospective)"
else
    echo "[AVERTISSEMENT] Données NIfTI non trouvées : $DATA_ROOT/Training_retrospective"
fi

# ─── Lancement multi-GPU (torchrun DDP 4×H100) ───────────────────────────────
export DATALOADER_NUM_WORKERS=8

echo "======================================================================="
echo " Démarrage torchrun (4 GPUs H100) — reprise depuis $RESUME_CKPT"
echo "======================================================================="

MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port="$MASTER_PORT" \
    src/cfm/train_mmfm_unet_3d.py \
    --config "$CONFIG" \
    --env remote \
    --resume "$RESUME_CKPT"

echo "======================================================================="
echo " Job terminé (ou interrompu pour requeue)"
echo "======================================================================="
