#!/bin/bash
# SLURM - MedVAE disentanglement v1.2 (anatomie/modalite) sur Jean Zay
# v1.2: progressive paired-loss warmup + tighter adversarial schedule

#SBATCH --job-name=medvae_dis_v12
#SBATCH -A crp@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=48:00:00
#SBATCH --output=logs/medvae_dis_v12_%x_%j.out
#SBATCH --error=logs/medvae_dis_v12_%x_%j.err

set -e

RUN_NAME="${1:-medvae_disentangle_v12}"
MODALITIES="${2:-T1W T2W T2FLAIR}"
FIELDS="${3:-0.1T 1.5T 3T 5T 7T}"

CODE_DIR="$WORK/MRIX/mrixfields_2026"
DATA_ROOT="$WORK/MRIX/data"
OUT_DIR="$WORK/MRIX/mrixfields_2026/outputs/medvae_disentangle_v1/runs/${RUN_NAME}"

mkdir -p "$CODE_DIR/logs"

module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

export PYTHONUSERBASE="$WORK/.local"
export PATH="$WORK/.local/bin:$PATH"

_pip_install() {
    python3 -c "import $1" 2>/dev/null && return 0
    echo "[INFO] Installation de '$2'..."
    pip install --user --no-cache-dir "$2" || {
        echo "[ERREUR] Impossible d'installer '$2'."
        echo "  Pré-installez depuis le nœud de login : bash src/slurm/setup_jeanzay.sh"
        exit 1
    }
}
_pip_install medvae  medvae
_pip_install nibabel nibabel
_pip_install einops  einops
_pip_install scipy   scipy

cd "$CODE_DIR"

python3 src/vae3d/train_medvae_disentangle_v1.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_DIR" \
  --splits retro_train \
  --modalities $MODALITIES \
  --fields $FIELDS \
  --batch-size 1 \
  --num-workers 16 \
  --steps 20000 \
  --save-every 200 \
  --print-every 20 \
  --use-amp \
  --amp-dtype bf16
