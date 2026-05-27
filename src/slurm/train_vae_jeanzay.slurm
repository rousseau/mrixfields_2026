#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Script SLURM — Entraînement VAE 3D sur Jean Zay (H100 SXM5 80GB)
#
# Partition gpu_p6 : H100 SXM5 80GB, 1 GPU/nœud (suffisant pour VAE)
# QOS : qos_gpu_h100-t4 (100h max)
#
# Usage :
#   sbatch src/slurm/train_vae_jeanzay.slurm <VAE_TYPE> [CONFIG] [RUN_NAME]
#
# Arguments :
#   VAE_TYPE  : aekl | medvae | vqvae   (obligatoire)
#   CONFIG    : chemin config YAML       (défaut : configs/vae3d_multimodal.yaml)
#   RUN_NAME  : sous-répertoire de sortie (défaut : {VAE_TYPE}_multimodal_remote)
#
# Exemples :
#   sbatch src/slurm/train_vae_jeanzay.slurm aekl
#   sbatch src/slurm/train_vae_jeanzay.slurm aekl configs/vae3d_multimodal.yaml
#   sbatch src/slurm/train_vae_jeanzay.slurm medvae configs/vae3d_multimodal.yaml medvae_multimodal_v1
#
# Tuning via variables d'environnement :
#   STEPS=30000 BATCH_SIZE=2 LR=1e-4 sbatch src/slurm/train_vae_jeanzay.slurm aekl
#
# Prérequis (via setup_jeanzay.sh) :
#   $WORK/MRIX/mrixfields_2026/   ← ce repo
#   $WORK/MRIX/data/              ← données NIfTI
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=vae3d
#SBATCH -A crp@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=48:00:00
#SBATCH --output=logs/vae3d_%x_%j.out
#SBATCH --error=logs/vae3d_%x_%j.err

set -euo pipefail

VAE_TYPE="${1:-}"
CONFIG="${2:-configs/vae3d_multimodal.yaml}"
RUN_NAME="${3:-${VAE_TYPE}_multimodal_remote}"

STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-4}"

# ── Validation ───────────────────────────────────────────────────────────────
if [[ -z "$VAE_TYPE" ]]; then
    echo "[ERREUR] VAE_TYPE manquant. Valeurs : aekl | medvae | vqvae"
    exit 1
fi
case "$VAE_TYPE" in
    aekl|medvae|vqvae) ;;
    *)
        echo "[ERREUR] VAE_TYPE invalide : $VAE_TYPE  (aekl | medvae | vqvae)"
        exit 1
        ;;
esac

# ── Chemins ──────────────────────────────────────────────────────────────────
CODE_DIR="$WORK/MRIX/mrixfields_2026"
mkdir -p "$CODE_DIR/logs"
cd "$CODE_DIR"

# ── Modules ──────────────────────────────────────────────────────────────────
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

export PYTHONUSERBASE="$WORK/.local"
export PATH="$WORK/.local/bin:$PATH"
export TORCH_HOME="$WORK/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="$CODE_DIR/src"

# ── Dépendances optionnelles ─────────────────────────────────────────────────
_install_if_missing() {
    python3 -c "import $1" >/dev/null 2>&1 || pip install --user --quiet "$2"
}
_install_if_missing nibabel nibabel
_install_if_missing scipy scipy
_install_if_missing einops einops

if [[ "$VAE_TYPE" == "medvae" ]]; then
    python3 -c "import medvae" >/dev/null 2>&1 || {
        echo "[ERREUR] Package 'medvae' introuvable."
        echo "Installez-le sur le nœud de connexion :"
        echo "  pip install --user git+https://github.com/StanfordMIMI/MedVAE.git"
        exit 1
    }
fi

# ── Résumé ───────────────────────────────────────────────────────────────────
echo "======================================================================="
echo " Job ID    : ${SLURM_JOB_ID:-N/A}"
echo " Nœud      : ${SLURM_JOB_NODELIST:-N/A}"
echo " VAE type  : $VAE_TYPE"
echo " Config    : $CONFIG"
echo " Run name  : $RUN_NAME"
echo " Env       : remote (configs/env/remote.yaml)"
echo " Steps     : $STEPS"
echo " Batch     : $BATCH_SIZE"
echo " LR        : $LR"
echo "======================================================================="

# ── Lancement ────────────────────────────────────────────────────────────────
python3 src/vae3d/train_vae_3d.py \
    --config "$CONFIG" \
    --env remote \
    --run-name "$RUN_NAME" \
    --vae-type "$VAE_TYPE" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR"

echo "======================================================================="
echo " Entraînement terminé : $VAE_TYPE / $RUN_NAME"
echo "======================================================================="
