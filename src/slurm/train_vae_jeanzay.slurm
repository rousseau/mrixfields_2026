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
#   VAE_TYPE  : aekl | medvae | medvae_finetune | vqvae | pythae_vae | pythae_vqvae | pythae_rhvae  (obligatoire)
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
    echo "[ERREUR] VAE_TYPE manquant. Valeurs : aekl | medvae | vqvae | pythae_vae | pythae_vqvae | pythae_rhvae"
    exit 1
fi
case "$VAE_TYPE" in
    aekl|medvae|medvae_finetune|vqvae|pythae_vae|pythae_vqvae|pythae_rhvae) ;;
    *)
        echo "[ERREUR] VAE_TYPE invalide : $VAE_TYPE  (aekl | medvae | medvae_finetune | vqvae | pythae_vae | pythae_vqvae | pythae_rhvae)"
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

# ── Vérification des dépendances Python ─────────────────────────────────────
# Les nœuds de calcul Jean Zay n'ont PAS accès à internet.
# Toutes les dépendances doivent être pré-installées depuis le nœud de login :
#   bash src/slurm/setup_jeanzay.sh
_check_import() {
    # $1 = module Python à importer, $2 = paquet pip correspondant
    python3 -c "import $1" 2>/dev/null && return 0
    echo "[ERREUR] Package '$1' introuvable dans \$WORK/.local."
    echo "  Pré-installez depuis le nœud de login : bash src/slurm/setup_jeanzay.sh"
    exit 1
}
_check_import nibabel nibabel
_check_import scipy   scipy
_check_import einops  einops
_check_import monai   "monai[all]"

# Vérifications conditionnelles selon VAE_TYPE
case "$VAE_TYPE" in
    pythae_vae|pythae_vqvae|pythae_rhvae)
        _check_import pythae pythae
        ;;
    medvae|medvae_finetune)
        _check_import medvae medvae
        # Vérifier que les poids MedVAE sont en cache HuggingFace local
        python3 -c "
import os; os.environ['HF_HUB_OFFLINE'] = '1'
try:
    from medvae import MVAE; MVAE(model_name='medvae_4_1_3d', modality='mri')
    print('[OK] Poids MedVAE disponibles dans le cache HuggingFace')
except Exception as e:
    print(f'[WARN] Poids MedVAE non mis en cache : {e}')
    print('  Sur le nœud de login, exécuter :')
    print('  python3 -c \"from medvae import MVAE; MVAE(model_name=\\\"medvae_4_1_3d\\\", modality=\\\"mri\\\")\"')
"
        ;;
esac

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
# Routing vers le script d'entraînement selon le type de VAE
if [[ "$VAE_TYPE" == "pythae_vae" ]]; then
    TRAIN_SCRIPT="src/vae3d/train_pythae_vae.py"
elif [[ "$VAE_TYPE" == "pythae_vqvae" ]]; then
    TRAIN_SCRIPT="src/vae3d/train_pythae_vqvae.py"
elif [[ "$VAE_TYPE" == "pythae_rhvae" ]]; then
    TRAIN_SCRIPT="src/vae3d/train_pythae_rhvae.py"
elif [[ "$VAE_TYPE" == "medvae_finetune" ]]; then
    TRAIN_SCRIPT="src/vae3d/train_maisi_finetune.py"
else
    TRAIN_SCRIPT="src/vae3d/train_vae_3d.py"
fi

python3 "$TRAIN_SCRIPT" \
    --config "$CONFIG" \
    --env remote \
    --run-name "$RUN_NAME" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR"

echo "======================================================================="
echo " Entraînement terminé : $VAE_TYPE / $RUN_NAME"
echo "======================================================================="
