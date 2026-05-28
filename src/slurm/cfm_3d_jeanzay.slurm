#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Script SLURM — Entraînement VAE 3D + CFM/MMFM sur Jean Zay (H100 SXM5 80GB)
#
# Partition gpu_p6 : H100 SXM5 80GB, 4 GPUs/nœud
# QOS : qos_gpu_h100-t4 (100h max)
#
# Lancement :
#   # Phase 2 — VAE 3D pré-entraînement
#   sbatch src/slurm/cfm_3d_jeanzay.slurm vae T1W
#
#   # Phase 3 — OT-CFM 3D en espace latent
#   sbatch src/slurm/cfm_3d_jeanzay.slurm cfm T1W configs/cfm3d_T1W_medvae_finetuned.yaml
#   # Variante bidomaine 0.1T↔7T uniquement :
#   sbatch src/slurm/cfm_3d_jeanzay.slurm cfm T1W configs/cfm3d_T1W_medvae_0p1T_7T.yaml
#
#   # Baseline MMFM v1 vectorisée sur latent MedVAE aplati
#   sbatch src/slurm/cfm_3d_jeanzay.slurm mmfm T1W configs/mmfm3d_medvae_multimodal.yaml
#
#   # MMFM-UNet 3D spatial sur latent MedVAE
#   sbatch src/slurm/cfm_3d_jeanzay.slurm mmfm_unet T1W configs/mmfm3d_unet_medvae_multimodal.yaml
#
# Prérequis (via setup_jeanzay.sh) :
#   $WORK/MRIX/mrixfields_2026/   ← ce repo
#   $WORK/MRIX/challenge/         ← code officiel du challenge
#   $WORK/MRIX/data/              ← données NIfTI
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=cfm3d
# SBATCH_ACCOUNT doit être défini dans l'environnement avant sbatch :
#   export SBATCH_ACCOUNT=<projet>@h100   (ex: crp@h100)
# SLURM lit automatiquement cette variable — ne pas hardcoder ici.
#SBATCH -A crp@h100
#SBATCH -C h100

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1               # torchrun gère les 4 sous-processus GPU
#SBATCH --cpus-per-task=64       # 16 CPUs × 4 GPUs (workers DataLoader)

#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=100:00:00

#SBATCH --output=logs/cfm3d_%x_%j.out
#SBATCH --error=logs/cfm3d_%x_%j.err

# Signal 120s avant la fin pour auto-requeue
#SBATCH --signal=USR1@120

set -e

PHASE="${1:-cfm}"          # "vae", "cfm", "mmfm" ou "mmfm_unet"
MODALITY="${2:-T1W}"
MODALITY="${MODALITY^^}"   # normalisation en majuscules (T1w → T1W)
CONFIG_ARG="${3:-}"
CODE_DIR="$WORK/MRIX/mrixfields_2026"
mkdir -p "$CODE_DIR/logs"

echo "======================================================================="
echo " Job ID    : $SLURM_JOB_ID"
echo " Nœud      : $SLURM_JOB_NODELIST"
echo " Phase     : $PHASE"
echo " Modalité  : $MODALITY"
echo " Code      : $CODE_DIR"
echo "======================================================================="

# ─── Modules Jean Zay ────────────────────────────────────────────────────────
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

# ─── Dépendances utilisateur ─────────────────────────────────────────────────
export PYTHONUSERBASE="$WORK/.local"
export PATH="$WORK/.local/bin:$PATH"

# Installation via pip --user --no-cache-dir (fonctionne depuis les nœuds de
# Les nœuds de calcul Jean Zay n'ont PAS accès à internet.
# Toutes les dépendances doivent être pré-installées : bash src/slurm/setup_jeanzay.sh
_check_import() {
    python3 -c "import $1" 2>/dev/null && return 0
    echo "[ERREUR] Package '$1' introuvable. Pré-installez : bash src/slurm/setup_jeanzay.sh"
    exit 1
}
_check_import torchcfm
_check_import ot
_check_import torchdiffeq
_check_import nibabel
_check_import monai
_check_import medvae

# Vérifie explicitement la disponibilité des modèles generative
_ensure_monai_generative_models() {
    if python3 - <<'PY' >/dev/null 2>&1
try:
    from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
except Exception:
    try:
        from monai.generative.networks.nets import AutoencoderKL, DiffusionModelUNet
    except Exception:
        try:
            from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
        except Exception:
            raise SystemExit(1)
raise SystemExit(0)
PY
    then
        echo "[INFO] AutoencoderKL/DiffusionModelUNet disponibles."
    else
        echo "[ERREUR] AutoencoderKL/DiffusionModelUNet introuvables dans l'environnement." >&2
        echo "         Pré-installez depuis le nœud de LOGIN : bash src/slurm/setup_jeanzay.sh" >&2
        exit 1
    fi
}
_ensure_monai_generative_models

# Package challenge (mrixfields) — doit être pré-installé via setup_jeanzay.sh
_check_import mrixfields

# ─── Cache PyTorch / HuggingFace dans $WORK (évite de saturer le quota HOME) ─
export TORCH_HOME="$WORK/.cache/torch"
export HF_HOME="$WORK/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1

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
    exit 1
fi
N_NIFTI=$(find "$DATA_PATH" -name "*.nii.gz" | wc -l)
echo "Données NIfTI  : OK ($N_NIFTI volumes dans $DATA_PATH)"
echo "GPU            : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ─── Sélection de la config ───────────────────────────────────────────────────
cd "$CODE_DIR"

if [ "$PHASE" = "vae" ]; then
    SCRIPT="src/vae3d/train_vae_3d.py"
    if [ -n "$CONFIG_ARG" ]; then
        CONFIG="$CONFIG_ARG"
    else
        CONFIG="configs/vae3d_multimodal.yaml"
    fi
elif [ "$PHASE" = "cfm" ]; then
    SCRIPT="src/cfm/train_cfm_3d.py"
    if [ -n "$CONFIG_ARG" ]; then
        CONFIG="$CONFIG_ARG"
    else
        CONFIG="configs/cfm3d_T1W_medvae_finetuned.yaml"
    fi
elif [ "$PHASE" = "mmfm" ]; then
    SCRIPT="src/cfm/train_mmfm_3d.py"
    if [ -n "$CONFIG_ARG" ]; then
        CONFIG="$CONFIG_ARG"
    else
        CONFIG="configs/mmfm3d_medvae_multimodal.yaml"
    fi
elif [ "$PHASE" = "mmfm_unet" ]; then
    SCRIPT="src/cfm/train_mmfm_unet_3d.py"
    if [ -n "$CONFIG_ARG" ]; then
        CONFIG="$CONFIG_ARG"
    else
        CONFIG="configs/mmfm3d_unet_medvae_multimodal.yaml"
    fi
else
    echo "[ERREUR] Phase inconnue : '$PHASE'. Utiliser 'vae', 'cfm', 'mmfm' ou 'mmfm_unet'."
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "[ERREUR] Config introuvable : $CONFIG"
    exit 1
fi

echo "Script         : $SCRIPT"
echo "Config         : $CONFIG"

# ─── Détection du dernier checkpoint pour reprise ────────────────────────────
# Lire output_subdir depuis la config YAML
OUTPUT_SUBDIR=$(python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
print(c.get('data', {}).get('output_subdir', ''))
" 2>/dev/null || echo "")

WEIGHTS_DIR="$WORK/MRIX/mrixfields_2026/outputs/${OUTPUT_SUBDIR}/weights"
RESUME_FLAG=""

if [ -d "$WEIGHTS_DIR" ]; then
    # Cherche le checkpoint le plus récent (checkpoint_XXXXX.pth)
    LAST_CKPT=$(ls -t "${WEIGHTS_DIR}"/checkpoint_*.pth "${WEIGHTS_DIR}"/model_final.pth 2>/dev/null | head -1 || true)
    if [ -n "$LAST_CKPT" ]; then
        RESUME_FLAG="--resume $LAST_CKPT"
        echo "Reprise depuis  : $LAST_CKPT"
    else
        echo "Aucun checkpoint trouvé — démarrage à zéro."
    fi
else
    echo "Aucun répertoire weights — démarrage à zéro."
fi

echo "Output dir     : $WEIGHTS_DIR"
echo "Env            : jeanzay"
echo "Config CFM/MMFM: $CONFIG"

# Optimisation DataLoader : 16 workers/GPU × 4 GPUs = 64 CPUs utilisés
export DATALOADER_NUM_WORKERS=16

# ─── Lancement multi-GPU (torchrun DDP 4×H100) ───────────────────────────────
echo "======================================================================="
echo " Démarrage torchrun (4 GPUs H100)"
echo "======================================================================="

MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port="$MASTER_PORT" \
    "$SCRIPT" \
    --config "$CONFIG" \
    --env remote \
    $RESUME_FLAG

echo "======================================================================="
echo " Job terminé (ou interrompu pour requeue)"
echo "======================================================================="
