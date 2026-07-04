#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_cfm3d_dgx.sh — Lancement local DGX GB10 (single-GPU or multi-GPU)
#
# Usage :
#   bash src/launch_cfm3d_dgx.sh <PHASE> <MODALITY> [N_GPUS] [CONFIG_OVERRIDE]
#
# Arguments :
#   PHASE    : vae | cfm | mmfm
#   MODALITY : T1W  (pour l'instant)
#   N_GPUS   : nombre de GPUs à utiliser (défaut : auto-détecté)
#   CONFIG   : chemin alternatif vers un config YAML (optionnel)
#
# Exemples :
#   bash src/launch_cfm3d_dgx.sh vae T1W          # auto-détection GPUs
#   bash src/launch_cfm3d_dgx.sh vae T1W 1        # single-GPU, débug rapide
#   bash src/slurm/launch_cfm3d_dgx.sh cfm T1W 4  # multi-GPU sur DGX GB10
#   bash src/slurm/launch_cfm3d_dgx.sh mmfm T1W 4 configs/mmfm3d_medvae_multimodal.yaml
#   bash src/slurm/launch_cfm3d_dgx.sh mmfm_unet T1W 4 configs/mmfm3d_unet_medvae_multimodal.yaml
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PHASE="${1:-cfm}"
MODALITY="${2:-T1W}"
N_GPUS="${3:-}"         # vide = auto-détection
CONFIG_OVERRIDE="${4:-}"

# ── Résolution des chemins ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

ENV_FILE="configs/env/local.yaml"
PYTHON=$(python3 -c "
import yaml
with open('$ENV_FILE') as f: e = yaml.safe_load(f)
print(e.get('python', 'python3'))
" 2>/dev/null || echo "python3")

# ── Config selon la phase ────────────────────────────────────────────────────
if [[ "$PHASE" == "vae" ]]; then
    CONFIG="${CONFIG_OVERRIDE:-configs/vae3d_multimodal.yaml}"
    SCRIPT="src/vae3d/train_vae_3d.py"
elif [[ "$PHASE" == "cfm" ]]; then
    CONFIG="${CONFIG_OVERRIDE:-configs/cfm3d_T1W_medvae_finetuned.yaml}"
    SCRIPT="src/cfm/train_cfm_3d.py"
elif [[ "$PHASE" == "mmfm" ]]; then
    CONFIG="${CONFIG_OVERRIDE:-configs/mmfm3d_medvae_multimodal.yaml}"
    SCRIPT="src/cfm/train_mmfm_3d.py"
elif [[ "$PHASE" == "mmfm_unet" ]]; then
    CONFIG="${CONFIG_OVERRIDE:-configs/mmfm3d_unet_medvae_multimodal.yaml}"
    SCRIPT="src/cfm/train_mmfm_unet_3d.py"
else
    echo "ERREUR : PHASE doit être 'vae', 'cfm', 'mmfm' ou 'mmfm_unet' (reçu : '$PHASE')" >&2
    exit 1
fi

# ── Auto-détection du nombre de GPUs ────────────────────────────────────────
if [[ -z "$N_GPUS" ]]; then
    N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
    if [[ "$N_GPUS" -eq 0 ]]; then
        N_GPUS=1
        echo "[WARN] Aucun GPU CUDA détecté, lancement CPU (débug uniquement)"
    fi
fi
echo "Lancement : PHASE=$PHASE | MODALITY=$MODALITY | N_GPUS=$N_GPUS | CONFIG=$CONFIG"
if [[ "$N_GPUS" -gt 1 ]]; then
    echo "  Multi-GPU DDP : batch effectif = $((2 * N_GPUS)) (batch_size=2/GPU × $N_GPUS GPUs)"
fi

# ── Résolution du dernier checkpoint (resume automatique) ───────────────────
OUTPUT_ROOT=$($PYTHON -c "
import yaml
with open('configs/env/local.yaml') as f: e = yaml.safe_load(f)
with open('$CONFIG') as f: c = yaml.safe_load(f)
subdir = c.get('data', {}).get('output_subdir', '')
print(e['output_root'] + '/' + subdir)
" 2>/dev/null || echo "")

RESUME_ARG=""
if [[ -n "$OUTPUT_ROOT" && -d "$OUTPUT_ROOT/weights" ]]; then
    LAST_CKPT=""
    for preferred in model_best.pth model_final.pth; do
        if [[ -f "$OUTPUT_ROOT/weights/$preferred" ]]; then
            LAST_CKPT="$OUTPUT_ROOT/weights/$preferred"
            break
        fi
    done
    if [[ -z "$LAST_CKPT" ]]; then
        LAST_CKPT=$(ls -t "$OUTPUT_ROOT/weights"/checkpoint_*.pth 2>/dev/null | head -1 || echo "")
    fi
    if [[ -n "$LAST_CKPT" ]]; then
        RESUME_ARG="--resume $LAST_CKPT"
        echo "  Reprise depuis : $LAST_CKPT"
    fi
fi

# ── Lancement avec limite de 24h ─────────────────────────────────────────────
MAX_HOURS="${MAX_HOURS:-24}"
MAX_SECONDS=$((MAX_HOURS * 3600))
echo "  Durée max : ${MAX_HOURS}h (MAX_HOURS=${MAX_HOURS})"

if [[ "$N_GPUS" -gt 1 ]]; then
    echo "  torchrun --nproc_per_node=$N_GPUS $SCRIPT"
    timeout --signal=TERM --kill-after=120s "${MAX_SECONDS}s" \
        torchrun \
            --nproc_per_node="$N_GPUS" \
            --master_port=$(( RANDOM % 10000 + 29500 )) \
            "$SCRIPT" \
            --config "$CONFIG" \
            --env local \
            $RESUME_ARG
else
    echo "  $PYTHON $SCRIPT (single-GPU)"
    timeout --signal=TERM --kill-after=120s "${MAX_SECONDS}s" \
        $PYTHON "$SCRIPT" \
            --config "$CONFIG" \
            --env local \
            $RESUME_ARG
fi
