#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# launch_cfm3d_dgx.sh — Lancement local multi-GPU (DGX Station, debug)
#
# Usage :
#   bash src/launch_cfm3d_dgx.sh <PHASE> <MODALITY> [N_GPUS] [CONFIG_OVERRIDE]
#
# Arguments :
#   PHASE    : vae | cfm
#   MODALITY : T1W  (pour l'instant)
#   N_GPUS   : nombre de GPUs à utiliser (défaut : auto-détecté)
#   CONFIG   : chemin alternatif vers un config YAML (optionnel)
#
# Exemples :
#   bash src/launch_cfm3d_dgx.sh vae T1W          # auto-détection GPUs
#   bash src/launch_cfm3d_dgx.sh vae T1W 1        # single-GPU, débug rapide
#   bash src/launch_cfm3d_dgx.sh cfm T1W 4        # 4 GPUs, entraînement complet
#   bash src/launch_cfm3d_dgx.sh cfm T1W 4 configs/cfm3d_latent_T1W_debug.yaml
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

ENV_FILE="configs/env/dgx.yaml"
PYTHON=$(python3 -c "
import yaml
with open('$ENV_FILE') as f: e = yaml.safe_load(f)
print(e.get('python', 'python3'))
" 2>/dev/null || echo "python3")

# ── Config selon la phase ────────────────────────────────────────────────────
if [[ "$PHASE" == "vae" ]]; then
    CONFIG="${CONFIG_OVERRIDE:-configs/vae3d_T1W.yaml}"
    SCRIPT="src/train_vae_3d.py"
elif [[ "$PHASE" == "cfm" ]]; then
    CONFIG="${CONFIG_OVERRIDE:-configs/cfm3d_latent_${MODALITY}.yaml}"
    SCRIPT="src/train_cfm_3d.py"
else
    echo "ERREUR : PHASE doit être 'vae' ou 'cfm' (reçu : '$PHASE')" >&2
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

# ── Résolution du dernier checkpoint (resume automatique) ───────────────────
OUTPUT_ROOT=$($PYTHON -c "
import yaml
with open('configs/env/dgx.yaml') as f: e = yaml.safe_load(f)
with open('$CONFIG') as f: c = yaml.safe_load(f)
subdir = c.get('data', {}).get('output_subdir', '')
print(e['output_root'] + '/' + subdir)
" 2>/dev/null || echo "")

RESUME_ARG=""
if [[ -n "$OUTPUT_ROOT" && -d "$OUTPUT_ROOT/weights" ]]; then
    LAST_CKPT=$(ls -t "$OUTPUT_ROOT/weights"/*.pth 2>/dev/null | head -1 || echo "")
    if [[ -n "$LAST_CKPT" ]]; then
        RESUME_ARG="--resume $LAST_CKPT"
        echo "  Reprise depuis : $LAST_CKPT"
    fi
fi

# ── Lancement ────────────────────────────────────────────────────────────────
if [[ "$N_GPUS" -gt 1 ]]; then
    echo "  torchrun --nproc_per_node=$N_GPUS $SCRIPT"
    torchrun \
        --nproc_per_node="$N_GPUS" \
        --master_port=$(( RANDOM % 10000 + 29500 )) \
        "$SCRIPT" \
        --config "$CONFIG" \
        --env dgx \
        $RESUME_ARG
else
    echo "  $PYTHON $SCRIPT (single-GPU)"
    $PYTHON "$SCRIPT" \
        --config "$CONFIG" \
        --env dgx \
        $RESUME_ARG
fi
