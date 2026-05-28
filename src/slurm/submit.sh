#!/usr/bin/env bash
# =============================================================================
# submit.sh — Wrapper sbatch avec nommage de job automatique
#
# Usage :
#   bash src/slurm/submit.sh <PHASE> <MODALITY> <CONFIG>
#
# Arguments :
#   PHASE    : vae | cfm | mmfm | mmfm_unet
#   MODALITY : T1W | T2W | T2FLAIR  (normalisé en majuscules)
#   CONFIG   : chemin vers le fichier YAML de config
#
# Exemples :
#   bash src/slurm/submit.sh mmfm_unet T1W configs/mmfm3d_unet_medvae_multimodal.yaml
#   bash src/slurm/submit.sh cfm       T1W configs/cfm3d_T1W_medvae.yaml
#   bash src/slurm/submit.sh mmfm      T1W configs/mmfm3d_medvae_multimodal.yaml
#
# Nom du job : dérivé de output_subdir dans le YAML, tronqué à 15 caractères.
# Logs       : logs/<job_name>_<SLURM_JOB_ID>.{out,err}
# =============================================================================
set -euo pipefail

PHASE="${1:-}"
MODALITY="${2:-}"
CONFIG="${3:-}"

if [[ -z "$PHASE" || -z "$MODALITY" || -z "$CONFIG" ]]; then
    echo "Usage: bash src/slurm/submit.sh <PHASE> <MODALITY> <CONFIG>"
    echo "       PHASE    : vae | cfm | mmfm | mmfm_unet"
    echo "       MODALITY : T1W | T2W | T2FLAIR"
    echo "       CONFIG   : chemin vers le YAML de config"
    exit 1
fi

# ── Résolution des chemins ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f "$CONFIG" ]]; then
    echo "[ERREUR] Config introuvable : $CONFIG"
    exit 1
fi

# ── Calcul du nom de job depuis output_subdir ───────────────────────────────
JOB_NAME=$(python3 - <<PYEOF
import yaml, re, sys
with open("$CONFIG") as f:
    cfg = yaml.safe_load(f)
subdir = cfg.get("data", {}).get("output_subdir", "run")
# Garder uniquement le nom du run (dernier composant du path)
slug = subdir.split("/")[-1]
# Remplacer tout caractère non alphanumérique par _
slug = re.sub(r"[^a-zA-Z0-9]", "_", slug)
# Tronquer à 15 caractères (limite SLURM)
print(slug[:15])
PYEOF
)

if [[ -z "$JOB_NAME" ]]; then
    JOB_NAME="${PHASE}_${MODALITY}"
fi

# ── Répertoire des logs ──────────────────────────────────────────────────────
mkdir -p "$PROJECT_ROOT/logs"

LOG_OUT="$PROJECT_ROOT/logs/${JOB_NAME}_%j.out"
LOG_ERR="$PROJECT_ROOT/logs/${JOB_NAME}_%j.err"

# ── Affichage du récapitulatif ───────────────────────────────────────────────
echo "======================================================================="
echo " submit.sh — Soumission Jean Zay"
echo "======================================================================="
echo "  Phase     : $PHASE"
echo "  Modalité  : ${MODALITY^^}"
echo "  Config    : $CONFIG"
echo "  Job name  : $JOB_NAME"
echo "  Log out   : $LOG_OUT"
echo "  Log err   : $LOG_ERR"
echo "======================================================================="

# ── Soumission ───────────────────────────────────────────────────────────────
JOB_ID=$(sbatch \
    --job-name="$JOB_NAME" \
    --output="$LOG_OUT" \
    --error="$LOG_ERR" \
    "$SCRIPT_DIR/cfm_3d_jeanzay.slurm" \
    "$PHASE" "${MODALITY^^}" "$CONFIG" \
    | grep -oP '\d+$')

echo "  Job soumis : $JOB_ID"
echo ""
echo "Suivre l'avancement :"
echo "  squeue -u \$USER"
echo "  tail -f logs/${JOB_NAME}_${JOB_ID}.out"
echo ""
echo "Synchroniser les résultats une fois terminé :"
echo "  bash src/slurm/sync_from_jeanzay.sh"
