#!/usr/bin/env bash
# Convenience wrapper to submit unified VAE training jobs to Jean Zay.
#
# Usage examples:
#   bash src/slurm/submit_train_vae_jeanzay.sh aekl
#   bash src/slurm/submit_train_vae_jeanzay.sh medvae medvae_t1w
#   bash src/slurm/submit_train_vae_jeanzay.sh vqvae vqvae_full "T1W T2W T2FLAIR" "0.1T 1.5T 3T 5T 7T"

set -euo pipefail

VAE_TYPE="${1:-}"
RUN_NAME="${2:-${VAE_TYPE}_jeanzay}"
MODALITIES="${3:-T1W T2W T2FLAIR}"
FIELDS="${4:-0.1T 1.5T 3T 5T 7T}"

if [[ -z "$VAE_TYPE" ]]; then
  echo "Usage: $0 <aekl|vqvae|medvae> [run_name] [\"modalities\"] [\"fields\"]"
  exit 1
fi

case "$VAE_TYPE" in
  aekl|vqvae|medvae) ;;
  *)
    echo "[ERREUR] type invalide: $VAE_TYPE"
    echo "Types acceptés: aekl | vqvae | medvae"
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/train_vae_jeanzay.slurm"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "[ERREUR] script introuvable: $SLURM_SCRIPT"
  exit 1
fi

echo "Soumission: VAE=$VAE_TYPE RUN=$RUN_NAME"
if [[ "$VAE_TYPE" == "vqvae" ]]; then
  echo "  Modalités: $MODALITIES"
  echo "  Fields   : $FIELDS"
fi

sbatch "$SLURM_SCRIPT" "$VAE_TYPE" "$RUN_NAME" "$MODALITIES" "$FIELDS"
