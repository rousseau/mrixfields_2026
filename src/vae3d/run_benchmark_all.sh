#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_benchmark_all.sh — Benchmark complet 4 AE × 3 modalités × 5 champs
#
# Lance le benchmark de comparaison des 4 autoencodeurs :
#   - AEKL (AutoencoderKL Jean Zay, 200 epochs, T1W)
#   - VQ-VAE (NeuroQuant, 17 800 steps, T1W+T2W+T2FLAIR)
#   - MedVAE frozen (poids HuggingFace, medvae_4_1_3d)
#   - MedVAE fine-tuné (20 000 steps, T1W+T2W+T2FLAIR)
#
# Usage :
#   cd PROJECT_ROOT && bash src/vae3d/run_benchmark_all.sh [--samples N]
#
# Options :
#   --samples N   Nombre de volumes par combinaison [défaut: 2]
#
# Sorties :
#   results/benchmark/benchmark_{MODALITY}_{FIELD}.csv
#   results/benchmark/summary.txt
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MAX_SAMPLES=2
SKIP_MEDVAE_FROZEN=0

# ── CLI parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --samples)     MAX_SAMPLES="$2"; shift 2 ;;
        --skip-medvae-frozen) SKIP_MEDVAE_FROZEN=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Paths (relatifs à la racine du projet) ────────────────────────────────────
AEKL_CKPT="outputs/vae3d/runs/vae3d_T1W_jeanzay/weights/model_final.pth"
VQVAE_CKPT="outputs/vqvae3d/runs/vqvae_final/weights/model_best.pth"
MEDVAE_FT_CKPT="outputs/medvae/runs/medvae_finetune_all/weights/model_final.pth"
OUT_DIR="results/benchmark"

SCRIPT="src/vae3d/benchmark_vae.py"

mkdir -p "$OUT_DIR"

# ── Common args ───────────────────────────────────────────────────────────────
COMMON_ARGS="--aekl-ckpt $AEKL_CKPT \
             --vqvae-ckpt $VQVAE_CKPT \
             --medvae-finetuned-ckpt $MEDVAE_FT_CKPT \
             --output-dir $OUT_DIR \
             --max-samples $MAX_SAMPLES"

if [[ $SKIP_MEDVAE_FROZEN -eq 1 ]]; then
    COMMON_ARGS="$COMMON_ARGS --skip-medvae"
fi

echo "============================================================"
echo " MRIxFields VAE Benchmark"
echo " max-samples=$MAX_SAMPLES  skip-medvae-frozen=$SKIP_MEDVAE_FROZEN"
echo "============================================================"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# T1W — tous les champs (4 modèles)
# ─────────────────────────────────────────────────────────────────────────────
for FIELD in 0.1T 1.5T 3T 5T 7T; do
    echo "──── T1W / $FIELD ────────────────────────────────────────────"
    python $SCRIPT --modality T1W --field $FIELD $COMMON_ARGS
    echo ""
done

# ─────────────────────────────────────────────────────────────────────────────
# T2W — champ 3T représentatif (skip AEKL : non entraîné sur T2W)
# ─────────────────────────────────────────────────────────────────────────────
echo "──── T2W / 3T ────────────────────────────────────────────────"
python $SCRIPT --modality T2W --field 3T --skip-aekl $COMMON_ARGS
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# T2FLAIR — champ 3T représentatif (skip AEKL : non entraîné sur T2FLAIR)
# ─────────────────────────────────────────────────────────────────────────────
echo "──── T2FLAIR / 3T ────────────────────────────────────────────"
python $SCRIPT --modality T2FLAIR --field 3T --skip-aekl $COMMON_ARGS
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Résumé agrégé
# ─────────────────────────────────────────────────────────────────────────────
SUMMARY="$OUT_DIR/summary.txt"
echo "===========================================================" > "$SUMMARY"
echo " VAE Benchmark — Summary" >> "$SUMMARY"
echo " Date: $(date)" >> "$SUMMARY"
echo " max-samples=$MAX_SAMPLES" >> "$SUMMARY"
echo "===========================================================" >> "$SUMMARY"
echo "" >> "$SUMMARY"

for CSV in "$OUT_DIR"/benchmark_*.csv; do
    BNAME=$(basename "$CSV" .csv)
    echo "── $BNAME ──" >> "$SUMMARY"
    cat "$CSV" >> "$SUMMARY"
    echo "" >> "$SUMMARY"
done

echo "Summary written to: $SUMMARY"
echo "All benchmarks done."
