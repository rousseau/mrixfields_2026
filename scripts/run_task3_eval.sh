#!/bin/bash
# Inference Task 3 + evaluation officielle pour une variante MMFM, sur les
# TROIS contrastes. Une marche de l'escalier experimental = un appel a ce script.
#
# Motif : chaque variante etait jusqu'ici evaluee a la main, ce qui ouvre la
# porte a un ecart de protocole entre deux runs qu'on croit comparables. Le
# projet a deja paye ce genre d'erreur (selection non stratifiee, volume de
# controle unique, EMA non chauffee). Un seul chemin, parametre.
#
# Usage :
#   bash scripts/run_task3_eval.sh <config.yaml> <method> <tag> [outdir]
# Exemple :
#   bash scripts/run_task3_eval.sh configs/mmfm/vectorized_r1_time.yaml \
#        mmfm3d_vectorized vec_r1_time results/mmfm/staircase_20260827
set -euo pipefail

CONFIG=${1:?config manquante}
METHOD=${2:?method manquante}
TAG=${3:?tag manquant}
OUTDIR=${4:-results/mmfm/staircase_20260827}

cd "$(dirname "$0")/.."
export PYTHONPATH=src

SUBDIR=$(python -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['data']['output_subdir'])")
CKPT="outputs/$SUBDIR/weights/model_final.pth"
PRED="outputs/$SUBDIR/predictions/task3"

[ -f "$CKPT" ] || { echo "checkpoint absent : $CKPT"; exit 1; }
mkdir -p "$OUTDIR"

echo "=== $TAG ==="
echo "config     : $CONFIG"
echo "checkpoint : $CKPT"

echo "--- inference (3 contrastes, 20 paires, 3 sujets) ---"
python src/cfm/infer_mmfm_unified.py \
    --config "$CONFIG" --checkpoint "$CKPT" \
    --output_dir "outputs/$SUBDIR/predictions" \
    --split Training_prospective \
    --modalities T1W T2W T2FLAIR \
    --field_norm_stats configs/mmfm/field_norm_stats.json \
    --skip_existing

echo "--- evaluation officielle ---"
for M in T1W T2W T2FLAIR; do
    python src/evaluation/evaluate.py \
        --method mmfm_v2 --task task3 --modality "$M" \
        --pred-dir "$PRED/$M" \
        --output-csv "$OUTDIR/task3_${TAG}_${M}.csv"
done

echo "--- resume ---"
python - "$OUTDIR" "$TAG" <<'PY'
import csv, sys
outdir, tag = sys.argv[1], sys.argv[2]
tot = []
for m in ("T1W", "T2W", "T2FLAIR"):
    rows = list(csv.DictReader(open(f"{outdir}/task3_{tag}_{m}.csv")))
    v = [float(r["nrmse_mean"]) for r in rows]
    s = [float(r["ssim_mean"]) for r in rows]
    l = [float(r["lpips_mean"]) for r in rows]
    tot += v
    print(f"  {m:8s} nRMSE {sum(v)/len(v):.4f}  SSIM {sum(s)/len(s):.4f}  "
          f"LPIPS {sum(l)/len(l):.4f}  ({len(v)} paires)")
print(f"  {'moyenne':8s} nRMSE {sum(tot)/len(tot):.4f}")
PY
