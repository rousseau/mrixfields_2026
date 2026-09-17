#!/bin/bash
# Evaluation officielle Task 3, UNE FOIS, sur le dossier de predictions
# PARTAGE rempli par les 3 replis LOO (scripts/run_task3_eval_loo.sh) --
# chaque sujet prospectif y a ete depose par le fine-tuning qui NE l'a PAS vu.
#
# Usage :
#   bash scripts/eval_task3_loo_aggregate.sh [shared_output_dir] [tag] [outdir]
set -euo pipefail

SHARED_OUT=${1:-outputs/mmfm/finetune_loo_aggregate/predictions}
TAG=${2:-finetune_loo}
OUTDIR=${3:-results/mmfm/finetune_loo_20260916}

cd "$(dirname "$0")/.."
export PYTHONPATH=src

PRED="$SHARED_OUT/task3"
mkdir -p "$OUTDIR"

echo "=== Évaluation officielle LOO agrégée ($TAG) ==="
echo "prédictions : $PRED"

for M in T1W T2W T2FLAIR; do
    python src/evaluation/evaluate.py \
        --method mmfm_v2 --task task3 --modality "$M" \
        --pred-dir "$PRED/$M" \
        --output-csv "$OUTDIR/task3_${TAG}_${M}.csv"
done

echo "--- résumé ---"
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
