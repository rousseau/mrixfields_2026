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
#   bash scripts/run_task3_eval.sh <config.yaml> <method> <tag> [outdir] [field_norm_stats.json]
# Exemple :
#   bash scripts/run_task3_eval.sh configs/mmfm/vectorized_r1_time.yaml \
#        mmfm3d_vectorized vec_r1_time results/mmfm/staircase_20260827
#
# DEUX CORRECTIFS DU 2026-09-07, tous deux mesures :
#
#  1. `--center_crop_only` est desormais passe. Sans lui, `infer_mmfm_unified.py`
#     decoupe 8 fenetres decalees de -16 a +7 voxels et les fusionne par Hann, alors
#     que TOUT le cache de latents a ete encode sur un crop centre unique. Mesure sur
#     l'ecart median entre fichiers de prediction : 104.0 s/volume pour les runs passes
#     par ce pilote (vec_rbest, vec_lpips, vec_batch8, ceiling_vectorized,
#     ceiling_lpips) contre 16-17 s/volume pour ceux en crop centre (production,
#     compare_20260905). Rapport 6.3x, soit exactement 8 passes. Les deux familles
#     etaient comparees dans le meme tableau. Voir CHANGELOG 2026-09-07 (soir).
#
#  2. Le chemin du JSON de normalisation etait CODE EN DUR sur
#     `configs/mmfm/field_norm_stats.json`. Il sert a normaliser la source ET a
#     denormaliser la sortie avec les stats du champ CIBLE : evaluer un modele
#     entraine sous une autre table produit une sortie a la mauvaise echelle, et le
#     run est perdu. Il est maintenant le 5e argument.
set -euo pipefail

CONFIG=${1:?config manquante}
METHOD=${2:?method manquante}
TAG=${3:?tag manquant}
OUTDIR=${4:-results/mmfm/staircase_20260827}
FIELD_NORM_STATS=${5:-configs/mmfm/field_norm_stats.json}

cd "$(dirname "$0")/.."
export PYTHONPATH=src

SUBDIR=$(python -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['data']['output_subdir'])")
CKPT="outputs/$SUBDIR/weights/model_final.pth"
PRED="outputs/$SUBDIR/predictions/task3"

[ -f "$CKPT" ] || { echo "checkpoint absent : $CKPT"; exit 1; }
mkdir -p "$OUTDIR"

[ -f "$FIELD_NORM_STATS" ] || { echo "field_norm_stats absent : $FIELD_NORM_STATS"; exit 1; }

echo "=== $TAG ==="
echo "config           : $CONFIG"
echo "checkpoint       : $CKPT"
echo "field_norm_stats : $FIELD_NORM_STATS"
echo "geometrie        : crop centre unique (--center_crop_only)"

echo "--- inference (3 contrastes, 20 paires, 3 sujets) ---"
python src/cfm/infer_mmfm_unified.py \
    --config "$CONFIG" --checkpoint "$CKPT" \
    --output_dir "outputs/$SUBDIR/predictions" \
    --split Training_prospective \
    --modalities T1W T2W T2FLAIR \
    --field_norm_stats "$FIELD_NORM_STATS" \
    --center_crop_only \
    --skip_existing

# ATTENTION : `--skip_existing` ne recalcule rien si les predictions existent deja.
# Les predictions produites AVANT le 2026-09-07 le sont en 8 fenetres ; pour re-mesurer
# une variante historique sous la geometrie correcte, il faut supprimer
# outputs/$SUBDIR/predictions au prealable.

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
