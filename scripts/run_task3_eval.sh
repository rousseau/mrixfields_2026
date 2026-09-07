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
#   bash scripts/run_task3_eval.sh <config.yaml> <method> <tag> [outdir] \
#        [field_norm_stats.json] [8win|cc]
# Exemple :
#   bash scripts/run_task3_eval.sh configs/mmfm/vectorized_r1_time.yaml \
#        mmfm3d_vectorized vec_r1_time results/mmfm/staircase_20260827
#
# TROIS CORRECTIFS DU 2026-09-07, tous mesures :
#
#  1. LA GEOMETRIE EST DESORMAIS EXPLICITE (6e argument), elle etait implicite.
#     `--center_crop_only` etait declare `action="store_true"` sans `default=None` et
#     n'etait lu par aucune cle de config : il fallait le taper, et ce pilote ne le
#     tapait pas. Consequence, mesuree sur l'ecart median entre fichiers de prediction :
#     104 s/volume pour les 15 runs passes par ce pilote (vec_rbest, vec_r1_time,
#     vec_batch8, vec_lpips, valid_rbest, ceiling_*, calib_*, ablations n_steps) contre
#     9-34 s/volume pour les 15 autres. Deux geometries comparees dans le meme tableau,
#     sans que rien ne le dise.
#
#     CE QUE LA MESURE A TRANCHE (results/mmfm/geometry_20260907/, R-best, 180 volumes,
#     seule la geometrie change) :
#       - region FULL, celle des chiffres publies : ecart +0.0005, 26/60 paires,
#         p = 0.37. INDISCERNABLE. Le tableau de reference historique n'est donc PAS
#         corrompu, et l'avance de R-best sur la production (0.0057) tient.
#       - region SLAB, celle que le classement note reellement : ecart +0.0034,
#         12/60 paires, p = 3.2e-06. Les 8 fenetres sont MEILLEURES, de facon
#         consistante, et de plus que le plancher de bruit (0.002) — plus meme que
#         AdaGN (0.0006), le flip (0.0001), l'ordre 3 (0.0031) ou les 4 correctifs du
#         flow (0.0031).
#
#     Le defaut reste donc `8win` : moyenner 8 generations decalees est une reduction
#     de variance qui paie sur la region notee. Ce n'est plus un accident, c'est un
#     choix, et il coute 6.5x le temps machine. `cc` sert a comparer aux chiffres
#     anterieurs au 2026-08-27 et va 6.5x plus vite.
#
#  2. Le chemin du JSON de normalisation etait CODE EN DUR sur
#     `configs/mmfm/field_norm_stats.json`. Il sert a normaliser la source ET a
#     denormaliser la sortie avec les stats du champ CIBLE : evaluer un modele
#     entraine sous une autre table produit une sortie a la mauvaise echelle, et le
#     run est perdu. Il est maintenant le 5e argument.
#
#  3. La geometrie et la table utilisees sont IMPRIMEES en tete de run.
set -euo pipefail

CONFIG=${1:?config manquante}
METHOD=${2:?method manquante}
TAG=${3:?tag manquant}
OUTDIR=${4:-results/mmfm/staircase_20260827}
FIELD_NORM_STATS=${5:-configs/mmfm/field_norm_stats.json}
GEOMETRY=${6:-8win}          # 8win (defaut, meilleur sur la region notee) | cc

case "$GEOMETRY" in
  8win) CROP_FLAG=() ;;
  cc)   CROP_FLAG=(--center_crop_only) ;;
  *)    echo "GEOMETRY doit valoir '8win' ou 'cc', pas '$GEOMETRY'"; exit 1 ;;
esac

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
if [ "$GEOMETRY" = "cc" ]; then GEO_DESC="un seul crop centre, comme le cache"
else GEO_DESC="8 fenetres decalees fusionnees par Hann"; fi
echo "geometrie        : $GEOMETRY  ($GEO_DESC)"

echo "--- inference (3 contrastes, 20 paires, 3 sujets) ---"
python src/cfm/infer_mmfm_unified.py \
    --config "$CONFIG" --checkpoint "$CKPT" \
    --output_dir "outputs/$SUBDIR/predictions" \
    --split Training_prospective \
    --modalities T1W T2W T2FLAIR \
    --field_norm_stats "$FIELD_NORM_STATS" \
    "${CROP_FLAG[@]}" \
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
