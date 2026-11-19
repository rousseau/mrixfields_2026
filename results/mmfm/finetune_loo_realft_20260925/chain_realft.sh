#!/bin/bash
# Plan 2x2 (2026-09-25) : recette LOO 0.2843 en VRAI fine-tuning (reprise CLI)
# depuis {production, GAN Palier 4}. La branche « depuis zero » = reference 0.2843.
# Par repli et par branche : 2 entrainements (T1W long+LPIPS 3750 it., T2FLAIR 5k MSE),
# diagnostic latent, selection par contraste (select_ckpt.py, regle verifiee), elagage
# des checkpoints non retenus (disque), inference 8win du seul sujet tenu a l'ecart.
set -euo pipefail
cd /home/rousseau/Exp/mrixfields_2026
export PYTHONPATH=src PYTHONUNBUFFERED=1
N=results/mmfm/finetune_loo_realft_20260925
NORM=configs/mmfm/field_norm_stats.json
declare -A BASE=([ftprod]=outputs/mmfm/vectorized_trajectory/weights/model_final.pth
                 [ftgan]=outputs/mmfm/vectorized_trajectory_gan/weights/model_final.pth)
declare -A T2W_SRC=([ftprod]=outputs/mmfm/vectorized_trajectory/predictions_8win_baseline_20260917/task3/T2W
                    [ftgan]=outputs/mmfm/vectorized_trajectory_gan/predictions_8win_palier4_20260923/task3/T2W)

train_checked() {  # $1 config, $2 base, $3 log
  python src/cfm/train_mmfm_unified.py --method mmfm3d_vectorized --config $1 --env local \
    --resume $2 --resume_weights_only > $3 2>&1 &
  local pid=$!
  until grep -q "Entraînement :" $3 2>/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "entrainement mort avant demarrage : $3"; exit 1; }
    sleep 5
  done
  grep -q "Reprise (poids seuls) depuis : $2" $3 || { kill $pid; echo "PAS DE REPRISE : $3"; exit 1; }
  wait $pid
}

for F in 0006 0007 0009; do
  for A in ftprod ftgan; do
    for V in long_lpips 5k; do
      C=configs/mmfm/vectorized_finetune_loo_excl${F}_${V}_${A}.yaml
      echo "[$(date)] train $F $A $V"
      train_checked $C ${BASE[$A]} logs/finetune_loo_excl${F}_${V}_${A}.out
      echo "[$(date)] diag $F $A $V"
      python src/cfm/diagnose_finetune_checkpoints.py --config $C --excl $F \
        --json-out $N/diag_excl${F}_${V}_${A}.json > logs/diag_excl${F}_${V}_${A}.out 2>&1
    done
    T1=$(python $N/select_ckpt.py $N/diag_excl${F}_long_lpips_${A}.json T1W)
    T2=$(python $N/select_ckpt.py $N/diag_excl${F}_5k_${A}.json T2FLAIR)
    [ -f "$T1" ] && [ -f "$T2" ] || { echo "selection invalide : [$T1] [$T2]"; exit 1; }
    echo "[$(date)] selection $F $A : T1W=$T1 T2FLAIR=$T2" | tee -a $N/selection.txt
    for V in long_lpips 5k; do
      for P in outputs/mmfm/finetune_loo_excl${F}_${V}_${A}/weights/checkpoint_*.pth; do
        [ "$P" = "$T1" ] || [ "$P" = "$T2" ] || rm -f "$P"
      done
    done
    PRED=outputs/mmfm/finetune_loo_${A}_8win/predictions
    for ITEM in "T1W $T1 long_lpips" "T2FLAIR $T2 5k"; do
      set -- $ITEM
      echo "[$(date)] infer $F $A $1"
      python src/cfm/infer_mmfm_unified.py --config configs/mmfm/vectorized_finetune_loo_excl${F}_${3}_${A}.yaml \
        --checkpoint $2 --output_dir $PRED --split Training_prospective --modalities $1 \
        --subjects $F --field_norm_stats $NORM --skip_existing --env local \
        >> logs/infer_loo_${A}_8win.log 2>&1
    done
  done
done
for A in ftprod ftgan; do
  PRED=outputs/mmfm/finetune_loo_${A}_8win/predictions
  mkdir -p $PRED/task3 && cp -r ${T2W_SRC[$A]} $PRED/task3/
  for M in T1W T2W T2FLAIR; do
    python src/evaluation/evaluate.py --method mmfm_v2 --task task3 --modality $M \
      --pred-dir $PRED/task3/$M --output-csv $N/task3_${A}_loo_8win_$M.csv
  done
done
python $N/compare_realft.py | tee $N/compare_realft.txt
echo "[$(date)] FIN"
