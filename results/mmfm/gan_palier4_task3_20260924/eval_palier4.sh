#!/bin/bash
# Evaluation officielle (evaluate.py, region slab par defaut) des predictions
# 8win du Palier 4 : GAN (5000 iters) et temoin lambda_gan=0 (5000 iters).
set -euo pipefail
cd /home/rousseau/Exp/mrixfields_2026
export PYTHONPATH=src
OUT=results/mmfm/gan_palier4_task3_20260924
for M in T1W T2W T2FLAIR; do
  python src/evaluation/evaluate.py --method mmfm_v2 --task task3 --modality $M \
    --pred-dir outputs/mmfm/vectorized_trajectory_gan/predictions_8win_palier4_20260923/task3/$M \
    --output-csv $OUT/task3_gan5000_8win_$M.csv
  python src/evaluation/evaluate.py --method mmfm_v2 --task task3 --modality $M \
    --pred-dir outputs/mmfm/vectorized_trajectory_gan_control/predictions_8win_palier4_20260924/task3/$M \
    --output-csv $OUT/task3_control5000_8win_$M.csv
done
python $OUT/compare_palier4.py | tee $OUT/compare_palier4.txt
