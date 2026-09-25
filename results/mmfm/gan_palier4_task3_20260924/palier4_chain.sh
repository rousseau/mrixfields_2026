#!/bin/bash
set -e
cd /home/rousseau/Exp/mrixfields_2026
export PYTHONPATH=src PYTHONUNBUFFERED=1
echo "[$(date)] 1/3 fin inference GAN T2FLAIR"
python src/cfm/infer_mmfm_unified.py --config configs/mmfm/vectorized_trajectory_gan.yaml \
  --checkpoint outputs/mmfm/vectorized_trajectory_gan/weights/model_final.pth \
  --output_dir outputs/mmfm/vectorized_trajectory_gan/predictions_8win_palier4_20260923 \
  --modalities T2FLAIR --split Training_prospective --field_norm_stats configs/mmfm/field_norm_stats.json --skip_existing --env local > logs/gan_palier4_infer_resume_20260924.log 2>&1
echo "[$(date)] 2/3 temoin 2000->5000"
python src/cfm/train_mmfm_unified.py --method mmfm3d_vectorized \
  --config configs/mmfm/vectorized_trajectory_gan_control.yaml --env local \
  --resume outputs/mmfm/vectorized_trajectory_gan_control/weights/model_final.pth \
  > logs/mmfm_vectorized_trajectory_gan_control_palier4.out 2>&1
echo "[$(date)] 3/3 inference temoin 8win"
python src/cfm/infer_mmfm_unified.py --config configs/mmfm/vectorized_trajectory_gan_control.yaml \
  --checkpoint outputs/mmfm/vectorized_trajectory_gan_control/weights/model_final.pth \
  --output_dir outputs/mmfm/vectorized_trajectory_gan_control/predictions_8win_palier4_20260924 \
  --split Training_prospective --field_norm_stats configs/mmfm/field_norm_stats.json --skip_existing --env local > logs/gan_control_palier4_infer_20260924.log 2>&1
echo "[$(date)] FIN"
