# Manifest — Validation stricte Task3 T1W (2026-07-17)

## Objectif
Comparer la performance maximale atteinte avec les artefacts déjà entraînés, sans auto-discovery implicite.

## Modèles évalués
1. MMFM vectoriel (latest)
   - Config: configs/mmfm3d_medvae_multimodal.yaml
   - Checkpoint: outputs/cfm3d/runs/mmfm3d_medvae_multimodal_vectorized_v1/weights/model_final.pth
   - Pred dir: outputs/predictions/mmfm/task3/T1W
   - Eval CSV: results/mmfm/analysis/protocol_20260717/task3_mmfm_vector_latest.csv

2. MMFM U-Net v1 (latest)
   - Config: configs/mmfm3d_unet_medvae_multimodal.yaml
   - Checkpoint: outputs/cfm3d/runs/mmfm3d_unet_medvae_multimodal/weights/model_final.pth
   - Pred dir: outputs/predictions/mmfm_unet/task3/T1W
   - Eval CSV: results/mmfm/analysis/protocol_20260717/task3_mmfm_unet_v1_latest.csv

3. MMFM U-Net v2 (latest available)
   - Config: configs/mmfm3d_unet_v2_medvae_multimodal.yaml
   - Checkpoint: outputs/cfm3d/runs/mmfm3d_unet_v2_medvae_multimodal/weights/checkpoint_115000.pth
   - Pred dir: outputs/predictions/mmfm_unet_v2/task3/T1W
   - Eval CSV: results/mmfm/analysis/protocol_20260717/task3_mmfm_unet_v2_latest.csv

## Commandes d'évaluation (mode strict)
- python src/evaluation/evaluate.py --method mmfm --task task3 --modality T1W --pred-dir outputs/predictions/mmfm/task3/T1W --output-csv results/mmfm/analysis/protocol_20260717/task3_mmfm_vector_latest.csv
- python src/evaluation/evaluate.py --method mmfm_unet --task task3 --modality T1W --pred-dir outputs/predictions/mmfm_unet/task3/T1W --output-csv results/mmfm/analysis/protocol_20260717/task3_mmfm_unet_v1_latest.csv
- python src/evaluation/evaluate.py --method mmfm_unet --task task3 --modality T1W --pred-dir outputs/predictions/mmfm_unet_v2/task3/T1W --output-csv results/mmfm/analysis/protocol_20260717/task3_mmfm_unet_v2_latest.csv

## Gates de validation
- Complétude (60/60): PASS pour les 3 modèles
- Conformité géométrique (shape/affine vs GT):
  - mmfm_vector: FAIL (sorties 128x128x80)
  - mmfm_unet_v1: PASS
  - mmfm_unet_v2: FAIL (sorties 128x128x80)

## Résumé global (moyenne sur 20 paires)
Voir: results/mmfm/analysis/protocol_20260717/protocol_summary_task3_T1W.csv

Classement (meilleur -> moins bon):
1. mmfm_vector_latest: nRMSE=1.2643, SSIM=0.7949, LPIPS=0.2140
2. mmfm_unet_v2_latest: nRMSE=1.4329, SSIM=0.7514, LPIPS=0.2317
3. mmfm_unet_v1_latest: nRMSE=2.0985, SSIM=0.7827, LPIPS=0.2406

## Remarque importante
Ce classement reflète les prédictions déjà présentes dans outputs/predictions et non une régénération contrôlée au même pipeline d'inférence. Les comparaisons sont valides quantitativement (mêmes paires/sujets), mais le gate géométrique indique une non-conformité de format pour vectoriel et U-Net v2.
