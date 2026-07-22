# Plan: RHVAE Retraining & Benchmark Update

## Situation Actuelle
- RHVAE checkpoint incomplet à epoch 4/200
- Benchmark récent montre MAE=0.2138 (pauvre)
- Training interrupté probablement le 29 mai

## Étape 1: Retrainer RHVAE à Complétion
### Commande
```bash
cd /home/rousseau/Exp/mrixfields_2026
python src/vae3d/train_pythae_rhvae.py \
    --config configs/pythae_rhvae_multimodal.yaml \
    --env local \
    --resume outputs/pythae_rhvae3d/runs/pythae_rhvae3d_multimodal/weights/model_best.pth
```

### Duration expected: 12-24h (GB10)
### Save checkpoints every 10 epochs

## Étape 2: Vérifier Avancement
```bash
# Check checkpoints created
ls -la outputs/pythae_rhvae3d/runs/pythae_rhvae3d_multimodal/weights/

# Verify best checkpoint quality
python -c "
import torch
state = torch.load('outputs/pythae_rhvae3d/runs/pythae_rhvae3d_multimodal/weights/model_best.pth', map_location='cpu', weights_only=False)
print(f'Epoch: {state[\"epoch\"]}')
print(f'Loss: {state[\"loss\"]:.2f}')
"
```

## Étape 3: Re-exécuter Benchmark
```bash
cd /home/rousseau/Exp/mrixfields_2026
rm results/benchmark_vae/metrics/benchmark_results.csv
PYTHONPATH=src python src/vae3d/benchmark_vae.py --skip-lpips
```

## Étape 4: Comparaison Post-Retraining
### Expected improvements:
- Loss: ~135K → ~50-100K (convergence typical)
- MAE: 0.21 → ~0.15-0.18 (potentiel)
- SSIM: 0.32 → ~0.45-0.60 (potentiel)

### Key metrics to watch:
- Loss evolution (plot from checkpoint logs)
- Benchmark summary table
- Compare with Pythae_VAE baseline

## Risques & Contre-mesures
- **Risk**: RHVAE vector latent inherently poor reconstruction
- **Mitigation**: If MAE > 0.15 after full training, consider:
  - Try smaller latent_dim (128 instead of 256)
  - Increase eps_lf for better HMC sampling
  - Use MMFM vectorisé plutôt que CFM spatial

## Success Criteria
1. Training completes to epoch 200
2. Loss convergence visible in logs
3. Benchmark MAE < 0.18
