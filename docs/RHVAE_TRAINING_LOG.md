# RHVAE Retraining Log

## Command
python src/vae3d/train_pythae_rhvae.py \
    --config configs/pythae_rhvae_multimodal.yaml \
    --env local

## Monitoring Commands
ls -la outputs/pythae_rhvae3d/runs/pythae_rhvae3d_multimodal/weights/
python -c "import torch; state = torch.load('outputs/pythae_rhvae3d/runs/pythae_rhvae3d_multimodal/weights/model_best.pth', map_location='cpu', weights_only=False); print(f'Epoch: {state[epoch]}'); print(f'Loss: {state[loss]:.2f}')"

## Expected Final Status
- Epoch: 200
- Loss: ~50-100K (converged)
- MAE: ~0.15-0.18 (expected)
