> **⚠️ INVALIDE — marqué le 2026-09-07.** Le classement ci-dessous provient de
> `src/vae3d/benchmark_vae.py`, qui alimentait les VAE en [0, 1] au lieu de [−1, 1]
> (+21 % de nRMSE mesurés) et dont `PatchedVAE` laissait 14.89 % des voxels à zéro
> tout en les comptant dans les métriques. La recommandation « Pythae_VAE » n'a pas
> d'autre source que ce banc et n'est donc **pas** étayée.
> Détail : `results/benchmark_vae/INVALIDE.md`. Référence de remplacement :
> `results/mmfm/representation_20260906/manifest.md`.

# VAE Performance Benchmarks — MRIxFields 2026

## Overall Ranking (by Mean MAE on prospective data)

| VAE | Mean MAE | Mean SSIM | Mean nRMSE | Latent |
|-----|----------|-----------|------------|-----|
| MedVAE_finetuned | 0.0057 | 0.9804 | 0.0503 | Spatial |
| MedVAE_frozen | 0.0125 | 0.9668 | 0.1160 | Spatial |
| NV_Generate | 0.0279 | 0.3580 | 0.1314 | Spatial |
| Pythae_VAE | 0.0413 | 0.6787 | 0.3582 | Spatial |
| Pythae_VQVAE | 0.0747 | 0.3665 | 0.3382 | Spatial |
| Pythae_RHVAE | 0.2138 | 0.3201 | 0.3151 | Vector |
| AEKL_multimodal | 0.0633 | 0.8188 | 0.4497 | Spatial |

## Notes

- MedVAE_finetuned: Best overall (MAE < 0.01)
- Pythae_RHVAE: Vector latent → poor reconstruction quality (MAE ~0.21)
- Pythae_VAE: Best among Pythae, good low-field reconstruction
- Pythae_VQVAE: Better high-field stability than Pythae_VAE
- AEKL_multimodal: Slightly worse than Pythae_VAE but still good
- NV_Generate: Moderate reconstruction, optimized for CT+MRI

## Fixes Applied

1. ✅ Moved legacy CSVs to `results/benchmark_vae/metrics/benchmark_results_legacy.csv`
2. ✅ Added DDP key remapping in vae_loader.py for Pythae models
3. ✅ Fixed encode_decode() for vector latents (RHVAE) in benchmark_vae.py
4. ✅ Updated RHVAE config eps_lf: 0.001 → 0.01

## Recommendations

1. Retrain RHVAE to completion (remove partial=True flag)
2. Retrain MedVAE_finetuned on full prospective dataset
3. Pythae_VAE is recommended baseline for CFM
