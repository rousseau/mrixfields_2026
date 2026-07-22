# Résultats RHVAE (latent vectoriel)

## Performance
| VAE | Mean MAE | Mean SSIM | Latent |
|-----|----------|-----------|-----|
| Pythae_RHVAE | 0.2138 | 0.3201 | Vector (256-D) |

## Commentaire
- MAE élevé (~0.21) par rapport aux VAE spatiaux (0.04-0.07)
- SSIM faible (~0.32) vs spatiaux (0.67-0.96)
- Latent vectoriel (256-D) plus difficile à optimiser pour la reconstruction

## Recommandations
1. Retrain RHVAE avec hyperparamètres optimisés
2. Utiliser Pythae_VAE comme baseline CFM (meilleur trade-off)
