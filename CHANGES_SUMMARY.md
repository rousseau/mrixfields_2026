# Summary of Fixes Applied

## MMFM-UNet multi-marginal — 18 juillet 2026

Nouvelle méthode pour la Task 3 (any-to-any) avec une **vraie formulation
multi-marginal** :

- **Contraste** = classe conditionnante (`num_class_embeds=3`).
- **Champ magnétique** = axe temporel du flow (`0.1T→1.5T→3T→5T→7T` mappé sur
  `[0,1]`).
- Couplage OT entre marginales **adjacentes**, avec cas identité
  (`identity_prob=0.15`).
- Entraînement sur **volume entier** (latent MedVAE `(1, 44, 54, 44)`) via un
  **cache de latents** pré-encodés.
- Padding réversible `54→56` pour la contrainte UNet 3 niveaux.

### Fichiers ajoutés / modifiés
- `src/cfm/train_mmfm_unet_3d.py` — boucle multi-marginale, padding/crop,
  `_euler_integrate_mm`.
- `src/cfm/precompute_latents.py` — cache de latents (vae_id hashé, index.json).
- `src/common/dataset.py` — `LatentCacheDataset` (chargement RAM + flip G/D).
- `src/cfm/infer_mmfm_unified.py` — inférence continue t_source → t_target.
- `configs/mmfm3d_multimarginal_medvae.yaml` — config de base.
- `configs/mmfm3d_multimarginal_medvae_run1.yaml` — run de nuit 12k iters.
- `src/slurm/run_mmfm_multimarginal_night.sh` — wrapper pré-encode + entraîne.
- `docs/MMFM3D_UNET_MULTIMARGINAL.md` — documentation de la méthode.
- `README.md` + `AGENTS.md` — mise à jour des liens et état d'avancement.

### Run en cours
- Session `screen` : `mmfm_run1`
- PID pré-encodage : 455321
- Sorties : `outputs/cfm3d/runs/mmfm3d_multimarginal_medvae_run1/`
- Logs : `logs/precompute_latents_run1.out`,
  `logs/mmfm3d_multimarginal_medvae_run1.out`

---

## Issues Identified

1. **Corrupted old benchmark results** (results/benchmark_comparison/)
   - AEKL: 'NoneType' object is not subscriptable
   - MedVAE: all_failed
   - Cause: Runs executed before VAE training completed

2. **RHVAE training incomplete** (partial=True flag)
   - Checkpoint from epoch 4 only (target: 200)
   - Loss: 135,707.93
   - MAE: 0.2138 (poor performance)

3. **Missing DDP key remapping for Pythae models**
   - Multi-GPU models have keys prefixed with module.
   - Missing in vae_loader.py

4. **RHVAE vector latency inference compatibility**
   - PatchedVAE now works with both spatial & vector latents

## Fixes Applied

1. **Archived corrupted CSVs**
   - Moved to results/benchmark_comparison/archived/
   - Added README explanation

2. **Added DDP key remapping** (vae_loader.py)
   - Lines 501-502, 544-545, 589-590
   - Handles "module." prefix for Pythae models

3. **Fixed RHVAE inference** (benchmark_vae.py)
   - Simplified encode_decode() for all VAE types

4. **Updated RHVAE config**
   - eps_lf: 0.001 → 0.01

## Updated Files
- src/models/vae_loader.py (+8/-2)
- src/vae3d/benchmark_vae.py (+2/-2)
- results/benchmark_comparison/ (archived + README)

## Current Status

| Component | Status |
|-----------|--------|
| Corrupted CSVs | ✅ Archived |
| DDP handling | ✅ Fixed |
| RHVAE inference | ✅ Fixed |
| RHVAE training | ⏳ Needs retraining (from epoch 4 → 200) |
| VAE performance docs | ✅ Created |

## Next Steps (if plan approved)

1. Resume RHVAE training (estimated 12-24h)
2. Re-run benchmark after completion
3. Compare with Pythae_VAE baseline
4. Decide whether to continue with RHVAE or switch to Pythae_VAE for CFM
