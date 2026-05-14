# 📊 VAE Benchmark Plan — MRIxFields Full-Resolution

## Objectif

Comparer 3 architectures VAE sur volumes complets **364×436×364** :
1. **AEKL** : AutoencoderKL (MONAI) — baseline simple
2. **VQ-VAE** : NeuroQuant-inspired hybrid — avec conditioning multi-champ + adversary
3. **MedVAE** : Pré-entraîné sur 1M images MRI/CT — terme de comparaison fort

**Enjeu** : Sélectionner le meilleur VAE pour CFM3D, sachant que CFM doit diffuser dans les latents.

---

## 🔧 Architecture technique

### Stratégie patch-based

Puisque les 3 VAE sont entraînés sur **112×128×80**, on utilise le wrapper `PatchedVAE` pour traiter les images complètes :

```
Volume 364×436×364
    ↓
Extract 112×128×80 patches (stride ~84px, overlap 25%)
    ↓
Encode chaque patch
    ↓
Recombiner avec blending Gaussian (soft fusion des overlaps)
```

**Avantages** :
- ✅ Pas de perte de résolution
- ✅ Pas d'OOM
- ✅ Compatible CFM (latents patchés)
- ⚠️ Perd dépendances spatiales à travers limites patches

### Fichiers

| Fichier | Rôle |
|---|---|
| `src/utils/patched_vae.py` | Classe `PatchedVAE` + blending |
| `src/benchmark_vae.py` | Script benchmark multi-VAE |
| `src/slurm/benchmark_vae_jeanzay.slurm` | SLURM 1×H100, 10h |
| `BENCHMARK_PLAN.md` | Cette doc |

---

## 📋 Phase 1 : Reconstruction Quality

### Test set
- **10 volumes** par (modality, field)
- Splits : `Training_retrospective` (pairs) + `Validating_prospective` (unpaired)
- Gamme complète : T1W/T2W/T2FLAIR × {0.1T, 1.5T, 3T, 5T, 7T}

### Métriques per-volume

| Métrique | Calcul | Seuil bon |
|---|---|---|
| **MAE** | $\frac{1}{HWD} \sum \|x - \hat{x}\|$ | < 0.15 |
| **MSE** | $\frac{1}{HWD} \sum (x - \hat{x})^2$ | < 0.04 |
| **SSIM** | SSIM 3D (slice-wise avg) | > 0.75 |
| **LPIPS** | Perceptual distance (opt) | < 0.2 |

### Résultats attendus

| VAE | Latent size | Compression | MAE estim. | SSIM estim. | Notes |
|---|---|---|---|---|---|
| **AEKL** | 8×14×16×10 = 17.9K | 8× | 0.12 | 0.82 | KL continu, simple |
| **VQ-VAE** | 64×14×16×10 = 179K | 8× | 0.14 | 0.78 | VQ discret, paired loss |
| **MedVAE 4×1** | 1×28×32×20 = 17.9K | 64× | 0.18 | 0.70 | Pré-entraîné, très comprimé |
| **MedVAE 8×1** | 1×14×16×10 = 2.2K | 512× | 0.25 | 0.60 | Ultra-comprimé, OOM risk |

---

## 🔬 Phase 2 : Latent Space Quality

### VQ-VAE specific

```python
# Métriques de codebook
perplexity = exp(-sum(p_i * log(p_i)))  # utilisation codebook
codebook_coverage = n_unique_codes / codebook_size
```

- **Cible** : perplexity > 200, coverage > 0.8

### Modality disentanglement (VQ-VAE)

```
z_anat_0p1T vs z_anat_7T (même patient, même modalité)
→ Distance intra-modalité < inter-modalité ?
```

### Latent variance

- Intra-classe (même sujet, différentes modalités) : variance $\sigma^2_{\text{intra}}$
- Inter-classe (différents sujets) : variance $\sigma^2_{\text{inter}}$
- Ratio : $\rho = \sigma^2_{\text{inter}} / \sigma^2_{\text{intra}}$ — plus élevé = mieux pour classification/CFM

---

## 🌊 Phase 3 : CFM Tractabilité

### Considérations pour CFM3D

| Aspect | AEKL | VQ-VAE | MedVAE |
|---|---|---|---|
| **Latent shape** | (8, 14, 16, 10) | (64, 14, 16, 10) | (1, 28/14, 32/16, 20/10) |
| **Tokens par vol** | 17.9K | 179K | 17.9K / 2.2K |
| **CFM diffusion steps** | ~1000 (tractable) | ~500 (large) | ~1000 / ~250 (vary) |
| **Paired/unpaired** | ✗ | ✅ (cross-modal) | ✗ (need fine-tune) |
| **Backward compat** | ✅ | ✅ | ⚠️ (need adapt) |

**Verdict CFM** :
- **AEKL** : Safe choice, mais pas de conditioning champ
- **VQ-VAE** : Best pour cross-field synthesis (conditioning FiLM), mais plus tokens
- **MedVAE** : Meilleur pré-entraînement, mais perte info (512× compression)

---

## 🚀 Workflow exécution

### Local (smoke test, CPU)

```bash
# Test sur 2 petits volumes, patches uniquement
python src/benchmark_vae.py \
  --data-root /path/to/MRIxFields \
  --modality T1W --field 0.1T \
  --max-samples 2 \
  --skip-medvae \
  --device cpu

# Temps attendu : ~5 min
```

### JeanZay (production, 1×H100)

```bash
sbatch src/slurm/benchmark_vae_jeanzay.slurm T1W 0.1T
sbatch src/slurm/benchmark_vae_jeanzay.slurm T2W 3T
sbatch src/slurm/benchmark_vae_jeanzay.slurm T2FLAIR 7T
# Temps par job : ~30-60 min
# Total : ~2h pour 5 champs × 3 modalités
```

### Post-processing

```bash
# Génère .csv des résultats
results/benchmark/runs/benchmark_T1W_0.1T.csv
results/benchmark/runs/benchmark_T2W_3T.csv
...

# Jupyter : agrégation + tableau comparatif
jupyter notebook results/benchmark_analysis.ipynb
```

---

## 📊 Résultats : Interprétation

### Scénario 1 : AEKL wins (MAE < 0.12, SSIM > 0.80)

```
Conclusion : Utiliser AEKL pour CFM
Raison : Simple, efficace, pré-entraîné sur MONAI VAE dataset
Adaptation CFM : Standard OT-CFM sur latents 8-channel
```

### Scénario 2 : VQ-VAE wins (paired loss + adversary significatif)

```
Conclusion : Utiliser VQ-VAE pour CFM avec conditioning multi-champ
Raison : Explicit paired/unpaired, field modulation, adversary invariance
Adaptation CFM : CFM + multi-condition (T1W→T2W@7T synthesis)
```

### Scénario 3 : MedVAE wins (MAE ≈ AEKL malgré compression)

```
Conclusion : Fine-tune MedVAE sur MRIxFields, puis CFM
Raison : Pré-entraîné fort, généralisabilité médicale
Adaptation : Fine-tune stage 1 (paired loss), puis CFM
```

### Scénario 4 : Trade-off (nul)

```
Conclusion : Ensemble voting ou cascade
Raison : Chaque VAE bon pour cas d'usage différents
Adaptation : VQ-VAE (paired) + AEKL (unpaired) en CFM
```

---

## ✅ Checklist

- [ ] `src/utils/patched_vae.py` compilé ✓
- [ ] `src/benchmark_vae.py` importable ✓
- [ ] Smoke test local réussi ✓
- [ ] SLURM script prêt ✓
- [ ] Métriques SSIM implémentées ✓
- [ ] CSV output parser écrit ✓
- [ ] Tableau comparatif généré ✓
- [ ] Décision VAE finale prise
- [ ] CFM3D adapter selon choix

---

## 🔗 Références

- **AEKL** : MONAI AutoencoderKL, `src/train_vae_3d.py`
- **VQ-VAE** : NeuroQuant-inspired, `src/train_vqvae.py`
- **MedVAE** : https://github.com/StanfordMIMI/MedVAE, https://arxiv.org/abs/2502.14753
- **PatchedVAE** : Custom wrapper, `src/utils/patched_vae.py`

---

## 📝 Notes

1. **Memory** : 1×H100 80GB suffisant pour 364×436×364 avec patches
2. **Time** : 30-60s par volume en patch-based (vs OOM en full-res)
3. **Quality** : Blending Gaussian préserve continuité > simple crop/pad
4. **CFM next** : Résultats benchmark → sélection VAE → train CFM sur latents
