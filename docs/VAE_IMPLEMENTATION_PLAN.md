# VAE Implementation Plan — MRIxFields 2026

Date: 2026-05-27
Status: Phase B — Pythae VAE + VQ-VAE 3D (completed)

---

## 1. Context & Challenge Rules

### Multi-task constraints
- **Task 1 & 2**: up to 12 separate models allowed (4 source fields × 3 modalities).
- **Task 3**: **single unified conditional model** mandatory. All field/modality combinations must be handled by **one checkpoint**.
- **Implication for VAE**: the VAE must be **shared across all modalities and fields**. Training on a single modality (e.g. T1W only) is insufficient for Task 3.

### Data
- Training retrospective: ~1900 unpaired volumes across T1W/T2W/T2FLAIR × 5 fields.
- Training prospective: 45 paired volumes (3 subjects × 5 fields × 3 modalities).
- Evaluation: nRMSE, SSIM, LPIPS, Dice, VolumeConsistency.

---

## 2. Methods

| # | Method | Type latent | Source | Pré-entraîné | Rôle |
|---|--------|-------------|--------|-------------|------|
| 1 | **AEKL** | Spatial `(C,H',W',D')` | MONAI AutoencoderKL | ❌ | Baseline scratch |
| 2 | **Pythae VAE 3D** | Spatial | Pythae + conv 3D custom | ❌ | Baseline propre (remplace NeuroQuant) |
| 3 | **Pythae VQ-VAE 3D** | Spatial | Pythae BaseAE + quantizer 5D | ❌ | VQ-VAE propre |
| 4 | **RHVAE 3D** | Vectoriel `(D_lat)` | Pythae RHVAE + conv 3D | ❌ | Riemannian Hamiltonian baseline |
| 5 | **MedVAE** | Spatial | `medvae` package | ✅ | Pré-entraîné médical |
| 6 | **MAISI** | Spatial | MONAI bundle / NV-Generate-CTMR | ✅ | Pré-entraîné large-échelle |

**Excluded**: MMHVAE (ReubenDo) — strictly 2D, multi-scale latent hierarchy incompatible with single-latent CFM.

### Patch size standard
| Parameter | Value | Justification |
|-----------|-------|---------------|
| `patch_size` | `(128, 128, 128)` | Multiple of 32; natively supported by MAISI; H100 80GB OK with AMP + checkpointing |
| `overlap` (inference) | `0.25` | PatchedVAE existing blending |
| Standard latent (8× down) | `(4, 16, 16, 16)` | 16k values when flattened |

---

## 3. Common API: MRIxFieldsVAE

All VAE inherit from `src/models/vae_base.py`:

```python
class MRIxFieldsVAE(nn.Module, ABC):
    latent_channels: int
    
    @property
    def latent_format() -> Literal["spatial", "vector"]: ...
    
    @property  
    def latent_shape() -> Tuple[int, ...]: ...
    
    def encode(self, x: (B,1,H,W,D)) -> z
    def decode(self, z) -> recon: (B,1,H,W,D)
    
    # For CFM compatibility:
    def to_vector(self, z) -> (B, D_flat)     # flatten if spatial
    def from_vector(self, z_vec) -> z          # reshape if spatial
    
    # Inference helpers:
    def infer_full_volume(self, volume_path, patch_size=128³, overlap=0.25) -> np.ndarray
    def extract_latent_nifti(self, volume_path, output_path=None) -> nib.Nifti1Image
```

### Dual CFM pipeline
| VAE type | `latent_format` | Compatible CFM |
|----------|----------------|----------------|
| Spatial | `"spatial"` | `train_cfm_3d.py` (UNet 3D on latent) **OR** `train_mmfm_3d.py` (after `to_vector()`) |
| Vectoriel | `"vector"` | `train_mmfm_3d.py` only |

For spatial VAEs → MMFM vectoriel: simple `flatten(z_spatial)` (dimension ≈ 16k). No learned projection needed for now.

---

## 4. Multi-modality Strategy

### Option chosen: "trust" (confiance)
The VAE learns a shared latent space implicitly via reconstruction on mixed T1W+T2W+T2FLAIR data. No explicit adversarial modality invariance (no GRL classifier). This is simpler and less prone to instability.

If UMAP analysis shows poor mixing, Option B (gradient-reversed modality classifier) can be added as a variant.

### Dataset
- `src/common/dataset_vae.py` — `MRIxFieldsMultimodalDataset`
  - Loads all modalities × all fields from `Training_retrospective`
  - Returns dict with `x`, `modality`, `field`, `mod_idx`, `field_idx`
  - Same dataset used by all VAE training scripts (AEKL, Pythae, RHVAE)

---

## 5. Architecture Details

### Pythae VQ-VAE 3D
Pythae's quantizer (`vq_vae_utils.py`) is hardcoded for 4D tensors `(B,C,H,W)`. For 5D spatial latents `(B,C,D,H,W)`, we rewrite the quantizer:
```python
# 5D quantizer (adapted from Pythae)
z = z.permute(0, 2, 3, 4, 1)          # (B,D,H,W,C)
quantized = quantized.permute(0, 4, 1, 2, 3)  # (B,C,D,H,W)
```
The `BaseAE`/`BaseEncoder`/`BaseDecoder` infrastructure and the trainer are perfectly reusable.

### RHVAE 3D
- Encoder ends with global pooling / flatten → linear → `embedding` + `log_covariance`
- Decoder starts from linear projection → reshape → transposed 3D conv upsampling
- Metric network operates on compressed feature vector (not full volume)
- Hyperparameters: `n_lf` (leapfrog steps), `eps_lf`, `beta_zero`, `temperature`, `regularization`

### MAISI
- Same architecture as AEKL (channels [64,128,256,512], latent_channels=4)
- Pretrained on 55k CT+MRI volumes
- License: `autoencoder_v1.pt` (NVIDIA Open Model) — safe for challenge use
- `autoencoder_v2.pt` is Non-Commercial — **avoid**

---

## 6. Evaluation Criteria

| Criterion | Script | Details |
|-----------|--------|---------|
| Reconstruction | `benchmark_vae.py` | MAE, MSE, SSIM 3D, LPIPS, time/volume |
| Compression | integrated in benchmark | `input_voxels / latent_voxels` |
| Latent regularity | `evaluate_latent_regularity.py` | Linear interpolation smoothness, Lipschitz constant |
| UMAP visualization | `visualize_latent_umap.py` | Scatter 2D, color-coded by modality and field |
| Latent → NIfTI | `extract_latent_nifti.py` | Spatial → 4D NIfTI (H',W',D',C); Vector → CSV |

### UMAP protocol
- Encode ~100 prospective subjects (3 train + validation subset if available)
- Collect z vectors (spatial VAEs: use `to_vector(z)`)
- Fit UMAP, plot with modality colors and field markers
- Goal: anatomically similar subjects should cluster regardless of field/modality

---

## 7. Roadmap

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| **A. Infrastructure** | 2-3 days | `vae_base.py`, refactor wrappers, `dataset_vae.py`, configs multimodal | ✅ |
| **B. Pythae VAE + VQ-VAE** | 3-4 days | `pythae_vae.py`, `pythae_vqvae.py`, 5D quantizer, scripts entraînement | ✅ |
| **C. RHVAE** | 2-3 days | `train_pythae_rhvae.py`, custom metric network | ⏳ |
| **D. MAISI wrapper** | 1-2 days | Load pretrained weights, validate preprocessing alignment | ⏳ |
| **E. Evaluation** | 2-3 days | UMAP, regularity, NIfTI extraction, cumulative CSV | ⏳ |
| **F. CFM integration** | 1-2 days | Verify `load_vae` + `to_vector` in all CFM scripts | ⏳ |

---

## 8. File Inventory

### Created / Modified (Phase A)
```
src/models/vae_base.py              ← NEW: MRIxFieldsVAE abstract base
src/models/vae_wrappers.py          ← MODIFIED: inherit MRIxFieldsVAE
src/models/vae_loader.py            ← MODIFIED: support maisi, placeholder pythae/rhvae
src/common/dataset_vae.py           ← NEW: MRIxFieldsMultimodalDataset
configs/vae3d_multimodal.yaml       ← NEW: AEKL 128³ multimodal config
docs/VAE_IMPLEMENTATION_PLAN.md     ← THIS FILE
```

### Created / Modified (Phase B)
```
src/models/pythae_vae.py            ← NEW: Encoder3D + Decoder3D + PythaeVAE3D wrapper
src/models/pythae_vqvae.py          ← NEW: VQEncoder3D + Quantizer5D + PythaeVQVAE3D
src/models/vae_loader.py            ← MODIFIED: _load_pythae_vae, _load_pythae_vqvae
src/models/vae_base.py              ← MODIFIED: from_vector dynamique (racine cubique)
src/vae3d/train_pythae_vae.py       ← NEW: entraînement multimodal VAE
src/vae3d/train_pythae_vqvae.py     ← NEW: entraînement multimodal VQ-VAE
configs/pythae_vae_multimodal.yaml  ← NEW: config VAE 128³ multimodal
configs/pythae_vqvae_multimodal.yaml← NEW: config VQ-VAE 128³ multimodal
src/slurm/train_vae_jeanzay.slurm   ← MODIFIED: routing pythae_vae / pythae_vqvae
```

### To Create (Phase B–F)
```
src/models/pythae_vae.py            # Encoder3D + Decoder3D + VAE model
src/models/pythae_vqvae.py          # VQ-VAE3D (5D quantizer)
src/models/pythae_rhvae.py          # RHVAE3D wrapper
src/models/maisi_vae.py             # MAISI wrapper (if different from vae_loader)
src/vae3d/train_pythae_vae.py
src/vae3d/train_pythae_vqvae.py
src/vae3d/train_pythae_rhvae.py
src/vae3d/visualize_latent_umap.py
src/vae3d/evaluate_latent_regularity.py
src/vae3d/extract_latent_nifti.py
configs/pythae_vae_multimodal.yaml
configs/pythae_vqvae_multimodal.yaml
configs/pythae_rhvae_multimodal.yaml
configs/maisi_vae_multimodal.yaml
```

### Deprecated (kept for backward compat, not used in new benchmarks)
```
src/vae3d/train_vqvae.py            # NeuroQuant VQ-VAE
configs/vqvae3d_T1W.yaml            # Mono-modale
configs/vae3d_T1W.yaml              # Mono-modale (replaced by multimodal)
```

---

## 9. Open Questions

1. **RHVAE metric network**: default `Metric_MLP` operates on flattened volume. For 128³ = 2M voxels, this is impractical. A custom `BaseMetric` on encoder features is needed.

2. **MAISI preprocessing alignment**: MAISI may expect a specific intensity range or normalization. Need to verify compatibility with our percentile normalization `[-1, 1]` before fine-tuning.

3. **AEKL re-training**: The existing AEKL T1W-only checkpoint (`vae3d_T1W/weights/model_best.pth`) will be superseded by `vae3d_multimodal`. Existing CFM experiments using the old AEKL will need re-training.
