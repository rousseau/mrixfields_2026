# CFM3D 0.1T → 7T Training and Inference Analysis

**Date**: 2026-05-23  
**Method**: OT-CFM 3D + MedVAE fine-tuned (bidomain: 0.1T ↔ 7T)  
**Task**: Task 2 (Low-field enhancement)

---

## 1. Training Configuration and Duration

### Hardware
- **Platform**: DGX Spark GB10 (single NVIDIA GB10 GPU, CUDA 12.4)
- **GPU Memory**: ~24GB per GB10
- **PyTorch Version**: 2.10.0+cu130 (warning: GB10 CUDA capability 12.1 exceeds supported range 8.0-12.0)

### Configuration
- **Batch size**: 2 volumes/batch
- **Total iterations**: 100,000 (reported as 100k in logs)
- **Learning rate**: 1.0e-4 (Adam optimizer)
- **OT method**: Exact Optimal Transport
- **Noise schedule**: σ = 0.0 (CFM, no noise)
- **Gradient clipping**: 1.0
- **EMA decay**: 0.9999
- ** AMP**: Enabled (mixed precision training)

### Training Duration
- **Duration**: ~32 hours (based on checkpoint timestamps: 2026-05-20 to 2026-05-21)
- **Iteration throughput**: ~3 iterations/hour (100k iterations / 32 hours)
- **Checkpoints saved**: every 5,000 iterations (20 total + final)
- **Final checkpoint**: `checkpoint_100000.pth` (iter 99,999)

### Model Architecture

#### UNet 3D (CFM backbone)
- **Parameters**: 178.14M
- **Spatial dims**: 3
- **Model channels**: 128
- **Res blocks per level**: 2
- **Channel multipliers**: [1, 2, 4]
- **Attention levels**: [false, true, true] (at resolutions 1/4 and 1/8)
- **Head channels**: 64
- **Norm groups**: 32
- **Gradient checkpointing**: enabled
- **Input channels**: 2 (concatenated latent: z_t + z_src)
- **Output channels**: 1 (velocity field)
- **Class embeddings**: 5 domains × 512 dims

#### VAE Encoder/Decoder (MedVAE fine-tuned)
- **Type**: MedVAE fine-tuned on MRIxFields
- **Compression**: 4× spatial, 1× latent channel → (B, 1, 32, 32, 20)
- **Latent size**: 20,480 voxels per volume
- **Status**: Fine-tuned weights loaded from `medvae_finetune_all/weights/model_best.pth`

### Data
- **Modalities**: T1W only
- **Domains**: 0.1T (100 volumes) + 7T (235 volumes) = 335 total
- **Split**: retro_train (retrospective training)
- **Target spacing**: [1.0, 1.0, 1.0] mm isotropic
- **Volume size**: [128, 128, 80]
- **Percentile clipping**: [0.5, 99.5]

---

## 2. Results Quality Issues

### Observed Problems (from inference on prospective subjects)

#### 2.1 Severe Intensity Scaling Mismatch
- **Prediction range**: ~0.67 (normalized, near upper bound)
- **Ground truth range**: ~0.04 (normalized, near center)
- **Mismatch factor**: ~17× intensity difference (0.67 / 0.04)
- **Effect**: Predictions are systematically over-brightened

#### 2.2 Low SSIM Values
- **SSIM range**: 0.008–0.013 (expected: >0.5 for reasonable quality)
- **Interpretation**: Predictions share virtually no structural similarity with ground truth
- **Context**: SSIM measures local structural correlation; values <0.05 indicate catastrophic failure

#### 2.3 Systematic Over-Brightness
- All predictions exhibit exaggerated intensity scaling
- Loss of fine anatomical detail
- Contrast inversion artifacts visible in visualizations

#### 2.4 Saturation at Extremes
- Predictions cluster near upper/lower bounds of normalized range
- Reduced dynamic range utilization
- Information loss in mid-intensity regions

### Visual Evidence (figures)
- `cfm3d_0p1T_7T_subject_0001.nii.png` — Over-bright predictions, poor alignment with GT
- `cfm3d_0p1T_7T_subject_0002.nii.png` — Similar intensity mismatch
- `cfm3d_0p1T_7T_subject_0003.nii.png` — Consistent over-brightness across subjects
- `cfm3d_0p1T_7T_all_subjects.png` — Systematic summary

---

## 3. Root Cause Analysis

### 3.1 Primary Cause: Training Objective Misalignment

#### CFM Flow Target vs. VAE Reconstruction
- **CFM objective**: Learn optimal transport map between domain distributions
  - Target: Move latent z_src (0.1T) → z_target (7T) along geodesic
  - Loss: Minimal displacement under OT metric
  
- **Reality**: MedVAE latent space compression + intensity normalization creates non-linear intensity distortion
  - VAE learned to reconstruct **normalized** volumes (range [-1, 1])
  - CFM learns flow in **normalized** latent space
  - But 0.1T vs 7T have different mean intensities even after normalization

#### Evidence
1. **VAE reconstruction quality is excellent** (SSIM 0.83–0.85 for intra-field recon)
2. **CFM translation quality is catastrophic** (SSIM 0.01 for cross-field)
3. This indicates the problem is **not** VAE failure but **CFM flow learning failure**

### 3.2 Secondary Causes

#### A. Data Distribution Shift
- **0.1T images**: Lower SNR, different contrast, wider intensity distribution
- **7T images**: Higher SNR, different tissue contrast, narrower distribution
- After normalization: Distributions overlap but with different statistics
- **Issue**: CFM assumes similar distributions; large shift breaks geodesic assumption

#### B. Bidomain Training Scarcity
- Only 100× 0.1T + 235× 7T = 335 paired samples
- **Comparison**: Full 5-domain config would have ~1900 volumes
- **Impact**: Insufficient diversity to learn robust cross-field mapping
- **Risk**: Overfitting to specific subject characteristics

#### C. Learning Rate Too High for Fine-Grained Mapping
- LR = 1.0e-4 is standard but may be too aggressive for:
  - Fine intensity scaling adjustments
  - High-resolution latent details (32×32×20 = 20k voxels)
- Early training may overshoot optimal flow

#### D. No Explicit Intensity Consistency Constraint
- CFM optimizes OT cost: E[‖z_1 - z_0‖²]
- **Missing**: Explicit constraint on reconstruction intensity
- VAE decoder may produce different intensity scale than training

#### E. Batch Size Too Small (B=2)
- Limited gradient statistics
- Batch normalization unstable
- EMA may not converge reliably

---

## 4. Next Steps and Recommendations

### High Priority: Fix Intensity Scaling

#### Option 1: Add Intensity Regression Head (RECOMMENDED)
- **Approach**: Concatenate mean/intensity features to CFM input
- **Implementation**:
  ```python
  # In train_cfm_3d.py, modify forward pass:
  z_cat = torch.cat([z_t, z_src, μ_src, μ_t], dim=1)  # add 2 scalar features
  ```
- **Benefit**: Directly teaches CFM to compensate for intensity shift
- **Complexity**: Low (2 additional channels)

#### Option 2: Post-Hoc Intensity Matching
- **Approach**: After CFM prediction, match intensity statistics to 7T training distribution
- **Implementation**:
  ```python
  pred = cfm_predict(src_0.1T)
  pred = match_intensity(pred, gt_7T_stats)  # histogram matching or regression
  ```
- **Benefit**: Simple, no training changes
- **Drawback**: Doesn't fix structural issues

#### Option 3: Train with Intensity-Conditioned CFM
- **Approach**: Condition CFM on intensity ratio r = μ_7T / μ_0.1T
- **Implementation**: Add intensity ratio to class embedding
- **Benefit**: Explicit modeling of intensity scaling
- **Complexity**: Medium

### Medium Priority: Improve Training Stability

#### A. lr Scheduler + Warmup
- **Change**: Linear warmup for first 5k iterations
- **Reason**: Stabilize early training on difficult cross-domain mapping

#### B. Increase Batch Size
- **Change**: Try B=4 or B=8 (if GPU memory allows)
- **Benefit**: Better gradient estimates, BN stability
- **Trade-off**: Longer iteration time but fewer iterations needed

#### C. Add Reconstruction Loss Weighting
- **Approach**: Multi-task loss:
  ```
  L_total = L_OT + λ_L1 * ‖Dec(z_hat) - x_target‖₁
  ```
- **Benefit**: Directly constrains intensity scaling at decoder
- **Risk**: May conflict with OT objective

### Low Priority: Data and Architecture Improvements

#### A. Restore 5-Domain Training
- **Rationale**: 0.1T ↔ 7T bidomain is too constrained
- **Benefit**: More diverse training, better generalization
- **Alternative**: Train 0.1T→1.5T, 1.5T→3T, 3T→5T, 5T→7T as chain

#### B. Pre-Train CFM on 5-Domains, Fine-Tune on Bidomain
- **Two-stage**: Learn general field translation first
- **Then**: Fine-tune on 0.1T↔7T data

#### C. Use Diffusion Model Instead of UNet
- **Rationale**: CFM is compatible with both
- **Benefit**: Potentially smoother flows
- **Cost**: Requires re-implementation

### Validation Strategy

#### 1. Quick Test (24h)
- Apply post-hoc intensity matching (Option 2)
- Evaluate on same subjects
- **Success criteria**: SSIM > 0.3

#### 2. Training Update (1 week)
- Add intensity regression head (Option 1)
- Reduce LR to 5.0e-5
- Add linear warmup (5k iters)
- Train for 100k iterations
- **Success criteria**: SSIM > 0.5

#### 3. Full Re-Training (2 weeks)
- Restore 5-domain config
- Train CFM from scratch
- Use best-performing vae (MedVAE fine-tuned)
- **Success criteria**: Match StarGAN 2D baseline

---

## 5. Summary Table

| Metric | Expected | Actual | Gap |
|--------|----------|--------|-----|
| SSIM | 0.5–0.7 | 0.008–0.013 | **50–60× low** |
| Intensity range | 0.0±0.2 | 0.67 | **3.5× offset** |
| Structural detail | Visible anatomy | Saturated blobs | NA |
| Visually plausible | Yes | No | Complete failure |

| Root Cause | Impact | Fix Complexity |
|------------|--------|----------------|
| Intensity scaling mismatch | Critical | Low |
| OT objective misalignment | Critical | Medium |
| Small bidomain dataset | High | Medium |
| lr too high | Medium | Low |
| batch_size=2 | Medium | Low |

---

## 6. Files Reference

### Training
- **Config**: `configs/cfm3d_T1W_medvae_0p1T_7T.yaml`
- **Script**: `src/cfm/train_cfm_3d.py`
- **Checkpoint**: `outputs/cfm3d/runs/cfm3d_T1W_medvae_0p1T_7T/weights/model_final.pth`

### Inference
- **Script**: `src/cfm/run_inference_viz_cfm3d.py`
- **Predictions**: `results/cfm/visuals/cfm3d_0p1T_7T/predictions/`
- **Visuals**: `results/cfm/visuals/cfm3d_0p1T_7T/figures/`

### VAE Weights
- **Checkpoint**: `outputs/medvae/runs/medvae_finetune_all/weights/model_best.pth`
- **Params**: ~178M UNet + MedVAE encoder/decoder (unchanged)

---

## Generated By
Analysis completed: 2026-05-23  
Based on training logs, checkpoint inspection, and inference visualizations.
