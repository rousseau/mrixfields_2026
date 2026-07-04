# Methods: Multimodal Flow Matching for MRI Cross-Field Translation

This section describes our two main approaches for the MRIxFields 2026 challenge: (i) a vectorized latent flow-matching baseline that operates on flattened MedVAE latents, and (ii) a spatial UNet flow-matching model that predicts a 3D velocity field in the latent space.

Both methods share the same high-level pipeline:

1. Preprocess the source NIfTI volume: resample to 1 mm isotropic spacing, normalize to $[-1, 1]$, and extract a $128 \times 128 \times 80$ crop.
2. Encode the volume with a frozen/fine-tuned MedVAE to obtain a spatial latent tensor $z \in \mathbb{R}^{1 \times 32 \times 32 \times 20}$.
3. Use conditional flow matching to transport $z_{\text{src}}$ to $z_{\text{tgt}}$ conditioned on the target domain $y$.
4. Decode the resulting latent with MedVAE and, for full-resolution inference, resample back to the original 0.5 mm spacing.

The target domain $y$ is one of 15 discrete classes: 3 modalities $\times$ 5 magnetic-field strengths.

---

## Notation and Flow-Matching Objective

Let $x \in \mathbb{R}^{1 \times H \times W \times D}$ denote a 3D MRI volume and $z = \mathrm{Enc}(x)$ its MedVAE latent. We formulate cross-field translation as learning a time-conditional velocity field $v_\theta(z_t, t, y)$ whose trajectories map a source latent $z_0$ to a target latent $z_1$.

We train with the optimal-transport conditional flow matching (OT-CFM) objective:

$$
\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t \sim \mathcal{U}(0,1), \, (z_0, z_1) \sim \pi} \left[ \left\| v_\theta(z_t, t, y) - u_t \right\|_2^2 \right],
$$

where $\pi$ is the OT coupling between source and target latent distributions. With the exact linear path used in OT-CFM:

$$
z_t = (1 - t) z_0 + t z_1, \qquad u_t = z_1 - z_0,
$$

and $\sigma = 0$. At inference time, we generate a prediction by integrating the learned velocity field with the Euler method:

$$
z_{\text{tgt}} = z_{\text{src}} + \int_0^1 v_\theta(z_t, t, y) \, dt.
$$

In practice, 50 Euler steps are used by default.

---

## 3.1 MMFM v1 — Vectorized Latent Flow Matching

### Architecture overview

The first baseline keeps the MedVAE encoder/decoder frozen and applies flow matching in a **vectorized latent space**. The spatial MedVAE latent $z \in \mathbb{R}^{1 \times 32 \times 32 \times 20}$ is flattened into a single vector of dimension $D = 1 \times 32 \times 32 \times 20 = 20\,480$. The velocity model is then a small multi-layer perceptron (MLP) that predicts a velocity vector of the same dimension.

### VectorMMFM model

The velocity network is implemented as:

```
input  = concat( z_t_vec,            # (B, 20480)
                 z_src_vec,          # (B, 20480)
                 time_emb(t),        # (B, time_embed_dim)
                 class_emb(y) )      # (B, class_embed_dim)
output = MLP(input)                  # (B, 20480)
```

The MLP consists of:
- A linear projection from input dimension to `hidden_dim`.
- A stack of residual MLP blocks. Each block applies LayerNorm, two linear layers with a 4$\times$ hidden-dimension bottleneck, and a residual skip connection.
- A final LayerNorm and linear layer projecting back to the latent dimension.

This design is intentionally simple: it tests whether a compact vector field can learn cross-field and cross-modal transformations without modifying the VAE or using a heavy spatial network.

### Latent reshaping

After flow integration, the predicted vector $z_{\text{tgt,vec}} \in \mathbb{R}^{20\,480}$ is reshaped back to $(1, 32, 32, 20)$ and decoded by MedVAE. The decoder reconstructs the target crop at $128 \times 128 \times 80$, 1 mm spacing.

---

## 3.2 MMFM-UNet v2 — Spatial Latent Flow Matching

### Motivation

The vectorized baseline discards spatial structure in the latent space, which may limit the model's ability to model local anatomical transformations. The second approach therefore predicts the velocity field as a **3D spatial tensor** using a UNet operating directly on the MedVAE latent.

### UNet backbone

We use MONAI's `DiffusionModelUNet` as the velocity network. It takes as input the concatenation of:
- $z_t \in \mathbb{R}^{C \times H' \times W' \times D'}$, the current latent state,
- time embedding $\gamma(t)$,
- class embedding $e(y)$,

and outputs a velocity tensor $v_\theta(z_t, t, y) \in \mathbb{R}^{C \times H' \times W' \times D'}$ with the same shape as the input latent.

Two architectural variants were explored:

| Variant | `channel_mult` | Bottleneck | Attention |
|---|---|---|---|
| V1 | `[1, 2, 4]` | $\times 8$ | Standard 3D attention at later levels |
| V2 | `[1, 2]` | $\times 4$ | Factorized 3D axial attention at bottleneck |

**V2 changes.** Reducing the number of downsampling stages from three to two avoids an impractically large attention matrix at full latent resolution. To still capture long-range dependencies, we replace the bottleneck attention with `FactorizedAttention3D`, which applies axial attention sequentially along the H, W, and D axes. This gives a global receptive field with memory cost $O(N^2)$ per axis instead of $O(N^6)$ for full 3D self-attention, where $N$ is the number of tokens along one axis.

### Training with random crops

A key issue observed during full-resolution inference was that the model underperformed outside the central brain region. Investigation showed that training used only center crops. We therefore added random spatial cropping during training: with probability 0.8, each training sample is a random $128 \times 128 \times 80$ crop from the 1 mm volume; otherwise the central crop is kept. This forces the model to learn anatomical consistency across the whole brain.

### Full-resolution inference

For test-time submission, volumes are $364 \times 436 \times 364$ at 0.5 mm. We:

1. Resample the source volume to 1 mm spacing.
2. Extract overlapping $128 \times 128 \times 80$ patches with 50% overlap.
3. For each patch, encode with MedVAE, integrate the flow field, and decode.
4. Blend patch predictions with Hann window weights.
5. Resample the blended 1 mm prediction back to 0.5 mm.
6. Mask the background using the source brain mask ($> 10^{-6}$).

The inference script also supports a `--center_crop_only` debug mode and alternative blending (`--blend_mode uniform`) for ablation studies.

---

## 3.3 Shared implementation details

### VAE

Both methods use MedVAE (`medvae_4x_1c_3d`) as the encoder/decoder. MedVAE reduces the spatial dimensions by a factor of 4 in each axis, so a $128^3$ crop gives a $32 \times 32 \times 20$ latent tensor with $C = 1$ channel. The VAE weights are either kept frozen or loaded from a locally fine-tuned checkpoint.

### Conditioning

Target domains are encoded as 15-dimensional class embeddings:

$$
y = \text{modality_idx} \times 5 + \text{field_idx}.
$$

During training, for each source sample we randomly sample one or more target domains, allowing the model to learn any-to-any translation within a single network.

### Optimization

| Hyperparameter | Vectorized v1 | UNet v2 |
|---|---|---|
| Optimizer | Adam | Adam |
| Learning rate | $10^{-4}$ | $10^{-4}$ |
| Batch size | 1 | 4 |
| Total iterations | 30 000 | 150 000 |
| Save interval | 2 000 | 5 000 |
| AMP | bfloat16 | bfloat16 |
| EMA decay | 0.9999 | 0.9999 |
| Number of targets per step | 2 | 2 |
| Gradient clipping | 1.0 | 1.0 |
| Random crop probability | 0.8 | 0.8 |

### Distributed training

Both models support multi-GPU training with PyTorch DDP. The UNet v2 model is trained on 4$\times$ H100 (Jean Zay) or on the local DGX GB10 (single 128 GB GPU). The vectorized model fits comfortably on a single GPU because of its compact MLP backbone.

---

## 3.4 Differences between vectorized and UNet variants

| Aspect | MMFM v1 vectorized | MMFM-UNet v2 |
|---|---|---|
| Latent operation | Flattened vector $\mathbb{R}^{20480}$ | Spatial tensor $\mathbb{R}^{1 \times 32 \times 32 \times 20}$ |
| Velocity network | Residual MLP | MONAI DiffusionModelUNet |
| Spatial awareness | None | Full 3D convolutions + factorized attention |
| Parameters | $\sim$97 M | $\sim$44 M |
| Memory per sample | Low | Higher (UNet feature maps) |
| Training time (local GB10) | $\sim$5 h for 30 k iters | $\sim$15 days for 150 k iters |
| Best use case | Fast baseline, ablations | Full-brain high-quality synthesis |

The vectorized baseline provides a strong, fast-to-train reference, while the UNet variant is designed to better model spatial dependencies and ultimately achieve higher-quality full-resolution predictions.
