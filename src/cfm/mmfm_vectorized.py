#!/usr/bin/env python3
"""Vectorized MMFM helpers for the MRIxFields v1 baseline.

This module intentionally stays free of MedVAE/MONAI-specific code so it can be
imported by smoke tests and reused by the MMFM training script.

The baseline keeps MedVAE untouched:
- MedVAE encodes a 3D volume into a 3D latent tensor.
- The latent tensor is flattened into a single vector before MMFM.
- MMFM predicts a vector field in that flattened latent space.
- The predicted vector is reshaped back to the MedVAE latent shape before
  decoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_time_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """Create a standard sinusoidal embedding for flow time conditioning."""

    timesteps = timesteps.float().reshape(-1)
    half = dim // 2
    if half == 0:
        return timesteps[:, None]

    device = timesteps.device
    dtype = timesteps.dtype
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=device, dtype=dtype)
        / max(half, 1)
    )
    args = timesteps[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


@dataclass(frozen=True)
class LatentVectorizer:
    """Flatten/unflatten helper for a fixed MedVAE latent shape."""

    latent_shape: Tuple[int, ...]

    @property
    def flat_dim(self) -> int:
        return int(math.prod(self.latent_shape))

    def flatten(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.reshape(latent.shape[0], -1)

    def unflatten(self, latent_vec: torch.Tensor) -> torch.Tensor:
        return latent_vec.reshape(latent_vec.shape[0], *self.latent_shape)


class ResidualMLPBlock(nn.Module):
    """Simple residual MLP block used by the vector field model."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        inner_dim = hidden_dim * 4
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.silu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + x


class VectorMMFM(nn.Module):
    """Vector-field MLP for the MMFM v1 baseline.

    Input:
      - z_t_vec: interpolated latent vector
      - z_src_vec: source latent vector
      - timesteps: scalar flow times sampled by the CFM sampler
      - class_labels: discrete target-domain ids (3 classes in v2 = modalités,
                       15 classes in v1 = modalités × champs)

    Output:
      - vector field with the same dimensionality as the flattened MedVAE latent
    """

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dim: int = 1024,
        depth: int = 4,
        time_embed_dim: int = 256,
        class_embed_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.class_embed_dim = class_embed_dim

        input_dim = 2 * latent_dim + time_embed_dim + class_embed_dim
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(depth)]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        z_t_vec: torch.Tensor,
        z_src_vec: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        time_feat = sinusoidal_time_embedding(timesteps, self.time_embed_dim)
        class_feat = self.class_embed(class_labels)
        h = torch.cat([z_t_vec, z_src_vec, time_feat, class_feat], dim=1)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        return self.output_head(h)


def cycle_rollout_vector(
    raw_mmfm: VectorMMFM,
    z_src_vec: torch.Tensor,
    y_tgt: torch.Tensor,
    t_i: float,
    t_j: float,
    n_steps: int,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable forward-then-backward Euler round trip in vector latent
    space: src(t_i) -> ~tgt(t_j) -> ~src(t_i).

    Self-supervised cycle-consistency check for unpaired multi-field training
    (see MMFM v2): `z_src_vec`/`y_tgt` are the FIXED conditioning inputs used
    throughout both legs, exactly mirroring how `_euler_integrate_vector`
    conditions on a fixed source at inference time — the only difference is
    this version is NOT wrapped in `torch.no_grad()`, so gradients flow back
    through the whole rollout into `raw_mmfm`'s parameters.

    `z` is kept in fp32 across steps (autocast only wraps each `raw_mmfm`
    call), matching `_euler_integrate_vector`'s `z = z + dt * vt.float()`
    pattern, to avoid accumulating bf16 rounding error over the multi-step
    backprop-through-time chain.

    `dt` is deliberately signed (no `abs()`/`max(dt, eps)`): t_j may be below
    t_i (target field below source field) — see the identical reasoning in
    train()'s v2 velocity-target computation.

    Returns:
        (z_tgt_hat, z_src_roundtrip) — the forward-leg endpoint and the
        round-tripped point, both fp32. Compare `z_src_roundtrip` against the
        real `z_src_vec` (known ground truth) for the cycle-consistency loss;
        `z_tgt_hat` is exposed too since edge-consistency (or future losses)
        may want the forward endpoint as well as the round trip.
    """
    device = z_src_vec.device
    dt_fwd = (t_j - t_i) / n_steps

    z = z_src_vec.float()
    for step_i in range(n_steps):
        t_val = t_i + step_i * dt_fwd
        t_vec = torch.full((z.shape[0],), t_val, dtype=torch.float32, device=device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")):
            v = raw_mmfm(z, z_src_vec, t_vec, y_tgt)
        z = z + dt_fwd * v.float()
    z_tgt_hat = z

    dt_bwd = (t_i - t_j) / n_steps
    z = z_tgt_hat
    for step_i in range(n_steps):
        t_val = t_j + step_i * dt_bwd
        t_vec = torch.full((z.shape[0],), t_val, dtype=torch.float32, device=device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")):
            v = raw_mmfm(z, z_src_vec, t_vec, y_tgt)
        z = z + dt_bwd * v.float()
    z_src_roundtrip = z

    return z_tgt_hat, z_src_roundtrip
