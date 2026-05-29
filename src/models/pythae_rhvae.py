#!/usr/bin/env python3
"""Pythae RHVAE 3D — Riemannian Hamiltonian VAE for MRIxFields.

Architecture:
  Encoder3D_vec : 3D conv stack (stride-2 ×3 → 8× compression)
                  + AdaptiveAvgPool3d(1) + flatten → (mu, log_var) vectors
  Decoder3D_vec : MLP reshape + 3D transposed conv stack (upsampling ×3)
  Metric3D      : Shares encoder feature extractor, outputs L ∈ (B, D, D)
                  where M = L @ L^T is the Riemannian metric matrix.

Latent:  vector (D_lat)  — compatible with MMFM vectorised (Étape 4).

The RHVAE Hamiltonian leapfrog integrator is handled entirely by Pythae.
We only need to provide compliant Encoder / Decoder / Metric modules.

Usage:
    from models.pythae_rhvae import build_pythae_rhvae_3d
    model = build_pythae_rhvae_3d(latent_dim=512, base_channels=32)

    # Inference
    model.eval()
    z  = model.encode(x)      # (B, D_lat)
    rx = model.decode(z)      # (B, 1, H, W, D)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pythae.models import RHVAE, RHVAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder, BaseMetric

from models.vae_base import MRIxFieldsVAE


# --------------------------------------------------------------------------- #
# Building blocks (shared with pythae_vae.py but kept local for independence) #
# --------------------------------------------------------------------------- #


class ResBlock3D(nn.Module):
    """3D residual block with GroupNorm + SiLU."""

    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        g = min(num_groups, channels)
        while channels % g != 0 and g > 1:
            g -= 1
        self.net = nn.Sequential(
            nn.GroupNorm(g, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(g, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _make_down_block(in_c: int, out_c: int, num_groups: int = 8) -> nn.Sequential:
    g = min(num_groups, in_c)
    while in_c % g != 0 and g > 1:
        g -= 1
    return nn.Sequential(
        nn.GroupNorm(g, in_c),
        nn.SiLU(),
        nn.Conv3d(in_c, out_c, 4, stride=2, padding=1),  # stride-2 halving
        ResBlock3D(out_c, num_groups),
    )


def _make_up_block(in_c: int, out_c: int, num_groups: int = 8) -> nn.Sequential:
    g = min(num_groups, in_c)
    while in_c % g != 0 and g > 1:
        g -= 1
    return nn.Sequential(
        nn.GroupNorm(g, in_c),
        nn.SiLU(),
        nn.ConvTranspose3d(in_c, out_c, 4, stride=2, padding=1),
        ResBlock3D(out_c, num_groups),
    )


# --------------------------------------------------------------------------- #
# Feature extractor (shared by encoder + metric)                              #
# --------------------------------------------------------------------------- #


class ConvFeatureExtractor3D(nn.Module):
    """3D conv backbone: (B,1,H,W,D) -> (B, base*4, H/8, W/8, D/8).

    Used by both the encoder head and the metric network.
    """

    def __init__(self, base_channels: int = 32, num_groups: int = 8):
        super().__init__()
        C = base_channels
        self.stem = nn.Conv3d(1, C, 3, padding=1)
        self.down1 = _make_down_block(C, C * 2, num_groups)     # /2
        self.down2 = _make_down_block(C * 2, C * 4, num_groups) # /4
        self.down3 = _make_down_block(C * 4, C * 8, num_groups) # /8
        self.out_channels = C * 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        return h  # (B, C*8, H/8, W/8, D/8)


# --------------------------------------------------------------------------- #
# Pythae-compliant Encoder (vectorial latent)                                 #
# --------------------------------------------------------------------------- #


class RHVAEEncoder3D(BaseEncoder):
    """3D conv encoder with global average pooling → vectorial latent.

    Returns ModelOutput(embedding=mu, log_covariance=log_var)
    where mu, log_var ∈ (B, latent_dim).
    """

    def __init__(
        self,
        latent_dim: int,
        base_channels: int = 32,
        num_groups: int = 8,
        model_config=None,  # unused, kept for Pythae compatibility
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.backbone = ConvFeatureExtractor3D(base_channels, num_groups)
        feat_dim = self.backbone.out_channels  # C*8
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mu_head = nn.Linear(feat_dim, latent_dim)
        self.lv_head = nn.Linear(feat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.backbone(x)            # (B, C*8, H/8, W/8, D/8)
        h = self.pool(h).flatten(1)     # (B, C*8)
        mu = self.mu_head(h)            # (B, D)
        log_var = self.lv_head(h)       # (B, D)
        return ModelOutput(embedding=mu, log_covariance=log_var)


# --------------------------------------------------------------------------- #
# Pythae-compliant Decoder (vectorial -> spatial)                             #
# --------------------------------------------------------------------------- #


class RHVAEDecoder3D(BaseDecoder):
    """MLP project + 3D transposed conv stack: z (B,D) -> (B,1,H,W,D).

    The spatial size reconstructed depends on `spatial_size` (H/8 of input).
    Default: 16 (for 128³ input → 16³ feature map before 3 upsamples).
    """

    def __init__(
        self,
        latent_dim: int,
        base_channels: int = 32,
        num_groups: int = 8,
        spatial_size: int = 16,
        model_config=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_size = spatial_size
        C = base_channels
        feat_channels = C * 8

        self.proj = nn.Linear(latent_dim, feat_channels * spatial_size ** 3)
        self.reshape_c = feat_channels
        self.reshape_s = spatial_size

        self.up1 = _make_up_block(feat_channels, C * 4, num_groups)    # ×2
        self.up2 = _make_up_block(C * 4, C * 2, num_groups)            # ×2
        self.up3 = _make_up_block(C * 2, C, num_groups)                # ×2
        g = min(num_groups, C)
        while C % g != 0 and g > 1:
            g -= 1
        self.out_conv = nn.Sequential(
            nn.GroupNorm(g, C),
            nn.SiLU(),
            nn.Conv3d(C, 1, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> ModelOutput:
        B = z.shape[0]
        h = self.proj(z)
        h = h.view(B, self.reshape_c, self.reshape_s, self.reshape_s, self.reshape_s)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        recon = self.out_conv(h)
        return ModelOutput(reconstruction=recon)


# --------------------------------------------------------------------------- #
# Pythae-compliant Metric network                                             #
# --------------------------------------------------------------------------- #


class RHVAEMetric3D(BaseMetric):
    """Riemannian metric network: image -> L ∈ (B, D, D) lower triangular.

    Shares the same feature extractor architecture as the encoder (but
    separate weights), following the RHVAE paper.

    Returns ModelOutput(L=L) where M = L @ L^T is the metric matrix.
    """

    def __init__(
        self,
        latent_dim: int,
        base_channels: int = 32,
        num_groups: int = 8,
        model_config=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.backbone = ConvFeatureExtractor3D(base_channels, num_groups)
        feat_dim = self.backbone.out_channels
        self.pool = nn.AdaptiveAvgPool3d(1)
        # Output the lower-triangular entries of L  (D*(D+1)/2 values)
        n_entries = latent_dim * (latent_dim + 1) // 2
        self.head = nn.Linear(feat_dim, n_entries)
        self._tril_idx = None  # lazily initialised on first forward

    def _get_tril_idx(self, device: torch.device):
        if self._tril_idx is None or self._tril_idx[0].device != device:
            idx = torch.tril_indices(self.latent_dim, self.latent_dim)
            self._tril_idx = (idx[0].to(device), idx[1].to(device))
        return self._tril_idx

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.backbone(x)
        h = self.pool(h).flatten(1)  # (B, feat_dim)
        entries = self.head(h)       # (B, D*(D+1)/2)

        B, D = entries.shape[0], self.latent_dim
        L = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
        rows, cols = self._get_tril_idx(x.device)
        L[:, rows, cols] = entries
        # Ensure positive diagonal (Cholesky stability)
        diag_idx = torch.arange(D, device=x.device)
        L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + 1e-6

        return ModelOutput(L=L)


# --------------------------------------------------------------------------- #
# MRIxFieldsVAE wrapper                                                       #
# --------------------------------------------------------------------------- #


class PythaeRHVAE3D(MRIxFieldsVAE):
    """RHVAE 3D wrapped as MRIxFieldsVAE.

    Latent format: vector (B, D_lat)
    encode(): returns mu (no sampling at inference).
    decode(): wraps RHVAEDecoder3D.
    forward_train(): wraps pythae RHVAE.forward() — returns ModelOutput with
                     loss, z, recon_x, etc.

    Note: update_metric() must be called at the end of each epoch (calls
    self.rhvae.update()) to freeze the metric from collected centroids.
    """

    def __init__(
        self,
        latent_dim: int,
        base_channels: int = 32,
        num_groups: int = 8,
        spatial_size: int = 16,
        # RHVAE hyper-params
        n_lf: int = 3,
        eps_lf: float = 0.001,
        beta_zero: float = 0.3,
        temperature: float = 1.5,
        regularization: float = 0.01,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_channels = latent_dim  # alias used by vae_base API
        self._spatial_size = spatial_size

        encoder = RHVAEEncoder3D(latent_dim, base_channels, num_groups)
        decoder = RHVAEDecoder3D(latent_dim, base_channels, num_groups, spatial_size)
        metric = RHVAEMetric3D(latent_dim, base_channels, num_groups)

        # input_dim must match the actual volume shape so that Pythae's
        # _log_p_x_given_z correctly reshapes recon_x and x before MSE.
        # spatial_size is patch_size / 8 (3 stride-2 downsamples), so the
        # full patch size is spatial_size * 8 on each spatial axis.
        patch_size = spatial_size * 8
        cfg = RHVAEConfig(
            input_dim=(1, patch_size, patch_size, patch_size),
            reconstruction_loss="mse",
            latent_dim=latent_dim,
            n_lf=n_lf,
            eps_lf=eps_lf,
            beta_zero=beta_zero,
            temperature=temperature,
            regularization=regularization,
        )
        self.rhvae = RHVAE(
            model_config=cfg,
            encoder=encoder,
            decoder=decoder,
            metric=metric,
        )

    # -- MRIxFieldsVAE contract ---------------------------------------------- #

    @property
    def latent_format(self):
        return "vector"

    @property
    def latent_shape(self):
        return (self.latent_dim,)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode → mu vector (no sampling at inference)."""
        enc_out = self.rhvae.encoder(x)
        return enc_out.embedding  # (B, D_lat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector → reconstructed volume."""
        dec_out = self.rhvae.decoder(z)
        return dec_out.reconstruction  # (B, 1, H, W, D)

    # -- Training helper ----------------------------------------------------- #

    def forward_train(self, x: torch.Tensor) -> ModelOutput:
        """Full RHVAE forward (with HMC leapfrog).

        Returns Pythae ModelOutput with fields:
            loss, recon_x, z, z0, mu, log_var, G_inv, G_log_det, ...
        """
        from pythae.data.datasets import DatasetOutput
        inputs = DatasetOutput(data=x, labels=torch.zeros(x.shape[0], device=x.device))
        return self.rhvae(inputs)

    def update_metric(self):
        """Call at end of each epoch to update the Riemannian metric."""
        self.rhvae.update()


# --------------------------------------------------------------------------- #
# Builder                                                                     #
# --------------------------------------------------------------------------- #


def build_pythae_rhvae_3d(
    latent_dim: int = 256,
    base_channels: int = 32,
    num_groups: int = 8,
    spatial_size: int = 16,
    n_lf: int = 3,
    eps_lf: float = 0.001,
    beta_zero: float = 0.3,
    temperature: float = 1.5,
    regularization: float = 0.01,
) -> PythaeRHVAE3D:
    """Build a PythaeRHVAE3D with given hyperparameters.

    Args:
        latent_dim:     vectorial latent dimension (e.g. 256 or 512)
        base_channels:  base feature channels (scales ×2, ×4, ×8 along encoder)
        num_groups:     GroupNorm groups
        spatial_size:   spatial size of the feature map before decoder upsampling
                        (= input_size / 8, e.g. 16 for 128³ input)
        n_lf:           number of leapfrog steps in HMC
        eps_lf:         leapfrog step size
        beta_zero:      initial inverse temperature
        temperature:    metric temperature
        regularization: metric regularization (lambda)

    Returns:
        PythaeRHVAE3D instance (untrained)
    """
    return PythaeRHVAE3D(
        latent_dim=latent_dim,
        base_channels=base_channels,
        num_groups=num_groups,
        spatial_size=spatial_size,
        n_lf=n_lf,
        eps_lf=eps_lf,
        beta_zero=beta_zero,
        temperature=temperature,
        regularization=regularization,
    )
