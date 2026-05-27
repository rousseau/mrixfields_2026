#!/usr/bin/env python3
"""Pythae VAE 3D — Encoder3D + Decoder3D + VAE wrapper for MRIxFields.

Architecture:
  Encoder: 3D conv stack (stride-2 downsampling x3 → 8× compression)
           + split head → (mu, log_var)
  Decoder: 3D transposed conv stack (upsampling x3)
  Latent:  spatial (C, H/8, W/8, D/8)  e.g. (8, 16, 16, 16) for 128³ input

Follows the Pythae BaseEncoder / BaseDecoder interface so it can be used
directly with pythae.models.VAE (for training) OR wrapped in PythaeVAEWrapper
(for CFM inference).

Usage (standalone):
    from models.pythae_vae import build_pythae_vae_3d
    vae = build_pythae_vae_3d(latent_channels=8, base_channels=32)

Usage (Pythae trainer):
    from pythae.models import VAE
    from pythae.models.vae.vae_config import VAEConfig
    from models.pythae_vae import Encoder3D, Decoder3D

    cfg = VAEConfig(input_dim=(1, 128, 128, 128), latent_dim=8*16*16*16)
    model = VAE(model_config=cfg, encoder=Encoder3D(cfg), decoder=Decoder3D(cfg))
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.models.vae.vae_config import VAEConfig

from models.vae_base import MRIxFieldsVAE


# --------------------------------------------------------------------------- #
# Building blocks                                                             #
# --------------------------------------------------------------------------- #


class ResBlock3D(nn.Module):
    """3D residual block with GroupNorm + SiLU."""

    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        ng = min(num_groups, channels)
        self.norm1 = nn.GroupNorm(ng, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(ng, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


def _make_norm(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(min(num_groups, channels), channels)


# --------------------------------------------------------------------------- #
# Encoder                                                                     #
# --------------------------------------------------------------------------- #


class Encoder3D(BaseEncoder):
    """3D convolutional encoder for Pythae VAE.

    Produces ModelOutput with `embedding` (mu) and `log_covariance` (log_var).
    Output shape (spatial latent): (B, latent_channels, H/8, W/8, D/8).
    """

    def __init__(
        self,
        model_config: VAEConfig,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_channels: int = 8,
        num_groups: int = 8,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        C = base_channels

        self.encoder = nn.Sequential(
            # stem
            nn.Conv3d(in_channels, C, 3, padding=1),
            ResBlock3D(C, num_groups),
            # down 1 : /2
            nn.Conv3d(C, C * 2, 4, stride=2, padding=1),
            _make_norm(C * 2, num_groups),
            nn.SiLU(),
            ResBlock3D(C * 2, num_groups),
            # down 2 : /4
            nn.Conv3d(C * 2, C * 4, 4, stride=2, padding=1),
            _make_norm(C * 4, num_groups),
            nn.SiLU(),
            ResBlock3D(C * 4, num_groups),
            # down 3 : /8
            nn.Conv3d(C * 4, C * 4, 4, stride=2, padding=1),
            _make_norm(C * 4, num_groups),
            nn.SiLU(),
            ResBlock3D(C * 4, num_groups),
        )
        # split into mu / log_var
        self.mu_head = nn.Conv3d(C * 4, latent_channels, 1)
        self.lv_head = nn.Conv3d(C * 4, latent_channels, 1)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.encoder(x)
        mu = self.mu_head(h)
        log_var = self.lv_head(h)
        # Pythae VAE.forward expects embedding (B,D) or (B,C,H',W',D')
        # and log_covariance with same shape.
        return ModelOutput(embedding=mu, log_covariance=log_var)


# --------------------------------------------------------------------------- #
# Decoder                                                                     #
# --------------------------------------------------------------------------- #


class Decoder3D(BaseDecoder):
    """3D convolutional decoder for Pythae VAE.

    Input: (B, latent_channels, H/8, W/8, D/8).
    Output ModelOutput with `reconstruction` (B, 1, H, W, D) in [-1, 1].
    """

    def __init__(
        self,
        model_config: VAEConfig,
        out_channels: int = 1,
        base_channels: int = 32,
        latent_channels: int = 8,
        num_groups: int = 8,
    ):
        super().__init__()
        C = base_channels

        self.decoder = nn.Sequential(
            # project latent
            nn.Conv3d(latent_channels, C * 4, 3, padding=1),
            ResBlock3D(C * 4, num_groups),
            # up 1 : ×2
            nn.ConvTranspose3d(C * 4, C * 4, 4, stride=2, padding=1),
            _make_norm(C * 4, num_groups),
            nn.SiLU(),
            ResBlock3D(C * 4, num_groups),
            # up 2 : ×4
            nn.ConvTranspose3d(C * 4, C * 2, 4, stride=2, padding=1),
            _make_norm(C * 2, num_groups),
            nn.SiLU(),
            ResBlock3D(C * 2, num_groups),
            # up 3 : ×8
            nn.ConvTranspose3d(C * 2, C, 4, stride=2, padding=1),
            _make_norm(C, num_groups),
            nn.SiLU(),
            ResBlock3D(C, num_groups),
            # output
            nn.Conv3d(C, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> ModelOutput:
        recon = self.decoder(z)
        return ModelOutput(reconstruction=recon)


# --------------------------------------------------------------------------- #
# Convenience builder                                                          #
# --------------------------------------------------------------------------- #


def build_pythae_vae_3d(
    latent_channels: int = 8,
    base_channels: int = 32,
    num_groups: int = 8,
) -> "PythaeVAE3D":
    """Build a standalone Pythae VAE 3D (encoder + decoder + VAE model).

    Returns a PythaeVAE3D instance (not a Pythae VAE — use that for training).
    For training with Pythae trainer, use build_pythae_vae_3d_for_trainer().
    """
    cfg = VAEConfig(
        input_dim=(1, 128, 128, 128),
        latent_dim=latent_channels * 16 * 16 * 16,  # informational
    )
    encoder = Encoder3D(cfg, latent_channels=latent_channels, base_channels=base_channels, num_groups=num_groups)
    decoder = Decoder3D(cfg, latent_channels=latent_channels, base_channels=base_channels, num_groups=num_groups)
    return PythaeVAE3D(encoder=encoder, decoder=decoder, latent_channels=latent_channels)


def build_pythae_vae_3d_for_trainer(
    latent_channels: int = 8,
    base_channels: int = 32,
    num_groups: int = 8,
    beta: float = 1.0,
):
    """Build a Pythae VAE model suitable for the Pythae BaseTrainer.

    Returns (model, VAEConfig) for use with pythae.trainers.BaseTrainer.
    """
    from pythae.models import VAE

    cfg = VAEConfig(
        input_dim=(1, 128, 128, 128),
        latent_dim=latent_channels * 16 * 16 * 16,
        reconstruction_loss="mse",
    )
    encoder = Encoder3D(cfg, latent_channels=latent_channels, base_channels=base_channels, num_groups=num_groups)
    decoder = Decoder3D(cfg, latent_channels=latent_channels, base_channels=base_channels, num_groups=num_groups)
    model = VAE(model_config=cfg, encoder=encoder, decoder=decoder)
    # Patch beta for beta-VAE
    if hasattr(model, "beta"):
        model.beta = beta
    return model, cfg


# --------------------------------------------------------------------------- #
# MRIxFieldsVAE wrapper                                                       #
# --------------------------------------------------------------------------- #


class PythaeVAE3D(MRIxFieldsVAE):
    """MRIxFieldsVAE wrapper around Pythae VAE 3D.

    Can be constructed from:
      - build_pythae_vae_3d(...)  (from scratch)
      - PythaeVAE3D.from_pythae_model(pythae_vae)  (from trained Pythae model)
    """

    def __init__(
        self,
        encoder: Encoder3D,
        decoder: Decoder3D,
        latent_channels: int = 8,
    ):
        super().__init__()
        self.encoder_net = encoder
        self.decoder_net = decoder
        self.latent_channels = latent_channels
        # Infer latent shape via dummy pass on CPU
        self._latent_shape: Tuple[int, ...] = self._infer_latent_shape()

    def _infer_latent_shape(self) -> Tuple[int, ...]:
        device = next(self.parameters()).device
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 32, 32, 32, device=device)
            out = self.encoder_net(dummy)
            z = out.embedding
        return tuple(z.shape[1:])

    @property
    def latent_format(self):
        return "spatial"

    @property
    def latent_shape(self) -> Tuple[int, ...]:
        return self._latent_shape

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return mean latent (mu), no sampling."""
        out = self.encoder_net(x)
        return out.embedding  # (B, C, H', W', D')

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder_net(z)
        return out.reconstruction  # (B, 1, H, W, D)

    @classmethod
    def from_pythae_model(cls, pythae_vae) -> "PythaeVAE3D":
        """Wrap a trained Pythae VAE model (pythae.models.VAE instance)."""
        encoder = pythae_vae.encoder
        decoder = pythae_vae.decoder
        lc = encoder.latent_channels if hasattr(encoder, "latent_channels") else 8
        obj = cls(encoder=encoder, decoder=decoder, latent_channels=lc)
        return obj

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, **kwargs) -> "PythaeVAE3D":
        """Load a PythaeVAE3D from a state-dict checkpoint."""
        latent_channels = kwargs.get("latent_channels", 8)
        base_channels = kwargs.get("base_channels", 32)
        num_groups = kwargs.get("num_groups", 8)
        model = build_pythae_vae_3d(
            latent_channels=latent_channels,
            base_channels=base_channels,
            num_groups=num_groups,
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        model.load_state_dict(state, strict=True)
        return model
