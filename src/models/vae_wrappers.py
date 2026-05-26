#!/usr/bin/env python3
"""Unified VAE wrapper interfaces.

All VAE architectures (AEKL, MedVAE, VQ-VAE, MedVAE-disentangle) implement
the VAEWrapper ABC so they can be used interchangeably by CFM training,
benchmark, and evaluation scripts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Abstract base                                                               #
# --------------------------------------------------------------------------- #


class VAEWrapper(nn.Module, ABC):
    """Unified interface for all VAE architectures.

    All subclasses must implement:
      - encode(x) -> z
      - decode(z) -> x

    And set self.latent_channels to the number of latent channels.
    """

    latent_channels: int

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent: (B,1,H,W,D) -> (B,C,H',W',D')."""

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image: (B,C,H',W',D') -> (B,1,H,W,D)."""


# --------------------------------------------------------------------------- #
# AEKL (MONAI AutoencoderKL)                                                  #
# --------------------------------------------------------------------------- #


class AEKLWrapper(VAEWrapper):
    """Wrapper for MONAI AutoencoderKL.

    encode() returns z_mu (deterministic, no sampling).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.latent_channels = _infer_aekl_latent_channels(model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, _ = self.model.encode(x)
        return z_mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z)


def _infer_aekl_latent_channels(model) -> int:
    """Infer latent channels from MONAI AutoencoderKL instance."""
    for attr in ("latent_channels", "z_channels", "latent_dim"):
        v = getattr(model, attr, None)
        if isinstance(v, int):
            return v
    # Fallback: forward pass with dummy
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 32, 32, 32, device=next(model.parameters()).device)
            z_mu, _ = model.encode(dummy)
            return z_mu.shape[1]
    except Exception:
        return 4


# --------------------------------------------------------------------------- #
# MedVAE (StanfordMIMI)                                                       #
# --------------------------------------------------------------------------- #


class MedVAEWrapper(VAEWrapper):
    """Wrapper for MedVAE (frozen or fine-tuned).

    Handles both (mean, logvar) tuple output and direct tensor output.
    """

    def __init__(self, model, latent_ch: Optional[int] = None):
        super().__init__()
        self.model = model
        if latent_ch is not None:
            self.latent_channels = latent_ch
        else:
            self.latent_channels = _infer_medvae_latent_channels(model)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.model.encode(x)
        if isinstance(z, (tuple, list)):
            z = z[0]
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.model.decode(z)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


def _infer_medvae_latent_channels(model) -> int:
    """Infer latent channels via forward pass dummy."""
    try:
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 32, 32, 32)
            if next(model.parameters()).is_cuda:
                dummy = dummy.cuda()
            _z = model.encode(dummy)
            if isinstance(_z, (tuple, list)):
                _z = _z[0]
            return int(_z.shape[1])
    except Exception:
        return 1


# --------------------------------------------------------------------------- #
# VQ-VAE (NeuroQuant)                                                         #
# --------------------------------------------------------------------------- #


class VQVAEWrapper(VAEWrapper):
    """Wrapper for NeuroQuantHybrid / VQ-VAE.

    The CFM operates on the quantized anatomical latent z_q.
    encode() returns z_q and caches z_mod for decode().
    decode() uses cached z_mod or zeros if not available.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._cached_z_mod: Optional[torch.Tensor] = None
        self.latent_channels = getattr(
            getattr(model, "quantizer", None), "embedding_dim", 64
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_anat, z_mod = self.model.encoder(x)
        z_q, _, _ = self.model.quantizer(z_anat)
        self._cached_z_mod = z_mod.detach()
        return z_q

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        b, device = z_q.shape[0], z_q.device
        z_mod = self._cached_z_mod
        if z_mod is None or z_mod.shape[0] != b:
            mod_ch = getattr(
                getattr(getattr(self.model, "encoder", None), "mod_head", None),
                "out_channels",
                32,
            )
            z_mod = torch.zeros(b, mod_ch, *z_q.shape[2:], device=device)
        else:
            z_mod = z_mod.to(device)
        mod_idx = torch.zeros(b, dtype=torch.long, device=device)
        field_idx = torch.zeros(b, dtype=torch.long, device=device)
        return self.model.decoder(z_q, z_mod, mod_idx, field_idx)


# --------------------------------------------------------------------------- #
# MedVAE Disentanglement (v1)                                                 #
# --------------------------------------------------------------------------- #


class MedVAEDisentangleWrapper(VAEWrapper):
    """Wrapper for MedVAEDisentanglerV1.

    encode() returns the fused latent z_hat decodable by MedVAE.
    decode() calls MedVAE decode directly.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.latent_channels = model.latent_channels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, z_a, z_m = self.model.encode_parts(x)
        # Default modality = 0; caller should override via model-specific logic
        mod_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        z_hat = self.model.fuse_to_latent(z_a, z_m, mod_idx)
        return z_hat

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.model.medvae.decode(z)
        if isinstance(out, (tuple, list)):
            out = out[0]
        # Resize if needed
        out_shape = tuple(z.shape[2:])
        if out.shape[2:] != out_shape:
            out = F.interpolate(out, size=out_shape, mode="trilinear", align_corners=False)
        return out
