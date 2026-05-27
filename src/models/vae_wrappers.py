#!/usr/bin/env python3
"""Unified VAE wrapper interfaces — all architectures share MRIxFieldsVAE base.

Wrappers provided:
  - AEKLWrapper        (MONAI AutoencoderKL)
  - MedVAEWrapper      (MedVAE frozen / fine-tuned)
  - VQVAEWrapper       (NeuroQuantHybrid — spatial quantized latent)
  - MedVAEDisentangleWrapper  (MedVAE frozen + anatomie/modalité projection)

All inherit MRIxFieldsVAE and expose:
  latent_format == "spatial"
  latent_shape  == (C, H', W', D')  (inferred at init via dummy pass)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vae_base import MRIxFieldsVAE

# Rétrocompatibilité : alias pour les imports externes
VAEWrapper = MRIxFieldsVAE


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _infer_latent_shape_spatial(
    encode_fn, dummy_size: Tuple[int, int, int] = (32, 32, 32)
) -> Tuple[int, int, int, int]:
    """Infer (C, H', W', D') by encoding a dummy tensor."""
    device = next(encode_fn.__self__.parameters()).device
    dummy = torch.zeros(1, 1, *dummy_size, device=device)
    with torch.no_grad():
        z = encode_fn(dummy)
    return tuple(z.shape[1:])


def _infer_aekl_latent_channels(model) -> int:
    for attr in ("latent_channels", "z_channels", "latent_dim"):
        v = getattr(model, attr, None)
        if isinstance(v, int):
            return v
    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            dummy = torch.zeros(1, 1, 32, 32, 32, device=device)
            z_mu, _ = model.encode(dummy)
            return z_mu.shape[1]
    except Exception:
        return 4


def _infer_medvae_latent_channels(model) -> int:
    try:
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            dummy = torch.zeros(1, 1, 32, 32, 32, device=device)
            _z = model.encode(dummy)
            if isinstance(_z, (tuple, list)):
                _z = _z[0]
            return int(_z.shape[1])
    except Exception:
        return 1


# --------------------------------------------------------------------------- #
# AEKL (MONAI AutoencoderKL)                                                  #
# --------------------------------------------------------------------------- #


class AEKLWrapper(MRIxFieldsVAE):
    """Wrapper for MONAI AutoencoderKL — deterministic mean encoding."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.latent_channels = _infer_aekl_latent_channels(model)
        # Infer spatial latent shape via dummy pass
        self._latent_shape = _infer_latent_shape_spatial(self.encode)

    @property
    def latent_format(self):
        return "spatial"

    @property
    def latent_shape(self) -> Tuple[int, ...]:
        return self._latent_shape

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, _ = self.model.encode(x)
        return z_mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z)


# --------------------------------------------------------------------------- #
# MedVAE (StanfordMIMI)                                                       #
# --------------------------------------------------------------------------- #


class MedVAEWrapper(MRIxFieldsVAE):
    """Wrapper for MedVAE (frozen or fine-tuned).

    encode() returns the mean (first element if tuple).
    decode() unwraps tuple/list outputs.
    """

    def __init__(self, model, latent_ch: Optional[int] = None):
        super().__init__()
        self.model = model
        self.latent_channels = latent_ch if latent_ch is not None else _infer_medvae_latent_channels(model)
        self._latent_shape = _infer_latent_shape_spatial(self.encode)

    @property
    def latent_format(self):
        return "spatial"

    @property
    def latent_shape(self) -> Tuple[int, ...]:
        return self._latent_shape

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


# --------------------------------------------------------------------------- #
# VQ-VAE (NeuroQuantHybrid)                                                   #
# --------------------------------------------------------------------------- #


class VQVAEWrapper(MRIxFieldsVAE):
    """Wrapper for NeuroQuantHybrid VQ-VAE.

    The CFM operates on the quantized anatomical latent z_q.
    encode() returns z_q and caches z_mod for decode().
    decode() uses cached z_mod or zeros if unavailable.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._cached_z_mod: Optional[torch.Tensor] = None
        self.latent_channels = getattr(
            getattr(model, "quantizer", None), "embedding_dim", 64
        )
        self._latent_shape = _infer_latent_shape_spatial(self.encode)

    @property
    def latent_format(self):
        return "spatial"

    @property
    def latent_shape(self) -> Tuple[int, ...]:
        return self._latent_shape

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_anat, z_mod = self.model.encoder(x)
        # Note: NeuroQuant quantizer may return (z_q, vq_loss, indices, perplexity)
        q_out = self.model.quantizer(z_anat)
        if isinstance(q_out, tuple):
            z_q = q_out[0]
        else:
            z_q = q_out
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
        # NeuroQuant decoder signature may vary; try common variants
        try:
            return self.model.decoder(z_q, z_mod, mod_idx, field_idx)
        except TypeError:
            try:
                return self.model.decoder(z_q, z_mod)
            except TypeError:
                # VQVAECompatWrapper style
                return self.model.decoder(z_q)


# --------------------------------------------------------------------------- #
# MedVAE Disentanglement (v1)                                                 #
# --------------------------------------------------------------------------- #


class MedVAEDisentangleWrapper(MRIxFieldsVAE):
    """Wrapper for MedVAEDisentanglerV1.

    encode() returns the fused latent z_hat decodable by MedVAE (spatial).
    decode() calls MedVAE decode directly.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.latent_channels = model.latent_channels
        self._latent_shape = _infer_latent_shape_spatial(self.encode)

    @property
    def latent_format(self):
        return "spatial"

    @property
    def latent_shape(self) -> Tuple[int, ...]:
        return self._latent_shape

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, z_a, z_m = self.model.encode_parts(x)
        mod_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        z_hat = self.model.fuse_to_latent(z_a, z_m, mod_idx)
        return z_hat

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.model.medvae.decode(z)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.shape[2:] != z.shape[2:]:
            out = F.interpolate(out, size=z.shape[2:], mode="trilinear", align_corners=False)
        return out
