#!/usr/bin/env python3
"""MedVAE / MAISI wrapper — Phase D of MRIxFields 2026.

Fournit deux wrappers MRIxFieldsVAE autour des modèles pré-entraînés :

  MedVAEFineTuneWrapper
    - Wraps medvae.MVAE (StanfordMIMI, HuggingFace) avec accès à la
      distribution postérieure (DiagonalGaussianDistribution).
    - Modes : frozen (inférence seule) ou fine-tuning (loss KL + reconstruction).
    - Variantes : medvae_4_1_3d (4×, 1 canal latent) ou medvae_8_1_3d (8×).
    - encode() retourne le mode (µ) de la distribution postérieure.
    - encode_dist() retourne la DiagonalGaussianDistribution complète (pour training).
    - forward_train() retourne ModelOutput(loss, recon_loss, kl_loss, recon, z).

Usage :
    from models.maisi_vae import build_medvae_wrapper
    model = build_medvae_wrapper(model_name='medvae_4_1_3d', frozen=True)
    z = model.encode(x)    # (B, 1, H/4, W/4, D/4)
    recon = model.decode(z)

    # Fine-tuning
    model = build_medvae_wrapper(model_name='medvae_4_1_3d', frozen=False)
    out = model.forward_train(x)
    out.loss.backward()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vae_base import MRIxFieldsVAE
from models.vae_base import ModelOutput as MRIModelOutput


# --------------------------------------------------------------------------- #
# Helper                                                                       #
# --------------------------------------------------------------------------- #


def _infer_medvae_latent_shape(encode_fn, input_size=(64, 64, 64)) -> Tuple[int, ...]:
    """Infer latent (C, H', W', D') from a dummy forward pass."""
    # We need the model reference — encode_fn is a bound method
    model_self = encode_fn.__self__
    device = next(model_self.parameters()).device
    dummy = torch.zeros(1, 1, *input_size, device=device)
    with torch.no_grad():
        z = encode_fn(dummy)
    return tuple(z.shape[1:])


# --------------------------------------------------------------------------- #
# MedVAEFineTuneWrapper                                                        #
# --------------------------------------------------------------------------- #


class MedVAEFineTuneWrapper(MRIxFieldsVAE):
    """MedVAE 3D wrapper compatible MRIxFieldsVAE.

    Expose:
      encode(x)         -> z mode (B, C, H', W', D')
      encode_dist(x)    -> DiagonalGaussianDistribution (pour training)
      decode(z)         -> (B, 1, H, W, D)
      forward_train(x)  -> MRIModelOutput(loss, recon_loss, kl_loss, recon, z)

    Args:
        medvae_model  : instance medvae.MVAE déjà chargée
        kl_weight     : poids du terme KL dans la loss de fine-tuning
        frozen        : si True, gèle tous les paramètres sauf quant_conv
    """

    def __init__(
        self,
        medvae_model,
        kl_weight: float = 1e-6,
        frozen: bool = True,
    ):
        super().__init__()
        self.medvae = medvae_model       # medvae.MVAE instance
        self.inner  = medvae_model.model  # AutoencoderKL (inner model)
        self.kl_weight = kl_weight

        # Latent channels : quant_conv output = moments/2
        # For 4×1 model: inner.quant_conv outputs 2 channels → C_lat=1
        with torch.no_grad():
            device = next(self.inner.parameters()).device
            dummy = torch.zeros(1, 1, 64, 64, 64, device=device)
            z0 = self.medvae.encode(dummy)  # returns tensor (mode)
        self.latent_channels = int(z0.shape[1])
        self._latent_shape = tuple(z0.shape[1:])

        # Freeze if requested
        if frozen:
            self._freeze_all()
        else:
            # Unfreeze full model for fine-tuning
            for p in self.parameters():
                p.requires_grad_(True)

    def _freeze_all(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_decoder(self):
        """Unfreeze decoder only (partial fine-tuning)."""
        for p in self.inner.decoder.parameters():
            p.requires_grad_(True)
        for p in self.inner.post_quant_conv.parameters():
            p.requires_grad_(True)

    def unfreeze_all(self):
        """Unfreeze entire model."""
        for p in self.parameters():
            p.requires_grad_(True)

    # -- MRIxFieldsVAE contract ---------------------------------------------- #

    @property
    def latent_format(self):
        return "spatial"

    @property
    def latent_shape(self) -> Tuple[int, ...]:
        return self._latent_shape

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode → mode (µ) de la distribution postérieure.

        Args:
            x: (B, 1, H, W, D) in [-1, 1]

        Returns:
            z: (B, C, H', W', D')
        """
        return self.medvae.encode(x)  # returns mode directly

    def encode_dist(self, x: torch.Tensor):
        """Encode → DiagonalGaussianDistribution (pour fine-tuning)."""
        return self.inner.encode(x)   # returns posterior object

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent → volume.

        Args:
            z: (B, C, H', W', D')

        Returns:
            recon: (B, 1, H, W, D)
        """
        return self.medvae.decode(z)

    # -- Training ------------------------------------------------------------ #

    def forward_train(self, x: torch.Tensor) -> MRIModelOutput:
        """Full VAE forward for fine-tuning.

        Returns MRIModelOutput with:
            loss      : total loss (recon + kl_weight * kl)
            recon_loss: L1 reconstruction loss
            kl_loss   : mean KL divergence
            recon     : reconstructed volume (B, 1, H, W, D)
            z         : sampled latent (B, C, H', W', D')
        """
        posterior = self.inner.encode(x)
        z = posterior.sample()
        recon = self.inner.decode(z)

        recon_loss = F.l1_loss(recon, x)
        kl_loss = posterior.kl().mean()
        total = recon_loss + self.kl_weight * kl_loss

        return MRIModelOutput(
            loss=total,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            recon=recon,
            z=z,
        )


# --------------------------------------------------------------------------- #
# Builder                                                                      #
# --------------------------------------------------------------------------- #


def build_medvae_wrapper(
    model_name: str = "medvae_4_1_3d",
    frozen: bool = True,
    kl_weight: float = 1e-6,
    checkpoint: Optional[str] = None,
) -> MedVAEFineTuneWrapper:
    """Load MedVAE from HuggingFace and wrap as MedVAEFineTuneWrapper.

    Args:
        model_name : 'medvae_4_1_3d' (4×, 1-ch) or 'medvae_8_1_3d' (8×, 1-ch)
        frozen     : freeze all parameters (inference only)
        kl_weight  : KL weight for fine-tuning loss
        checkpoint : optional local checkpoint path (overrides HuggingFace weights)

    Returns:
        MedVAEFineTuneWrapper (on CPU; call .to(device) after)
    """
    try:
        from medvae import MVAE
    except ImportError:
        raise ImportError(
            "medvae not installed. "
            "pip install medvae  (or pip install git+https://github.com/StanfordMIMI/MedVAE)"
        )

    model = MVAE(model_name=model_name, modality="mri")

    if checkpoint is not None:
        import os
        from pathlib import Path
        ckpt_path = Path(checkpoint)
        if ckpt_path.exists():
            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            state = state.get("model", state)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"  [WARN] MedVAE fine-tune: {len(missing)} missing keys")
            if unexpected:
                print(f"  [WARN] MedVAE fine-tune: {len(unexpected)} unexpected keys")
            print(f"  MedVAE fine-tuned checkpoint loaded from {ckpt_path}")
        else:
            print(f"  [WARN] MedVAE checkpoint not found: {checkpoint} — using pretrained HuggingFace weights")

    wrapper = MedVAEFineTuneWrapper(model, kl_weight=kl_weight, frozen=frozen)
    return wrapper
