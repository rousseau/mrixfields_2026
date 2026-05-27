#!/usr/bin/env python3
"""Pythae VQ-VAE 3D — 5D Quantizer + MRIxFieldsVAE wrapper for MRIxFields.

Architecture:
  Encoder:  3D conv stack → spatial embeddings (B, C, H', W', D')
  Quantizer: adapted Pythae Quantizer to 5D (B, H', W', D', C)
  Decoder:  3D transposed conv stack (B, C, H', W', D') → (B, 1, H, W, D)

The key adaptation over Pythae's built-in VQ-VAE is the Quantizer5D class,
which handles 5D tensors (B, C, D, H, W) instead of 4D (B, C, H, W).

Shares the same Encoder3D / Decoder3D backbone as pythae_vae.py, but the
encoder does NOT produce mu/log_var — it produces a single embedding tensor
passed directly to the quantizer.

Usage (standalone):
    from models.pythae_vqvae import build_pythae_vqvae_3d
    vae = build_pythae_vqvae_3d(latent_channels=8, base_channels=32,
                                 num_embeddings=1024)

Usage (training loop):
    vae = build_pythae_vqvae_3d(...)
    output = vae.forward_train(x)   # returns ModelOutput with loss
    loss = output.loss.mean()
    loss.backward()
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.models.vq_vae.vq_vae_config import VQVAEConfig

from models.vae_base import MRIxFieldsVAE
from models.pythae_vae import ResBlock3D, _make_norm


# --------------------------------------------------------------------------- #
# 5D Quantizer                                                                #
# --------------------------------------------------------------------------- #


class Quantizer5D(nn.Module):
    """Vector quantizer adapted for 5D spatial latents (B, C, H', W', D').

    Adaptation of Pythae's Quantizer (2D) to 3D volumes:
      - Input/output: (B, C, H', W', D')
      - Internal operation on (B*H'*W'*D', C) flattened vectors

    Supports EMA codebook update (use_ema=True) for stable training.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 8,
        commitment_loss_factor: float = 0.25,
        quantization_loss_factor: float = 1.0,
        use_ema: bool = True,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_loss_factor = commitment_loss_factor
        self.quantization_loss_factor = quantization_loss_factor
        self.use_ema = use_ema
        self.decay = decay
        self.eps = eps

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1 / num_embeddings, 1 / num_embeddings)

        if use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
            self.register_buffer("ema_w", self.embeddings.weight.clone())

    def forward(self, z: torch.Tensor, uses_ddp: bool = False) -> ModelOutput:
        """Quantize 5D spatial latent.

        Args:
            z: (B, C, H', W', D')

        Returns:
            ModelOutput with:
              quantized_vector: (B, C, H', W', D')
              quantized_indices: (B, 1, H', W', D')
              loss: scalar
        """
        B, C, H, W, D = z.shape
        # (B, C, H, W, D) → (B, H, W, D, C) → (N, C)  with N = B*H*W*D
        z_flat = z.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (N, C)

        # Compute distances to codebook entries
        distances = (
            (z_flat ** 2).sum(dim=-1, keepdim=True)
            + (self.embeddings.weight ** 2).sum(dim=-1)
            - 2 * z_flat @ self.embeddings.weight.T
        )  # (N, K)

        closest = distances.argmin(-1)  # (N,)

        # One-hot for quantization
        one_hot = F.one_hot(closest, num_classes=self.num_embeddings).float()  # (N, K)
        quantized_flat = one_hot @ self.embeddings.weight  # (N, C)

        # EMA codebook update (training only)
        if self.use_ema and self.training:
            with torch.no_grad():
                n = one_hot.sum(0)  # (K,)
                if uses_ddp:
                    torch.distributed.all_reduce(n)
                self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * n
                # Laplace smoothing
                n_smooth = (
                    (self.ema_cluster_size + self.eps)
                    / (B * H * W * D + self.num_embeddings * self.eps)
                    * B * H * W * D
                )
                dw = one_hot.T @ z_flat  # (K, C)
                if uses_ddp:
                    torch.distributed.all_reduce(dw)
                self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
                self.embeddings.weight.data = self.ema_w / n_smooth.unsqueeze(1)

        # Straight-through estimator
        quantized_flat_st = z_flat + (quantized_flat - z_flat).detach()

        # Losses
        commitment_loss = F.mse_loss(quantized_flat.detach(), z_flat)
        embedding_loss = F.mse_loss(quantized_flat, z_flat.detach())
        loss = (
            self.commitment_loss_factor * commitment_loss
            + self.quantization_loss_factor * embedding_loss
        )

        # Reshape back to (B, C, H, W, D)
        quantized = quantized_flat_st.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        indices = closest.reshape(B, 1, H, W, D)

        return ModelOutput(
            quantized_vector=quantized,
            quantized_indices=indices,
            loss=loss,
        )


# --------------------------------------------------------------------------- #
# Encoder (no mu/log_var split — single embedding)                            #
# --------------------------------------------------------------------------- #


class VQEncoder3D(BaseEncoder):
    """3D encoder for VQ-VAE (no mu/log_var split).

    Output ModelOutput with `embedding` = (B, latent_channels, H/8, W/8, D/8).
    """

    def __init__(
        self,
        model_config: VQVAEConfig,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_channels: int = 8,
        num_groups: int = 8,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        C = base_channels

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, C, 3, padding=1),
            ResBlock3D(C, num_groups),
            nn.Conv3d(C, C * 2, 4, stride=2, padding=1),
            _make_norm(C * 2, num_groups), nn.SiLU(),
            ResBlock3D(C * 2, num_groups),
            nn.Conv3d(C * 2, C * 4, 4, stride=2, padding=1),
            _make_norm(C * 4, num_groups), nn.SiLU(),
            ResBlock3D(C * 4, num_groups),
            nn.Conv3d(C * 4, C * 4, 4, stride=2, padding=1),
            _make_norm(C * 4, num_groups), nn.SiLU(),
            ResBlock3D(C * 4, num_groups),
            nn.Conv3d(C * 4, latent_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        embedding = self.encoder(x)  # (B, latent_channels, H/8, W/8, D/8)
        return ModelOutput(embedding=embedding)


# --------------------------------------------------------------------------- #
# Decoder (same as VAE decoder but takes quantized vector directly)           #
# --------------------------------------------------------------------------- #


class VQDecoder3D(BaseDecoder):
    """3D decoder for VQ-VAE.

    Input:  (B, latent_channels, H/8, W/8, D/8)
    Output: ModelOutput with `reconstruction` (B, 1, H, W, D) in [-1, 1].
    """

    def __init__(
        self,
        model_config: VQVAEConfig,
        out_channels: int = 1,
        base_channels: int = 32,
        latent_channels: int = 8,
        num_groups: int = 8,
    ):
        super().__init__()
        C = base_channels

        self.decoder = nn.Sequential(
            nn.Conv3d(latent_channels, C * 4, 3, padding=1),
            ResBlock3D(C * 4, num_groups),
            nn.ConvTranspose3d(C * 4, C * 4, 4, stride=2, padding=1),
            _make_norm(C * 4, num_groups), nn.SiLU(),
            ResBlock3D(C * 4, num_groups),
            nn.ConvTranspose3d(C * 4, C * 2, 4, stride=2, padding=1),
            _make_norm(C * 2, num_groups), nn.SiLU(),
            ResBlock3D(C * 2, num_groups),
            nn.ConvTranspose3d(C * 2, C, 4, stride=2, padding=1),
            _make_norm(C, num_groups), nn.SiLU(),
            ResBlock3D(C, num_groups),
            nn.Conv3d(C, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> ModelOutput:
        return ModelOutput(reconstruction=self.decoder(z))


# --------------------------------------------------------------------------- #
# MRIxFieldsVAE wrapper                                                       #
# --------------------------------------------------------------------------- #


class PythaeVQVAE3D(MRIxFieldsVAE):
    """MRIxFieldsVAE wrapper around a 3D VQ-VAE with 5D quantizer.

    encode() returns the quantized latent z_q (straight-through, no gradient
    through quantization at inference time).
    For training, use forward_train() which also returns the VQ loss.

    Construct with build_pythae_vqvae_3d() or PythaeVQVAE3D.from_checkpoint().
    """

    def __init__(
        self,
        encoder: VQEncoder3D,
        quantizer: Quantizer5D,
        decoder: VQDecoder3D,
        latent_channels: int = 8,
    ):
        super().__init__()
        self.encoder_net = encoder
        self.quantizer = quantizer
        self.decoder_net = decoder
        self.latent_channels = latent_channels
        self._latent_shape: Tuple[int, ...] = self._infer_latent_shape()

    def _infer_latent_shape(self) -> Tuple[int, ...]:
        device = next(self.parameters()).device
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 32, 32, 32, device=device)
            out = self.encoder_net(dummy)
        return tuple(out.embedding.shape[1:])

    @property
    def latent_format(self):
        return "spatial"

    @property
    def latent_shape(self) -> Tuple[int, ...]:
        return self._latent_shape

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode + quantize. Returns z_q (straight-through at training,
        hard quantized at inference)."""
        enc_out = self.encoder_net(x)
        q_out = self.quantizer(enc_out.embedding)
        return q_out.quantized_vector  # (B, C, H', W', D')

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder_net(z)
        return out.reconstruction

    def forward_train(self, x: torch.Tensor) -> ModelOutput:
        """Full training forward: encode → quantize → decode + VQ loss.

        Returns ModelOutput with:
          recon:  (B, 1, H, W, D)
          loss:   scalar (recon_loss + vq_loss) — use for .backward()
          recon_loss: reconstruction MSE
          vq_loss:    commitment + embedding loss
        """
        enc_out = self.encoder_net(x)
        q_out = self.quantizer(enc_out.embedding)
        dec_out = self.decoder_net(q_out.quantized_vector)
        recon = dec_out.reconstruction

        recon_loss = F.mse_loss(recon, x)
        vq_loss = q_out.loss
        total_loss = recon_loss + vq_loss

        return ModelOutput(
            recon=recon,
            loss=total_loss,
            recon_loss=recon_loss,
            vq_loss=vq_loss,
        )

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, **kwargs) -> "PythaeVQVAE3D":
        """Load from a state-dict checkpoint."""
        model = build_pythae_vqvae_3d(
            latent_channels=kwargs.get("latent_channels", 8),
            base_channels=kwargs.get("base_channels", 32),
            num_embeddings=kwargs.get("num_embeddings", 512),
            num_groups=kwargs.get("num_groups", 8),
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        model.load_state_dict(state, strict=True)
        return model


# --------------------------------------------------------------------------- #
# Convenience builder                                                          #
# --------------------------------------------------------------------------- #


def build_pythae_vqvae_3d(
    latent_channels: int = 8,
    base_channels: int = 32,
    num_embeddings: int = 512,
    commitment_loss_factor: float = 0.25,
    quantization_loss_factor: float = 1.0,
    use_ema: bool = True,
    decay: float = 0.99,
    num_groups: int = 8,
) -> PythaeVQVAE3D:
    """Build a PythaeVQVAE3D from scratch.

    Args:
        latent_channels: number of channels in the quantized latent space
        base_channels: base channel width for encoder/decoder (×1, ×2, ×4)
        num_embeddings: codebook size
        commitment_loss_factor: weight for commitment loss (beta in VQ-VAE)
        quantization_loss_factor: weight for embedding loss
        use_ema: use EMA codebook updates (recommended for stability)
        decay: EMA decay rate
        num_groups: GroupNorm groups

    Returns:
        PythaeVQVAE3D instance
    """
    cfg = VQVAEConfig(
        input_dim=(1, 128, 128, 128),
        latent_dim=latent_channels * 16 * 16 * 16,
        num_embeddings=num_embeddings,
        use_ema=use_ema,
        decay=decay,
        commitment_loss_factor=commitment_loss_factor,
        quantization_loss_factor=quantization_loss_factor,
    )
    encoder = VQEncoder3D(cfg, latent_channels=latent_channels, base_channels=base_channels, num_groups=num_groups)
    quantizer = Quantizer5D(
        num_embeddings=num_embeddings,
        embedding_dim=latent_channels,
        commitment_loss_factor=commitment_loss_factor,
        quantization_loss_factor=quantization_loss_factor,
        use_ema=use_ema,
        decay=decay,
    )
    decoder = VQDecoder3D(cfg, latent_channels=latent_channels, base_channels=base_channels, num_groups=num_groups)
    return PythaeVQVAE3D(encoder=encoder, quantizer=quantizer, decoder=decoder, latent_channels=latent_channels)
