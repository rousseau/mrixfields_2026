"""
Generic patched VAE wrapper for full-resolution processing.

Handles patch extraction, encoding, decoding, and reconstruction blending
for any VAE architecture (AEKL, VQ-VAE, MedVAE).
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class PatchedVAE(nn.Module):
    """
    Wraps any VAE to process full-resolution volumes via patch-based inference.
    
    Supports:
    - Sliding window extraction with configurable overlap
    - Reconstruction blending for seamless output
    - Memory-efficient batch processing
    """

    def __init__(
        self,
        vae_model: nn.Module,
        patch_size: Tuple[int, int, int] = (112, 128, 80),
        stride: Optional[Tuple[int, int, int]] = None,
        overlap: float = 0.25,
        blend_mode: str = "gaussian",
    ):
        """
        Args:
            vae_model: base VAE (AutoencoderKL, NeuroQuantVQVAE, MVAE)
            patch_size: spatial size of each patch
            stride: step size for sliding window; if None, computed from overlap
            overlap: overlap ratio (0.0-1.0); ignored if stride is provided
            blend_mode: "gaussian" (smooth), "linear", or "ones" (average)
        """
        super().__init__()
        self.vae = vae_model
        self.patch_size = patch_size
        self.blend_mode = blend_mode

        # Compute stride from overlap if not provided
        if stride is None:
            stride = tuple(int(p * (1.0 - overlap)) for p in patch_size)
        self.stride = stride

        # Precompute blending weights (Hann window)
        self.register_buffer("blend_weights", self._create_blend_weights(patch_size, blend_mode))

    def _create_blend_weights(self, size: Tuple[int, int, int], mode: str) -> torch.Tensor:
        """Create blending weights for patch overlaps."""
        h, w, d = size
        if mode == "gaussian":
            # Gaussian falloff at edges
            wh = torch.exp(-((torch.arange(h, dtype=torch.float32) - h/2) ** 2) / (2 * (h/8) ** 2))
            ww = torch.exp(-((torch.arange(w, dtype=torch.float32) - w/2) ** 2) / (2 * (w/8) ** 2))
            wd = torch.exp(-((torch.arange(d, dtype=torch.float32) - d/2) ** 2) / (2 * (d/8) ** 2))
        elif mode == "linear":
            # Hann window
            wh = torch.hann_window(h, periodic=False)
            ww = torch.hann_window(w, periodic=False)
            wd = torch.hann_window(d, periodic=False)
        else:  # "ones"
            wh = torch.ones(h)
            ww = torch.ones(w)
            wd = torch.ones(d)

        # Combine into 3D weights: (H, W, D)
        weights = wh[:, None, None] * ww[None, :, None] * wd[None, None, :]
        return weights / (weights.max() + 1e-8)  # normalize to [0, 1]

    def _get_patches(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, Tuple[int, int, int]]]:
        """Extract patches from input volume."""
        b, c, h, w, d = x.shape
        ph, pw, pd = self.patch_size
        sh, sw, sd = self.stride

        patches = []
        positions = []

        # Sliding window extraction
        for i in range(0, h - ph + 1, sh):
            for j in range(0, w - pw + 1, sw):
                for k in range(0, d - pd + 1, sd):
                    patch = x[:, :, i:i+ph, j:j+pw, k:k+pd]
                    patches.append(patch)
                    positions.append((i, j, k))

        # Handle boundaries with padding
        # Right/bottom/depth boundaries
        for i in range(0, h - ph + 1, sh):
            for j in range(0, w - ph + 1, sw):
                if d + pd > x.shape[4]:  # depth boundary
                    i_start = max(0, h - ph)
                    j_start = max(0, w - pw)
                    k_start = max(0, d - pd)
                    patch = x[:, :, i_start:i_start+ph, j_start:j_start+pw, k_start:k_start+pd]
                    if (i_start, j_start, k_start) not in positions:
                        patches.append(patch)
                        positions.append((i_start, j_start, k_start))

        return patches, positions

    def encode(self, x: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        """
        Encode full-resolution volume into latent patches.
        
        Returns latent tensor with preserved spatial structure.
        """
        b, c, h, w, d = x.shape
        device = x.device

        # Extract patches
        patches, positions = self._get_patches(x)
        if not patches:
            raise ValueError(f"No patches extracted from volume {x.shape}")

        # Encode in batches
        latents = []
        for batch_start in range(0, len(patches), batch_size):
            batch_end = min(batch_start + batch_size, len(patches))
            batch_patches = torch.cat(patches[batch_start:batch_end], dim=0)

            with torch.no_grad():
                # Call appropriate encode method based on VAE type
                if hasattr(self.vae, 'encode'):  # AEKL, MedVAE
                    z = self.vae.encode(batch_patches)
                    if isinstance(z, tuple):
                        z = z[0]  # Handle (mean, logvar) output
                elif hasattr(self.vae, 'encoder'):  # VQ-VAE
                    z_anat, z_mod = self.vae.encoder(batch_patches)
                    z = z_anat
                else:
                    raise ValueError("VAE must have encode() or encoder attribute")

            latents.append(z.cpu())

        # Concatenate all latents
        all_latents = torch.cat(latents, dim=0)
        return all_latents, positions

    def decode(self, latents: torch.Tensor, positions: List[Tuple[int, int, int]], 
               full_shape: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
        """
        Decode latent patches and blend into full-resolution reconstruction.
        
        Args:
            latents: stacked latent patches (N_patches, latent_channels, h_lat, w_lat, d_lat)
            positions: list of (i, j, k) patch positions
            full_shape: target full-resolution shape (H, W, D)
            device: device for processing
        """
        h_full, w_full, d_full = full_shape
        ph, pw, pd = self.patch_size

        # Initialize accumulator and weight map
        reconstruction = torch.zeros(1, 1, h_full, w_full, d_full, device=device, dtype=torch.float32)
        weights_map = torch.zeros(1, 1, h_full, w_full, d_full, device=device, dtype=torch.float32)

        # Decode patches
        latents = latents.to(device)
        for patch_idx, (i, j, k) in enumerate(positions):
            z_patch = latents[patch_idx:patch_idx+1]

            with torch.no_grad():
                if hasattr(self.vae, 'decode'):  # AEKL, MedVAE
                    x_rec = self.vae.decode(z_patch)
                elif hasattr(self.vae, 'decoder'):  # VQ-VAE
                    # For VQ-VAE, need z_anat and z_mod; here use quantized + dummy mod
                    if hasattr(self.vae, 'quantizer'):
                        # Assuming z_patch is already quantized in encode path
                        # Create dummy z_mod (mean pooled or zeros)
                        z_mod_dummy = torch.zeros(z_patch.shape[0], 32, 
                                                 z_patch.shape[2], z_patch.shape[3], z_patch.shape[4],
                                                 device=device)
                        # Dummy modality/field indices
                        mod_idx = torch.zeros(z_patch.shape[0], dtype=torch.long, device=device)
                        field_idx = torch.zeros(z_patch.shape[0], dtype=torch.long, device=device)
                        x_rec = self.vae.decoder(z_patch, z_mod_dummy, mod_idx, field_idx)
                    else:
                        x_rec = self.vae.decoder(z_patch)
                else:
                    raise ValueError("VAE must have decode() or decoder attribute")

            # Ensure reconstruction is single channel
            if x_rec.shape[1] > 1:
                x_rec = x_rec[:, :1]

            # Blend into accumulator using weights
            weights = self.blend_weights.to(device)
            reconstruction[:, :, i:i+ph, j:j+pw, k:k+pd] += x_rec * weights[None, None, :, :, :]
            weights_map[:, :, i:i+ph, j:j+pw, k:k+pd] += weights[None, None, :, :, :]

        # Normalize by weights
        weights_map = torch.clamp(weights_map, min=1e-8)
        reconstruction = reconstruction / weights_map

        return reconstruction.squeeze(0).squeeze(0)  # Remove batch and channel dims

    def forward(self, x: torch.Tensor, encode_only: bool = False, 
                batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """
        Full encode-decode cycle on patched full-resolution volume.
        
        Args:
            x: input volume (1, 1, H, W, D)
            encode_only: if True, return only latent encoding
            batch_size: batch size for patch processing
        
        Returns:
            dict with 'latent' and optionally 'reconstruction'
        """
        device = x.device
        h, w, d = x.shape[2:]

        # Encode
        latents, positions = self.encode(x, batch_size=batch_size)

        result = {"latent": latents, "positions": positions}

        if not encode_only:
            # Decode
            reconstruction = self.decode(latents, positions, (h, w, d), device)
            result["reconstruction"] = reconstruction.to(device)

        return result


def create_patched_vae(
    vae: nn.Module,
    vae_type: str = "aekl",
    patch_size: Tuple[int, int, int] = (112, 128, 80),
    overlap: float = 0.25,
) -> PatchedVAE:
    """
    Factory function to create a PatchedVAE wrapper.
    
    Args:
        vae: base VAE model
        vae_type: "aekl", "vqvae", or "medvae"
        patch_size: spatial patch dimensions
        overlap: overlap ratio for sliding window
    
    Returns:
        PatchedVAE instance
    """
    return PatchedVAE(vae, patch_size=patch_size, overlap=overlap, blend_mode="gaussian")
