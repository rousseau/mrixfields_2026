#!/usr/bin/env python3
"""Base VAE abstraction for MRIxFields.

Defines MRIxFieldsVAE, the common interface for all VAE architectures
(AEKL, MedVAE, VQ-VAE, RHVAE, MAISI, Pythae-derived).
Key features:
- spatial vs vector latent format
- to_vector / from_vector helpers for cross-format Flow Matching
- full-volume inference helpers (patched sliding window)
- NIfTI latent extraction
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Literal, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn


class ModelOutput:
    """Lightweight container for encoder/decoder outputs.
    Compatible with pythae base_architectures.ModelOutput."""

    def __init__(self, **kwargs: torch.Tensor):
        self.__dict__.update(kwargs)


class MRIxFieldsVAE(nn.Module, ABC):
    """Abstract base for all VAE architectures in MRIxFields.

    Contract:
        - encode(x)  -> z  (latent representation)
        - decode(z)  -> recon  (image reconstruction)
        - latent_format  == "spatial" | "vector"
        - latent_shape   depends on format:
            spatial : Tuple[int, int, int, int] = (C, H', W', D')
            vector  : Tuple[int]               = (D_latent,)

    The model is **multi-modal by default** (trained on T1W+T2W+T2FLAIR).
    It does not receive modality/field labels at inference unless subclasses
    expose conditioning methods.
    """

    # ----------------------------------------------------------------------- #
    # Subclass contract                                                       #
    # ----------------------------------------------------------------------- #

    latent_channels: int  # used by downstream CFM for channel-in or dim

    @property
    @abstractmethod
    def latent_format(self) -> Literal["spatial", "vector"]:
        """"spatial" if z is (B,C,H',W',D'), "vector" if z is (B,D_lat)."""
        ...

    @property
    def latent_shape(self) -> Tuple[int, ...]:
        """Shape of a single latent sample.
        Must be overridden if not inferable from a dummy pass.
        """
        return ()

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input volume -> latent.

        Args:
            x: (B, 1, H, W, D)  – single-channel 3D volume, range [-1, 1]

        Returns:
            z: (B, C, H', W', D')  if spatial
               (B, D_lat)          if vector
        """

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent -> volume.

        Args:
            z: depend on `latent_format`

        Returns:
            recon: (B, 1, H, W, D) in [-1, 1] range
        """

    # ----------------------------------------------------------------------- #
    # Convenience forward / helpers                                           #
    # ----------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full encode-decode. Returns (recon, z)."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    # ----------------------------------------------------------------------- #
    # Spatial <-> Vector conversion for Flow Matching                         #
    # ----------------------------------------------------------------------- #

    def to_vector(self, z: torch.Tensor) -> torch.Tensor:
        """Flatten a spatial latent into a vector.

        spatial (B,C,H',W',D') -> (B, C*H'*W'*D')
        vector  (B,D_lat)      -> identity
        """
        if self.latent_format == "spatial":
            return z.flatten(start_dim=1)
        return z

    def from_vector(self, z_vec: torch.Tensor) -> torch.Tensor:
        """Reshape a vector back into spatial latent (only for spatial VAEs).

        vector  (B,D_flat) -> (B,C,H',W',D')   if spatial
        vector  (B,D_lat)  -> identity          if vector

        NOTE: prefer LatentVectorizer(latent_shape).unflatten() when the
        exact latent_shape is known at call-site (e.g. from _infer_latent_shape).
        This method tries to reconstruct the shape from D_flat + latent_channels,
        first as a cube, then falling back to the stored latent_shape if the
        flat dimension matches.
        """
        if self.latent_format == "spatial":
            C = self.latent_channels
            D_flat = z_vec.shape[1]
            spatial = D_flat // C
            # Try perfect cube
            s = round(spatial ** (1/3))
            if s ** 3 == spatial:
                return z_vec.view(-1, C, s, s, s)
            # Fallback: stored latent_shape if total elements match
            stored_total = 1
            for v in self.latent_shape:
                stored_total *= v
            if stored_total == D_flat:
                return z_vec.view(-1, *self.latent_shape)
            raise ValueError(
                f"from_vector: cannot infer spatial shape from D_flat={D_flat}, "
                f"latent_channels={C}, stored latent_shape={self.latent_shape}. "
                "Use LatentVectorizer(latent_shape).unflatten(z_vec) instead."
            )
        return z_vec

    @property
    def vector_dim(self) -> int:
        """Dimension of the flattened / vector latent."""
        d = 1
        for s in self.latent_shape:
            d *= s
        return int(d)

    # ----------------------------------------------------------------------- #
    # Full-volume (patched) inference                                         #
    # ----------------------------------------------------------------------- #

    @torch.no_grad()
    def infer_full_volume(
        self,
        volume: Union[np.ndarray, Path, str],
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        overlap: float = 0.25,
        blend_mode: Literal["gaussian", "linear", "ones"] = "gaussian",
        normalize: bool = True,
        percentile_lower: float = 0.5,
        percentile_upper: float = 99.5,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 1,
    ) -> np.ndarray:
        """Reconstruct a full-resolution volume via patch-based sliding window.

        Args:
            volume: numpy array (H,W,D) or path to .nii.gz
            patch_size: spatial size of each patch
            overlap: overlap ratio [0, 1)
            blend_mode: how to blend overlapping patches
            normalize: whether to apply percentile normalization before encoding
            percentile_lower, percentile_upper: normalization bounds
            device: compute device; defaults to model's current device
            batch_size: number of patches processed in parallel

        Returns:
            reconstruction: np.ndarray (H, W, D) in original spatial shape
        """
        from utils.patched_vae import PatchedVAE

        # Load volume
        if isinstance(volume, (str, Path)):
            img = nib.load(str(volume))
            vol = img.get_fdata(dtype=np.float32)
        else:
            vol = volume.astype(np.float32)

        orig_shape = vol.shape

        # Normalize
        if normalize:
            lo = np.percentile(vol, percentile_lower)
            hi = np.percentile(vol, percentile_upper)
            if hi > lo:
                vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
                vol = (vol * 2.0 - 1.0).astype(np.float32)
            else:
                vol = np.zeros_like(vol, dtype=np.float32)

        x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)

        if device is None:
            device = next(self.parameters()).device
        x = x.to(device)

        # Patched inference
        wrapper = PatchedVAE(
            self,
            patch_size=patch_size,
            overlap=overlap,
            blend_mode=blend_mode,
        )
        wrapper = wrapper.to(device)
        result = wrapper.forward(x, encode_only=False, batch_size=batch_size)
        recon = result["reconstruction"].squeeze().cpu().numpy()

        # Ensure original shape (PatcheVAE may pad slightly)
        if recon.shape != orig_shape:
            recon = self._crop_or_pad_np(recon, orig_shape)

        return recon

    @torch.no_grad()
    def extract_latent_nifti(
        self,
        volume: Union[np.ndarray, Path, str],
        output_path: Optional[Union[str, Path]] = None,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        overlap: float = 0.25,
        device: Optional[Union[str, torch.device]] = None,
        compress: bool = False,
    ) -> nib.Nifti1Image:
        """Extract latent representation and save as NIfTI.

        For spatial VAEs: returns a 4D NIfTI (H',W',D',C).
        For vector VAEs: returns a 1D NIfTI (D_lat,).

        Args:
            volume: input volume or path
            output_path: optional path to save .nii.gz
            patch_size, overlap, device: passed to patched inference
            compress: gzip the output
        """
        from utils.patched_vae import PatchedVAE

        # Load volume
        if isinstance(volume, (str, Path)):
            img = nib.load(str(volume))
            affine = img.affine.copy().astype(float)
            vol = img.get_fdata(dtype=np.float32)
        else:
            vol = volume.astype(np.float32)
            affine = np.eye(4)

        # Normalize
        lo = np.percentile(vol, 0.5)
        hi = np.percentile(vol, 99.5)
        if hi > lo:
            vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
            vol = (vol * 2.0 - 1.0).astype(np.float32)
        else:
            vol = np.zeros_like(vol, dtype=np.float32)

        x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
        if device is None:
            device = next(self.parameters()).device
        x = x.to(device)

        # Encode patched
        wrapper = PatchedVAE(
            self,
            patch_size=patch_size,
            overlap=overlap,
            blend_mode="ones",  # no blending needed for latents average
        )
        wrapper = wrapper.to(device)
        result = wrapper.forward(x, encode_only=True, batch_size=1)
        latents = result["latent"].cpu().numpy()  # (N_patches, C, h', w', d')

        # Spatial VAE: aggregate latent patches (simple averaging)
        if self.latent_format == "spatial":
            #TODO: implement proper latent blending for spatial aggregation
            # For now, use first patch's latent as proxy
            latent_vol = latents[0]  # (C, h', w', d')
            latent_vol = np.transpose(latent_vol, (1, 2, 3, 0))  # (h',w',d',C)
        else:
            latent_vol = latents[0]  # (D_lat,)

        nifti = nib.Nifti1Image(latent_vol.astype(np.float32), affine)

        if output_path is not None:
            nib.save(nifti, str(output_path))

        return nifti

    # ----------------------------------------------------------------------- #
    # Static helpers                                                          #
    # ----------------------------------------------------------------------- #

    @staticmethod
    def _crop_or_pad_np(vol: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
        """Center-crop or pad a 3D numpy array to target shape."""
        th, tw, td = target
        h, w, d = vol.shape[:3]
        # pad
        ph = max(0, th - h)
        pw = max(0, tw - w)
        pd = max(0, td - d)
        if ph or pw or pd:
            vol = np.pad(
                vol,
                [(ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (pd // 2, pd - pd // 2)],
                mode="reflect",
            )
            h, w, d = vol.shape[:3]
        # crop
        sh = (h - th) // 2
        sw = (w - tw) // 2
        sd = (d - td) // 2
        return vol[sh : sh + th, sw : sw + tw, sd : sd + td]
