#!/usr/bin/env python3
"""Unified NIfTI I/O and image preprocessing utilities.

Extracted and consolidated from:
  - src/cfm/train_cfm_3d.py
  - src/vae3d/train_vae_3d.py
  - src/vae3d/train_vqvae.py
  - src/vae3d/benchmark_vae.py
  - src/cfm/train_mmfm_3d.py
  - src/evaluation/evaluate.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom as scipy_zoom

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

DOMAINS: List[str] = ["0.1T", "1.5T", "3T", "5T", "7T"]
DOMAIN_TO_IDX: dict = {d: i for i, d in enumerate(DOMAINS)}
NUM_DOMAINS = len(DOMAINS)

MODALITIES: List[str] = ["T1W", "T2W", "T2FLAIR"]
MODALITY_TO_IDX: dict = {m: i for i, m in enumerate(MODALITIES)}
NUM_MODALITIES = len(MODALITIES)

SPLIT_MAP: dict = {
    "retro_train": "Training_retrospective",
    "pro_train": "Training_prospective",
    "pro_val": "Validating_prospective",
    "pro_test": "Testing_prospective",
}
IDX_TO_SPLIT = {v: k for k, v in SPLIT_MAP.items()}

# Filename regex: {R,P}_{modality}_{field}_{subject_id}.nii.gz
FILE_RE = re.compile(r"^[A-Z]_([A-Z0-9]+)_([0-9.]+T)_(\d+)\.nii\.gz$")


# --------------------------------------------------------------------------- #
# Volume preprocessing                                                        #
# --------------------------------------------------------------------------- #


def normalize_volume(
    vol: np.ndarray,
    lo_pct: float = 0.5,
    hi_pct: float = 99.5,
) -> np.ndarray:
    """Percentile normalization → [0, 1] → [-1, 1].

    Args:
        vol: Input volume (any shape).
        lo_pct: Lower percentile for clipping.
        hi_pct: Upper percentile for clipping.

    Returns:
        Normalized volume in [-1, 1] as float32.
    """
    lo = np.percentile(vol, lo_pct)
    hi = np.percentile(vol, hi_pct)
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return (vol * 2.0 - 1.0).astype(np.float32)


def resample_volume(
    vol: np.ndarray,
    original_spacing,
    target_spacing: Tuple[float, float, float],
) -> np.ndarray:
    """Rééchantillonne un volume 3D vers target_spacing (mm).

    Exemple : 364×436×364 @ 0.5mm → 182×218×182 @ 1mm (8× moins de voxels).
    """
    orig = np.asarray(original_spacing[:3], dtype=float)
    tgt = np.asarray(target_spacing, dtype=float)
    factors = orig / tgt
    if np.allclose(factors, 1.0, atol=0.02):
        return vol.astype(np.float32)
    return scipy_zoom(vol, factors, order=1).astype(np.float32)


def center_crop_or_pad_np(
    vol: np.ndarray,
    target_size: Tuple[int, int, int],
    mode: str = "reflect",
) -> np.ndarray:
    """Centre-crop ou pad un volume 3D NumPy vers target_size (H, W, D).

    Args:
        vol: Input volume (H, W, D).
        target_size: Target shape (th, tw, td).
        mode: Padding mode for np.pad.

    Returns:
        Volume of shape target_size.
    """
    th, tw, td = target_size
    h, w, d = vol.shape[:3]

    # Pad si le volume est trop petit
    ph = max(0, th - h)
    pw = max(0, tw - w)
    pd = max(0, td - d)
    if ph > 0 or pw > 0 or pd > 0:
        vol = np.pad(
            vol,
            [(ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (pd // 2, pd - pd // 2)],
            mode=mode,
        )
        h, w, d = vol.shape[:3]

    # Crop centré
    sh = max((h - th) // 2, 0)
    sw = max((w - tw) // 2, 0)
    sd = max((d - td) // 2, 0)
    return vol[sh : sh + th, sw : sw + tw, sd : sd + td]


def random_crop_or_pad_np(
    vol: np.ndarray,
    target_size: Tuple[int, int, int],
    mode: str = "reflect",
) -> np.ndarray:
    """Random-crop ou pad un volume 3D NumPy vers target_size (H, W, D).

    Le coin supérieur-gauche du crop est tiré aléatoirement dans les limites
    valides. Si le volume est plus petit que target_size dans une dimension,
    on pad centré comme dans center_crop_or_pad_np.

    Args:
        vol: Input volume (H, W, D).
        target_size: Target shape (th, tw, td).
        mode: Padding mode for np.pad.

    Returns:
        Volume of shape target_size.
    """
    th, tw, td = target_size
    h, w, d = vol.shape[:3]

    # Pad si le volume est trop petit (centré, pour garder tout le contenu)
    ph = max(0, th - h)
    pw = max(0, tw - w)
    pd = max(0, td - d)
    if ph > 0 or pw > 0 or pd > 0:
        vol = np.pad(
            vol,
            [(ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (pd // 2, pd - pd // 2)],
            mode=mode,
        )
        h, w, d = vol.shape[:3]

    # Tirage aléatoire des offsets de crop
    sh = np.random.randint(0, h - th + 1) if h > th else 0
    sw = np.random.randint(0, w - tw + 1) if w > tw else 0
    sd = np.random.randint(0, d - td + 1) if d > td else 0
    return vol[sh : sh + th, sw : sw + tw, sd : sd + td]


def center_crop_or_pad_tensor(
    tensor: torch.Tensor,
    target_size: Tuple[int, int, int],
) -> torch.Tensor:
    """Centre-crop ou pad un tensor 5D (B, C, H, W, D)."""
    th, tw, td = target_size
    h, w, d = tensor.shape[2:]

    ph = max(0, th - h)
    pw = max(0, tw - w)
    pd = max(0, td - d)
    if ph > 0 or pw > 0 or pd > 0:
        tensor = torch.nn.functional.pad(
            tensor,
            (
                pd // 2,
                pd - pd // 2,
                pw // 2,
                pw - pw // 2,
                ph // 2,
                ph - ph // 2,
            ),
            mode="reflect",
        )
        h, w, d = tensor.shape[2:]

    sh = max((h - th) // 2, 0)
    sw = max((w - tw) // 2, 0)
    sd = max((d - td) // 2, 0)
    return tensor[:, :, sh : sh + th, sw : sw + tw, sd : sd + td]


def load_nifti_volume(
    path: Path,
    target_spacing: Optional[Tuple[float, float, float]] = None,
    volume_size: Optional[Tuple[int, int, int]] = None,
    normalize: bool = True,
    lo_pct: float = 0.5,
    hi_pct: float = 99.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Charge un volume NIfTI, optionnellement le rééchantillonne, crop/pad, normalise.

    Args:
        path: Path to .nii.gz file.
        target_spacing: Target voxel spacing (mm). None = no resampling.
        volume_size: Target shape (H, W, D). None = no crop/pad.
        normalize: Whether to apply percentile normalization.
        lo_pct: Lower percentile for normalization.
        hi_pct: Upper percentile for normalization.

    Returns:
        (volume_array, affine_matrix)
    """
    img = nib.load(str(path))
    affine = img.affine.copy().astype(float)
    vol = img.get_fdata(dtype=np.float32)

    if target_spacing is not None:
        spacing = np.abs(np.diag(affine)[:3])
        vol = resample_volume(vol, spacing, target_spacing)

    if volume_size is not None:
        vol = center_crop_or_pad_np(vol, volume_size)

    if normalize:
        vol = normalize_volume(vol, lo_pct, hi_pct)

    return vol, affine


def volume_to_tensor(vol: np.ndarray) -> torch.Tensor:
    """Convert a numpy volume (H, W, D) to a 5D torch tensor (1, 1, H, W, D)."""
    return torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)


def tensor_to_volume(tensor: torch.Tensor) -> np.ndarray:
    """Convert a 5D torch tensor (B, 1, H, W, D) to a numpy volume (H, W, D)."""
    return tensor.squeeze().cpu().numpy()


def adjust_affine_for_crop_pad(
    affine: np.ndarray,
    original_shape: Tuple[int, int, int],
    crop_pad_shape: Optional[Tuple[int, int, int]] = None,
    resampled_shape: Optional[Tuple[int, int, int]] = None,
    target_spacing: Optional[Tuple[float, float, float]] = None,
    original_spacing: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    """Adjust an affine matrix for resampling + center crop/pad operations.

    Args:
        affine: Original affine (4x4).
        original_shape: Original volume shape before any processing.
        crop_pad_shape: Shape after resampling and crop/pad.
        resampled_shape: Shape after resampling (before crop/pad).
        target_spacing: Target spacing used for resampling.
        original_spacing: Original voxel spacing.

    Returns:
        Adjusted affine matrix.
    """
    out_affine = affine.copy().astype(float)

    if target_spacing is not None and original_spacing is not None:
        for i in range(3):
            scale_i = target_spacing[i] / max(float(original_spacing[i]), 1e-8)
            out_affine[:3, i] *= scale_i

    shape_for_crop = resampled_shape if resampled_shape is not None else original_shape
    if crop_pad_shape is not None:
        th, tw, td = crop_pad_shape
        sh_off = int(max((shape_for_crop[0] - th) // 2, 0))
        sw_off = int(max((shape_for_crop[1] - tw) // 2, 0))
        sd_off = int(max((shape_for_crop[2] - td) // 2, 0))
        out_affine[:3, 3] += out_affine[:3, :3] @ np.array([sh_off, sw_off, sd_off])

    return out_affine


# --------------------------------------------------------------------------- #
# Data splitting / matching                                                   #
# --------------------------------------------------------------------------- #


def extract_subject_id(filename: str) -> str:
    """Extract subject ID from NIfTI filename.

    Format: {R,P}_{modality}_{field}_{ID}.nii.gz -> ID (e.g., '0001')
    """
    base = filename.replace(".nii.gz", "")
    parts = base.split("_")
    if len(parts) >= 4:
        return parts[-1]
    return parts[0]


def list_nifti_files(
    data_root: Path,
    split: str,
    modality: str,
    domain: str,
) -> List[Path]:
    """List all .nii.gz files for a given split/modality/domain."""
    split_dir = SPLIT_MAP.get(split, split)
    domain_dir = Path(data_root) / split_dir / modality / domain
    return sorted(domain_dir.glob("*.nii.gz"))
