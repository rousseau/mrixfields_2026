#!/usr/bin/env python3
"""Unified metrics computation for MRIxFields.

Consolidated from:
  - src/evaluation/evaluate.py (nRMSE, SSIM, LPIPS, Dice, Volume)
  - src/vae3d/train_vqvae.py (ssim3d differentiable)
  - src/vae3d/benchmark_vae.py (SSIM per-slice numpy)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

# --------------------------------------------------------------------------- #
# LPIPS singleton                                                             #
# --------------------------------------------------------------------------- #

_lpips_fn = None


def get_lpips_fn(device: str = "cuda"):
    """Get cached LPIPS function (AlexNet)."""
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import lpips as lpips_module
        except ImportError:
            raise ImportError("lpips non installé. pip install lpips")
        if not torch.cuda.is_available() and device == "cuda":
            device = "cpu"
        _lpips_fn = lpips_module.LPIPS(net="alex").to(device)
        _lpips_fn.eval()
    return _lpips_fn


# --------------------------------------------------------------------------- #
# Voxel-level metrics                                                         #
# --------------------------------------------------------------------------- #


def compute_nrmse(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Normalized Root Mean Square Error.

    Args:
        pred, target: Arrays in [0, 1].
        mask: Optional brain mask.
    """
    pred, target = pred.astype(np.float64), target.astype(np.float64)
    if mask is not None:
        pred, target = pred[mask > 0], target[mask > 0]
    norm = np.linalg.norm(target)
    return float(np.linalg.norm(pred - target) / norm) if norm > 1e-10 else 0.0


def compute_ssim(
    pred: np.ndarray,
    target: np.ndarray,
    slice_axis: int = 2,
) -> float:
    """Structural Similarity Index (3D, computed slice-wise)."""
    pred, target = pred.astype(np.float64), target.astype(np.float64)
    data_range = float(target.max() - target.min())
    if data_range < 1e-10:
        return 1.0

    if pred.ndim == 2:
        return float(structural_similarity(pred, target, data_range=data_range))

    vals = []
    for i in range(pred.shape[slice_axis]):
        s = [slice(None)] * pred.ndim
        s[slice_axis] = i
        ps, ts = pred[tuple(s)], target[tuple(s)]
        if ts.max() - ts.min() < 1e-10:
            continue
        vals.append(structural_similarity(ps, ts, data_range=data_range))
    return float(np.mean(vals)) if vals else 1.0


def compute_ssim3d_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    sigma: float = 1.5,
) -> float:
    """3D SSIM using scipy Gaussian filter (numpy).

    This is the implementation from benchmark_vae.py, retained for
    comparison purposes.
    """
    from scipy import ndimage

    c1, c2 = 0.01, 0.03
    mu1 = ndimage.gaussian_filter(pred.astype(np.float64), sigma=sigma)
    mu2 = ndimage.gaussian_filter(target.astype(np.float64), sigma=sigma)
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = ndimage.gaussian_filter(pred.astype(np.float64) ** 2, sigma=sigma) - mu1_sq
    sigma2_sq = ndimage.gaussian_filter(target.astype(np.float64) ** 2, sigma=sigma) - mu2_sq
    sigma12 = ndimage.gaussian_filter(pred.astype(np.float64) * target.astype(np.float64), sigma=sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    )
    return float(np.mean(ssim_map))


def compute_ssim3d_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 7,
    data_range: float = 2.0,
) -> torch.Tensor:
    """Differentiable 3D SSIM (mean over the volume).

    Adapted from train_vqvae.py. Protected against division by zero.
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    if x.shape[2] == 1:
        # 2D fallback
        x2 = x[:, :, 0]
        y2 = y[:, :, 0]
        coords = torch.arange(window_size, dtype=x.dtype, device=x.device)
        coords = coords - (window_size - 1) / 2.0
        g = torch.exp(-(coords**2) / (2.0 * 1.5**2))
        g = g / g.sum()
        k = (g[:, None] * g[None, :]).view(1, 1, window_size, window_size)
        pad = window_size // 2
        mu_x = F.conv2d(x2, k, padding=pad)
        mu_y = F.conv2d(y2, k, padding=pad)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sigma_x2 = F.conv2d(x2 * x2, k, padding=pad) - mu_x2
        sigma_y2 = F.conv2d(y2 * y2, k, padding=pad) - mu_y2
        sigma_xy = F.conv2d(x2 * y2, k, padding=pad) - mu_xy
        denom = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-8
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / denom
        return ssim_map.mean()

    coords = torch.arange(window_size, dtype=x.dtype, device=x.device)
    coords = coords - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * 1.5**2))
    g = g / g.sum()
    k3 = g[:, None, None] * g[None, :, None] * g[None, None, :]
    k3 = k3.view(1, 1, window_size, window_size, window_size)
    pad = window_size // 2
    mu_x = F.conv3d(x, k3, padding=pad)
    mu_y = F.conv3d(y, k3, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv3d(x * x, k3, padding=pad) - mu_x2
    sigma_y2 = F.conv3d(y * y, k3, padding=pad) - mu_y2
    sigma_xy = F.conv3d(x * y, k3, padding=pad) - mu_xy
    denom = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-8
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / denom
    return ssim_map.mean()


def compute_lpips(
    pred: np.ndarray,
    target: np.ndarray,
    device: str = "cuda",
    slice_axis: int = 2,
) -> float:
    """Learned Perceptual Image Patch Similarity (AlexNet).

    Uses a cached model instance for efficiency.
    """
    fn = get_lpips_fn(device)
    pred_n = pred.astype(np.float64) * 2.0 - 1.0
    target_n = target.astype(np.float64) * 2.0 - 1.0

    def _2d(p: np.ndarray, t: np.ndarray) -> float:
        pt = (
            torch.from_numpy(p)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, 3, 1, 1)
            .to(next(fn.parameters()).device)
        )
        tt = (
            torch.from_numpy(t)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, 3, 1, 1)
            .to(next(fn.parameters()).device)
        )
        with torch.no_grad():
            return float(fn(pt, tt).item())

    if pred.ndim == 2:
        return _2d(pred_n, target_n)

    vals = []
    for i in range(pred.shape[slice_axis]):
        s = [slice(None)] * pred.ndim
        s[slice_axis] = i
        if np.abs(target_n[tuple(s)]).max() < 1e-10:
            continue
        vals.append(_2d(pred_n[tuple(s)], target_n[tuple(s)]))

    return float(np.mean(vals)) if vals else 0.0


# --------------------------------------------------------------------------- #
# Segmentation-based metrics                                                  #
# --------------------------------------------------------------------------- #

# 14 deep gray matter structures (7 bilateral pairs)
DGM_LABELS = {
    "L_Thalamus": 10,
    "R_Thalamus": 49,
    "L_Caudate": 11,
    "R_Caudate": 50,
    "L_Putamen": 12,
    "R_Putamen": 51,
    "L_Pallidum": 13,
    "R_Pallidum": 52,
    "L_Hippocampus": 17,
    "R_Hippocampus": 53,
    "L_Amygdala": 18,
    "R_Amygdala": 54,
    "L_Accumbens": 26,
    "R_Accumbens": 58,
}


def compute_dice(
    pred_seg: np.ndarray,
    target_seg: np.ndarray,
    labels: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """Compute Dice score for each DGM structure."""
    if labels is None:
        labels = DGM_LABELS

    scores = {}
    for name, lid in labels.items():
        pm = pred_seg == lid
        tm = target_seg == lid
        inter = np.sum(pm & tm)
        total = np.sum(pm) + np.sum(tm)
        scores[name] = 1.0 if total == 0 else float(2.0 * inter / total)
    return scores


def compute_volume_consistency(
    pred_seg: np.ndarray,
    target_seg: np.ndarray,
    voxel_volume: float = 1.0,
    labels: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """Compute normalized volume consistency for each DGM structure."""
    if labels is None:
        labels = DGM_LABELS

    results = {}
    for name, lid in labels.items():
        vp = np.sum(pred_seg == lid) * voxel_volume
        vt = np.sum(target_seg == lid) * voxel_volume
        if vt < 1e-10:
            results[name] = 1.0 if vp < 1e-10 else 0.0
        else:
            results[name] = float(1.0 - abs(vp - vt) / vt)
    return results
