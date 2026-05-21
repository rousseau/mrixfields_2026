#!/usr/bin/env python3
"""
VQ-VAE 3D multimodal (paired + unpaired) pour MRIxFields.

Objectif:
- Conserver les briques principales de NeuroQuant (dual-stream, VQ EMA, FiLM, adversary)
- Ajouter un entraînement hybride pour exploiter:
  1) données unpaired: reconstruction intra-modale
  2) données paired: reconstruction cross-modale supervisée
- Se rapprocher au maximum de l'implémentation NeuroQuant originale (CVPR 2026)
  * Factorized convs + Multi-axis attention
  * Dead-code revival + k-means init
  * MLP FiLM avec identity init
  * Mid block entre encoder/decoder
  * SSIM3D + foreground weighting

Ce script est volontairement autonome pour tester l'approche avant intégration CFM.
"""

import argparse
import gc
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy.ndimage import zoom as scipy_zoom
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

MODALITIES = ["T1W", "T2W", "T2FLAIR"]
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
SPLIT_MAP = {
    "retro_train": "Training_retrospective",
    "pro_train": "Training_prospective",
    "pro_val": "Validating_prospective",
    "pro_test": "Testing_prospective",
}
FILE_RE = re.compile(r"^[A-Z]_([A-Z0-9]+)_([0-9.]+T)_(\d+)\.nii\.gz$")


def _resample_volume(
    vol: np.ndarray, original_spacing, target_spacing: Tuple[float, float, float]
) -> np.ndarray:
    orig = np.asarray(original_spacing[:3], dtype=float)
    tgt = np.asarray(target_spacing, dtype=float)
    factors = orig / tgt
    if np.allclose(factors, 1.0, atol=0.02):
        return vol.astype(np.float32)
    return scipy_zoom(vol, factors, order=1).astype(np.float32)


def _normalize(vol: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    lo = np.percentile(vol, lo_pct)
    hi = np.percentile(vol, hi_pct)
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return (vol * 2.0 - 1.0).astype(np.float32)


def _center_crop_or_pad(vol: np.ndarray, size: Tuple[int, int, int]) -> np.ndarray:
    th, tw, td = size
    h, w, d = vol.shape

    ph = max(0, th - h)
    pw = max(0, tw - w)
    pd = max(0, td - d)
    if ph > 0 or pw > 0 or pd > 0:
        vol = np.pad(
            vol,
            [(ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (pd // 2, pd - pd // 2)],
            mode="reflect",
        )
        h, w, d = vol.shape

    sh = max((h - th) // 2, 0)
    sw = max((w - tw) // 2, 0)
    sd = max((d - td) // 2, 0)
    return vol[sh : sh + th, sw : sw + tw, sd : sd + td]


@dataclass(frozen=True)
class SampleMeta:
    path: Path
    split: str
    modality: str
    field: str
    subject_id: str


class MRIxFieldsHybridDataset(Dataset):
    """Dataset hybride pour entraînement paired/unpaired.

    - Retourne toujours une source x_src
    - Peut retourner une cible paired x_tgt si disponible et tirée
    """

    def __init__(
        self,
        data_root: Path,
        splits: Sequence[str],
        modalities: Sequence[str],
        fields: Sequence[str],
        volume_size: Tuple[int, int, int],
        paired_prob: float = 0.5,
        percentile_lower: float = 0.5,
        percentile_upper: float = 99.5,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.volume_size = volume_size
        self.paired_prob = paired_prob
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.target_spacing = target_spacing

        self.samples: List[SampleMeta] = []
        self.by_key: Dict[Tuple[str, str, str], Dict[str, SampleMeta]] = {}

        for split in splits:
            split_dir = SPLIT_MAP.get(split, split)
            for modality in modalities:
                for field in fields:
                    d = self.data_root / split_dir / modality / field
                    for p in sorted(d.glob("*.nii.gz")):
                        m = FILE_RE.match(p.name)
                        if m is None:
                            continue
                        subj = m.group(3)
                        meta = SampleMeta(
                            path=p,
                            split=split,
                            modality=modality,
                            field=field,
                            subject_id=subj,
                        )
                        self.samples.append(meta)
                        key = (split, field, subj)
                        if key not in self.by_key:
                            self.by_key[key] = {}
                        self.by_key[key][modality] = meta

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if not self.samples:
            raise FileNotFoundError(
                "Aucun fichier NIfTI détecté pour les paramètres fournis."
            )

        self.mod_to_idx = {m: i for i, m in enumerate(modalities)}
        self.field_to_idx = {f: i for i, f in enumerate(fields)}

        n_pairable = 0
        for s in self.samples:
            if len(self.by_key[(s.split, s.field, s.subject_id)]) > 1:
                n_pairable += 1
        print(f"Dataset: {len(self.samples)} samples | pairables: {n_pairable}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_tensor(self, meta: SampleMeta) -> torch.Tensor:
        img = nib.load(str(meta.path))
        vol = img.get_fdata(dtype=np.float32)
        if self.target_spacing is not None:
            spacing = np.abs(np.diag(img.affine)[:3])
            vol = _resample_volume(vol, spacing, self.target_spacing)
        vol = _normalize(vol, self.percentile_lower, self.percentile_upper)
        vol = _center_crop_or_pad(vol, self.volume_size)
        return torch.from_numpy(vol).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src = self.samples[idx]
        x_src = self._load_tensor(src)
        src_mod_idx = self.mod_to_idx[src.modality]
        src_field_idx = self.field_to_idx[src.field]

        candidates = self.by_key[(src.split, src.field, src.subject_id)]
        other_mods = [m for m in candidates.keys() if m != src.modality]

        is_paired = bool(other_mods) and (random.random() < self.paired_prob)

        if is_paired:
            tgt_mod = random.choice(other_mods)
            tgt = candidates[tgt_mod]
            x_tgt = self._load_tensor(tgt)
            tgt_mod_idx = self.mod_to_idx[tgt.modality]
            tgt_field_idx = self.field_to_idx[tgt.field]
        else:
            x_tgt = torch.zeros_like(x_src)
            tgt_mod_idx = -1
            tgt_field_idx = -1

        return {
            "x_src": x_src,
            "x_tgt": x_tgt,
            "src_mod": torch.tensor(src_mod_idx, dtype=torch.long),
            "src_field": torch.tensor(src_field_idx, dtype=torch.long),
            "tgt_mod": torch.tensor(tgt_mod_idx, dtype=torch.long),
            "tgt_field": torch.tensor(tgt_field_idx, dtype=torch.long),
            "is_paired": torch.tensor(1 if is_paired else 0, dtype=torch.float32),
        }


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


# ──── NeuroQuant-style blocks ──────────────────────────────────────


class Normalize(nn.Module):
    """GroupNorm with num_groups=32 (or min(32, channels))."""

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=min(num_groups, channels),
            num_channels=channels,
            eps=1e-6,
            affine=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class FactoredConv3d(nn.Module):
    """Spatial (H,W) conv + Depth (D) conv, factorized.

    In 2D mode (depth_mode="2d"), the depth conv is skipped, so the model
    can process single slices (D=1) without artifacts.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        spatial_kernel: int = 3,
        depth_kernel: int = 3,
        stride: int = 1,
        depth_stride: int = 1,
    ):
        super().__init__()
        sp = spatial_kernel // 2
        dp = depth_kernel // 2

        self.spatial_conv = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=(1, spatial_kernel, spatial_kernel),
            stride=(1, stride, stride),
            padding=(0, sp, sp),
        )
        self.depth_conv = nn.Conv3d(
            out_ch,
            out_ch,
            kernel_size=(depth_kernel, 1, 1),
            stride=(depth_stride, 1, 1),
            padding=(dp, 1, 1) if depth_stride > 1 else (dp, 0, 0),
        )
        # Initialize depth conv as identity-like for smooth 2D->3D transition
        nn.init.dirac_(self.depth_conv.weight)
        if self.depth_conv.bias is not None:
            nn.init.zeros_(self.depth_conv.bias)

    def forward(self, x: torch.Tensor, depth_mode: str = "3d") -> torch.Tensor:
        x = self.spatial_conv(x)
        if depth_mode == "3d":
            x = self.depth_conv(x)
        return x


class FactoredResBlock(nn.Module):
    """ResBlock built on factored spatial + depth convolutions."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = Normalize(in_ch)
        self.conv1 = FactoredConv3d(in_ch, out_ch)
        self.norm2 = Normalize(out_ch)
        self.conv2 = FactoredConv3d(out_ch, out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, depth_mode: str = "3d") -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)), depth_mode)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))), depth_mode)
        return h + self.skip(x)


class AxisAttention(nn.Module):
    """Self-attention along a single axis."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = Normalize(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, axis: str = "d") -> torch.Tensor:
        B, C, D, H, W = x.shape
        h = self.norm(x)

        if axis == "d":
            h = rearrange(h, "b c d h w -> (b h w) d c")
        elif axis == "h":
            h = rearrange(h, "b c d h w -> (b d w) h c")
        elif axis == "w":
            h = rearrange(h, "b c d h w -> (b d h) w c")

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        head_dim = C // self.num_heads
        q = rearrange(q, "b s (nh hd) -> b nh s hd", nh=self.num_heads, hd=head_dim)
        k = rearrange(k, "b s (nh hd) -> b nh s hd", nh=self.num_heads, hd=head_dim)
        v = rearrange(v, "b s (nh hd) -> b nh s hd", nh=self.num_heads, hd=head_dim)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b nh s hd -> b s (nh hd)")
        out = self.proj(out)

        if axis == "d":
            out = rearrange(out, "(b h w) d c -> b c d h w", b=B, h=H, w=W)
        elif axis == "h":
            out = rearrange(out, "(b d w) h c -> b c d h w", b=B, d=D, w=W)
        elif axis == "w":
            out = rearrange(out, "(b d h) w c -> b c d h w", b=B, d=D, h=H)

        return x + out


class MultiAxisAttention(nn.Module):
    """Sequential attention along all three axes (D, H, W)."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.attn_d = AxisAttention(channels, num_heads)
        self.attn_h = AxisAttention(channels, num_heads)
        self.attn_w = AxisAttention(channels, num_heads)

    def forward(self, x: torch.Tensor, depth_mode: str = "3d") -> torch.Tensor:
        if depth_mode == "3d":
            x = self.attn_d(x, axis="d")
        x = self.attn_h(x, axis="h")
        x = self.attn_w(x, axis="w")
        return x


class Downsample3D(nn.Module):
    """Spatial 2x downsample + optional depth 2x downsample."""

    def __init__(self, channels: int, downsample_depth: bool = True):
        super().__init__()
        self.downsample_depth = downsample_depth
        if downsample_depth:
            self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)
        else:
            self.conv = nn.Conv3d(
                channels, channels, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
            )

    def forward(self, x: torch.Tensor, depth_mode: str = "3d") -> torch.Tensor:
        if depth_mode == "3d" and self.downsample_depth:
            return self.conv(x)
        elif not self.downsample_depth:
            return self.conv(x)
        else:
            return F.interpolate(
                x, scale_factor=(1, 0.5, 0.5), mode="trilinear", align_corners=False
            )


class Upsample3D(nn.Module):
    """Spatial 2x upsample + optional depth 2x upsample."""

    def __init__(self, channels: int, upsample_depth: bool = True):
        super().__init__()
        self.upsample_depth = upsample_depth
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, depth_mode: str = "3d") -> torch.Tensor:
        if depth_mode == "3d" and self.upsample_depth:
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        else:
            x = F.interpolate(
                x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False
            )
        return self.conv(x)


def weighted_recon_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mode: str = "l1",
    fg_weight: float = 5.0,
    bg_threshold: float = -0.9,
) -> torch.Tensor:
    """L1 / L2 reconstruction loss with brain-foreground weighting."""
    if mode == "l1":
        pixel_loss = (recon - target).abs()
    elif mode == "mse":
        pixel_loss = (recon - target) ** 2
    else:
        raise ValueError(f"Unknown recon loss mode: {mode}")

    with torch.no_grad():
        fg_mask = (target > bg_threshold).float()
        weight = 1.0 + (fg_weight - 1.0) * fg_mask
        weight = weight / weight.mean()

    return (pixel_loss * weight).mean()


def ssim3d(
    x: torch.Tensor, y: torch.Tensor, window_size: int = 7, data_range: float = 2.0
) -> torch.Tensor:
    """Differentiable 3D SSIM (mean over the volume).

    Protégé contre la division par zéro (variance=0).
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
        # + eps prevents division by zero
        denom = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-8
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / denom
        return ssim_map.mean()

    # 3D Gaussian kernel
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
    # + 1e-8 prevents division by zero (variance = 0 → constant region)
    denom = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-8
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / denom
    return ssim_map.mean()


class EMAVectorQuantizer(nn.Module):
    """EMA VectorQuantizer with k-means cold-start and adaptive dead-code revival (NeuroQuant)."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
        beta: float = 0.25,
        revive_dead: bool = True,
        revive_threshold: float = 0.1,
    ):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.revive_dead = revive_dead
        self.revive_threshold = revive_threshold

        # The codebook is a buffer (NOT a parameter): updated by EMA, not by grad.
        embed = torch.empty(num_embeddings, embedding_dim)
        embed.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        self.register_buffer("embedding", embed)

        # EMA accumulators
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", embed.clone())
        self.register_buffer("initialized", torch.tensor(False))

        # Reservoir for cold-start k-means init: lets us accumulate encoder
        # vectors across multiple forward calls when a single batch has fewer
        # tokens than K. We keep at most 2*K vectors; once full, init fires.
        self.register_buffer("init_reservoir", torch.empty(0, embedding_dim))

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, encodings: torch.Tensor):
        """Update codebook in place via EMA."""
        # 1) update cluster sizes
        cluster_size_new = encodings.sum(dim=0)  # (K,)
        self.ema_cluster_size.mul_(self.decay).add_(
            cluster_size_new, alpha=1.0 - self.decay
        )

        # 2) update sums of assigned vectors
        dw = encodings.t() @ flat  # (K, D)
        self.ema_w.mul_(self.decay).add_(dw, alpha=1.0 - self.decay)

        # 3) Laplace smoothing
        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + self.eps) / (n + self.K * self.eps) * n
        self.embedding.copy_(self.ema_w / smoothed.unsqueeze(1))

        if self.revive_dead:
            avg = self.ema_cluster_size.mean()
            threshold = max(
                self.revive_threshold * avg, torch.tensor(1e-3, device=avg.device)
            )
            dead = self.ema_cluster_size < threshold
            n_dead = int(dead.sum().item())
            if n_dead > 0 and flat.size(0) > 0:
                rand_idx = torch.randint(0, flat.size(0), (n_dead,), device=flat.device)
                self.embedding[dead] = flat[rand_idx]
                # reset their EMA stats so they get a fresh start
                self.ema_cluster_size[dead] = self.revive_threshold
                self.ema_w[dead] = flat[rand_idx]

    def forward(
        self, z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: (B,C,H,W,D)
        b, c, h, w, d = z_e.shape
        # Compute VQ assignment in fp32 for numerical stability under AMP.
        z_flat = z_e.float().permute(0, 2, 3, 4, 1).contiguous().view(-1, c)
        embedding_fp32 = self.embedding.float()

        # (B, C, D, H, W) -> (B, D, H, W, C) -> (N, C)
        z_perm = z_e.permute(0, 2, 3, 4, 1).contiguous()

        if self.training and not bool(self.initialized.item()):
            with torch.no_grad():
                self.init_reservoir = torch.cat(
                    [
                        self.init_reservoir.to(z_flat.device, dtype=z_flat.dtype),
                        z_flat.detach(),
                    ],
                    dim=0,
                )
                # Cap reservoir at 2*K to keep memory bounded if init is slow.
                if self.init_reservoir.size(0) > 2 * self.K:
                    perm = torch.randperm(
                        self.init_reservoir.size(0), device=z_flat.device
                    )[: 2 * self.K]
                    self.init_reservoir = self.init_reservoir[perm]

                if self.init_reservoir.size(0) >= self.K:
                    idx = torch.randperm(
                        self.init_reservoir.size(0), device=z_flat.device
                    )[: self.K]
                    self.embedding.copy_(self.init_reservoir[idx])
                    self.ema_w.copy_(self.embedding)
                    self.ema_cluster_size.fill_(1.0)
                    self.initialized.fill_(True)
                    # Free reservoir
                    self.init_reservoir = torch.empty(
                        0, self.D, device=z_flat.device, dtype=z_flat.dtype
                    )

        # Squared L2 distances: ||z||^2 + ||e||^2 - 2 z·e
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + embedding_fp32.pow(2).sum(dim=1)
            - 2.0 * z_flat @ embedding_fp32.t()
        )

        indices_flat = dist.argmin(dim=1)  # (N,)
        encodings = F.one_hot(indices_flat, num_classes=self.K).type(z_flat.dtype)
        z_q_flat = encodings @ embedding_fp32  # (N, D)
        z_q = z_q_flat.view(z_perm.shape)  # (B, D, H, W, C)

        # EMA codebook update (only in training)
        if self.training:
            self._ema_update(z_flat.detach(), encodings.detach())

        # Commitment loss only — codebook is updated by EMA, not by grad.
        commitment_loss = F.mse_loss(z_perm, z_q.detach())
        vq_loss = self.beta * commitment_loss

        # Straight-through estimator
        z_q = z_perm + (z_q - z_perm).detach()
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)

        # Perplexity (codebook usage)
        with torch.no_grad():
            avg_probs = encodings.mean(dim=0)
            perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        indices = indices_flat.view(z_perm.shape[:-1])  # (B, D, H, W)
        return z_q, vq_loss, indices, perplexity


class ConvBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, min(8, cout // 4)), num_channels=cout),
            nn.SiLU(inplace=True),
            nn.Conv3d(cout, cout, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, min(8, cout // 4)), num_channels=cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DualStreamEncoder(nn.Module):
    """Dual-stream encoder with factorized convs + multi-axis attention (NeuroQuant)."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        anat_channels: int = 64,
        mod_channels: int = 32,
        dropout: float = 0.0,
        attention_levels: Tuple[int, ...] = (2, 3),
        num_heads: int = 8,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.channels = [base_channels * m for m in channel_multipliers]
        if len(self.channels) < 2:
            raise ValueError("channel_multipliers must contain at least 2 values")

        self.conv_in = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for i, out_ch in enumerate(self.channels):
            block = nn.ModuleDict()
            res_blocks = nn.ModuleList()
            for _ in range(2):  # num_res_blocks=2
                res_blocks.append(FactoredResBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
            block["res"] = res_blocks
            if i in attention_levels:
                block["attn"] = MultiAxisAttention(out_ch, num_heads)
            block["down"] = Downsample3D(out_ch, downsample_depth=True)
            self.down_blocks.append(block)

        # Mid block (global context) - NeuroQuant style
        self.mid_res1 = FactoredResBlock(self.channels[-1], self.channels[-1], dropout)
        self.mid_attn = MultiAxisAttention(self.channels[-1], num_heads)
        self.mid_res2 = FactoredResBlock(self.channels[-1], self.channels[-1], dropout)

        self.norm_out = Normalize(self.channels[-1])

        # Dual-stream heads - both see the same shared feature map
        self.head_anat = nn.Conv3d(self.channels[-1], anat_channels, 1)
        self.head_mod = nn.Conv3d(self.channels[-1], mod_channels, 1)

    def _run_stage(
        self, block: nn.ModuleDict, h: torch.Tensor, depth_mode: str = "3d"
    ) -> torch.Tensor:
        for res in block["res"]:
            h = res(h, depth_mode)
        if "attn" in block:
            h = block["attn"](h, depth_mode)
        h = block["down"](h, depth_mode)
        return h

    def _run_mid(self, h: torch.Tensor, depth_mode: str = "3d") -> torch.Tensor:
        h = self.mid_res1(h, depth_mode)
        h = self.mid_attn(h, depth_mode)
        h = self.mid_res2(h, depth_mode)
        return h

    def forward(
        self, x: torch.Tensor, depth_mode: str = "3d"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)

        for block in self.down_blocks:
            if self.gradient_checkpointing and self.training:
                h = torch_checkpoint(
                    self._run_stage, block, h, depth_mode, use_reentrant=False
                )
            else:
                h = self._run_stage(block, h, depth_mode)

        if self.gradient_checkpointing and self.training:
            h = torch_checkpoint(self._run_mid, h, depth_mode, use_reentrant=False)
        else:
            h = self._run_mid(h, depth_mode)

        h = F.silu(self.norm_out(h))  # shared F^(t)
        z_anat = self.head_anat(h)  # (B, C_a, D', H', W')
        z_mod = self.head_mod(h)  # (B, C_m, D', H', W')
        return z_anat, z_mod


class FiLMGenerator(nn.Module):
    """MLP that predicts (gamma_l, beta_l) for each decoder layer (NeuroQuant)."""

    def __init__(self, in_dim: int, layer_channels: List[int], hidden_dim: int = 256):
        super().__init__()
        self.layer_channels = layer_channels
        total = sum(2 * c for c in layer_channels)  # gamma + beta per layer
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, total),
        )
        # Initialize last layer so initial FiLM ≈ identity (gamma=1, beta=0)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, u: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        raw = self.mlp(u)  # (B, total)
        params = []
        offset = 0
        for c in self.layer_channels:
            gamma = 1.0 + raw[:, offset : offset + c]  # identity init
            offset += c
            beta = raw[:, offset : offset + c]
            offset += c
            params.append((gamma, beta))
        return params


def film_apply(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Channel-wise affine: h' = gamma * h + beta. h is (B, C, D, H, W)."""
    g = gamma.view(gamma.size(0), gamma.size(1), 1, 1, 1)
    b = beta.view(beta.size(0), beta.size(1), 1, 1, 1)
    return g * x + b


class FiLMDecoder3D(nn.Module):
    """FiLM decoder 3D (NeuroQuant style)."""

    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        anat_channels: int = 64,
        dropout: float = 0.0,
        attention_levels: Tuple[int, ...] = (2, 3),
        num_heads: int = 8,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.channels = [base_channels * m for m in channel_multipliers]

        self.conv_in = nn.Conv3d(anat_channels, self.channels[-1], 3, padding=1)

        # Mid block
        self.mid_res1 = FactoredResBlock(self.channels[-1], self.channels[-1], dropout)
        self.mid_attn = MultiAxisAttention(self.channels[-1], num_heads)
        self.mid_res2 = FactoredResBlock(self.channels[-1], self.channels[-1], dropout)

        # Up blocks (reverse order)
        self.up_blocks = nn.ModuleList()
        in_ch = self.channels[-1]
        rev_channels = list(reversed(self.channels))
        rev_attn_levels = [len(self.channels) - 1 - l for l in attention_levels]
        for i, out_ch in enumerate(rev_channels):
            block = nn.ModuleDict()
            block["up"] = Upsample3D(in_ch, upsample_depth=True)
            res_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                res_blocks.append(FactoredResBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
            block["res"] = res_blocks
            if i in rev_attn_levels:
                block["attn"] = MultiAxisAttention(out_ch, num_heads)
            self.up_blocks.append(block)

        self.norm_out = Normalize(self.channels[0])
        self.conv_out = nn.Conv3d(self.channels[0], out_channels, 3, padding=1)

        # Channel sequence used by the FiLM generator: 1 entry per modulated stage.
        # Order: [mid (channels[-1]), up_block_0_out, up_block_1_out, ..., final_out]
        self.film_layer_channels = [self.channels[-1]] + list(rev_channels)

    def _run_mid(self, h: torch.Tensor, depth_mode: str = "3d") -> torch.Tensor:
        h = self.mid_res1(h, depth_mode)
        h = self.mid_attn(h, depth_mode)
        h = self.mid_res2(h, depth_mode)
        return h

    def _run_stage(
        self, block: nn.ModuleDict, h: torch.Tensor, depth_mode: str = "3d"
    ) -> torch.Tensor:
        h = block["up"](h, depth_mode)
        for res in block["res"]:
            h = res(h, depth_mode)
        if "attn" in block:
            h = block["attn"](h, depth_mode)
        return h

    def forward(
        self,
        z_q: torch.Tensor,
        film_params: List[Tuple[torch.Tensor, torch.Tensor]],
        depth_mode: str = "3d",
    ) -> torch.Tensor:
        assert len(film_params) == len(self.film_layer_channels), (
            f"FiLM expects {len(self.film_layer_channels)} (gamma,beta) pairs, "
            f"got {len(film_params)}"
        )

        h = self.conv_in(z_q)

        # Mid block + FiLM_0
        if self.gradient_checkpointing and self.training:
            h = torch_checkpoint(self._run_mid, h, depth_mode, use_reentrant=False)
        else:
            h = self._run_mid(h, depth_mode)
        gamma, beta = film_params[0]
        h = film_apply(h, gamma, beta)

        # Up blocks + FiLM_{l>=1}
        for i, block in enumerate(self.up_blocks):
            if self.gradient_checkpointing and self.training:
                h = torch_checkpoint(
                    self._run_stage, block, h, depth_mode, use_reentrant=False
                )
            else:
                h = self._run_stage(block, h, depth_mode)
            gamma, beta = film_params[i + 1]
            h = film_apply(h, gamma, beta)

        h = self.conv_out(F.silu(self.norm_out(h)))
        return torch.tanh(h)


class ModalityAdversary(nn.Module):
    """Gradient-reversed modality classifier on z_anat (NeuroQuant)."""

    def __init__(self, in_channels: int, num_modalities: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, 1),
            nn.GroupNorm(min(32, hidden), hidden),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_modalities),
        )

    def forward(self, z_anat: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        z_rev = grad_reverse(z_anat, alpha)
        return self.net(z_rev)


class NeuroQuantHybrid(nn.Module):
    """Dual-stream 3D VQ-VAE (NeuroQuant-inspired, with cross-modal swap)."""

    def __init__(
        self,
        n_modalities: int = 3,
        n_fields: int = 5,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        anat_channels: int = 64,
        mod_channels: int = 32,
        codebook_size: int = 4096,
        vq_decay: float = 0.99,
        vq_beta: float = 0.25,
        modality_embed_dim: int = 32,
        film_hidden: int = 256,
        dropout: float = 0.0,
        attention_levels: Tuple[int, ...] = (2, 3),
        num_heads: int = 8,
        adv_alpha: float = 1.0,
    ):
        super().__init__()
        self.adv_alpha = adv_alpha

        self.encoder = DualStreamEncoder(
            in_channels=1,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            anat_channels=anat_channels,
            mod_channels=mod_channels,
            dropout=dropout,
            attention_levels=attention_levels,
            num_heads=num_heads,
        )

        self.quantizer = EMAVectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=anat_channels,
            decay=vq_decay,
            eps=1e-5,
            beta=vq_beta,
            revive_dead=True,
            revive_threshold=0.1,
        )

        self.decoder = FiLMDecoder3D(
            out_channels=1,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=2,
            anat_channels=anat_channels,
            dropout=dropout,
            attention_levels=attention_levels,
            num_heads=num_heads,
        )

        # Per-modality learned embedding s_m (NeuroQuant)
        self.modality_embedding = nn.Embedding(max(n_modalities, 2), modality_embed_dim)
        self.film_generator = FiLMGenerator(
            in_dim=mod_channels + modality_embed_dim,
            layer_channels=self.decoder.film_layer_channels,
            hidden_dim=film_hidden,
        )

        self.adversary = ModalityAdversary(
            in_channels=anat_channels,
            num_modalities=max(n_modalities, 2),
        )

    def enable_gradient_checkpointing(self) -> None:
        self.encoder.gradient_checkpointing = True
        self.decoder.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.encoder.gradient_checkpointing = False
        self.decoder.gradient_checkpointing = False

    def compute_film(
        self, z_mod: torch.Tensor, modality: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Build u_m = concat(GAP(z_mod), s_m) and predict FiLM params."""
        gap = F.adaptive_avg_pool3d(z_mod, 1).flatten(1)  # (B, C_m)
        s_m = self.modality_embedding(modality)  # (B, C_s)
        u_m = torch.cat([gap, s_m], dim=1)
        return self.film_generator(u_m)

    def forward_src(
        self,
        x_src: torch.Tensor,
        src_mod: torch.Tensor,
        src_field: torch.Tensor,
        adv_alpha: Optional[float] = None,
        depth_mode: str = "3d",
    ) -> Dict[str, torch.Tensor]:
        z_anat, z_mod = self.encoder(x_src, depth_mode)
        z_anat_q, vq_loss, indices, perplexity = self.quantizer(z_anat)

        film_params = self.compute_film(z_mod, src_mod)
        recon = self.decoder(z_anat_q, film_params, depth_mode)

        mod_logits = None
        alpha = self.adv_alpha if adv_alpha is None else float(adv_alpha)
        mod_logits = self.adversary(z_anat, alpha=alpha)

        return {
            "z_anat": z_anat,
            "z_mod": z_mod,
            "z_q": z_anat_q,
            "indices": indices,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "film_params": film_params,
            "x_rec": recon,
            "mod_logits": mod_logits,
        }


def _to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _setup_distributed() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_dist = world_size > 1
    if use_dist and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return use_dist, world_size, rank, local_rank


def _sync_quantizer_buffers(model: NeuroQuantHybrid, world_size: int) -> None:
    """Average EMA codebook buffers across ranks to keep VQ state consistent."""
    q = model.quantizer
    for name in ["embedding", "ema_cluster_size", "ema_w"]:
        buf = getattr(q, name)
        if not torch.is_floating_point(buf):
            continue
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        buf.div_(float(world_size))

    # initialized is a scalar bool tensor; use max to propagate True to all ranks.
    init_i32 = q.initialized.to(dtype=torch.int32)
    dist.all_reduce(init_i32, op=dist.ReduceOp.MAX)
    q.initialized.copy_(init_i32.bool())


def train(args: argparse.Namespace) -> None:
    use_dist, world_size, rank, local_rank = _setup_distributed()

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if use_dist else "cuda")
    else:
        device = torch.device("cpu")

    if use_dist and device.type != "cuda":
        raise RuntimeError("DDP requires CUDA devices on Jean Zay.")
    if use_dist:
        torch.cuda.set_device(local_rank)

    use_amp = args.use_amp and device.type == "cuda"
    if args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    ds = MRIxFieldsHybridDataset(
        data_root=Path(args.data_root),
        splits=args.splits,
        modalities=args.modalities,
        fields=args.fields,
        volume_size=tuple(args.volume_size),
        paired_prob=args.paired_prob,
        percentile_lower=args.percentile_lower,
        percentile_upper=args.percentile_upper,
        target_spacing=tuple(args.target_spacing) if args.target_spacing else None,
        max_samples=args.max_samples,
    )

    sampler = (
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        if use_dist
        else None
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = NeuroQuantHybrid(
        n_modalities=len(args.modalities),
        n_fields=len(args.fields),
        base_channels=args.base_channels,
        channel_multipliers=tuple(args.channel_multipliers),
        anat_channels=args.anat_channels,
        mod_channels=args.mod_channels,
        codebook_size=args.codebook_size,
        vq_decay=args.vq_decay,
        vq_beta=args.vq_beta,
        modality_embed_dim=32,
        film_hidden=256,
        dropout=0.0,
        attention_levels=(2, 3),
        num_heads=8,
        adv_alpha=1.0,
    ).to(device)
    raw_model = model

    if use_dist:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    if args.gradient_checkpointing:
        raw_model.enable_gradient_checkpointing()

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    is_main = rank == 0
    if is_main:
        print(
            f"Device: {device} | AMP: {use_amp} | DDP: {use_dist} "
            f"(world_size={world_size})"
        )
        print(f"AMP dtype: {str(amp_dtype).replace('torch.', '')}")
        print(f"Grad checkpointing: {args.gradient_checkpointing}")
        print(f"Batch size/GPU: {args.batch_size} | Global: {args.batch_size * world_size}")
        print(f"Steps: {args.steps}")
        print(f"Output: {out_dir}")

    step = 0
    t0 = time.time()
    best_recon_loss = float("inf")
    consecutive_nonfinite = 0

    # ── Reprise depuis un checkpoint (--resume) ───────────────────────────────────
    if args.resume and Path(args.resume).exists():
        if is_main:
            print(f"Chargement du checkpoint : {args.resume}")
        ckpt_r = torch.load(args.resume, map_location=device, weights_only=False)

        # Poids du modèle
        missing, unexpected = raw_model.load_state_dict(ckpt_r["model"], strict=False)
        if is_main:
            if missing:
                print(f"  ⚠  {len(missing)} clé(s) manquante(s) dans le checkpoint")
            if unexpected:
                print(f"  ⚠  {len(unexpected)} clé(s) inattendue(s) dans le checkpoint")

        # Vérification que les poids ne sont pas NaN/Inf (checkpoint corrompu)
        n_bad = sum(1 for p in model.parameters() if not torch.isfinite(p.data).all())
        if n_bad > 0:
            if is_main:
                print(
                    f"  ✗ Checkpoint corrompu : {n_bad} paramètre(s) non-finis (NaN/Inf) détectés."
                )
                print("    Le checkpoint est inutilisable — démarrage à zéro.")
            raw_model.apply(
                lambda m: (
                    m.reset_parameters() if hasattr(m, "reset_parameters") else None
                )
            )
            # Ne pas restaurer step ni best_recon_loss
        else:
            # État de l'optimiseur
            if "optimizer" in ckpt_r:
                try:
                    opt.load_state_dict(ckpt_r["optimizer"])
                except Exception as e:
                    if is_main:
                        print(f"  ⚠  Optimiseur non restauré (incompatibilité) : {e}")

            # État de l'AMP scaler
            if "scaler" in ckpt_r:
                try:
                    scaler.load_state_dict(ckpt_r["scaler"])
                except Exception as e:
                    if is_main:
                        print(f"  ⚠  Scaler non restauré : {e}")

            # État de l'entraînement
            ckpt_step = int(ckpt_r.get("step", 0))
            best_recon_loss = float(ckpt_r.get("best_recon_loss", float("inf")))

            # Sécurité : si le checkpoint a déjà atteint le nombre de steps cible
            # (ex : run NaN qui a loopé jusqu'à steps=20000 sans rien apprendre),
            # on repart de zéro avec les poids tels quels.
            if ckpt_step >= args.steps:
                if is_main:
                    print(
                        f"  ⚠  Le checkpoint indique step={ckpt_step} >= steps={args.steps}."
                    )
                    print(
                        "    Le training était peut-être corrompu. Démarrage à zéro (poids conservés)."
                    )
                step = 0
                best_recon_loss = float("inf")
            else:
                step = ckpt_step
                if is_main:
                    print(
                        f"  ✓ Reprise à partir du step {step}  (best_recon={best_recon_loss:.4f})"
                    )

    elif args.resume:
        if is_main:
            print(f"  ⚠  Checkpoint introuvable ({args.resume}) — démarrage à zéro.")

    if use_dist:
        dist.barrier()

    def _linear_ramp(
        step_id: int, start: int, ramp_steps: int, max_val: float
    ) -> float:
        if step_id < start:
            return 0.0
        if ramp_steps <= 0:
            return float(max_val)
        p = min(1.0, (step_id - start) / max(1, ramp_steps))
        return float(max_val * p)

    def _grl_sigmoid(
        step_id: int, start: int, total_steps: int, max_alpha: float
    ) -> float:
        if step_id < start:
            return 0.0
        p = min(1.0, (step_id - start) / max(1, total_steps - start))
        return float(max_alpha * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))

    epoch = 0
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
            epoch += 1

        for batch in loader:
            if step >= args.steps:
                break
            step += 1

            batch = _to_device(batch, device)
            x_src = batch["x_src"]
            x_tgt = batch["x_tgt"]
            src_mod = batch["src_mod"]
            src_field = batch["src_field"]
            tgt_mod = batch["tgt_mod"]
            tgt_field = batch["tgt_field"]
            is_paired = batch["is_paired"]

            model.train()
            opt.zero_grad(set_to_none=True)

            cross_w = _linear_ramp(
                step, args.cross_start_step, args.cross_ramp_steps, args.lambda_cross
            )
            adv_w = _linear_ramp(
                step, args.adv_start_step, args.adv_ramp_steps, args.lambda_adv
            )
            grl_lambda = _grl_sigmoid(
                step, args.adv_start_step, args.steps, args.adv_grl_lambda
            )

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out_src = raw_model.forward_src(
                    x_src,
                    src_mod,
                    src_field,
                    adv_alpha=grl_lambda,
                )
                x_rec = out_src["x_rec"]

                # SSIM3D + foreground weighting (NeuroQuant)
                recon_loss = weighted_recon_loss(
                    x_rec, x_src, "l1", fg_weight=5.0, bg_threshold=-0.9
                )
                ssim_term = 1.0 - ssim3d(
                    x_rec.float(), x_src.float(), window_size=7, data_range=2.0
                )
                recon_loss = recon_loss + 0.5 * ssim_term  # ssim_weight=0.5

                vq_loss = out_src["vq_loss"]

                # Adversary modalité sur code anatomique (invariance)
                logits_adv = out_src["mod_logits"]
                adv_loss = F.cross_entropy(logits_adv, src_mod)

                # Cross loss (paired uniquement) - NeuroQuant style
                paired_mask = is_paired > 0.5
                if paired_mask.any():
                    # Get z_mod for target modality
                    with torch.no_grad():
                        _, z_mod_all_tgt = raw_model.encoder(x_tgt[paired_mask])

                    # Compute FiLM params for target modality
                    film_params_tgt = raw_model.compute_film(
                        z_mod_all_tgt, tgt_mod[paired_mask]
                    )

                    # Reconstruct with z_q_src + film_params_tgt
                    x_cross = raw_model.decoder(
                        out_src["z_q"][paired_mask], film_params_tgt
                    )
                    cross_loss = F.l1_loss(x_cross, x_tgt[paired_mask])
                else:
                    cross_loss = torch.zeros([], device=device)

                total = (
                    args.lambda_recon * recon_loss
                    + args.lambda_vq * vq_loss
                    + adv_w * adv_loss
                    + cross_w * cross_loss
                )

            finite_tensor = torch.isfinite(total).to(dtype=torch.int32)
            if use_dist:
                dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
            all_finite = bool(finite_tensor.item())

            if not all_finite:
                consecutive_nonfinite += 1
                if is_main:
                    print(
                        f"[WARN] step={step} non-finite loss detected "
                        "on at least one rank. Skipping optimizer step."
                    )
                if consecutive_nonfinite >= args.max_consecutive_nonfinite:
                    raise RuntimeError(
                        "Too many consecutive non-finite steps "
                        f"({consecutive_nonfinite} >= {args.max_consecutive_nonfinite}). "
                        "Stopping early to avoid wasting compute; resume from last healthy checkpoint."
                    )
                opt.zero_grad(set_to_none=True)
                continue

            consecutive_nonfinite = 0

            # GradScaler is required for fp16, but not for bf16.
            if use_scaler:
                prev_scale = scaler.get_scale()
                scaler.scale(total).backward()

                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)

                scaler.step(opt)  # skipped automatically by GradScaler on overflow
                scaler.update()

                new_scale = scaler.get_scale()
                if new_scale < prev_scale:
                    print(
                        f"  [AMP-fp16] step={step} overflow détecté "
                        f"(scale {prev_scale:.1f} -> {new_scale:.1f}), optimizer step ignoré automatiquement."
                    )
            else:
                total.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
                opt.step()

            if use_dist:
                _sync_quantizer_buffers(raw_model, world_size)

            # Periodic memory cleanup to prevent CUDA OOM during long training runs
            if step % 50 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            metric_tensor = torch.stack(
                [
                    total.detach(),
                    recon_loss.detach(),
                    vq_loss.detach(),
                    adv_loss.detach(),
                    cross_loss.detach(),
                    is_paired.mean().detach(),
                    out_src["perplexity"].detach(),
                ]
            ).float()
            if use_dist:
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                metric_tensor /= float(world_size)

            total_m, recon_m, vq_m, adv_m, cross_m, paired_ratio_m, ppl_m = (
                metric_tensor.tolist()
            )

            if (step % args.print_every == 0 or step == 1) and is_main:
                elapsed = time.time() - t0
                print(
                    f"[{step:5d}/{args.steps}] "
                    f"loss={total_m:.4f} "
                    f"recon={recon_m:.4f} "
                    f"vq={vq_m:.4f} "
                    f"adv={adv_m:.4f} "
                    f"cross={cross_m:.4f} "
                    f"w_adv={adv_w:.4f} "
                    f"w_cross={cross_w:.4f} "
                    f"grl={grl_lambda:.3f} "
                    f"paired={paired_ratio_m:.2f} "
                    f"ppl={ppl_m:.1f} "
                    f"t={elapsed / 60:.1f}min"
                )

            if (step % args.save_every == 0 or step == args.steps) and is_main:
                ckpt = {
                    "step": step,
                    "model": raw_model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_recon_loss": best_recon_loss,
                    "args": vars(args),
                }
                # Checkpoint périodique nommé (pour reprise auto-requeue)
                step_ckpt = weights_dir / f"vqvae_step_{step:06d}.pth"
                torch.save(ckpt, step_ckpt)
                print(f"  -> {step_ckpt.name}")

                # model_best : meilleure reconstruction
                if recon_m < best_recon_loss:
                    best_recon_loss = recon_m
                    ckpt["best_recon_loss"] = best_recon_loss
                    torch.save(ckpt, weights_dir / "model_best.pth")
                    print(f"  -> model_best.pth (recon={best_recon_loss:.4f})")

                # Nettoyage : ne garder que les 3 derniers checkpoints périodiques
                step_ckpts = sorted(weights_dir.glob("vqvae_step_*.pth"))
                for old in step_ckpts[:-3]:
                    old.unlink(missing_ok=True)

                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if is_main:
        final_ckpt = {
            "step": step,
            "model": raw_model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "best_recon_loss": best_recon_loss,
            "args": vars(args),
        }
        final_path = weights_dir / "model_final.pth"
        torch.save(final_ckpt, final_path)
        print(f"Training terminé. Modèle final: {final_path}")

    if use_dist and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train VQ-VAE 3D hybride paired/unpaired (MRIxFields)"
    )
    p.add_argument(
        "--data-root", type=str, default="/home/rousseau/Data/MRIxFields_20260414"
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/vqvae3d",
        help="Output directory for weights and checkpoints (relative to mrixfields_2026/)",
    )

    p.add_argument("--splits", nargs="+", default=["retro_train"])
    p.add_argument("--modalities", nargs="+", default=MODALITIES)
    p.add_argument("--fields", nargs="+", default=FIELDS)

    p.add_argument("--volume-size", nargs=3, type=int, default=[112, 128, 80])
    p.add_argument("--target-spacing", nargs=3, type=float, default=None)
    p.add_argument("--percentile-lower", type=float, default=0.5)
    p.add_argument("--percentile-upper", type=float, default=99.5)
    p.add_argument("--paired-prob", type=float, default=0.5)
    p.add_argument("--max-samples", type=int, default=None)

    p.add_argument(
        "--base-channels", type=int, default=64, help="NeuroQuant base_channels=64"
    )
    p.add_argument("--channel-multipliers", nargs="+", type=int, default=[1, 2, 4, 4])
    p.add_argument("--anat-channels", type=int, default=64)
    p.add_argument("--mod-channels", type=int, default=32)
    p.add_argument(
        "--codebook-size", type=int, default=4096, help="NeuroQuant codebook_size=4096"
    )
    p.add_argument("--vq-decay", type=float, default=0.99, help="NeuroQuant decay=0.99")
    p.add_argument("--vq-beta", type=float, default=0.25)

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--lambda-recon", type=float, default=1.0)
    p.add_argument("--lambda-vq", type=float, default=1.0)
    p.add_argument("--lambda-adv", type=float, default=1e-3)
    p.add_argument("--lambda-cross", type=float, default=0.5)
    p.add_argument("--adv-grl-lambda", type=float, default=0.5)
    p.add_argument("--cross-start-step", type=int, default=500)
    p.add_argument("--cross-ramp-steps", type=int, default=1000)
    p.add_argument("--adv-start-step", type=int, default=1500)
    p.add_argument("--adv-ramp-steps", type=int, default=1000)
    p.add_argument(
        "--max-consecutive-nonfinite",
        type=int,
        default=50,
        help="Stop training early after this many consecutive non-finite steps.",
    )

    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--use-amp", action="store_true")
    p.add_argument(
        "--amp-dtype",
        type=str,
        choices=["fp16", "bf16"],
        default="bf16",
        help="AMP dtype. Sur H100, bf16 est recommandé pour la stabilité.",
    )
    # gradient_checkpointing désactivé par défaut : incompatible avec AMP → NaN
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Activé seulement si AMP=off. Par défaut: désactivé (cause de NaN avec AMP).",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Chemin vers un checkpoint .pth pour reprendre l'entraînement.",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
