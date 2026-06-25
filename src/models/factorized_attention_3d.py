"""Factorized 3D Self-Attention for latent space UNet bottleneck.

Replaces dense O(N^6) attention with sequential axial attention:
  - O(H^2) on X-axis → O(W^2) on Y-axis → O(D^2) on Z-axis

Total complexity per volume: O(H*W*D * (H + W + D)) vs O((H*W*D)^2)
For a 16x16x10 latent: ~4k ops vs ~65k ops per attention head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AxialAttention(nn.Module):
    """Single-axis self-attention."""

    def __init__(self, dim: int, num_heads: int = 8, axis: int = -1, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.axis = axis  # which spatial axis to attend over (2,3,4 for H,W,D)
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W, D)
        Returns: (B, C, H, W, D) with attention applied along self.axis
        """
        B, C, H, W, D = x.shape
        orig_shape = x.shape

        # Permute so target axis is last
        if self.axis == 2:      # H axis
            x = x.permute(0, 3, 4, 2, 1)  # (B, W, D, H, C)
        elif self.axis == 3:    # W axis
            x = x.permute(0, 2, 4, 3, 1)  # (B, H, D, W, C)
        elif self.axis == 4:    # D axis
            x = x.permute(0, 2, 3, 4, 1)  # (B, H, W, D, C)
        else:
            raise ValueError(f"axis must be 2,3,4 for H,W,D. Got {self.axis}")

        # Flatten to (B*flat_spatial, seq_len, C)
        B2, S1, S2, seq_len, C = x.shape
        x = x.reshape(B2 * S1 * S2, seq_len, C)

        # QKV projection
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)  # (B*S1*S2, seq_len, 3*C)
        qkv = qkv.reshape(B2 * S1 * S2, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*S1*S2, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*S1*S2, heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B*S1*S2, heads, seq_len, head_dim)
        out = out.transpose(1, 2).reshape(B2 * S1 * S2, seq_len, C)
        out = self.proj(out)
        out = out + x  # residual

        # Reshape back
        out = out.reshape(B2, S1, S2, seq_len, C)
        if self.axis == 2:
            out = out.permute(0, 4, 3, 1, 2)  # (B, C, H, W, D)
        elif self.axis == 3:
            out = out.permute(0, 4, 1, 3, 2)  # (B, C, H, W, D)
        elif self.axis == 4:
            out = out.permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)

        return out


class FactorizedAttention3D(nn.Module):
    """Sequential axial attention over H, W, D axes."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.attn_h = AxialAttention(dim, num_heads, axis=2, dropout=dropout)
        self.attn_w = AxialAttention(dim, num_heads, axis=3, dropout=dropout)
        self.attn_d = AxialAttention(dim, num_heads, axis=4, dropout=dropout)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W, D)
        out: (B, C, H, W, D)
        """
        x = self.attn_h(x)
        x = self.attn_w(x)
        x = self.attn_d(x)
        return x
