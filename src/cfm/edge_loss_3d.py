#!/usr/bin/env python3
"""3D edge-consistency loss — self-supervised anatomy-preservation regularizer.

Direct 3D port of the 2D Sobel edge-consistency term in the official challenge
baseline (~/Code/MRIxFields2026/Baseline/mrixfields/losses/structure.py,
StructureLoss._edge_map), applied here on VAE-decoded volumes rather than 2D
slices.

Two usages in train_mmfm_3d.py:
  - round-trip (lambda_edge): `decode(z_src_vec)` vs `decode(z_src_roundtrip)`
    (the cycle-consistency round trip) — SAME-FIELD/SAME-CONTRAST images.
  - forward (lambda_edge_fwd): `decode(z_src_vec)` vs `decode(z_tgt_hat)` (the
    actual forward prediction) — CROSS-FIELD/CROSS-CONTRAST. This relies on
    Sobel3D returning gradient *magnitude* only (no polarity/sign), on the
    assumption that edge *position* — not which side is brighter — is stable
    across field strength for the same subject anatomy. Unlike the round-trip
    usage, this is a real assumption, not a same-domain identity check: it
    could fight legitimate contrast-driven changes in edge visibility between
    e.g. 0.1T and 7T if that assumption doesn't hold everywhere.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Sobel3D(nn.Module):
    """3D Sobel gradient magnitude, via 3 orthogonal separable 3x3x3 kernels.

    Each kernel is a derivative ([-1,0,1]) along one axis combined with
    smoothing ([1,2,1]) along the other two — the standard 3D generalization
    of the 2D Sobel operator used by the baseline's StructureLoss.
    """

    def __init__(self):
        super().__init__()
        smooth = torch.tensor([1.0, 2.0, 1.0])
        deriv = torch.tensor([-1.0, 0.0, 1.0])

        kx = deriv.view(3, 1, 1) * smooth.view(1, 3, 1) * smooth.view(1, 1, 3)
        ky = smooth.view(3, 1, 1) * deriv.view(1, 3, 1) * smooth.view(1, 1, 3)
        kz = smooth.view(3, 1, 1) * smooth.view(1, 3, 1) * deriv.view(1, 1, 3)

        self.register_buffer("kx", kx.view(1, 1, 3, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3, 3))
        self.register_buffer("kz", kz.view(1, 1, 3, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W, D) -> gradient magnitude, same shape."""
        gx = F.conv3d(x, self.kx.to(x.dtype), padding=1)
        gy = F.conv3d(x, self.ky.to(x.dtype), padding=1)
        gz = F.conv3d(x, self.kz.to(x.dtype), padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + 1e-8)


def edge_consistency_loss(sobel3d: Sobel3D, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """L1 distance between Sobel edge maps of two same-contrast volumes."""
    return F.l1_loss(sobel3d(pred), sobel3d(ref))
