"""Hybrid UNet 3D + Transformer bottleneck for MMFM v2.

Wraps MONAI DiffusionModelUNet and injects factorized axial attention
at the bottleneck (middle_block), providing global receptive field
without the O(N^6) cost of dense 3D self-attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

try:
    from monai.networks.nets import DiffusionModelUNet
except ImportError:
    try:
        from monai.generative.networks.nets import DiffusionModelUNet
    except ImportError:
        from generative.networks.nets import DiffusionModelUNet

from models.factorized_attention_3d import FactorizedAttention3D


class _MiddleBlockWithAttention(nn.Module):
    """Wraps the original middle_block and appends factorized attention.

    MONAI's middle_block is a TimestepEmbedSequential; its forward takes
    keyword arguments (hidden_states, temb, context). We delegate to the
    original block and then apply attention on the returned tensor.
    """

    def __init__(self, original_middle: nn.Module, attn: FactorizedAttention3D):
        super().__init__()
        self.original = original_middle
        self.attn = attn

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, context=None) -> torch.Tensor:
        out = self.original(hidden_states, temb, context)
        out = self.attn(out)
        return out


class HybridUNetTransformer(nn.Module):
    """UNet with optional factorized attention at bottleneck.

    Injects axial self-attention (H→W→D) after the UNet's middle_block,
    giving the model a global receptive field at the deepest level.
    """

    def __init__(
        self,
        unet: DiffusionModelUNet,
        bottleneck_channels: int,
        use_factorized_attention: bool = True,
        num_attn_heads: int = 8,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.unet = unet
        self.use_factorized_attention = use_factorized_attention

        if use_factorized_attention:
            self.factorized_attn = FactorizedAttention3D(
                dim=bottleneck_channels,
                num_heads=num_attn_heads,
                dropout=attn_dropout,
            )
            # Inject attention after middle_block
            if hasattr(unet, "middle_block"):
                original_middle = unet.middle_block
                unet.middle_block = _MiddleBlockWithAttention(
                    original_middle, self.factorized_attn
                )
            else:
                raise AttributeError(
                    "DiffusionModelUNet does not have a 'middle_block' attribute. "
                    "Cannot inject bottleneck attention."
                )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.unet(x, timesteps, context, class_labels)

    def state_dict(self, *args, **kwargs):
        return self.unet.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.unet.load_state_dict(state_dict, strict=strict)
