#!/usr/bin/env python3
"""Small shared utilities not fitting in other common modules."""

import torch


class GradReverse(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)
