#!/usr/bin/env python3
"""Distributed training utilities and EMA.

Consolidated from:
  - src/cfm/train_cfm_3d.py
  - src/vae3d/train_vae_3d.py
  - src/cfm/train_mmfm_3d.py
  - src/vae3d/train_vqvae.py
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# --------------------------------------------------------------------------- #
# Process / rank helpers                                                      #
# --------------------------------------------------------------------------- #


def is_main_process() -> bool:
    """Return True if this is the main (rank 0) process or not in DDP."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def setup_distributed(backend: str = "nccl") -> Tuple[bool, int, int, int]:
    """Initialize distributed process group if WORLD_SIZE > 1.

    Returns:
        (is_distributed, world_size, rank, local_rank)
    """
    world_size = get_world_size()
    rank = get_rank()
    local_rank = get_local_rank()
    is_distributed = world_size > 1

    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    if is_distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return is_distributed, world_size, rank, local_rank


def cleanup_distributed():
    """Destroy process group if initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# EMA                                                                         #
# --------------------------------------------------------------------------- #


class EMAModel:
    """Exponential Moving Average of model weights.

    Usage:
        ema = EMAModel(model, decay=0.9999)
        # During training:
        ema.update(model)  # typically after optimizer.step()
        # For inference:
        ema_model = ema.shadow  # or ema.apply_shadow(model)
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        src = model.module if hasattr(model, "module") else model
        for s_param, param in zip(self.shadow.parameters(), src.parameters()):
            s_param.copy_(s_param * self.decay + param.data * (1.0 - self.decay))

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state: dict):
        self.shadow.load_state_dict(state)

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module):
        """Copy EMA weights into the given model (for inference)."""
        src = model.module if hasattr(model, "module") else model
        for s_param, param in zip(self.shadow.parameters(), src.parameters()):
            param.copy_(s_param)

    @torch.no_grad()
    def restore_original(self, model: torch.nn.Module, original_state: dict):
        """Restore original weights from a saved state dict."""
        src = model.module if hasattr(model, "module") else model
        src.load_state_dict(original_state)


# --------------------------------------------------------------------------- #
# Checkpoint helpers                                                          #
# --------------------------------------------------------------------------- #


def save_checkpoint(
    path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    step_or_epoch: int,
    **extras,
) -> None:
    """Save a standard checkpoint dict.

    Args:
        path: Save path.
        model: Model (or DDP-wrapped model).
        optimizer: Optimizer.
        scaler: AMP GradScaler or None.
        step_or_epoch: Current training step or epoch.
        **extras: Any extra fields (cfg_path, ema, best_loss, etc.)
    """
    raw_model = model.module if hasattr(model, "module") else model
    ckpt = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step_or_epoch,
    }
    if scaler is not None and hasattr(scaler, "state_dict"):
        ckpt["scaler"] = scaler.state_dict()
    ckpt.update(extras)
    torch.save(ckpt, path)


def load_checkpoint(
    path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scaler=None,
    strict: bool = True,
    device=None,
):
    """Load checkpoint into model (and optionally optimizer, scaler).

    Returns:
        Loaded checkpoint dict with metadata.
    """
    state = torch.load(path, map_location=device, weights_only=False)

    raw_model = model.module if hasattr(model, "module") else model

    model_state = state.get("model", state)
    if isinstance(model_state, dict):
        raw_model.load_state_dict(model_state, strict=strict)

    if optimizer is not None and "optimizer" in state:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except Exception:
            pass

    if scaler is not None and "scaler" in state:
        try:
            scaler.load_state_dict(state["scaler"])
        except Exception:
            pass

    return state
