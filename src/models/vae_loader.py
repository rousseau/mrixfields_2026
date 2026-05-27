#!/usr/bin/env python3
"""Unified VAE loading utilities.

Single entry point `load_vae(cfg, device)` used by all training/inference/eval scripts.

Supported vae_types:
  - "aekl"              : MONAI AutoencoderKL
  - "maisi"             : MONAI bundle MAISI / NV-Generate-CTMR autoencoder
  - "medvae"            : Stanford MIMI MedVAE
  - "medvae_disentangle": MedVAE disentanglement v1
  - "vqvae"             : NeuroQuantHybrid VQ-VAE (deprecated, kept for compat)
  - "pythae_vae"        : Pythae VAE 3D (conv encoder/decoder + reparameterization)
  - "pythae_vqvae"      : Pythae VQ-VAE 3D (5D quantizer, EMA codebook)
  - "pythae_rhvae"      : Pythae RHVAE 3D (future — Phase C)
"""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml

try:
    from monai.networks.nets import AutoencoderKL
except ImportError:
    try:
        from monai.generative.networks.nets import AutoencoderKL
    except ImportError:
        from generative.networks.nets import AutoencoderKL

# Imports locaux
from models.vae_base import MRIxFieldsVAE
from models.vae_wrappers import (
    AEKLWrapper,
    MedVAEWrapper,
    MedVAEDisentangleWrapper,
    VQVAEWrapper,
    _infer_medvae_latent_channels,
)

# Rétrocompatibilité
VAEWrapper = MRIxFieldsVAE

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def load_vae(cfg: dict, device: torch.device) -> MRIxFieldsVAE:
    """Load and wrap a VAE based on cfg['vae'].

    Args:
        cfg: Config dict with 'vae' sub-dict.
        device: Target device.

    Returns:
        Wrapped VAE (frozen, on device).
    """
    vae_cfg = cfg.get("vae", {})
    vae_type = vae_cfg.get("vae_type", "aekl").lower()
    vae_source = vae_cfg.get("source", "local")

    print(f"  [load_vae] type={vae_type} source={vae_source}")

    if vae_type == "medvae":
        wrapper = _load_medvae(vae_cfg, device, vae_source)
    elif vae_type == "vqvae":
        wrapper = _load_vqvae(vae_cfg, device)
    elif vae_type == "medvae_disentangle":
        wrapper = _load_medvae_disentangle(vae_cfg, device)
    elif vae_type == "maisi":
        wrapper = _load_maisi(vae_cfg, device, vae_source)
    elif vae_type == "pythae_vae":
        wrapper = _load_pythae_vae(vae_cfg, device)
    elif vae_type == "pythae_vqvae":
        wrapper = _load_pythae_vqvae(vae_cfg, device)
    elif vae_type == "pythae_rhvae":
        raise NotImplementedError(
            "pythae_rhvae support not yet implemented. "
            "Please run Phase C of the VAE implementation plan."
        )
    else:
        wrapper = _load_aekl(vae_cfg, device, vae_source)

    wrapper = wrapper.to(device)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    print(f"  [load_vae] latent_format={wrapper.latent_format} "
          f"latent_channels={wrapper.latent_channels} "
          f"latent_shape={wrapper.latent_shape}")
    return wrapper


# --------------------------------------------------------------------------- #
# AEKL                                                                        #
# --------------------------------------------------------------------------- #


def _load_aekl(vae_cfg: dict, device: torch.device, source: str) -> AEKLWrapper:
    """Load AutoencoderKL MONAI (local | maisi | random)."""
    if source == "maisi":
        model = _build_maisi_autoencoder(vae_cfg.get("maisi_cache_dir"))
    else:
        vae_config_path = vae_cfg.get("vae_config", "configs/vae3d_multimodal.yaml")
        if not os.path.isabs(vae_config_path):
            project_root = Path(__file__).resolve().parents[2]
            vae_config_path = str(project_root / vae_config_path)

        with open(vae_config_path) as f:
            vqvae_cfg = yaml.safe_load(f)
        model = _build_aekl_from_config(vqvae_cfg)

        if source != "random":
            ckpt_path = vae_cfg.get("checkpoint", "")
            if ckpt_path and Path(ckpt_path).exists():
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state = state.get("model", state)
                # Handle key remapping from older MONAI versions
                state = _remap_aekl_keys(state)
                model.load_state_dict(state, strict=False)
                print(f"  AEKL loaded from {ckpt_path}")
            else:
                print(f"  [WARN] AEKL checkpoint not found: {ckpt_path}")

    return AEKLWrapper(model)


def _build_aekl_from_config(cfg: dict) -> AutoencoderKL:
    m = cfg["model"]
    sig = inspect.signature(AutoencoderKL.__init__).parameters
    kwargs = {
        "spatial_dims": m["spatial_dims"],
        "in_channels": m["in_channels"],
        "out_channels": m["out_channels"],
        "latent_channels": m["latent_channels"],
        "num_res_blocks": m["num_res_blocks"],
        "norm_num_groups": m["norm_num_groups"],
        "attention_levels": tuple(m["attention_levels"]),
        "with_encoder_nonlocal_attn": m["with_encoder_nonlocal_attn"],
        "with_decoder_nonlocal_attn": m["with_decoder_nonlocal_attn"],
    }
    channels = tuple(m["channels"])
    if "channels" in sig:
        kwargs["channels"] = channels
    elif "num_channels" in sig:
        kwargs["num_channels"] = channels
    filtered = {k: v for k, v in kwargs.items() if k in sig}
    return AutoencoderKL(**filtered)


def _remap_aekl_keys(state: dict) -> dict:
    """Remap decoder keys from .conv.conv. to .postconv.conv. for MONAI compat."""
    fixed = {}
    for k, v in state.items():
        if k.startswith("decoder."):
            k = k.replace(".conv.conv.", ".postconv.conv.")
        fixed[k] = v
    return fixed


# --------------------------------------------------------------------------- #
# MAISI / NV-Generate-CTMR                                                    #
# --------------------------------------------------------------------------- #


def _load_maisi(vae_cfg: dict, device: torch.device, source: str) -> AEKLWrapper:
    """Load MAISI autoencoder (MONAI bundle or NVIDIA NV-Generate-CTMR).

    Internally uses _build_maisi_autoencoder which downloads the MONAI bundle.
    In the future, this may load NVIDIA's v1/v2 checkpoints directly.
    """
    model = _build_maisi_autoencoder(vae_cfg.get("maisi_cache_dir"))
    return AEKLWrapper(model)


def _build_maisi_autoencoder(cache_dir: Optional[str] = None) -> AutoencoderKL:
    """Download and load MAISI autoencoder from MONAI Model Zoo."""
    try:
        from monai.bundle import download
    except ImportError:
        raise ImportError("MONAI bundle API required. pip install monai>=1.3")

    bundle_dir = Path(cache_dir or os.path.expanduser("~/.cache/monai/bundles"))
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = "maisi_ct_generative"

    download(name=bundle_name, source="monai", bundle_dir=str(bundle_dir))

    models_dir = bundle_dir / bundle_name / "models"
    candidates = sorted(models_dir.glob("*autoencoder*"))
    if not candidates:
        raise FileNotFoundError(f"MAISI autoencoder not found in {models_dir}")
    ckpt_path = candidates[0]

    cfg = {
        "model": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "latent_channels": 4,
            "channels": [64, 128, 256, 512],
            "num_res_blocks": 2,
            "norm_num_groups": 32,
            "attention_levels": [False, False, False, False],
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
        }
    }
    vae = _build_aekl_from_config(cfg)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = state.get("state_dict", state.get("model", state))
    vae.load_state_dict(state, strict=True)
    print(f"  MAISI VAE loaded from {ckpt_path}")
    return vae


# --------------------------------------------------------------------------- #
# MedVAE                                                                      #
# --------------------------------------------------------------------------- #


def _load_medvae(vae_cfg: dict, device: torch.device, source: str) -> MedVAEWrapper:
    """Load MedVAE (frozen from HuggingFace or fine-tuned from local checkpoint)."""
    model_name = vae_cfg.get("model_name", "medvae_4_1_3d")
    try:
        from medvae import MVAE
    except ImportError:
        raise ImportError("medvae not installed. pip install medvae")

    model = MVAE(model_name=model_name, modality="mri")

    if source == "local":
        ckpt_path = vae_cfg.get("checkpoint", "")
        if ckpt_path and Path(ckpt_path).exists():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = state.get("model", state)
            model.load_state_dict(state, strict=False)
            print(f"  MedVAE fine-tuned loaded from {ckpt_path}")
        else:
            print(f"  [WARN] MedVAE checkpoint not found: {ckpt_path}")
    else:
        print(f"  MedVAE frozen loaded from HuggingFace ({model_name})")

    latent_ch = _infer_medvae_latent_channels(model)
    return MedVAEWrapper(model, latent_ch)


# --------------------------------------------------------------------------- #
# VQ-VAE (NeuroQuant)                                                         #
# --------------------------------------------------------------------------- #


def _load_vqvae(vae_cfg: dict, device: torch.device) -> VQVAEWrapper:
    """Load NeuroQuantHybrid VQ-VAE."""
    try:
        from vae3d.train_vqvae import NeuroQuantHybrid
    except ImportError:
        from train_vqvae import NeuroQuantHybrid

    vae_config_path = vae_cfg.get("vae_config", "configs/vqvae3d_T1W.yaml")
    if not os.path.isabs(vae_config_path):
        project_root = Path(__file__).resolve().parents[2]
        vae_config_path = str(project_root / vae_config_path)

    if Path(vae_config_path).exists():
        with open(vae_config_path) as f:
            cfg = yaml.safe_load(f)
        m = cfg.get("model", {})
        model = NeuroQuantHybrid(
            n_modalities=m.get("n_modalities", 3),
            n_fields=m.get("n_fields", 5),
            base_channels=m.get("base_channels", 64),
            anat_channels=m.get("anat_channels", 64),
            mod_channels=m.get("mod_channels", 32),
            codebook_size=m.get("codebook_size", 4096),
            vq_decay=m.get("vq_decay", 0.99),
            vq_beta=m.get("vq_beta", 0.25),
        )
    else:
        print(f"  [WARN] VQ-VAE config not found: {vae_config_path}")
        model = NeuroQuantHybrid()

    ckpt_path = vae_cfg.get("checkpoint", "")
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        current = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
        skipped = len(state) - len(compatible)
        model.load_state_dict(compatible, strict=False)
        print(f"  VQ-VAE loaded from {ckpt_path} ({skipped} keys skipped)")
    else:
        print(f"  [WARN] VQ-VAE checkpoint not found: {ckpt_path}")

    return VQVAEWrapper(model)


# --------------------------------------------------------------------------- #
# MedVAE Disentanglement                                                      #
# --------------------------------------------------------------------------- #


def _load_medvae_disentangle(vae_cfg: dict, device: torch.device) -> MedVAEDisentangleWrapper:
    """Load MedVAE DisentanglerV1."""
    try:
        from vae3d.train_medvae_disentangle_v1 import MedVAEDisentanglerV1, load_medvae
    except ImportError:
        from train_medvae_disentangle_v1 import MedVAEDisentanglerV1, load_medvae

    ckpt_path = vae_cfg.get("checkpoint", "")
    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(f"MedVAE disentangle checkpoint required: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = raw.get("model", raw)
    args_d = raw.get("args", {}) if isinstance(raw, dict) else {}

    anat_channels = int(args_d.get("anat_channels", 8))
    style_dim = int(args_d.get("style_dim", 32))
    film_hidden = int(args_d.get("film_hidden", 128))
    n_modalities = int(args_d.get("n_modalities", len(vae_cfg.get("modalities", ["T1W", "T2W", "T2FLAIR"]))))

    medvae = load_medvae(vae_cfg.get("model_name", "medvae_4_1_3d"), device)
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 112, 128, 80, device=device)
        z_dummy = medvae.encode(dummy)
        if isinstance(z_dummy, (tuple, list)):
            z_dummy = z_dummy[0]
        latent_channels = int(z_dummy.shape[1])

    model = MedVAEDisentanglerV1(
        medvae=medvae,
        latent_channels=latent_channels,
        n_modalities=n_modalities,
        anat_channels=anat_channels,
        style_dim=style_dim,
        film_hidden=film_hidden,
    ).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] MedVAE-disentangle {len(missing)} missing keys")
    if unexpected:
        print(f"  [WARN] MedVAE-disentangle {len(unexpected)} unexpected keys")

    model.eval()
    return MedVAEDisentangleWrapper(model)


# --------------------------------------------------------------------------- #
# Pythae VAE 3D                                                               #
# --------------------------------------------------------------------------- #


def _load_pythae_vae(vae_cfg: dict, device: torch.device):
    """Load a Pythae VAE 3D (PythaeVAE3D wrapper)."""
    from models.pythae_vae import build_pythae_vae_3d, PythaeVAE3D

    latent_channels = int(vae_cfg.get("latent_channels", 8))
    base_channels = int(vae_cfg.get("base_channels", 32))
    num_groups = int(vae_cfg.get("num_groups", 8))

    model = build_pythae_vae_3d(
        latent_channels=latent_channels,
        base_channels=base_channels,
        num_groups=num_groups,
    )

    ckpt_path = vae_cfg.get("checkpoint", "")
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        model.load_state_dict(state, strict=True)
        print(f"  Pythae VAE 3D loaded from {ckpt_path}")
    else:
        print(f"  [WARN] Pythae VAE 3D checkpoint not found: {ckpt_path} — using random weights")

    return model


# --------------------------------------------------------------------------- #
# Pythae VQ-VAE 3D                                                            #
# --------------------------------------------------------------------------- #


def _load_pythae_vqvae(vae_cfg: dict, device: torch.device):
    """Load a Pythae VQ-VAE 3D (PythaeVQVAE3D wrapper)."""
    from models.pythae_vqvae import build_pythae_vqvae_3d, PythaeVQVAE3D

    latent_channels = int(vae_cfg.get("latent_channels", 8))
    base_channels = int(vae_cfg.get("base_channels", 32))
    num_embeddings = int(vae_cfg.get("num_embeddings", 512))
    commitment_loss_factor = float(vae_cfg.get("commitment_loss_factor", 0.25))
    quantization_loss_factor = float(vae_cfg.get("quantization_loss_factor", 1.0))
    use_ema = bool(vae_cfg.get("use_ema", True))
    decay = float(vae_cfg.get("decay", 0.99))
    num_groups = int(vae_cfg.get("num_groups", 8))

    model = build_pythae_vqvae_3d(
        latent_channels=latent_channels,
        base_channels=base_channels,
        num_embeddings=num_embeddings,
        commitment_loss_factor=commitment_loss_factor,
        quantization_loss_factor=quantization_loss_factor,
        use_ema=use_ema,
        decay=decay,
        num_groups=num_groups,
    )

    ckpt_path = vae_cfg.get("checkpoint", "")
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        model.load_state_dict(state, strict=True)
        print(f"  Pythae VQ-VAE 3D loaded from {ckpt_path}")
    else:
        print(f"  [WARN] Pythae VQ-VAE 3D checkpoint not found: {ckpt_path} — using random weights")

    return model
