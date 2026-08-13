#!/usr/bin/env python3
"""MONAI DiffusionModelUNet architecture adapter for the unified MMFM trainer
(mmfm_core.py).

This is the *only* file that should know about the spatial UNet's call
convention (cat(z_t, z_src) along the channel axis, multiple-of-4 spatial
padding) — everything else in the training loop (sampler, OT-CFM, EMA,
optimizer, checkpointing, cycle/edge regularizers) is architecture-agnostic
and lives in `mmfm_core.py`.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Callable, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from monai.networks.nets import DiffusionModelUNet
except ImportError:
    try:
        from monai.generative.networks.nets import DiffusionModelUNet
    except ImportError:
        from generative.networks.nets import DiffusionModelUNet

from common.dataset import LatentCacheDataset

REQUIRES_SPATIAL_VAE = True
DEFAULT_CACHE_ROOT = "outputs/mmfm/latent_cache/unet"


def _pad_to_multiple(z: Tensor, mult: int = 4) -> Tuple[Tensor, Tuple[int, int, int]]:
    """Zero-pad the spatial dims of a (B, C, H, W, D) tensor up to a multiple
    of `mult` (the 3-level UNet's downsample constraint). Returns
    (z_padded, orig_spatial_shape) so the output can be cropped back."""
    h, w, d = z.shape[-3:]
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    pd = (mult - d % mult) % mult
    if ph or pw or pd:
        z = F.pad(z, (0, pd, 0, pw, 0, ph), mode="constant", value=0.0)
    return z, (h, w, d)


def _crop_to_shape(z: Tensor, shape: Tuple[int, int, int]) -> Tensor:
    h, w, d = shape
    return z[..., :h, :w, :d]


def remap_monai_attention_keys(state_dict: dict) -> dict:
    """Remap attention submodule keys between MONAI API versions.

    Old (Jean Zay pytorch-gpu/py3/2.5.0, MONAI <=1.3):
        <prefix>.to_q / to_k / to_v / proj_attn
    New (MONAI >=1.4, local):
        <prefix>.attn.to_q / attn.to_k / attn.to_v / attn.out_proj
    """
    new_sd = {}
    _remap = {
        r"(\.attentions\.\d+)\.to_q\.(weight|bias)$":    r"\1.attn.to_q.\2",
        r"(\.attentions\.\d+)\.to_k\.(weight|bias)$":    r"\1.attn.to_k.\2",
        r"(\.attentions\.\d+)\.to_v\.(weight|bias)$":    r"\1.attn.to_v.\2",
        r"(\.attentions\.\d+)\.proj_attn\.(weight|bias)$": r"\1.attn.out_proj.\2",
        r"(\.attention)\.to_q\.(weight|bias)$":           r"\1.attn.to_q.\2",
        r"(\.attention)\.to_k\.(weight|bias)$":           r"\1.attn.to_k.\2",
        r"(\.attention)\.to_v\.(weight|bias)$":           r"\1.attn.to_v.\2",
        r"(\.attention)\.proj_attn\.(weight|bias)$":      r"\1.attn.out_proj.\2",
    }
    for k, v in state_dict.items():
        new_k = k
        for pat, repl in _remap.items():
            new_k2 = re.sub(pat, repl, new_k)
            if new_k2 != new_k:
                new_k = new_k2
                break
        new_sd[new_k] = v
    return new_sd


def build_unet_3d(cfg: dict, latent_channels: int, n_classes: int) -> nn.Module:
    """UNet 3D conditionné sur le temps (champ) et la classe (contraste) cible.

    V1 : DiffusionModelUNet standard
    V2 : HybridUNetTransformer avec attention factorisée au bottleneck
    """
    m = cfg["model"]
    channel_mult = tuple(m.get("channel_mult", [1, 2, 4]))
    base_channels = m.get("model_channels", 128)
    channels = tuple(base_channels * c for c in channel_mult)

    _sig = inspect.signature(DiffusionModelUNet.__init__).parameters
    _ch_kwarg = "num_channels" if "num_channels" in _sig else "channels"

    num_class_embeds = int(m.get("num_class_embeds", n_classes))

    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=2 * latent_channels,
        out_channels=latent_channels,
        **{_ch_kwarg: channels},
        attention_levels=tuple(m.get("attention_levels", [False, True, True])),
        num_res_blocks=m.get("num_res_blocks", 2),
        num_head_channels=m.get("num_head_channels", 64),
        norm_num_groups=m.get("norm_num_groups", 32),
        use_flash_attention=m.get("use_flash_attention", False),
        num_class_embeds=num_class_embeds,
        with_conditioning=False,
        resblock_updown=True,
    )

    # AdaGN : la group-norm de chaque bloc résiduel devient modulée (gain +
    # décalage) au lieu de recevoir un simple décalage additif AVANT une
    # normalisation qui en efface une partie. Voir models/adagn_conditioning.py.
    # Conversion faite APRÈS construction : on reprend les sous-modules de MONAI
    # tels quels, sans dépendre de la signature de son constructeur.
    if m.get("use_adagn", False):
        from models.adagn_conditioning import convert_unet_to_adagn, count_resnet_blocks
        n_blocks = count_resnet_blocks(unet)
        n_conv = convert_unet_to_adagn(unet, zero_init=bool(m.get("adagn_zero_init", True)))
        if n_conv != n_blocks or n_conv == 0:
            raise RuntimeError(
                f"Conversion AdaGN incomplète : {n_conv}/{n_blocks} blocs convertis. "
                "La structure de DiffusionModelUNet a probablement changé de version."
            )
        print(f"  [build_unet_3d] AdaGN activé : {n_conv} blocs résiduels convertis "
              f"(zero_init={m.get('adagn_zero_init', True)})")

    use_factorized = m.get("use_factorized_attention", False)
    if use_factorized:
        from models.hybrid_unet_transformer import HybridUNetTransformer
        bottleneck_ch = base_channels * max(channel_mult)
        return HybridUNetTransformer(
            unet=unet,
            bottleneck_channels=bottleneck_ch,
            use_factorized_attention=True,
            num_attn_heads=m.get("factorized_attn_heads", 8),
            attn_dropout=m.get("factorized_attn_dropout", 0.0),
        )

    return unet


def _validate_cache_shape(ds, latent_shape: Tuple[int, ...]) -> None:
    cache_shape = tuple(ds.latent_shape) if ds.latent_shape else None
    if cache_shape and cache_shape != tuple(latent_shape):
        raise RuntimeError(
            f"Incohérence cache de latents : latent_shape du cache={cache_shape}, "
            f"attendu (VAE + volume_size courants)={latent_shape}. "
            "Le cache a probablement été généré avec une config différente — "
            "relancer precompute_unet_latents.py."
        )


def _build_cache_dataset(cache_dir: Path, cache_root: Path, data_cfg: dict):
    return LatentCacheDataset(
        cache_dir=cache_dir,
        cache_root=cache_root,
        preload_ram=bool(data_cfg.get("latent_preload_ram", True)),
        flip_lr_prob=float(data_cfg.get("flip_lr_prob", 0.0)),
        flip_axis=int(data_cfg.get("flip_axis", 0)),
    )


def make_adapter(cfg: dict, latent_shape: Tuple[int, ...], n_classes: int):
    from cfm.mmfm_core import ArchAdapter  # deferred: avoids a circular import at module load

    latent_channels = int(latent_shape[0])

    def build_model() -> nn.Module:
        return build_unet_3d(cfg, latent_channels, n_classes)

    def make_model_fn(raw_model: nn.Module) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
        def _fn(z_t: Tensor, z_src: Tensor, t: Tensor, y: Tensor) -> Tensor:
            z_in = torch.cat([z_t, z_src], dim=1)
            return raw_model(x=z_in, timesteps=t, class_labels=y)
        return _fn

    def prep_latent(vae, z: Tensor) -> Tuple[Tensor, Any]:
        return _pad_to_multiple(z, 4)

    def restore_latent(z: Tensor, meta: Any) -> Tensor:
        return _crop_to_shape(z, meta)

    def arch_meta_dict() -> dict:
        return {"latent_channels": latent_channels, "num_class_embeds": int(n_classes)}

    return ArchAdapter(
        name="mmfm3d_unet",
        build_model=build_model,
        make_model_fn=make_model_fn,
        prep_latent=prep_latent,
        restore_latent=restore_latent,
        checkpoint_key_remap=remap_monai_attention_keys,
        arch_meta_dict=arch_meta_dict,
        # LatentCacheDataset stores raw (unpadded) spatial latents — padding
        # is always applied fresh at train time, cache or not (unlike the
        # vectorized flat cache, which pre-bakes the flatten — see
        # arch_vector.py).
        cache_prebakes_prep=False,
        build_cache_dataset=_build_cache_dataset,
        validate_cache_shape=_validate_cache_shape,
        default_cache_root=DEFAULT_CACHE_ROOT,
    )
