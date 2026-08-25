#!/usr/bin/env python3
"""Vector-MLP architecture adapter for the unified MMFM trainer (mmfm_core.py).

Wraps `VectorMMFM` (src/cfm/mmfm_vectorized.py) behind the `ArchAdapter`
protocol defined in `mmfm_core.py`. This is the *only* file that should know
about the vectorized MLP's call convention (flat vector in/out, no spatial
structure) — everything else in the training loop (sampler, OT-CFM, EMA,
optimizer, checkpointing, cycle/edge regularizers) is architecture-agnostic
and lives in `mmfm_core.py`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor, nn

from cfm.mmfm_vectorized import LatentVectorizer, VectorMMFM
from common.dataset import FlatLatentCacheDataset

REQUIRES_SPATIAL_VAE = False
DEFAULT_CACHE_ROOT = "outputs/mmfm/latent_cache/vectorized"


def build_vector_mmfm(cfg: dict, latent_dim: int, n_classes: int) -> VectorMMFM:
    m = cfg["model"]
    return VectorMMFM(
        latent_dim=latent_dim,
        num_classes=n_classes,
        hidden_dim=int(m.get("hidden_dim", 1024)),
        depth=int(m.get("num_blocks", 4)),
        time_embed_dim=int(m.get("time_embed_dim", 256)),
        class_embed_dim=int(m.get("class_embed_dim", 128)),
        dropout=float(m.get("dropout", 0.0)),
        # (0, 1) par defaut : sans effet, les checkpoints existants restent valides.
        latent_mean=float(m.get("latent_mean", 0.0)),
        latent_scale=float(m.get("latent_scale", 1.0)),
    )


def _validate_cache_shape(ds, latent_shape: Tuple[int, ...]) -> None:
    latent_dim = int(math.prod(latent_shape))
    if ds.flat_dim is not None and ds.flat_dim != latent_dim:
        raise RuntimeError(
            f"Incohérence cache de latents : flat_dim du cache={ds.flat_dim}, "
            f"attendu (VAE + volume_size courants)={latent_dim}. "
            "Le cache a probablement été généré avec une config différente — "
            "relancer precompute_mmfm_latents.py."
        )


def make_adapter(cfg: dict, latent_shape: Tuple[int, ...], n_classes: int):
    from cfm.mmfm_core import ArchAdapter  # deferred: avoids a circular import at module load

    vectorizer = LatentVectorizer(latent_shape)
    latent_dim = vectorizer.flat_dim

    def build_model() -> nn.Module:
        return build_vector_mmfm(cfg, latent_dim, n_classes)

    def make_model_fn(raw_model: nn.Module) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
        return lambda z_t, z_src, t, y: raw_model(z_t, z_src, t, y)

    def prep_latent(vae, z: Tensor) -> Tuple[Tensor, Any]:
        # Phase F precedent (train_mmfm_3d.py): vae.to_vector() is the
        # canonical flatten — identity for vector-native VAEs (RHVAE),
        # reshape-to-flat for spatial VAEs (MedVAE/AEKL).
        return vae.to_vector(z), None

    def restore_latent(z: Tensor, meta: Any) -> Tensor:
        # Deliberately NOT vae.from_vector(): that assumes a perfect cube
        # latent shape (see train_mmfm_3d.py inference comment), which does
        # not hold for non-cubic volume_size. LatentVectorizer.unflatten()
        # reshapes against the actual latent_shape instead.
        return vectorizer.unflatten(z)

    def checkpoint_key_remap(state_dict: dict) -> dict:
        return state_dict

    def arch_meta_dict() -> dict:
        return {"latent_shape": tuple(int(v) for v in latent_shape)}

    def _build_cache_dataset(cache_dir: Path, cache_root: Path, data_cfg: dict):
        # Fermeture (et non fonction de module) pour capturer `latent_shape` :
        # le cache plat n'enregistre que `flat_dim`, or retourner un latent
        # aplati exige de connaître sa forme spatiale. C'est ce qui manquait —
        # `flip_lr_prob` était déclaré dans la config mais la classe ne
        # l'acceptait pas, donc seul l'UNet recevait l'augmentation.
        return FlatLatentCacheDataset(
            cache_dir=cache_dir,
            cache_root=cache_root,
            preload_ram=bool(data_cfg.get("latent_preload_ram", True)),
            flip_lr_prob=float(data_cfg.get("flip_lr_prob", 0.0)),
            flip_axis=int(data_cfg.get("flip_axis", 0)),
            latent_shape=tuple(int(v) for v in latent_shape),
        )

    return ArchAdapter(
        name="mmfm3d_vectorized",
        build_model=build_model,
        make_model_fn=make_model_fn,
        prep_latent=prep_latent,
        restore_latent=restore_latent,
        checkpoint_key_remap=checkpoint_key_remap,
        arch_meta_dict=arch_meta_dict,
        # FlatLatentCacheDataset pre-flattens at precompute time (see
        # precompute_mmfm_latents.py) — prep_latent must NOT run again on
        # cached samples, unlike the spatial UNet cache (arch_unet.py).
        cache_prebakes_prep=True,
        build_cache_dataset=_build_cache_dataset,
        validate_cache_shape=_validate_cache_shape,
        default_cache_root=DEFAULT_CACHE_ROOT,
    )
