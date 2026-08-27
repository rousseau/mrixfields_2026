#!/usr/bin/env python3
"""Adaptateur d'architecture pour le probleme SYNTHETIQUE (cfm/synthetic_marginals.py).

Purement ADDITIF : aucune ligne du code d'entrainement partage n'est modifiee.
`mmfm_core.train()` tourne tel quel, ce qui est exactement l'interet du harnais
— il exerce `_sample_step_plan`, `_compute_flow`, le sampler OT-CFM, la loss,
l'EMA et `euler_integrate` REELS, pas une reimplementation qui pourrait diverger
du code de production et valider quelque chose qui n'est pas teste.

Le modele de flow est `VectorMMFM` (arch_vector.build_vector_mmfm), le meme que
le vectorise et l'INR : le correctif d'echelle du temps y est donc teste sur le
chemin de code reel.

Le jeu de donnees etant genere, il n'y a pas de cache a lire : `build_cache_dataset`
ignore les chemins et construit le dataset synthetique. `use_latent_cache: true`
dans la config est donc un branchement, pas un vrai cache disque.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Tuple

import torch
from torch import Tensor, nn

from cfm.arch_vector import build_vector_mmfm
from cfm.synthetic_marginals import SyntheticLatentDataset, SyntheticSpec

REQUIRES_SPATIAL_VAE = False
DEFAULT_CACHE_ROOT = "outputs/mmfm/synthetic"


def spec_from_cfg(cfg: dict) -> SyntheticSpec:
    s = cfg.get("synthetic", {})
    data = cfg.get("data", {})
    return SyntheticSpec(
        dim=int(s.get("dim", 64)),
        n_fields=len(data.get("fields", [0, 1, 2, 3, 4])),
        n_contrasts=len(data.get("modalities", [0, 1, 2])),
        bend=float(s.get("bend", 0.6)),
        subject_scale=float(s.get("subject_scale", 1.0)),
        noise=float(s.get("noise", 0.05)),
        seed=int(s.get("seed", 0)),
    )


def make_adapter(cfg: dict, latent_shape: Tuple[int, ...], n_classes: int):
    from cfm.mmfm_core import ArchAdapter  # differe : evite un import circulaire

    spec = spec_from_cfg(cfg)
    latent_dim = spec.dim

    def build_model() -> nn.Module:
        return build_vector_mmfm(cfg, latent_dim, n_classes)

    def make_model_fn(raw_model: nn.Module) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
        def _fn(z_t: Tensor, z_src: Tensor, t: Tensor, y: Tensor) -> Tensor:
            return raw_model(z_t, z_src, t, y)
        return _fn

    def prep_latent(vae, z: Tensor) -> Tuple[Tensor, Any]:
        return z.reshape(z.shape[0], -1), None

    def restore_latent(z: Tensor, meta: Any) -> Tensor:
        return z

    def arch_meta_dict() -> dict:
        return {
            "latent_dim": latent_dim,
            "synthetic_spec": {
                "dim": spec.dim, "bend": spec.bend, "noise": spec.noise,
                "subject_scale": spec.subject_scale, "seed": spec.seed,
            },
        }

    def build_cache_dataset(cache_dir: Path, cache_root: Path, data_cfg: dict):
        # Les chemins sont ignores : les donnees sont generees, pas lues.
        return SyntheticLatentDataset(
            spec,
            n_per_class=int(data_cfg.get("n_per_class", 256)),
            seed=int(data_cfg.get("data_seed", 1)),
        )

    def validate_cache_shape(ds, expected_latent_shape: Tuple[int, ...]) -> None:
        if ds.flat_dim != latent_dim:
            raise RuntimeError(
                f"dimension synthetique incoherente : dataset {ds.flat_dim}, "
                f"modele {latent_dim}."
            )

    return ArchAdapter(
        name="mmfm3d_synthetic",
        build_model=build_model,
        make_model_fn=make_model_fn,
        prep_latent=prep_latent,
        restore_latent=restore_latent,
        checkpoint_key_remap=lambda sd: sd,
        arch_meta_dict=arch_meta_dict,
        cache_prebakes_prep=True,
        build_cache_dataset=build_cache_dataset,
        validate_cache_shape=validate_cache_shape,
        default_cache_root=DEFAULT_CACHE_ROOT,
    )
