#!/usr/bin/env python3
"""INR architecture adapter for the unified MMFM trainer (mmfm_core.py).

Wraps a frozen, separately meta-trained INR backbone (SIREN + hypernetwork,
cfm/inr_backbone.py, trained by cfm/train_inr_backbone.py) behind the
ArchAdapter protocol. Unlike arch_vector.py/arch_unet.py, this adapter does
NOT use MedVAE at all — the INR operates directly on raw volumes (confirmed
design decision, see docs/MMFM_INR_STATE_OF_THE_ART.md section (C)): the
config's `vae:` block must be `vae_type: identity`
(models/vae_wrappers.py's IdentityVAEWrapper), a pure pass-through that lets
mmfm_core.py's train()/infer() run completely unmodified while the REAL
per-volume work (INR meta-fitting) happens in this adapter's
prep_latent/restore_latent.

Since the INR's fitted latent `z` is a flat vector — exactly like the
vectorized MLP's flattened MedVAE latent — the flow model itself is
`VectorMMFM` (see arch_vector.py): no new flow architecture needed.

The backbone + coordinate grid are loaded lazily, on the device of the first
tensor prep_latent/restore_latent actually sees — make_adapter() itself
receives no device argument (see mmfm_core.ArchAdapter), so guessing it here
would silently break under DDP (LOCAL_RANK > 0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch
from torch import Tensor, nn

from cfm.arch_vector import build_vector_mmfm
from cfm.inr_backbone import (
    DEFAULT_FIT_CHUNK, INRBackbone, INRBackboneConfig, decode_volume, fit_new_volume, make_coord_grid,
)
from common.dataset import FlatLatentCacheDataset

REQUIRES_SPATIAL_VAE = False
DEFAULT_CACHE_ROOT = "outputs/mmfm/latent_cache/inr"


def _backbone_config_from_cfg(cfg: dict) -> INRBackboneConfig:
    b = cfg.get("inr_backbone", {})
    return INRBackboneConfig(
        latent_dim=int(b.get("latent_dim", 512)),
        hidden_dim=int(b.get("hidden_dim", 256)),
        num_hidden_layers=int(b.get("num_hidden_layers", 6)),
        omega_0=float(b.get("omega_0", 30.0)),
        omega_hidden=float(b.get("omega_hidden", 30.0)),
        hyper_hidden_dim=int(b.get("hyper_hidden_dim", 64)),
        inner_lr=float(b.get("inner_lr", 1e-2)),
        inner_steps_train=int(b.get("inner_steps_train", 5)),
        inner_steps_eval=int(b.get("inner_steps_eval", 10)),
        fg_weight=float(b.get("fg_weight", 5.0)),
        bg_threshold=float(b.get("bg_threshold", -0.9)),
        modulate_scale=bool(b.get("modulate_scale", False)),
        lora_rank=int(b.get("lora_rank", 0)),
    )


def load_inr_backbone(cfg: dict, device: torch.device) -> INRBackbone:
    """Load the frozen, meta-trained INR backbone (theta, psi) — trained
    separately via train_inr_backbone.py, mirroring how load_vae() loads a
    frozen MedVAE checkpoint. Prefers EMA weights when present, matching
    mmfm_core.py's inference convention for the flow models."""
    ckpt_path = cfg.get("inr_backbone", {}).get("checkpoint")
    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"inr_backbone.checkpoint introuvable : {ckpt_path!r}. "
            "Entraîner d'abord via train_inr_backbone.py "
            "(configs/mmfm/inr_backbone.yaml)."
        )
    backbone = INRBackbone(_backbone_config_from_cfg(cfg)).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if state.get("ema"):
        backbone.load_state_dict(state["ema"])
    else:
        backbone.load_state_dict(state["model"])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def _validate_cache_shape(ds, latent_shape: Tuple[int, ...]) -> None:
    # latent_shape here is the identity-VAE's dummy-encode shape (1, H, W, D)
    # — a raw *volume* shape, not the INR's fitted flat latent_dim, so it
    # can't validate the cache directly. The cache is already protected by
    # its own content-addressed cache_id (backbone checkpoint + fitting
    # hyperparameters + preprocessing — see precompute_inr_latents.py).
    pass


def _build_cache_dataset(cache_dir: Path, cache_root: Path, data_cfg: dict):
    # PAS de flip ici, et ce n'est pas un oubli : le latent de cette
    # architecture est le `z` ajusté par l'Algorithme 2 de NOIR, un vecteur de
    # modulation GLOBAL sans aucune structure spatiale. Le retourner n'a pas de
    # sens géométrique — la seule façon correcte d'augmenter par symétrie
    # serait de réajuster `z` sur le volume retourné, donc de refaire le
    # precompute (~6 h) avec des volumes retournés.
    # `latent_shape` vaut ici (C, *volume_size) et non la forme d'un latent
    # spatial (cf. make_adapter) : l'utiliser pour un flip remettrait `z` dans
    # une forme qui n'est pas la sienne.
    if float(data_cfg.get("flip_lr_prob", 0.0)) > 0:
        raise ValueError(
            "flip_lr_prob > 0 est incompatible avec l'architecture INR : son latent "
            "est un vecteur de modulation global, pas un latent spatial — le "
            "retourner n'a pas de sens. Mettre flip_lr_prob: 0.0 dans "
            "configs/mmfm/inr.yaml (voir le commentaire de cette clé)."
        )
    return FlatLatentCacheDataset(
        cache_dir=cache_dir, cache_root=cache_root,
        preload_ram=bool(data_cfg.get("latent_preload_ram", True)),
    )


def make_adapter(cfg: dict, latent_shape: Tuple[int, ...], n_classes: int):
    from cfm.mmfm_core import ArchAdapter  # deferred: avoids a circular import at module load

    # latent_shape = _infer_latent_shape(identity_vae, volume_size, device)
    # = (channels, *volume_size) since encode() is a no-op — see IdentityVAEWrapper.
    volume_size = tuple(int(v) for v in latent_shape[1:])
    backbone_cfg = _backbone_config_from_cfg(cfg)
    if not cfg.get("inr_backbone", {}).get("checkpoint"):
        raise FileNotFoundError("inr_backbone.checkpoint manquant dans la config.")
    latent_dim = backbone_cfg.latent_dim
    # Fitting/decoding a full 1mm grid (8.25M coordinates) in one pass would
    # need ~100 GB of SIREN activations, so both are chunked (exactly — see
    # inr_backbone.fit_latent). `fit_points` must match whatever
    # precompute_inr_latents.py used for the cache the flow was trained on:
    # a z fitted on a different point budget is a different z, and inference
    # would then feed the flow an out-of-distribution source latent.
    fit_points = cfg.get("inr_backbone", {}).get("fit_points")
    fit_points = int(fit_points) if fit_points else None
    fit_chunk = int(cfg.get("inr_backbone", {}).get("fit_chunk", DEFAULT_FIT_CHUNK))

    # Lazy, device-keyed cache: avoids guessing the device inside make_adapter
    # (not part of the ArchAdapter signature) and stays correct under DDP.
    _state: Dict[str, Any] = {"device": None, "backbone": None, "coords": None}

    def _ensure_loaded(device: torch.device) -> None:
        if _state["device"] == device:
            return
        _state["backbone"] = load_inr_backbone(cfg, device)
        _state["coords"] = make_coord_grid(volume_size, device=device)
        _state["device"] = device

    def build_model() -> nn.Module:
        return build_vector_mmfm(cfg, latent_dim, n_classes)

    def make_model_fn(raw_model: nn.Module) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
        return lambda z_t, z_src, t, y: raw_model(z_t, z_src, t, y)

    def prep_latent(vae, z: Tensor) -> Tuple[Tensor, Any]:
        # `z` here is the identity-VAE "encoded" tensor, i.e. the raw volume
        # unchanged (B, 1, H, W, D). The real work — meta-learned fitting
        # (NOIR Algorithm 2) — happens here, not in the (no-op) vae.
        _ensure_loaded(z.device)
        zfit = fit_new_volume(
            _state["backbone"], z, _state["coords"],
            num_steps=backbone_cfg.inner_steps_eval, lr=backbone_cfg.inner_lr,
            num_points=fit_points, chunk_size=fit_chunk,
        )
        return zfit, None

    def restore_latent(z: Tensor, meta: Any) -> Tensor:
        _ensure_loaded(z.device)
        return decode_volume(_state["backbone"], z, _state["coords"], volume_size, chunk_size=fit_chunk)

    def checkpoint_key_remap(state_dict: dict) -> dict:
        return state_dict

    def arch_meta_dict() -> dict:
        return {
            "latent_dim": int(latent_dim),
            "volume_size": tuple(int(v) for v in volume_size),
            "inr_backbone_checkpoint": str(cfg.get("inr_backbone", {}).get("checkpoint", "")),
            "fit_points": fit_points,
        }

    return ArchAdapter(
        name="mmfm3d_inr",
        build_model=build_model,
        make_model_fn=make_model_fn,
        prep_latent=prep_latent,
        restore_latent=restore_latent,
        checkpoint_key_remap=checkpoint_key_remap,
        arch_meta_dict=arch_meta_dict,
        # FlatLatentCacheDataset pre-fits (Algorithm 2) at precompute time
        # (precompute_inr_latents.py) — prep_latent must NOT run again on
        # cached samples: unlike MedVAE's cheap forward-pass encode, the INR
        # fit is a multi-step optimization, far too expensive to redo every
        # training step.
        cache_prebakes_prep=True,
        build_cache_dataset=_build_cache_dataset,
        validate_cache_shape=_validate_cache_shape,
        default_cache_root=DEFAULT_CACHE_ROOT,
    )
