#!/usr/bin/env python3
"""Unified MMFM training/inference core — shared by the vectorized-MLP and
spatial-UNet architectures.

This module owns everything that is NOT architecture-specific: the
multi-marginal sampler, the OT-CFM flow construction, the cycle/edge
anatomy-preservation regularizers, EMA/optimizer/scheduler, checkpointing,
and logging. The only point of legitimate variation between architectures is
the `ArchAdapter` each one provides (see arch_vector.py / arch_unet.py):

  - build_model()            : construct the raw nn.Module
  - make_model_fn(raw_model) : wrap it into model_fn(z_t, z_src, t, y) -> v_t
  - prep_latent(vae, z)      : VAE-latent -> model-space latent (+ meta)
  - restore_latent(z, meta)  : model-space latent -> VAE-latent
  - checkpoint_key_remap     : state_dict compat shim (no-op for the MLP)
  - arch_meta_dict()         : architecture metadata stored in checkpoints
  - cache_prebakes_prep      : whether the latent cache already applied
                                prep_latent at precompute time
  - build_cache_dataset      : construct the arch's latent-cache Dataset
  - validate_cache_shape     : sanity-check the cache against latent_shape
  - default_cache_root       : default outputs/latent_cache_* root

Both `cycle_rollout_vector` (cfm/mmfm_vectorized.py) and
`Sobel3D`/`edge_consistency_loss` (cfm/edge_loss_3d.py) are already
architecture-agnostic — they operate purely through `model_fn`, never on the
raw module — and need no adapter of their own.

The multi-marginal sampler is availability-aware (only samples transitions
within a contrast's actually-loaded fields, honoring `num_targets_per_step`)
and the flow is always constructed via real OT coupling
(`FM.sample_location_and_conditional_flow`) for genuine transitions. Both
architectures used to have their own, non-equivalent implementations here
(the vectorized MLP never used real OT-CFM, and ignored
`num_targets_per_step`) — unified in favor of the UNet's already-correct
behavior (see the unification plan for the A/B verification that preceded
this switch).
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from common.config import load_yaml_with_include, load_env, resolve_paths
from common.dataset import MultiModalNIfTILatentDataset, flat_latent_cache_id
from common.distributed import is_main_process, EMAModel
from common.io import (
    DOMAINS,
    MODALITIES,
    adjust_affine_for_crop_pad,
    denormalize_from_01,
    load_nifti_volume,
    resample_volume,
)
from models.tiled_vae import tiled_encode
from models.vae_loader import load_vae

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)

from cfm.mmfm_vectorized import cycle_rollout_vector
from cfm.edge_loss_3d import Sobel3D, edge_consistency_loss


# ===========================================================================
# ArchAdapter — the sole point of architecture-specific variation
# ===========================================================================


@dataclass
class ArchAdapter:
    name: str
    build_model: Callable[[], Any]
    make_model_fn: Callable[[Any], Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]]
    prep_latent: Callable[[Any, Tensor], Tuple[Tensor, Any]]
    restore_latent: Callable[[Tensor, Any], Tensor]
    checkpoint_key_remap: Callable[[dict], dict]
    arch_meta_dict: Callable[[], dict]
    cache_prebakes_prep: bool
    build_cache_dataset: Callable[[Path, Path, dict], Any]
    validate_cache_shape: Callable[[Any, Tuple[int, ...]], None]
    default_cache_root: str


def _resolve_arch_module(method: str):
    if method == "mmfm3d_vectorized":
        from cfm import arch_vector as arch_module
    elif method == "mmfm3d_unet":
        from cfm import arch_unet as arch_module
    elif method == "mmfm3d_inr":
        from cfm import arch_inr as arch_module
    else:
        raise ValueError(
            f"method='{method}' inconnu — attendu 'mmfm3d_vectorized', 'mmfm3d_unet' ou 'mmfm3d_inr'."
        )
    return arch_module


# ===========================================================================
# Helpers (architecture-agnostic)
# ===========================================================================


def _flat_class(mod_idx: int, field_idx: int, n_fields: int) -> int:
    return mod_idx * n_fields + field_idx


def _unflat_class(flat: int, n_fields: int) -> Tuple[int, int]:
    return flat // n_fields, flat % n_fields


def _field_to_time(field_idx: int, n_fields: int) -> float:
    if n_fields <= 1:
        return 0.0
    return float(field_idx) / (n_fields - 1)


def _lerp_lambda(init: float, final: float, step: int, total_iters: int) -> float:
    """Linear interpolation of a regularizer weight from `init` (step=0) to
    `final` (step=total_iters). Returns `init` unchanged if final == init."""
    if total_iters <= 0 or final == init:
        return init
    frac = min(1.0, max(0.0, step / total_iters))
    return init + (final - init) * frac


def _make_infinite(loader: DataLoader):
    while True:
        yield from loader


def _infer_latent_shape(vae, volume_size: Tuple[int, ...], device: torch.device) -> Tuple[int, ...]:
    dummy = torch.zeros(1, 1, *volume_size, device=device)
    with torch.no_grad():
        z = vae.encode(dummy)
    return tuple(int(v) for v in z.shape[1:])


def _encode_like_cache(vae, x: Tensor, tile, margin: int, use_amp: bool,
                       amp_dtype: torch.dtype) -> Tensor:
    """Encode EXACTEMENT comme le precompute et l'inférence.

    Sans ce détour, l'entraînement sans cache (`use_latent_cache: false`)
    appelait `vae.encode()` — donc la fenêtre glissante interne de MedVAE —
    alors que `precompute_*_latents.py` et `infer_mmfm_unified.py` utilisent
    `tiled_encode` (tuiles + marge de contexte). Deux schémas d'encodage
    différents produisent des latents de distributions différentes : le modèle
    se serait entraîné sur l'un et aurait été évalué sur l'autre, SANS le
    moindre signal — les formes coïncident, seules les valeurs diffèrent.
    L'encodage par tuiles n'est pas un détail de performance : mesuré, il donne
    SSIM 0.980/0.959 contre 0.934/0.919 pour un découpage naïf.

    `tiled_encode` ne traite qu'un volume à la fois (sa sortie est allouée avec
    une dimension de lot de 1) : on boucle donc sur le lot plutôt que de lui
    passer un tenseur qu'il n'accepte pas.
    """
    if not tile:
        return vae.encode(x)
    return torch.cat(
        [tiled_encode(vae, x[i:i + 1], tile=tuple(tile), margin=margin,
                      use_amp=use_amp, amp_dtype=amp_dtype)
         for i in range(x.shape[0])],
        dim=0,
    )


def _sample_class(ds, i: int) -> int:
    """class_idx accessor: ds.samples is a list of (path, mod, field, class)
    tuples for MultiModalNIfTILatentDataset, or a list of dicts for the
    latent-cache datasets."""
    s = ds.samples[i]
    return s[3] if isinstance(s, tuple) else s["class_idx"]


def _build_class_loaders(
    ds, n_loader_classes: int, batch_size: int, num_workers: int, is_distributed: bool,
) -> Dict[int, Any]:
    persistent = num_workers > 0
    class_loaders: Dict[int, Any] = {}
    for c_idx in range(n_loader_classes):
        class_indices = [i for i in range(len(ds.samples)) if _sample_class(ds, i) == c_idx]
        if not class_indices:
            continue
        subset = torch.utils.data.Subset(ds, class_indices)
        sampler = DistributedSampler(subset, shuffle=True) if is_distributed else None
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=persistent,
            prefetch_factor=(4 if num_workers > 0 else None),
        )
        class_loaders[c_idx] = _make_infinite(loader)
    return class_loaders


def _build_contrast_fields(
    available_classes: List[int], n_fields: int,
) -> Tuple[Dict[int, List[int]], List[int]]:
    contrast_fields: Dict[int, List[int]] = {}
    for flat in available_classes:
        m_idx, f_idx = _unflat_class(flat, n_fields)
        contrast_fields.setdefault(m_idx, []).append(f_idx)
    for m_idx in contrast_fields:
        contrast_fields[m_idx] = sorted(contrast_fields[m_idx])
    usable_contrasts = [m for m, fs in contrast_fields.items() if len(fs) >= 2]
    if not usable_contrasts:
        raise RuntimeError(
            "Aucun contraste n'a >= 2 champs disponibles ; impossible de définir "
            "une trajectoire multi-marginal."
        )
    return contrast_fields, usable_contrasts


def _sample_step_plan(
    contrast_fields: Dict[int, List[int]],
    usable_contrasts: List[int],
    num_targets_per_step: int,
    identity_prob: float,
    adjacent_only: bool,
) -> Tuple[int, List[Tuple[int, int]]]:
    """Pick a contrast (that has >=2 available fields) and `num_targets_per_step`
    (field_i -> field_j) transitions within that contrast's available fields."""
    contrast = random.choice(usable_contrasts)
    fields_avail = contrast_fields[contrast]
    transitions = []
    for _ in range(num_targets_per_step):
        if random.random() < identity_prob:
            fi = random.choice(fields_avail)
            transitions.append((fi, fi))
            continue
        if adjacent_only and len(fields_avail) >= 2:
            pos = random.randrange(len(fields_avail) - 1)
            fi, fj = fields_avail[pos], fields_avail[pos + 1]
            if random.random() < 0.5:
                fi, fj = fj, fi
        else:
            fi, fj = random.sample(fields_avail, 2)
        transitions.append((fi, fj))
    return contrast, transitions


def _compute_flow(
    FM,
    z_src: Tensor,
    z_tgt: Tensor,
    t_i: float,
    t_j: float,
    same: bool,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Construct (t_global, z_t, ut_global) for one src(t_i) -> tgt(t_j)
    transition. Works identically for flat (B, D) or spatial (B, C, H, W, D)
    tensors — no architecture-specific code here.

    same=True (identity, f_src==f_tgt): stationary trajectory — z_t = z_src
    exactly (no interpolation, no OT coupling), ut=0 everywhere, t sampled
    uniformly (velocity is zero at every time along it).

    Otherwise: real OT/independent coupling via
    `FM.sample_location_and_conditional_flow` (torchcfm), projected from its
    local t in [0,1] onto the trajectory's global [t_i, t_j] interval.
    """
    B = z_src.shape[0]
    if same:
        t_global = torch.rand(B, device=device)
        return t_global, z_src, torch.zeros_like(z_src)

    dt = t_j - t_i
    t_local, z_t, ut_local = FM.sample_location_and_conditional_flow(z_src, z_tgt)
    t_local = t_local.to(device)
    t_global = t_i + t_local * dt
    ut_global = torch.clamp(ut_local.to(device) / dt, min=-1e6, max=1e6)
    return t_global, z_t, ut_global


@torch.no_grad()
def euler_integrate(
    model_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    z_src: Tensor,
    y: Tensor,
    t_start: float,
    t_end: float,
    n_steps: int,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Euler integration of the multi-marginal flow from t_start to t_end,
    conditioned on a FIXED source anchor `z_src` throughout (never updated —
    only the running state `z` is). Identical for both architectures once
    `model_fn` unifies the call convention; supports field ascent
    (t_end>t_start) and descent (t_end<t_start) alike via a signed dt."""
    z = z_src.clone().to(device).float()
    z_anchor = z_src.clone().to(device).float()
    dt = (t_end - t_start) / n_steps
    for step_i in range(n_steps):
        t_val = t_start + step_i * dt
        t_vec = torch.full((z.shape[0],), t_val, dtype=torch.float32, device=device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")):
            vt = model_fn(z, z_anchor, t_vec, y)
        z = z + dt * vt.float()
    return z


def _check_arch_meta(ckpt: dict, adapter: ArchAdapter) -> None:
    saved = ckpt.get("arch_meta")
    if saved is None:
        return  # legacy (pre-unification) checkpoint — nothing to check
    current = adapter.arch_meta_dict()
    if saved != current:
        raise RuntimeError(
            f"Incohérence arch_meta : checkpoint={saved}, config courante={current}. "
            "Utilisez la même config qu'à l'entraînement."
        )


def _save_checkpoint(
    path: Path,
    step: int,
    model,
    ema: EMAModel,
    optimizer,
    scheduler,
    scaler,
    use_scaler: bool,
    cfg_path: str,
    method: str,
    arch_meta: dict,
    modalities: List[str],
    fields: List[str],
) -> None:
    torch.save(
        {
            "iter": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            # `num_updates` pilote l'échauffement de l'EMA (voir EMAModel.current_decay) :
            # sans lui, une reprise repartirait d'une décroissance faible et
            # écraserait la moyenne accumulée.
            "ema_num_updates": ema.num_updates,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if use_scaler else None,
            "cfg_path": str(cfg_path),
            "method": method,
            "arch_meta": arch_meta,
            "modalities": modalities,
            "fields": fields,
        },
        path,
    )


CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)\.pth$")


def _rotate_checkpoints(weights_dir: Path, keep: int) -> List[Path]:
    """Ne conserve que les `keep` checkpoints intermédiaires les plus récents.

    Un checkpoint UNet pèse 2.9 Go ; avec `save_every: 2500` sur 25000
    itérations, un seul entraînement laisse 30 Go derrière lui. Les
    intermédiaires ont une vraie utilité — ils ont permis une reprise à chaud
    après divergence, et de choisir le point de sauvegarde sur une mesure de
    reconstruction plutôt que sur la dernière itération — mais en garder dix
    n'apporte rien de plus que d'en garder deux.

    `model_final.pth` n'est JAMAIS concerné : il ne suit pas le motif
    `checkpoint_<n>.pth`, et c'est lui que l'inférence charge par défaut.

    Le tri se fait sur le NUMÉRO D'ITÉRATION, jamais sur le nom ni sur la date :
    l'ordre lexicographique placerait `checkpoint_10000` avant `checkpoint_2500`
    et supprimerait donc le mauvais fichier ; et une reprise à chaud réécrit des
    dates qui ne reflètent plus l'ordre d'entraînement.

    Retourne la liste des fichiers supprimés.
    """
    if keep <= 0:
        return []
    found = []
    for p in weights_dir.glob("checkpoint_*.pth"):
        m = CHECKPOINT_RE.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    removed = []
    for _, p in sorted(found, key=lambda x: x[0], reverse=True)[keep:]:
        try:
            p.unlink()
            removed.append(p)
        except OSError as e:            # disque plein/lecture seule, course — ne doit PAS tuer le run
            print(f"  ⚠ rotation : impossible de supprimer {p.name} ({e})")
    return removed


# ===========================================================================
# Training
# ===========================================================================


def train(
    cfg_path: str,
    method: str,
    env_path: Optional[str] = None,
    resume: Optional[str] = None,
    resume_weights_only: bool = False,
    field_norm_stats_path: Optional[str] = None,
) -> None:
    arch_module = _resolve_arch_module(method)

    cfg = load_yaml_with_include(cfg_path)
    cfg = resolve_paths(cfg, load_env(env_path))
    if resume is not None:
        cfg["resume"] = resume

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    data_root = data_cfg.get("data_root")
    use_latent_cache = bool(data_cfg.get("use_latent_cache", False))
    # Tuilage de l'encodage : DOIT valoir exactement ce qu'ont utilisé le
    # precompute et l'inférence, sinon les latents d'entraînement ne seraient
    # pas dans la même distribution (voir _encode_like_cache).
    raw_tile = data_cfg.get("encode_tile")
    encode_tile = tuple(int(v) for v in raw_tile) if raw_tile else None
    encode_tile_margin = int(data_cfg.get("encode_tile_margin", 16))
    if data_root is None and not use_latent_cache:
        raise RuntimeError("data_root requis dans la config ou l'env (sauf use_latent_cache=True).")

    modalities: List[str] = data_cfg.get("modalities", MODALITIES)
    fields: List[str] = data_cfg.get("fields", DOMAINS)
    n_fields = len(fields)
    n_classes = len(modalities)  # conditioning class = contrast only; field = time
    n_loader_classes = len(modalities) * n_fields

    output_dir = Path(data_cfg["output_dir"])
    split = data_cfg.get("split", "retro_train")
    p_lo = data_cfg.get("percentile_lower", 0.5)
    p_hi = data_cfg.get("percentile_upper", 99.5)
    max_per_class = data_cfg.get("max_volumes_per_class", None)
    random_crop_prob = float(data_cfg.get("random_crop_prob", 0.0))
    flip_lr_prob = float(data_cfg.get("flip_lr_prob", 0.0))

    raw_vs = data_cfg.get("volume_size", None)
    if raw_vs is None:
        raise RuntimeError("volume_size est requis.")
    volume_size = tuple(int(v) for v in raw_vs)

    raw_ts = data_cfg.get("target_spacing", None)
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else None

    field_norm_stats = None
    if field_norm_stats_path:
        with open(field_norm_stats_path) as _f:
            field_norm_stats = json.load(_f)["stats"]

    total_iters = int(train_cfg.get("total_iters", 10000))
    batch_size = int(train_cfg.get("batch_size", 1))
    num_workers = int(train_cfg.get("num_workers", 4))
    lr = float(train_cfg.get("lr", 1e-4))
    sigma = float(train_cfg.get("sigma", 0.0))
    ot_method = train_cfg.get("ot_method", "exact")  # which FM class (Exact-OT vs plain CFM)
    save_every = int(train_cfg.get("save_every", 2000))
    print_every = int(train_cfg.get("print_every", 100))
    # Nombre de checkpoints intermédiaires conservés (0 = tous, ancien
    # comportement). `model_final.pth` n'est jamais concerné.
    keep_checkpoints = int(train_cfg.get("keep_checkpoints", 2))
    use_amp = bool(train_cfg.get("use_amp", True))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    ema_decay = float(train_cfg.get("ema_decay", 0.9999))
    amp_dtype_name = train_cfg.get("amp_dtype", "bf16")
    num_targets_per_step = int(train_cfg.get("num_targets_per_step", 2))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    accumulation_steps = int(train_cfg.get("accumulation_steps", 1))
    adjacent_only = bool(train_cfg.get("adjacent_only", False))
    identity_prob = float(train_cfg.get("identity_prob", 0.1))

    # Self-supervised anatomy-preservation regularizers (cycle/edge). Disabled
    # by default (lambda=0.0). See cycle_rollout_vector / edge_loss_3d.py.
    lambda_cycle = float(train_cfg.get("lambda_cycle", 0.0))
    lambda_cycle_final = float(train_cfg.get("lambda_cycle_final", lambda_cycle))
    cycle_n_steps = int(train_cfg.get("cycle_n_steps", 2))
    lambda_edge = float(train_cfg.get("lambda_edge", 0.0))
    lambda_edge_final = float(train_cfg.get("lambda_edge_final", lambda_edge))
    lambda_edge_fwd = float(train_cfg.get("lambda_edge_fwd", 0.0))
    lambda_edge_fwd_final = float(train_cfg.get("lambda_edge_fwd_final", lambda_edge_fwd))
    edge_every = int(train_cfg.get("edge_every", 10))
    edge_batch_subset = int(train_cfg.get("edge_batch_subset", 2))

    lambda_cycle_active = lambda_cycle > 0.0 or lambda_cycle_final > 0.0
    lambda_edge_active = lambda_edge > 0.0 or lambda_edge_final > 0.0
    lambda_edge_fwd_active = lambda_edge_fwd > 0.0 or lambda_edge_fwd_final > 0.0

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        (output_dir / "weights").mkdir(parents=True, exist_ok=True)
        print(f"Output dir : {output_dir}")
        print(f"World size : {world_size} | Device : {device} | Méthode : {method}")
        print(f"Modalités  : {modalities}  |  Champs : {fields}  |  Classes (contrastes) : {n_classes}")
        if field_norm_stats_path:
            print(f"Normalisation par champ (fixe) : {field_norm_stats_path}")

    # ── VAE (frozen) — chargé même en mode cache : requis pour latent_shape
    # (métadonnée de checkpoint / build_model) et filet de sécurité si le
    # cache s'avère incomplet.
    vae = load_vae(cfg, device)
    if arch_module.REQUIRES_SPATIAL_VAE and vae.latent_format != "spatial":
        raise RuntimeError(
            f"method='{method}' requiert un VAE spatial (latent_format='spatial'), "
            f"mais '{cfg['vae'].get('vae_type', '?')}' a latent_format='{vae.latent_format}'."
        )
    for _p in vae.parameters():
        _p.requires_grad_(False)
    vae.eval()

    latent_shape = _infer_latent_shape(vae, volume_size, device)
    adapter = arch_module.make_adapter(cfg, latent_shape, n_classes)

    # ── Dataset : latents pré-encodés (cache) OU volumes à encoder à la volée
    if use_latent_cache:
        cache_root = Path(data_cfg.get("latent_cache_root", adapter.default_cache_root))
        cache_dir_override = data_cfg.get("latent_cache_dir")
        if cache_dir_override:
            cache_dir = Path(cache_dir_override)
        else:
            cache_id = flat_latent_cache_id(
                cfg, target_spacing, volume_size, p_lo, p_hi,
                field_norm_stats_path=field_norm_stats_path,
            )
            cache_dir = cache_root / cache_id / split
        ds = adapter.build_cache_dataset(cache_dir, cache_root, data_cfg)
        adapter.validate_cache_shape(ds, latent_shape)
        if is_main_process():
            print(f"  Cache latents : {cache_dir}")
        loader_num_workers = 0  # RAM-resident tensors — worker processes add pure overhead.
    else:
        ds = MultiModalNIfTILatentDataset(
            data_root=Path(data_root),
            split=split,
            modalities=modalities,
            fields=fields,
            percentile_lower=p_lo,
            percentile_upper=p_hi,
            max_per_class=max_per_class,
            target_spacing=target_spacing,
            volume_size=volume_size,
            random_crop_prob=random_crop_prob,
            field_norm_stats=field_norm_stats,
        )
        loader_num_workers = num_workers

    class_loaders = _build_class_loaders(ds, n_loader_classes, batch_size, loader_num_workers, is_distributed)
    available_classes = sorted(class_loaders.keys())
    if len(available_classes) < 2:
        raise RuntimeError("Il faut au moins 2 classes (modalité, champ) non vides.")
    contrast_fields, usable_contrasts = _build_contrast_fields(available_classes, n_fields)

    if is_main_process():
        print(f"  Classes disponibles : {len(available_classes)}/{n_loader_classes}")
        print(f"  Contrastes utilisables : {usable_contrasts} | champs/contraste="
              f"{ {m: contrast_fields[m] for m in usable_contrasts} }")
        print(f"  identity_prob={identity_prob} | adjacent_only={adjacent_only} | "
              f"num_targets_per_step={num_targets_per_step}")

    sobel3d = Sobel3D().to(device) if (lambda_edge_active or lambda_edge_fwd_active) else None

    model = adapter.build_model().to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if is_distributed else model
    model_fn = adapter.make_model_fn(raw_model)

    ema = EMAModel(raw_model, decay=ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        decay_start = total_iters // 2
        if step < decay_start:
            return 1.0
        return max(0.0, 1.0 - (step - decay_start) / max(total_iters - decay_start, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    if amp_dtype_name == "bf16":
        amp_dtype = torch.bfloat16
    elif amp_dtype_name == "fp16":
        amp_dtype = torch.float16
    else:
        raise ValueError("amp_dtype doit être 'bf16' ou 'fp16'.")
    use_scaler = use_amp and device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if ot_method == "exact":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    else:
        FM = ConditionalFlowMatcher(sigma=sigma)

    start_iter = 0
    resume_path = cfg.get("resume")
    if resume_path and Path(resume_path).exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        _check_arch_meta(state, adapter)
        raw_model.load_state_dict(adapter.checkpoint_key_remap(state["model"]))
        if "ema" in state:
            ema.load_state_dict(adapter.checkpoint_key_remap(state["ema"]))
            # checkpoints antérieurs au correctif : pas de compteur, on prend
            # l'itération, qui en est le bon substitut.
            ema.num_updates = int(state.get("ema_num_updates", state.get("iter", 0)))
        if resume_weights_only:
            if is_main_process():
                print(f"Reprise (poids seuls) depuis : {resume_path}")
        else:
            optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state and state["scheduler"] is not None:
                scheduler.load_state_dict(state["scheduler"])
            if "scaler" in state and use_scaler and state["scaler"] is not None:
                scaler.load_state_dict(state["scaler"])
            start_iter = state.get("iter", 0) + 1
            if is_main_process():
                print(f"Reprise depuis iter {start_iter} : {resume_path}")

    if is_main_process():
        n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"{adapter.name} : {n_params / 1e6:.1f}M params | latent_shape={latent_shape}")
        print(
            f"Entraînement : {total_iters} iters | batch={batch_size} | "
            f"targets/step={num_targets_per_step} | lr={lr} | AMP={use_amp} ({amp_dtype_name})"
        )

    weights_dir = output_dir / "weights"
    metrics_path = output_dir / "train_metrics.jsonl"
    t0 = time.time()
    last_log_t = t0
    recent_losses: List[float] = []
    recent_cycle_losses: List[float] = []
    recent_edge_losses: List[float] = []
    recent_edge_fwd_losses: List[float] = []
    model.train()
    grad_norm = 0.0  # persists across accumulation micro-steps for logging

    msg_len = 1 + 2 * num_targets_per_step

    for step in range(start_iter, total_iters):
        if is_distributed:
            if dist.get_rank() == 0:
                contrast, transitions = _sample_step_plan(
                    contrast_fields, usable_contrasts,
                    num_targets_per_step, identity_prob, adjacent_only,
                )
                flat = [contrast]
                for fi, fj in transitions:
                    flat += [fi, fj]
                msg = torch.tensor(flat, dtype=torch.long, device=device)
            else:
                msg = torch.zeros(msg_len, dtype=torch.long, device=device)
            dist.broadcast(msg, src=0)
            contrast = int(msg[0].item())
            transitions = [
                (int(msg[1 + 2 * i].item()), int(msg[2 + 2 * i].item()))
                for i in range(num_targets_per_step)
            ]
        else:
            contrast, transitions = _sample_step_plan(
                contrast_fields, usable_contrasts,
                num_targets_per_step, identity_prob, adjacent_only,
            )

        k = len(transitions)
        step_losses: List[float] = []
        step_cycle_losses: List[float] = []
        step_edge_losses: List[float] = []
        step_edge_fwd_losses: List[float] = []

        cur_lambda_cycle = _lerp_lambda(lambda_cycle, lambda_cycle_final, step, total_iters)
        cur_lambda_edge = _lerp_lambda(lambda_edge, lambda_edge_final, step, total_iters)
        cur_lambda_edge_fwd = _lerp_lambda(lambda_edge_fwd, lambda_edge_fwd_final, step, total_iters)

        for (fi, fj) in transitions:
            same = (fi == fj)
            src_flat = _flat_class(contrast, fi, n_fields)
            tgt_flat = _flat_class(contrast, fj, n_fields)

            src_item = next(class_loaders[src_flat])[0].to(device)
            tgt_item = src_item if same else next(class_loaders[tgt_flat])[0].to(device)

            if use_latent_cache and adapter.cache_prebakes_prep:
                z_src, meta = src_item.float(), None
                z_tgt = z_src if same else tgt_item.float()
            else:
                if use_latent_cache:
                    z_src_enc = src_item.float()
                    z_tgt_enc = z_src_enc if same else tgt_item.float()
                else:
                    with torch.no_grad(), torch.amp.autocast(
                        "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
                    ):
                        z_src_enc = _encode_like_cache(
                            vae, src_item, encode_tile, encode_tile_margin, use_amp, amp_dtype)
                        z_tgt_enc = z_src_enc if same else _encode_like_cache(
                            vae, tgt_item, encode_tile, encode_tile_margin, use_amp, amp_dtype)
                z_src, meta = adapter.prep_latent(vae, z_src_enc)
                z_tgt = z_src if same else adapter.prep_latent(vae, z_tgt_enc)[0]

            t_i = _field_to_time(fi, n_fields)
            t_j = _field_to_time(fj, n_fields)
            t_global, z_t, ut_global = _compute_flow(FM, z_src, z_tgt, t_i, t_j, same, device)

            y_tgt = torch.full((z_src.shape[0],), contrast, dtype=torch.long, device=device)
            t_vec = t_global.float()

            with torch.amp.autocast(
                "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
            ):
                v_t = model_fn(z_t, z_src, t_vec, y_tgt)
                loss = F.l1_loss(v_t, ut_global) / float(k)

            z_src_roundtrip = None
            z_tgt_hat = None
            if lambda_cycle_active and not same:
                z_tgt_hat, z_src_roundtrip = cycle_rollout_vector(
                    model_fn, z_src, y_tgt, t_i, t_j, cycle_n_steps, amp_dtype, use_amp,
                )
                loss_cycle = F.l1_loss(z_src_roundtrip, z_src)
                loss = loss + cur_lambda_cycle * loss_cycle
                step_cycle_losses.append(float(loss_cycle.item()))

            if (
                (lambda_edge_active or lambda_edge_fwd_active)
                and z_src_roundtrip is not None
                and (step % edge_every == 0)
            ):
                n_sub = min(edge_batch_subset, z_src.shape[0])
                with torch.amp.autocast(
                    "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
                ):
                    dec_src = vae.decode(adapter.restore_latent(z_src[:n_sub], meta))
                    if lambda_edge_active:
                        dec_roundtrip = vae.decode(adapter.restore_latent(z_src_roundtrip[:n_sub], meta))
                    if lambda_edge_fwd_active:
                        dec_tgt_hat = vae.decode(adapter.restore_latent(z_tgt_hat[:n_sub], meta))
                if lambda_edge_active:
                    loss_edge = edge_consistency_loss(sobel3d, dec_roundtrip.float(), dec_src.float())
                    loss = loss + cur_lambda_edge * loss_edge
                    step_edge_losses.append(float(loss_edge.item()))
                if lambda_edge_fwd_active:
                    loss_edge_fwd = edge_consistency_loss(sobel3d, dec_tgt_hat.float(), dec_src.float())
                    loss = loss + cur_lambda_edge_fwd * loss_edge_fwd
                    step_edge_fwd_losses.append(float(loss_edge_fwd.item()))

            if use_scaler:
                scaler.scale(loss / accumulation_steps).backward()
            else:
                (loss / accumulation_steps).backward()
            step_losses.append(float(loss.item() * k))

        if (step + 1) % accumulation_steps == 0:
            if use_scaler:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            ema.update(raw_model)

        mean_step_loss = float(np.mean(step_losses)) if step_losses else 0.0
        recent_losses.append(mean_step_loss)
        if len(recent_losses) > print_every:
            recent_losses.pop(0)
        if step_cycle_losses:
            recent_cycle_losses.append(float(np.mean(step_cycle_losses)))
            if len(recent_cycle_losses) > print_every:
                recent_cycle_losses.pop(0)
        if step_edge_losses:
            recent_edge_losses.append(float(np.mean(step_edge_losses)))
            if len(recent_edge_losses) > print_every:
                recent_edge_losses.pop(0)
        if step_edge_fwd_losses:
            recent_edge_fwd_losses.append(float(np.mean(step_edge_fwd_losses)))
            if len(recent_edge_fwd_losses) > print_every:
                recent_edge_fwd_losses.pop(0)

        if is_main_process() and (step + 1) % print_every == 0:
            avg_recent = float(np.mean(recent_losses))
            elapsed = time.time() - t0
            win_dt = time.time() - last_log_t
            it_s = print_every / max(win_dt, 1e-9)
            eta_s = (total_iters - step - 1) / max(it_s, 1e-9)
            lr_cur = scheduler.get_last_lr()[0]
            mem_gb = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                if device.type == "cuda" else 0.0
            )
            extra_log = ""
            if lambda_cycle_active:
                avg_cycle = float(np.mean(recent_cycle_losses)) if recent_cycle_losses else float("nan")
                extra_log += f" cycle={avg_cycle:.4f}(λ={cur_lambda_cycle:.2f})"
            if lambda_edge_active:
                avg_edge = float(np.mean(recent_edge_losses)) if recent_edge_losses else float("nan")
                extra_log += f" edge={avg_edge:.4f}(λ={cur_lambda_edge:.2f})"
            if lambda_edge_fwd_active:
                avg_edge_fwd = float(np.mean(recent_edge_fwd_losses)) if recent_edge_fwd_losses else float("nan")
                extra_log += f" edge_fwd={avg_edge_fwd:.4f}(λ={cur_lambda_edge_fwd:.2f})"
            print(
                f"[{step + 1:6d}/{total_iters}] loss={avg_recent:.4f} grad={float(grad_norm):.2f} "
                f"lr={lr_cur:.2e} contrast={contrast}→fields{transitions} speed={it_s:.2f} it/s "
                f"eta={eta_s / 3600:.2f}h t={elapsed / 60:.1f}min mem={mem_gb:.1f}GB{extra_log}"
            )
            record = {
                "iter": step + 1,
                "loss": round(avg_recent, 6),
                "grad_norm": round(float(grad_norm), 4),
                "lr": round(lr_cur, 8),
                "speed_it_s": round(it_s, 3),
                "elapsed_s": round(elapsed, 1),
                "mem_gb": round(mem_gb, 2),
            }
            if lambda_cycle_active:
                record["loss_cycle"] = round(avg_cycle, 6)
                record["lambda_cycle"] = round(cur_lambda_cycle, 6)
            if lambda_edge_active:
                record["loss_edge"] = round(avg_edge, 6)
                record["lambda_edge"] = round(cur_lambda_edge, 6)
            if lambda_edge_fwd_active:
                record["loss_edge_fwd"] = round(avg_edge_fwd, 6)
                record["lambda_edge_fwd"] = round(cur_lambda_edge_fwd, 6)
            with open(metrics_path, "a") as _mf:
                _mf.write(json.dumps(record) + "\n")
            last_log_t = time.time()

        if is_main_process() and (step + 1) % save_every == 0:
            ckpt_path = weights_dir / f"checkpoint_{step + 1}.pth"
            _save_checkpoint(
                ckpt_path, step, raw_model, ema, optimizer, scheduler, scaler, use_scaler,
                cfg_path, method, adapter.arch_meta_dict(), modalities, fields,
            )
            print(f"  → Checkpoint : {ckpt_path}")
            # Rotation APRÈS écriture réussie : si `_save_checkpoint` avait
            # échoué, purger d'abord aurait détruit les anciens sans avoir le
            # nouveau. `keep_checkpoints: 0` désactive la rotation.
            for _p in _rotate_checkpoints(weights_dir, keep_checkpoints):
                print(f"    rotation : {_p.name} supprimé")

    if is_main_process():
        final_path = weights_dir / "model_final.pth"
        _save_checkpoint(
            final_path, total_iters - 1, raw_model, ema, optimizer, scheduler, scaler, use_scaler,
            cfg_path, method, adapter.arch_meta_dict(), modalities, fields,
        )
        print(f"\nEntraînement terminé. Modèle final : {final_path}")

    if is_distributed:
        dist.destroy_process_group()


# ===========================================================================
# Inference (single-shot, direct-latent — no patching/blending; see
# infer_mmfm_unified.py for the patch-based full-resolution pipeline)
# ===========================================================================


def infer(
    cfg_path: str,
    method: str,
    checkpoint: str,
    output_dir: str,
    source_field: str,
    source_modality: str,
    target_field: str,
    target_modality: str,
    env_path: Optional[str] = None,
    input_dir: Optional[str] = None,
    input_volume: Optional[str] = None,
    n_steps: Optional[int] = None,
    use_ema: bool = True,
    field_norm_stats_path: Optional[str] = None,
) -> None:
    arch_module = _resolve_arch_module(method)

    cfg = load_yaml_with_include(cfg_path)
    cfg = resolve_paths(cfg, load_env(env_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    amp_dtype_name = train_cfg.get("amp_dtype", "bf16")
    use_amp = bool(train_cfg.get("use_amp", True))
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16

    n_steps = n_steps or cfg.get("inference", {}).get("n_steps", 20)
    p_lo = data_cfg.get("percentile_lower", 0.5)
    p_hi = data_cfg.get("percentile_upper", 99.5)

    field_norm_stats = None
    if field_norm_stats_path:
        with open(field_norm_stats_path) as _f:
            field_norm_stats = json.load(_f)["stats"]
        print(f"  Normalisation par champ (fixe) : {field_norm_stats_path}")

    raw_vs = data_cfg.get("volume_size", None)
    volume_size = tuple(int(v) for v in raw_vs) if raw_vs else None
    raw_ts = data_cfg.get("target_spacing", None)
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else None

    modalities: List[str] = data_cfg.get("modalities", MODALITIES)
    fields: List[str] = data_cfg.get("fields", DOMAINS)
    n_fields = len(fields)
    n_classes = len(modalities)

    for name, val, lst in [
        ("source_field", source_field, fields),
        ("target_field", target_field, fields),
        ("source_modality", source_modality, modalities),
        ("target_modality", target_modality, modalities),
    ]:
        if val not in lst:
            raise ValueError(f"{name}='{val}' non présent dans la config ({lst})")

    if source_modality != target_modality:
        raise ValueError(
            "MMFM multi-marginal : le contraste est invariant le long de la "
            f"trajectoire de champ. source_modality={source_modality} doit égaler "
            f"target_modality={target_modality}. (Task 3 = translation de champ.)"
        )

    contrast = modalities.index(source_modality)
    src_field_idx = fields.index(source_field)
    tgt_field_idx = fields.index(target_field)
    t_start = _field_to_time(src_field_idx, n_fields)
    t_end = _field_to_time(tgt_field_idx, n_fields)

    print(
        f"Inférence MMFM ({method}) : {source_modality}@{source_field} → "
        f"{target_modality}@{target_field}  |  contrast={contrast}  |  "
        f"t:{t_start:.3f}→{t_end:.3f}  |  {n_steps} steps Euler"
    )

    vae = load_vae(cfg, device)
    if arch_module.REQUIRES_SPATIAL_VAE and vae.latent_format != "spatial":
        raise RuntimeError(f"Requiert un VAE spatial, got latent_format='{vae.latent_format}'.")
    latent_shape = _infer_latent_shape(vae, volume_size, device)
    adapter = arch_module.make_adapter(cfg, latent_shape, n_classes)
    print(f"  VAE : latent_shape={latent_shape}")

    model = adapter.build_model().to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    _check_arch_meta(ckpt, adapter)

    loaded_from = "model"
    if use_ema and "ema" in ckpt and ckpt["ema"]:
        ema_state = ckpt["ema"]
        shadow = ema_state.get("shadow_params", None) if isinstance(ema_state, dict) else None
        if shadow is not None:
            model.load_state_dict(adapter.checkpoint_key_remap(shadow))
            loaded_from = "ema.shadow_params"
        elif isinstance(ema_state, dict) and ema_state:
            model.load_state_dict(adapter.checkpoint_key_remap(ema_state))
            loaded_from = "ema"
        else:
            model.load_state_dict(adapter.checkpoint_key_remap(ckpt["model"]))
    else:
        model.load_state_dict(adapter.checkpoint_key_remap(ckpt["model"]))

    model.eval()
    model_fn = adapter.make_model_fn(model)
    trained_at = ckpt.get("iter", "?")
    print(f"  Modèle chargé (clé: {loaded_from}, iter={trained_at}) depuis {checkpoint}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_volume is not None:
        input_files = [Path(input_volume)]
    elif input_dir is not None:
        input_files = sorted(Path(input_dir).glob("*.nii.gz"))
        if not input_files:
            raise FileNotFoundError(f"Aucun fichier .nii.gz dans {input_dir}")
    else:
        raise ValueError("Fournir --input_dir ou --input_volume.")

    print(f"  {len(input_files)} volume(s) → {out_dir}")

    src_fixed_lo = src_fixed_hi = None
    tgt_fixed_lo = tgt_fixed_hi = None
    if field_norm_stats is not None:
        src_entry = field_norm_stats.get(source_modality, {}).get(source_field)
        tgt_entry = field_norm_stats.get(target_modality, {}).get(target_field)
        if src_entry is None or tgt_entry is None:
            raise RuntimeError(
                f"field_norm_stats fourni mais entrée manquante pour "
                f"{source_modality}@{source_field} ou {target_modality}@{target_field}"
            )
        src_fixed_lo, src_fixed_hi = src_entry["lo"], src_entry["hi"]
        tgt_fixed_lo, tgt_fixed_hi = tgt_entry["lo"], tgt_entry["hi"]

    for nii_path in input_files:
        t_start_time = time.time()

        vol, _ = load_nifti_volume(
            nii_path,
            target_spacing=target_spacing,
            volume_size=volume_size,
            normalize=True,
            lo_pct=p_lo,
            hi_pct=p_hi,
            fixed_lo=src_fixed_lo,
            fixed_hi=src_fixed_hi,
        )

        img_nib = nib.load(str(nii_path))
        orig_spacing = np.abs(np.diag(img_nib.affine)[:3])
        orig_shape = np.array(img_nib.shape[:3])
        if target_spacing is not None:
            resampled_arr = resample_volume(
                np.zeros(orig_shape.tolist(), dtype=np.float32), orig_spacing, target_spacing,
            )
            resampled_shape = resampled_arr.shape
        else:
            resampled_shape = None

        out_affine = adjust_affine_for_crop_pad(
            img_nib.affine.copy().astype(float),
            orig_shape, volume_size, resampled_shape, target_spacing, orig_spacing,
        )

        vol_tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
        ):
            z_src_enc = vae.encode(vol_tensor)
        z_src, meta = adapter.prep_latent(vae, z_src_enc)
        z_src = z_src.float()

        y = torch.tensor([contrast], dtype=torch.long, device=device)
        z_tgt = euler_integrate(
            model_fn, z_src, y, t_start, t_end, n_steps, device, use_amp, amp_dtype,
        )
        z_tgt = adapter.restore_latent(z_tgt, meta)

        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
        ):
            recon = vae.decode(z_tgt)

        pred_vol = recon.squeeze().cpu().float().numpy()
        pred_vol = (np.clip(pred_vol, -1.0, 1.0) + 1.0) / 2.0  # [-1,1] -> [0,1]

        if field_norm_stats is not None:
            pred_vol = denormalize_from_01(pred_vol, tgt_fixed_lo, tgt_fixed_hi)

        brain_mask = (vol > -0.99).astype(np.float32)
        pred_vol = pred_vol * brain_mask

        stem = nii_path.name.replace(".nii.gz", "")
        out_name = f"{stem}_{target_modality}_{target_field}_{adapter.name}.nii.gz"
        out_path = out_dir / out_name
        nib.save(nib.Nifti1Image(pred_vol, out_affine), str(out_path))

        elapsed = time.time() - t_start_time
        print(f"  {nii_path.name} → {out_name}  ({elapsed:.1f}s)")

    print(f"\nInférence terminée. Prédictions dans : {out_dir}")
