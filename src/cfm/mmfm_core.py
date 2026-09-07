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
    elif method == "mmfm3d_synthetic":
        # Harnais de validation : probleme a 5 marginales dont la reponse est
        # connue analytiquement (cfm/synthetic_marginals.py). Branche ICI pour
        # que le harnais exerce la VRAIE boucle d'entrainement plutot qu'une
        # copie qui pourrait diverger du code de production.
        from cfm import arch_synthetic as arch_module
    else:
        raise ValueError(
            f"method='{method}' inconnu — attendu 'mmfm3d_vectorized', "
            f"'mmfm3d_unet', 'mmfm3d_inr' ou 'mmfm3d_synthetic'."
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


# ===========================================================================
# Flow multi-marginal RÉEL — couplage OT chaîné + spline à travers les marginales
#
# Ce bloc remplace `_compute_flow` (droites indépendantes entre 2 marginales,
# recousues dans un temps global) par l'algorithme de la référence que ce projet
# prétend implémenter, Genentech/MMFM :
#
#   1. external/MMFM/src/mmfm/data.py:123 `couple_samples_no_int` — couplages OT
#      CHAÎNÉS entre marginales consécutives (`ot.da.EMDTransport`), calculés sur
#      TOUT le jeu de données, puis composés (`apply_couplings_no_int`) pour
#      transformer des marginales non appariées en TUPLES COUPLÉS à travers tous
#      les temps. C'est exactement notre situation : 0 sujet apparié dans
#      `retro_train`, des ensembles de sujets différents par champ.
#   2. external/MMFM/src/mmfm/multi_marginal_fm.py:334 — une SPLINE CUBIQUE à
#      travers les K marginales du tuple, dont la dérivée `P(t, 1)` est la cible
#      de vitesse. Une seule courbe lisse, pas des segments contradictoires.
#
# Pourquoi c'était fatal. Avec deux DataLoader indépendants (l'ancien chemin),
# z_tgt est indépendant de z_src, donc E[z_tgt | z_src] = E[z_tgt] : la vitesse
# optimale de Bayes ne dépend PAS du sujet, et le réseau converge vers une
# constante. Mesuré sur le checkpoint de production le 2026-09-04 :
# cos(v(t=0), v(t=1)) = 1.00000000, cos(v(sujet A), v(sujet B)) = 0.99997336,
# les 4 déplacements cibles colinéaires à 1.000000 avec des normes en rapport
# exact 1:2:3:4. Le modèle n'était pas cassé : il avait convergé vers l'optimum
# d'un objectif qui n'avait qu'une constante à offrir.
# ===========================================================================


_SPLINE_CACHE: Dict[Tuple[Tuple[float, ...], str], Any] = {}


def _spline_weights(
    t_anchor: Tuple[float, ...],
    t_query: np.ndarray,
    derivative: int,
) -> np.ndarray:
    """Poids (T, K) tels que `P(t) = W @ Z` et `P'(t) = W' @ Z`.

    Les temps d'ancrage sont FIXES (t_k = k/(K-1), les 5 champs), donc la spline
    cubique de scipy est un opérateur LINÉAIRE fixe sur les K valeurs d'ancrage.
    On l'obtient exactement en ajustant `interpolate.CubicSpline` sur la base
    canonique une fois pour toutes, ce qui évite de faire un aller-retour
    CPU/numpy sur un latent de 129 024 dimensions à chaque pas — tout en gardant
    la sémantique de scipy au bit près (mêmes conditions de bord « not-a-knot »
    que la référence, qui appelle `CubicSpline` directement).

    Vérifié : `W @ Z` égale `CubicSpline(t_anchor, Z)(t)` à 6e-16 près pour
    nu=0 et 7e-15 pour nu=1 ; les poids somment à 1 (nu=0) et 0 (nu=1) ; et
    `P(t_k) = z_k` exactement.
    """
    from scipy import interpolate  # local: scipy n'est pas requis à l'inférence

    key = (t_anchor, f"nu{derivative}")
    spline = _SPLINE_CACHE.get(key)
    if spline is None:
        spline = interpolate.CubicSpline(
            np.asarray(t_anchor, dtype=np.float64),
            np.eye(len(t_anchor), dtype=np.float64),
            axis=0,
        )
        _SPLINE_CACHE[key] = spline
    return np.asarray(spline(np.asarray(t_query, dtype=np.float64), nu=derivative))


def build_chained_ot_trajectories(
    ds,
    contrast_fields: Dict[int, List[int]],
    usable_contrasts: List[int],
    n_fields: int,
    verbose: bool = True,
) -> Dict[int, np.ndarray]:
    """Couplages OT chaînés : construit des TRAJECTOIRES à partir de marginales
    non appariées.

    Pour chaque contraste, on résout un transport optimal entre les latents du
    champ k et ceux du champ k+1, puis on chaîne les affectations. Le résultat
    est un tableau `(n_traj, K)` d'INDICES dans `ds` : la ligne n donne, pour
    chaque champ, le sujet que l'OT a apparié au sujet n du premier champ.

    Différence assumée avec la référence : `couple_samples_no_int` lève
    `NotImplementedError` sur une matrice de couplage non carrée. Nos classes
    sont très déséquilibrées (43 à 235 volumes selon la cellule), donc on
    généralise en prenant, pour chaque source, l'argmax de sa ligne du plan de
    transport — l'affectation barycentrique dégénérée, qui coïncide avec la
    permutation de la référence dans le cas carré.

    Coût : la matrice de coût est n_src x n_tgt en dimension D. Pour la plus
    grosse cellule (235 x 235 en 129 024 dimensions) c'est ~7 GFLOP, quelques
    secondes, une seule fois avant l'entraînement.
    """
    import ot as pot  # local: seule cette fonction en dépend

    by_class: Dict[int, List[int]] = {}
    for i in range(len(ds.samples)):
        by_class.setdefault(_sample_class(ds, i), []).append(i)

    def _latents(indices: List[int]) -> np.ndarray:
        rows = [np.asarray(ds[i][0], dtype=np.float64).reshape(-1) for i in indices]
        return np.stack(rows, axis=0)

    trajectories: Dict[int, np.ndarray] = {}
    for contrast in usable_contrasts:
        fields = contrast_fields[contrast]
        idx_per_field = [by_class[_flat_class(contrast, f, n_fields)] for f in fields]
        # La trajectoire est indexée par les sujets du PREMIER champ disponible.
        traj = np.zeros((len(idx_per_field[0]), len(fields)), dtype=np.int64)
        traj[:, 0] = np.asarray(idx_per_field[0], dtype=np.int64)
        cur = _latents(idx_per_field[0])
        for k in range(len(fields) - 1):
            nxt_idx = idx_per_field[k + 1]
            nxt = _latents(nxt_idx)
            cost = pot.dist(cur, nxt)  # coût quadratique, comme pot.da.EMDTransport
            a = np.full(cost.shape[0], 1.0 / cost.shape[0])
            b = np.full(cost.shape[1], 1.0 / cost.shape[1])
            plan = pot.emd(a, b, cost, numItermax=1_000_000)
            assign = plan.argmax(axis=1)
            traj[:, k + 1] = np.asarray(nxt_idx, dtype=np.int64)[assign]
            # On avance en gardant l'ALIGNEMENT : la ligne n représente toujours
            # la même trajectoire, ce que fait `apply_couplings_no_int` en
            # composant les couplages.
            cur = nxt[assign]
        trajectories[contrast] = traj
        if verbose and is_main_process():
            uniq = [len(np.unique(traj[:, k])) for k in range(traj.shape[1])]
            print(
                f"  couplage OT chaîné, contraste {contrast} : {traj.shape[0]} trajectoires "
                f"sur champs {fields} | sujets distincts par champ = {uniq}"
            )
    return trajectories


def _compute_flow_trajectory(
    Z: Tensor,
    t_anchor: Tuple[float, ...],
    anchor_pos: int,
    sigma: float,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Construit (t, z_t, u_t, z_anchor) le long de la spline multi-marginale.

    Args:
        Z: (B, K, *rest) — les K marginales couplées de chaque trajectoire.
            `rest` est quelconque : plat (D,) pour le cache vectorisé/INR,
            spatial (C, H, W, D) pour le cache UNet. La spline est un opérateur
            linéaire appliqué le long de l'axe K, donc élément par élément sur
            `rest` — aucune hypothèse de forme.
        t_anchor: temps des K marginales (fixes).
        anchor_pos: index de la marginale servant d'ancre de conditionnement.
            À l'inférence, `euler_integrate` conditionne sur le volume SOURCE
            pendant toute la trajectoire ; l'entraînement doit donc voir la même
            convention, avec une ancre tirée uniformément parmi les K champs
            pour couvrir toutes les inférences possibles (montantes et
            descendantes).
        sigma: écart-type du chemin conditionnel. 0 = chemin déterministe ; la
            référence n'ajoute alors aucun terme de dérivée
            (`compute_conditional_flow`, cas `isinstance(self.sigma, float)`).

    Returns:
        (t, z_t, u_t, z_anchor) — t de forme (B,), les autres (B, *rest).
    """
    B, K = Z.shape[0], Z.shape[1]
    if K != len(t_anchor):
        raise RuntimeError(
            f"Z porte {K} marginales mais t_anchor en déclare {len(t_anchor)}."
        )
    t = torch.rand(B, device=device)
    tq = t.detach().cpu().numpy()
    w0 = torch.as_tensor(_spline_weights(t_anchor, tq, 0), dtype=Z.dtype, device=device)
    w1 = torch.as_tensor(_spline_weights(t_anchor, tq, 1), dtype=Z.dtype, device=device)
    # Contraction le long de l'axe des marginales, quelle que soit la forme de
    # queue : (B, K) x (B, K, *rest) -> (B, *rest).
    flat = Z.reshape(B, K, -1)
    z_t = torch.bmm(w0.unsqueeze(1), flat).reshape(B, *Z.shape[2:])
    u_t = torch.bmm(w1.unsqueeze(1), flat).reshape(B, *Z.shape[2:])
    if sigma > 0.0:
        # Variance constante : sa dérivée est nulle, donc u_t est inchangé — même
        # traitement que la référence.
        z_t = z_t + sigma * torch.randn_like(z_t)
    return t, z_t, u_t, Z[:, anchor_pos]


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
    flip_axis = int(data_cfg.get("flip_axis", 0))

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

    # ── PERTE. Le théorème du flow matching identifie le champ de vitesse
    # marginal à l'ESPÉRANCE conditionnelle E[u_t | x_t] : c'est une régression
    # quadratique qui la retrouve, une régression L1 retrouve la MÉDIANE
    # conditionnelle. Les trois références utilisent une erreur quadratique
    # (torchcfm, external/MMFM/src/mmfm/multi_marginal_fm.py,
    # facebookresearch/flow_matching). Ce pipeline utilisait `l1` depuis
    # l'origine. Le défaut est désormais `mse` : changer la perte n'affecte
    # AUCUN checkpoint existant (elle n'intervient qu'à l'entraînement), donc
    # aucun chiffre publié ne devient irreproductible — seul un
    # ré-entraînement diffère, ce qui est l'intention.
    loss_kind = str(train_cfg.get("loss", "mse")).lower()
    if loss_kind not in ("mse", "l1"):
        raise ValueError(f"train.loss doit valoir 'mse' ou 'l1', pas {loss_kind!r}")
    loss_fn = F.mse_loss if loss_kind == "mse" else F.l1_loss

    # ── MODE MARGINAL.
    #   "trajectory" (défaut) : couplage OT chaîné hors boucle + spline cubique à
    #     travers les K marginales — l'algorithme de Genentech/MMFM (voir
    #     build_chained_ot_trajectories / _compute_flow_trajectory).
    #   "pairwise" : l'ancien chemin, deux DataLoader indépendants et des droites
    #     entre 2 marginales recousues dans le temps global. Conservé pour
    #     rejouer un run historique, jamais pour en produire un nouveau : dans ce
    #     régime la vitesse optimale ne dépend pas du sujet (mesuré le
    #     2026-09-04 : cos(v(sujet A), v(sujet B)) = 0.99997336 sur le
    #     checkpoint de production).
    marginal_mode = str(train_cfg.get("marginal_mode", "trajectory")).lower()
    if marginal_mode not in ("trajectory", "pairwise"):
        raise ValueError(
            f"train.marginal_mode doit valoir 'trajectory' ou 'pairwise', pas {marginal_mode!r}"
        )

    # ── FLIP. `FlatLatentCacheDataset.__getitem__` tire son propre flip, et le
    # loop appelait le dataset DEUX fois par pas (source puis cible) : avec
    # flip_lr_prob=0.5, une transition sur deux demandait au flow de transformer
    # un cerveau EN SON IMAGE MIROIR. `flip_per_step` retire le tirage au
    # dataset et l'applique une seule fois par pas, à toutes les marginales de
    # la trajectoire.
    flip_per_step = bool(train_cfg.get("flip_per_step", True))

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

    if marginal_mode == "trajectory" and (
        lambda_cycle_active or lambda_edge_active or lambda_edge_fwd_active
    ):
        # Les régularisateurs cycle/edge sont écrits autour d'un aller-retour
        # entre DEUX marginales (t_i -> t_j -> t_i). En mode trajectoire il n'y a
        # plus de couple (t_i, t_j) : il faudrait redéfinir ce qu'ils
        # régularisent. Refuser explicitement plutôt que de leur donner
        # silencieusement des bornes qui ne veulent rien dire.
        raise ValueError(
            "lambda_cycle / lambda_edge / lambda_edge_fwd ne sont pas définis en "
            "marginal_mode='trajectory' (ils supposent un aller-retour entre deux "
            "marginales). Les mettre à 0.0, ou repasser en marginal_mode='pairwise'."
        )

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
                encode_tile=encode_tile, encode_tile_margin=encode_tile_margin,
                amp_dtype=amp_dtype_name,
            )
            cache_dir = cache_root / cache_id / split
        ds_cfg = dict(data_cfg)
        if flip_per_step and flip_lr_prob > 0.0:
            # Le flip est retiré au dataset et appliqué une seule fois par pas
            # dans la boucle, à toutes les marginales à la fois.
            ds_cfg["flip_lr_prob"] = 0.0
        ds = adapter.build_cache_dataset(cache_dir, cache_root, ds_cfg)
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
        print(f"  marginal_mode={marginal_mode} | loss={loss_kind} | "
              f"flip_per_step={flip_per_step} (flip_lr_prob={flip_lr_prob})")

    # ── Couplage OT chaîné : une seule fois, avant la boucle.
    trajectories: Optional[Dict[int, np.ndarray]] = None
    traj_t_anchor: Dict[int, Tuple[float, ...]] = {}
    if marginal_mode == "trajectory":
        if is_main_process():
            print("  Construction des trajectoires par couplage OT chaîné "
                  "(Genentech/MMFM, data.py:123) …", flush=True)
        t0_couple = time.time()
        trajectories = build_chained_ot_trajectories(
            ds, contrast_fields, usable_contrasts, n_fields,
        )
        for c in usable_contrasts:
            traj_t_anchor[c] = tuple(
                _field_to_time(f, n_fields) for f in contrast_fields[c]
            )
        if is_main_process():
            print(f"  Couplage terminé en {time.time() - t0_couple:.1f}s | "
                  f"temps d'ancrage par contraste = {traj_t_anchor}", flush=True)
            for c in usable_contrasts:
                if len(contrast_fields[c]) < 3:
                    print(f"  ATTENTION contraste {c} : seulement "
                          f"{len(contrast_fields[c])} marginales — la spline cubique "
                          "dégénère vers une interpolation de bas degré.")

    flat_latent_numel = int(np.prod(latent_shape))

    def _flip_batch(z: Tensor) -> Tensor:
        """Miroir gauche-droite, appliqué à TOUTES les marginales d'un même pas.

        Accepte indifféremment un latent déjà spatial (dims de queue égales à
        `latent_shape`, cache UNet) ou aplati (…, prod(latent_shape), cache
        vectorisé) : les dimensions de tête, quel que soit leur nombre (lot,
        marginales), sont préservées. `flip_axis` est un axe IMAGE, d'où le +1
        qui saute la dimension de canaux — même convention que
        `FlatLatentCacheDataset` / `LatentCacheDataset`.
        """
        k = len(latent_shape)
        if z.shape[-k:] == tuple(latent_shape):
            return torch.flip(z, dims=[z.ndim - k + 1 + flip_axis])
        if z.shape[-1] != flat_latent_numel:
            raise RuntimeError(
                f"Forme de latent inattendue pour le flip : {tuple(z.shape)}, "
                f"attendu des dims de queue {tuple(latent_shape)} ou "
                f"(..., {flat_latent_numel})."
            )
        lead = z.shape[:-1]
        v = z.reshape(*lead, *latent_shape)
        v = torch.flip(v, dims=[len(lead) + 1 + flip_axis])
        return v.reshape(*lead, -1)

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
        # RNG de pas, déterministe et DISTINCT par rang : en mode trajectoire il
        # remplace le broadcast (chaque rang tire son propre lot, DDP moyenne les
        # gradients), tout en restant reproductible à partir de `step`.
        step_rng = np.random.default_rng(step * max(world_size, 1) + local_rank)

        if marginal_mode == "trajectory":
            # La trajectoire traverse tous les champs disponibles du contraste :
            # il n'y a qu'un « plan » à tirer, le contraste.
            contrast = int(step_rng.choice(usable_contrasts))
            transitions = [(0, 0)]
        elif is_distributed:
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
            if marginal_mode == "trajectory":
                # ── Chemin multi-marginal RÉEL : une trajectoire couplée par OT,
                # une spline cubique à travers ses K marginales, sa dérivée pour
                # cible. Ni `transitions`, ni `identity_prob`, ni `adjacent_only`
                # n'ont de sens ici : la trajectoire traverse TOUS les champs.
                traj = trajectories[contrast]
                rows = step_rng.integers(0, traj.shape[0], size=batch_size)
                idx = traj[rows]  # (B, K)
                Z = torch.stack([
                    torch.stack([ds[int(j)][0] for j in row]) for row in idx
                ]).to(device).float()  # (B, K, *rest)

                if flip_per_step and flip_lr_prob > 0.0 and random.random() < flip_lr_prob:
                    # UN SEUL tirage, appliqué aux K marginales : les
                    # correspondances de la trajectoire restent cohérentes.
                    Z = _flip_batch(Z)

                meta = None
                if not (use_latent_cache and adapter.cache_prebakes_prep):
                    b, kk = Z.shape[0], Z.shape[1]
                    prepped, meta = adapter.prep_latent(vae, Z.reshape(b * kk, *Z.shape[2:]))
                    Z = prepped.reshape(b, kk, *prepped.shape[1:])

                t_anchor = traj_t_anchor[contrast]
                anchor_pos = int(step_rng.integers(0, len(t_anchor)))
                t_global, z_t, ut_global, z_src = _compute_flow_trajectory(
                    Z, t_anchor, anchor_pos, sigma, device,
                )
                same = False
                t_i, t_j = t_anchor[anchor_pos], t_anchor[anchor_pos]
            else:
                same = (fi == fj)
                src_flat = _flat_class(contrast, fi, n_fields)
                tgt_flat = _flat_class(contrast, fj, n_fields)

                src_item = next(class_loaders[src_flat])[0].to(device)
                tgt_item = src_item if same else next(class_loaders[tgt_flat])[0].to(device)

                if flip_per_step and flip_lr_prob > 0.0 and random.random() < flip_lr_prob:
                    # Le MÊME tirage pour la source et la cible. Deux tirages
                    # indépendants (l'ancien comportement, hérité de deux appels
                    # séparés à `__getitem__`) demandaient au flow, une fois sur
                    # deux, de transformer un cerveau en son image miroir.
                    src_item = _flip_batch(src_item)
                    tgt_item = src_item if same else _flip_batch(tgt_item)

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
                loss = loss_fn(v_t, ut_global) / float(k)

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
                # 4 CHIFFRES SIGNIFICATIFS, pas 4 decimales : avec round(x, 4)
                # toute norme < 5e-5 s'ecrivait "0.0". Le flow INR (latent a
                # l'echelle 2.6e-4, donc gradients ~1e-4) apparaissait ainsi avec
                # 103 gradients "nuls" sur 250 points, ce qui a ete pris pour un
                # entrainement mort le 2026-08-27 alors que le gradient etait sain.
                "grad_norm": float(f"{float(grad_norm):.4g}"),
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
