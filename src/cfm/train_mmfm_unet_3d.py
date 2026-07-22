#!/usr/bin/env python3
"""MMFM-UNet 3D — Any-to-any multimodal avec UNet 3D spatial.

Extension de MMFM-MLP (train_mmfm_3d.py) qui remplace le MLP vectoriel par un
UNet 3D spatial (MONAI DiffusionModelUNet), similaire à train_cfm_3d.py mais
étendu au cadre multimodal any-to-any (15 classes = 3 mod × 5 champs).

Différences clés vs MMFM-MLP (train_mmfm_3d.py) :
  - Architecture : DiffusionModelUNet 3D (pas de flatten du latent)
  - Input : cat(z_t, z_src) → (B, 2*C_lat, H', W', D')  [comme CFM]
  - Output : champ vectoriel spatial (B, C_lat, H', W', D')
  - Conditioning : AdaGN à chaque résolution (profond), num_class_embeds configurable
  - Multi-marginal : num_class_embeds = 3 contrastes, le temps = champ magnétique
  - Pas de LatentVectorizer — tout reste spatial

Différences clés vs CFM 3D (train_cfm_3d.py) :
  - Multimodal : 15 loaders (1/classe = 1/(modalité, champ))
  - Boucle : num_targets_per_step paires source→cible par step (comme MMFM-MLP)
  - build_unet_3d() avec num_class_embeds=15 (vs 5 dans CFM)
  - Synchronisation DDP : src_class + tgt_classes (vecteur de longueur variable)

Pipeline :
  1. NIfTI → normalize → crop/pad → (B, 1, H, W, D)
  2. VAE encode (frozen) → z  (B, C_lat, H', W', D')
  3. OT-CFM : (z_src, z_tgt) → (t, z_t, ut) spatial
  4. UNet(cat(z_t, z_src), t, class_cible) → v_t spatial
  5. loss = MSE(v_t, ut)
  6. [infer] Euler spatial → z_tgt → VAE decode → NIfTI

Usage :
  # Entraînement local (single-GPU)
  PYTHONPATH=src python src/cfm/train_mmfm_unet_3d.py \\
      --config configs/mmfm3d_unet_medvae_multimodal.yaml --env local

  # Entraînement multi-GPU (DGX local)
  bash src/slurm/launch_cfm3d_dgx.sh mmfm_unet T1W 4 \\
      configs/mmfm3d_unet_medvae_multimodal.yaml

  # Jean Zay (4×H100)
  sbatch src/slurm/cfm_3d_jeanzay.slurm mmfm_unet T1W \\
      configs/mmfm3d_unet_medvae_multimodal.yaml

  # Inférence
  PYTHONPATH=src python src/cfm/train_mmfm_unet_3d.py --mode infer \\
      --config configs/mmfm3d_unet_medvae_multimodal.yaml --env local \\
      --checkpoint outputs/cfm3d/runs/mmfm3d_unet_medvae_multimodal/weights/model_final.pth \\
      --input_volume /data/T1W/0.1T/sub_0001.nii.gz \\
      --output_dir outputs/predictions/mmfm_unet/ \\
      --source_field 0.1T --source_modality T1W \\
      --target_field 7T   --target_modality T1W
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config import load_yaml_with_include, load_env, resolve_paths
from common.dataset import MultiModalNIfTILatentDataset, LatentCacheDataset
from common.distributed import is_main_process, EMAModel
from common.io import (
    DOMAINS,
    MODALITIES,
    adjust_affine_for_crop_pad,
    load_nifti_volume,
    resample_volume,
)
from models.vae_loader import load_vae
from models.factorized_attention_3d import FactorizedAttention3D

try:
    from monai.networks.nets import DiffusionModelUNet
except ImportError:
    try:
        from monai.generative.networks.nets import DiffusionModelUNet
    except ImportError:
        from generative.networks.nets import DiffusionModelUNet

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _flat_class(mod_idx: int, field_idx: int, n_fields: int) -> int:
    """Index de classe unique : mod_idx * n_fields + field_idx.

    NB : conservé pour rétro-compatibilité (loaders par (modalité, champ)).
    Dans la formulation multi-marginal, la CLASSE conditionnante du réseau est
    le CONTRASTE (mod_idx), et le CHAMP magnétique (field_idx) est mappé sur le
    TEMPS via `_field_to_time`.
    """
    return mod_idx * n_fields + field_idx


def _unflat_class(flat: int, n_fields: int) -> Tuple[int, int]:
    """Inverse de _flat_class : flat -> (mod_idx, field_idx)."""
    return flat // n_fields, flat % n_fields


def _field_to_time(field_idx: int, n_fields: int) -> float:
    """Mappe un index de champ magnétique ordonné sur un temps uniforme [0,1].

    Ex. 5 champs (0.1T,1.5T,3T,5T,7T) -> t = 0, 0.25, 0.5, 0.75, 1.0
    """
    if n_fields <= 1:
        return 0.0
    return field_idx / (n_fields - 1)


def _make_infinite(loader: DataLoader):
    while True:
        try:
            yield from loader
        except StopIteration:
            pass  # epoch fini, on relance
        except Exception as e:
            print(f"[DEBUG] _make_infinite: loader raised {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            # ne pas relancer — redémarrer le loader à l'epoch suivante
            continue


def _pad_to_multiple(z: torch.Tensor, mult: int = 4):
    """Pad zéro les dims spatiales d'un tenseur (B, C, H, W, D) vers un multiple.

    Retourne (z_padded, orig_spatial_shape) pour pouvoir cropper la sortie.
    """
    h, w, d = z.shape[-3:]
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    pd = (mult - d % mult) % mult
    if ph or pw or pd:
        # F.pad ordre : (d_l,d_r, w_l,w_r, h_l,h_r)
        z = F.pad(z, (0, pd, 0, pw, 0, ph), mode="constant", value=0.0)
    return z, (h, w, d)


def _crop_to_shape(z: torch.Tensor, shape) -> torch.Tensor:
    """Crop les dims spatiales d'un (B, C, H, W, D) vers `shape` (H,W,D)."""
    h, w, d = shape
    return z[..., :h, :w, :d]


# ===========================================================================
# Compatibilité MONAI : remapping des clés d'attention
# ===========================================================================


def _remap_monai_attention_keys(state_dict: dict) -> dict:
    """Remappe les clés d'attention de l'ancienne API MONAI (≤1.3) vers la nouvelle (≥1.4).

    Ancienne (Jean Zay pytorch-gpu/py3/2.5.0) :
        <prefix>.to_q / to_k / to_v / proj_attn
    Nouvelle (MONAI ≥1.4, local) :
        <prefix>.attn.to_q / attn.to_k / attn.to_v / attn.out_proj
    """
    import re
    new_sd = {}
    # Pattern : tout ce qui se termine par .to_q, .to_k, .to_v, .proj_attn
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
    n_remapped = sum(1 for k, nk in zip(state_dict, new_sd) if k != nk)
    if n_remapped:
        print(f"  [remap_monai_attn] {n_remapped} clés reméppées (ancienne API → MONAI ≥1.4)")
    return new_sd


# ===========================================================================
# Build UNet 3D multimodal
# ===========================================================================


def build_unet_3d(cfg: dict, latent_channels: int, n_classes: int) -> nn.Module:
    """UNet 3D conditionné sur le temps et la classe (modalité, champ) cible.

    V1 : standard DiffusionModelUNet
    V2 : HybridUNetTransformer avec attention factorisée au bottleneck
    """
    m = cfg["model"]
    channel_mult = tuple(m.get("channel_mult", [1, 2, 4]))
    base_channels = m.get("model_channels", 128)
    channels = tuple(base_channels * c for c in channel_mult)

    # Compatibilité MONAI : arg "channels" ou "num_channels" selon la version
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


# ===========================================================================
# Training
# ===========================================================================


def train(
    cfg_path: str,
    env_path: Optional[str] = None,
    resume: Optional[str] = None,
) -> None:
    cfg = load_yaml_with_include(cfg_path)
    cfg = resolve_paths(cfg, load_env(env_path))
    if resume is not None:
        cfg["resume"] = resume

    # Réglages GPU : TF32 + cudnn autotune (accélère sur GB10, VRAM large)
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
    if data_root is None:
        raise RuntimeError("data_root requis dans la config ou l'env.")

    modalities: List[str] = data_cfg.get("modalities", MODALITIES)
    fields: List[str] = data_cfg.get("fields", DOMAINS)
    n_fields = len(fields)
    n_classes = len(modalities) * n_fields

    output_dir = Path(data_cfg["output_dir"])
    split = data_cfg.get("split", "retro_train")
    p_lo = data_cfg.get("percentile_lower", 0.5)
    p_hi = data_cfg.get("percentile_upper", 99.5)
    max_per_class = data_cfg.get("max_volumes_per_class", None)
    random_crop_prob = float(data_cfg.get("random_crop_prob", 0.0))

    raw_vs = data_cfg.get("volume_size", None)
    if raw_vs is None:
        raise RuntimeError("volume_size est requis.")
    volume_size: Tuple[int, ...] = tuple(int(v) for v in raw_vs)

    raw_ts = data_cfg.get("target_spacing", None)
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else None

    total_iters = int(train_cfg.get("total_iters", 150000))
    batch_size = int(train_cfg.get("batch_size", 2))
    num_workers = int(train_cfg.get("num_workers", 4))
    lr = float(train_cfg.get("lr", 1e-4))
    sigma = float(train_cfg.get("sigma", 0.0))
    ot_method = train_cfg.get("ot_method", "exact")
    save_every = int(train_cfg.get("save_every", 5000))
    print_every = int(train_cfg.get("print_every", 200))
    use_amp = bool(train_cfg.get("use_amp", True))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    ema_decay = float(train_cfg.get("ema_decay", 0.9999))
    amp_dtype_name = train_cfg.get("amp_dtype", "bf16")
    num_targets_per_step = int(train_cfg.get("num_targets_per_step", 2))

    # ── AMP dtype ────────────────────────────────────────────────────────────
    if amp_dtype_name == "bf16":
        amp_dtype = torch.bfloat16
    elif amp_dtype_name == "fp16":
        amp_dtype = torch.float16
    else:
        raise ValueError("amp_dtype doit être 'bf16' ou 'fp16'.")
    use_scaler = use_amp and amp_dtype == torch.float16

    # ── Distributed ──────────────────────────────────────────────────────────
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
        print(f"World size : {world_size} | Device : {device}")
        print(f"Modalités  : {modalities}  |  Champs : {fields}  |  Classes : {n_classes}")

    # ── VAE (frozen) — chargé pour fallback encode & (toujours) métadonnées ───
    use_latent_cache = bool(data_cfg.get("use_latent_cache", False))
    flip_lr_prob = float(data_cfg.get("flip_lr_prob", 0.0))

    vae = None
    if not use_latent_cache:
        vae = load_vae(cfg, device)
        if vae.latent_format != "spatial":
            raise RuntimeError(
                f"train_mmfm_unet_3d.py requiert un VAE spatial (latent_format='spatial'), "
                f"mais '{cfg['vae'].get('vae_type', '?')}' a latent_format='{vae.latent_format}'."
            )
        latent_channels = vae.latent_channels
    else:
        latent_channels = int(cfg["vae"].get("latent_channels", 1))

    # ── Dataset : latents pré-encodés (cache) OU volumes à encoder à la volée ──
    def _sample_class(sample):
        """Retourne l'indice de classe d'un échantillon (tuple ou dict)."""
        return sample["class_idx"] if isinstance(sample, dict) else sample[3]

    if use_latent_cache:
        cache_root = Path(data_cfg.get("latent_cache_root", "outputs/latent_cache"))
        cache_dir = data_cfg.get("latent_cache_dir")
        if cache_dir is None:
            raise RuntimeError(
                "use_latent_cache=True : renseignez data.latent_cache_dir "
                "(ex. outputs/latent_cache/<vae_id>/retro_train)."
            )
        ds = LatentCacheDataset(
            cache_dir=Path(cache_dir),
            cache_root=cache_root,
            preload_ram=bool(data_cfg.get("latent_preload_ram", True)),
            flip_lr_prob=flip_lr_prob,
            flip_axis=int(data_cfg.get("flip_axis", 0)),
        )
        if is_main_process():
            print(f"  Cache latents : {cache_dir} | latent_shape={ds.latent_shape} "
                  f"| flip_lr_prob={flip_lr_prob} | preload_ram={ds.preload_ram}")
        if latent_channels != ds.latent_shape[0]:
            latent_channels = ds.latent_shape[0]
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
        )

    persistent = num_workers > 0
    class_loaders: Dict[int, any] = {}
    for c_idx in range(n_classes):
        class_indices = [i for i, s in enumerate(ds.samples) if _sample_class(s) == c_idx]
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

    available_classes = sorted(class_loaders.keys())
    if len(available_classes) < 2:
        raise RuntimeError("Il faut au moins 2 classes (modalité, champ) non vides.")

    # ── Multi-marginal : organiser les loaders par (contraste, champ) ──────────
    # La CLASSE conditionnante du réseau est le CONTRASTE (mod_idx).
    # Le CHAMP magnétique (field_idx) est un point temporel ordonné.
    # contrast_fields[mod_idx] = liste triée des field_idx disponibles
    contrast_fields: Dict[int, List[int]] = {}
    for flat in available_classes:
        m_idx, f_idx = _unflat_class(flat, n_fields)
        contrast_fields.setdefault(m_idx, []).append(f_idx)
    for m_idx in contrast_fields:
        contrast_fields[m_idx] = sorted(contrast_fields[m_idx])

    # Contrastes utilisables : ceux ayant >= 2 champs (pour définir une trajectoire)
    usable_contrasts = [m for m, fs in contrast_fields.items() if len(fs) >= 2]
    if not usable_contrasts:
        raise RuntimeError(
            "Aucun contraste n'a >= 2 champs disponibles ; impossible de définir "
            "une trajectoire multi-marginal."
        )

    identity_prob = float(train_cfg.get("identity_prob", 0.1))
    adjacent_only = bool(train_cfg.get("adjacent_only", True))

    if is_main_process():
        print(f"  Classes disponibles : {len(available_classes)}/{n_classes} "
              f"(volumes/classe min: 1, batch_size={batch_size})")
        print(f"  Multi-marginal : contrastes utilisables={usable_contrasts} | "
              f"champs/contraste={ {m: contrast_fields[m] for m in usable_contrasts} }")
        print(f"  identity_prob={identity_prob} | adjacent_only={adjacent_only} | "
              f"num_targets_per_step={num_targets_per_step}")

    # ── UNet 3D ───────────────────────────────────────────────────────────────
    unet = build_unet_3d(cfg, latent_channels, n_classes).to(device)
    if is_distributed:
        unet = DDP(unet, device_ids=[local_rank])
    raw_unet = unet.module if is_distributed else unet

    if is_main_process():
        n_params = sum(p.numel() for p in raw_unet.parameters() if p.requires_grad)
        # MONAI stocke l'embedding de classe sous class_embedding.weight (N, D)
        n_cls = raw_unet.state_dict().get("class_embedding.weight", raw_unet.state_dict().get("class_embed.weight"))
        n_cls = n_cls.shape[0] if n_cls is not None else n_classes
        print(f"UNet 3D MMFM : {n_params / 1e6:.1f}M params | "
              f"latent_channels={latent_channels} | n_class_embeds={n_cls}")

    ema = EMAModel(raw_unet, decay=ema_decay)

    # ── Optimizer / scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-4)

    def _lr_lambda(step: int) -> float:
        decay_start = total_iters // 2
        if step < decay_start:
            return 1.0
        return max(0.0, 1.0 - (step - decay_start) / max(total_iters - decay_start, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ── Flow Matcher ──────────────────────────────────────────────────────────
    if ot_method == "exact":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        if is_main_process():
            print(f"Flow matcher: OT-CFM exact, sigma={sigma}")
    else:
        FM = ConditionalFlowMatcher(sigma=sigma)
        if is_main_process():
            print(f"Flow matcher: CFM indépendant, sigma={sigma}")

    # ── Reprise ───────────────────────────────────────────────────────────────
    start_iter = 0
    resume_path = cfg.get("resume")
    if resume_path and Path(resume_path).exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        raw_unet.load_state_dict(_remap_monai_attention_keys(state["model"]))
        optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state and state["scheduler"] is not None:
            scheduler.load_state_dict(state["scheduler"])
        if "ema" in state:
            ema.load_state_dict(_remap_monai_attention_keys(state["ema"]))
        if "scaler" in state and use_scaler and state["scaler"] is not None:
            scaler.load_state_dict(state["scaler"])
        start_iter = state.get("iter", 0) + 1
        if is_main_process():
            print(f"Reprise depuis iter {start_iter} : {resume_path}")

    weights_dir = output_dir / "weights"
    metrics_path = output_dir / "train_metrics.jsonl"

    if is_main_process():
        print(
            f"\nEntraînement MMFM-UNet 3D : {total_iters} iters"
            f" | batch={batch_size} | targets/step={num_targets_per_step}"
            f" | lr={lr} | AMP={use_amp} ({amp_dtype_name})\n"
        )

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    unet.train()
    t0 = time.time()
    last_log_t = t0
    recent_losses: deque = deque(maxlen=print_every)

    print(f"[DEBUG] start_iter={start_iter} total_iters={total_iters} range_len={len(range(start_iter, total_iters))}", flush=True)

    for step in range(start_iter, total_iters):        # ── Échantillonnage multi-marginal ────────────────────────────────────
        # On tire un CONTRASTE puis num_targets_per_step transitions de champ.
        # Chaque transition = (field_i -> field_j) le long de l'axe temporel.
        # rank 0 décide, puis broadcast en DDP.
        def _sample_step_plan():
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
                    # sens aléatoire (montée/descente de champ)
                    if random.random() < 0.5:
                        fi, fj = fj, fi
                else:
                    fi, fj = random.sample(fields_avail, 2)
                transitions.append((fi, fj))
            return contrast, transitions

        if is_distributed:
            if dist.get_rank() == 0:
                contrast, transitions = _sample_step_plan()
                flat = [contrast]
                for (fi, fj) in transitions:
                    flat += [fi, fj]
                msg = torch.tensor(flat, dtype=torch.long, device=device)
            else:
                msg = torch.zeros(1 + 2 * num_targets_per_step, dtype=torch.long, device=device)
            dist.broadcast(msg, src=0)
            contrast = int(msg[0].item())
            transitions = [
                (int(msg[1 + 2 * i].item()), int(msg[2 + 2 * i].item()))
                for i in range(num_targets_per_step)
            ]
        else:
            contrast, transitions = _sample_step_plan()

        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        k = len(transitions)

        for (fi, fj) in transitions:
            src_flat = _flat_class(contrast, fi, n_fields)
            tgt_flat = _flat_class(contrast, fj, n_fields)

            try:
                src_item = next(class_loaders[src_flat])[0].to(device)
                if fi == fj:
                    tgt_item = src_item  # cas identité : même distribution
                else:
                    tgt_item = next(class_loaders[tgt_flat])[0].to(device)
            except Exception as e:
                print(f"[DEBUG] loader error at step {step}: {type(e).__name__}: {e}", flush=True)
                import traceback; traceback.print_exc()
                raise

            if use_latent_cache:
                # Les items sont déjà des latents (C, H', W', D') → (B, C, H', W', D')
                z_src = src_item.float()
                z_tgt = z_src if fi == fj else tgt_item.float()
            else:
                # Encode (VAE frozen, no grad)
                with torch.no_grad(), torch.amp.autocast(
                    "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
                ):
                    z_src = vae.encode(src_item)   # (B, C_lat, H', W', D')
                    z_tgt = z_src if fi == fj else vae.encode(tgt_item)

            # Pad zéro des dims spatiales vers multiple de 4 (contrainte UNet 3 niveaux)
            z_src, _ = _pad_to_multiple(z_src, 4)
            z_tgt, _ = _pad_to_multiple(z_tgt, 4)

            # Temps globaux des deux marginales sur l'axe champ
            t_i = _field_to_time(fi, n_fields)
            t_j = _field_to_time(fj, n_fields)
            dt_field = t_j - t_i  # peut être négatif (descente de champ)

            if fi == fj:
                # Cas identité : vitesse nulle, temps uniforme sur [0,1].
                B = z_src.shape[0]
                s = torch.rand(B, device=device)
                t_global = s  # n'importe quel temps → vitesse nulle
                z_t = z_src
                ut_global = torch.zeros_like(z_src)
                # z_cond = z_src (ancre l'anatomie)
                z_cond = z_src
            else:
                # OT-CFM entre les deux marginales adjacentes (données non appariées)
                t_local, z_t, ut_local = FM.sample_location_and_conditional_flow(z_src, z_tgt)
                # t_local ~ U[0,1] : position dans l'intervalle [t_i, t_j]
                t_global = t_i + t_local.to(device) * dt_field
                # Vitesse en temps GLOBAL : dz/dt_global = (z_j - z_i)/(t_j - t_i)
                ut_global = ut_local / dt_field
                # Conditionnement anatomique : le point de départ de la trajectoire (z au champ source fi)
                z_cond = z_src

            z_in = torch.cat([z_t, z_cond], dim=1)  # (B, 2*C_lat, H', W', D')
            t_vec = t_global.float()
            y = torch.full(
                (z_src.shape[0],), contrast, dtype=torch.long, device=device
            )

            with torch.amp.autocast(
                "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
            ):
                vt = raw_unet(x=z_in, timesteps=t_vec, class_labels=y)
                loss = F.mse_loss(vt, ut_global) / float(k)

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            step_losses.append(float(loss.item() * k))


        # Gradient step
        if use_scaler:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_unet.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_unet.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        ema.update(raw_unet)

        mean_step_loss = float(np.mean(step_losses)) if step_losses else 0.0
        recent_losses.append(mean_step_loss)

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
            print(
                f"[{step + 1:6d}/{total_iters}]"
                f"  loss={avg_recent:.4f}"
                f"  grad={float(grad_norm):.2f}"
                f"  lr={lr_cur:.2e}"
                f"  contrast={contrast}→fields{transitions}"
                f"  speed={it_s:.2f} it/s"
                f"  eta={eta_s / 3600:.2f}h"
                f"  t={elapsed / 60:.1f}min"
                f"  mem={mem_gb:.1f}GB"
            )
            # ── Écriture train_metrics.jsonl ──────────────────────────────
            record = {
                "iter": step + 1,
                "loss": round(avg_recent, 6),
                "grad_norm": round(float(grad_norm), 4),
                "lr": round(lr_cur, 8),
                "speed_it_s": round(it_s, 3),
                "elapsed_s": round(elapsed, 1),
                "mem_gb": round(mem_gb, 2),
            }
            with open(metrics_path, "a") as _mf:
                _mf.write(json.dumps(record) + "\n")
            last_log_t = time.time()

        if is_main_process() and (step + 1) % save_every == 0:
            ckpt_path = weights_dir / f"checkpoint_{step + 1}.pth"
            torch.save(
                {
                    "iter": step,
                    "model": raw_unet.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if use_scaler else None,
                    "cfg_path": str(cfg_path),
                    "n_classes": n_classes,
                    "num_class_embeds": int(cfg["model"].get("num_class_embeds", len(modalities))),
                    "modalities": modalities,
                    "fields": fields,
                },
                ckpt_path,
            )
            print(f"  → Checkpoint : {ckpt_path}")

    if is_main_process():
        final_path = weights_dir / "model_final.pth"
        torch.save(
            {
                "iter": total_iters - 1,
                "model": raw_unet.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if use_scaler else None,
                "cfg_path": str(cfg_path),
                "n_classes": n_classes,
                "num_class_embeds": int(cfg["model"].get("num_class_embeds", len(modalities))),
                "modalities": modalities,
                "fields": fields,
            },
            final_path,
        )
        print(f"\nEntraînement MMFM-UNet terminé. Modèle final : {final_path}")

    if is_distributed:
        dist.destroy_process_group()


# ===========================================================================
# Inference
# ===========================================================================


@torch.no_grad()
def _euler_integrate(
    unet: DiffusionModelUNet,
    z_src: torch.Tensor,
    tgt_class: int,
    n_steps: int,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """[LEGACY] Intégration Euler spatiale sur [0,1] conditionnée par classe.

    Conservée pour compatibilité avec les scripts d'inférence hérités
    (ex. scripts/infer_mmfm_unet_v2_batch.py). La formulation multi-marginal
    utilise `_euler_integrate_mm`.
    """
    dt = 1.0 / n_steps
    z = z_src.clone().to(device)
    y = torch.tensor([tgt_class], dtype=torch.long, device=device)

    for step_i in range(n_steps):
        t_val = step_i * dt
        t_vec = torch.tensor([t_val], dtype=torch.float32, device=device)
        z_in = torch.cat([z, z_src], dim=1)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            vt = unet(x=z_in, timesteps=t_vec, class_labels=y)
        z = z + dt * vt.float()

    return z


@torch.no_grad()
def _euler_integrate_mm(
    unet: DiffusionModelUNet,
    z_src: torch.Tensor,
    contrast_class: int,
    t_start: float,
    t_end: float,
    n_steps: int,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Intégration multi-marginal : trajectoire continue de t_start à t_end.

    Le CHAMP magnétique est l'axe temporel (t_start, t_end ∈ [0,1] donnés par
    `_field_to_time`). Le CONTRASTE est la classe conditionnante. L'anatomie
    source `z_src` est concaténée à chaque pas (ancre structurelle).

    Intègre l'EDO globale dz/dt = v_θ(z, t, contraste) de t_start vers t_end en
    `n_steps` pas Euler (passe par les champs intermédiaires : trajectoire unique).
    Supporte la montée (t_end>t_start) et la descente (t_end<t_start) de champ.

    Args:
        z_src: Latent source au champ de départ, shape (1, C_lat, H', W', D').
        contrast_class: indice de contraste (0=T1W, 1=T2W, 2=T2FLAIR).
        t_start, t_end: temps globaux (champ source, champ cible).
        n_steps: nombre de pas Euler sur l'intervalle [t_start, t_end].

    Returns:
        Latent prédit au champ cible, shape (1, C_lat, H', W', D').
    """
    z_anchor = z_src.clone().to(device)
    z = z_src.clone().to(device)
    y = torch.tensor([contrast_class], dtype=torch.long, device=device)

    dt = (t_end - t_start) / n_steps  # signé (peut être négatif)
    for step_i in range(n_steps):
        t_val = t_start + step_i * dt
        t_vec = torch.tensor([t_val], dtype=torch.float32, device=device)
        z_in = torch.cat([z, z_anchor], dim=1)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            vt = unet(x=z_in, timesteps=t_vec, class_labels=y)
        z = z + dt * vt.float()

    return z



def infer(
    cfg_path: str,
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
) -> None:
    """Inférence MMFM-UNet : encode → intégration Euler spatiale → decode."""
    cfg = load_yaml_with_include(cfg_path)
    cfg = resolve_paths(cfg, load_env(env_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    amp_dtype_name = train_cfg.get("amp_dtype", "bf16")
    use_amp = bool(train_cfg.get("use_amp", True))
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16

    n_steps = n_steps or cfg.get("inference", {}).get("n_steps", 50)
    p_lo = data_cfg.get("percentile_lower", 0.5)
    p_hi = data_cfg.get("percentile_upper", 99.5)

    raw_vs = data_cfg.get("volume_size", None)
    volume_size = tuple(int(v) for v in raw_vs) if raw_vs else None
    raw_ts = data_cfg.get("target_spacing", None)
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else None

    modalities: List[str] = data_cfg.get("modalities", MODALITIES)
    fields: List[str] = data_cfg.get("fields", DOMAINS)
    n_fields = len(fields)
    n_contrasts = len(modalities)
    n_classes = n_contrasts  # multi-marginal : la classe conditionnante = contraste

    # Valider source/cible
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

    contrast_class = modalities.index(source_modality)
    src_field_idx = fields.index(source_field)
    tgt_field_idx = fields.index(target_field)
    t_start = _field_to_time(src_field_idx, n_fields)
    t_end = _field_to_time(tgt_field_idx, n_fields)

    print(
        f"Inférence MMFM multi-marginal : {source_modality}@{source_field} → "
        f"{target_modality}@{target_field}  |  contrast={contrast_class}  |  "
        f"t:{t_start:.3f}→{t_end:.3f}  |  {n_steps} steps Euler (multi-pas)"
    )

    # ── VAE ───────────────────────────────────────────────────────────────────
    vae = load_vae(cfg, device)
    if vae.latent_format != "spatial":
        raise RuntimeError(
            f"Requiert un VAE spatial, got latent_format='{vae.latent_format}'."
        )
    latent_channels = vae.latent_channels

    # ── UNet ──────────────────────────────────────────────────────────────────
    unet = build_unet_3d(cfg, latent_channels, n_classes).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    # Vérifier cohérence du nombre de classes conditionnantes (contrastes)
    saved_nce = ckpt.get("num_class_embeds", None)
    if saved_nce is not None and saved_nce != n_classes:
        raise RuntimeError(
            f"Incohérence num_class_embeds : checkpoint={saved_nce}, "
            f"config courante={n_classes}. Utilisez la même config qu'à l'entraînement."
        )

    # Charger EMA si disponible
    loaded_from = "model"
    if use_ema and "ema" in ckpt and ckpt["ema"]:
        ema_state = ckpt["ema"]
        shadow = ema_state.get("shadow_params", None)
        if shadow is not None:
            unet.load_state_dict(_remap_monai_attention_keys(shadow))
            loaded_from = "ema.shadow_params"
        elif isinstance(ema_state, dict) and ema_state:
            # EMAModel.state_dict() plat (OrderedDict de poids)
            unet.load_state_dict(_remap_monai_attention_keys(ema_state))
            loaded_from = "ema"
        else:
            unet.load_state_dict(_remap_monai_attention_keys(ckpt["model"]))
    else:
        unet.load_state_dict(_remap_monai_attention_keys(ckpt["model"]))

    unet.eval()
    trained_at = ckpt.get("iter", "?")
    print(f"  UNet chargé (clé: {loaded_from}, iter={trained_at}) depuis {checkpoint}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collecter fichiers
    if input_volume is not None:
        input_files = [Path(input_volume)]
    elif input_dir is not None:
        input_files = sorted(Path(input_dir).glob("*.nii.gz"))
        if not input_files:
            raise FileNotFoundError(f"Aucun fichier .nii.gz dans {input_dir}")
    else:
        raise ValueError("Fournir --input_dir ou --input_volume.")

    print(f"  {len(input_files)} volume(s) → {out_dir}")

    for nii_path in input_files:
        t_start = time.time()

        vol, _ = load_nifti_volume(
            nii_path,
            target_spacing=target_spacing,
            volume_size=volume_size,
            normalize=True,
            lo_pct=p_lo,
            hi_pct=p_hi,
        )

        # Affine corrigé
        img_nib = nib.load(str(nii_path))
        orig_spacing = np.abs(np.diag(img_nib.affine)[:3])
        orig_shape = np.array(img_nib.shape[:3])
        if target_spacing is not None:
            resampled_arr = resample_volume(
                np.zeros(orig_shape.tolist(), dtype=np.float32),
                orig_spacing, target_spacing,
            )
            resampled_shape = resampled_arr.shape
        else:
            resampled_shape = None

        out_affine = adjust_affine_for_crop_pad(
            img_nib.affine.copy().astype(float),
            orig_shape, volume_size, resampled_shape, target_spacing, orig_spacing,
        )

        vol_tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)

        # Encode
        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
        ):
            z_src = vae.encode(vol_tensor)

        # Pad zéro des dims latentes vers multiple de 4 (contrainte UNet 3 niveaux),
        # puis crop retour à la forme latente d'origine avant décodage.
        z_src_p, lat_shape = _pad_to_multiple(z_src, 4)

        # Intégration multi-marginal : trajectoire continue t_start → t_end
        z_tgt_p = _euler_integrate_mm(
            unet, z_src_p, contrast_class, t_start, t_end, n_steps,
            device, use_amp, amp_dtype,
        )
        z_tgt = _crop_to_shape(z_tgt_p, lat_shape)

        # Decode
        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
        ):
            recon = vae.decode(z_tgt)

        pred_vol = recon.squeeze().cpu().float().numpy()
        pred_vol = (np.clip(pred_vol, -1.0, 1.0) + 1.0) / 2.0  # [-1,1] → [0,1]

        # Apply source brain mask (background normalized to -1 is excluded)
        brain_mask = (vol > -0.99).astype(np.float32)
        pred_vol = pred_vol * brain_mask

        stem = nii_path.name.replace(".nii.gz", "")
        out_name = f"{stem}_{target_modality}_{target_field}_mmfm_unet.nii.gz"
        out_path = out_dir / out_name
        nib.save(nib.Nifti1Image(pred_vol, out_affine), str(out_path))

        elapsed = time.time() - t_start
        print(f"  {nii_path.name} → {out_name}  ({elapsed:.1f}s)")

    print(f"\nInférence MMFM-UNet terminée. Prédictions dans : {out_dir}")


# ===========================================================================
# CLI
# ===========================================================================


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MMFM-UNet 3D — Any-to-any multimodal avec UNet 3D spatial"
    )
    p.add_argument("--mode", default="train", choices=["train", "infer"])
    p.add_argument("--config", required=True, help="Chemin vers le YAML de configuration")
    p.add_argument("--env", default=None, help="Env YAML (local / remote / chemin)")
    # Train-only
    p.add_argument("--resume", default=None, help="[train] Reprendre depuis ce checkpoint")
    # Infer-only
    p.add_argument("--checkpoint", default=None,
                   help="[infer] Chemin vers le checkpoint (.pth)")
    p.add_argument("--input_dir", default=None,
                   help="[infer] Répertoire de volumes .nii.gz source")
    p.add_argument("--input_volume", default=None,
                   help="[infer] Volume .nii.gz unique source")
    p.add_argument("--output_dir", default=None,
                   help="[infer] Répertoire de sortie des prédictions")
    p.add_argument("--source_field", default=None,
                   help="[infer] Champ magnétique source (ex. '0.1T')")
    p.add_argument("--source_modality", default=None,
                   help="[infer] Modalité source (ex. 'T1W')")
    p.add_argument("--target_field", default=None,
                   help="[infer] Champ magnétique cible (ex. '7T')")
    p.add_argument("--target_modality", default=None,
                   help="[infer] Modalité cible (ex. 'T1W')")
    p.add_argument("--n_steps", type=int, default=None,
                   help="[infer] Nombre de pas Euler (défaut: 50)")
    p.add_argument("--no_ema", action="store_true",
                   help="[infer] Ignorer les poids EMA")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.mode == "train":
        train(args.config, env_path=args.env, resume=args.resume)
    else:
        missing = [
            name for name, val in [
                ("--checkpoint", args.checkpoint),
                ("--output_dir", args.output_dir),
                ("--source_field", args.source_field),
                ("--source_modality", args.source_modality),
                ("--target_field", args.target_field),
                ("--target_modality", args.target_modality),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(
                f"Mode infer : arguments manquants : {', '.join(missing)}\n"
                "Fournir aussi --input_dir ou --input_volume."
            )
        if args.input_dir is None and args.input_volume is None:
            raise ValueError("Mode infer : fournir --input_dir ou --input_volume.")
        infer(
            cfg_path=args.config,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            source_field=args.source_field,
            source_modality=args.source_modality,
            target_field=args.target_field,
            target_modality=args.target_modality,
            env_path=args.env,
            input_dir=args.input_dir,
            input_volume=args.input_volume,
            n_steps=args.n_steps,
            use_ema=not args.no_ema,
        )
