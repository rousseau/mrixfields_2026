#!/usr/bin/env python3
"""MMFM-UNet 3D — Any-to-any multimodal avec UNet 3D spatial.

Extension de MMFM-MLP (train_mmfm_3d.py) qui remplace le MLP vectoriel par un
UNet 3D spatial (MONAI DiffusionModelUNet), similaire à train_cfm_3d.py mais
étendu au cadre multimodal any-to-any (15 classes = 3 mod × 5 champs).

Différences clés vs MMFM-MLP (train_mmfm_3d.py) :
  - Architecture : DiffusionModelUNet 3D (pas de flatten du latent)
  - Input : cat(z_t, z_src) → (B, 2*C_lat, H', W', D')  [comme CFM]
  - Output : champ vectoriel spatial (B, C_lat, H', W', D')
  - Conditioning : AdaGN à chaque résolution (profond), num_class_embeds=15
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
from common.dataset import MultiModalNIfTILatentDataset
from common.distributed import is_main_process, EMAModel
from common.io import (
    DOMAINS,
    MODALITIES,
    adjust_affine_for_crop_pad,
    load_nifti_volume,
    resample_volume,
)
from models.vae_loader import load_vae

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
    """Index de classe unique : mod_idx * n_fields + field_idx."""
    return mod_idx * n_fields + field_idx


def _make_infinite(loader: DataLoader):
    while True:
        yield from loader


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


def build_unet_3d(cfg: dict, latent_channels: int, n_classes: int) -> DiffusionModelUNet:
    """UNet 3D conditionné sur le temps et la classe (modalité, champ) cible.

    Identique à train_cfm_3d.py::build_unet_3d, mais :
      - num_class_embeds = n_classes (15 pour 3 mod × 5 champs)
      - lu depuis cfg["model"]["num_class_embeds"] si présent
    """
    m = cfg["model"]
    channel_mult = tuple(m.get("channel_mult", [1, 2, 4]))
    base_channels = m.get("model_channels", 128)
    channels = tuple(base_channels * c for c in channel_mult)

    # Compatibilité MONAI : arg "channels" ou "num_channels" selon la version
    _sig = inspect.signature(DiffusionModelUNet.__init__).parameters
    _ch_kwarg = "num_channels" if "num_channels" in _sig else "channels"

    num_class_embeds = int(m.get("num_class_embeds", n_classes))

    return DiffusionModelUNet(
        spatial_dims=3,
        in_channels=2 * latent_channels,      # cat(z_t, z_src)
        out_channels=latent_channels,          # champ de vitesse spatial
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

    # ── VAE (frozen) ─────────────────────────────────────────────────────────
    vae = load_vae(cfg, device)
    if vae.latent_format != "spatial":
        raise RuntimeError(
            f"train_mmfm_unet_3d.py requiert un VAE spatial (latent_format='spatial'), "
            f"mais '{cfg['vae'].get('vae_type', '?')}' a latent_format='{vae.latent_format}'. "
            "Pour les VAE vectoriels (RHVAE), utilisez train_mmfm_3d.py."
        )
    latent_channels = vae.latent_channels

    # ── Dataset multimodal — 15 loaders (1 par classe) ───────────────────────
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
    )

    class_loaders: Dict[int, any] = {}
    for c_idx in range(n_classes):
        class_indices = [i for i, (_, _, _, c) in enumerate(ds.samples) if c == c_idx]
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
        )
        class_loaders[c_idx] = _make_infinite(loader)

    available_classes = sorted(class_loaders.keys())
    if len(available_classes) < 2:
        raise RuntimeError("Il faut au moins 2 classes (modalité, champ) non vides.")

    if is_main_process():
        print(f"  Classes disponibles : {len(available_classes)}/{n_classes} "
              f"(volumes/classe min: 1, batch_size={batch_size})")

    # ── UNet 3D ───────────────────────────────────────────────────────────────
    unet = build_unet_3d(cfg, latent_channels, n_classes).to(device)
    if is_distributed:
        unet = DDP(unet, device_ids=[local_rank])
    raw_unet = unet.module if is_distributed else unet

    if is_main_process():
        n_params = sum(p.numel() for p in raw_unet.parameters() if p.requires_grad)
        print(f"UNet 3D MMFM : {n_params / 1e6:.1f}M params | "
              f"latent_channels={latent_channels} | n_classes={n_classes}")

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
        if "ema" in state:
            ema.load_state_dict(state["ema"])
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

    for step in range(start_iter, total_iters):
        # Synchroniser le choix src/tgt entre tous les ranks DDP
        if is_distributed:
            if dist.get_rank() == 0:
                src_idx = torch.tensor(
                    [random.randrange(len(available_classes))],
                    dtype=torch.long, device=device,
                )
                tgt_candidates = [c for c in available_classes
                                  if c != available_classes[src_idx.item()]]
                k = min(num_targets_per_step, len(tgt_candidates))
                tgt_idx_t = torch.tensor(
                    random.sample(range(len(tgt_candidates)), k),
                    dtype=torch.long, device=device,
                )
                # Broadcast : [src_idx, k, tgt_idx_0, tgt_idx_1, ...]
                msg = torch.cat([src_idx, torch.tensor([k], device=device), tgt_idx_t])
            else:
                # Taille max = 1 + 1 + num_targets_per_step
                msg = torch.zeros(2 + num_targets_per_step, dtype=torch.long, device=device)
            dist.broadcast(msg, src=0)
            src_class = available_classes[msg[0].item()]
            k = int(msg[1].item())
            tgt_candidates = [c for c in available_classes if c != src_class]
            tgt_classes = [tgt_candidates[msg[2 + i].item()] for i in range(k)]
        else:
            src_class = random.choice(available_classes)
            tgt_candidates = [c for c in available_classes if c != src_class]
            k = min(num_targets_per_step, len(tgt_candidates))
            tgt_classes = random.sample(tgt_candidates, k=k)

        src_batch = next(class_loaders[src_class])
        src_x = src_batch[0].to(device)

        optimizer.zero_grad(set_to_none=True)
        step_losses = []

        for tgt_class in tgt_classes:
            tgt_batch = next(class_loaders[tgt_class])
            tgt_x = tgt_batch[0].to(device)

            # Encode (VAE frozen, no grad)
            with torch.no_grad(), torch.amp.autocast(
                "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
            ):
                z_src = vae.encode(src_x)   # (B, C_lat, H', W', D')
                z_tgt = vae.encode(tgt_x)

            # OT-CFM en espace latent spatial
            t_batch, z_t, ut = FM.sample_location_and_conditional_flow(z_src, z_tgt)

            z_in = torch.cat([z_t, z_src], dim=1)        # (B, 2*C_lat, H', W', D')
            t_vec = t_batch.to(device).float()
            y = torch.full(
                (z_src.shape[0],), tgt_class, dtype=torch.long, device=device
            )

            with torch.amp.autocast(
                "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
            ):
                vt = raw_unet(x=z_in, timesteps=t_vec, class_labels=y)
                loss = F.mse_loss(vt, ut) / float(k)

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
                f"  src={src_class}→{tgt_classes}"
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
                    "scaler": scaler.state_dict() if use_scaler else None,
                    "cfg_path": str(cfg_path),
                    "n_classes": n_classes,
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
                "scaler": scaler.state_dict() if use_scaler else None,
                "cfg_path": str(cfg_path),
                "n_classes": n_classes,
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
    """Intégration Euler spatiale : z_src → z_tgt.

    Args:
        unet: DiffusionModelUNet chargé, en mode eval.
        z_src: Latent source, shape (1, C_lat, H', W', D').
        tgt_class: Indice de classe cible (mod_idx * n_fields + field_idx).
        n_steps: Nombre de pas Euler.
        device: Périphérique de calcul.

    Returns:
        Latent prédit, shape (1, C_lat, H', W', D').
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
    n_classes = len(modalities) * n_fields

    # Valider source/cible
    for name, val, lst in [
        ("source_field", source_field, fields),
        ("target_field", target_field, fields),
        ("source_modality", source_modality, modalities),
        ("target_modality", target_modality, modalities),
    ]:
        if val not in lst:
            raise ValueError(f"{name}='{val}' non présent dans la config ({lst})")

    tgt_mod_idx = modalities.index(target_modality)
    tgt_field_idx = fields.index(target_field)
    tgt_class = _flat_class(tgt_mod_idx, tgt_field_idx, n_fields)

    print(
        f"Inférence MMFM-UNet : {source_modality}@{source_field} → "
        f"{target_modality}@{target_field}  |  tgt_class={tgt_class}  |  {n_steps} steps Euler"
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

    # Vérifier cohérence n_classes
    saved_n_classes = ckpt.get("n_classes", None)
    if saved_n_classes is not None and saved_n_classes != n_classes:
        raise RuntimeError(
            f"Incohérence n_classes : checkpoint={saved_n_classes}, "
            f"config courante={n_classes}. Utilisez la même config que lors de l'entraînement."
        )

    # Charger EMA si disponible
    loaded_from = "model"
    if use_ema and "ema" in ckpt and ckpt["ema"]:
        ema_state = ckpt["ema"]
        shadow = ema_state.get("shadow_params", None)
        if shadow is not None:
            unet.load_state_dict(_remap_monai_attention_keys(shadow))
            loaded_from = "ema.shadow_params"
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

        # Intégration Euler spatiale
        z_tgt = _euler_integrate(unet, z_src, tgt_class, n_steps, device, use_amp, amp_dtype)

        # Decode
        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
        ):
            recon = vae.decode(z_tgt)

        pred_vol = recon.squeeze().cpu().float().numpy()
        pred_vol = (np.clip(pred_vol, -1.0, 1.0) + 1.0) / 2.0  # [-1,1] → [0,1]

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
