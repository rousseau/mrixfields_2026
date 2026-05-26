#!/usr/bin/env python3
"""OT-CFM 3D en espace latent — Traduction de contraste IRM 3D volumétrique.

Architecture :
  - VAE 3D pré-entraîné (AEKL / MedVAE / VQ-VAE) → espace latent
  - OT-CFM en espace latent : UNet 3D conditionné sur domaine cible
  - Input du UNet : cat(z_t, z_src)  →  (B, 2*C_lat, H', W', D')
  - Output : champ vectoriel de vitesse → (B, C_lat, H', W', D')
  - Entraîné avec ExactOT conditional flow matching (torchcfm)

VAE supportés (étape 2) :
  - aekl    : AutoencoderKL MONAI (4 canaux latents, 8x compression)
  - medvae  : MedVAE Stanford (frozen HuggingFace ou fine-tuné local)
  - vqvae   : NeuroQuant adapté (src/vae3d/train_vqvae.py, données paired+unpaired)

Usage :
  # Single-GPU
  python src/cfm/train_cfm_3d.py --config configs/cfm3d_T1W_aekl.yaml --env local
  python src/cfm/train_cfm_3d.py --config configs/cfm3d_T1W_medvae.yaml --env local
  python src/cfm/train_cfm_3d.py --config configs/cfm3d_T1W_vqvae.yaml --env local

  # Multi-GPU (torchrun DDP)
  torchrun --nproc_per_node=4 src/cfm/train_cfm_3d.py \\
      --config configs/cfm3d_T1W_aekl.yaml --env jeanzay

  # Inférence
  python src/cfm/train_cfm_3d.py --mode infer \\
      --config configs/cfm3d_T1W_aekl.yaml \\
      --checkpoint outputs/cfm3d/runs/cfm3d_T1W_aekl/weights/model_final.pth \\
      --input_dir /data/T1W/0.1T/ --output_dir /data/predictions/ \\
      --source_domain 0.1T --target_domain 7T
"""

import argparse
import inspect
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
from common.dataset import NIfTILatentDataset
from common.distributed import is_main_process, EMAModel
from common.io import (
    DOMAINS,
    DOMAIN_TO_IDX,
    center_crop_or_pad_np,
    load_nifti_volume,
    normalize_volume,
    resample_volume,
    adjust_affine_for_crop_pad,
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


NUM_DOMAINS = len(DOMAINS)


# ===========================================================================
# Build UNet 3D pour le flow matching en espace latent
# ===========================================================================


def build_unet_3d(cfg: dict, latent_channels: int) -> DiffusionModelUNet:
    """UNet 3D conditionné sur le temps et le domaine cible."""
    m = cfg["model"]
    channel_mult = tuple(m.get("channel_mult", [1, 2, 4]))
    base_channels = m.get("model_channels", 128)
    channels = tuple(base_channels * c for c in channel_mult)

    _sig = inspect.signature(DiffusionModelUNet.__init__).parameters
    _ch_kwarg = "num_channels" if "num_channels" in _sig else "channels"

    return DiffusionModelUNet(
        spatial_dims=3,
        in_channels=2 * latent_channels,
        out_channels=latent_channels,
        **{_ch_kwarg: channels},
        attention_levels=tuple(m.get("attention_levels", [False, True, True])),
        num_res_blocks=m.get("num_res_blocks", 2),
        num_head_channels=m.get("num_head_channels", 64),
        norm_num_groups=m.get("norm_num_groups", 32),
        use_flash_attention=m.get("use_flash_attention", False),
        num_class_embeds=NUM_DOMAINS,
        with_conditioning=False,
        resblock_updown=True,
    )


# ===========================================================================
# Infinite data loader helper
# ===========================================================================


def _make_infinite(loader: DataLoader):
    while True:
        yield from loader


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

    data_root = cfg["data"].get("data_root")
    if data_root is None:
        raise RuntimeError("data_root requis dans la config ou l'env.")

    output_dir = Path(cfg["data"]["output_dir"])
    modality = cfg["data"]["modality"]
    split = cfg["data"].get("split", "retro_train")
    domains = cfg["data"].get("domains", DOMAINS)
    p_lo = cfg["data"].get("percentile_lower", 0.5)
    p_hi = cfg["data"].get("percentile_upper", 99.5)
    max_per_dom = cfg["data"].get("max_volumes_per_domain", None)
    raw_vs = cfg["data"].get("volume_size", None)
    volume_size = tuple(int(v) for v in raw_vs) if raw_vs else None
    raw_ts = cfg["data"].get("target_spacing", None)
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else None

    total_iters = cfg["train"]["total_iters"]
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"].get("num_workers", 4)
    lr = cfg["train"]["lr"]
    sigma = cfg["train"].get("sigma", 0.0)
    ot_method = cfg["train"].get("ot_method", "exact")
    save_every = cfg["train"].get("save_every", 5000)
    print_every = cfg["train"].get("print_every", 200)
    use_amp = cfg["train"].get("use_amp", True)
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    ema_decay = cfg["train"].get("ema_decay", 0.9999)

    # ── Distributed ─────────────────────────────────────────────────────────
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

    if is_main_process() and target_spacing:
        print(f"  Resampling → spacing cible : {target_spacing} mm")

    # ── VAE (figé) ───────────────────────────────────────────────────────────
    vae = load_vae(cfg, device)
    latent_channels = vae.latent_channels

    # ── Dataset ─────────────────────────────────────────────────────────────
    train_ds = NIfTILatentDataset(
        data_root=Path(data_root),
        split=split,
        modality=modality,
        domains=domains,
        percentile_lower=p_lo,
        percentile_upper=p_hi,
        max_per_domain=max_per_dom,
        target_spacing=target_spacing,
        volume_size=volume_size,
    )

    # Pour chaque domaine, un loader infini séparé
    domain_loaders: Dict[str, any] = {}
    for d in domains:
        domain_indices = [
            i for i, (_, di) in enumerate(train_ds.samples) if di == DOMAIN_TO_IDX[d]
        ]
        domain_subset = torch.utils.data.Subset(train_ds, domain_indices)
        if not domain_subset:
            continue
        sampler = (
            DistributedSampler(domain_subset, shuffle=True) if is_distributed else None
        )
        loader = DataLoader(
            domain_subset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        domain_loaders[d] = _make_infinite(loader)

    available_domains = list(domain_loaders.keys())
    if len(available_domains) < 2:
        raise RuntimeError(
            f"Besoin d'au moins 2 domaines, {len(available_domains)} disponible(s)."
        )

    # ── UNet 3D ──────────────────────────────────────────────────────────────
    unet = build_unet_3d(cfg, latent_channels).to(device)
    if is_distributed:
        unet = DDP(unet, device_ids=[local_rank])

    n_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    if is_main_process():
        print(f"UNet 3D : {n_params / 1e6:.1f}M paramètres")

    ema = EMAModel(unet.module if is_distributed else unet, decay=ema_decay)

    # ── Optimizer / scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-4)

    def _lr_lambda(step: int) -> float:
        decay_start = total_iters // 2
        if step < decay_start:
            return 1.0
        return max(0.0, 1.0 - (step - decay_start) / (total_iters - decay_start))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    # ── Flow Matcher ─────────────────────────────────────────────────────────
    try:
        if ot_method == "exact":
            FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
        else:
            FM = ConditionalFlowMatcher(sigma=sigma)
        if is_main_process():
            print(f"Flow matcher: OT-CFM ({ot_method}), sigma={sigma}")
    except Exception:
        FM = ConditionalFlowMatcher(sigma=sigma)
        if is_main_process():
            print("Flow matcher: CFM (indépendant, fallback)")

    # ── Reprise ───────────────────────────────────────────────────────────────
    start_iter = 0
    resume_path = cfg.get("resume")
    if resume_path and Path(resume_path).exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        raw_unet = unet.module if is_distributed else unet
        raw_unet.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "ema" in state:
            ema.load_state_dict(state["ema"])
        start_iter = state.get("iter", 0) + 1
        if is_main_process():
            print(f"Reprise depuis iter {start_iter} : {resume_path}")

    weights_dir = output_dir / "weights"
    amp_dtype = torch.float16 if use_amp else torch.float32

    if is_main_process():
        print(
            f"\nEntraînement OT-CFM 3D latent : {total_iters} iters"
            f" | batch={batch_size}"
            f" | lr={lr}"
            f" | AMP={'oui' if use_amp else 'non'}\n"
        )

    unet.train()
    t0 = time.time()
    last_log_t = t0
    ema_loss: Optional[float] = None
    recent_losses: deque = deque(maxlen=print_every)
    raw_unet = unet.module if is_distributed else unet

    for step in range(start_iter, total_iters):
        src_domain, tgt_domain = random.sample(available_domains, 2)
        tgt_idx = DOMAIN_TO_IDX[tgt_domain]

        src_vol, _ = next(domain_loaders[src_domain])
        tgt_vol, _ = next(domain_loaders[tgt_domain])
        src_vol = src_vol.to(device)
        tgt_vol = tgt_vol.to(device)

        with (
            torch.no_grad(),
            torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp),
        ):
            z_src = vae.encode(src_vol)
            z_tgt = vae.encode(tgt_vol)

        t_batch, z_t, ut = FM.sample_location_and_conditional_flow(z_src, z_tgt)

        z_in = torch.cat([z_t, z_src], dim=1)
        t_vec = t_batch.to(device).float()
        y = torch.full((z_src.shape[0],), tgt_idx, dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            vt = raw_unet(x=z_in, timesteps=t_vec, class_labels=y)
            loss = F.mse_loss(vt, ut)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_unet.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        ema.update(unet)

        loss_val = float(loss.item())
        recent_losses.append(loss_val)
        ema_loss = loss_val if ema_loss is None else (0.98 * ema_loss + 0.02 * loss_val)

        if is_main_process() and (step + 1) % print_every == 0:
            avg_recent = sum(recent_losses) / len(recent_losses)
            elapsed = time.time() - t0
            window_dt = time.time() - last_log_t
            iter_per_s = print_every / max(window_dt, 1e-9)
            eta_sec = (total_iters - step - 1) / max(iter_per_s, 1e-9)
            lr_cur = scheduler.get_last_lr()[0]
            mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            print(
                f"[{step + 1:6d}/{total_iters}]"
                f"  loss={avg_recent:.4f}"
                f"  ema={ema_loss:.4f}"
                f"  grad={float(grad_norm):.2f}"
                f"  lr={lr_cur:.2e}"
                f"  pair={src_domain}→{tgt_domain}"
                f"  speed={iter_per_s:.2f} it/s"
                f"  eta={eta_sec / 3600:.2f}h"
                f"  t={elapsed / 60:.1f}min"
                f"  mem={mem_gb:.1f}GB"
            )
            last_log_t = time.time()

        if is_main_process() and (step + 1) % save_every == 0:
            ckpt_path = weights_dir / f"checkpoint_{step + 1}.pth"
            torch.save(
                {
                    "iter": step,
                    "model": raw_unet.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg_path": str(cfg_path),
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
                "cfg_path": str(cfg_path),
            },
            final_path,
        )
        print(f"\nEntraînement terminé. Modèle final : {final_path}")

    if is_distributed:
        dist.destroy_process_group()


# ===========================================================================
# Inference
# ===========================================================================


@torch.no_grad()
def _euler_integrate(
    unet: DiffusionModelUNet,
    z_src: torch.Tensor,
    tgt_idx: int,
    n_steps: int,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype=torch.float16,
) -> torch.Tensor:
    """Intégration Euler : z_src → z_tgt via le champ vectoriel appris."""
    dt = 1.0 / n_steps
    z = z_src.clone().to(device)
    y = torch.tensor([tgt_idx], dtype=torch.long, device=device)

    for step_i in range(n_steps):
        t_val = step_i * dt
        t_vec = torch.tensor([t_val], dtype=torch.float32, device=device)
        z_in = torch.cat([z, z_src], dim=1)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            vt = unet(x=z_in, timesteps=t_vec, class_labels=y)
        z = z + dt * vt

    return z


def infer(
    cfg_path: str,
    checkpoint: str,
    input_dir: str,
    output_dir: str,
    source_domain: str,
    target_domain: str,
    env_path: Optional[str] = None,
    n_steps: Optional[int] = None,
    use_ema: bool = True,
) -> None:
    """Inférence complète : encode → intégration ODE → decode."""
    cfg = load_yaml_with_include(cfg_path)
    cfg = resolve_paths(cfg, load_env(env_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["train"].get("use_amp", False)
    amp_dtype = torch.float16 if use_amp else torch.float32
    n_steps = n_steps or cfg["inference"].get("n_steps", 50)
    tgt_idx = DOMAIN_TO_IDX[target_domain]
    p_lo = cfg["data"].get("percentile_lower", 0.5)
    p_hi = cfg["data"].get("percentile_upper", 99.5)

    raw_vs = cfg["data"].get("volume_size", None)
    volume_size_inf = tuple(int(v) for v in raw_vs) if raw_vs else None
    raw_ts = cfg["data"].get("target_spacing", None)
    target_spacing_inf = tuple(float(v) for v in raw_ts) if raw_ts else None

    print(f"Inférence CFM 3D : {source_domain} → {target_domain} | {n_steps} steps")

    # ── Charger VAE ──────────────────────────────────────────────────────────
    vae = load_vae(cfg, device)
    latent_channels = vae.latent_channels

    # ── Charger UNet ─────────────────────────────────────────────────────────
    unet = build_unet_3d(cfg, latent_channels).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    key = "ema" if (use_ema and "ema" in state) else "model"
    unet.load_state_dict(state[key])
    unet.eval()
    print(f"  UNet chargé depuis : {checkpoint} (clé: {key})")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(Path(input_dir).glob("*.nii.gz"))
    if not input_files:
        raise FileNotFoundError(f"Aucun fichier .nii.gz dans {input_dir}")
    print(f"  {len(input_files)} volumes à inférer → {out_dir}")

    for nii_path in input_files:
        t_start = time.time()
        vol, out_affine = load_nifti_volume(
            nii_path,
            target_spacing=target_spacing_inf,
            volume_size=volume_size_inf,
            normalize=True,
            lo_pct=p_lo,
            hi_pct=p_hi,
        )

        img_nib = nib.load(str(nii_path))
        orig_spacing = np.abs(np.diag(img_nib.affine)[:3])
        orig_shape = img_nib.shape[:3]

        # Adjust affine for resampling + crop/pad
        if target_spacing_inf is not None:
            resampled_shape = np.array(nib.load(str(nii_path)).get_fdata(dtype=np.float32).shape[:3])
            # Need to compute actual resampled shape
            resampled_float = resample_volume(
                np.zeros(orig_shape, dtype=np.float32),
                orig_spacing,
                target_spacing_inf,
            )
            resampled_shape = resampled_float.shape
        else:
            resampled_shape = orig_shape

        out_affine = adjust_affine_for_crop_pad(
            img_nib.affine.copy().astype(float),
            orig_shape,
            volume_size_inf,
            resampled_shape if target_spacing_inf else None,
            target_spacing_inf,
            orig_spacing,
        )

        vol_tensor = (
            torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
        )  # (1, 1, H, W, D)

        # Encode
        with (
            torch.no_grad(),
            torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp),
        ):
            z_src = vae.encode(vol_tensor)

        # ODE integration
        z_pred = _euler_integrate(
            unet, z_src, tgt_idx, n_steps, device, use_amp, amp_dtype
        )

        # Decode
        with (
            torch.no_grad(),
            torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp),
        ):
            recon = vae.decode(z_pred)

        pred_vol = recon.squeeze().cpu().numpy()
        pred_vol = (np.clip(pred_vol, -1.0, 1.0) + 1.0) / 2.0

        out_nii = nib.Nifti1Image(pred_vol, out_affine)
        out_path = out_dir / nii_path.name
        nib.save(out_nii, str(out_path))

        elapsed = time.time() - t_start
        print(f"  {nii_path.name} → {out_path.name}  ({elapsed:.1f}s)")

    print(f"\nInférence terminée. Prédictions dans : {out_dir}")


# ===========================================================================
# CLI
# ===========================================================================


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OT-CFM 3D — Traduction de contraste IRM 3D"
    )
    p.add_argument("--mode", default="train", choices=["train", "infer"])
    p.add_argument("--config", required=True)
    p.add_argument("--env", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--input_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--source_domain", default=None)
    p.add_argument("--target_domain", default=None)
    p.add_argument("--n_steps", type=int, default=None)
    p.add_argument("--no_ema", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.mode == "train":
        train(args.config, env_path=args.env, resume=args.resume)
    else:
        if not all(
            [
                args.checkpoint,
                args.input_dir,
                args.output_dir,
                args.source_domain,
                args.target_domain,
            ]
        ):
            raise ValueError(
                "Mode infer : --checkpoint, --input_dir, --output_dir, "
                "--source_domain, --target_domain requis."
            )
        infer(
            cfg_path=args.config,
            checkpoint=args.checkpoint,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            source_domain=args.source_domain,
            target_domain=args.target_domain,
            env_path=args.env,
            n_steps=args.n_steps,
            use_ema=not args.no_ema,
        )
