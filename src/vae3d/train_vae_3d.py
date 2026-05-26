#!/usr/bin/env python3
"""VAE 3D — Pré-entraînement d'un autoencodeur variationnel 3D pour IRM.

Architecture : AutoencoderKL 3D (MONAI) — 3 niveaux de compression → 8× spatiale.
Pour un volume 512×512×192 → latent 64×64×24 (en latent_channels=4).

Usage :
  # Single-GPU
  python src/train_vae_3d.py --config configs/vae3d_T1W.yaml --env local

  # Multi-GPU (torchrun, DDP)
  torchrun --nproc_per_node=4 src/train_vae_3d.py \\
      --config configs/vae3d_T1W.yaml --env jeanzay

  # Reprendre depuis un checkpoint
  python src/train_vae_3d.py --config configs/vae3d_T1W.yaml \\
      --resume outputs/vae3d/runs/vae3d_T1W/weights/epoch_050.pth
"""

import argparse
import inspect
import os
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

# MONAI generative models
try:
    from monai.networks.nets import AutoencoderKL
except ImportError:
    try:
        from monai.generative.networks.nets import AutoencoderKL
    except ImportError:
        from generative.networks.nets import AutoencoderKL

from common.config import load_env, resolve_paths
from common.distributed import is_main_process
from common.io import DOMAINS, normalize_volume, resample_volume, list_nifti_files
from pathlib import Path
import nibabel as nib


# --------------------------------------------------------------------------- #
# Dataset with patch extraction                                               #
# --------------------------------------------------------------------------- #


class PatchedNIfTIVolumeDataset(Dataset):
    """Dataset with random/center patch extraction for VAE pre-training.

    Uses common.io preprocessing functions for resampling and normalization.
    """

    def __init__(
        self,
        data_root: Path,
        split: str,
        modality: str,
        domains: List[str],
        patch_size: Optional[Tuple[int, int, int]] = (112, 128, 80),
        percentile_lower: float = 0.5,
        percentile_upper: float = 99.5,
        is_training: bool = True,
        target_spacing: Optional[Tuple[float, float, float]] = None,
    ):
        self.patch_size = patch_size
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.is_training = is_training
        self.target_spacing = target_spacing

        self.volumes: List[Path] = []
        for domain in domains:
            files = list_nifti_files(data_root, split, modality, domain)
            if not files:
                print(f"  [WARN] Aucun volume dans {split}/{modality}/{domain}")
            self.volumes.extend(files)

        if not self.volumes:
            raise FileNotFoundError(
                f"Aucun volume NIfTI trouvé dans {data_root}/{split}/{modality}/"
            )
        print(
            f"  PatchedNIfTIVolumeDataset: {len(self.volumes)} volumes ({modality}, {split})"
        )

    def __len__(self) -> int:
        return len(self.volumes)

    def _load_and_preprocess(self, idx: int) -> np.ndarray:
        img_nib = nib.load(str(self.volumes[idx]))
        if self.target_spacing is not None:
            spacing = np.abs(np.diag(img_nib.affine)[:3])
            vol = img_nib.get_fdata(dtype=np.float32)
            vol = resample_volume(vol, spacing, self.target_spacing)
        else:
            vol = img_nib.get_fdata(dtype=np.float32)
        vol = normalize_volume(vol, self.percentile_lower, self.percentile_upper)
        return vol

    def _random_crop(self, vol: np.ndarray) -> np.ndarray:
        ph, pw, pd = self.patch_size
        h, w, d = vol.shape
        ph_pad = max(0, ph - h)
        pw_pad = max(0, pw - w)
        pd_pad = max(0, pd - d)
        if ph_pad > 0 or pw_pad > 0 or pd_pad > 0:
            vol = np.pad(
                vol,
                [(0, ph_pad), (0, pw_pad), (0, pd_pad)],
                mode="reflect",
            )
            h, w, d = vol.shape
        sh = np.random.randint(0, h - ph + 1)
        sw = np.random.randint(0, w - pw + 1)
        sd = np.random.randint(0, d - pd + 1)
        return vol[sh : sh + ph, sw : sw + pw, sd : sd + pd]

    def _center_crop(self, vol: np.ndarray) -> np.ndarray:
        ph, pw, pd = self.patch_size
        h, w, d = vol.shape
        ph_pad = max(0, ph - h)
        pw_pad = max(0, pw - w)
        pd_pad = max(0, pd - d)
        if ph_pad > 0 or pw_pad > 0 or pd_pad > 0:
            vol = np.pad(vol, [(0, ph_pad), (0, pw_pad), (0, pd_pad)], mode="reflect")
            h, w, d = vol.shape
        sh = (h - ph) // 2
        sw = (w - pw) // 2
        sd = (d - pd) // 2
        return vol[sh : sh + ph, sw : sw + pw, sd : sd + pd]

    def __getitem__(self, idx: int) -> torch.Tensor:
        vol = self._load_and_preprocess(idx)
        if self.patch_size is not None:
            if self.is_training:
                vol = self._random_crop(vol)
            else:
                vol = self._center_crop(vol)
        return torch.from_numpy(vol).unsqueeze(0)  # (1, H, W, D)


# --------------------------------------------------------------------------- #
# Build VAE 3D                                                                #
# --------------------------------------------------------------------------- #


def build_vae(cfg: dict) -> AutoencoderKL:
    """Construit un AutoencoderKL 3D via MONAI."""
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

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig}
    return AutoencoderKL(**filtered_kwargs)


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #


def train(
    cfg_path: str, env_path: Optional[str] = None, resume: Optional[str] = None
) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
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
    patch_size = tuple(cfg["data"]["patch_size"])
    p_lo = cfg["data"].get("percentile_lower", 0.5)
    p_hi = cfg["data"].get("percentile_upper", 99.5)
    raw_ts = cfg["data"].get("target_spacing", None)
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else None
    if target_spacing and is_main_process():
        print(f"  Resampling → spacing cible : {target_spacing} mm")

    total_epochs = cfg["train"]["total_epochs"]
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"].get("num_workers", 4)
    lr = cfg["train"]["lr"]
    kl_weight = cfg["model"]["kl_weight"]
    kl_warmup = cfg["train"].get("kl_warmup_epochs", 10)
    save_every = cfg["train"].get("save_every_epochs", 10)
    print_every = cfg["train"].get("print_every", 100)
    use_amp = cfg["train"].get("use_amp", True)
    grad_clip = cfg["train"].get("grad_clip", 1.0)

    # ── Distributed setup ───────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend=cfg["train"].get("dist_backend", "nccl"))
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "weights").mkdir(exist_ok=True)
        print(f"Output dir : {output_dir}")
        print(f"World size : {world_size}")
        print(f"Device     : {device}")

    # ── Dataset ─────────────────────────────────────────────────────────────
    train_ds = PatchedNIfTIVolumeDataset(
        data_root=Path(data_root),
        split=split,
        modality=modality,
        domains=domains,
        patch_size=patch_size,
        percentile_lower=p_lo,
        percentile_upper=p_hi,
        is_training=True,
        target_spacing=target_spacing,
    )

    # Validation split
    val_split_fraction = cfg["data"].get("val_split_fraction", 0.2)
    try:
        val_ds = PatchedNIfTIVolumeDataset(
            data_root=Path(data_root),
            split="pro_val",
            modality=modality,
            domains=domains,
            patch_size=patch_size,
            percentile_lower=p_lo,
            percentile_upper=p_hi,
            is_training=False,
            target_spacing=target_spacing,
        )
        if is_main_process():
            print(f"  → Validation dataset chargé depuis Validating_prospective")
    except FileNotFoundError:
        if is_main_process():
            print(f"  [INFO] Validation dataset non trouvé")
            print(
                f"  → Création d'un split train/val depuis {split} "
                f"({int((1 - val_split_fraction) * 100)}% train, {int(val_split_fraction * 100)}% val)"
            )
        full_ds = train_ds
        n_total = len(full_ds)
        n_val = max(1, int(n_total * val_split_fraction))
        indices = list(range(n_total))
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        train_ds = Subset(full_ds, train_indices)
        val_ds = Subset(full_ds, val_indices)

    train_sampler = (
        DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ── Modèle ──────────────────────────────────────────────────────────────
    model = build_vae(cfg).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        print(f"VAE 3D : {n_params / 1e6:.1f}M paramètres")

    # ── Optimizer & scaler ───────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    # ── Reprise ──────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    resume_path = cfg.get("resume")
    if resume_path and Path(resume_path).exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model = model.module if is_distributed else model
        raw_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state.get("epoch", 0) + 1
        best_val_loss = state.get("best_val_loss", float("inf"))
        if is_main_process():
            print(f"Reprise depuis epoch {start_epoch} : {resume_path}")

    weights_dir = output_dir / "weights"
    amp_dtype = torch.float16 if use_amp else torch.float32

    if is_main_process():
        print(
            f"\nEntraînement VAE 3D : {total_epochs} epochs"
            f" | batch={batch_size}"
            f" | lr={lr}"
            f" | patch={patch_size}"
            f" | kl_weight={kl_weight}"
            f" | AMP={'oui' if use_amp else 'non'}\n"
        )

    for epoch in range(start_epoch, total_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        kl_factor = min(1.0, (epoch + 1) / max(kl_warmup, 1))
        effective_kl = kl_weight * kl_factor

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0
        t_epoch_start = time.time()
        recent_losses: deque = deque(maxlen=print_every)

        raw_model = model.module if is_distributed else model

        for i, batch in enumerate(train_loader):
            images = batch.to(device)  # (B, 1, H, W, D)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                recon, z_mu, z_sigma = raw_model(images)
                recon_loss = F.l1_loss(recon, images)
                kl_loss = 0.5 * torch.mean(z_mu.pow(2) + z_sigma.exp() - z_sigma - 1.0)
                loss = recon_loss + effective_kl * kl_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            loss_val = float(loss.item())
            epoch_loss += loss_val
            epoch_recon += float(recon_loss.item())
            epoch_kl += float(kl_loss.item())
            n_batches += 1
            recent_losses.append(loss_val)

            if is_main_process() and (i + 1) % print_every == 0:
                avg_recent = sum(recent_losses) / len(recent_losses)
                elapsed = time.time() - t_epoch_start
                iter_per_s = (i + 1) / max(elapsed, 1e-9)
                n_iters_left = len(train_loader) * (total_epochs - epoch) - i
                eta_h = n_iters_left / max(iter_per_s, 1e-9) / 3600
                mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                print(
                    f"  [E{epoch + 1:03d}/{total_epochs}  {i + 1:4d}/{len(train_loader)}]"
                    f"  loss={avg_recent:.4f}"
                    f"  recon={epoch_recon / n_batches:.4f}"
                    f"  kl={epoch_kl / n_batches:.5f}"
                    f"  kl_w={effective_kl:.2e}"
                    f"  speed={iter_per_s:.2f} it/s"
                    f"  eta={eta_h:.1f}h"
                    f"  mem={mem_gb:.1f}GB"
                )

        # ── Validation ────────────────────────────────────────────────────
        val_loss = _validate(
            raw_model, val_loader, device, use_amp, amp_dtype, effective_kl
        )
        if is_distributed:
            val_t = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_t, op=dist.ReduceOp.AVG)
            val_loss = float(val_t.item())

        if is_main_process():
            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed_total = time.time() - t_epoch_start
            print(
                f"Epoch {epoch + 1:3d}/{total_epochs}"
                f"  train={avg_loss:.4f}"
                f"  val={val_loss:.4f}"
                f"  time={elapsed_total / 60:.1f}min"
                + (" ← best" if val_loss < best_val_loss else "")
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "cfg_path": str(cfg_path),
                    },
                    weights_dir / "model_best.pth",
                )

    if is_main_process():
        torch.save(
            {
                "epoch": total_epochs - 1,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "cfg_path": str(cfg_path),
            },
            weights_dir / "model_final.pth",
        )
        print(f"\nEntraînement terminé. Best val_loss={best_val_loss:.4f}")
        print(f"Modèle final : {weights_dir / 'model_final.pth'}")

    if is_distributed:
        dist.destroy_process_group()


@torch.no_grad()
def _validate(
    model,
    val_loader,
    device,
    use_amp,
    amp_dtype,
    effective_kl,
):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in val_loader:
        images = batch.to(device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            recon, z_mu, z_sigma = model(images)
            recon_loss = F.l1_loss(recon, images)
            kl_loss = 0.5 * torch.mean(z_mu.pow(2) + z_sigma.exp() - z_sigma - 1.0)
            loss = recon_loss + effective_kl * kl_loss
        total_loss += float(loss.item())
        n += 1
    model.train()
    return total_loss / max(n, 1)


# --------------------------------------------------------------------------- #
# Inference helpers                                                           #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def encode_volume(
    model: AutoencoderKL,
    volume: np.ndarray,
    device: torch.device,
    percentile_lower: float = 0.5,
    percentile_upper: float = 99.5,
    use_amp: bool = False,
) -> torch.Tensor:
    """Encode un volume NIfTI numpy → latent tensor (1, C, H', W', D')."""
    vol = normalize_volume(volume, percentile_lower, percentile_upper)
    t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", enabled=use_amp):
        z_mu, _ = model.encode(t)
    return z_mu


@torch.no_grad()
def decode_volume(
    model: AutoencoderKL,
    latent: torch.Tensor,
    device: torch.device,
    use_amp: bool = False,
) -> np.ndarray:
    """Decode un latent tensor → volume numpy dans [-1, 1]."""
    with torch.amp.autocast("cuda", enabled=use_amp):
        recon = model.decode(latent)
    return np.clip(recon.squeeze().cpu().numpy(), -1.0, 1.0)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAE 3D — Pré-entraînement pour IRM")
    p.add_argument("--config", required=True, help="Chemin vers le YAML de config")
    p.add_argument("--env", default=None, help="Env YAML ou nom (local, jeanzay)")
    p.add_argument(
        "--resume", default=None, help="Chemin vers un checkpoint à reprendre"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    train(args.config, env_path=args.env, resume=args.resume)
