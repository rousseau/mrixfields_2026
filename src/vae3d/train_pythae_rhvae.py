#!/usr/bin/env python3
"""Entraînement Pythae RHVAE 3D — MRIxFields 2026.

Entraîne un PythaeRHVAE3D (Riemannian Hamiltonian VAE, latent vectoriel) sur
les volumes IRM multimodaux (T1W + T2W + T2FLAIR) de tous les champs magnétiques.

Particularités vs VAE/VQ-VAE classiques :
  - Loss gérée entièrement par Pythae RHVAE (HMC leapfrog + énergie riemannienne)
  - update_metric() appelé à la fin de chaque epoch pour geler les centroids
  - Latent vectoriel (B, D_lat) → compatible MMFM vectorisé

Usage :
  # Local (DGX GB10)
  python src/vae3d/train_pythae_rhvae.py \\
      --config configs/pythae_rhvae_multimodal.yaml --env local

  # Multi-GPU (torchrun)
  torchrun --nproc_per_node=4 src/vae3d/train_pythae_rhvae.py \\
      --config configs/pythae_rhvae_multimodal.yaml --env local

  # Remote (Jean Zay — via SLURM)
  sbatch src/slurm/train_vae_jeanzay.slurm pythae_rhvae configs/pythae_rhvae_multimodal.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import yaml

# Ajouter src/ au PYTHONPATH
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.config import load_env, resolve_paths
from common.dataset_vae import MRIxFieldsMultimodalDataset, vae_multimodal_collate
from models.pythae_rhvae import build_pythae_rhvae_3d, PythaeRHVAE3D


# ── Distributed helpers ───────────────────────────────────────────────────────

def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", 0)) == 0

def print_main(*args, **kwargs):
    if is_main_process():
        print(*args, flush=True, **kwargs)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg_path: str, env_path: Optional[str] = None,
          resume: Optional[str] = None, overrides: Optional[dict] = None) -> None:

    # ── Config ───────────────────────────────────────────────────────────────
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = resolve_paths(cfg, load_env(env_path))
    if resume is not None:
        cfg["resume"] = resume
    if overrides:
        for section, values in overrides.items():
            cfg.setdefault(section, {}).update(values)

    data_root = cfg["data"].get("data_root")
    if data_root is None:
        raise RuntimeError("data_root requis dans la config ou l'env.")

    output_dir = Path(cfg["data"]["output_dir"])
    modalities = cfg["data"].get("modalities", ["T1W", "T2W", "T2FLAIR"])
    if isinstance(modalities, str):
        modalities = [modalities]
    splits = cfg["data"].get("splits", ["retro_train"])
    fields = cfg["data"].get("fields", ["0.1T", "1.5T", "3T", "5T", "7T"])
    patch_size = tuple(cfg["data"]["patch_size"])
    p_lo = cfg["data"].get("percentile_lower", 0.5)
    p_hi = cfg["data"].get("percentile_upper", 99.5)

    total_epochs = cfg["train"]["total_epochs"]
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"].get("num_workers", 4)
    lr = cfg["train"]["lr"]
    save_every = cfg["train"].get("save_every_epochs", 10)
    print_every = cfg["train"].get("print_every", 50)
    # AMP désactivé par défaut pour RHVAE (leapfrog instable en fp16)
    use_amp = cfg["train"].get("use_amp", False)
    grad_clip = cfg["train"].get("grad_clip", 1.0)

    vae_cfg = cfg.get("vae", {})
    latent_dim   = int(vae_cfg.get("latent_dim", 256))
    base_channels = int(vae_cfg.get("base_channels", 32))
    num_groups   = int(vae_cfg.get("num_groups", 8))
    spatial_size = int(vae_cfg.get("spatial_size", 16))
    n_lf         = int(vae_cfg.get("n_lf", 3))
    eps_lf       = float(vae_cfg.get("eps_lf", 0.001))
    beta_zero    = float(vae_cfg.get("beta_zero", 0.3))
    temperature  = float(vae_cfg.get("temperature", 1.5))
    regularization = float(vae_cfg.get("regularization", 0.01))

    # ── Distributed ───────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend=cfg["train"].get("dist_backend", "nccl"))
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = MRIxFieldsMultimodalDataset(
        data_root=data_root,
        modalities=modalities,
        fields=fields,
        splits=splits,
        patch_size=patch_size,
        percentile_lower=p_lo,
        percentile_upper=p_hi,
    )
    sampler = (
        torch.utils.data.DistributedSampler(dataset, shuffle=True)
        if is_distributed else None
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=vae_multimodal_collate,
        pin_memory=True,
        drop_last=True,
    )
    print_main(f"  Dataset: {len(dataset)} patches | {len(loader)} batches/epoch")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_pythae_rhvae_3d(
        latent_dim=latent_dim,
        base_channels=base_channels,
        num_groups=num_groups,
        spatial_size=spatial_size,
        n_lf=n_lf,
        eps_lf=eps_lf,
        beta_zero=beta_zero,
        temperature=temperature,
        regularization=regularization,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # find_unused_parameters=True car metric + encoder partagent le backbone
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    ckpt_resume = cfg.get("resume")
    if ckpt_resume and Path(ckpt_resume).exists():
        ckpt = torch.load(ckpt_resume, map_location="cpu", weights_only=False)
        raw_model = model.module if is_distributed else model
        raw_model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print_main(f"  Reprise depuis epoch {start_epoch} ({ckpt_resume})")

    # ── Output dirs ───────────────────────────────────────────────────────────
    if is_main_process():
        (output_dir / "weights").mkdir(parents=True, exist_ok=True)
        print_main(f"  Output: {output_dir}")
        print_main(f"  latent_dim={latent_dim} | base_channels={base_channels} | "
                   f"n_lf={n_lf} | eps_lf={eps_lf} | beta_zero={beta_zero}")

    # ── Training ──────────────────────────────────────────────────────────────
    best_loss = float("inf")

    for epoch in range(start_epoch, total_epochs):
        if is_distributed:
            sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(loader):
            x = batch["x"].to(device, non_blocking=True)  # (B, 1, H, W, D)

            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                raw_model = model.module if is_distributed else model
                out = raw_model.forward_train(x)
                loss = out.loss

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1

            if is_main_process() and (step + 1) % print_every == 0:
                print(f"  Ep {epoch:04d} | step {step+1:05d}/{len(loader)} "
                      f"| loss={loss.item():.4f}", flush=True)

        avg_loss = epoch_loss / max(1, n_batches)
        print_main(f"Epoch {epoch:04d}/{total_epochs} | loss={avg_loss:.4f}")

        # ── Update Riemannian metric (end of epoch) ────────────────────────────
        # Must be called on raw (non-DDP) model — syncs metric centroids across
        # accumulated M / centroids stored during the epoch.
        raw_model = model.module if is_distributed else model
        raw_model.update_metric()

        # ── Checkpoint ────────────────────────────────────────────────────────
        if is_main_process():
            state = {
                "epoch": epoch,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
                "cfg": cfg,
            }
            if (epoch + 1) % save_every == 0:
                torch.save(state, output_dir / "weights" / f"epoch_{epoch+1:04d}.pth")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(state, output_dir / "weights" / "model_best.pth")

    if is_main_process():
        raw_model = model.module if is_distributed else model
        torch.save(
            {"epoch": total_epochs - 1, "model": raw_model.state_dict(), "cfg": cfg},
            output_dir / "weights" / "model_final.pth",
        )
        print(f"Entraînement terminé. Meilleure loss : {best_loss:.4f}", flush=True)

    if is_distributed:
        dist.destroy_process_group()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pythae RHVAE 3D — Entraînement multimodal")
    p.add_argument("--config", required=True, help="Chemin vers le YAML de config")
    p.add_argument("--env", default=None, help="Env YAML ou nom (local, remote)")
    p.add_argument("--resume", default=None, help="Checkpoint à reprendre")
    p.add_argument("--run-name", default=None, help="Override data.output_subdir suffix")
    p.add_argument("--steps", default=None, type=int, help="Override train.total_epochs")
    p.add_argument("--batch-size", default=None, type=int, help="Override train.batch_size")
    p.add_argument("--lr", default=None, type=float, help="Override train.lr")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    overrides: dict = {}
    if args.run_name   is not None: overrides.setdefault("data",  {})["run_name"]     = args.run_name
    if args.steps      is not None: overrides.setdefault("train", {})["total_epochs"] = args.steps
    if args.batch_size is not None: overrides.setdefault("train", {})["batch_size"]   = args.batch_size
    if args.lr         is not None: overrides.setdefault("train", {})["lr"]           = args.lr
    train(args.config, env_path=args.env, resume=args.resume, overrides=overrides)
