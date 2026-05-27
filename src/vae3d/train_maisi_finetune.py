#!/usr/bin/env python3
"""MedVAE Fine-tuning — Phase D (MRIxFields 2026).

Entraîne (ou évalue) un MedVAE (StanfordMIMI) pré-entraîné sur les données
MRIxFields. Trois modes via configs/medvae_finetune_multimodal.yaml :
  frozen          : évaluation seule (pas de gradient)
  decoder_only    : fine-tuning du décodeur uniquement
  false           : fine-tuning complet (encodeur + décodeur)

Usage :
  # Single-GPU
  python src/vae3d/train_maisi_finetune.py \
      --config configs/medvae_finetune_multimodal.yaml --env local

  # Multi-GPU (torchrun)
  torchrun --nproc_per_node=4 src/vae3d/train_maisi_finetune.py \
      --config configs/medvae_finetune_multimodal.yaml --env local
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from common.config import load_env, resolve_paths
from common.distributed import is_main_process
from common.io import DOMAINS
from common.dataset_vae import MRIxFieldsMultimodalDataset, vae_multimodal_collate
from models.maisi_vae import build_medvae_wrapper, MedVAEFineTuneWrapper


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #


def train(
    cfg_path: str,
    env_path: Optional[str] = None,
    resume: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = resolve_paths(cfg, load_env(env_path))
    if resume is not None:
        cfg["train"]["resume"] = resume
    if overrides:
        for section, values in overrides.items():
            cfg.setdefault(section, {}).update(values)

    # ── Env / paths ──────────────────────────────────────────────────────────
    data_root = cfg["data"].get("data_root")
    if data_root is None:
        raise RuntimeError("data_root requis dans la config ou l'env.")
    output_dir = Path(cfg["data"]["output_dir"])

    # ── Distributed ──────────────────────────────────────────────────────────
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
        print(f"Device     : {device}  |  World: {world_size}")

    # ── Hyper-params ─────────────────────────────────────────────────────────
    vae_cfg = cfg.get("vae", {})
    frozen_cfg = vae_cfg.get("frozen", True)
    model_name  = vae_cfg.get("model_name", "medvae_4_1_3d")
    kl_weight   = float(vae_cfg.get("kl_weight", 1e-6))

    tcfg       = cfg["train"]
    total_epochs= tcfg["total_epochs"]
    batch_size  = tcfg["batch_size"]
    num_workers = tcfg.get("num_workers", 4)
    lr          = tcfg["lr"]
    save_every  = tcfg.get("save_every_epochs", 5)
    print_every = tcfg.get("print_every", 50)
    use_amp     = tcfg.get("use_amp", True)
    grad_clip   = tcfg.get("grad_clip", 1.0)

    dcfg = cfg["data"]
    modalities  = dcfg.get("modalities", ["T1W", "T2W", "T2FLAIR"])
    fields      = dcfg.get("fields", DOMAINS)
    splits      = dcfg.get("splits", ["retro_train"])
    patch_size  = tuple(dcfg["patch_size"])
    p_lo        = dcfg.get("percentile_lower", 0.5)
    p_hi        = dcfg.get("percentile_upper", 99.5)
    raw_ts      = dcfg.get("target_spacing", None)
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else None

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_ds = MRIxFieldsMultimodalDataset(
        data_root=Path(data_root),
        splits=splits,
        modalities=modalities,
        fields=fields,
        patch_size=patch_size,
        target_spacing=target_spacing,
        percentile_lower=p_lo,
        percentile_upper=p_hi,
        is_training=True,
    )

    try:
        val_ds = MRIxFieldsMultimodalDataset(
            data_root=Path(data_root),
            splits=["pro_val"],
            modalities=modalities,
            fields=fields,
            patch_size=patch_size,
            target_spacing=target_spacing,
            percentile_lower=p_lo,
            percentile_upper=p_hi,
            is_training=False,
        )
        if is_main_process():
            print(f"  Validation: prospective split ({len(val_ds)} volumes)")
    except FileNotFoundError:
        n_total = len(train_ds)
        n_val = max(1, int(n_total * 0.1))
        indices = list(range(n_total))
        np.random.RandomState(42).shuffle(indices)
        val_ds   = Subset(train_ds, indices[:n_val])
        train_ds = Subset(train_ds, indices[n_val:])
        if is_main_process():
            print(f"  Validation: auto-split ({n_val} volumes)")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=num_workers,
        pin_memory=True, drop_last=True, collate_fn=vae_multimodal_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=vae_multimodal_collate,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    ckpt_path_vae = vae_cfg.get("checkpoint") or ""
    wrapper: MedVAEFineTuneWrapper = build_medvae_wrapper(
        model_name=model_name,
        frozen=(frozen_cfg is True or frozen_cfg == "frozen"),
        kl_weight=kl_weight,
        checkpoint=ckpt_path_vae if ckpt_path_vae else None,
    )

    # Partial unfreeze
    if isinstance(frozen_cfg, str) and frozen_cfg == "decoder_only":
        wrapper.unfreeze_decoder()
        if is_main_process():
            print(f"  MedVAE ({model_name}): decoder fine-tuning only")
    elif frozen_cfg is False or frozen_cfg == "false":
        wrapper.unfreeze_all()
        if is_main_process():
            print(f"  MedVAE ({model_name}): full fine-tuning")
    else:
        if is_main_process():
            print(f"  MedVAE ({model_name}): frozen — eval mode only")

    wrapper = wrapper.to(device)

    if is_distributed:
        wrapper = DDP(wrapper, device_ids=[local_rank], find_unused_parameters=True)

    raw_model: MedVAEFineTuneWrapper = wrapper.module if is_distributed else wrapper

    trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in wrapper.parameters())
    if is_main_process():
        print(f"  Params trainable: {trainable/1e6:.1f}M / {total_p/1e6:.1f}M total")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable_params = [p for p in wrapper.parameters() if p.requires_grad]
    if not trainable_params:
        if is_main_process():
            print("  [INFO] No trainable parameters — running evaluation only.")
        _evaluate_frozen(raw_model, val_loader, device, use_amp, output_dir)
        if is_distributed:
            dist.destroy_process_group()
        return

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    amp_dtype = torch.float16 if use_amp else torch.float32

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    resume_path = tcfg.get("resume")
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if is_main_process():
            print(f"  Resumed from epoch {start_epoch}: {resume_path}")

    weights_dir = output_dir / "weights"

    if is_main_process():
        print(
            f"\nFine-tuning MedVAE: {total_epochs} epochs"
            f" | batch={batch_size} | lr={lr} | kl={kl_weight}"
            f" | patch={patch_size} | AMP={'on' if use_amp else 'off'}\n"
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs):
        wrapper.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = epoch_recon = epoch_kl = 0.0
        n_batches = 0
        t_start = time.time()
        recent: deque = deque(maxlen=print_every)

        for i, batch_dict in enumerate(train_loader):
            images = batch_dict["x"].to(device)   # (B, 1, H, W, D)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = raw_model.forward_train(images)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss  += float(loss.item())
            epoch_recon += float(out.recon_loss.item())
            epoch_kl    += float(out.kl_loss.item())
            n_batches   += 1
            recent.append(float(loss.item()))

            if is_main_process() and (i + 1) % print_every == 0:
                avg = sum(recent) / len(recent)
                speed = (i + 1) / max(time.time() - t_start, 1e-9)
                eta_h = (len(train_loader) * (total_epochs - epoch) - i) / max(speed, 1e-9) / 3600
                mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                print(
                    f"  [E{epoch+1:03d}/{total_epochs} {i+1:4d}/{len(train_loader)}]"
                    f"  loss={avg:.4f}"
                    f"  recon={epoch_recon/n_batches:.4f}"
                    f"  kl={epoch_kl/n_batches:.2e}"
                    f"  speed={speed:.2f}it/s  eta={eta_h:.1f}h  mem={mem_gb:.1f}GB"
                )

        # ── Validation ────────────────────────────────────────────────────────
        val_loss = _validate(raw_model, val_loader, device, use_amp, amp_dtype)
        if is_distributed:
            vt = torch.tensor(val_loss, device=device)
            dist.all_reduce(vt, op=dist.ReduceOp.AVG)
            val_loss = float(vt.item())

        if is_main_process():
            avg_train = epoch_loss / max(n_batches, 1)
            elapsed   = (time.time() - t_start) / 60
            is_best   = val_loss < best_val_loss
            print(
                f"Epoch {epoch+1:3d}/{total_epochs}"
                f"  train={avg_train:.4f}  val={val_loss:.4f}"
                f"  time={elapsed:.1f}min"
                + (" ← best" if is_best else "")
            )

            if is_best:
                best_val_loss = val_loss
                torch.save(
                    {"epoch": epoch, "model": raw_model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "best_val_loss": best_val_loss, "cfg_path": cfg_path},
                    weights_dir / "model_best.pth",
                )

            if (epoch + 1) % save_every == 0:
                torch.save(
                    {"epoch": epoch, "model": raw_model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "best_val_loss": best_val_loss, "cfg_path": cfg_path},
                    weights_dir / f"epoch_{epoch+1:04d}.pth",
                )

    if is_main_process():
        torch.save(
            {"epoch": total_epochs - 1, "model": raw_model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "best_val_loss": best_val_loss, "cfg_path": cfg_path},
            weights_dir / "model_final.pth",
        )
        print(f"\nDone. Best val_loss={best_val_loss:.4f}")
        print(f"Final checkpoint: {weights_dir / 'model_final.pth'}")

    if is_distributed:
        dist.destroy_process_group()


@torch.no_grad()
def _validate(model, val_loader, device, use_amp, amp_dtype) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch_dict in val_loader:
        images = batch_dict["x"].to(device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = model.forward_train(images)
        total += float(out.loss.item())
        n += 1
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def _evaluate_frozen(model, val_loader, device, use_amp, output_dir: Path) -> None:
    """Évalue le modèle gelé sur le val set et sauvegarde les métriques."""
    import torch.nn.functional as F
    amp_dtype = torch.float16 if use_amp else torch.float32
    model.eval()
    total_l1 = total_kl = 0.0
    n = 0
    for batch_dict in val_loader:
        images = batch_dict["x"].to(device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = model.forward_train(images)
        total_l1 += float(out.recon_loss.item())
        total_kl  += float(out.kl_loss.item())
        n += 1
    print(f"\n[Frozen eval] L1={total_l1/max(n,1):.4f}  KL={total_kl/max(n,1):.2e}  (N={n})")
    results_csv = output_dir / "frozen_eval.csv"
    with open(results_csv, "w") as f:
        f.write("l1_recon,kl\n")
        f.write(f"{total_l1/max(n,1):.6f},{total_kl/max(n,1):.6f}\n")
    print(f"Eval results saved to {results_csv}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedVAE Fine-tuning — MRIxFields 2026")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--env",    default=None,  help="Env YAML or name (local, remote)")
    p.add_argument("--resume", default=None,  help="Resume from checkpoint path")
    p.add_argument("--frozen", default=None,  help="Override vae.frozen (true|false|decoder_only)")
    p.add_argument("--lr",     default=None,  type=float, help="Override train.lr")
    p.add_argument("--batch-size", default=None, type=int, help="Override train.batch_size")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    overrides: dict = {}
    if args.frozen     is not None:
        frozen_val = args.frozen
        if frozen_val.lower() in ("true", "1", "frozen"):
            frozen_val = True
        elif frozen_val.lower() in ("false", "0"):
            frozen_val = False
        overrides.setdefault("vae", {})["frozen"] = frozen_val
    if args.lr         is not None: overrides.setdefault("train", {})["lr"]         = args.lr
    if args.batch_size is not None: overrides.setdefault("train", {})["batch_size"] = args.batch_size
    train(args.config, env_path=args.env, resume=args.resume, overrides=overrides)
