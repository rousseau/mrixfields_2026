#!/usr/bin/env python3
"""Meta-training script for the shared INR backbone (SIREN + hypernetwork).

Trains the shared (theta, psi) — see cfm/inr_backbone.py — on the full
retro_train set (pooled across modalities/fields: the backbone doesn't need
contrast/field conditioning, since a per-volume `z` already absorbs whatever
makes each volume distinct). This is the "frozen shared encoder" step the
INR variant needs before it can be used by arch_inr.py — analogous to how
MedVAE is trained once, frozen, then consumed by arch_vector.py/arch_unet.py.

Training tricks are deliberately kept consistent with the other two "trained
once, then frozen" / "shared" pipelines in this project (mmfm_core.py's flow
training, MedVAE fine-tuning) rather than picking ad hoc choices per script:
grad clipping (all three), AdamW + weight_decay=1e-4 (all three), a
warmup+decay LR schedule (all three — this script previously had none, a
real inconsistency vs. both other pipelines), and EMA (mmfm_core.py already
uses it for the flow models; MedVAE fine-tuning currently doesn't — see
docs/MMFM_INR_STATE_OF_THE_ART.md / project memory for that known gap,
deliberately not backported to MedVAE without an explicit decision given the
cost of invalidating the already-reported vectorized/unet results).

The resulting checkpoint is what cfm/precompute_inr_latents.py and
cfm/arch_inr.py load via cfg['inr_backbone']['checkpoint'] (EMA weights
preferred when present, matching mmfm_core.py's inference convention).

Usage:
    PYTHONPATH=src python src/cfm/train_inr_backbone.py \\
        --config configs/mmfm/inr_backbone.yaml --env local
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_yaml_with_include, load_env, resolve_paths
from common.dataset import MultiModalNIfTILatentDataset
from common.distributed import EMAModel
from common.io import DOMAINS, MODALITIES
from dataclasses import asdict

from cfm.inr_backbone import INRBackbone, INRBackboneConfig, make_coord_grid, meta_train_step


def _make_infinite(loader: DataLoader):
    while True:
        yield from loader


def _backbone_config_from_cfg(cfg: dict) -> INRBackboneConfig:
    m = cfg.get("model", {})
    return INRBackboneConfig(
        latent_dim=int(m.get("latent_dim", 512)),
        hidden_dim=int(m.get("hidden_dim", 256)),
        num_hidden_layers=int(m.get("num_hidden_layers", 6)),
        omega_0=float(m.get("omega_0", 30.0)),
        omega_hidden=float(m.get("omega_hidden", 30.0)),
        hyper_hidden_dim=int(m.get("hyper_hidden_dim", 64)),
        inner_lr=float(m.get("inner_lr", 1e-2)),
        inner_steps_train=int(m.get("inner_steps_train", 5)),
        inner_steps_eval=int(m.get("inner_steps_eval", 10)),
        fg_weight=float(m.get("fg_weight", 5.0)),
        bg_threshold=float(m.get("bg_threshold", -0.9)),
        modulate_scale=bool(m.get("modulate_scale", False)),
        lora_rank=int(m.get("lora_rank", 0)),
    )


def _save_checkpoint(path: Path, step: int, backbone, ema: EMAModel, optimizer, scheduler, cfg_path: str) -> None:
    torch.save({
        "iter": step,
        "model": backbone.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cfg_path": str(cfg_path),
        # D0 : config RESOLUE du backbone au moment de l'entraînement — source
        # de vérité lue par load_inr_backbone (arch_inr.py) pour détecter une
        # divergence config/consommation qui serait invisible à load_state_dict
        # (inner_lr, inner_steps_*, fg_weight, bg_threshold — mêmes formes de
        # poids, mais un z source hors distribution). Absent des checkpoints
        # antérieurs → le garde-fou se tait sur ceux-ci (diff_inr_configs).
        "backbone_cfg": asdict(backbone.cfg),
    }, path)


def main():
    ap = argparse.ArgumentParser(description="Meta-entraînement du backbone INR partagé (theta, psi)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--env", default="local")
    ap.add_argument("--resume", default=None, help="Reprendre depuis ce checkpoint (poids seuls)")
    args = ap.parse_args()

    cfg = load_yaml_with_include(args.config)
    cfg = resolve_paths(cfg, load_env(args.env))

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modalities: List[str] = data_cfg.get("modalities", MODALITIES)
    fields: List[str] = data_cfg.get("fields", DOMAINS)
    volume_size = tuple(int(v) for v in data_cfg["volume_size"])
    raw_ts = data_cfg.get("target_spacing")
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else None

    output_dir = Path(data_cfg["output_dir"])
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"

    total_iters = int(train_cfg.get("total_iters", 50000))
    batch_size = int(train_cfg.get("batch_size", 2))
    points_per_step = int(train_cfg.get("points_per_step", 16384))
    lr = float(train_cfg.get("lr", 1e-4))
    warmup_steps = int(train_cfg.get("warmup_steps", 2000))
    ema_decay = float(train_cfg.get("ema_decay", 0.9999))
    save_every = int(train_cfg.get("save_every", 2500))
    print_every = int(train_cfg.get("print_every", 100))
    num_workers = int(train_cfg.get("num_workers", data_cfg.get("num_workers", 4)))

    print(f"Output dir : {output_dir}", flush=True)
    print(f"Device : {device}  |  volume_size={volume_size}  |  target_spacing={target_spacing}", flush=True)

    ds = MultiModalNIfTILatentDataset(
        data_root=Path(data_cfg["data_root"]),
        split=data_cfg.get("split", "retro_train"),
        modalities=modalities,
        fields=fields,
        percentile_lower=data_cfg.get("percentile_lower", 0.5),
        percentile_upper=data_cfg.get("percentile_upper", 99.5),
        max_per_class=data_cfg.get("max_volumes_per_class", None),
        target_spacing=target_spacing,
        volume_size=volume_size,
        random_crop_prob=0.0,
    )
    print(f"  Dataset : {len(ds)} volumes (pooled — le backbone ne conditionne pas sur mod/champ)", flush=True)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0),
    )
    data_iter = _make_infinite(loader)

    backbone_cfg = _backbone_config_from_cfg(cfg)
    backbone = INRBackbone(backbone_cfg).to(device)
    n_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"  INRBackbone : {n_params / 1e6:.2f}M params | {backbone_cfg}", flush=True)

    ema = EMAModel(backbone, decay=ema_decay)
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=lr, weight_decay=1e-4)
    coords_full = make_coord_grid(volume_size, device=device)

    start_iter = 0
    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device, weights_only=False)
        backbone.load_state_dict(state["model"])
        if "ema" in state:
            ema.load_state_dict(state["ema"])
        start_iter = state.get("iter", 0) + 1
        print(f"  Reprise (poids seuls) depuis iter {start_iter} : {args.resume}", flush=True)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        decay_start = total_iters // 2
        if step < decay_start:
            return 1.0
        return max(0.0, 1.0 - (step - decay_start) / max(total_iters - decay_start, 1))

    # last_epoch=start_iter-1 so the schedule picks up exactly at start_iter
    # (the first scheduler.step() in the loop then lands on start_iter) —
    # constructing with the default last_epoch=-1 always begins _lr_lambda's
    # own counter at 0 regardless of start_iter, silently replaying a fresh
    # warmup/plateau/decay cycle offset from the real iteration count on any
    # --resume. Caught via the LR values logged by a real resumed run
    # (lr=2.5e-07 at step 50100, i.e. _lr_lambda(100), not _lr_lambda(50100)).
    # PyTorch requires 'initial_lr' on the param groups whenever last_epoch>=0
    # (normally restored via optimizer.load_state_dict(); this script always
    # builds a fresh optimizer, even on --resume, so it's set explicitly).
    for _g in optimizer.param_groups:
        _g.setdefault("initial_lr", _g["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch=start_iter - 1)

    grad_clip = train_cfg.get("grad_clip", 1.0)
    # Smoothness penalty on z (Hutchinson VJP, see inr_backbone.py::meta_train_step)
    # — 0.0 (default) leaves training identical to before this option existed.
    # Ramped 0 -> z_reg_weight over the first z_reg_warmup_frac of training
    # (not applied full-strength from step 0): this training loop is already
    # known to be sensitive to early instability (see meta_train_step's
    # docstring on the grad-clip fix), and a new loss term appearing abruptly
    # is exactly the kind of change that's cheap to make safer this way.
    z_reg_weight = float(train_cfg.get("z_reg_weight", 0.0))
    z_reg_warmup_frac = float(train_cfg.get("z_reg_warmup_frac", 0.1))
    z_reg_warmup_iters = max(1, int(total_iters * z_reg_warmup_frac))

    t0 = time.time()
    last_log_t = t0
    recent_losses: List[float] = []
    recent_grad_norms: List[float] = []
    recent_jac_penalties: List[float] = []

    for step in range(start_iter, total_iters):
        batch = next(data_iter)[0].to(device)  # (B, 1, H, W, D)
        cur_z_reg_weight = z_reg_weight * min(1.0, step / z_reg_warmup_iters)
        loss, grad_norm, jac_penalty = meta_train_step(
            backbone, batch, coords_full, optimizer, points_per_step, grad_clip=grad_clip,
            z_reg_weight=cur_z_reg_weight,
        )
        scheduler.step()
        ema.update(backbone)
        recent_losses.append(loss)
        recent_grad_norms.append(grad_norm)
        recent_jac_penalties.append(jac_penalty)
        if len(recent_losses) > print_every:
            recent_losses.pop(0)
            recent_grad_norms.pop(0)
            recent_jac_penalties.pop(0)

        if (step + 1) % print_every == 0:
            avg = float(np.mean(recent_losses))
            avg_grad = float(np.mean(recent_grad_norms))
            max_grad = float(np.max(recent_grad_norms))
            avg_jac = float(np.mean(recent_jac_penalties))
            lr_cur = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            win_dt = time.time() - last_log_t
            it_s = print_every / max(win_dt, 1e-9)
            eta_s = (total_iters - step - 1) / max(it_s, 1e-9)
            print(
                f"[{step + 1:6d}/{total_iters}] loss={avg:.5f} grad={avg_grad:.2f}(max={max_grad:.2f}) "
                f"jac_pen={avg_jac:.5f}(w={cur_z_reg_weight:.2e}) "
                f"lr={lr_cur:.2e} speed={it_s:.2f} it/s eta={eta_s / 3600:.2f}h t={elapsed / 60:.1f}min",
                flush=True,
            )
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "iter": step + 1, "loss": round(avg, 6), "grad_norm": round(avg_grad, 4),
                    "grad_norm_max": round(max_grad, 4), "jac_penalty": round(avg_jac, 6),
                    "z_reg_weight": round(cur_z_reg_weight, 8), "lr": round(lr_cur, 8),
                    "elapsed_s": round(elapsed, 1),
                }) + "\n")
            last_log_t = time.time()

        if (step + 1) % save_every == 0:
            ckpt_path = weights_dir / f"checkpoint_{step + 1}.pth"
            _save_checkpoint(ckpt_path, step, backbone, ema, optimizer, scheduler, args.config)
            print(f"  → Checkpoint : {ckpt_path}", flush=True)

    final_path = weights_dir / "model_final.pth"
    _save_checkpoint(final_path, total_iters - 1, backbone, ema, optimizer, scheduler, args.config)
    print(f"\nEntraînement terminé. Backbone final : {final_path}", flush=True)


if __name__ == "__main__":
    main()
