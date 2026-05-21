#!/usr/bin/env python3
"""MMFM v1 baseline for MRIxFields.

This script keeps MedVAE unchanged and vectorizes the MedVAE latent before the
flow model. The core flow model is a small residual MLP operating on flattened
latents, which makes the implementation faithful to the original MMFM idea
without introducing a convolutional latent backbone.

Pipeline:
  1. 3D NIfTI volume
  2. MedVAE encode -> latent tensor
  3. Flatten latent tensor into a single vector
  4. Conditional flow matching in vector space
  5. Predict vector field
  6. Unflatten the predicted vector back to latent tensor shape
  7. Decode with MedVAE

The training setup keeps the existing 15 domain classes (3 modalities x 5
fields) as discrete target conditions so the baseline remains comparable to the
previous multimodal prototype.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from cfm.mmfm_vectorized import LatentVectorizer, VectorMMFM  # noqa: E402
from cfm.train_cfm_3d import (  # noqa: E402
    EMAModel,
    _load_env,
    _make_infinite,
    _resolve_paths,
    _resample_volume,
    is_main_process,
    load_vae,
)


MODALITIES: List[str] = ["T1W", "T2W", "T2FLAIR"]
FIELDS: List[str] = ["0.1T", "1.5T", "3T", "5T", "7T"]
SPLIT_MAP = {
    "retro_train": "Training_retrospective",
    "pro_train": "Training_prospective",
    "pro_val": "Validating_prospective",
    "pro_test": "Testing_prospective",
}
FILE_RE = re.compile(r"^[A-Z]_([A-Z0-9]+)_([0-9.]+T)_(\d+)\.nii\.gz$")


def _flat_class(mod_idx: int, field_idx: int, n_fields: int) -> int:
    return mod_idx * n_fields + field_idx


def _center_crop_or_pad_np(vol: np.ndarray, size: Tuple[int, int, int]) -> np.ndarray:
    th, tw, td = size
    h, w, d = vol.shape

    ph = max(0, th - h)
    pw = max(0, tw - w)
    pd = max(0, td - d)
    if ph > 0 or pw > 0 or pd > 0:
        vol = np.pad(
            vol,
            [(ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (pd // 2, pd - pd // 2)],
            mode="reflect",
        )
        h, w, d = vol.shape

    sh = max((h - th) // 2, 0)
    sw = max((w - tw) // 2, 0)
    sd = max((d - td) // 2, 0)
    return vol[sh : sh + th, sw : sw + tw, sd : sd + td]


class MultiModalNIfTILatentDataset(Dataset):
    """Multi-modal / multi-field dataset for MMFM v1."""

    def __init__(
        self,
        data_root: Path,
        split: str,
        modalities: List[str],
        fields: List[str],
        percentile_lower: float = 0.5,
        percentile_upper: float = 99.5,
        max_per_class: Optional[int] = None,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        volume_size: Optional[Tuple[int, int, int]] = None,
    ):
        self.modalities = modalities
        self.fields = fields
        self.mod_to_idx = {m: i for i, m in enumerate(modalities)}
        self.field_to_idx = {f: i for i, f in enumerate(fields)}
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.target_spacing = target_spacing
        self.volume_size = volume_size

        self.samples: List[Tuple[Path, int, int, int]] = []

        split_dir = SPLIT_MAP.get(split, split)
        for modality in modalities:
            for field in fields:
                class_files = sorted((Path(data_root) / split_dir / modality / field).glob("*.nii.gz"))
                if max_per_class is not None:
                    class_files = class_files[:max_per_class]
                m_idx = self.mod_to_idx[modality]
                f_idx = self.field_to_idx[field]
                c_idx = _flat_class(m_idx, f_idx, len(fields))
                for p in class_files:
                    if FILE_RE.match(p.name) is None:
                        continue
                    self.samples.append((p, m_idx, f_idx, c_idx))

        if not self.samples:
            raise FileNotFoundError("Aucun volume NIfTI trouvé pour les classes MMFM.")

        counts: Dict[int, int] = {}
        for _, _, _, c in self.samples:
            counts[c] = counts.get(c, 0) + 1
        print(
            f"MultiModalNIfTILatentDataset: {len(self.samples)} volumes | "
            f"classes présentes={len(counts)}/{len(modalities) * len(fields)}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize(self, vol: np.ndarray) -> np.ndarray:
        lo = np.percentile(vol, self.percentile_lower)
        hi = np.percentile(vol, self.percentile_upper)
        vol = np.clip((vol - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
        return vol * 2.0 - 1.0

    def __getitem__(self, idx: int):
        path, mod_idx, field_idx, class_idx = self.samples[idx]
        img = nib.load(str(path))
        vol = img.get_fdata(dtype=np.float32)

        if self.target_spacing is not None:
            spacing = np.abs(np.diag(img.affine)[:3])
            vol = _resample_volume(vol, spacing, self.target_spacing)

        vol = self._normalize(vol)
        if self.volume_size is not None:
            vol = _center_crop_or_pad_np(vol, self.volume_size)

        x = torch.from_numpy(vol).unsqueeze(0)
        return (
            x,
            torch.tensor(mod_idx, dtype=torch.long),
            torch.tensor(field_idx, dtype=torch.long),
            torch.tensor(class_idx, dtype=torch.long),
        )


def _infer_latent_shape(vae, volume_size: Tuple[int, int, int], device: torch.device) -> Tuple[int, ...]:
    dummy = torch.zeros(1, 1, *volume_size, device=device)
    with torch.no_grad():
        z = vae.encode(dummy)
    return tuple(int(v) for v in z.shape[1:])


def build_vector_mmfm(cfg: dict, latent_dim: int, n_classes: int) -> VectorMMFM:
    m = cfg["model"]
    return VectorMMFM(
        latent_dim=latent_dim,
        num_classes=n_classes,
        hidden_dim=int(m.get("hidden_dim", 1024)),
        depth=int(m.get("num_blocks", 4)),
        time_embed_dim=int(m.get("time_embed_dim", 256)),
        class_embed_dim=int(m.get("class_embed_dim", 128)),
        dropout=float(m.get("dropout", 0.0)),
    )


def _save_checkpoint(
    path: Path,
    step: int,
    model: torch.nn.Module,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    use_scaler: bool,
    cfg_path: str,
    latent_shape: Tuple[int, ...],
) -> None:
    torch.save(
        {
            "iter": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_scaler else None,
            "cfg_path": str(cfg_path),
            "latent_shape": latent_shape,
        },
        path,
    )


def train(cfg_path: str, env_path: Optional[str] = None, resume: Optional[str] = None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_paths(cfg, _load_env(env_path))
    if resume is not None:
        cfg["resume"] = resume

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    data_root = data_cfg.get("data_root")
    if data_root is None:
        raise RuntimeError("data_root requis dans config/env")

    modalities = data_cfg.get("modalities", MODALITIES)
    fields = data_cfg.get("fields", FIELDS)

    output_dir = Path(data_cfg["output_dir"])
    split = data_cfg.get("split", "retro_train")
    p_lo = data_cfg.get("percentile_lower", 0.5)
    p_hi = data_cfg.get("percentile_upper", 99.5)
    max_per_class = data_cfg.get("max_volumes_per_class", None)

    raw_vs = data_cfg.get("volume_size", None)
    if raw_vs is None:
        raise RuntimeError("volume_size est requis pour la baseline vectorisée MMFM v1.")
    volume_size = tuple(int(v) for v in raw_vs)

    raw_ts = data_cfg.get("target_spacing", None)
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else None

    total_iters = int(train_cfg.get("total_iters", 10000))
    batch_size = int(train_cfg.get("batch_size", 1))
    num_workers = int(train_cfg.get("num_workers", 4))
    lr = float(train_cfg.get("lr", 1e-4))
    sigma = float(train_cfg.get("sigma", 0.0))
    ot_method = train_cfg.get("ot_method", "exact")
    save_every = int(train_cfg.get("save_every", 2000))
    print_every = int(train_cfg.get("print_every", 100))
    use_amp = bool(train_cfg.get("use_amp", True))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    ema_decay = float(train_cfg.get("ema_decay", 0.9999))
    amp_dtype_name = train_cfg.get("amp_dtype", "bf16")
    num_targets_per_step = int(train_cfg.get("num_targets_per_step", 2))

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
    n_classes = len(modalities) * len(fields)

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
        raise RuntimeError("Il faut au moins 2 classes modalite/champ non vides.")

    vae = load_vae(cfg, device)
    latent_shape = _infer_latent_shape(vae, volume_size, device)
    vectorizer = LatentVectorizer(latent_shape)
    latent_dim = vectorizer.flat_dim

    mmfm = build_vector_mmfm(cfg, latent_dim, n_classes).to(device)
    if is_distributed:
        mmfm = DDP(mmfm, device_ids=[local_rank])
    raw_mmfm = mmfm.module if is_distributed else mmfm

    ema = EMAModel(raw_mmfm, decay=ema_decay)
    optimizer = torch.optim.AdamW(mmfm.parameters(), lr=lr, weight_decay=1e-4)

    def _lr_lambda(step: int) -> float:
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
        raw_mmfm.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "ema" in state:
            ema.load_state_dict(state["ema"])
        if "scaler" in state and use_scaler:
            scaler.load_state_dict(state["scaler"])
        start_iter = state.get("iter", 0) + 1
        if is_main_process():
            print(f"Reprise depuis iter {start_iter}: {resume_path}")

    if is_main_process():
        n_params = sum(p.numel() for p in raw_mmfm.parameters() if p.requires_grad)
        print(
            f"Vector MMFM: {n_params/1e6:.1f}M params | classes={n_classes} | "
            f"latent_shape={latent_shape} | flat_dim={latent_dim}"
        )
        print(
            f"Training MMFM v1: iters={total_iters} batch={batch_size} "
            f"targets/step={num_targets_per_step} amp={use_amp} dtype={amp_dtype_name}"
        )

    weights_dir = output_dir / "weights"
    t0 = time.time()
    last_log_t = t0
    recent_losses: List[float] = []
    mmfm.train()

    for step in range(start_iter, total_iters):
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

            with torch.no_grad(), torch.amp.autocast(
                "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
            ):
                z_src = vae.encode(src_x)
                z_tgt = vae.encode(tgt_x)

            z_src_vec = vectorizer.flatten(z_src)
            z_tgt_vec = vectorizer.flatten(z_tgt)
            t_batch, z_t, ut = FM.sample_location_and_conditional_flow(z_src_vec, z_tgt_vec)
            t_vec = t_batch.to(device).float().reshape(z_src_vec.shape[0], -1).squeeze(-1)
            y_tgt = torch.full((z_src_vec.shape[0],), tgt_class, dtype=torch.long, device=device)

            with torch.amp.autocast(
                "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
            ):
                v_t = raw_mmfm(z_t, z_src_vec, t_vec, y_tgt)
                loss = F.mse_loss(v_t, ut) / float(k)

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            step_losses.append(float(loss.item() * k))

        if use_scaler:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_mmfm.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_mmfm.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        ema.update(raw_mmfm)

        mean_step_loss = float(np.mean(step_losses)) if step_losses else 0.0
        recent_losses.append(mean_step_loss)
        if len(recent_losses) > print_every:
            recent_losses.pop(0)

        if is_main_process() and (step + 1) % print_every == 0:
            avg_recent = float(np.mean(recent_losses))
            elapsed = time.time() - t0
            win_dt = time.time() - last_log_t
            it_s = print_every / max(win_dt, 1e-9)
            eta_s = (total_iters - step - 1) / max(it_s, 1e-9)
            lr_cur = scheduler.get_last_lr()[0]
            mem_gb = (
                torch.cuda.max_memory_allocated(device) / (1024**3)
                if device.type == "cuda"
                else 0.0
            )
            print(
                f"[{step+1:6d}/{total_iters}] loss={avg_recent:.4f} grad={float(grad_norm):.2f} "
                f"lr={lr_cur:.2e} src={src_class} tgts={tgt_classes} speed={it_s:.2f} it/s "
                f"eta={eta_s/3600:.2f}h t={elapsed/60:.1f}min mem={mem_gb:.1f}GB"
            )
            last_log_t = time.time()

        if is_main_process() and (step + 1) % save_every == 0:
            ckpt_path = weights_dir / f"checkpoint_{step+1}.pth"
            _save_checkpoint(
                ckpt_path,
                step,
                raw_mmfm,
                ema,
                optimizer,
                scaler,
                use_scaler,
                cfg_path,
                latent_shape,
            )
            print(f"  -> Checkpoint: {ckpt_path}")

    if is_main_process():
        final_path = weights_dir / "model_final.pth"
        _save_checkpoint(
            final_path,
            total_iters - 1,
            raw_mmfm,
            ema,
            optimizer,
            scaler,
            use_scaler,
            cfg_path,
            latent_shape,
        )
        print(f"Training MMFM terminé. Modèle final: {final_path}")

    if is_distributed:
        dist.destroy_process_group()


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMFM v1 vectorized baseline")
    p.add_argument("--config", required=True)
    p.add_argument("--env", default=None)
    p.add_argument("--resume", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    train(args.config, env_path=args.env, resume=args.resume)
