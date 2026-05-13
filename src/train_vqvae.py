#!/usr/bin/env python3
"""
VQ-VAE 3D multimodal (paired + unpaired) pour MRIxFields.

Objectif:
- Conserver les briques principales de NeuroQuant (dual-stream, VQ EMA, FiLM, adversary)
- Ajouter un entraînement hybride pour exploiter:
  1) données unpaired: reconstruction intra-modale
  2) données paired: reconstruction cross-modale supervisée

Ce script est volontairement autonome pour tester l'approche avant intégration CFM.
"""

import argparse
import gc
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as scipy_zoom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint as torch_checkpoint


MODALITIES = ["T1W", "T2W", "T2FLAIR"]
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
SPLIT_MAP = {
    "retro_train": "Training_retrospective",
    "pro_train": "Training_prospective",
    "pro_val": "Validating_prospective",
    "pro_test": "Testing_prospective",
}
FILE_RE = re.compile(r"^[A-Z]_([A-Z0-9]+)_([0-9.]+T)_(\d+)\.nii\.gz$")


def _resample_volume(vol: np.ndarray, original_spacing, target_spacing: Tuple[float, float, float]) -> np.ndarray:
    orig = np.asarray(original_spacing[:3], dtype=float)
    tgt = np.asarray(target_spacing, dtype=float)
    factors = orig / tgt
    if np.allclose(factors, 1.0, atol=0.02):
        return vol.astype(np.float32)
    return scipy_zoom(vol, factors, order=1).astype(np.float32)


def _normalize(vol: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    lo = np.percentile(vol, lo_pct)
    hi = np.percentile(vol, hi_pct)
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return (vol * 2.0 - 1.0).astype(np.float32)


def _center_crop_or_pad(vol: np.ndarray, size: Tuple[int, int, int]) -> np.ndarray:
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


@dataclass(frozen=True)
class SampleMeta:
    path: Path
    split: str
    modality: str
    field: str
    subject_id: str


class MRIxFieldsHybridDataset(Dataset):
    """Dataset hybride pour entraînement paired/unpaired.

    - Retourne toujours une source x_src
    - Peut retourner une cible paired x_tgt si disponible et tirée
    """

    def __init__(
        self,
        data_root: Path,
        splits: Sequence[str],
        modalities: Sequence[str],
        fields: Sequence[str],
        volume_size: Tuple[int, int, int],
        paired_prob: float = 0.5,
        percentile_lower: float = 0.5,
        percentile_upper: float = 99.5,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.volume_size = volume_size
        self.paired_prob = paired_prob
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.target_spacing = target_spacing

        self.samples: List[SampleMeta] = []
        self.by_key: Dict[Tuple[str, str, str], Dict[str, SampleMeta]] = {}

        for split in splits:
            split_dir = SPLIT_MAP.get(split, split)
            for modality in modalities:
                for field in fields:
                    d = self.data_root / split_dir / modality / field
                    for p in sorted(d.glob("*.nii.gz")):
                        m = FILE_RE.match(p.name)
                        if m is None:
                            continue
                        subj = m.group(3)
                        meta = SampleMeta(path=p, split=split, modality=modality, field=field, subject_id=subj)
                        self.samples.append(meta)
                        key = (split, field, subj)
                        if key not in self.by_key:
                            self.by_key[key] = {}
                        self.by_key[key][modality] = meta

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if not self.samples:
            raise FileNotFoundError("Aucun fichier NIfTI détecté pour les paramètres fournis.")

        self.mod_to_idx = {m: i for i, m in enumerate(modalities)}
        self.field_to_idx = {f: i for i, f in enumerate(fields)}

        n_pairable = 0
        for s in self.samples:
            if len(self.by_key[(s.split, s.field, s.subject_id)]) > 1:
                n_pairable += 1
        print(f"Dataset: {len(self.samples)} samples | pairables: {n_pairable}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_tensor(self, meta: SampleMeta) -> torch.Tensor:
        img = nib.load(str(meta.path))
        vol = img.get_fdata(dtype=np.float32)
        if self.target_spacing is not None:
            spacing = np.abs(np.diag(img.affine)[:3])
            vol = _resample_volume(vol, spacing, self.target_spacing)
        vol = _normalize(vol, self.percentile_lower, self.percentile_upper)
        vol = _center_crop_or_pad(vol, self.volume_size)
        return torch.from_numpy(vol).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src = self.samples[idx]
        x_src = self._load_tensor(src)
        src_mod_idx = self.mod_to_idx[src.modality]
        src_field_idx = self.field_to_idx[src.field]

        candidates = self.by_key[(src.split, src.field, src.subject_id)]
        other_mods = [m for m in candidates.keys() if m != src.modality]

        is_paired = bool(other_mods) and (random.random() < self.paired_prob)

        if is_paired:
            tgt_mod = random.choice(other_mods)
            tgt = candidates[tgt_mod]
            x_tgt = self._load_tensor(tgt)
            tgt_mod_idx = self.mod_to_idx[tgt.modality]
            tgt_field_idx = self.field_to_idx[tgt.field]
        else:
            x_tgt = torch.zeros_like(x_src)
            tgt_mod_idx = -1
            tgt_field_idx = -1

        return {
            "x_src": x_src,
            "x_tgt": x_tgt,
            "src_mod": torch.tensor(src_mod_idx, dtype=torch.long),
            "src_field": torch.tensor(src_field_idx, dtype=torch.long),
            "tgt_mod": torch.tensor(tgt_mod_idx, dtype=torch.long),
            "tgt_field": torch.tensor(tgt_field_idx, dtype=torch.long),
            "is_paired": torch.tensor(1 if is_paired else 0, dtype=torch.float32),
        }


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, decay: float = 0.99, eps: float = 1e-5, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps
        self.beta = beta

        # Match NeuroQuant-style small codebook init to avoid early distance explosions.
        embed = torch.empty(num_embeddings, embedding_dim)
        embed.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_count", torch.zeros(num_embeddings))
        self.register_buffer("ema_weight", embed.clone())

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: (B,C,H,W,D)
        b, c, h, w, d = z_e.shape
        # Compute VQ assignment in fp32 for numerical stability under AMP.
        z_flat = z_e.float().permute(0, 2, 3, 4, 1).contiguous().view(-1, c)
        embedding_fp32 = self.embedding.float()

        # distances L2
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ embedding_fp32.t()
            + embedding_fp32.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = torch.argmin(dist, dim=1)
        z_q = embedding_fp32.index_select(0, indices)
        z_q = z_q.view(b, h, w, d, c).permute(0, 4, 1, 2, 3).contiguous()
        z_q = z_q.to(dtype=z_e.dtype)

        if self.training:
            onehot = F.one_hot(indices, self.num_embeddings).type_as(z_flat)
            count = onehot.sum(dim=0)
            weight = onehot.t() @ z_flat

            self.ema_count.mul_(self.decay).add_(count, alpha=1 - self.decay)
            self.ema_weight.mul_(self.decay).add_(weight, alpha=1 - self.decay)

            n = self.ema_count.sum()
            smoothed = (self.ema_count + self.eps) / (n + self.num_embeddings * self.eps) * n
            smoothed = smoothed.clamp_min(1e-6)
            self.embedding.copy_(self.ema_weight / smoothed.unsqueeze(1))

        # straight-through + commitment
        z_q_st = z_e + (z_q - z_e).detach()
        commit_loss = self.beta * F.mse_loss(z_e.float(), z_q.detach().float())

        probs = F.one_hot(indices, self.num_embeddings).float().mean(dim=0)
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())

        return z_q_st, commit_loss, perplexity


class ConvBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, min(8, cout // 4)), num_channels=cout),
            nn.SiLU(inplace=True),
            nn.Conv3d(cout, cout, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, min(8, cout // 4)), num_channels=cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DualStreamEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        anat_channels: int = 64,
        mod_channels: int = 32,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        channels = [base_channels * m for m in channel_multipliers]
        if len(channels) < 2:
            raise ValueError("channel_multipliers must contain at least 2 values")

        self.stem = ConvBlock3D(in_channels, channels[0])

        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels:
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.SiLU(inplace=True),
                    nn.Conv3d(out_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.SiLU(inplace=True),
                )
            )
            in_ch = out_ch

        hidden = channels[-1]
        self.anat_head = nn.Conv3d(hidden, anat_channels, kernel_size=1)
        self.mod_head = nn.Conv3d(hidden, mod_channels, kernel_size=1)

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing and x.requires_grad:
            return torch_checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._run_block(self.stem, x)
        for block in self.down_blocks:
            h = self._run_block(block, h)
        z_anat = self.anat_head(h)
        z_mod = self.mod_head(h)
        return z_anat, z_mod


class FiLMDecoder(nn.Module):
    def __init__(
        self,
        anat_channels: int,
        mod_channels: int,
        n_modalities: int,
        n_fields: int,
        base_channels: int = 32,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        out_channels: int = 1,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.mod_emb = nn.Embedding(n_modalities, 16)
        self.field_emb = nn.Embedding(n_fields, 8)

        channels = [base_channels * m for m in channel_multipliers]
        if len(channels) < 2:
            raise ValueError("channel_multipliers must contain at least 2 values")
        rev_channels = list(reversed(channels))

        style_in = mod_channels + 16 + 8
        film_total = 2 * sum(rev_channels)
        self.style_mlp = nn.Sequential(
            nn.Linear(style_in, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, film_total),
        )

        self.up_blocks = nn.ModuleList()
        in_ch = anat_channels
        for out_ch in rev_channels:
            self.up_blocks.append(nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            in_ch = out_ch
        self.film_channels = rev_channels
        self.out = nn.Conv3d(rev_channels[-1], out_channels, kernel_size=3, padding=1)

    def _film(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # gamma/beta: (B,C)
        return x * (1.0 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing and x.requires_grad:
            return torch_checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, z_q: torch.Tensor, z_mod: torch.Tensor, mod_idx: torch.Tensor, field_idx: torch.Tensor) -> torch.Tensor:
        z_mod_pool = z_mod.mean(dim=(2, 3, 4))
        style = torch.cat([z_mod_pool, self.mod_emb(mod_idx), self.field_emb(field_idx)], dim=1)
        gb = self.style_mlp(style)

        split_sizes: List[int] = []
        for c in self.film_channels:
            split_sizes.extend([c, c])
        chunks = torch.split(gb, split_sizes, dim=1)

        x = z_q
        offset = 0
        for block, c in zip(self.up_blocks, self.film_channels):
            gamma = chunks[offset]
            beta = chunks[offset + 1]
            offset += 2
            x = self._run_block(block, x)
            if gamma.shape[1] != c or beta.shape[1] != c:
                raise RuntimeError("FiLM parameter shape mismatch")
            x = F.silu(self._film(x, gamma, beta))
        x = torch.tanh(self.out(x))
        return x


class ModalityAdversary(nn.Module):
    def __init__(self, anat_channels: int, n_modalities: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(anat_channels, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, n_modalities),
        )

    def forward(self, z_anat: torch.Tensor, grl_lambda: float) -> torch.Tensor:
        pooled = z_anat.mean(dim=(2, 3, 4))
        pooled = grad_reverse(pooled, grl_lambda)
        return self.head(pooled)


class NeuroQuantHybrid(nn.Module):
    def __init__(
        self,
        n_modalities: int,
        n_fields: int,
        base_channels: int = 32,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        anat_channels: int = 64,
        mod_channels: int = 32,
        codebook_size: int = 1024,
        vq_decay: float = 0.99,
        vq_beta: float = 0.25,
    ):
        super().__init__()
        self.encoder = DualStreamEncoder(
            in_channels=1,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            anat_channels=anat_channels,
            mod_channels=mod_channels,
        )
        self.quantizer = EMAVectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=anat_channels,
            decay=vq_decay,
            beta=vq_beta,
        )
        self.decoder = FiLMDecoder(
            anat_channels=anat_channels,
            mod_channels=mod_channels,
            n_modalities=n_modalities,
            n_fields=n_fields,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            out_channels=1,
        )
        self.adversary = ModalityAdversary(anat_channels=anat_channels, n_modalities=n_modalities)

    def enable_gradient_checkpointing(self) -> None:
        self.encoder.gradient_checkpointing = True
        self.decoder.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.encoder.gradient_checkpointing = False
        self.decoder.gradient_checkpointing = False

    def forward_src(self, x_src: torch.Tensor, src_mod: torch.Tensor, src_field: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_anat, z_mod = self.encoder(x_src)
        z_q, vq_loss, perplexity = self.quantizer(z_anat)
        x_rec = self.decoder(z_q, z_mod, src_mod, src_field)
        return {
            "z_anat": z_anat,
            "z_mod": z_mod,
            "z_q": z_q,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "x_rec": x_rec,
        }


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = args.use_amp and device.type == "cuda"
    amp_dtype = torch.float16

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    ds = MRIxFieldsHybridDataset(
        data_root=Path(args.data_root),
        splits=args.splits,
        modalities=args.modalities,
        fields=args.fields,
        volume_size=tuple(args.volume_size),
        paired_prob=args.paired_prob,
        percentile_lower=args.percentile_lower,
        percentile_upper=args.percentile_upper,
        target_spacing=tuple(args.target_spacing) if args.target_spacing else None,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = NeuroQuantHybrid(
        n_modalities=len(args.modalities),
        n_fields=len(args.fields),
        base_channels=args.base_channels,
        channel_multipliers=tuple(args.channel_multipliers),
        anat_channels=args.anat_channels,
        mod_channels=args.mod_channels,
        codebook_size=args.codebook_size,
        vq_decay=args.vq_decay,
        vq_beta=args.vq_beta,
    ).to(device)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Grad checkpointing: {args.gradient_checkpointing}")
    print(f"Batch size: {args.batch_size} | Steps: {args.steps}")
    print(f"Output: {out_dir}")

    step = 0
    t0 = time.time()

    def _linear_ramp(step_id: int, start: int, ramp_steps: int, max_val: float) -> float:
        if step_id < start:
            return 0.0
        if ramp_steps <= 0:
            return float(max_val)
        p = min(1.0, (step_id - start) / max(1, ramp_steps))
        return float(max_val * p)

    def _grl_sigmoid(step_id: int, start: int, total_steps: int, max_alpha: float) -> float:
        if step_id < start:
            return 0.0
        p = min(1.0, (step_id - start) / max(1, total_steps - start))
        return float(max_alpha * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            step += 1

            batch = _to_device(batch, device)
            x_src = batch["x_src"]
            x_tgt = batch["x_tgt"]
            src_mod = batch["src_mod"]
            src_field = batch["src_field"]
            tgt_mod = batch["tgt_mod"]
            tgt_field = batch["tgt_field"]
            is_paired = batch["is_paired"]

            model.train()
            opt.zero_grad(set_to_none=True)

            cross_w = _linear_ramp(step, args.cross_start_step, args.cross_ramp_steps, args.lambda_cross)
            adv_w = _linear_ramp(step, args.adv_start_step, args.adv_ramp_steps, args.lambda_adv)
            grl_lambda = _grl_sigmoid(step, args.adv_start_step, args.steps, args.adv_grl_lambda)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out_src = model.forward_src(x_src, src_mod, src_field)
                x_rec = out_src["x_rec"]
                recon_loss = F.l1_loss(x_rec, x_src)
                vq_loss = out_src["vq_loss"]

                # Adversary modalité sur code anatomique (invariance)
                logits_adv = model.adversary(out_src["z_anat"], grl_lambda=grl_lambda)
                adv_loss = F.cross_entropy(logits_adv, src_mod)

                # Cross loss (paired uniquement)
                paired_mask = is_paired > 0.5
                if paired_mask.any():
                    z_mod_tgt = torch.zeros_like(out_src["z_mod"])
                    with torch.no_grad():
                        _, z_mod_all_tgt = model.encoder(x_tgt[paired_mask])
                    z_mod_tgt[paired_mask] = z_mod_all_tgt

                    # Placeholder indices sûrs pour unpaired (pas utilisés par le masque)
                    safe_tgt_mod = tgt_mod.clone()
                    safe_tgt_field = tgt_field.clone()
                    safe_tgt_mod[~paired_mask] = src_mod[~paired_mask]
                    safe_tgt_field[~paired_mask] = src_field[~paired_mask]

                    x_cross = model.decoder(out_src["z_q"], z_mod_tgt, safe_tgt_mod, safe_tgt_field)
                    cross_loss = F.l1_loss(x_cross[paired_mask], x_tgt[paired_mask])
                else:
                    cross_loss = torch.zeros([], device=device)

                total = (
                    args.lambda_recon * recon_loss
                    + args.lambda_vq * vq_loss
                    + adv_w * adv_loss
                    + cross_w * cross_loss
                )

            if not torch.isfinite(total):
                print(
                    f"[WARN] step={step} non-finite loss detected "
                    f"(total={float(total.detach().cpu().item())}, "
                    f"recon={float(recon_loss.detach().cpu().item())}, "
                    f"vq={float(vq_loss.detach().cpu().item())}, "
                    f"adv={float(adv_loss.detach().cpu().item())}, "
                    f"cross={float(cross_loss.detach().cpu().item())}). "
                    "Skipping optimizer step."
                )
                opt.zero_grad(set_to_none=True)
                continue

            scaler.scale(total).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            # Periodic memory cleanup to prevent CUDA OOM during long training runs
            if step % 50 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if step % args.print_every == 0 or step == 1:
                elapsed = time.time() - t0
                paired_ratio = float(is_paired.mean().item())
                print(
                    f"[{step:5d}/{args.steps}] "
                    f"loss={float(total.item()):.4f} "
                    f"recon={float(recon_loss.item()):.4f} "
                    f"vq={float(vq_loss.item()):.4f} "
                    f"adv={float(adv_loss.item()):.4f} "
                    f"cross={float(cross_loss.item()):.4f} "
                    f"w_adv={adv_w:.4f} "
                    f"w_cross={cross_w:.4f} "
                    f"grl={grl_lambda:.3f} "
                    f"paired={paired_ratio:.2f} "
                    f"ppl={float(out_src['perplexity'].item()):.1f} "
                    f"t={elapsed/60:.1f}min"
                )

            if step % args.save_every == 0 or step == args.steps:
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "args": vars(args),
                }
                ckpt_path = weights_dir / f"vqvae_step_{step:06d}.pth"
                torch.save(ckpt, ckpt_path)
                print(f"  -> checkpoint: {ckpt_path}")
                
                # Clean up after checkpoint save
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    final_path = weights_dir / "vqvae_final.pth"
    torch.save({"step": step, "model": model.state_dict(), "optimizer": opt.state_dict(), "args": vars(args)}, final_path)
    print(f"Training terminé. Modèle final: {final_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VQ-VAE 3D hybride paired/unpaired (MRIxFields)")
    p.add_argument("--data-root", type=str, default="/home/rousseau/Data/MRIxFields_20260414")
    p.add_argument("--output-dir", type=str, default="outputs/vqvae3d/runs/vqvae3d_hybrid")

    p.add_argument("--splits", nargs="+", default=["retro_train"])
    p.add_argument("--modalities", nargs="+", default=MODALITIES)
    p.add_argument("--fields", nargs="+", default=FIELDS)

    p.add_argument("--volume-size", nargs=3, type=int, default=[112, 128, 80])
    p.add_argument("--target-spacing", nargs=3, type=float, default=None)
    p.add_argument("--percentile-lower", type=float, default=0.5)
    p.add_argument("--percentile-upper", type=float, default=99.5)
    p.add_argument("--paired-prob", type=float, default=0.5)
    p.add_argument("--max-samples", type=int, default=None)

    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--channel-multipliers", nargs="+", type=int, default=[1, 2, 4, 4])
    p.add_argument("--anat-channels", type=int, default=64)
    p.add_argument("--mod-channels", type=int, default=32)
    p.add_argument("--codebook-size", type=int, default=1024)
    p.add_argument("--vq-decay", type=float, default=0.99)
    p.add_argument("--vq-beta", type=float, default=0.25)

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--lambda-recon", type=float, default=1.0)
    p.add_argument("--lambda-vq", type=float, default=1.0)
    p.add_argument("--lambda-adv", type=float, default=1e-3)
    p.add_argument("--lambda-cross", type=float, default=0.5)
    p.add_argument("--adv-grl-lambda", type=float, default=0.5)
    p.add_argument("--cross-start-step", type=int, default=500)
    p.add_argument("--cross-ramp-steps", type=int, default=1000)
    p.add_argument("--adv-start-step", type=int, default=1500)
    p.add_argument("--adv-ramp-steps", type=int, default=1000)

    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
