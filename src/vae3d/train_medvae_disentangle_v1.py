#!/usr/bin/env python3
"""
MedVAE frozen disentanglement v1 (anatomie / modalite).

Principe:
- MedVAE encodeur/decodeur figes (poids non trainables)
- Projection latent anatomie z_a (spatiale)
- Projection latent modalite z_m (globale)
- Fusion conditionnelle (FiLM) vers un latent decodable par MedVAE

v1 scope:
- anatomie / modalite uniquement
- field conserve dans le batch mais non modelise explicitement
"""

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Imports locaux (meme pattern que benchmark_vae.py)
_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "vae3d"))

from vae3d.train_vqvae import MRIxFieldsHybridDataset, ssim3d


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


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _encode_medvae_deterministic(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    z = model.encode(x)
    if isinstance(z, (tuple, list)):
        # MedVAE peut retourner (mean, logvar, ...)
        z = z[0]
    return z


def _decode_medvae(model: nn.Module, z: torch.Tensor, out_shape: Tuple[int, int, int]) -> torch.Tensor:
    x = model.decode(z)
    if isinstance(x, (tuple, list)):
        x = x[0]
    if x.shape[2:] != out_shape:
        x = F.interpolate(x, size=out_shape, mode="trilinear", align_corners=False)
    return x


class MedVAEDisentanglerV1(nn.Module):
    def __init__(
        self,
        medvae: nn.Module,
        latent_channels: int,
        n_modalities: int,
        anat_channels: int = 8,
        style_dim: int = 32,
        film_hidden: int = 128,
    ):
        super().__init__()
        self.medvae = medvae
        self.latent_channels = latent_channels
        self.anat_channels = anat_channels
        self.style_dim = style_dim

        # Anatomie (spatial)
        self.anat_proj = nn.Sequential(
            nn.Conv3d(latent_channels, anat_channels, kernel_size=1),
            nn.GroupNorm(num_groups=max(1, min(8, anat_channels)), num_channels=anat_channels),
            nn.SiLU(),
        )

        # Modalite (global)
        self.style_proj = nn.Sequential(
            nn.Conv3d(latent_channels, style_dim, kernel_size=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(style_dim, style_dim),
            nn.SiLU(),
        )

        self.modality_embed = nn.Embedding(n_modalities, style_dim)
        self.film = nn.Sequential(
            nn.Linear(2 * style_dim, film_hidden),
            nn.SiLU(),
            nn.Linear(film_hidden, 2 * anat_channels),
        )

        # Retour vers latent MedVAE
        self.latent_reproj = nn.Conv3d(anat_channels, latent_channels, kernel_size=1)

        # Auxiliaires
        self.mod_classifier = nn.Linear(style_dim, n_modalities)
        self.adv_classifier = nn.Sequential(
            nn.Linear(anat_channels, anat_channels),
            nn.SiLU(),
            nn.Linear(anat_channels, n_modalities),
        )

    def encode_parts(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # medvae freeze: on ne calcule pas de grad pour limiter memoire/compute
        with torch.no_grad():
            z = _encode_medvae_deterministic(self.medvae, x)
        z_a = self.anat_proj(z)
        z_m = self.style_proj(z)
        return z, z_a, z_m

    def fuse_to_latent(self, z_a: torch.Tensor, z_m: torch.Tensor, mod_idx: torch.Tensor) -> torch.Tensor:
        emb = self.modality_embed(mod_idx)
        cond = torch.cat([z_m, emb], dim=1)
        gamma_beta = self.film(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        z_cond = (1.0 + gamma) * z_a + beta
        return self.latent_reproj(z_cond)

    def forward(self, x_src: torch.Tensor, src_mod: torch.Tensor, adv_alpha: float = 0.0):
        z_raw, z_a, z_m = self.encode_parts(x_src)
        z_hat = self.fuse_to_latent(z_a, z_m, src_mod)
        x_rec = _decode_medvae(self.medvae, z_hat, out_shape=x_src.shape[2:])

        mod_logits = self.mod_classifier(z_m)
        z_a_pool = F.adaptive_avg_pool3d(z_a, 1).flatten(1)
        adv_logits = self.adv_classifier(grad_reverse(z_a_pool, adv_alpha))

        return {
            "z_raw": z_raw,
            "z_a": z_a,
            "z_m": z_m,
            "x_rec": x_rec,
            "mod_logits": mod_logits,
            "adv_logits": adv_logits,
        }


def covariance_penalty(z_a: torch.Tensor, z_m: torch.Tensor) -> torch.Tensor:
    # z_a: (B, C_a, D,H,W) -> (B, C_a)
    a = F.adaptive_avg_pool3d(z_a, 1).flatten(1)
    m = z_m
    a = a - a.mean(dim=0, keepdim=True)
    m = m - m.mean(dim=0, keepdim=True)
    cov = (a.T @ m) / max(1, a.shape[0] - 1)
    return (cov.pow(2)).mean()


def linear_ramp(step: int, start_step: int, ramp_steps: int) -> float:
    if step < start_step:
        return 0.0
    return float(min(1.0, (step - start_step) / max(1, ramp_steps)))


def load_medvae(model_name: str, device: torch.device) -> nn.Module:
    from medvae import MVAE

    model = MVAE(model_name=model_name, modality="mri").to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = args.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

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

    medvae = load_medvae(args.medvae_model_name, device)

    # Infere la taille des canaux latents MedVAE
    with torch.no_grad():
        dummy = torch.zeros(1, 1, *args.volume_size, device=device)
        z_dummy = _encode_medvae_deterministic(medvae, dummy)
        latent_ch = int(z_dummy.shape[1])

    model = MedVAEDisentanglerV1(
        medvae=medvae,
        latent_channels=latent_ch,
        n_modalities=len(args.modalities),
        anat_channels=args.anat_channels,
        style_dim=args.style_dim,
        film_hidden=args.film_hidden,
    ).to(device)

    # Optimiser uniquement les tetes trainables (MedVAE est fige)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    print(f"Device={device} | AMP={use_amp}({args.amp_dtype})")
    print(f"MedVAE frozen={args.medvae_model_name} | latent_ch={latent_ch}")
    print(f"Output={out_dir}")

    step = 0
    t0 = time.time()
    best_loss = float("inf")

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            step += 1

            batch = _to_device(batch, device)
            x_src = batch["x_src"]
            x_tgt = batch["x_tgt"]
            src_mod = batch["src_mod"]
            tgt_mod = batch["tgt_mod"]
            is_paired = batch["is_paired"]

            opt.zero_grad(set_to_none=True)

            adv_alpha = linear_ramp(step, args.adv_start_step, args.adv_ramp_steps)
            cross_alpha = linear_ramp(step, args.cross_start_step, args.cross_ramp_steps)
            paired_alpha = linear_ramp(step, args.paired_start_step, args.paired_ramp_steps)

            lambda_cross_eff = args.lambda_cross * cross_alpha
            lambda_anat_eff = args.lambda_anat * paired_alpha
            lambda_latent_eff = args.lambda_latent * paired_alpha
            lambda_tgt_mod_eff = args.lambda_tgt_mod * paired_alpha

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(x_src, src_mod, adv_alpha=adv_alpha)

                rec_l1 = F.l1_loss(out["x_rec"], x_src)
                rec_ssim = 1.0 - ssim3d(out["x_rec"].float(), x_src.float(), window_size=7, data_range=2.0)
                rec_loss = rec_l1 + args.lambda_ssim * rec_ssim

                mod_loss = F.cross_entropy(out["mod_logits"], src_mod)
                adv_loss = F.cross_entropy(out["adv_logits"], src_mod)
                decor_loss = covariance_penalty(out["z_a"], out["z_m"])
                anat_loss = torch.zeros([], device=device)
                latent_align_loss = torch.zeros([], device=device)
                tgt_mod_loss = torch.zeros([], device=device)

                paired_mask = is_paired > 0.5
                if paired_mask.any():
                    with torch.no_grad():
                        z_tgt_raw, _, _ = model.encode_parts(x_tgt[paired_mask])

                    _, z_a_src, _ = model.encode_parts(x_src[paired_mask])
                    _, z_a_tgt, z_m_tgt = model.encode_parts(x_tgt[paired_mask])

                    # Same subject + same field + different modality: anatomy should match.
                    anat_loss = F.l1_loss(z_a_src, z_a_tgt)
                    tgt_mod_loss = F.cross_entropy(model.mod_classifier(z_m_tgt), tgt_mod[paired_mask])

                    z_cross = model.fuse_to_latent(z_a_src, z_m_tgt, tgt_mod[paired_mask])
                    x_cross = _decode_medvae(model.medvae, z_cross, out_shape=x_src.shape[2:])
                    cross_loss = F.l1_loss(x_cross, x_tgt[paired_mask])

                    # Stronger paired supervision: the cross latent should land near the target MedVAE latent.
                    latent_align_loss = F.l1_loss(z_cross, z_tgt_raw)
                else:
                    cross_loss = torch.zeros([], device=device)

                total = (
                    args.lambda_rec * rec_loss
                    + args.lambda_mod * mod_loss
                    + lambda_tgt_mod_eff * tgt_mod_loss
                    + args.lambda_adv * adv_loss
                    + lambda_cross_eff * cross_loss
                    + lambda_anat_eff * anat_loss
                    + lambda_latent_eff * latent_align_loss
                    + args.lambda_decor * decor_loss
                )

            if not torch.isfinite(total):
                print(f"[WARN] step={step} non-finite loss, skip")
                continue

            if scaler.is_enabled():
                scaler.scale(total).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                total.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                opt.step()

            if step % args.print_every == 0 or step == 1:
                elapsed = (time.time() - t0) / 60.0
                print(
                    f"[{step:5d}/{args.steps}] loss={float(total.item()):.4f} "
                    f"rec={float(rec_loss.item()):.4f} mod={float(mod_loss.item()):.4f} "
                    f"tgt_mod={float(tgt_mod_loss.item()):.4f} adv={float(adv_loss.item()):.4f} "
                    f"cross={float(cross_loss.item()):.4f} anat={float(anat_loss.item()):.4f} "
                    f"lat={float(latent_align_loss.item()):.4f} decor={float(decor_loss.item()):.4f} "
                    f"a_adv={adv_alpha:.3f} a_cross={cross_alpha:.3f} a_pair={paired_alpha:.3f} "
                    f"w_cross={lambda_cross_eff:.3f} w_anat={lambda_anat_eff:.3f} "
                    f"w_lat={lambda_latent_eff:.3f} w_tgt={lambda_tgt_mod_eff:.3f} "
                    f"t={elapsed:.1f}min"
                )

            if step % args.save_every == 0 or step == args.steps:
                state = {
                    k: v for k, v in model.state_dict().items() if not k.startswith("medvae.")
                }
                ckpt = {
                    "step": step,
                    "model": state,
                    "optimizer": opt.state_dict(),
                    "best_loss": best_loss,
                    "args": vars(args),
                }
                step_ckpt = weights_dir / f"disent_step_{step:06d}.pth"
                torch.save(ckpt, step_ckpt)
                if float(total.item()) < best_loss:
                    best_loss = float(total.item())
                    ckpt["best_loss"] = best_loss
                    torch.save(ckpt, weights_dir / "model_best.pth")
                torch.save(ckpt, weights_dir / "model_final.pth")

    print(f"Training done. best_loss={best_loss:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MedVAE disentanglement v1 (anatomie/modalite)")
    p.add_argument("--data-root", type=str, default="/home/rousseau/Data/MRIxFields_20260414")
    p.add_argument("--output-dir", type=str, default="outputs/medvae_disentangle_v1/runs/default")
    p.add_argument("--splits", nargs="+", default=["retro_train"])
    p.add_argument("--modalities", nargs="+", default=["T1W", "T2W", "T2FLAIR"])
    p.add_argument("--fields", nargs="+", default=["0.1T", "1.5T", "3T", "5T", "7T"])

    p.add_argument("--volume-size", nargs=3, type=int, default=[112, 128, 80])
    p.add_argument("--target-spacing", nargs=3, type=float, default=None)
    p.add_argument("--percentile-lower", type=float, default=0.5)
    p.add_argument("--percentile-upper", type=float, default=99.5)
    p.add_argument("--paired-prob", type=float, default=0.5)
    p.add_argument("--max-samples", type=int, default=None)

    p.add_argument("--medvae-model-name", type=str, default="medvae_4_1_3d")
    p.add_argument("--anat-channels", type=int, default=8)
    p.add_argument("--style-dim", type=int, default=32)
    p.add_argument("--film-hidden", type=int, default=128)

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--lambda-rec", type=float, default=1.0)
    p.add_argument("--lambda-ssim", type=float, default=0.5)
    p.add_argument("--lambda-mod", type=float, default=1.0)
    p.add_argument("--lambda-tgt-mod", type=float, default=0.2)
    p.add_argument("--lambda-adv", type=float, default=0.002)
    p.add_argument("--lambda-cross", type=float, default=0.4)
    p.add_argument("--lambda-anat", type=float, default=0.35)
    p.add_argument("--lambda-latent", type=float, default=0.25)
    p.add_argument("--lambda-decor", type=float, default=0.01)
    p.add_argument("--adv-start-step", type=int, default=800)
    p.add_argument("--adv-ramp-steps", type=int, default=2500)
    p.add_argument("--cross-start-step", type=int, default=300)
    p.add_argument("--cross-ramp-steps", type=int, default=1200)
    p.add_argument("--paired-start-step", type=int, default=900)
    p.add_argument("--paired-ramp-steps", type=int, default=2200)

    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--amp-dtype", type=str, choices=["fp16", "bf16"], default="bf16")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
