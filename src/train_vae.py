#!/usr/bin/env python3
"""
Unified VAE training script for MRIxFields.

Supports 3 architectures:
1. AEKL (AutoencoderKL from MONAI)
2. VQ-VAE (NeuroQuant-inspired, hybrid paired/unpaired)
3. MedVAE (pre-trained on 1M medical images, from medvae package)

Usage:
  # AEKL
  python3 src/train_vae.py --vae aekl --config configs/vae3d_T1W.yaml --steps 100

  # VQ-VAE
  python3 src/train_vae.py --vae vqvae --data-root /path/to/MRIxFields \
    --modalities T1W T2W --fields 0.1T 3T --steps 100

  # MedVAE (fine-tuning)
  python3 src/train_vae.py --vae medvae --data-root /path/to/MRIxFields \
    --steps 100 --lr 1e-5

  # Smoke test (1 step, all VAEs)
  python3 src/train_vae.py --vae aekl --steps 1 --batch-size 1 --device cpu
"""

import argparse
import inspect
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as scipy_zoom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# MONAI
try:
    from monai.networks.nets import AutoencoderKL
except ImportError:
    try:
        from monai.generative.networks.nets import AutoencoderKL
    except ImportError:
        try:
            from generative.networks.nets import AutoencoderKL
        except ImportError as e:
            raise ImportError("AutoencoderKL not found") from e


# ============================================================================
# Constants
# ============================================================================
SPLIT_MAP = {
    "retro_train": "Training_retrospective",
    "pro_train": "Training_prospective",
    "pro_val": "Validating_prospective",
    "pro_test": "Testing_prospective",
}

# ============================================================================
# Utility Functions
# ============================================================================

def _resample_volume(
    vol: np.ndarray,
    original_spacing: Tuple,
    target_spacing: Tuple[float, float, float],
) -> np.ndarray:
    """Resample 3D volume to target spacing."""
    orig = np.asarray(original_spacing[:3], dtype=float)
    tgt = np.asarray(target_spacing, dtype=float)
    factors = orig / tgt
    if np.allclose(factors, 1.0, atol=0.02):
        return vol.astype(np.float32)
    return scipy_zoom(vol, factors, order=1).astype(np.float32)


def _normalize(
    vol: np.ndarray,
    lo_pct: float = 0.5,
    hi_pct: float = 99.5,
) -> np.ndarray:
    """Normalize to [-1, 1]."""
    lo = np.percentile(vol, lo_pct)
    hi = np.percentile(vol, hi_pct)
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return (vol * 2.0 - 1.0).astype(np.float32)


def _center_crop_or_pad(
    vol: np.ndarray,
    size: Tuple[int, int, int],
) -> np.ndarray:
    """Crop or pad to target size."""
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


# ============================================================================
# Datasets
# ============================================================================

class SimpleMRIDataset(Dataset):
    """Simple dataset for AEKL: single modality, single field."""

    def __init__(
        self,
        data_root: Path,
        split: str = "pro_train",
        modality: str = "T1W",
        field: str = "0.1T",
        volume_size: Tuple[int, int, int] = (128, 128, 64),
        target_spacing: Optional[Tuple] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.volume_size = volume_size
        self.target_spacing = target_spacing or (1.0, 1.0, 1.0)

        split_dir = SPLIT_MAP.get(split, split)
        d = self.data_root / split_dir / modality / field

        if not d.exists():
            raise FileNotFoundError(f"Directory not found: {d}")

        self.paths = list(sorted(d.glob("*.nii.gz")))
        if max_samples is not None:
            self.paths = self.paths[:max_samples]

        if not self.paths:
            raise FileNotFoundError(f"No NIfTI files in {d}")

        print(f"SimpleMRIDataset: {len(self.paths)} volumes from {d}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = nib.load(path)
        vol = img.get_fdata().astype(np.float32)
        spacing = img.header.get_zooms()

        # Resample if needed
        if self.target_spacing and not np.allclose(spacing[:3], self.target_spacing):
            vol = _resample_volume(vol, spacing, self.target_spacing)

        # Normalize
        vol = _normalize(vol)

        # Crop/pad
        vol = _center_crop_or_pad(vol, self.volume_size)

        return torch.from_numpy(vol[None]).float()  # (1, H, W, D)


class HybridMRIDataset(Dataset):
    """Hybrid dataset for VQ-VAE: multi-modality, multi-field, paired/unpaired."""

    import re

    FILE_RE = re.compile(r"^[A-Z]_([A-Z0-9]+)_([0-9.]+T)_(\d+)\.nii\.gz$")

    def __init__(
        self,
        data_root: Path,
        splits: List[str] = ["pro_train"],
        modalities: List[str] = ["T1W"],
        fields: List[str] = ["0.1T"],
        volume_size: Tuple[int, int, int] = (128, 128, 64),
        paired_prob: float = 0.5,
        target_spacing: Optional[Tuple] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.volume_size = volume_size
        self.paired_prob = paired_prob
        self.target_spacing = target_spacing or (1.0, 1.0, 1.0)

        self.samples = []
        self.by_key = {}  # (split, field, subject_id) -> {modality -> path}

        for split in splits:
            split_dir = SPLIT_MAP.get(split, split)
            for modality in modalities:
                for field in fields:
                    d = self.data_root / split_dir / modality / field
                    if not d.exists():
                        continue
                    for p in sorted(d.glob("*.nii.gz")):
                        m = self.FILE_RE.match(p.name)
                        if m is None:
                            continue
                        subj = m.group(3)
                        self.samples.append((p, split, modality, field, subj))
                        key = (split, field, subj)
                        if key not in self.by_key:
                            self.by_key[key] = {}
                        self.by_key[key][modality] = p

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if not self.samples:
            raise FileNotFoundError("No NIfTI files found")

        self.mod_to_idx = {m: i for i, m in enumerate(modalities)}
        self.field_to_idx = {f: i for i, f in enumerate(fields)}

        n_pairable = sum(
            1
            for s in self.samples
            if len(self.by_key[(s[1], s[3], s[4])]) > 1
        )
        print(f"HybridMRIDataset: {len(self.samples)} samples | pairables: {n_pairable}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_src, split, mod_src, field, subj = self.samples[idx]

        # Load source
        img = nib.load(path_src)
        vol_src = img.get_fdata().astype(np.float32)
        spacing = img.header.get_zooms()
        if not np.allclose(spacing[:3], self.target_spacing):
            vol_src = _resample_volume(vol_src, spacing, self.target_spacing)
        vol_src = _normalize(vol_src)
        vol_src = _center_crop_or_pad(vol_src, self.volume_size)

        # Try to load paired target
        is_paired = False
        vol_tgt = None
        if np.random.rand() < self.paired_prob:
            key = (split, field, subj)
            if key in self.by_key and len(self.by_key[key]) > 1:
                mods_available = list(self.by_key[key].keys())
                mods_other = [m for m in mods_available if m != mod_src]
                if mods_other:
                    mod_tgt = np.random.choice(mods_other)
                    path_tgt = self.by_key[key][mod_tgt]
                    img_tgt = nib.load(path_tgt)
                    vol_tgt = img_tgt.get_fdata().astype(np.float32)
                    spacing_tgt = img_tgt.header.get_zooms()
                    if not np.allclose(spacing_tgt[:3], self.target_spacing):
                        vol_tgt = _resample_volume(vol_tgt, spacing_tgt, self.target_spacing)
                    vol_tgt = _normalize(vol_tgt)
                    vol_tgt = _center_crop_or_pad(vol_tgt, self.volume_size)
                    is_paired = True

        src_dict = {
            "image": torch.from_numpy(vol_src[None]).float(),
            "modality_idx": torch.tensor(self.mod_to_idx[mod_src], dtype=torch.long),
            "field_idx": torch.tensor(self.field_to_idx[field], dtype=torch.long),
            "is_paired": torch.tensor(is_paired, dtype=torch.bool),
            # Always keep this key so DataLoader collation stays stable.
            "paired_image": torch.from_numpy((vol_tgt if (is_paired and vol_tgt is not None) else vol_src)[None]).float(),
        }

        return src_dict


# ============================================================================
# VAE Factories
# ============================================================================

def create_aekl_vae(device: str = "cuda") -> nn.Module:
    """Create AEKL model with MONAI-version-compatible kwargs."""
    sig = inspect.signature(AutoencoderKL.__init__).parameters

    channels = (64, 128, 256)
    kwargs = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "latent_channels": 4,
        "num_res_blocks": 2,
        "norm_num_groups": 32,
        "attention_levels": (False, False, False),
        "with_encoder_nonlocal_attn": False,
        "with_decoder_nonlocal_attn": False,
    }

    # MONAI API can expose either channels or num_channels.
    if "channels" in sig:
        kwargs["channels"] = channels
    elif "num_channels" in sig:
        kwargs["num_channels"] = channels

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig}
    model = AutoencoderKL(**filtered_kwargs)
    return model.to(device)


def create_vqvae(
    n_modalities: int = 3,
    n_fields: int = 5,
    device: str = "cuda",
) -> nn.Module:
    """Create VQ-VAE model."""
    from train_vqvae import NeuroQuantHybrid
    model = NeuroQuantHybrid(
        n_modalities=n_modalities,
        n_fields=n_fields,
        base_channels=32,
        anat_channels=64,
        mod_channels=32,
        codebook_size=1024,
    )
    return model.to(device)


def create_medvae(
    device: str = "cuda",
) -> Optional[nn.Module]:
    """Create MedVAE model (requires medvae package)."""
    try:
        from medvae import MVAE
        model = MVAE(model_name="medvae_4_1_3d", modality="mri")
        return model.to(device)
    except ImportError:
        print("⚠ MedVAE not installed: pip install medvae")
        return None


# ============================================================================
# Training Loop
# ============================================================================

def train_step(
    model: nn.Module,
    batch: Dict,
    device: str,
    vae_type: str,
    use_amp: bool = True,
    scaler=None,
    optimizer=None,
) -> float:
    """Single training step."""
    model.train()

    x = batch["image"].to(device) if isinstance(batch, dict) else batch.to(device)
    optimizer.zero_grad(set_to_none=True)

    def _compute_loss() -> torch.Tensor:
        if vae_type == "aekl":
            # AutoencoderKL returns (recon, z_mu, z_logvar)
            recon, z_mu, z_logvar = model(x)
            recon_loss = F.mse_loss(recon, x, reduction="mean")
            kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
            return recon_loss + 1e-6 * kl_loss

        if vae_type == "vqvae":
            src_mod = batch["modality_idx"].to(device)
            src_field = batch["field_idx"].to(device)
            out = model.forward_src(x, src_mod, src_field)
            recon_loss = F.l1_loss(out["x_rec"], x)
            vq_loss = out["vq_loss"]

            cross_loss = torch.zeros([], device=x.device)
            if "paired_image" in batch:
                paired = batch["paired_image"].to(device)
                paired_mask = batch["is_paired"].to(device) > 0.5
                if paired_mask.any():
                    cross_loss = F.l1_loss(out["x_rec"][paired_mask], paired[paired_mask])

            return recon_loss + vq_loss + 0.5 * cross_loss

        if vae_type == "medvae":
            # Replace MedVAE loss branch with robust encode/decode reconstruction loss
            z = model.encode(x)
            recon = model.decode(z)
            return F.mse_loss(recon, x, reduction="mean")
        raise ValueError(f"Unknown VAE type: {vae_type}")

    if use_amp and scaler is not None:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = _compute_loss()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss = _compute_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return loss.item()


def train_vae(
    vae_type: str,
    data_root: Path,
    steps: int = 100,
    batch_size: int = 2,
    lr: float = 1e-4,
    device: str = "cuda",
    output_dir: Optional[Path] = None,
    use_amp: bool = True,
    modalities: Optional[List[str]] = None,
    fields: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
):
    """Train VAE."""
    if output_dir is None:
        output_dir = Path(f"outputs/train_{vae_type}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    if vae_type == "aekl":
        dataset = SimpleMRIDataset(
            data_root=data_root,
            modality="T1W",
            field="0.1T",
            volume_size=(128, 128, 64),
            max_samples=max_samples,
        )
    elif vae_type == "vqvae":
        if modalities is None:
            modalities = ["T1W", "T2W"]
        if fields is None:
            fields = ["0.1T", "3T"]
        dataset = HybridMRIDataset(
            data_root=data_root,
            modalities=modalities,
            fields=fields,
            volume_size=(128, 128, 64),
            paired_prob=0.5,
            max_samples=max_samples,
        )
    elif vae_type == "medvae":
        dataset = SimpleMRIDataset(
            data_root=data_root,
            modality="T1W",
            field="0.1T",
            volume_size=(64, 64, 32),
            max_samples=max_samples,
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Create model
    if vae_type == "aekl":
        model = create_aekl_vae(device=device)
    elif vae_type == "vqvae":
        n_mods = len(modalities or ["T1W"])
        n_fields = len(fields or ["0.1T"])
        model = create_vqvae(n_modalities=n_mods, n_fields=n_fields, device=device)
    elif vae_type == "medvae":
        model = create_medvae(device=device)
        if model is None:
            return
    else:
        raise ValueError(f"Unknown VAE type: {vae_type}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device == "cuda" else None

    # Training loop
    print(f"\n{'='*70}")
    print(f"  Training {vae_type.upper()}")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Steps: {steps}")
    print(f"Output: {output_dir}\n")

    t0 = time.time()
    for step in range(steps):
        for batch_idx, batch in enumerate(loader):
            if step * len(loader) + batch_idx >= steps:
                break

            loss = train_step(
                model=model,
                batch=batch,
                device=device,
                vae_type=vae_type,
                use_amp=use_amp,
                scaler=scaler,
                optimizer=optimizer,
            )

            elapsed = time.time() - t0
            print(f"[{step:04d}] loss={loss:.4f} ({elapsed:.1f}s)")

            if step >= steps:
                break

    # Save checkpoint
    ckpt_path = output_dir / f"{vae_type}_final.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✓ Saved checkpoint: {ckpt_path}")
    print(f"{'='*70}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified VAE training")
    parser.add_argument(
        "--vae",
        choices=["aekl", "vqvae", "medvae"],
        required=True,
        help="VAE architecture to train",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/rousseau/Data/MRIxFields_20260414"),
        help="Path to MRIxFields data",
    )
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/train_<vae>)",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=True,
        help="Use AMP (automatic mixed precision)",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        help="Modalities for VQ-VAE (default: T1W T2W)",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=None,
        help="Fields for VQ-VAE (default: 0.1T 3T)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per split (for smoke test)",
    )

    args = parser.parse_args()

    train_vae(
        vae_type=args.vae,
        data_root=args.data_root,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        use_amp=args.use_amp and torch.cuda.is_available(),
        modalities=args.modalities,
        fields=args.fields,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
