#!/usr/bin/env python3
"""Inférence batch MMFM-UNet V2 — Task 3 full-resolution (0.5 mm).

Ce script génère des prédictions pleine résolution compatibles avec la soumission
MRIxFields 2026 :
  - shape (364, 436, 364)
  - spacing 0.5 mm isotrope
  - affine/header identiques à la source
  - nom officiel P_{MOD}_{TARGET_FIELD}_{ID}.nii.gz
  - fond masqué avec le masque source (> 1e-6)

Pipeline par volume :
  1. Charger source à 0.5 mm.
  2. Rééchantillonner en 1 mm.
  3. Normaliser globalement (percentile 0.5/99.5 → [-1,1]).
  4. Extraire des patches (128,128,80) avec recouvrement 50 %.
  5. Pour chaque patch : encoder (VAE) → flow (UNet) → décoder.
  6. Recombiner par mélange à fenêtre (Hann) avec padding de bordure.
  7. Rééchantillonner 1 mm → 0.5 mm.
  8. Appliquer le masque source.
  9. Sauvegarder au format officiel.

Usage:
  PYTHONPATH=src python scripts/infer_mmfm_unet_v2_batch.py \
      --config configs/mmfm3d_unet_v2_medvae_multimodal.yaml \
      --checkpoint outputs/cfm3d/runs/mmfm3d_unet_v2_medvae_multimodal/weights/checkpoint_20000.pth \
      --output_dir outputs/predictions/mmfm_unet/task3 \
      --split Training_prospective \
      --n_steps 50
"""

import argparse
import sys
import time
from pathlib import Path

import nibabel as nib
import nibabel.processing as nib_proc
import numpy as np
import torch

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

from cfm.train_mmfm_unet_3d import (
    build_unet_3d,
    load_vae,
    _euler_integrate,
    _flat_class,
    _remap_monai_attention_keys,
)
from common.config import load_yaml_with_include, load_env, resolve_paths
from common.io import DOMAINS, MODALITIES


# --------------------------------------------------------------------------- #
#  Patch extraction / blending
# --------------------------------------------------------------------------- #


def _create_blend_weights(patch_size, mode="hann"):
    """Create 3D blending weights for a patch."""
    h, w, d = patch_size
    if mode == "hann":
        wh = torch.hann_window(h, periodic=False)
        ww = torch.hann_window(w, periodic=False)
        wd = torch.hann_window(d, periodic=False)
    else:
        wh = torch.ones(h)
        ww = torch.ones(w)
        wd = torch.ones(d)
    weights = wh[:, None, None] * ww[None, :, None] * wd[None, None, :]
    return (weights / weights.max()).float()


def _normalize_global(vol, p_lo=0.5, p_hi=99.5):
    """Percentile normalization → [0, 1] → [-1, 1]."""
    lo = np.percentile(vol, p_lo)
    hi = np.percentile(vol, p_hi)
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return (vol * 2.0 - 1.0).astype(np.float32)


def _extract_patches(vol, patch_size, stride, pad=16):
    """Extract overlapping patches with boundary padding.

    Returns:
        patches: list of (ph, pw, pd) arrays
        positions: list of (i, j, k) positions in padded coordinates
    """
    h, w, d = vol.shape
    ph, pw, pd = patch_size
    sh, sw, sd = stride

    # Reflect-pad to avoid zero-weight boundary voxels
    pad_hw = [(pad, pad), (pad, pad), (pad, pad)]
    vol_padded = np.pad(vol, pad_hw, mode="reflect")

    positions = set()
    hp, wp, dp = vol_padded.shape
    for i in range(0, hp, sh):
        i0 = min(i, max(0, hp - ph)) if hp >= ph else 0
        for j in range(0, wp, sw):
            j0 = min(j, max(0, wp - pw)) if wp >= pw else 0
            for k in range(0, dp, sd):
                k0 = min(k, max(0, dp - pd)) if dp >= pd else 0
                positions.add((i0, j0, k0))

    patches = []
    positions = list(positions)
    for (i0, j0, k0) in positions:
        patch = vol_padded[i0:i0 + ph, j0:j0 + pw, k0:k0 + pd]
        if patch.shape != (ph, pw, pd):
            patch = np.pad(
                patch,
                [(0, ph - patch.shape[0]), (0, pw - patch.shape[1]), (0, pd - patch.shape[2])],
                mode="reflect",
            )
        patches.append(patch)

    return patches, positions, vol_padded.shape


def _blend_patches(patch_outputs, positions, weights, padded_shape, original_shape, pad):
    """Blend predicted patches into a full volume and crop back."""
    recon = torch.zeros((1, 1, *padded_shape), dtype=torch.float32)
    wsum = torch.zeros((1, 1, *padded_shape), dtype=torch.float32)
    weights_5d = weights[None, None, :, :, :]

    for patch, (i0, j0, k0) in zip(patch_outputs, positions):
        recon[:, :, i0:i0 + patch.shape[2], j0:j0 + patch.shape[3], k0:k0 + patch.shape[4]] += (
            patch * weights_5d
        )
        wsum[:, :, i0:i0 + patch.shape[2], j0:j0 + patch.shape[3], k0:k0 + patch.shape[4]] += weights_5d

    wsum = torch.clamp(wsum, min=1e-8)
    recon = recon / wsum

    # Crop back to original size
    h, w, d = original_shape
    recon_cropped = recon[0, 0, pad:pad + h, pad:pad + w, pad:pad + d]
    return recon_cropped.numpy()


# --------------------------------------------------------------------------- #
#  Full-resolution inference for one volume
# --------------------------------------------------------------------------- #


def _infer_patch(patch_tensor, vae, unet, tgt_class, n_steps, device, use_amp, amp_dtype):
    """Run VAE encode → flow → VAE decode on a single patch."""
    with torch.no_grad(), torch.amp.autocast(
        "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
    ):
        z_src = vae.encode(patch_tensor)
        z_tgt = _euler_integrate(unet, z_src, tgt_class, n_steps, device, use_amp, amp_dtype)
        recon = vae.decode(z_tgt)

    pred = recon.squeeze().cpu().float().numpy()
    return (np.clip(pred, -1.0, 1.0) + 1.0) / 2.0


def process_volume(
    nii_path: Path,
    vae,
    unet,
    tgt_class: int,
    n_steps: int,
    patch_size,
    stride,
    pad: int,
    p_lo: float,
    p_hi: float,
    device,
    use_amp: bool,
    amp_dtype,
):
    """Generate a full-resolution prediction from a source NIfTI."""
    img_src = nib.load(str(nii_path))
    affine_src = img_src.affine.copy()
    header_src = img_src.header.copy()

    # 1. Resample source to 1 mm
    img_src_1mm = nib_proc.resample_to_output(img_src, voxel_sizes=(1.0, 1.0, 1.0), order=1)
    vol_1mm = img_src_1mm.get_fdata(dtype=np.float32)

    # 2. Global normalization in 1 mm space
    vol_1mm_norm = _normalize_global(vol_1mm, p_lo, p_hi)

    # 3. Extract patches
    patches, positions, padded_shape = _extract_patches(vol_1mm_norm, patch_size, stride, pad=pad)
    blend_weights = _create_blend_weights(patch_size, mode="hann")

    # 4. Process each patch
    patch_outputs = []
    for patch in patches:
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
        pred_patch = _infer_patch(patch_tensor, vae, unet, tgt_class, n_steps, device, use_amp, amp_dtype)
        pred_patch_tensor = torch.from_numpy(pred_patch).unsqueeze(0).unsqueeze(0).float()
        patch_outputs.append(pred_patch_tensor)

    # 5. Blend into full 1 mm prediction
    pred_1mm = _blend_patches(
        patch_outputs, positions, blend_weights, padded_shape, vol_1mm_norm.shape, pad
    )

    # 6. Resample prediction back to 0.5 mm source space
    img_pred_1mm = nib.Nifti1Image(pred_1mm.astype(np.float32), img_src_1mm.affine)
    img_pred_05mm = nib_proc.resample_from_to(img_pred_1mm, img_src, order=1)
    pred_05mm = img_pred_05mm.get_fdata(dtype=np.float32)

    # 7. Apply source brain mask
    vol_src_05mm = img_src.get_fdata(dtype=np.float32)
    mask = vol_src_05mm > 1e-6
    pred_05mm_masked = pred_05mm * mask

    return pred_05mm_masked.astype(np.float32), affine_src, header_src


# --------------------------------------------------------------------------- #
#  Batch orchestration
# --------------------------------------------------------------------------- #


def infer_batch(
    cfg_path: str,
    checkpoint: str,
    output_dir: str,
    split: str = "Training_prospective",
    modalities=None,
    env_path=None,
    n_steps: int = 50,
    patch_size=None,
    stride=None,
    pad: int = 16,
    use_ema: bool = True,
    skip_existing: bool = False,
):
    cfg = load_yaml_with_include(cfg_path)
    cfg = resolve_paths(cfg, load_env(env_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    amp_dtype_name = train_cfg.get("amp_dtype", "bf16")
    use_amp = bool(train_cfg.get("use_amp", True))
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16

    p_lo = data_cfg.get("percentile_lower", 0.5)
    p_hi = data_cfg.get("percentile_upper", 99.5)

    patch_size = tuple(patch_size) if patch_size else (128, 128, 80)
    stride = tuple(stride) if stride else (64, 64, 40)

    all_modalities = data_cfg.get("modalities", MODALITIES)
    fields = data_cfg.get("fields", DOMAINS)
    n_fields = len(fields)
    n_classes = len(all_modalities) * n_fields

    modalities = modalities if modalities is not None else all_modalities

    # Load VAE
    print("Loading VAE...")
    vae = load_vae(cfg, device)
    if vae.latent_format != "spatial":
        raise RuntimeError(f"Requires spatial VAE, got {vae.latent_format}")
    latent_channels = vae.latent_channels

    # Load UNet
    print("Loading UNet...")
    unet = build_unet_3d(cfg, latent_channels, n_classes).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    saved_n_classes = ckpt.get("n_classes", None)
    if saved_n_classes is not None and saved_n_classes != n_classes:
        raise RuntimeError(f"n_classes mismatch: {saved_n_classes} vs {n_classes}")

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
    print(f"  UNet loaded ({loaded_from}, iter={ckpt.get('iter','?')})")

    data_root_env = cfg.get("data_root") or cfg.get("data", {}).get("data_root")
    data_root = Path(data_root_env) if data_root_env else Path("/home/rousseau/Data/MRIxFields_20260414")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Task 3: all directed pairs
    task_pairs = [(s, t) for s in fields for t in fields if s != t]

    total_volumes = len(modalities) * len(task_pairs)  # per subject
    start_all = time.time()

    for mod in modalities:
        mod_idx = all_modalities.index(mod)
        for src, tgt in task_pairs:
            tgt_idx = _flat_class(mod_idx, fields.index(tgt), n_fields)
            pair_out_dir = out_root / "task3" / mod / f"{src}_to_{tgt}"
            pair_out_dir.mkdir(parents=True, exist_ok=True)

            input_dir = data_root / split / mod / src
            if not input_dir.exists():
                print(f"[WARN] Input dir missing: {input_dir}")
                continue

            input_files = sorted(input_dir.glob("*.nii.gz"))
            print(f"\n[{mod}] {src} → {tgt} : {len(input_files)} sujets")

            for nii_path in input_files:
                sid_match = nii_path.name.split("_")[-1].replace(".nii.gz", "")
                official_name = f"P_{mod}_{tgt}_{sid_match}.nii.gz"
                out_path = pair_out_dir / official_name

                if skip_existing and out_path.exists():
                    print(f"  SKIP {official_name}")
                    continue

                t0 = time.time()
                pred_vol, affine, header = process_volume(
                    nii_path,
                    vae,
                    unet,
                    tgt_class=tgt_idx,
                    n_steps=n_steps,
                    patch_size=patch_size,
                    stride=stride,
                    pad=pad,
                    p_lo=p_lo,
                    p_hi=p_hi,
                    device=device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )

                nib.save(nib.Nifti1Image(pred_vol, affine, header), str(out_path))
                print(f"  {nii_path.name} → {out_path}  ({time.time() - t0:.1f}s)")

    elapsed = time.time() - start_all
    print(f"\n✅ Batch inference done in {elapsed / 3600:.2f} h")
    print(f"   Predictions saved to: {out_root}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description="Batch full-resolution MMFM-UNet V2 inference (Task 3)")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", default="Training_prospective",
                   choices=["Training_prospective", "Validating_prospective", "Testing_prospective"])
    p.add_argument("--modalities", nargs="+", default=None,
                   help="Subset of modalities to process (default: all from config)")
    p.add_argument("--env", default="local")
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 80])
    p.add_argument("--stride", type=int, nargs=3, default=[64, 64, 40])
    p.add_argument("--pad", type=int, default=16)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--no_ema", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer_batch(
        cfg_path=args.config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        modalities=args.modalities,
        env_path=args.env,
        n_steps=args.n_steps,
        patch_size=args.patch_size,
        stride=args.stride,
        pad=args.pad,
        use_ema=not args.no_ema,
        skip_existing=args.skip_existing,
    )
