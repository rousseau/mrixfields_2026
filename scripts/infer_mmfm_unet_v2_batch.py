#!/usr/bin/env python3
"""Inférence batch MMFM-UNet V2 — Task 1/2/3 (180 prédictions)."""

import argparse
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

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
from common.io import (
    DOMAINS,
    MODALITIES,
    load_nifti_volume,
    resample_volume,
    adjust_affine_for_crop_pad,
)


def infer_batch(cfg_path: str, checkpoint: str, output_dir: str, env_path=None, n_steps=50, use_ema=True):
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
    raw_vs = data_cfg.get("volume_size", None)
    volume_size = tuple(int(v) for v in raw_vs) if raw_vs else None
    raw_ts = data_cfg.get("target_spacing", None)
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else None

    modalities = data_cfg.get("modalities", MODALITIES)
    fields = data_cfg.get("fields", DOMAINS)
    n_fields = len(fields)
    n_classes = len(modalities) * n_fields

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

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path("/home/rousseau/Data/MRIxFields_20260414/Training_prospective")
    subjects = ["0006", "0007", "0009"]

    total = 0
    count = 0
    for mod in modalities:
        for src in fields:
            for tgt in fields:
                if src == tgt:
                    continue
                input_dir = data_root / mod / src
                if not input_dir.exists():
                    continue
                total += 1

    start_all = time.time()
    for mod in modalities:
        for src in fields:
            for tgt in fields:
                if src == tgt:
                    continue
                input_dir = data_root / mod / src
                if not input_dir.exists():
                    continue
                count += 1
                src_mod_idx = modalities.index(mod)
                src_field_idx = fields.index(src)
                tgt_mod_idx = modalities.index(mod)
                tgt_field_idx = fields.index(tgt)
                tgt_class = _flat_class(tgt_mod_idx, tgt_field_idx, n_fields)

                print(f"\n[{count}/{total}] {mod}: {src} → {tgt}  (class={tgt_class})")
                input_files = sorted(input_dir.glob("*.nii.gz"))
                for nii_path in input_files:
                    t0 = time.time()
                    vol, _ = load_nifti_volume(
                        nii_path,
                        target_spacing=target_spacing,
                        volume_size=volume_size,
                        normalize=True,
                        lo_pct=p_lo,
                        hi_pct=p_hi,
                    )
                    img_nib = nib.load(str(nii_path))
                    orig_spacing = np.abs(np.diag(img_nib.affine)[:3])
                    orig_shape = np.array(img_nib.shape[:3])
                    if target_spacing is not None:
                        resampled_arr = resample_volume(
                            np.zeros(orig_shape.tolist(), dtype=np.float32),
                            orig_spacing, target_spacing,
                        )
                        resampled_shape = resampled_arr.shape
                    else:
                        resampled_shape = None

                    out_affine = adjust_affine_for_crop_pad(
                        img_nib.affine.copy().astype(float),
                        orig_shape, volume_size, resampled_shape, target_spacing, orig_spacing,
                    )

                    vol_tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)

                    with torch.no_grad(), torch.amp.autocast(
                        "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
                    ):
                        z_src = vae.encode(vol_tensor)
                        z_tgt = _euler_integrate(unet, z_src, tgt_class, n_steps, device, use_amp, amp_dtype)
                        recon = vae.decode(z_tgt)

                    pred_vol = recon.squeeze().cpu().float().numpy()
                    pred_vol = (np.clip(pred_vol, -1.0, 1.0) + 1.0) / 2.0

                    stem = nii_path.name.replace(".nii.gz", "")
                    out_name = f"{stem}_{mod}_{tgt}_mmfm_unet.nii.gz"
                    out_path = out_dir / out_name
                    nib.save(nib.Nifti1Image(pred_vol, out_affine), str(out_path))
                    print(f"  {nii_path.name} → {out_name}  ({time.time()-t0:.1f}s)")

    elapsed = time.time() - start_all
    print(f"\n✅ Batch inference done: {count} pairs, {elapsed/60:.1f} min total")
    print(f"   Predictions saved to: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Batch inference MMFM-UNet V2")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--env", default="local")
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--no_ema", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer_batch(
        cfg_path=args.config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        env_path=args.env,
        n_steps=args.n_steps,
        use_ema=not args.no_ema,
    )
