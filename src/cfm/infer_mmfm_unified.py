#!/usr/bin/env python3
"""Inférence MMFM unifiée — vectoriel (MLP) et UNet (V1/V2).

Gère :
  - Modèle vectoriel : mmfm3d / mmfm3d_vectorized / mmfm3d_vectorized_v1
  - Modèle UNet : mmfm3d_unet / mmfm3d_unet_v2
  - Single image ou batch Task 3
  - Full-resolution avec brain mask source (comme baseline officielle)
  - EMA correct (flat state dict ou shadow_params)

Usage single:
    PYTHONPATH=src python src/cfm/infer_mmfm_unified.py \
        --config configs/mmfm3d_medvae_multimodal.yaml \
        --checkpoint outputs/cfm3d/runs/mmfm3d_medvae_multimodal_vectorized_v1/weights/model_final.pth \
        --input /path/to/P_T1W_0.1T_0006.nii.gz --tgt-field 3T --output /tmp/pred.nii.gz

Usage batch Task 3:
    PYTHONPATH=src python src/cfm/infer_mmfm_unified.py \
        --config configs/mmfm3d_unet_v2_medvae_multimodal.yaml \
        --checkpoint outputs/cfm3d/runs/mmfm3d_unet_v2_medvae_multimodal/weights/checkpoint_115000.pth \
        --output_dir outputs/predictions/mmfm_unet_v2_fixed/task3 \
        --split Training_prospective --modalities T1W
"""

import argparse
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import nibabel.processing as nib_proc
import numpy as np
import torch

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))
# Permet d'importer les helpers depuis scripts/ (non package)
_PROJECT_ROOT = _SRC.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from cfm.mmfm_vectorized import LatentVectorizer
from cfm.train_mmfm_3d import _euler_integrate_vector
from cfm.train_mmfm_unet_3d import (
    build_unet_3d,
    load_vae,
    _euler_integrate,
    _euler_integrate_mm,
    _flat_class,
    _field_to_time,
    _pad_to_multiple,
    _crop_to_shape,
    _remap_monai_attention_keys,
)
from common.config import load_yaml_with_include, load_env, resolve_paths
from common.io import DOMAINS, MODALITIES

# Import process_volume from batch script (re-use full-res logic)
from scripts.infer_mmfm_unet_v2_batch import (
    process_volume,
    _create_blend_weights,
    _extract_patches,
    _blend_patches,
    _normalize_global,
    _normalize_with_params,
)


def _resolve_checkpoint(cfg: dict, checkpoint: Optional[str] = None) -> Path:
    if checkpoint is not None:
        return Path(checkpoint)
    output_dir = cfg.get("data", {}).get("output_dir")
    if not output_dir:
        raise ValueError("No checkpoint provided and no data.output_dir in config")
    task_name = cfg.get("task_name", "")
    output_dir = output_dir.replace("{{TASK_NAME}}", task_name)
    weights_dir = Path(output_dir) / "weights"
    ckpts = sorted(weights_dir.glob("checkpoint_*.pth"))
    candidates = []
    if ckpts:
        def _iter_number(p: Path) -> int:
            try:
                return int(p.stem.split("_")[1])
            except (ValueError, IndexError):
                return -1
        candidates.append(max(ckpts, key=_iter_number))
    candidates.extend([weights_dir / "model_final.pth", weights_dir / "model_best.pth"])
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No checkpoint found in {weights_dir}")


def _resolve_n_steps(cfg: dict, n_steps: Optional[int]) -> int:
    if n_steps is not None:
        return n_steps
    return cfg.get("inference", {}).get("n_steps", 50)


def _resolve_norm_mode(cfg: dict, norm_mode: Optional[str]) -> str:
    if norm_mode is not None:
        return norm_mode
    return cfg.get("inference", {}).get("norm_mode", "global")


def _build_model(cfg: dict, vae, device: torch.device):
    """Build and return the flow model (vectorial MLP or UNet)."""
    method = cfg.get("method", "mmfm3d")
    data_cfg = cfg["data"]
    modalities = data_cfg.get("modalities", MODALITIES)
    fields = data_cfg.get("fields", DOMAINS)
    n_fields = len(fields)
    # V2 : n_classes = modalités uniquement (3) ; V1 : modalités × champs (15)
    n_classes = len(modalities) if method == "mmfm3d_vectorized_v2" else len(modalities) * n_fields

    if method in ("mmfm3d", "mmfm", "mmfm3d_vectorized", "mmfm3d_vectorized_v1", "mmfm3d_vectorized_v2"):
        from cfm.train_mmfm_3d import build_vector_mmfm
        # Determine latent shape for vectorizer
        with torch.no_grad():
            dummy = torch.zeros((1, 1, 128, 128, 80), device=device)
            z_dummy = vae.encode(dummy)
            latent_shape = tuple(z_dummy.shape[1:])
            flat_dim = int(np.prod(latent_shape))
        mmfm = build_vector_mmfm(cfg, flat_dim, n_classes).to(device)
        return mmfm, "vectorial", latent_shape

    elif method in ("mmfm3d_unet", "mmfm3d_unet_v2"):
        unet = build_unet_3d(cfg, vae.latent_channels, n_classes).to(device)
        return unet, "unet", None

    raise ValueError(f"Unsupported method for unified inference: {method}")


def _load_model_weights(model, ckpt: dict, use_ema: bool, model_type: str):
    """Load EMA/model weights into vectorial MLP or UNet."""
    loaded_from = "model"
    state = ckpt["model"]
    if use_ema and "ema" in ckpt and ckpt["ema"]:
        ema_state = ckpt["ema"]
        if isinstance(ema_state, dict) and "shadow_params" in ema_state:
            state = ema_state["shadow_params"]
            loaded_from = "ema.shadow_params"
        elif isinstance(ema_state, dict) and ema_state:
            state = ema_state
            loaded_from = "ema"

    if model_type == "unet":
        model.load_state_dict(_remap_monai_attention_keys(state))
    else:
        model.load_state_dict(state)
    model.eval()
    return loaded_from


def _make_flow_spec(model_type: str, mod_idx: int, src_field_idx: int,
                    tgt_field_idx: int, n_fields: int, use_v2: bool = False) -> dict:
    """Construire la spécification de flow pour une translation.

    - vectorial v1 (legacy) : conditionne sur la classe cible (mod, champ cible).
    - vectorial v2 : conditionne sur la modalité seule.
    - unet (multi-marginal) : contraste + temps (champ) source->cible.
    """
    if model_type == "vectorial":
        if use_v2:
            # V2 : tgt_class = mod_idx (modalité seule, dans [0,2])
            return {"tgt_class": mod_idx}
        else:
            # V1 : tgt_class = _flat_class(mod_idx, tgt_field_idx, n_fields)
            return {"tgt_class": _flat_class(mod_idx, tgt_field_idx, n_fields)}
    return {
        "contrast": mod_idx,
        "t_start": _field_to_time(src_field_idx, n_fields),
        "t_end": _field_to_time(tgt_field_idx, n_fields),
    }


def _infer_patch_unified(
    patch_tensor: torch.Tensor,
    vae,
    model,
    model_type: str,
    latent_vectorizer,
    flow_spec: dict,
    n_steps: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    """Run VAE encode → flow → VAE decode on a single patch.

    flow_spec:
      - vectorial : {"tgt_class": int}
      - unet (multi-marginal) : {"contrast": int, "t_start": float, "t_end": float}
    """
    with torch.no_grad(), torch.amp.autocast(
        "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
    ):
        z_src = vae.encode(patch_tensor)

        if model_type == "vectorial":
            z_src_vec = vae.to_vector(z_src).float()
            z_tgt_vec = _euler_integrate_vector(
                model, z_src_vec, flow_spec["tgt_class"], n_steps, device, use_amp, amp_dtype
            )
            z_tgt = latent_vectorizer.unflatten(z_tgt_vec)
        else:
            z_src_p, lat_shape = _pad_to_multiple(z_src, 4)
            z_tgt_p = _euler_integrate_mm(
                model, z_src_p, flow_spec["contrast"],
                flow_spec["t_start"], flow_spec["t_end"],
                n_steps, device, use_amp, amp_dtype,
            )
            z_tgt = _crop_to_shape(z_tgt_p, lat_shape)

        recon = vae.decode(z_tgt)

    pred = recon.squeeze().cpu().float().numpy()
    return (np.clip(pred, -1.0, 1.0) + 1.0) / 2.0


def process_volume_unified(
    nii_path: Path,
    vae,
    model,
    model_type: str,
    latent_vectorizer,
    flow_spec: dict,
    n_steps: int,
    patch_size,
    stride,
    pad: int,
    p_lo: float,
    p_hi: float,
    device,
    use_amp: bool,
    amp_dtype,
    norm_mode: str = "global",
    center_crop_only: bool = False,
    blend_mode: str = "hann",
    center_aligned: bool = False,
):
    """Full-resolution prediction, dispatching vectorial or UNet model."""
    img_src = nib.load(str(nii_path))
    affine_src = img_src.affine.copy()
    header_src = img_src.header.copy()

    img_src_1mm = nib_proc.resample_to_output(img_src, voxel_sizes=(1.0, 1.0, 1.0), order=1)
    vol_1mm = img_src_1mm.get_fdata(dtype=np.float32)

    if norm_mode == "crop_percentile":
        h, w, d = vol_1mm.shape
        ph, pw, pd = patch_size
        sh, sw, sd = max(0, (h - ph) // 2), max(0, (w - pw) // 2), max(0, (d - pd) // 2)
        crop = vol_1mm[sh:sh + ph, sw:sw + pw, sd:sd + pd]
        lo, hi = np.percentile(crop, p_lo), np.percentile(crop, p_hi)
        vol_1mm_norm = _normalize_with_params(vol_1mm, lo, hi)
    elif norm_mode == "per_patch":
        vol_1mm_norm = vol_1mm.astype(np.float32)
        lo = hi = None
    else:
        vol_1mm_norm = _normalize_global(vol_1mm, p_lo, p_hi)
        lo = hi = None

    if center_crop_only:
        h, w, d = vol_1mm_norm.shape
        ph, pw, pd = patch_size
        sh, sw, sd = max(0, (h - ph) // 2), max(0, (w - pw) // 2), max(0, (d - pd) // 2)
        crop = vol_1mm_norm[sh:sh + ph, sw:sw + pw, sd:sd + pd]
        crop_tensor = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).to(device)
        pred_crop = _infer_patch_unified(
            crop_tensor, vae, model, model_type, latent_vectorizer,
            flow_spec, n_steps, device, use_amp, amp_dtype
        )
        pred_1mm = np.zeros(vol_1mm_norm.shape, dtype=np.float32)
        pred_1mm[sh:sh + ph, sw:sw + pw, sd:sd + pd] = pred_crop
    else:
        patches, positions, padded_shape = _extract_patches(
            vol_1mm_norm, patch_size, stride, pad=pad, center_aligned=center_aligned
        )
        blend_weights = _create_blend_weights(patch_size, mode=blend_mode)

        patch_outputs = []
        for patch in patches:
            if norm_mode == "per_patch":
                patch = _normalize_global(patch, p_lo, p_hi)
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
            pred_patch = _infer_patch_unified(
                patch_tensor, vae, model, model_type, latent_vectorizer,
                flow_spec, n_steps, device, use_amp, amp_dtype
            )
            patch_outputs.append(torch.from_numpy(pred_patch).unsqueeze(0).unsqueeze(0).float())

        pred_1mm = _blend_patches(
            patch_outputs, positions, blend_weights, padded_shape, vol_1mm_norm.shape, pad
        )

    img_pred_1mm = nib.Nifti1Image(pred_1mm.astype(np.float32), img_src_1mm.affine)
    img_pred_05mm = nib_proc.resample_from_to(img_pred_1mm, img_src, order=1)
    pred_05mm = img_pred_05mm.get_fdata(dtype=np.float32)

    vol_src_05mm = img_src.get_fdata(dtype=np.float32)
    mask = vol_src_05mm > 1e-6
    pred_05mm_masked = pred_05mm * mask

    return pred_05mm_masked.astype(np.float32), affine_src, header_src


def _parse_official_filename(name: str) -> Optional[Dict]:
    pattern = re.compile(r"^P_([A-Z0-9]+)_([\d\.]+T)_(\d{4})\.nii.*$")
    m = pattern.match(name)
    if m:
        return {"modality": m.group(1), "field": m.group(2), "subject": m.group(3)}
    return None


def infer_single(
    cfg_path: str,
    checkpoint: Optional[str],
    input_path: str,
    tgt_field: str,
    output_path: Optional[str] = None,
    src_field: Optional[str] = None,
    modality: Optional[str] = None,
    env_path=None,
    n_steps: Optional[int] = None,
    norm_mode: Optional[str] = None,
    center_crop_only: bool = False,
    use_ema: bool = True,
    device: str = "cuda",
):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    parsed = _parse_official_filename(input_path.name)
    if parsed is None:
        raise ValueError(f"Filename does not match official format: {input_path.name}")

    mod = modality or parsed["field"]
    src = src_field or parsed["field"]

    cfg = load_yaml_with_include(cfg_path)
    cfg = resolve_paths(cfg, load_env(env_path))

    checkpoint = str(_resolve_checkpoint(cfg, checkpoint))
    n_steps = _resolve_n_steps(cfg, n_steps)
    norm_mode = _resolve_norm_mode(cfg, norm_mode)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    amp_dtype_name = train_cfg.get("amp_dtype", "bf16")
    use_amp = bool(train_cfg.get("use_amp", True))
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16

    p_lo = data_cfg.get("percentile_lower", 0.5)
    p_hi = data_cfg.get("percentile_upper", 99.5)
    raw_vs = data_cfg.get("volume_size", None)
    patch_size = tuple(int(v) for v in raw_vs) if raw_vs else (128, 128, 80)
    stride = tuple(max(s // 2, 1) for s in patch_size)
    pad = 16

    modalities = data_cfg.get("modalities", MODALITIES)
    fields = data_cfg.get("fields", DOMAINS)
    n_fields = len(fields)
    n_classes = len(modalities) * n_fields

    if mod not in modalities:
        raise ValueError(f"Unknown modality '{mod}'. Config has {modalities}")
    if src not in fields:
        raise ValueError(f"Unknown source field '{src}'. Config has {fields}")
    if tgt_field not in fields:
        raise ValueError(f"Unknown target field '{tgt_field}'. Config has {fields}")

    mod_idx = modalities.index(mod)
    src_field_idx = fields.index(src)
    tgt_field_idx = fields.index(tgt_field)
    
    # V2 method flag
    use_v2 = (cfg.get("method", "mmfm3d") == "mmfm3d_vectorized_v2")

    print(f"[{mod}] {src} → {tgt_field} | Loading VAE...")
    vae = load_vae(cfg, dev)
    if vae.latent_format != "spatial":
        raise RuntimeError(f"Requires spatial VAE, got {vae.latent_format}")

    print("Loading flow model...")
    model, model_type, latent_shape = _build_model(cfg, vae, dev)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    loaded_from = _load_model_weights(model, ckpt, use_ema, model_type)
    print(f"  Model loaded ({loaded_from}, iter={ckpt.get('iter', '?')}) from {checkpoint}")

    latent_vectorizer = LatentVectorizer(latent_shape) if model_type == "vectorial" else None
    flow_spec = _make_flow_spec(model_type, mod_idx, src_field_idx, tgt_field_idx, n_fields, use_v2)

    t0 = time.time()
    pred_vol, affine, header = process_volume_unified(
        input_path, vae, model, model_type, latent_vectorizer,
        flow_spec=flow_spec, n_steps=n_steps, patch_size=patch_size, stride=stride, pad=pad,
        p_lo=p_lo, p_hi=p_hi, device=dev, use_amp=use_amp, amp_dtype=amp_dtype,
        norm_mode=norm_mode, center_crop_only=center_crop_only,
    )

    if output_path:
        out_path = Path(output_path)
    else:
        sid = parsed["subject"]
        out_path = Path(f"P_{mod}_{tgt_field}_{sid}.nii.gz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nib.save(nib.Nifti1Image(pred_vol, affine, header), str(out_path))
    print(f"  {input_path.name} → {out_path}  ({time.time() - t0:.1f}s)")
    return out_path


def infer_batch(
    cfg_path: str,
    checkpoint: Optional[str],
    output_dir: str,
    split: str = "Training_prospective",
    modalities=None,
    pairs_filter=None,
    env_path=None,
    n_steps: Optional[int] = None,
    norm_mode: Optional[str] = None,
    center_crop_only: bool = False,
    use_ema: bool = True,
    skip_existing: bool = False,
    device: str = "cuda",
):
    cfg = load_yaml_with_include(cfg_path)
    cfg = resolve_paths(cfg, load_env(env_path))

    checkpoint = str(_resolve_checkpoint(cfg, checkpoint))
    n_steps = _resolve_n_steps(cfg, n_steps)
    norm_mode = _resolve_norm_mode(cfg, norm_mode)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    amp_dtype_name = train_cfg.get("amp_dtype", "bf16")
    use_amp = bool(train_cfg.get("use_amp", True))
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16

    p_lo = data_cfg.get("percentile_lower", 0.5)
    p_hi = data_cfg.get("percentile_upper", 99.5)
    raw_vs = data_cfg.get("volume_size", None)
    patch_size = tuple(int(v) for v in raw_vs) if raw_vs else (128, 128, 80)
    stride = tuple(max(s // 2, 1) for s in patch_size)
    pad = 16

    all_modalities = data_cfg.get("modalities", MODALITIES)
    fields = data_cfg.get("fields", DOMAINS)
    n_fields = len(fields)
    n_classes = len(all_modalities) if cfg.get("method", "mmfm3d") == "mmfm3d_vectorized_v2" else len(all_modalities) * n_fields
    
    # V2 method flag
    use_v2 = (cfg.get("method", "mmfm3d") == "mmfm3d_vectorized_v2")

    modalities = modalities if modalities is not None else all_modalities

    print("Loading VAE...")
    vae = load_vae(cfg, dev)
    if vae.latent_format != "spatial":
        raise RuntimeError(f"Requires spatial VAE, got {vae.latent_format}")

    print("Loading flow model...")
    model, model_type, latent_shape = _build_model(cfg, vae, dev)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    loaded_from = _load_model_weights(model, ckpt, use_ema, model_type)
    print(f"  Model loaded ({loaded_from}, iter={ckpt.get('iter', '?')}) from {checkpoint}")

    latent_vectorizer = LatentVectorizer(latent_shape) if model_type == "vectorial" else None

    data_root_env = cfg.get("data_root") or cfg.get("data", {}).get("data_root")
    data_root = Path(data_root_env) if data_root_env else Path("/home/rousseau/Data/MRIxFields_20260414")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if pairs_filter is not None:
        task_pairs = pairs_filter
    else:
        task_pairs = [(s, t) for s in fields for t in fields if s != t]

    start_all = time.time()
    for mod in modalities:
        mod_idx = all_modalities.index(mod)
        for src, tgt in task_pairs:
            flow_spec = _make_flow_spec(
                model_type, mod_idx, fields.index(src), fields.index(tgt), n_fields, use_v2
            )
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
                pred_vol, affine, header = process_volume_unified(
                    nii_path, vae, model, model_type, latent_vectorizer,
                    flow_spec=flow_spec, n_steps=n_steps, patch_size=patch_size, stride=stride, pad=pad,
                    p_lo=p_lo, p_hi=p_hi, device=dev, use_amp=use_amp, amp_dtype=amp_dtype,
                    norm_mode=norm_mode, center_crop_only=center_crop_only,
                )
                nib.save(nib.Nifti1Image(pred_vol, affine, header), str(out_path))
                print(f"  {nii_path.name} → {out_path}  ({time.time() - t0:.1f}s)")

    elapsed = time.time() - start_all
    print(f"\n✅ Batch inference done in {elapsed / 3600:.2f} h")
    print(f"   Predictions saved to: {out_root}")


def parse_args():
    p = argparse.ArgumentParser(description="MMFM unified inference — vectorial or UNet")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--input", default=None, help="Single input NIfTI path")
    p.add_argument("--tgt-field", default=None, help="Target field for single mode")
    p.add_argument("--src-field", default=None)
    p.add_argument("--modality", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--output_dir", default=None, help="Batch mode output root")
    p.add_argument("--split", default="Training_prospective",
                   choices=["Training_prospective", "Validating_prospective", "Testing_prospective"])
    p.add_argument("--modalities", nargs="+", default=None)
    p.add_argument("--pairs", default=None, help="Subset of pairs, e.g. '0.1T_to_7T,1.5T_to_3T'")
    p.add_argument("--env", default="local")
    p.add_argument("--n_steps", type=int, default=None)
    p.add_argument("--norm_mode", default=None, choices=["global", "crop_percentile", "per_patch"])
    p.add_argument("--center_crop_only", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    if args.input:
        infer_single(
            cfg_path=args.config, checkpoint=args.checkpoint, input_path=args.input,
            tgt_field=args.tgt_field, output_path=args.output,
            src_field=args.src_field, modality=args.modality, env_path=args.env,
            n_steps=args.n_steps, norm_mode=args.norm_mode,
            center_crop_only=args.center_crop_only,
            use_ema=not args.no_ema, device=args.device,
        )
    else:
        if not args.output_dir:
            print("ERROR: --output_dir required in batch mode", file=sys.stderr)
            sys.exit(2)
        pairs_filter = None
        if args.pairs:
            pairs_filter = []
            for pair_str in args.pairs.split(","):
                src, tgt = pair_str.strip().split("_to_")
                pairs_filter.append((src, tgt))
        infer_batch(
            cfg_path=args.config, checkpoint=args.checkpoint, output_dir=args.output_dir,
            split=args.split, modalities=args.modalities, pairs_filter=pairs_filter,
            env_path=args.env, n_steps=args.n_steps, norm_mode=args.norm_mode,
            center_crop_only=args.center_crop_only,
            use_ema=not args.no_ema, skip_existing=args.skip_existing, device=args.device,
        )


if __name__ == "__main__":
    main()
