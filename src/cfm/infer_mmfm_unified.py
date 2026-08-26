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
        --config configs/mmfm/vectorized.yaml \
        --checkpoint outputs/mmfm/vectorized/weights/model_final.pth \
        --input /path/to/P_T1W_0.1T_0006.nii.gz --tgt-field 3T --output /tmp/pred.nii.gz

Usage batch Task 3:
    PYTHONPATH=src python src/cfm/infer_mmfm_unified.py \
        --config configs/mmfm/unet.yaml \
        --checkpoint outputs/mmfm/unet/weights/model_final.pth \
        --output_dir outputs/mmfm/unet/predictions/task3 \
        --split Training_prospective --modalities T1W
"""

import argparse
import json
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

from cfm import arch_inr, arch_unet, arch_vector
from models.tiled_vae import tiled_decode, tiled_encode
from cfm.mmfm_core import euler_integrate, _infer_latent_shape, _flat_class, _field_to_time
from common.config import load_yaml_with_include, load_env, resolve_paths
from common.io import DOMAINS, MODALITIES, denormalize_from_01, center_crop_or_pad_np
from models.vae_loader import load_vae

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


def _build_model(cfg: dict, vae, device: torch.device, volume_size: Tuple[int, int, int]):
    """Build and return the flow model (vectorial MLP or UNet), plus its
    ArchAdapter (see mmfm_core.py) so the caller can encode/decode/checkpoint
    -load through the same architecture-agnostic contract used at training
    time.

    Supports both the current unified method strings (mmfm3d_vectorized,
    mmfm3d_unet) and the legacy pre-unification ones still referenced by
    historical checkpoints/configs (mmfm3d, mmfm3d_vectorized_v1/v2,
    mmfm3d_unet_v2) — this script loads and evaluates OLD checkpoints too,
    not just ones trained via train_mmfm_unified.py.
    """
    method = cfg.get("method", "mmfm3d")
    data_cfg = cfg["data"]
    modalities = data_cfg.get("modalities", MODALITIES)
    fields = data_cfg.get("fields", DOMAINS)
    n_fields = len(fields)
    # Legacy v1 conditions on (modality, field) jointly (n_classes=15); every
    # other scheme (v2, unified UNet/vectorized) conditions on modality alone.
    n_classes = len(modalities) * n_fields if method == "mmfm3d_vectorized_v1" else len(modalities)

    if method in ("mmfm3d", "mmfm", "mmfm3d_vectorized", "mmfm3d_vectorized_v1", "mmfm3d_vectorized_v2"):
        latent_shape = _infer_latent_shape(vae, volume_size, device)
        adapter = arch_vector.make_adapter(cfg, latent_shape, n_classes)
        mmfm = adapter.build_model().to(device)
        return mmfm, "vectorial", latent_shape, adapter

    elif method in ("mmfm3d_unet", "mmfm3d_unet_v2"):
        latent_shape = _infer_latent_shape(vae, volume_size, device)
        adapter = arch_unet.make_adapter(cfg, latent_shape, len(modalities))
        unet = adapter.build_model().to(device)
        return unet, "unet", latent_shape, adapter

    elif method == "mmfm3d_inr":
        # vae here is IdentityVAEWrapper — _infer_latent_shape(vae, ...) is a
        # no-op dummy pass through it, returning (channels, *volume_size),
        # exactly what arch_inr.make_adapter expects (see its docstring).
        latent_shape = _infer_latent_shape(vae, volume_size, device)
        adapter = arch_inr.make_adapter(cfg, latent_shape, len(modalities))
        inr_flow = adapter.build_model().to(device)
        return inr_flow, "inr", latent_shape, adapter

    raise ValueError(f"Unsupported method for unified inference: {method}")


def _load_model_weights(model, ckpt: dict, use_ema: bool, adapter):
    """Load EMA/model weights into the flow model via the adapter's
    checkpoint_key_remap (MONAI-version compat shim for UNet, no-op for the
    vectorial MLP)."""
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

    model.load_state_dict(adapter.checkpoint_key_remap(state))
    model.eval()
    return loaded_from


def _make_flow_spec(model_type: str, mod_idx: int, src_field_idx: int,
                    tgt_field_idx: int, n_fields: int, use_v1: bool = False) -> dict:
    """Build the (target class, time interval) flow spec for one translation.

    - vectorial v1 (legacy): conditions on the flat (modality, target-field)
      class, integrates over the fixed t in [0,1] (no field-based time axis).
    - vectorial v2/unified, and unet (multi-marginal): condition on the
      modality/contrast alone; the field is the time axis (t_start/t_end).
    """
    if model_type == "vectorial" and use_v1:
        return {
            "y": _flat_class(mod_idx, tgt_field_idx, n_fields),
            "t_start": 0.0, "t_end": 1.0,
        }
    return {
        "y": mod_idx,
        "t_start": _field_to_time(src_field_idx, n_fields),
        "t_end": _field_to_time(tgt_field_idx, n_fields),
    }


def _infer_patch_unified(
    patch_tensor: torch.Tensor,
    vae,
    model,
    adapter,
    flow_spec: dict,
    n_steps: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    encode_tile=None,
    encode_tile_margin: int = 16,
) -> np.ndarray:
    """Run VAE encode -> flow -> VAE decode on a single patch. Identical for
    both architectures — the adapter absorbs the flatten/pad and call-
    convention differences (see mmfm_core.py).

    `encode_tile` (non None) encode/décode le patch PAR TUILES : au-delà de 2mm
    MedVAE ne tient pas en mémoire sur un volume entier (attention quadratique
    au goulot, OOM dès 192x224x192). Doit reprendre EXACTEMENT les valeurs
    `data.encode_tile`/`encode_tile_margin` utilisées au precompute, sinon les
    latents d'inférence ne seraient pas dans la même distribution que ceux du
    cache d'entraînement."""
    model_fn = adapter.make_model_fn(model)
    with torch.no_grad(), torch.amp.autocast(
        "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
    ):
        if encode_tile is not None:
            z_src_enc = tiled_encode(vae, patch_tensor, tile=tuple(encode_tile),
                                     margin=encode_tile_margin,
                                     use_amp=use_amp, amp_dtype=amp_dtype)
        else:
            z_src_enc = vae.encode(patch_tensor)
        z_src, meta = adapter.prep_latent(vae, z_src_enc)
        z_src = z_src.float()
        y = torch.tensor([flow_spec["y"]], dtype=torch.long, device=device)
        z_tgt = euler_integrate(
            model_fn, z_src, y, flow_spec["t_start"], flow_spec["t_end"],
            n_steps, device, use_amp, amp_dtype,
        )
        z_tgt = adapter.restore_latent(z_tgt, meta)
        if encode_tile is not None:
            recon = tiled_decode(vae, z_tgt, tile=tuple(encode_tile),
                                 margin=encode_tile_margin,
                                 use_amp=use_amp, amp_dtype=amp_dtype)
        else:
            recon = vae.decode(z_tgt)

    pred = recon.squeeze().cpu().float().numpy()
    return (np.clip(pred, -1.0, 1.0) + 1.0) / 2.0


def process_volume_unified(
    nii_path: Path,
    vae,
    model,
    model_type: str,
    adapter,
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
    fixed_lo: Optional[float] = None,
    fixed_hi: Optional[float] = None,
    tgt_lo: Optional[float] = None,
    tgt_hi: Optional[float] = None,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    compat_orientation: bool = False,
    compat_source_norm: bool = False,
    upsample_order: int = 1,
    encode_tile=None,
    encode_tile_margin: int = 16,
):
    """Full-resolution prediction, dispatching vectorial or UNet model.

    Variable names below (vol_1mm, img_pred_1mm, ...) are historical — the
    actual resampling target is `target_spacing`, not necessarily 1mm.
    Must match training's data.target_spacing for a valid comparison: the
    model only ever saw inputs resampled to that spacing.
    """
    img_src = nib.load(str(nii_path))
    affine_src = img_src.affine.copy()
    header_src = img_src.header.copy()

    img_src_1mm = nib_proc.resample_to_output(img_src, voxel_sizes=target_spacing, order=1)
    vol_1mm = img_src_1mm.get_fdata(dtype=np.float32)

    if norm_mode == "field_fixed":
        # Fixed per-(modality, source field) percentile bounds from
        # compute_field_norm_stats.py, matching the normalization the model
        # was trained with (see MultiModalNIfTILatentDataset / field_norm_stats)
        # instead of per-volume percentiles computed on the fly — the latter
        # erases genuine field-to-field dynamic-range differences (5T/7T
        # natively occupy a much narrower range than 0.1T/1.5T/3T), which is
        # the root cause of the intensity-amplification bug on those targets.
        if fixed_lo is None or fixed_hi is None:
            raise ValueError(
                "norm_mode='field_fixed' requires fixed_lo/fixed_hi "
                "(pass --field_norm_stats and let the caller resolve them)."
            )
        vol_1mm_norm = _normalize_with_params(vol_1mm, fixed_lo, fixed_hi)
    elif norm_mode == "crop_percentile":
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

    if compat_orientation:
        # ORIENTATION. `resample_to_output` ci-dessus réoriente en canonique RAS ;
        # le PRECOMPUTE (`common.io.load_nifti_volume` -> `resample_volume`,
        # scipy.zoom sur le tableau brut) garde la native, ici LAS. L'axe 0 est
        # donc INVERSÉ entre le volume sur lequel le flow a été entraîné et celui
        # qu'il reçoit. Mesuré : 0.343 d'écart L2 tel quel, 0.057 après
        # retournement (results/mmfm/audit_20260825/manifest.md).
        #
        # On retourne ici et on remet à l'endroit sur la prédiction plutôt que
        # d'appeler le chemin du precompute : celui-ci ne maintient pas d'affine,
        # dont on a besoin pour revenir en 0.5mm.
        #
        # `flip_lr_prob: 0.5` rend le vectorisé et l'UNet insensibles au miroir,
        # d'où l'invisibilité du bug pendant des mois ; `arch_inr.py` REFUSE le
        # flip (un vecteur de modulation global n'a pas de structure spatiale à
        # retourner), donc l'INR était la seule sans tolérance.
        vol_1mm_norm = np.ascontiguousarray(vol_1mm_norm[::-1])
    if compat_source_norm:
        # NORMALISATION DE LA SOURCE. Le cache INR de production porte
        # `field_norm_stats_path: None` : il a été construit avec les percentiles
        # PAR VOLUME. La dénormalisation de SORTIE reste celle du champ cible,
        # seule échelle disponible à l'inférence.
        vn = _normalize_global(vol_1mm, p_lo, p_hi)
        vol_1mm_norm = np.ascontiguousarray(vn[::-1]) if compat_orientation else vn

    if center_crop_only:
        # center_crop_or_pad_np (not raw slicing) so this also handles the
        # "fullfov" case where patch_size exceeds the native resampled FOV
        # (e.g. vectorized/UNet harmonized comparison, volume_size=96x112x96
        # @ 2mm > native ~91x109x91) — plain slicing would silently return a
        # smaller-than-expected array and either crash the vectorial model
        # (fixed flat_dim) or feed the UNet a shape it never trained on.
        native_shape = vol_1mm_norm.shape
        crop = center_crop_or_pad_np(vol_1mm_norm, patch_size)
        crop_tensor = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).to(device)
        pred_crop = _infer_patch_unified(
            crop_tensor, vae, model, adapter,
            flow_spec, n_steps, device, use_amp, amp_dtype,
            encode_tile=encode_tile, encode_tile_margin=encode_tile_margin,
        )
        # Invert the same crop/pad symmetrically back to the native shape.
        pred_1mm = center_crop_or_pad_np(pred_crop, native_shape)
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
                patch_tensor, vae, model, adapter,
                flow_spec, n_steps, device, use_amp, amp_dtype,
                encode_tile=encode_tile, encode_tile_margin=encode_tile_margin,
            )
            patch_outputs.append(torch.from_numpy(pred_patch).unsqueeze(0).unsqueeze(0).float())

        pred_1mm = _blend_patches(
            patch_outputs, positions, blend_weights, padded_shape, vol_1mm_norm.shape, pad
        )

    if compat_orientation:
        pred_1mm = np.ascontiguousarray(pred_1mm[::-1])

    img_pred_1mm = nib.Nifti1Image(pred_1mm.astype(np.float32), img_src_1mm.affine)
    # Dernier reechantillonnage de la chaine : de la resolution de travail vers la
    # grille native 0.5mm sur laquelle l'evaluateur compare. `upsample_order` par
    # defaut a 1 (lineaire) -- c'est ce qui a produit TOUS les chiffres publies.
    #
    # L'ordre 3 a ete teste le 2026-08-26 et le resultat est NEUTRE, contrairement a
    # ce que le plafond laissait esperer :
    #   - sur un volume PARFAIT (aller-retour 0.5 -> 1 -> 0.5mm), l'ordre 3 fait
    #     passer la nettete de 0.615 a 0.910 et le nRMSE de 0.0382 a 0.0251 : le
    #     plafond de la chaine monte nettement ;
    #   - de bout en bout, il ne se passe presque rien. Vectorise T1W nRMSE
    #     0.4353 -> 0.4374, SSIM 0.8997 -> 0.8966, LPIPS 0.0983 -> 0.0969 ; INR
    #     0.4070 -> 0.4075 / 0.8606 -> 0.8602 / 0.1572 -> 0.1561.
    # Lecture : le plafond n'est PAS contraignant, le modele n'a rien a mettre dans
    # la bande de frequences que l'ordre 3 laisse passer. Un peu de LPIPS gagne, un
    # peu de nRMSE perdu -- un arbitrage perception/distorsion, pas un gain.
    # Le defaut reste donc 1, pour ne pas invalider les resultats en place.
    # Voir results/mmfm/audit_20260826_order3/manifest.md.
    img_pred_05mm = nib_proc.resample_from_to(img_pred_1mm, img_src, order=upsample_order)
    pred_05mm = img_pred_05mm.get_fdata(dtype=np.float32)

    if norm_mode == "field_fixed":
        # Model output stays in the training-time always-full-[0,1]-range
        # scale (matching the source's stretched normalize_volume_fixed
        # domain) until explicitly mapped back to the TARGET field's native
        # (narrow) range — otherwise it's compared against native-scale GT
        # at a mismatched intensity range. See common.io.denormalize_from_01.
        if tgt_lo is None or tgt_hi is None:
            raise ValueError(
                "norm_mode='field_fixed' requires tgt_lo/tgt_hi (target field's "
                "field_norm_stats entry) to denormalize the prediction."
            )
        pred_05mm = denormalize_from_01(pred_05mm, tgt_lo, tgt_hi)

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
    field_norm_stats_path: Optional[str] = None,
    compat_orientation: bool = False,
    compat_source_norm: bool = False,
    upsample_order: int = 1,
):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    parsed = _parse_official_filename(input_path.name)
    if parsed is None:
        raise ValueError(f"Filename does not match official format: {input_path.name}")

    mod = modality or parsed["field"]
    src = src_field or parsed["field"]

    fixed_lo = fixed_hi = tgt_lo = tgt_hi = None
    if field_norm_stats_path:
        with open(field_norm_stats_path) as _f:
            field_norm_stats = json.load(_f)["stats"]
        entry = field_norm_stats.get(mod, {}).get(src)
        if entry is None:
            raise RuntimeError(
                f"field_norm_stats fourni mais entrée manquante pour {mod}@{src}"
            )
        fixed_lo, fixed_hi = entry["lo"], entry["hi"]
        tgt_entry = field_norm_stats.get(mod, {}).get(tgt_field)
        if tgt_entry is None:
            raise RuntimeError(
                f"field_norm_stats fourni mais entrée manquante pour {mod}@{tgt_field}"
            )
        tgt_lo, tgt_hi = tgt_entry["lo"], tgt_entry["hi"]
        print(f"  Normalisation par champ (fixe) : {field_norm_stats_path} -> "
              f"src lo={fixed_lo:.4f} hi={fixed_hi:.4f} | tgt lo={tgt_lo:.4f} hi={tgt_hi:.4f}")

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
    raw_ts = data_cfg.get("target_spacing", None)
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else (1.0, 1.0, 1.0)
    # Tuilage MedVAE : DOIT reprendre les valeurs du precompute (cf. data.encode_tile)
    _raw_tile = data_cfg.get("encode_tile", None)
    encode_tile = tuple(int(v) for v in _raw_tile) if _raw_tile else None
    encode_tile_margin = int(data_cfg.get("encode_tile_margin", 16))

    modalities = data_cfg.get("modalities", MODALITIES)
    fields = data_cfg.get("fields", DOMAINS)
    n_fields = len(fields)

    if mod not in modalities:
        raise ValueError(f"Unknown modality '{mod}'. Config has {modalities}")
    if src not in fields:
        raise ValueError(f"Unknown source field '{src}'. Config has {fields}")
    if tgt_field not in fields:
        raise ValueError(f"Unknown target field '{tgt_field}'. Config has {fields}")

    mod_idx = modalities.index(mod)
    src_field_idx = fields.index(src)
    tgt_field_idx = fields.index(tgt_field)

    # Legacy v1 method flag (see _make_flow_spec)
    use_v1 = (cfg.get("method", "mmfm3d") == "mmfm3d_vectorized_v1")

    print(f"[{mod}] {src} → {tgt_field} | Loading VAE...")
    vae = load_vae(cfg, dev)
    if vae.latent_format != "spatial":
        raise RuntimeError(f"Requires spatial VAE, got {vae.latent_format}")

    print("Loading flow model...")
    model, model_type, latent_shape, adapter = _build_model(cfg, vae, dev, patch_size)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    loaded_from = _load_model_weights(model, ckpt, use_ema, adapter)
    print(f"  Model loaded ({loaded_from}, iter={ckpt.get('iter', '?')}) from {checkpoint}")

    flow_spec = _make_flow_spec(model_type, mod_idx, src_field_idx, tgt_field_idx, n_fields, use_v1)

    t0 = time.time()
    pred_vol, affine, header = process_volume_unified(
        input_path, vae, model, model_type, adapter,
        flow_spec=flow_spec, n_steps=n_steps, patch_size=patch_size, stride=stride, pad=pad,
        p_lo=p_lo, p_hi=p_hi, device=dev, use_amp=use_amp, amp_dtype=amp_dtype,
        norm_mode=norm_mode, center_crop_only=center_crop_only,
        fixed_lo=fixed_lo, fixed_hi=fixed_hi, tgt_lo=tgt_lo, tgt_hi=tgt_hi,
                    compat_orientation=compat_orientation,
                    compat_source_norm=compat_source_norm,
                    upsample_order=upsample_order,
        target_spacing=target_spacing,
        encode_tile=encode_tile, encode_tile_margin=encode_tile_margin,
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
    field_norm_stats_path: Optional[str] = None,
    compat_orientation: bool = False,
    compat_source_norm: bool = False,
    upsample_order: int = 1,
):
    field_norm_stats = None
    if field_norm_stats_path:
        with open(field_norm_stats_path) as _f:
            field_norm_stats = json.load(_f)["stats"]
        print(f"Normalisation par champ (fixe) : {field_norm_stats_path}")

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
    raw_ts = data_cfg.get("target_spacing", None)
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else (1.0, 1.0, 1.0)
    # Tuilage MedVAE : DOIT reprendre les valeurs du precompute (cf. data.encode_tile)
    _raw_tile = data_cfg.get("encode_tile", None)
    encode_tile = tuple(int(v) for v in _raw_tile) if _raw_tile else None
    encode_tile_margin = int(data_cfg.get("encode_tile_margin", 16))

    all_modalities = data_cfg.get("modalities", MODALITIES)
    fields = data_cfg.get("fields", DOMAINS)
    n_fields = len(fields)

    # Legacy v1 method flag (see _make_flow_spec)
    use_v1 = (cfg.get("method", "mmfm3d") == "mmfm3d_vectorized_v1")

    modalities = modalities if modalities is not None else all_modalities

    print("Loading VAE...")
    vae = load_vae(cfg, dev)
    if vae.latent_format != "spatial":
        raise RuntimeError(f"Requires spatial VAE, got {vae.latent_format}")

    print("Loading flow model...")
    model, model_type, latent_shape, adapter = _build_model(cfg, vae, dev, patch_size)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    loaded_from = _load_model_weights(model, ckpt, use_ema, adapter)
    print(f"  Model loaded ({loaded_from}, iter={ckpt.get('iter', '?')}) from {checkpoint}")

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
                model_type, mod_idx, fields.index(src), fields.index(tgt), n_fields, use_v1
            )
            pair_out_dir = out_root / "task3" / mod / f"{src}_to_{tgt}"
            pair_out_dir.mkdir(parents=True, exist_ok=True)

            fixed_lo = fixed_hi = tgt_lo = tgt_hi = None
            if field_norm_stats is not None:
                entry = field_norm_stats.get(mod, {}).get(src)
                tgt_entry = field_norm_stats.get(mod, {}).get(tgt)
                if entry is None or tgt_entry is None:
                    print(f"[WARN] field_norm_stats sans entrée pour {mod}@{src} ou {mod}@{tgt}, skip pair")
                    continue
                fixed_lo, fixed_hi = entry["lo"], entry["hi"]
                tgt_lo, tgt_hi = tgt_entry["lo"], tgt_entry["hi"]

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
                    nii_path, vae, model, model_type, adapter,
                    flow_spec=flow_spec, n_steps=n_steps, patch_size=patch_size, stride=stride, pad=pad,
                    p_lo=p_lo, p_hi=p_hi, device=dev, use_amp=use_amp, amp_dtype=amp_dtype,
                    norm_mode=norm_mode, center_crop_only=center_crop_only,
                    fixed_lo=fixed_lo, fixed_hi=fixed_hi, tgt_lo=tgt_lo, tgt_hi=tgt_hi,
                    compat_orientation=compat_orientation,
                    compat_source_norm=compat_source_norm,
                    upsample_order=upsample_order,
                    target_spacing=target_spacing,
                    encode_tile=encode_tile, encode_tile_margin=encode_tile_margin,
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
    p.add_argument("--norm_mode", default=None,
                   choices=["global", "crop_percentile", "per_patch", "field_fixed"])
    p.add_argument("--field_norm_stats", default=None,
                   help="Chemin JSON produit par compute_field_norm_stats.py — requis avec "
                        "--norm_mode field_fixed")
    p.add_argument("--center_crop_only", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--upsample_order", type=int, default=None,
                   help="ordre d'interpolation du dernier reechantillonnage vers 0.5mm "
                        "(defaut 1 = celui de tous les chiffres publies ; 3 teste et NEUTRE)")
    p.add_argument("--compat_orientation", action="store_true", default=None,
                   help="Aligne l'orientation de l'inference sur celle du precompute "
                        "(defaut : cle inference.compat_orientation de la config).")
    p.add_argument("--compat_source_norm", action="store_true", default=None,
                   help="Normalise la source par ses propres percentiles, comme le cache "
                        "(defaut : cle inference.compat_source_norm de la config).")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def _flag(cfg_infer: dict, key: str, cli_value) -> bool:
    """Drapeau CLI prioritaire sur la config, config prioritaire sur le défaut."""
    return bool(cfg_infer.get(key, False)) if cli_value is None else bool(cli_value)


def _check_cache_consistency(cfg: dict, norm_mode: str, compat_source_norm: bool) -> None:
    """GARDE-FOU — le cache de latents dit avec quelle normalisation il a été bâti ;
    l'inférence doit s'y conformer.

    C'est exactement le contrôle qui manquait : le cache INR de production
    (`inr_932b0b37`) porte `field_norm_stats_path: None`, donc des percentiles PAR
    VOLUME, tandis que l'inférence tournait en `field_fixed`. Le flow recevait un
    latent déplacé de 69 % en L2 relatif. Rien ne le signalait, parce que
    `cache_prebakes_prep=True` fait que l'entraînement ne rappelle jamais
    `prep_latent` : les deux chemins ne se croisent nulle part.
    """
    cache_dir = cfg.get("data", {}).get("latent_cache_dir")
    if not cache_dir:
        return
    index = Path(cache_dir) / "index.json"
    if not index.exists():
        return
    with open(index) as f:
        built_with_field_norm = json.load(f).get("field_norm_stats_path") is not None
    infer_uses_field_norm = (norm_mode == "field_fixed") and not compat_source_norm
    if built_with_field_norm != infer_uses_field_norm:
        raise SystemExit(
            "INCOHÉRENCE cache / inférence sur la normalisation de la source.\n"
            f"  cache     {index} : field_norm_stats_path "
            f"{'renseigné' if built_with_field_norm else 'ABSENT (percentiles par volume)'}\n"
            f"  inférence : norm_mode={norm_mode!r}, compat_source_norm={compat_source_norm}\n"
            "Le flow recevrait un latent hors de la distribution sur laquelle il a été\n"
            "entraîné. Mettre inference.compat_source_norm à true (ou régénérer le cache\n"
            "avec --field-norm-stats). Voir results/mmfm/audit_20260825/manifest.md."
        )


def main():
    args = parse_args()
    cfg_infer = load_yaml_with_include(args.config).get("inference", {}) or {}
    norm_mode = _resolve_norm_mode({"inference": cfg_infer}, args.norm_mode)
    compat_source_norm = _flag(cfg_infer, "compat_source_norm", args.compat_source_norm)
    _check_cache_consistency(
        resolve_paths(load_yaml_with_include(args.config), load_env(args.env)),
        norm_mode, compat_source_norm,
    )
    if args.input:
        infer_single(
            cfg_path=args.config, checkpoint=args.checkpoint, input_path=args.input,
            tgt_field=args.tgt_field, output_path=args.output,
            src_field=args.src_field, modality=args.modality, env_path=args.env,
            n_steps=args.n_steps, norm_mode=args.norm_mode,
            center_crop_only=args.center_crop_only,
            use_ema=not args.no_ema, device=args.device,
            compat_orientation=_flag(cfg_infer, 'compat_orientation', args.compat_orientation),
            compat_source_norm=_flag(cfg_infer, 'compat_source_norm', args.compat_source_norm),
            upsample_order=int(args.upsample_order if args.upsample_order is not None
                               else cfg_infer.get('upsample_order', 1)),
            field_norm_stats_path=args.field_norm_stats,
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
            compat_orientation=_flag(cfg_infer, 'compat_orientation', args.compat_orientation),
            compat_source_norm=_flag(cfg_infer, 'compat_source_norm', args.compat_source_norm),
            upsample_order=int(args.upsample_order if args.upsample_order is not None
                               else cfg_infer.get('upsample_order', 1)),
            field_norm_stats_path=args.field_norm_stats,
        )


if __name__ == "__main__":
    main()
