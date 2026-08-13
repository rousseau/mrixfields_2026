#!/usr/bin/env python3
"""Lightweight validation-curve computation across training checkpoints.

Not the official leaderboard metric: nRMSE is computed directly in our own
working grid/normalization (target_spacing/volume_size from the config,
field_norm_stats-normalized [0,1]-ish range) instead of the official
evaluator's resample-to-native-364x436x364 + subprocess-per-pair pipeline
(src/evaluation/evaluate.py). This trades absolute comparability to the
leaderboard for speed, so we can afford one point per checkpoint instead of
only the training endpoint.

Usage:
  PYTHONPATH=src python src/cfm/validation_curve.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import nibabel as nib
import nibabel.processing
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cfm import arch_vector
from cfm.mmfm_core import euler_integrate, _infer_latent_shape, _field_to_time
from cfm.mmfm_vectorized import LatentVectorizer
from common.config import load_yaml_with_include, load_env, resolve_paths
from common.io import load_nifti_volume
from models.vae_loader import load_vae

DATA_ROOT = Path("/home/rousseau/Data/MRIxFields_20260414/Training_prospective")
SUBJECTS = ["0006", "0007", "0009"]
PAIRS = [("0.1T", "7T"), ("3T", "7T"), ("0.1T", "3T")]
MODALITY = "T1W"

RUNS = {
    "phase1": {
        "config": "configs/mmfm3d_medvae_multimodal_v2_fullfov_regularized.yaml",
        "weights_dir": "outputs/cfm3d/runs/mmfm3d_medvae_multimodal_vectorized_v2_fullfov_regularized/weights",
        "iters": list(range(5000, 50001, 5000)),
        "x_offset": 0,
    },
    "phase2": {
        "config": "configs/mmfm3d_medvae_multimodal_v2_fullfov_regularized_v2.yaml",
        "weights_dir": "outputs/cfm3d/runs/mmfm3d_medvae_multimodal_vectorized_v2_fullfov_regularized_v2/weights",
        "iters": list(range(5000, 30001, 5000)),
        "x_offset": 50000,
    },
}
BASELINE_600K = {
    "config": "configs/mmfm3d_medvae_multimodal_v2_fullfov.yaml",
    "checkpoint": "outputs/cfm3d/runs/mmfm3d_medvae_multimodal_vectorized_v2_fullfov/weights/model_final.pth",
}
FIELD_NORM_STATS_PATH = "configs/field_norm_stats.json"
OUT_JSON = "results/validation_curve_nrmse.json"


def nrmse(pred: np.ndarray, gt: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    rng = float(np.percentile(gt, 99) - np.percentile(gt, 1)) + 1e-8
    return rmse / rng


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(FIELD_NORM_STATS_PATH) as f:
        field_norm_stats = json.load(f)["stats"]

    results = []

    # One VAE + preprocessing pass per (config, pair, subject) — configs share
    # the same VAE/volume_size/target_spacing, but we reload per RUNS entry
    # for simplicity since VAE load is a few seconds, not the bottleneck here
    # (Euler integration + decode per checkpoint dominates).
    for run_name, run_cfg in {**RUNS, "baseline_600k": {"config": BASELINE_600K["config"], "weights_dir": None, "iters": [None], "x_offset": None}}.items():
        cfg = load_yaml_with_include(run_cfg["config"])
        cfg = resolve_paths(cfg, load_env("local"))
        data_cfg = cfg["data"]
        train_cfg = cfg["train"]

        amp_dtype = torch.bfloat16
        use_amp = bool(train_cfg.get("use_amp", True))
        p_lo = data_cfg.get("percentile_lower", 0.5)
        p_hi = data_cfg.get("percentile_upper", 99.5)
        volume_size = tuple(int(v) for v in data_cfg["volume_size"])
        target_spacing = tuple(float(v) for v in data_cfg["target_spacing"])
        modalities = data_cfg["modalities"]
        fields = data_cfg["fields"]
        n_classes = len(modalities)  # v2 only

        vae = load_vae(cfg, device)
        latent_shape = _infer_latent_shape(vae, volume_size, device)
        adapter = arch_vector.make_adapter(cfg, latent_shape, n_classes)
        mmfm = adapter.build_model().to(device)
        model_fn = adapter.make_model_fn(mmfm)

        # Pre-load + preprocess all (pair, subject) source/GT volumes once per
        # run config (identical across checkpoints within a run).
        cache = {}
        for src_f, tgt_f in PAIRS:
            for subj in SUBJECTS:
                src_path = DATA_ROOT / MODALITY / src_f / f"P_{MODALITY}_{src_f}_{subj}.nii.gz"
                tgt_path = DATA_ROOT / MODALITY / tgt_f / f"P_{MODALITY}_{tgt_f}_{subj}.nii.gz"
                if not src_path.exists() or not tgt_path.exists():
                    continue
                src_entry = field_norm_stats[MODALITY][src_f]
                tgt_entry = field_norm_stats[MODALITY][tgt_f]
                src_vol, _ = load_nifti_volume(
                    src_path, target_spacing=target_spacing, volume_size=volume_size,
                    normalize=True, lo_pct=p_lo, hi_pct=p_hi,
                    fixed_lo=src_entry["lo"], fixed_hi=src_entry["hi"],
                )
                tgt_vol, _ = load_nifti_volume(
                    tgt_path, target_spacing=target_spacing, volume_size=volume_size,
                    normalize=True, lo_pct=p_lo, hi_pct=p_hi,
                    fixed_lo=tgt_entry["lo"], fixed_hi=tgt_entry["hi"],
                )
                src_tensor = torch.from_numpy(src_vol).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad(), torch.amp.autocast(
                    "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
                ):
                    z_src = vae.encode(src_tensor)
                z_src_vec = vae.to_vector(z_src).float()
                cache[(src_f, tgt_f, subj)] = (z_src_vec, tgt_vol)

        for it in run_cfg["iters"]:
            if run_name == "baseline_600k":
                ckpt_path = BASELINE_600K["checkpoint"]
                x_val = None
            else:
                ckpt_path = str(Path(run_cfg["weights_dir"]) / f"checkpoint_{it}.pth")
                x_val = it + run_cfg["x_offset"]
            if not Path(ckpt_path).exists():
                print(f"  [skip] {ckpt_path} introuvable")
                continue

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "ema" in ckpt and ckpt["ema"]:
                shadow = ckpt["ema"].get("shadow_params")
                mmfm.load_state_dict(adapter.checkpoint_key_remap(shadow if shadow is not None else ckpt["ema"]))
            else:
                mmfm.load_state_dict(adapter.checkpoint_key_remap(ckpt["model"]))
            mmfm.eval()

            t0 = time.time()
            for src_f, tgt_f in PAIRS:
                pair_nrmses = []
                for subj in SUBJECTS:
                    key = (src_f, tgt_f, subj)
                    if key not in cache:
                        continue
                    z_src_vec, tgt_vol = cache[key]
                    tgt_mod_idx = modalities.index(MODALITY)
                    y = torch.tensor([tgt_mod_idx], device=device)
                    n_fields = len(fields)
                    z_tgt_vec = euler_integrate(
                        model_fn, z_src_vec, y,
                        t_start=_field_to_time(fields.index(src_f), n_fields),
                        t_end=_field_to_time(fields.index(tgt_f), n_fields),
                        n_steps=20, device=device, use_amp=use_amp, amp_dtype=amp_dtype,
                    )
                    z_tgt = LatentVectorizer(latent_shape).unflatten(z_tgt_vec)
                    with torch.no_grad(), torch.amp.autocast(
                        "cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda")
                    ):
                        dec = vae.decode(z_tgt)
                    # normalize_volume_fixed (used for tgt_vol below) returns
                    # [-1, 1], matching decode()'s raw output range directly —
                    # no (x+1)/2 shift or denormalize_from_01 needed here,
                    # unlike infer()'s native-intensity output path.
                    pred_vol = np.clip(dec.float().cpu().numpy()[0, 0], -1.0, 1.0)
                    val = nrmse(pred_vol, tgt_vol)
                    pair_nrmses.append(val)
                    results.append({
                        "run": run_name, "iter": x_val, "pair": f"{src_f}_to_{tgt_f}",
                        "subject": subj, "nrmse": val,
                    })
                if pair_nrmses:
                    print(f"  [{run_name} it={x_val}] {src_f}->{tgt_f}: nRMSE={np.mean(pair_nrmses):.4f} (n={len(pair_nrmses)})")
            print(f"  [{run_name} it={x_val}] done in {time.time()-t0:.1f}s")

        del vae, mmfm, cache
        torch.cuda.empty_cache()

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_JSON} ({len(results)} points)")


if __name__ == "__main__":
    main()
