#!/usr/bin/env python3
"""Pré-encodage des latents MedVAE — cache pour entraînement MMFM rapide.

Encode chaque VOLUME ENTIER (1 mm, sans crop) en latent MedVAE fp16 et le
sauvegarde sur disque. L'entraînement MMFM lit ensuite ces latents directement
(plus de resample/normalize/encode à chaque itération).

Chemin de cache : outputs/latent_cache/<vae_id>/<split>/<mod>/<field>/<subject>.pt
  <vae_id> = <vae_type>_<hash8(checkpoint)>  (invalidation automatique si le VAE change)

Usage :
    PYTHONPATH=src python src/cfm/precompute_latents.py \
        --config configs/mmfm3d_multimarginal_medvae.yaml --env local
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_yaml_with_include, load_env, resolve_paths
from common.io import DOMAINS, MODALITIES, SPLIT_MAP, FILE_RE, load_nifti_volume
from models.vae_loader import load_vae


def _vae_id(cfg: dict) -> str:
    vae_cfg = cfg.get("vae", {})
    vtype = vae_cfg.get("vae_type", "vae")
    ckpt = vae_cfg.get("checkpoint", "") or ""
    h = hashlib.sha1(str(ckpt).encode()).hexdigest()[:8]
    return f"{vtype}_{h}"


def main():
    ap = argparse.ArgumentParser(description="Pré-encodage des latents MedVAE")
    ap.add_argument("--config", required=True)
    ap.add_argument("--env", default="local")
    ap.add_argument("--cache-root", default="outputs/latent_cache")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="Limite le nombre de volumes par classe (debug/smoke).")
    args = ap.parse_args()

    cfg = load_yaml_with_include(args.config)
    cfg = resolve_paths(cfg, load_env(args.env))
    data_cfg = cfg["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modalities = data_cfg.get("modalities", MODALITIES)
    fields = data_cfg.get("fields", DOMAINS)
    split = data_cfg.get("split", "retro_train")
    p_lo = data_cfg.get("percentile_lower", 0.5)
    p_hi = data_cfg.get("percentile_upper", 99.5)
    raw_ts = data_cfg.get("target_spacing", None)
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else None

    data_root = Path(data_cfg["data_root"])
    split_dir = SPLIT_MAP.get(split, split)

    amp_dtype_name = cfg.get("train", {}).get("amp_dtype", "bf16")
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16

    vae = load_vae(cfg, device)
    if vae.latent_format != "spatial":
        raise RuntimeError(f"VAE spatial requis, got {vae.latent_format}")

    vae_id = _vae_id(cfg)
    cache_dir = Path(args.cache_root) / vae_id / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache latents : {cache_dir}")

    index = []
    t0 = time.time()
    n_done = n_skip = 0
    latent_shape = None

    for m_idx, modality in enumerate(modalities):
        for f_idx, field in enumerate(fields):
            src_dir = data_root / split_dir / modality / field
            if not src_dir.exists():
                print(f"[WARN] absent : {src_dir}")
                continue
            out_dir = cache_dir / modality / field
            out_dir.mkdir(parents=True, exist_ok=True)
            files = sorted(src_dir.glob("*.nii.gz"))
            if args.max_per_class is not None:
                files = files[:args.max_per_class]
            class_idx = m_idx * len(fields) + f_idx

            for p in files:
                if FILE_RE.match(p.name) is None:
                    continue
                out_path = out_dir / (p.name.replace(".nii.gz", ".pt"))
                rel = str(out_path.relative_to(Path(args.cache_root)))
                if out_path.exists() and not args.overwrite:
                    n_skip += 1
                    index.append({"path": rel, "mod_idx": m_idx, "field_idx": f_idx,
                                  "class_idx": class_idx})
                    continue

                try:
                    # Volume ENTIER 1 mm, normalisé [-1,1], SANS crop
                    vol, _ = load_nifti_volume(
                        p, target_spacing=target_spacing, volume_size=None,
                        normalize=True, lo_pct=p_lo, hi_pct=p_hi,
                    )
                    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
                    with torch.no_grad(), torch.amp.autocast(
                        "cuda", dtype=amp_dtype, enabled=(device.type == "cuda")
                    ):
                        z = vae.encode(x)
                    z = z.squeeze(0).to(torch.float16).cpu()  # (C, H', W', D')
                    if latent_shape is None:
                        latent_shape = tuple(z.shape)
                    torch.save(z, out_path)
                    n_done += 1
                    index.append({"path": rel, "mod_idx": m_idx, "field_idx": f_idx,
                                  "class_idx": class_idx})
                    if n_done <= 10 or n_done % 50 == 0:
                        dt = time.time() - t0
                        per_vol = dt / n_done
                        print(f"  [{n_done:04d}] {rel} | vol={tuple(vol.shape)} | "
                              f"latent={latent_shape} | per_vol={per_vol:.1f}s | "
                              f"elapsed={dt/60:.1f}min", flush=True)
                except Exception as e:
                    print(f"\n❌ ERREUR sur {p}: {e}\n", flush=True)
                    import traceback; traceback.print_exc()
                    continue

    index_meta = {
        "vae_id": vae_id,
        "split": split,
        "modalities": modalities,
        "fields": fields,
        "latent_shape": list(latent_shape) if latent_shape else None,
        "target_spacing": list(target_spacing) if target_spacing else None,
        "percentile_lower": p_lo,
        "percentile_upper": p_hi,
        "samples": index,
    }
    index_path = cache_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index_meta, f, indent=2)

    print(f"\n✅ Terminé : {n_done} encodés, {n_skip} déjà présents.")
    print(f"   Latent shape : {latent_shape}")
    print(f"   Index : {index_path}")
    print(f"   Durée : {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
