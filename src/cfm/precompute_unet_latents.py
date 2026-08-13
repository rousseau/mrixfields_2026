#!/usr/bin/env python3
"""Pré-encodage des latents spatiaux (non-aplatis) pour train_mmfm_unet_3d.py.

Combine :
  - le prétraitement EXACT de precompute_mmfm_latents.py (resample(target_spacing)
    -> normalize(fixed field_norm_stats ou percentiles) -> center_crop_or_pad(volume_size)),
    reproduisant MultiModalNIfTILatentDataset._load_tensor à l'identique ;
  - la sortie SPATIALE (C, H', W', D') de precompute_latents.py (pas de
    vae.to_vector()), consommable directement par common.dataset.LatentCacheDataset
    sans aucun changement à cette classe.

precompute_latents.py encode le volume ENTIER SANS crop (pipeline différent, pas
adapté à train_mmfm_unet_3d.py qui attend un volume_size fixe) — ce script comble
cet écart plutôt que de modifier l'un des deux scripts existants.

Racine de cache dédiée (outputs/mmfm/latent_cache/unet/) : ne partage PAS
outputs/latent_cache/ (précompute_latents.py, volume entier) ni
outputs/mmfm/latent_cache/vectorized/ (précompute_mmfm_latents.py, vecteur
aplati), pour éviter toute collision entre schémas de payload différents même
en cas de collision de hash improbable.

Usage :
    PYTHONPATH=src python src/cfm/precompute_unet_latents.py \
        --config configs/mmfm/unet.yaml --env local \
        --field-norm-stats configs/mmfm/field_norm_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_yaml_with_include, load_env, resolve_paths
from common.io import FILE_RE, MODALITIES, DOMAINS, SPLIT_MAP, load_nifti_volume, center_crop_or_pad_np
from common.dataset import flat_latent_cache_id
from models.tiled_vae import tiled_encode
from models.vae_loader import load_vae


def main():
    ap = argparse.ArgumentParser(description="Pré-encodage des latents spatiaux MMFM-UNet (avec crop)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--env", default="local")
    ap.add_argument("--cache-root", default="outputs/mmfm/latent_cache/unet")
    ap.add_argument("--split", default=None, help="Défaut : data.split de la config")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--field-norm-stats", default=None,
                     help="Chemin vers un JSON produit par compute_field_norm_stats.py "
                          "(normalisation fixe par champ au lieu de percentile par volume)")
    args = ap.parse_args()

    field_norm_stats = None
    if args.field_norm_stats:
        with open(args.field_norm_stats) as f:
            field_norm_stats = json.load(f)["stats"]
        print(f"Normalisation par champ (fixe) : {args.field_norm_stats}")

    cfg = load_yaml_with_include(args.config)
    cfg = resolve_paths(cfg, load_env(args.env))
    data_cfg = cfg["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modalities = data_cfg.get("modalities", MODALITIES)
    fields = data_cfg.get("fields", DOMAINS)
    split = args.split or data_cfg.get("split", "retro_train")
    p_lo = data_cfg.get("percentile_lower", 0.5)
    p_hi = data_cfg.get("percentile_upper", 99.5)
    raw_ts = data_cfg.get("target_spacing", None)
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else None
    raw_vs = data_cfg.get("volume_size", None)
    if raw_vs is None:
        raise RuntimeError("data.volume_size est requis pour ce script.")
    volume_size = tuple(int(v) for v in raw_vs)

    # Tuilage de l'encodage (obligatoire au-delà de 2mm). Par défaut on tuile
    # dès que le volume dépasse ce qui tient en mémoire d'une seule passe.
    raw_tile = data_cfg.get("encode_tile", None)
    tile = tuple(int(v) for v in raw_tile) if raw_tile else None
    tile_margin = int(data_cfg.get("encode_tile_margin", 16))

    data_root = Path(data_cfg["data_root"])
    split_dir = SPLIT_MAP.get(split, split)

    amp_dtype_name = cfg.get("train", {}).get("amp_dtype", "bf16")
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16

    print("Chargement du VAE...")
    vae = load_vae(cfg, device)
    if vae.latent_format != "spatial":
        raise RuntimeError(f"VAE spatial requis, got {vae.latent_format}")

    # Même clé de hash que precompute_mmfm_latents.py (VAE + prétraitement) —
    # seul le payload (spatial vs aplati) diffère, sous une racine distincte.
    cache_id = flat_latent_cache_id(
        cfg, target_spacing, volume_size, p_lo, p_hi,
        field_norm_stats_path=args.field_norm_stats,
    )
    cache_dir = Path(args.cache_root) / cache_id / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache : {cache_dir}")
    print(f"  target_spacing={target_spacing}  volume_size={volume_size}  "
          f"percentiles=[{p_lo},{p_hi}]")

    index = []
    t0 = time.time()
    n_done = n_skip = n_err = 0
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
                files = files[: args.max_per_class]
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
                    fixed_lo = fixed_hi = None
                    if field_norm_stats is not None:
                        entry = field_norm_stats.get(modality, {}).get(field)
                        if entry is not None:
                            fixed_lo, fixed_hi = entry["lo"], entry["hi"]
                    # Reproduit exactement MultiModalNIfTILatentDataset._load_tensor :
                    # resample + normalize (SANS crop), puis center-crop/pad séparément.
                    vol, _ = load_nifti_volume(
                        p, target_spacing=target_spacing, volume_size=None,
                        normalize=True, lo_pct=p_lo, hi_pct=p_hi,
                        fixed_lo=fixed_lo, fixed_hi=fixed_hi,
                    )
                    vol = center_crop_or_pad_np(vol, volume_size)
                    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
                    # Encodage par TUILES : au-delà de 2mm, MedVAE ne peut pas
                    # encoder un volume entier en une passe (attention quadratique
                    # au goulot -> OOM dès 192x224x192). Les tuiles s'assemblent
                    # exactement en un latent unique (voir models/tiled_vae.py).
                    # tile=None => encodage direct (comportement historique 2mm).
                    if tile is not None:
                        z = tiled_encode(vae, x, tile=tile, margin=tile_margin,
                                         use_amp=(device.type == "cuda"), amp_dtype=amp_dtype)
                    else:
                        with torch.no_grad(), torch.amp.autocast(
                            "cuda", dtype=amp_dtype, enabled=(device.type == "cuda")
                        ):
                            z = vae.encode(x)
                    z = z.squeeze(0).to(torch.float16).cpu()  # (C, H', W', D') -- pas de to_vector()
                    if latent_shape is None:
                        latent_shape = tuple(z.shape)
                    torch.save(z, out_path)
                    n_done += 1
                    index.append({"path": rel, "mod_idx": m_idx, "field_idx": f_idx,
                                  "class_idx": class_idx})
                    if n_done <= 10 or n_done % 100 == 0:
                        dt = time.time() - t0
                        print(f"  [{n_done:04d}] {rel} | latent={latent_shape} | "
                              f"{dt/max(n_done,1):.2f}s/vol | elapsed={dt/60:.1f}min", flush=True)
                except Exception as e:
                    n_err += 1
                    print(f"\n❌ ERREUR sur {p}: {e}\n", flush=True)
                    import traceback; traceback.print_exc()
                    continue

    index_meta = {
        "cache_id": cache_id,
        "split": split,
        "modalities": modalities,
        "fields": fields,
        "latent_shape": list(latent_shape) if latent_shape else None,
        "target_spacing": list(target_spacing) if target_spacing else None,
        "volume_size": list(volume_size),
        "percentile_lower": p_lo,
        "percentile_upper": p_hi,
        "field_norm_stats_path": args.field_norm_stats,
        "samples": index,
    }
    index_path = cache_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index_meta, f, indent=2)

    print(f"\n✅ Terminé : {n_done} encodés, {n_skip} déjà présents, {n_err} erreurs.")
    print(f"   latent_shape : {latent_shape}")
    print(f"   Index : {index_path}")
    print(f"   Durée : {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
