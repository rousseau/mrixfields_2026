#!/usr/bin/env python3
"""Convertit un cache de latents SPATIAUX (UNet) en cache APLATI (vectorisé).

Motivation — équité de comparaison : les deux architectures doivent voir
EXACTEMENT les mêmes latents, pas seulement des latents produits par le même
protocole. Ré-encoder les 1939 volumes une seconde fois donnerait des latents
identiques en théorie (l'encodage est déterministe : on prend le mode de la
gaussienne, pas un échantillon) mais coûterait ~3h de GPU pour rien. On relit
donc le cache UNet et on écrit sa version aplatie.

Le seul écart entre les deux formats de cache :
  - UNet       : tenseur (C, H', W', D')  + index avec `latent_shape`/`class_idx`
  - vectorisé  : tenseur (flat_dim,)      + index avec `flat_dim`/`modality`/`field`
(voir precompute_unet_latents.py et precompute_mmfm_latents.py)

Usage:
    PYTHONPATH=src python src/cfm/convert_unet_cache_to_flat.py \
        --src outputs/mmfm/latent_cache/unet/medvae_finetune_c4d1e200/retro_train \
        --dst-root outputs/mmfm/latent_cache/vectorized
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Répertoire du cache UNet (contenant index.json)")
    ap.add_argument("--dst-root", default="outputs/mmfm/latent_cache/vectorized")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_dir = Path(args.src)
    meta = json.loads((src_dir / "index.json").read_text())
    cache_id = meta["cache_id"]
    split = meta["split"]
    src_root = src_dir.parent.parent          # .../latent_cache/unet
    dst_dir = Path(args.dst_root) / cache_id / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source : {src_dir}\nCible  : {dst_dir}")
    print(f"latent_shape source : {meta.get('latent_shape')}")

    new_samples, flat_dim = [], None
    t0 = time.time()
    for i, s in enumerate(meta["samples"], 1):
        src_path = src_root / s["path"]
        # `path` est relatif à la racine du cache et commence par le cache_id ;
        # on conserve la même arborescence côté cible.
        rel = s["path"]
        dst_path = Path(args.dst_root) / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if not dst_path.exists() or args.overwrite:
            z = torch.load(src_path, map_location="cpu")
            z_flat = z.reshape(-1).contiguous()
            torch.save(z_flat, dst_path)
        else:
            z_flat = torch.load(dst_path, map_location="cpu")

        if flat_dim is None:
            flat_dim = int(z_flat.shape[0])

        modality = Path(rel).parts[-3]
        field = Path(rel).parts[-2]
        new_samples.append({
            "path": rel, "modality": modality, "field": field,
            "mod_idx": s["mod_idx"], "field_idx": s["field_idx"],
        })
        if i % 500 == 0:
            print(f"  [{i}/{len(meta['samples'])}] flat_dim={flat_dim} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    out_meta = dict(meta)
    out_meta.pop("latent_shape", None)
    out_meta["flat_dim"] = flat_dim
    out_meta["samples"] = new_samples
    out_meta["converted_from"] = str(src_dir)
    (dst_dir / "index.json").write_text(json.dumps(out_meta, indent=2))

    print(f"\n✅ {len(new_samples)} latents convertis, flat_dim={flat_dim}")
    print(f"   index : {dst_dir / 'index.json'}")


if __name__ == "__main__":
    main()
