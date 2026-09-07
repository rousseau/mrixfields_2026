#!/usr/bin/env python3
"""Pré-encodage des vecteurs latents MMFM — cache pour accélérer train_mmfm_3d.py.

Contrairement à precompute_latents.py (qui encode le volume ENTIER sans crop,
pour un usage différent), ce script reproduit EXACTEMENT le pipeline de
prétraitement de MultiModalNIfTILatentDataset._load_tensor :

    resample(target_spacing) -> normalize(percentiles) -> center_crop_or_pad(volume_size)

puis encode via le VAE et aplatit (vae.to_vector) — le résultat est directement
consommable par train_mmfm_3d.py sans plus aucun traitement CPU/GPU coûteux.

Le crop est toujours CENTRÉ (déterministe) : le cache ne peut pas reproduire
l'augmentation random_crop_prob>0 de l'entraînement (une seule version par
volume est mise en cache). Pour un volume_size proche/supérieur au volume
natif rééchantillonné (cas des configs "full-FOV"), le random crop de
l'entraînement ne fait de toute façon presque que du padding — perte
d'augmentation négligeable dans ce cas précis.

Clé de cache : outputs/mmfm/latent_cache/vectorized/<cache_id>/<split>/<mod>/<field>/<subject>.pt
  <cache_id> = hash8(vae_checkpoint, target_spacing, volume_size, percentiles)
  -> invalidation automatique si l'un de ces paramètres change.

Usage :
    PYTHONPATH=src python src/cfm/precompute_mmfm_latents.py \
        --config configs/mmfm/vectorized.yaml --env local \
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
from models.vae_loader import load_vae


def main():
    ap = argparse.ArgumentParser(description="Pré-encodage des vecteurs latents MMFM (avec crop)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--env", default="local")
    ap.add_argument("--cache-root", default="outputs/mmfm/latent_cache/vectorized")
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

    data_root = Path(data_cfg["data_root"])
    split_dir = SPLIT_MAP.get(split, split)

    amp_dtype_name = cfg.get("train", {}).get("amp_dtype", "bf16")
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16

    print("Chargement du VAE...")
    vae = load_vae(cfg, device)

    # ATTENTION — CE SCRIPT NE TUILE PAS. `vae.encode(x)` ci-dessous passe par la
    # fenêtre glissante gaussienne interne de MedVAE (`roi_size_calc`, `gpu_dim=160`),
    # pas par `tiled_encode`. Ce n'est donc PAS le chemin d'encodage de la production :
    # le cache de production `medvae_finetune_c4d1e200` a été produit par
    # `precompute_unet_latents.py` (qui tuile) puis aplati par
    # `convert_unet_cache_to_flat.py`. Écart mesuré entre les deux schémas :
    # 0.1140 contre 0.1168 de nRMSE sur la tranche notée
    # (`results/mmfm/representation_20260906/manifest.md`).
    #
    # Jusqu'au 2026-09-07 le schéma d'encodage n'entrait pas dans la clé de cache, de
    # sorte que les caches `1989e9d1` (LPIPS) et `ff550d64` (identity) — écrits par ce
    # script, donc non tuilés — ont été comparés à `c4d1e200` qui l'est. Les deux
    # verdicts négatifs correspondants portaient deux variables et pas une.
    #
    # `encode_tile=None` ci-dessous dit la vérité sur ce que ce script fait ; pour
    # reproduire la production, utiliser precompute_unet_latents.py + conversion.
    cache_id = flat_latent_cache_id(
        cfg, target_spacing, volume_size, p_lo, p_hi,
        field_norm_stats_path=args.field_norm_stats,
        encode_tile=None, encode_tile_margin=None, amp_dtype=amp_dtype_name,
    )
    cache_dir = Path(args.cache_root) / cache_id / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache : {cache_dir}")
    print(f"  target_spacing={target_spacing}  volume_size={volume_size}  "
          f"percentiles=[{p_lo},{p_hi}]")

    index = []
    t0 = time.time()
    n_done = n_skip = n_err = 0
    flat_dim = None

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

            for p in files:
                if FILE_RE.match(p.name) is None:
                    continue
                out_path = out_dir / (p.name.replace(".nii.gz", ".pt"))
                rel = str(out_path.relative_to(Path(args.cache_root)))
                if out_path.exists() and not args.overwrite:
                    n_skip += 1
                    index.append({"path": rel, "modality": modality, "field": field,
                                  "mod_idx": m_idx, "field_idx": f_idx})
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
                    with torch.no_grad(), torch.amp.autocast(
                        "cuda", dtype=amp_dtype, enabled=(device.type == "cuda")
                    ):
                        z = vae.encode(x)
                        z_vec = vae.to_vector(z)
                    z_vec = z_vec.squeeze(0).to(torch.float16).cpu()  # (flat_dim,)
                    if flat_dim is None:
                        flat_dim = int(z_vec.shape[0])
                    torch.save(z_vec, out_path)
                    n_done += 1
                    index.append({"path": rel, "modality": modality, "field": field,
                                  "mod_idx": m_idx, "field_idx": f_idx})
                    if n_done <= 10 or n_done % 100 == 0:
                        dt = time.time() - t0
                        print(f"  [{n_done:04d}] {rel} | flat_dim={flat_dim} | "
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
        "flat_dim": flat_dim,
        "target_spacing": list(target_spacing) if target_spacing else None,
        "volume_size": list(volume_size),
        "percentile_lower": p_lo,
        "percentile_upper": p_hi,
        "field_norm_stats_path": args.field_norm_stats,
        "vae_checkpoint": cfg.get("vae", {}).get("checkpoint"),
        # PROVENANCE DE L'ENCODAGE (2026-09-07). Sans ces champs, rien dans le cache
        # ne disait avec quel schéma il avait été encodé — et deux schémas
        # incompatibles portaient le même `cache_id`. C'est `index.json` qui fait
        # foi : `_check_cache_consistency` le relit avant toute inférence.
        "encode_scheme": "medvae_sliding_window",
        "encode_tile": None,
        "encode_tile_margin": None,
        "amp_dtype": amp_dtype_name,
        "samples": index,
    }
    index_path = cache_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index_meta, f, indent=2)

    print(f"\n✅ Terminé : {n_done} encodés, {n_skip} déjà présents, {n_err} erreurs.")
    print(f"   flat_dim : {flat_dim}")
    print(f"   Index : {index_path}")
    print(f"   Durée : {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
