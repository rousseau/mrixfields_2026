#!/usr/bin/env python3
"""Pré-encodage des latents INR — cache pour accélérer train_mmfm_unified.py --method mmfm3d_inr.

Reproduit EXACTEMENT le pipeline de prétraitement de
MultiModalNIfTILatentDataset._load_tensor :

    resample(target_spacing) -> normalize(percentiles) -> center_crop_or_pad(volume_size)

puis fitte le latent `z` (NOIR Algorithme 2 — theta/psi du backbone gelés,
descente de gradient sur z uniquement, voir cfm/inr_backbone.py) au lieu
d'un encode VAE. Le résultat (vecteur plat) est directement consommable par
arch_inr.py sans plus aucun fitting coûteux dans la boucle d'entraînement —
c'est précisément pourquoi ce cache existe : contrairement à l'encode VAE
(un simple forward pass), le fitting INR est un processus itératif bien trop
coûteux pour être refait à chaque step.

Le fitting utilise ici TOUS les voxels (dense), pas le sous-échantillonnage
sparse utilisé pendant le meta-entraînement du backbone (--fit-points permet
de sous-échantillonner si nécessaire pour la vitesse) : ce cache est calculé
une seule fois, la qualité prime sur la vitesse.

Clé de cache : outputs/mmfm/latent_cache/inr/<cache_id>/<split>/<mod>/<field>/<subject>.pt
  <cache_id> = hash8(backbone_checkpoint, inner_lr, inner_steps_eval, fit_points,
                      target_spacing, volume_size, percentiles)
  -> invalidation automatique si l'un de ces paramètres change.

  NB: cache_id est calculé LOCALEMENT à ce script (pas via
  common.dataset.flat_latent_cache_id, qui ne connaît que cfg['vae'] et donc
  ne verrait pas un changement de checkpoint du backbone INR — voir
  docs/MMFM_INR_STATE_OF_THE_ART.md). configs/mmfm/inr.yaml référence le
  chemin exact en dur (latent_cache_dir), comme vectorized.yaml/unet.yaml.

Usage :
    PYTHONPATH=src python src/cfm/precompute_inr_latents.py \\
        --config configs/mmfm/inr.yaml --env local
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_yaml_with_include, load_env, resolve_paths
from common.io import FILE_RE, MODALITIES, DOMAINS, SPLIT_MAP, load_nifti_volume, center_crop_or_pad_np
from cfm.arch_inr import load_inr_backbone
from cfm.inr_backbone import DEFAULT_FIT_CHUNK, fit_new_volume, make_coord_grid


def _inr_cache_id(
    cfg: dict, target_spacing, volume_size, p_lo: float, p_hi: float,
    fit_points, field_norm_stats_path=None,
) -> str:
    """INR-specific cache id — deliberately independent of
    common.dataset.flat_latent_cache_id (see module docstring).

    Includes the checkpoint's mtime (not just its path): the path is a
    stable string that doesn't change when a backbone is retrained in place
    (e.g. re-running train_inr_backbone.py against the same
    output_dir/model_final.pth) — without the mtime, a re-fit under a newer
    backbone would silently reuse latents cached under the old one.
    """
    b = cfg.get("inr_backbone", {})
    ckpt_path = b.get("checkpoint", "")
    ckpt_mtime = Path(ckpt_path).stat().st_mtime if ckpt_path and Path(ckpt_path).exists() else 0.0
    key = "|".join([
        "inr",
        str(ckpt_path),
        f"{ckpt_mtime}",
        str(b.get("inner_lr", "")),
        str(b.get("inner_steps_eval", "")),
        str(fit_points),
        str(target_spacing),
        str(volume_size),
        f"{p_lo}",
        f"{p_hi}",
        f"field_norm={field_norm_stats_path or ''}",
    ])
    h = hashlib.sha1(key.encode()).hexdigest()[:8]
    return f"inr_{h}"


def main():
    ap = argparse.ArgumentParser(description="Pré-encodage des vecteurs latents INR (fitting Algorithme 2)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--env", default="local")
    ap.add_argument("--cache-root", default="outputs/mmfm/latent_cache/inr")
    ap.add_argument("--split", default=None, help="Défaut : data.split de la config")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--fit-points", type=int, default=None,
                     help="Sous-échantillonnage du fitting (défaut : inr_backbone.fit_points de la "
                          "config, ou dense/tous les voxels si absent)")
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

    # Le budget de points du fitting fait partie de la DÉFINITION du latent :
    # arch_inr.py doit refitter à l'identique au moment de l'inférence, sinon le
    # flow reçoit un z source hors distribution. On le lit donc dans la config
    # (source unique de vérité, partagée avec arch_inr.py), --fit-points ne
    # servant plus qu'à surcharger ponctuellement.
    cfg_fit_points = cfg.get("inr_backbone", {}).get("fit_points")
    fit_points = args.fit_points if args.fit_points is not None else (
        int(cfg_fit_points) if cfg_fit_points else None)
    fit_chunk = int(cfg.get("inr_backbone", {}).get("fit_chunk", DEFAULT_FIT_CHUNK))

    print("Chargement du backbone INR...")
    backbone = load_inr_backbone(cfg, device)
    inner_lr = backbone.cfg.inner_lr
    inner_steps = backbone.cfg.inner_steps_eval
    coords_full = make_coord_grid(volume_size, device=device)
    print(f"  latent_dim={backbone.cfg.latent_dim}  inner_steps_eval={inner_steps}  inner_lr={inner_lr}"
          f"  fit_points={'dense' if fit_points is None else fit_points}  fit_chunk={fit_chunk}")

    cache_id = _inr_cache_id(
        cfg, target_spacing, volume_size, p_lo, p_hi, fit_points,
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
                    vol, _ = load_nifti_volume(
                        p, target_spacing=target_spacing, volume_size=None,
                        normalize=True, lo_pct=p_lo, hi_pct=p_hi,
                        fixed_lo=fixed_lo, fixed_hi=fixed_hi,
                    )
                    vol = center_crop_or_pad_np(vol, volume_size)
                    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
                    z = fit_new_volume(
                        backbone, x, coords_full,
                        num_steps=inner_steps, lr=inner_lr, num_points=fit_points,
                        chunk_size=fit_chunk,
                    )
                    z_vec = z.squeeze(0).to(torch.float32).cpu()  # (latent_dim,)
                    if flat_dim is None:
                        flat_dim = int(z_vec.shape[0])
                    torch.save(z_vec, out_path)
                    n_done += 1
                    index.append({"path": rel, "modality": modality, "field": field,
                                  "mod_idx": m_idx, "field_idx": f_idx})
                    if n_done <= 10 or n_done % 50 == 0:
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
