#!/usr/bin/env python3
"""Enrichit un `index.json` de cache de latents (vectorisé OU UNet) avec, pour
chaque échantillon, le champ `level` (voir `common.io.foreground_level`) :
le niveau d'intensité observé du volume NATIF, avant `normalize_volume` —
l'information que la normalisation par percentile détruit et que le témoin
`srclevel` lit directement dans la source (voir CHANGELOG.md, entrée du
2026-09-04, « le témoin qui manquait »).

N'encode RIEN et ne recalcule aucun latent : les fichiers `.pt` du cache sont
inchangés, seule la métadonnée `index.json` gagne un champ par échantillon.
Coût : un `load_nifti_volume` (lecture + rééchantillonnage, CPU, pas de VAE)
par volume — sans commune mesure avec une régénération complète du cache.

`cache_id` n'est PAS recalculé : ce script ne change aucun des paramètres
qui entrent dans son hachage (voir flat_latent_cache_id), donc les checkpoints
existants entraînés sur ce cache restent valides et rechargeables.

Usage :
    PYTHONPATH=src python src/cfm/augment_cache_index_with_src_level.py \
        --cache-dir outputs/mmfm/latent_cache/vectorized/medvae_finetune_c4d1e200/retro_train \
        --data-root /chemin/vers/les/donnees --env local
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_env, resolve_paths
from common.io import SPLIT_MAP, foreground_level, load_nifti_volume


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True,
                     help="Répertoire du split contenant index.json "
                          "(ex. outputs/mmfm/latent_cache/vectorized/<cache_id>/<split>)")
    ap.add_argument("--data-root", default=None,
                     help="Racine des volumes NIfTI. Défaut : data.data_root de --env "
                          "via resolve_paths, comme les autres scripts de precompute.")
    ap.add_argument("--env", default="local")
    ap.add_argument("--overwrite", action="store_true",
                     help="Recalcule `level` même pour les échantillons qui l'ont déjà.")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    index_path = cache_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index introuvable : {index_path}")

    meta = json.loads(index_path.read_text())
    samples = meta["samples"]

    if args.data_root is not None:
        data_root = Path(args.data_root)
    else:
        env_cfg = load_env(args.env)
        resolved = resolve_paths({"data": {}}, env_cfg)
        data_root = Path(resolved["data"]["data_root"])

    split_dir = SPLIT_MAP.get(meta["split"], meta["split"])
    target_spacing = tuple(float(v) for v in meta["target_spacing"]) if meta.get("target_spacing") else None

    backup_path = index_path.with_suffix(".json.bak")
    if not backup_path.exists():
        shutil.copy2(index_path, backup_path)
        print(f"Sauvegarde : {backup_path}")

    n_done = n_skip = n_err = 0
    t0 = time.time()
    for i, s in enumerate(samples, 1):
        if "level" in s and not args.overwrite:
            n_skip += 1
            continue
        modality = s.get("modality") or Path(s["path"]).parts[-3]
        field = s.get("field") or Path(s["path"]).parts[-2]
        nii_name = Path(s["path"]).stem + ".nii.gz"
        nii_path = data_root / split_dir / modality / field / nii_name
        try:
            # normalize=False : rééchantillonné mais NON normalisé, échelle
            # native — exactement ce que `normalize_volume` s'apprête à
            # détruire. Pas de crop : `foreground_level` porte sur le volume
            # entier rééchantillonné, pas sur le pavé de crop de l'entraînement
            # (le niveau global du sujet ne dépend pas du crop centré).
            vol, _ = load_nifti_volume(nii_path, target_spacing=target_spacing, normalize=False)
            s["level"] = foreground_level(vol)
            n_done += 1
        except Exception as e:
            n_err += 1
            print(f"[ERREUR] {nii_path}: {e}")
            continue
        if n_done <= 5 or n_done % 200 == 0:
            print(f"  [{i}/{len(samples)}] level={s['level']:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    index_path.write_text(json.dumps(meta, indent=2))
    print(f"\nTerminé : {n_done} calculés, {n_skip} déjà présents, {n_err} erreurs.")
    print(f"Index mis à jour : {index_path}")


if __name__ == "__main__":
    main()
