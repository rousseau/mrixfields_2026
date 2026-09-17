#!/usr/bin/env python3
"""Construit 3 index de cache `pro_train` en leave-one-subject-out.

`pro_train` (`medvae_finetune_c4d1e200/pro_train/`) porte les 3 SEULS sujets du
jeu de données réellement acquis à plusieurs champs (0006, 0007, 0009) — voir
`diagnose_coupling_real.py`/`diagnose_true_displacement.py`. Pour fine-tuner le
flow dessus sans fuite, il faut exclure le sujet sur lequel on évaluera : ce
script écrit un `index.json` filtré par repli, sans dupliquer les `.pt` (les
chemins internes pointent toujours vers les fichiers déjà présents sous
`pro_train/`).

Prérequis : le cache `pro_train` doit être TUILÉ (`encode_scheme: tiled`),
cohérent avec `retro_train` — voir l'entrée CHANGELOG du 2026-09-16 (le cache
d'origine ne l'était pas, corrigé avant ce script).

Usage :
    PYTHONPATH=src python src/cfm/make_pro_train_loo_indexes.py
"""
from __future__ import annotations

import json
from pathlib import Path

CACHE_ROOT = Path("outputs/mmfm/latent_cache/vectorized")
CACHE_ID = "medvae_finetune_c4d1e200"
PRO_TRAIN = CACHE_ROOT / CACHE_ID / "pro_train"


def subject_id(path: str) -> str:
    return Path(path).stem.split("_")[-1]


def main() -> None:
    meta = json.loads((PRO_TRAIN / "index.json").read_text())
    if meta.get("encode_scheme") != "tiled":
        raise RuntimeError(
            f"pro_train n'est pas tuilé (encode_scheme={meta.get('encode_scheme')!r}) "
            "— corriger le cache avant de générer les index LOO (voir CHANGELOG 2026-09-16)."
        )

    samples = meta["samples"]
    subjects = sorted({subject_id(s["path"]) for s in samples})
    if len(subjects) != 3:
        raise RuntimeError(f"Attendu 3 sujets dans pro_train, trouvé {subjects}")
    print(f"Sujets dans pro_train : {subjects}")

    for excl in subjects:
        kept = [s for s in samples if subject_id(s["path"]) != excl]
        if len(kept) != len(samples) - 15:  # 3 contrastes x 5 champs
            raise RuntimeError(
                f"Filtrage inattendu pour excl={excl} : {len(kept)} restants "
                f"(attendu {len(samples) - 15})"
            )
        out_dir = CACHE_ROOT / CACHE_ID / f"pro_train_loo_excl{excl}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_meta = dict(meta)
        out_meta["split"] = f"pro_train_loo_excl{excl}"
        out_meta["samples"] = kept
        out_meta["loo_excluded_subject"] = excl
        (out_dir / "index.json").write_text(json.dumps(out_meta, indent=2))
        print(f"  excl{excl} : {len(kept)} entrées -> {out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
