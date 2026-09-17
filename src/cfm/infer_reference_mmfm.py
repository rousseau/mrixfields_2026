#!/usr/bin/env python3
"""Inférence Task 3 pour le modèle de REFERENCE Genentech/MMFM
(voir `diagnose_reference_mmfm.py` -- ce script entraîne le checkpoint que
celui-ci évalue).

RÉUTILISE tel quel `process_volume_unified`/`_make_flow_spec`/`load_vae` de
`infer_mmfm_unified.py` (VAE, patchs, dénormalisation, géométrie, garde-fous de
cohérence cache/inférence) -- seul le modèle/adapter diffère : `VectorFieldModel`
de référence au lieu de `VectorMMFM`/`arch_vector.py`. `prep_latent`/
`restore_latent` réutilisent `LatentVectorizer`, la MÊME classe que
`arch_vector.make_adapter` -- le format du latent plat est identique (même VAE,
même cache), seul le réseau qui l'habite change.

Géométrie : `cc` (un seul crop centré) par défaut ici, pas `8win` -- ce script
sert un diagnostic (code/architecture vs données), pas une nouvelle ligne de
production ; `cc` va 6.5x plus vite (9-34 s/volume contre 104 s/volume, mesuré
le 2026-09-07) et reste comparable au protocole historique (pré-2026-08-27),
avec un biais connu et petit vs. 8win (+0.0034 sur la région notée).

Usage (Task 3 complet, 3 contrastes x 20 paires x 3 sujets) :
    PYTHONPATH=src python src/cfm/infer_reference_mmfm.py \\
        --config configs/mmfm/vectorized_trajectory.yaml \\
        --ref-checkpoint outputs/mmfm/reference_genentech/real/real_step004000.pth \\
        --output_dir outputs/mmfm/reference_genentech/predictions \\
        --field_norm_stats configs/mmfm/field_norm_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import nibabel as nib
import torch

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))
_PROJECT_ROOT = _SRC.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from cfm.mmfm_core import _infer_latent_shape, _field_to_time  # noqa: E402
from cfm.mmfm_vectorized import LatentVectorizer  # noqa: E402
from cfm.diagnose_reference_mmfm import build_reference_model, make_model_fn  # noqa: E402
from cfm.infer_mmfm_unified import _make_flow_spec, process_volume_unified  # noqa: E402
from common.config import load_yaml_with_include, load_env, resolve_paths  # noqa: E402
from common.io import DOMAINS, MODALITIES  # noqa: E402
from models.vae_loader import load_vae  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="config de PRODUCTION (VAE/cache/résolution) -- pas de "
                         "checkpoint de modèle lu depuis lui, voir --ref-checkpoint")
    ap.add_argument("--ref-checkpoint", required=True,
                    help="checkpoint écrit par diagnose_reference_mmfm.py real")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split", default="Training_prospective")
    ap.add_argument("--modalities", nargs="+", default=None)
    ap.add_argument("--field_norm_stats", required=True)
    ap.add_argument("--geometry", choices=["8win", "cc"], default="cc")
    ap.add_argument("--max_subjects", type=int, default=None)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--env", default="local")
    ap.add_argument("--n_steps", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    with open(a.field_norm_stats) as f:
        field_norm_stats = json.load(f)["stats"]

    cfg = load_yaml_with_include(a.config)
    cfg = resolve_paths(cfg, load_env(a.env))
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    p_lo = data_cfg.get("percentile_lower", 0.5)
    p_hi = data_cfg.get("percentile_upper", 99.5)
    patch_size = tuple(int(v) for v in data_cfg["volume_size"])
    stride = tuple(max(s // 2, 1) for s in patch_size)
    target_spacing = tuple(float(v) for v in data_cfg.get("target_spacing", (1.0, 1.0, 1.0)))
    raw_tile = data_cfg.get("encode_tile", None)
    encode_tile = tuple(int(v) for v in raw_tile) if raw_tile else None
    encode_tile_margin = int(data_cfg.get("encode_tile_margin", 16))
    all_modalities = data_cfg.get("modalities", MODALITIES)
    fields = data_cfg.get("fields", DOMAINS)
    n_fields = len(fields)
    modalities = a.modalities if a.modalities is not None else all_modalities

    print("Loading VAE (identique à la config de production)...")
    vae = load_vae(cfg, dev)
    if vae.latent_format != "spatial":
        raise RuntimeError(f"Requires spatial VAE, got {vae.latent_format}")

    latent_shape = _infer_latent_shape(vae, patch_size, dev)
    vectorizer = LatentVectorizer(latent_shape)
    print(f"latent_shape={latent_shape}  flat_dim={vectorizer.flat_dim}")

    print(f"Loading modèle de RÉFÉRENCE : {a.ref_checkpoint}")
    ck = torch.load(a.ref_checkpoint, map_location="cpu", weights_only=False)
    model = build_reference_model(ck["data_dim"], ck["n_classes"], ck["hidden_dim"], dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"  step={ck.get('step')}  data_dim={ck['data_dim']}  hidden_dim={ck['hidden_dim']}")
    if ck["data_dim"] != vectorizer.flat_dim:
        raise RuntimeError(
            f"data_dim du checkpoint ({ck['data_dim']}) != flat_dim du VAE de "
            f"cette config ({vectorizer.flat_dim}) -- mauvais cache/config.")

    adapter = SimpleNamespace(
        make_model_fn=lambda raw_model: make_model_fn(raw_model),
        prep_latent=lambda vae_, z: (vae_.to_vector(z), None),
        restore_latent=lambda z, meta: vectorizer.unflatten(z),
    )

    data_root = Path(cfg.get("data_root") or "/home/rousseau/Data/MRIxFields_20260414")
    out_root = Path(a.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    task_pairs = [(s, t) for s in fields for t in fields if s != t]

    center_crop_only = (a.geometry == "cc")

    start_all = time.time()
    for mod in modalities:
        mod_idx = all_modalities.index(mod)
        for src, tgt in task_pairs:
            flow_spec = _make_flow_spec("vectorial", mod_idx, fields.index(src),
                                        fields.index(tgt), n_fields)
            pair_out_dir = out_root / "task3" / mod / f"{src}_to_{tgt}"
            pair_out_dir.mkdir(parents=True, exist_ok=True)

            entry = field_norm_stats.get(mod, {}).get(src)
            tgt_entry = field_norm_stats.get(mod, {}).get(tgt)
            if entry is None or tgt_entry is None:
                print(f"[WARN] field_norm_stats sans entrée pour {mod}@{src} ou {mod}@{tgt}, skip")
                continue
            fixed_lo, fixed_hi = entry["lo"], entry["hi"]
            tgt_lo, tgt_hi = tgt_entry["lo"], tgt_entry["hi"]

            input_dir = data_root / a.split / mod / src
            if not input_dir.exists():
                print(f"[WARN] Input dir missing: {input_dir}")
                continue
            input_files = sorted(input_dir.glob("*.nii.gz"))
            if a.max_subjects is not None:
                input_files = input_files[:a.max_subjects]
            print(f"\n[{mod}] {src} → {tgt} : {len(input_files)} sujets")

            for nii_path in input_files:
                sid = nii_path.name.split("_")[-1].replace(".nii.gz", "")
                official_name = f"P_{mod}_{tgt}_{sid}.nii.gz"
                out_path = pair_out_dir / official_name
                if a.skip_existing and out_path.exists():
                    print(f"  SKIP {official_name}")
                    continue

                t0 = time.time()
                pred_vol, affine, header = process_volume_unified(
                    nii_path, vae, model, "vectorial", adapter,
                    flow_spec=flow_spec, n_steps=a.n_steps, patch_size=patch_size,
                    stride=stride, pad=16, p_lo=p_lo, p_hi=p_hi, device=dev,
                    use_amp=True, amp_dtype=torch.bfloat16,
                    norm_mode="field_fixed", center_crop_only=center_crop_only,
                    fixed_lo=fixed_lo, fixed_hi=fixed_hi, tgt_lo=tgt_lo, tgt_hi=tgt_hi,
                    target_spacing=target_spacing,
                    encode_tile=encode_tile, encode_tile_margin=encode_tile_margin,
                )
                nib.save(nib.Nifti1Image(pred_vol, affine, header), str(out_path))
                print(f"  {nii_path.name} → {out_path}  ({time.time() - t0:.1f}s)")

    elapsed = time.time() - start_all
    print(f"\nDone in {elapsed / 60:.1f} min. Predictions -> {out_root}")


if __name__ == "__main__":
    main()
