#!/usr/bin/env python3
"""Porte rapide avant l'éval Task 3 : le fine-tuning LOO progresse-t-il ?

Contrairement au 2026-09-15 (code de référence, own_share comme proxy faute de
vérité terrain), on a ICI la VRAIE cible pour le sujet tenu à l'écart -- inutile
d'un proxy : nRMSE latent DIRECT contre la vérité, pour chaque checkpoint d'un
repli. Sert à choisir 1-2 checkpoints avant de payer l'inférence Task 3 complète
(décodage MedVAE), et à repérer un sur-apprentissage sur les 2 sujets
d'entraînement avant de le découvrir au chiffre Task 3 (leçon du 2026-09-15 :
own_share qui grimpe ne veut pas dire mieux -- ici, le nRMSE latent contre la
vraie cible ne ment pas de la même façon).

Usage :
    PYTHONPATH=src python src/cfm/diagnose_finetune_checkpoints.py --excl 0006
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cfm.arch_vector import build_vector_mmfm
from cfm.mmfm_core import _field_to_time, euler_integrate

FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
MODALITIES = ["T1W", "T2W", "T2FLAIR"]
PRO_TRAIN = Path("outputs/mmfm/latent_cache/vectorized/medvae_finetune_c4d1e200/pro_train")
LATENT_DIM = 129024


def load_subject_latents(subject: str) -> Dict[str, Dict[str, torch.Tensor]]:
    """{contraste: {champ: latent}} pour UN sujet, depuis le cache pro_train complet."""
    idx = json.loads((PRO_TRAIN / "index.json").read_text())["samples"]
    base = PRO_TRAIN.parent.parent
    out: Dict[str, Dict[str, torch.Tensor]] = {m: {} for m in MODALITIES}
    for r in idx:
        if Path(r["path"]).stem.split("_")[-1] != subject:
            continue
        z = torch.load(base / r["path"], map_location="cpu", weights_only=False)
        out[r["modality"]][r["field"]] = torch.as_tensor(z).float().reshape(1, -1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--excl", required=True, help="sujet tenu à l'écart, ex. 0006")
    ap.add_argument("--src-field", default="0.1T")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subj_latents = load_subject_latents(a.excl)

    cfg_path = f"configs/mmfm/vectorized_finetune_loo_excl{a.excl}.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    use_amp = bool(cfg["train"].get("use_amp", True))

    weights_dir = Path("outputs") / cfg["data"]["output_subdir"] / "weights"
    ckpts = sorted(weights_dir.glob("checkpoint_*.pth"),
                    key=lambda p: int(p.stem.split("_")[1]))
    ckpts.append(weights_dir / "model_final.pth")

    prod_ckpt = Path(cfg["train"]["resume"])
    all_ckpts = [("production (avant fine-tuning)", prod_ckpt)] + [
        (f"iter {p.stem.split('_')[-1] if 'checkpoint' in p.stem else 'final'}", p)
        for p in ckpts
    ]

    t_src = _field_to_time(FIELDS.index(a.src_field), len(FIELDS))
    print(f"Sujet tenu à l'écart : {a.excl}  |  source : {a.src_field}  |  n_steps={a.n_steps}\n")
    print(f"  {'checkpoint':32s}{'nRMSE latent (vs vrai)':>24s}")

    results = []
    for label, ckpt_path in all_ckpts:
        if not ckpt_path.exists():
            print(f"  {label:32s}  (absent, {ckpt_path})")
            continue
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_vector_mmfm(cfg, LATENT_DIM, len(MODALITIES)).to(dev).eval()
        model.load_state_dict(ck["ema"] if "ema" in ck and ck["ema"] else ck["model"])

        nrmses: List[float] = []
        for mi, mod in enumerate(MODALITIES):
            z_src = subj_latents[mod][a.src_field].to(dev)
            y = torch.full((1,), mi, dtype=torch.long, device=dev)
            for f in FIELDS:
                if f == a.src_field:
                    continue
                t_tgt = _field_to_time(FIELDS.index(f), len(FIELDS))
                with torch.no_grad():
                    z_pred = euler_integrate(
                        lambda z, zs, t, yy: model(z, zs, t, yy),
                        z_src, y, t_src, t_tgt, a.n_steps, dev, use_amp=use_amp,
                    ).cpu()
                z_true = subj_latents[mod][f]
                nrmse = float(((z_pred - z_true).norm() / z_true.norm().clamp_min(1e-12)).item())
                nrmses.append(nrmse)
        mean_nrmse = float(np.mean(nrmses))
        print(f"  {label:32s}{mean_nrmse:24.4f}")
        results.append({"label": label, "checkpoint": str(ckpt_path), "nrmse_mean": mean_nrmse,
                        "nrmses": nrmses})

    if results:
        best = min(results[1:], key=lambda r: r["nrmse_mean"]) if len(results) > 1 else results[0]
        prod = results[0]
        print(f"\n  production (repère)      : {prod['nrmse_mean']:.4f}")
        print(f"  meilleur checkpoint fine-tuné : {best['label']} -> {best['nrmse_mean']:.4f}")
        if best["nrmse_mean"] < prod["nrmse_mean"]:
            print("  => amélioration en espace latent, candidat pour l'éval Task 3.")
        else:
            print("  => AUCUNE amélioration en espace latent sur ce sujet tenu à l'écart.")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"excl": a.excl, "src_field": a.src_field, "results": results},
                   open(a.json_out, "w"), indent=2)
        print(f"\n  écrit -> {a.json_out}")


if __name__ == "__main__":
    main()
