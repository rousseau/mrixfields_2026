#!/usr/bin/env python3
"""Diagnostic per-cellule du Palier 2 (plan GAN) : d_acc par (contraste, champ
cible), pour verifier que le discriminateur ne memorise pas les petites
classes (T2W@5T=43 volumes, T2W@7T=81) alors que le d_acc GLOBAL est sain.

Reutilise directement les briques de production (arch_vector, euler_integrate,
LatentDiscriminator) sur le checkpoint deja entraine -- pas de reimplementation
du calcul, seulement une orchestration de mesure post-hoc.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/rousseau/Exp/mrixfields_2026/src")

from cfm import arch_vector
from cfm.mmfm_core import _field_to_time, _flat_class, _infer_latent_shape, euler_integrate
from common.config import load_env, load_yaml_with_include, resolve_paths
from models.vae_loader import load_vae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = load_yaml_with_include("configs/mmfm/vectorized_trajectory_gan.yaml")
cfg = resolve_paths(cfg, load_env("local"))

vae = load_vae(cfg, device)
for p in vae.parameters():
    p.requires_grad_(False)
vae.eval()

volume_size = tuple(cfg["data"]["volume_size"])
latent_shape = _infer_latent_shape(vae, volume_size, device)
n_fields = len(cfg["data"]["fields"])
n_classes = len(cfg["data"]["modalities"])
adapter = arch_vector.make_adapter(cfg, latent_shape, n_classes)

ds_cfg = dict(cfg["data"])
ds_cfg["flip_lr_prob"] = 0.0  # lecture propre, sans augmentation, pour ce diagnostic
ds = adapter.build_cache_dataset(
    Path(cfg["data"]["latent_cache_dir"]), Path(cfg["data"]["latent_cache_root"]), ds_cfg,
)
print(f"Cache : {len(ds)} echantillons")

ckpt = torch.load(
    "outputs/mmfm/vectorized_trajectory_gan/weights/model_final.pth",
    map_location=device, weights_only=False,
)
model = adapter.build_model().to(device)
model.load_state_dict(ckpt["ema"])  # poids EMA, comme eval_quantitative_3arch.py
model.eval()
model_fn = adapter.make_model_fn(model)

disc = arch_vector.build_latent_discriminator(
    cfg, int(np.prod(latent_shape)), n_classes,
).to(device)
disc.load_state_dict(ckpt["disc"])
disc.eval()

# index par class_idx -> liste d'indices dataset (comme _sample_class, mais
# une seule passe au lieu d'un scan par classe).
by_class: dict[int, list[int]] = {}
for i, s in enumerate(ds.samples):
    c = s[3] if isinstance(s, tuple) else s["class_idx"]
    by_class.setdefault(c, []).append(i)

CLASS_SIZES = {c: len(v) for c, v in by_class.items()}
print("Tailles de classe (mod*n_fields+field) :", CLASS_SIZES)

N = 16
g = torch.Generator().manual_seed(0)
rows = []
for contrast in range(n_classes):
    src_flat = _flat_class(contrast, 0, n_fields)
    if src_flat not in by_class or len(by_class[src_flat]) == 0:
        continue
    src_pool = by_class[src_flat]
    src_idx = [src_pool[i] for i in torch.randint(0, len(src_pool), (N,), generator=g).tolist()]
    z_src = torch.stack([ds[i][0] for i in src_idx]).to(device).float()
    y = torch.full((N,), contrast, dtype=torch.long, device=device)

    for tgt_field in range(1, n_fields):
        tgt_flat = _flat_class(contrast, tgt_field, n_fields)
        if tgt_flat not in by_class or len(by_class[tgt_flat]) == 0:
            continue
        tgt_pool = by_class[tgt_flat]
        tgt_idx = [tgt_pool[i] for i in torch.randint(0, len(tgt_pool), (N,), generator=g).tolist()]
        z_real = torch.stack([ds[i][0] for i in tgt_idx]).to(device).float()

        t_end = _field_to_time(tgt_field, n_fields)
        with torch.no_grad():
            z_fake = euler_integrate(model_fn, z_src, y, 0.0, t_end, 8, device,
                                      use_amp=True, amp_dtype=torch.bfloat16)
            t_vec = torch.full((N,), t_end, device=device)
            d_real = disc(z_real, t_vec, y)
            d_fake = disc(z_fake, t_vec, y)
            acc = 0.5 * ((d_real > 0.5).float().mean() + (d_fake < 0.5).float().mean())

        rows.append({
            "contrast": contrast, "tgt_field": tgt_field,
            "n_tgt_class": len(tgt_pool), "d_acc": round(float(acc.item()), 3),
            "d_real_mean": round(float(d_real.mean().item()), 3),
            "d_fake_mean": round(float(d_fake.mean().item()), 3),
        })

for r in sorted(rows, key=lambda r: r["d_acc"], reverse=True):
    flag = "  <-- petite classe" if r["n_tgt_class"] < 100 else ""
    print(f"  contraste={r['contrast']} champ_cible={r['tgt_field']} "
          f"n_classe_cible={r['n_tgt_class']:4d}  d_acc={r['d_acc']:.3f}  "
          f"D(reel)={r['d_real_mean']:+.3f}  D(faux)={r['d_fake_mean']:+.3f}{flag}")

d_accs = [r["d_acc"] for r in rows]
print(f"\nMoyenne des {len(rows)} cellules : {np.mean(d_accs):.3f}  "
      f"(min {min(d_accs):.3f}, max {max(d_accs):.3f})")

Path("results/mmfm/gan_latent_paliers01_20260922/percell_diagnostic_palier2.json").write_text(
    json.dumps({"class_sizes": CLASS_SIZES, "cells": rows}, indent=2)
)
print("Ecrit -> results/mmfm/gan_latent_paliers01_20260922/percell_diagnostic_palier2.json")
