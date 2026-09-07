#!/usr/bin/env python3
"""Banc de capacite de la REPRESENTATION INR — que peut vraiment representer un INR ici ?

POURQUOI CE SCRIPT EXISTE
L'auto-reconstruction INR de production vaut nRMSE ~0.35 la ou MedVAE vaut ~0.08 dans
le meme espace. Trois explications sont possibles et le depot ne les separe pas :

  (1) un INR ne peut pas representer ces volumes — la famille de modeles est en cause ;
  (2) l'INR peut, mais la MODULATION par volume (1536 nombres) est trop etroite ;
  (3) l'INR peut et la modulation suffirait, mais l'OPTIMISATION qui ajuste z
      (20 pas de descente de gradient simple a pas fixe 0.01) ne converge pas.

Ces trois causes appellent des correctifs opposes : (1) abandonner, (2) elargir la
modulation (modulate_scale / lora_rank, deja presents dans le code mais desactives),
(3) changer d'optimiseur. Ce banc les separe en mesurant, sur les MEMES volumes et
dans le MEME espace normalise que `bench_representation.py` :

  A. `siren_free_*`  : un SIREN entraine librement sur UN volume (tous poids libres).
     C'est le plafond de la famille INR, tel que la litterature le mesure.
  B. `prod_*`        : le backbone de production gele, z ajuste par la procedure de
     production (`fit_new_volume`, SGD a pas fixe) puis par Adam bien regle.
     L'ecart entre les deux est le cout de l'OPTIMISATION.
  C. `mod_*`         : le meme SIREN gele, mais la modulation optimisee DIRECTEMENT
     (sans hypernetwork), avec shift seul / + scale / + LoRA de rang r.
     L'ecart entre eux est le cout de la CAPACITE.

Les chiffres sont directement comparables aux colonnes `*_norm` de
`bench_representation.py` : meme volume, meme normalisation par champ, meme grille
1mm 192x224x192, meme metrique.

Usage :
    PYTHONPATH=src python src/cfm/bench_inr_capacity.py \
        --arms prod_sgd20,prod_adam,mod_shift,mod_scale,mod_lora4,siren_free_prod \
        --modalities T1W --fields 0.1T,3T,7T --outdir results/mmfm/inr_capacity_20260906
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib
import nibabel.processing as nib_proc
import torch
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_env
from common.io import center_crop_or_pad_np
from cfm.inr_backbone import (
    INRBackbone, INRBackboneConfig, ModulatedSIREN, make_coord_grid,
    _weighted_mse, fit_new_volume, decode_volume,
)

MODALITIES = ["T1W", "T2W", "T2FLAIR"]
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
SPLIT_DIR = "Training_prospective"
FIELD_NORM_STATS = Path("configs/mmfm/field_norm_stats.json")
BACKBONE_CKPT = Path("outputs/mmfm/inr_backbone/weights/model_final.pth")

VOLUME = (192, 224, 192)
SPACING = (1.0, 1.0, 1.0)
POINTS_PER_STEP = 262_144
EVAL_CHUNK = 1_048_576


# --------------------------------------------------------------------------- #


def nrmse(p, t):
    p, t = np.asarray(p, np.float64), np.asarray(t, np.float64)
    n = np.linalg.norm(t)
    return float(np.linalg.norm(p - t) / n) if n > 1e-10 else 0.0


def ssim3d(p, t, axis=2):
    p, t = np.asarray(p, np.float64), np.asarray(t, np.float64)
    dr = t.max() - t.min()
    if dr < 1e-10:
        return 1.0
    vals = []
    for i in range(p.shape[axis]):
        s = [slice(None)] * 3
        s[axis] = i
        s = tuple(s)
        if t[s].max() - t[s].min() < 1e-10:
            continue
        vals.append(structural_similarity(p[s], t[s], data_range=dr))
    return float(np.mean(vals)) if vals else 1.0


def psnr(p, t, data_range=2.0):
    mse = float(np.mean((np.asarray(p, np.float64) - np.asarray(t, np.float64)) ** 2))
    return 99.0 if mse <= 0 else float(10.0 * np.log10(data_range ** 2 / mse))


def load_normalized(path: Path, lo: float, hi: float) -> np.ndarray:
    """Meme pretraitement que bench_representation.py, variante `prod`."""
    img = nib.load(str(path))
    img1 = nib_proc.resample_to_output(img, voxel_sizes=SPACING, order=1)
    v = img1.get_fdata(dtype=np.float32)
    v = np.clip((v - lo) / (hi - lo), 0.0, 1.0) * 2.0 - 1.0
    return center_crop_or_pad_np(v.astype(np.float32), VOLUME)


# --------------------------------------------------------------------------- #
#  Le backbone de production
# --------------------------------------------------------------------------- #


def load_production_backbone(device) -> INRBackbone:
    ck = torch.load(BACKBONE_CKPT, map_location="cpu", weights_only=False)
    cfg_d = ck.get("backbone_cfg") or ck.get("inr_backbone_cfg") or ck.get("cfg") or {}
    if not cfg_d:
        raise RuntimeError(f"aucune config de backbone dans {BACKBONE_CKPT} — clefs : {list(ck)}")
    fields = INRBackboneConfig.__dataclass_fields__
    cfg = INRBackboneConfig(**{k: v for k, v in cfg_d.items() if k in fields})
    b = INRBackbone(cfg)
    state = ck["model"]
    if "ema" in ck and ck["ema"] and isinstance(ck["ema"], dict) and "shadow_params" in ck["ema"]:
        state = ck["ema"]["shadow_params"]
    b.load_state_dict(state)
    return b.to(device).eval()


def siren_with_capacity(base: ModulatedSIREN, modulate_scale: bool, lora_rank: int,
                        device) -> ModulatedSIREN:
    """Meme SIREN (memes poids appris), avec une modulation plus riche.

    Les parametres de `ModulatedSIREN` ne dependent PAS de modulate_scale/lora_rank
    (scale et LoRA arrivent par `gamma`), donc le state_dict se transfere tel quel :
    on mesure la capacite de la modulation, pas un autre reseau.
    """
    s = ModulatedSIREN(
        in_dim=base.in_dim, out_dim=1, hidden_dim=base.hidden_dim,
        num_hidden_layers=base.num_hidden_layers,
        omega_0=base.first.omega_0, omega_hidden=base.hidden[0].omega_0,
        modulate_scale=modulate_scale, lora_rank=lora_rank,
    )
    s.load_state_dict(base.state_dict())
    return s.to(device).eval()


def gamma_layout(siren: ModulatedSIREN) -> torch.Tensor:
    """Masque booleen (modulation_dim,) : True sur les composantes LoRA.

    Sert a donner aux facteurs de rang faible leur PROPRE pas d'apprentissage.
    Avec le pas des shifts (1e-2), une LoRA de rang 16 fait diverger le reseau des
    le premier pas (mesure : nRMSE 0.83, SSIM 0.09, contre 0.19/0.72 sans elle) —
    et on conclurait a tort que la capacite supplementaire ne sert a rien.
    """
    mask = torch.zeros(siren.modulation_dim, dtype=torch.bool)
    r = siren.lora_rank
    if r == 0:
        return mask
    off = 0
    for fin, fout in siren._layer_dims():
        off += fout
        if siren.modulate_scale:
            off += fout
        mask[off:off + fout * r + fin * r] = True
        off += fout * r + fin * r
    return mask


def init_gamma(siren: ModulatedSIREN, device, b_scale: float = 1e-2) -> torch.Tensor:
    """gamma initial. shift=0 et scale=0 redonnent exactement le reseau non module.

    Les facteurs LoRA ne peuvent PAS partir tous les deux de zero : le produit
    `A B^T` a alors un gradient nul des deux cotes (dL/dA proportionnel a x@B = 0,
    dL/dB proportionnel a A = 0) et la mise a jour de rang faible reste morte pour
    toujours. Convention standard (Hu et al. 2021) : B aleatoire, A a zero, de sorte
    que le produit est nul a l'initialisation mais differentiable.

    C'est un piege reel pour la production : `fit_latent` initialise z a zero, donc
    si `lora_rank > 0` etait active avec `hyper_hidden_dim=0` (modulation directe),
    la LoRA serait morte des le depart et la capacite annoncee ne servirait a rien.
    """
    g = torch.zeros(1, siren.modulation_dim, device=device)
    r = siren.lora_rank
    if r > 0:
        gen = torch.Generator(device=device); gen.manual_seed(7)
        off = 0
        for fin, fout in siren._layer_dims():
            off += fout
            if siren.modulate_scale:
                off += fout
            off += fout * r                                   # A reste a zero
            g[0, off:off + fin * r] = torch.randn(fin * r, device=device, generator=gen) * b_scale
            off += fin * r
    return g


# --------------------------------------------------------------------------- #
#  Optimisations
# --------------------------------------------------------------------------- #


@torch.no_grad()
def render(forward, coords_full, device) -> np.ndarray:
    out = torch.empty(coords_full.shape[0], device=device)
    for i in range(0, coords_full.shape[0], EVAL_CHUNK):
        sl = slice(i, min(i + EVAL_CHUNK, coords_full.shape[0]))
        out[sl] = forward(coords_full[sl].unsqueeze(0)).squeeze(0).squeeze(-1)
    return out.reshape(VOLUME).float().cpu().numpy()


def fit_by_adam(forward_with, params, coords_full, values, steps, lr, device,
                fg_weight=5.0, bg_threshold=-0.9, log_every=0) -> float:
    """Adam sur `params`, points echantillonnes a chaque pas (MedFuncta-style).

    `params` accepte la forme groupes-de-parametres de PyTorch pour donner un pas
    distinct a chaque famille. Renvoie la derniere perte.
    """
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    n = coords_full.shape[0]
    g = torch.Generator(device=device); g.manual_seed(1234)
    for step in range(steps):
        idx = torch.randint(0, n, (POINTS_PER_STEP,), device=device, generator=g)
        c = coords_full[idx].unsqueeze(0)
        t = values[idx].view(1, -1, 1)
        pred = forward_with(c)
        loss = _weighted_mse(pred, t, fg_weight, bg_threshold)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"      step {step+1:5d}/{steps}  loss {loss.item():.5f}", flush=True)
    return float(loss.item())


# --------------------------------------------------------------------------- #
#  Variantes
# --------------------------------------------------------------------------- #

ARMS = {
    "prod_sgd20": "backbone de production gele, z ajuste par fit_new_volume "
                  "(SGD pas fixe, inner_steps_eval=20) — LA recette de production",
    "prod_sgd200": "identique, 200 pas de SGD",
    "prod_adam": "backbone gele, z (129024-d) ajuste par Adam 1000 pas via le hypernetwork",
    "mod_shift": "SIREN gele, modulation (1536 shifts) optimisee DIRECTEMENT par Adam",
    "mod_scale": "SIREN gele, shift + scale (3072 valeurs) par Adam",
    "mod_lora4": "SIREN gele, shift + LoRA rang 4 (~13k valeurs) par Adam",
    "mod_lora16": "SIREN gele, shift + LoRA rang 16 (~51k valeurs) par Adam",
    "mod_lora64": "SIREN gele, shift + LoRA rang 64 (~200k valeurs) par Adam",
    "siren_free_prod": "SIREN NEUF 256x6 (~330k params), TOUS poids libres, un volume",
    "siren_free_big": "SIREN NEUF 512x8 (~1.8M params), TOUS poids libres, un volume",
    "grid1536": "TEMOIN TRIVIAL : depenser les 1536 nombres de la modulation INR en une "
                "grille 11x13x11 reinterpolee lineairement",
    "grid129024": "TEMOIN TRIVIAL : depenser les 129024 nombres du latent MedVAE en une "
                  "grille 48x56x48 reinterpolee lineairement",
}


def grid_control(vol: np.ndarray, budget: int) -> np.ndarray:
    """Reference d'interpretation : ce que vaut un budget de N nombres depense de la
    facon la plus bete possible.

    Sans ce temoin, « l'INR reconstruit a 21 dB » ne veut rien dire : on ignore si
    c'est mauvais pour son budget ou deja bien au-dessus de ce que ce budget permet.
    """
    from scipy.ndimage import zoom
    f = (budget / float(np.prod(VOLUME))) ** (1.0 / 3.0)
    small = zoom(vol, f, order=1)
    rec = zoom(small, np.array(VOLUME) / np.array(small.shape), order=1)
    return center_crop_or_pad_np(rec.astype(np.float32), VOLUME)


def run_cell(arm: str, vol: np.ndarray, backbone: INRBackbone, device,
             steps: int) -> dict:
    t0 = time.time()
    coords = make_coord_grid(VOLUME, device=device)
    values = torch.from_numpy(vol.reshape(-1)).to(device)
    vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)

    if arm.startswith("grid"):
        budget = int(arm[4:])
        rec = grid_control(vol, budget)

    elif arm.startswith("prod_"):
        for p in backbone.parameters():
            p.requires_grad_(False)
        if arm == "prod_sgd20":
            z = fit_new_volume(backbone, vol_t, coords, num_steps=20)
        elif arm == "prod_sgd200":
            z = fit_new_volume(backbone, vol_t, coords, num_steps=200)
        else:  # prod_adam : Adam sur z, a travers le hypernetwork
            z = torch.zeros(1, backbone.cfg.latent_dim, device=device, requires_grad=True)
            fit_by_adam(lambda c: backbone.decode(c, z), [z], coords, values,
                        steps, 1e-2, device)
            z = z.detach()
        rec = decode_volume(backbone, z, coords, VOLUME).squeeze().float().cpu().numpy()

    elif arm.startswith("mod_"):
        spec = {"mod_shift": (False, 0), "mod_scale": (True, 0),
                "mod_lora4": (False, 4), "mod_lora16": (False, 16),
                "mod_lora64": (False, 64)}[arm]
        siren = siren_with_capacity(backbone.siren, spec[0], spec[1], device)
        for p in siren.parameters():
            p.requires_grad_(False)
        mask = gamma_layout(siren).to(device)
        n_cap = int(siren.modulation_dim)

        # Les facteurs de rang faible n'ont pas la meme echelle que les shifts : on
        # leur donne leur propre pas et on BALAIE ce pas, pour qu'un mauvais reglage
        # ne se lise pas comme une absence de capacite.
        lr_grid = [1e-2] if spec[1] == 0 else [1e-3, 1e-4]
        best = None
        for lr_lora in lr_grid:
            g0 = init_gamma(siren, device)
            base = (g0 * (~mask)).clone().requires_grad_(True)
            lora = (g0 * mask).clone().requires_grad_(True)
            fwd = lambda c, b=base, l=lora: siren(c, torch.where(mask, l, b))
            groups = [{"params": [base], "lr": 1e-2}, {"params": [lora], "lr": lr_lora}]
            final = fit_by_adam(fwd, groups, coords, values, steps, 1e-2, device)
            if best is None or final < best[0]:
                best = (final, base.detach(), lora.detach(), lr_lora)
        _, base, lora, lr_used = best
        gamma = torch.where(mask, lora, base)
        rec = render(lambda c: siren(c, gamma), coords, device)

    else:  # siren_free_*
        hd, nl = (256, 6) if arm == "siren_free_prod" else (512, 8)
        siren = ModulatedSIREN(in_dim=3, out_dim=1, hidden_dim=hd,
                               num_hidden_layers=nl).to(device)
        zero = torch.zeros(1, siren.modulation_dim, device=device)
        for p in siren.parameters():
            p.requires_grad_(True)
        fit_by_adam(lambda c: siren(c, zero), list(siren.parameters()),
                    coords, values, steps, 1e-4, device)
        rec = render(lambda c: siren(c, zero), coords, device)

    rec = np.clip(rec, -1.0, 1.0)
    fg = vol > -0.9
    if arm.startswith("grid"):
        capacity = int(arm[4:])
    elif arm.startswith("mod_"):
        capacity = n_cap
    elif arm.startswith("prod_"):
        capacity = int(backbone.siren.modulation_dim)
    else:
        capacity = int(sum(p.numel() for p in siren.parameters()))
    return {
        "nrmse_norm": nrmse(rec, vol), "ssim_norm": ssim3d(rec, vol),
        "psnr_norm": psnr(rec, vol), "nrmse_norm_fg": nrmse(rec[fg], vol[fg]),
        "capacity": capacity, "seconds": time.time() - t0,
    }


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="prod_sgd20,mod_shift,mod_lora16,siren_free_prod")
    ap.add_argument("--modalities", default="T1W")
    ap.add_argument("--fields", default="0.1T,3T,7T")
    ap.add_argument("--subjects", default="0006")
    ap.add_argument("--steps", type=int, default=1000, help="pas d'Adam pour les variantes Adam")
    ap.add_argument("--outdir", default="results/mmfm/inr_capacity")
    ap.add_argument("--env", default="local")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for k, v in ARMS.items():
            print(f"  {k:18s} {v}")
        return

    names = list(ARMS) if a.arms == "all" else [s.strip() for s in a.arms.split(",")]
    bad = [n for n in names if n not in ARMS]
    if bad:
        raise SystemExit(f"variantes inconnues : {bad}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(load_env(a.env)["data_root"]) / SPLIT_DIR
    stats = json.load(open(FIELD_NORM_STATS))["stats"]
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    need_backbone = any(n.startswith(("prod_", "mod_")) for n in names)
    backbone = load_production_backbone(device) if need_backbone else None
    if backbone is not None:
        print(f"Backbone de production : latent_dim={backbone.cfg.latent_dim}, "
              f"modulation_dim={backbone.siren.modulation_dim}, "
              f"hidden={backbone.cfg.hidden_dim}x{backbone.cfg.num_hidden_layers}, "
              f"hyper_hidden={backbone.cfg.hyper_hidden_dim}")

    rows = []
    for arm in names:
        print(f"\n=== {arm} — {ARMS[arm]}")
        for mod in [s.strip() for s in a.modalities.split(",")]:
            for fld in [s.strip() for s in a.fields.split(",")]:
                for sid in [s.strip() for s in a.subjects.split(",")]:
                    p = root / mod / fld / f"P_{mod}_{fld}_{sid}.nii.gz"
                    if not p.exists():
                        print(f"  [manquant] {p}"); continue
                    e = stats[mod][fld]
                    vol = load_normalized(p, e["lo"], e["hi"])
                    try:
                        m = run_cell(arm, vol, backbone, device, a.steps)
                    except Exception as ex:
                        print(f"  [ERREUR] {arm} {mod} {fld}: {type(ex).__name__}: {ex}")
                        import traceback; traceback.print_exc()
                        continue
                    rows.append({"arm": arm, "modality": mod, "field": fld,
                                 "subject": sid, **m})
                    print(f"  {mod:8s} {fld:5s} {sid}  nRMSE {m['nrmse_norm']:.4f}  "
                          f"SSIM {m['ssim_norm']:.4f}  PSNR {m['psnr_norm']:5.2f}  "
                          f"({m['seconds']:.0f}s)", flush=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
        if rows:
            with open(outdir / "cells.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 78)
    print(f"{'variante':20s} {'nRMSE':>8s} {'SSIM':>8s} {'PSNR':>8s} {'n':>4s}")
    print("-" * 78)
    summary = []
    for arm in names:
        sel = [r for r in rows if r["arm"] == arm]
        if not sel:
            continue
        s = {"arm": arm, "n": len(sel)}
        for k in ("nrmse_norm", "ssim_norm", "psnr_norm", "nrmse_norm_fg", "seconds"):
            s[k] = float(np.mean([r[k] for r in sel]))
        summary.append(s)
        print(f"{arm:20s} {s['nrmse_norm']:8.4f} {s['ssim_norm']:8.4f} "
              f"{s['psnr_norm']:8.2f} {s['n']:4d}")
    print("=" * 78)
    with open(outdir / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    print(f"CSV : {outdir}/cells.csv et {outdir}/summary.csv")


if __name__ == "__main__":
    main()
