#!/usr/bin/env python3
"""Figures qualitatives : capacite de representation INR (SIREN, production
CORRIGEE) vs MedVAE pre-entraine -- auto-reconstruction a 1mm, geometrie et
checkpoints ACTUELS (2026-09-09).

POURQUOI CE SCRIPT EXISTE
Les seules figures INR vs MedVAE du depot (`figures_representation_capacity.py`,
dossier `results/mmfm/comparison_20260801_final/figures_representation_capacity/`)
datent du 2026-08-01/07 et utilisent un INR PERIME : latent z=4096 (remplace par
129024 le 2026-08-10), et surtout ANTERIEURES a l'audit du 2026-08-25 qui a
corrige le biais EMA non chauffe et l'inversion d'orientation LAS/RAS (effet
combine mesure : nRMSE T1W 0.6383 -> 0.3787, voir results/mmfm/audit_20260825/).
Aucune figure a jour ne montrait le backbone INR reellement deploye aujourd'hui.

Ce script ne reimplemente rien : il reutilise le code deja valide par les
campagnes chiffrees du 2026-09-06/07 (meme principe que
figures_representation_protocol.py : une figure batie sur un chemin parallele
finirait par illustrer autre chose que ce qu'on a mesure) :
  - MedVAE : `cfm.bench_representation.roundtrip(ARMS["prod"], ...)`.
  - INR    : `cfm.bench_inr_capacity.load_production_backbone` (le checkpoint
    `outputs/mmfm/inr_backbone/weights/model_final.pth` qui a servi a batir le
    cache de production `inr_932b0b37`) + `fit_new_volume`/`decode_volume`
    (procedure exacte de production : 20 pas de SGD, arm "prod_sgd20").

Deux jeux de figures :

  A. PRODUCTION vs PRODUCTION (par defaut 9 cellules : T1W x {0006,0007,0009}
     x {0.1T,3T,7T}) -- memes sujets/champs que les figures perimees de
     2026-08-01, pour un avant/apres direct. GT | MedVAE | INR (deploye) |
     erreur commune (3 plans : axial/coronal/sagittal).

  B. PLAFOND DE CAPACITE (par defaut 3 cellules : T1W x 0006 x {0.1T,3T,7T},
     memes cellules que results/mmfm/inr_capacity_20260906/) -- ajoute une 4e
     colonne, la modulation LoRA rang 64 optimisee DIRECTEMENT par Adam (meme
     SIREN gele, pas de reentrainement du backbone) : c'est la mesure qui a
     etabli que l'ecart INR/MedVAE est un facteur BUDGET (x252 en nombres par
     volume), pas un defaut de famille. Rend ce chiffre visible, pas seulement
     chiffre dans un CSV.

Usage :
    PYTHONPATH=src python src/cfm/figures_representation_capacity_current.py \
        --outdir results/mmfm/qualitative_representation_20260909

    # test rapide, une seule cellule, sans le plafond (couteux) :
    PYTHONPATH=src python src/cfm/figures_representation_capacity_current.py \
        --subjects 0006 --fields 3T --skip-set-b \
        --outdir /tmp/smoke_repr
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_env
from cfm.bench_representation import ARMS, FIELD_NORM_STATS, SPLIT_DIR, nrmse, ssim3d, roundtrip
from cfm.bench_inr_capacity import (
    load_production_backbone, load_normalized, siren_with_capacity, gamma_layout,
    init_gamma, fit_by_adam, render, VOLUME,
)
from cfm.inr_backbone import fit_new_volume, decode_volume, make_coord_grid

MODALITY = "T1W"
LORA_STEPS_DEFAULT = 1000  # identique a inr_capacity_20260906 (mod_lora64)


def _to01(x: np.ndarray) -> np.ndarray:
    return np.clip((x + 1) / 2, 0, 1)


def _nii_path(root: Path, mod: str, fld: str, sid: str) -> Path:
    return root / mod / fld / f"P_{mod}_{fld}_{sid}.nii.gz"


def _load_cell(root: Path, stats: dict, mod: str, fld: str, sid: str) -> Optional[np.ndarray]:
    p = _nii_path(root, mod, fld, sid)
    if not p.exists():
        print(f"  [skip] {p} introuvable")
        return None
    e = stats[mod][fld]
    return load_normalized(p, e["lo"], e["hi"])  # (192,224,192) numpy, [-1,1]


def _plot_row_figure(gt: np.ndarray, panels: List[Tuple[str, np.ndarray, float, float]],
                      title: str, out_path: Path) -> None:
    """panels : liste de (label, reconstruction, nrmse, ssim). Layout : 3 plans
    (axial / coronal / sagittal) x (GT + une colonne par panel + une colonne
    d'erreur commune, max sur tous les panels, meme echelle par ligne)."""
    h, w, d = gt.shape
    slices = {"Axiale": (slice(None), slice(None), d // 2),
              "Coronale": (slice(None), w // 2, slice(None)),
              "Sagittale": (h // 2, slice(None), slice(None))}
    gt01 = _to01(gt)
    n_cols = 1 + len(panels) + 1
    fig, axes = plt.subplots(3, n_cols, figsize=(3.1 * n_cols, 9.0))

    for row, (plane_name, sl) in enumerate(slices.items()):
        gt_sl = gt01[sl]
        recon_sls = [_to01(np.clip(recon, -1, 1))[sl] for _, recon, _, _ in panels]
        diffs = [np.abs(r - gt_sl) for r in recon_sls]
        emax = float(np.percentile(np.stack(diffs), 99.5)) if diffs else 0.5

        axes[row, 0].imshow(np.rot90(gt_sl), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(plane_name, fontsize=11)
        for k, ((label, _, nr, ss), r_sl) in enumerate(zip(panels, recon_sls), start=1):
            axes[row, k].imshow(np.rot90(r_sl), cmap="gray", vmin=0, vmax=1)
            if row == 0:
                axes[row, k].set_title(f"{label}\nnRMSE={nr:.3f} SSIM={ss:.3f}", fontsize=10)
        im_d = axes[row, -1].imshow(np.rot90(np.maximum.reduce(diffs)), cmap="hot", vmin=0, vmax=emax)
        if row == 0:
            axes[row, 0].set_title("GT", fontsize=12)
            axes[row, -1].set_title(f"|erreur| max, echelle commune\n(p99.5={emax:.2f})", fontsize=10)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _inr_deployed(backbone, vol: np.ndarray, coords: torch.Tensor, device) -> np.ndarray:
    """Procedure EXACTE de production : fit_new_volume, 20 pas de SGD (arm
    'prod_sgd20' de bench_inr_capacity.py)."""
    vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    z = fit_new_volume(backbone, vol_t, coords, num_steps=20)
    rec = decode_volume(backbone, z, coords, VOLUME).squeeze().float().cpu().numpy()
    return np.clip(rec, -1, 1)


def _inr_ceiling_lora64(backbone, vol: np.ndarray, coords: torch.Tensor, device,
                        steps: int) -> np.ndarray:
    """Meme SIREN gele (poids du backbone), modulation LoRA rang 64 optimisee
    DIRECTEMENT par Adam (arm 'mod_lora64' de bench_inr_capacity.py) --
    reproduit a l'identique, y compris le balayage de lr sur les facteurs
    LoRA (necessaire : un mauvais lr fait diverger la LoRA des le 1er pas,
    voir gamma_layout()/init_gamma())."""
    siren = siren_with_capacity(backbone.siren, modulate_scale=False, lora_rank=64, device=device)
    for p in siren.parameters():
        p.requires_grad_(False)
    mask = gamma_layout(siren).to(device)
    values = torch.from_numpy(vol.reshape(-1)).to(device)

    best = None
    for lr_lora in (1e-3, 1e-4):
        g0 = init_gamma(siren, device)
        base = (g0 * (~mask)).clone().requires_grad_(True)
        lora = (g0 * mask).clone().requires_grad_(True)
        fwd = lambda c, b=base, l=lora: siren(c, torch.where(mask, l, b))
        groups = [{"params": [base], "lr": 1e-2}, {"params": [lora], "lr": lr_lora}]
        final = fit_by_adam(fwd, groups, coords, values, steps, 1e-2, device)
        if best is None or final < best[0]:
            best = (final, base.detach(), lora.detach())
    _, base, lora = best
    gamma = torch.where(mask, lora, base)
    rec = render(lambda c: siren(c, gamma), coords, device)
    return np.clip(rec, -1, 1)


def run_set_a(root, stats, backbone, coords, device, outdir: Path,
              subjects: List[str], fields: List[str],
              all_rows: List[dict], csv_path: Path) -> List[dict]:
    rows = []
    for sid in subjects:
        for fld in fields:
            vol = _load_cell(root, stats, MODALITY, fld, sid)
            if vol is None:
                continue
            print(f"=== [A] {MODALITY} {fld} sujet {sid}")
            try:
                vae_rec = np.clip(roundtrip(ARMS["prod"], vol, device), -1, 1)
                vae_nr, vae_ss = nrmse(vae_rec, vol), ssim3d(vae_rec, vol)

                inr_rec = _inr_deployed(backbone, vol, coords, device)
                inr_nr, inr_ss = nrmse(inr_rec, vol), ssim3d(inr_rec, vol)
            except Exception as e:
                print(f"  [ERREUR] {sid} {fld}: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue

            panels = [("MedVAE pre-entraine (production)", vae_rec, vae_nr, vae_ss),
                      ("INR SIREN (production, 20 pas SGD)", inr_rec, inr_nr, inr_ss)]
            title = (f"Capacite de representation -- {MODALITY} {fld}, sujet {sid}\n"
                     f"(auto-reconstruction, 1mm, backbone/checkpoints de production ACTUELS)")
            out = outdir / f"repr_current_{MODALITY}_{sid}_{fld}.png"
            _plot_row_figure(vol, panels, title, out)
            print(f"  MedVAE nRMSE={vae_nr:.4f} SSIM={vae_ss:.4f} | "
                  f"INR nRMSE={inr_nr:.4f} SSIM={inr_ss:.4f}  -> {out}")
            rows.append({"set": "A", "modality": MODALITY, "field": fld, "subject": sid,
                        "vae_nrmse": vae_nr, "vae_ssim": vae_ss,
                        "inr_nrmse": inr_nr, "inr_ssim": inr_ss})
            _write_csv(all_rows + rows, csv_path)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def run_set_b(root, stats, backbone, coords, device, outdir: Path,
              subject: str, fields: List[str], lora_steps: int,
              all_rows: List[dict], csv_path: Path) -> List[dict]:
    rows = []
    for fld in fields:
        vol = _load_cell(root, stats, MODALITY, fld, subject)
        if vol is None:
            continue
        print(f"=== [B] {MODALITY} {fld} sujet {subject} (plafond de capacite, "
              f"lora_steps={lora_steps})")
        try:
            vae_rec = np.clip(roundtrip(ARMS["prod"], vol, device), -1, 1)
            vae_nr, vae_ss = nrmse(vae_rec, vol), ssim3d(vae_rec, vol)

            inr_rec = _inr_deployed(backbone, vol, coords, device)
            inr_nr, inr_ss = nrmse(inr_rec, vol), ssim3d(inr_rec, vol)

            ceil_rec = _inr_ceiling_lora64(backbone, vol, coords, device, lora_steps)
            ceil_nr, ceil_ss = nrmse(ceil_rec, vol), ssim3d(ceil_rec, vol)
        except Exception as e:
            print(f"  [ERREUR] {subject} {fld}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        panels = [("MedVAE pre-entraine", vae_rec, vae_nr, vae_ss),
                  ("INR deploye (20 pas SGD, ~512 eff.)", inr_rec, inr_nr, inr_ss),
                  ("INR plafond (LoRA r=64, ~182k eff.)", ceil_rec, ceil_nr, ceil_ss)]
        title = (f"Plafond de capacite INR vs MedVAE -- {MODALITY} {fld}, sujet {subject}\n"
                 f"meme SIREN gele : seule la modulation autorisee par volume change")
        out = outdir / f"plafond_{MODALITY}_{fld}_{subject}.png"
        _plot_row_figure(vol, panels, title, out)
        print(f"  MedVAE nRMSE={vae_nr:.4f} | INR deploye nRMSE={inr_nr:.4f} | "
              f"INR plafond nRMSE={ceil_nr:.4f}  -> {out}")
        rows.append({"set": "B", "modality": MODALITY, "field": fld, "subject": subject,
                    "vae_nrmse": vae_nr, "vae_ssim": vae_ss,
                    "inr_deployed_nrmse": inr_nr, "inr_deployed_ssim": inr_ss,
                    "inr_ceiling_nrmse": ceil_nr, "inr_ceiling_ssim": ceil_ss})
        _write_csv(all_rows + rows, csv_path)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def _write_csv(rows: List[dict], out: Path) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="results/mmfm/qualitative_representation_20260909")
    ap.add_argument("--env", default="local")
    ap.add_argument("--subjects", default="0006,0007,0009", help="sujets du jeu A")
    ap.add_argument("--fields", default="0.1T,3T,7T", help="champs du jeu A")
    ap.add_argument("--set-b-subject", default="0006")
    ap.add_argument("--set-b-fields", default="0.1T,3T,7T")
    ap.add_argument("--skip-set-a", action="store_true")
    ap.add_argument("--skip-set-b", action="store_true", help="ne pas calculer le plafond LoRA (couteux)")
    ap.add_argument("--lora-steps", type=int, default=LORA_STEPS_DEFAULT)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(load_env(a.env)["data_root"]) / SPLIT_DIR
    stats = json.load(open(FIELD_NORM_STATS))["stats"]
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Chargement du backbone INR de production...")
    backbone = load_production_backbone(device)
    print(f"  latent_dim={backbone.cfg.latent_dim} modulation_dim={backbone.siren.modulation_dim} "
          f"hidden={backbone.cfg.hidden_dim}x{backbone.cfg.num_hidden_layers}")
    coords = make_coord_grid(VOLUME, device=device)

    csv_path = outdir / "metrics.csv"
    existing = _read_csv(csv_path)
    rows: List[dict] = list(existing)
    if not a.skip_set_a:
        rows += run_set_a(root, stats, backbone, coords, device, outdir,
                          [s.strip() for s in a.subjects.split(",")],
                          [s.strip() for s in a.fields.split(",")],
                          existing, csv_path)
    if not a.skip_set_b:
        base = list(rows)
        rows += run_set_b(root, stats, backbone, coords, device, outdir,
                          a.set_b_subject.strip(),
                          [s.strip() for s in a.set_b_fields.split(",")],
                          a.lora_steps, base, csv_path)

    print(f"\nFigures et metrics.csv sauves dans : {outdir}")


if __name__ == "__main__":
    main()
