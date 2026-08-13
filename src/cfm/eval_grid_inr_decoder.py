#!/usr/bin/env python3
"""Auto-reconstruction : décodeur implicite (grille) VS décodeur convolutif MedVAE.

Question posée
--------------
Sur des LATENTS IDENTIQUES (ceux déjà en cache, encodeur MedVAE gelé), le
décodeur implicite conditionné par grille reconstruit-il aussi bien que le
décodeur convolutif de MedVAE — et surtout, gagne-t-il quelque chose là où il
est censé gagner, c'est-à-dire en décodant DIRECTEMENT à 0.5mm ?

Le seul argument sérieux pour réintroduire un INR dans ce pipeline est que
l'évaluateur du challenge ré-interpole les prédictions vers 0.5mm : le décodeur
convolutif produit 1mm puis subit une interpolation trilinéaire (qui n'invente
rien), tandis que le décodeur implicite peut être interrogé directement aux
coordonnées 0.5mm. Si ce gain n'existe pas, l'architecture n'a pas d'intérêt et
il faut le dire.

Quatre mesures, deux comparaisons
---------------------------------
  @1mm     conv  = tiled_decode(MedVAE)            vs  inr = decode_grid(1mm)
           -> l'INR atteint-il seulement le niveau du décodeur qu'il remplace ?
  @0.5mm   conv  = tiled_decode puis trilinéaire x2 vs  inr = decode_grid(0.5mm)
           -> C'EST LA MESURE QUI DÉCIDE. Même latent, même information
              disponible ; seul le mode de restitution diffère.

Précautions
-----------
- Sujets HORS ÉCHANTILLON, lus dans `val_subjects.json` écrit par
  l'entraînement (partage par sujet, pas par volume). Le décodeur convolutif
  MedVAE n'a jamais vu ces données non plus (poids pré-entraînés) : les deux
  sont donc logés à la même enseigne.
- Normalisation : lo/hi calculés UNE FOIS sur le volume rééchantillonné à 1mm
  (exactement comme au precompute des latents), puis appliqués tels quels au
  volume 0.5mm. Recalculer les percentiles par résolution introduirait un
  décalage d'intensité entre les deux vérités terrain.
- Poids EMA par défaut (`--weights ema`), comme partout ailleurs dans le projet.

Usage:
    PYTHONPATH=src python src/cfm/eval_grid_inr_decoder.py \\
        --config configs/mmfm/grid_inr_decoder.yaml --env local
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_yaml_with_include, load_env, resolve_paths
from common.io import (SPLIT_MAP, center_crop_or_pad_np, load_nifti_volume,
                       normalize_volume_fixed)
from common.metrics import compute_nrmse, compute_ssim
from cfm.grid_inr_decoder import GridConditionedINR, GridINRConfig
from cfm.train_grid_inr_decoder import subject_of
from models.tiled_vae import tiled_decode
from models.vae_loader import load_vae

OUT_CSV = Path("results/mmfm/grid_inr_decoder/reconstruction.csv")


def to01(x: np.ndarray) -> np.ndarray:
    """[-1, 1] -> [0, 1], convention de métriques du projet."""
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


def score(rec: np.ndarray, gt01: np.ndarray, fg: np.ndarray) -> Dict[str, float]:
    r = to01(rec)
    return {"nrmse": float(compute_nrmse(r, gt01, mask=fg)),
            "nrmse_full": float(compute_nrmse(r, gt01)),
            "ssim": float(compute_ssim(r, gt01))}


def ground_truth_pair(nii: Path, size_1mm: Tuple[int, int, int],
                      p_lo: float, p_hi: float) -> Tuple[np.ndarray, np.ndarray]:
    """(volume 1mm, volume 0.5mm) normalisés sur la MÊME échelle d'intensité.

    lo/hi viennent du volume 1mm — c'est ce qu'a vu le precompute des latents.
    """
    raw1, _ = load_nifti_volume(nii, target_spacing=(1.0, 1.0, 1.0), volume_size=None,
                                normalize=False)
    lo, hi = float(np.percentile(raw1, p_lo)), float(np.percentile(raw1, p_hi))
    vol1 = center_crop_or_pad_np(normalize_volume_fixed(raw1, lo, hi), size_1mm)

    raw05, _ = load_nifti_volume(nii, target_spacing=(0.5, 0.5, 0.5), volume_size=None,
                                 normalize=False)
    size_05 = tuple(2 * s for s in size_1mm)
    vol05 = center_crop_or_pad_np(normalize_volume_fixed(raw05, lo, hi), size_05)
    return vol1, vol05


def main():
    ap = argparse.ArgumentParser(description="Auto-reconstruction : décodeur grille vs MedVAE")
    ap.add_argument("--config", default="configs/mmfm/grid_inr_decoder.yaml")
    ap.add_argument("--vae-config", default="configs/mmfm/unet.yaml",
                    help="Config portant le bloc `vae:` ayant produit le cache de latents")
    ap.add_argument("--env", default="local")
    ap.add_argument("--weights", choices=["ema", "raw"], default="ema")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max-volumes", type=int, default=None,
                    help="Sous-ensemble STRATIFIÉ par (contraste, champ). "
                         "Défaut : tous les volumes réservés.")
    ap.add_argument("--skip-half-mm", action="store_true",
                    help="Ne mesurer qu'à 1mm (rapide, mise au point)")
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    args = ap.parse_args()

    env = load_env(args.env)
    cfg = resolve_paths(load_yaml_with_include(args.config), env)
    d, m = cfg["data"], cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size_1mm = tuple(int(v) for v in d["volume_size"])
    p_lo = float(d.get("percentile_lower", 0.5))
    p_hi = float(d.get("percentile_upper", 99.5))
    output_dir = Path(d["output_dir"])

    # --- sujets réservés ---------------------------------------------------
    val_path = output_dir / "val_subjects.json"
    if not val_path.exists():
        raise SystemExit(f"{val_path} absent : lancer d'abord l'entraînement "
                         "(il écrit le partage par sujet).")
    val_subs = set(json.loads(val_path.read_text())["subjects"])

    index = json.loads((Path(d["latent_cache_dir"]) / "index.json").read_text())
    samples = sorted([s for s in index["samples"] if subject_of(s["path"]) in val_subs],
                     key=lambda s: s["path"])
    if args.max_volumes and args.max_volumes < len(samples):
        # Sous-échantillonnage STRATIFIÉ par (contraste, champ), en tourniquet.
        # Une simple troncature du tri alphabétique ne retiendrait que des T1W
        # (mesuré : 24/24) et laisserait T2W et T2FLAIR hors de l'évaluation —
        # or c'est précisément là que les écarts se creusent sur ce projet.
        strata: Dict[tuple, List[dict]] = {}
        for s in samples:
            strata.setdefault(tuple(Path(s["path"]).parts[-3:-1]), []).append(s)
        picked, keys = [], sorted(strata)
        while len(picked) < args.max_volumes:
            progressed = False
            for k in keys:
                if strata[k]:
                    picked.append(strata[k].pop(0)); progressed = True
                    if len(picked) == args.max_volumes:
                        break
            if not progressed:
                break
        samples = sorted(picked, key=lambda s: s["path"])
    counts = Counter(tuple(Path(s["path"]).parts[-3:-1]) for s in samples)
    print(f"Évaluation sur {len(samples)} volumes hors échantillon "
          f"({len(val_subs)} sujets réservés) — "
          + ", ".join(f"{m}/{f}:{n}" for (m, f), n in sorted(counts.items())), flush=True)

    # --- décodeur implicite ------------------------------------------------
    ckpt_path = Path(args.checkpoint or (output_dir / "weights" / "model_final.pth"))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = GridConditionedINR(GridINRConfig(
        latent_channels=int(m.get("latent_channels", 1)),
        feat_dim=int(m.get("feat_dim", 64)),
        n_res_blocks=int(m.get("n_res_blocks", 2)),
        hidden_dim=int(m.get("hidden_dim", 256)),
        num_hidden_layers=int(m.get("num_hidden_layers", 5)),
        omega_0=float(m.get("omega_0", 30.0)),
        omega_hidden=float(m.get("omega_hidden", 30.0)),
    )).to(device)
    state = ckpt["model"]
    if args.weights == "ema" and ckpt.get("ema"):
        ema = ckpt["ema"]
        ema = ema.get("shadow", ema) if isinstance(ema, dict) else ema
        if isinstance(ema, dict) and set(ema) >= set(state):
            state = {k: ema[k] for k in state}
            print("  poids EMA")
        else:
            print("  ⚠ EMA illisible, repli sur les poids bruts")
    model.load_state_dict(state)
    model.eval()
    print(f"  checkpoint : {ckpt_path} (iter {ckpt.get('iter', '?')})", flush=True)

    # --- décodeur convolutif de référence ----------------------------------
    vae_cfg = resolve_paths(load_yaml_with_include(args.vae_config), env)
    vae = load_vae(vae_cfg, device)
    tile = tuple(int(v) for v in vae_cfg["data"].get("encode_tile", [96, 112, 96]))
    margin = int(vae_cfg["data"].get("encode_tile_margin", 16))

    data_root = Path(d["data_root"])
    split_dir = SPLIT_MAP[d.get("split", "retro_train")]
    cache_root = Path(d["latent_cache_root"])
    rows: List[dict] = []

    for n, s in enumerate(samples, 1):
        rel = Path(s["path"])
        mod, field, name = rel.parts[-3], rel.parts[-2], rel.stem
        z = torch.load(cache_root / rel, map_location="cpu").float().unsqueeze(0).to(device)
        nii = data_root / split_dir / mod / field / f"{name}.nii.gz"

        vol1, vol05 = ground_truth_pair(nii, size_1mm, p_lo, p_hi)
        gt1, gt05 = to01(vol1), to01(vol05)
        fg1, fg05 = gt1 > 0.025, gt05 > 0.025

        t0 = time.time()
        with torch.no_grad():
            conv1 = tiled_decode(vae, z, tile=tile, margin=margin)      # (1,1,192,224,192)
            inr1 = model.decode_grid(z, size_1mm)
        r = {"conv@1mm": score(conv1[0, 0].cpu().numpy(), gt1, fg1),
             "inr@1mm": score(inr1[0, 0].cpu().numpy(), gt1, fg1)}

        if not args.skip_half_mm:
            size_05 = tuple(2 * v for v in size_1mm)
            with torch.no_grad():
                # Le décodeur convolutif ne SAIT PAS produire du 0.5mm : on lui
                # applique l'interpolation que l'évaluateur ferait de toute façon.
                conv05 = F.interpolate(conv1, size=size_05, mode="trilinear",
                                       align_corners=False)
                inr05 = model.decode_grid(z, size_05)
            r["conv@0.5mm"] = score(conv05[0, 0].cpu().numpy(), gt05, fg05)
            r["inr@0.5mm"] = score(inr05[0, 0].cpu().numpy(), gt05, fg05)
            del conv05, inr05
        del conv1, inr1
        torch.cuda.empty_cache()

        for variant, sc in r.items():
            rows.append({"volume": name, "modality": mod, "field": field,
                         "subject": subject_of(s["path"]), "variant": variant,
                         **{k: round(v, 5) for k, v in sc.items()}})
        print(f"  [{n:3d}/{len(samples)}] {name:28s} " +
              " | ".join(f"{v}: nRMSE={sc['nrmse']:.4f} SSIM={sc['ssim']:.4f}"
                         for v, sc in r.items()) + f"  ({time.time() - t0:.0f}s)",
              flush=True)

        out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    # --- synthèse -----------------------------------------------------------
    print(f"\n{'=' * 78}\nRÉSUMÉ ({len(samples)} volumes hors échantillon)\n{'=' * 78}")
    print(f"{'variante':14s} {'nRMSE (fg)':>12s} {'nRMSE':>10s} {'SSIM':>10s}")
    order = ["conv@1mm", "inr@1mm", "conv@0.5mm", "inr@0.5mm"]
    means = {}
    for v in order:
        sub = [r for r in rows if r["variant"] == v]
        if not sub:
            continue
        means[v] = {k: float(np.mean([r[k] for r in sub]))
                    for k in ("nrmse", "nrmse_full", "ssim")}
        print(f"{v:14s} {means[v]['nrmse']:12.4f} {means[v]['nrmse_full']:10.4f} "
              f"{means[v]['ssim']:10.4f}")

    def verdict(a: str, b: str, label: str) -> None:
        if a not in means or b not in means:
            return
        ds = means[b]["ssim"] - means[a]["ssim"]
        dn = means[b]["nrmse"] - means[a]["nrmse"]
        sign = "GAGNE" if (ds > 0 and dn < 0) else ("PERD" if (ds < 0 and dn > 0) else "MIXTE")
        print(f"\n{label} : l'INR {sign}  (ΔSSIM={ds:+.4f}, ΔnRMSE={dn:+.4f})")

    verdict("conv@1mm", "inr@1mm", "À 1mm, contre le décodeur qu'il remplace")
    verdict("conv@0.5mm", "inr@0.5mm",
            "À 0.5mm — LA MESURE QUI DÉCIDE (même latent, restitution différente)")

    # Ventilation : une moyenne globale peut masquer des comportements opposés
    # (sur ce projet, 5T et 7T se comportent différemment du reste — voir le
    # manifest, section sur la calibration MedVAE).
    for axis, label in (("modality", "contraste"), ("field", "champ")):
        print(f"\n--- @0.5mm par {label} ---")
        print(f"{label:10s} {'n':>3s} {'conv SSIM':>10s} {'inr SSIM':>9s} {'Δ':>8s} "
              f"{'conv nRMSE':>11s} {'inr nRMSE':>10s} {'Δ':>8s}")
        for key in sorted({r[axis] for r in rows}):
            c = [r for r in rows if r[axis] == key and r["variant"] == "conv@0.5mm"]
            i = [r for r in rows if r[axis] == key and r["variant"] == "inr@0.5mm"]
            if not c or not i:
                continue
            cs, is_ = np.mean([r["ssim"] for r in c]), np.mean([r["ssim"] for r in i])
            cn, in_ = np.mean([r["nrmse"] for r in c]), np.mean([r["nrmse"] for r in i])
            print(f"{key:10s} {len(c):3d} {cs:10.4f} {is_:9.4f} {is_ - cs:+8.4f} "
                  f"{cn:11.4f} {in_:10.4f} {in_ - cn:+8.4f}")

    print(f"\nCSV : {args.out_csv}")


if __name__ == "__main__":
    main()
