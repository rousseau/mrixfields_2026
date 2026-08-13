#!/usr/bin/env python3
"""Figure qualitative : décodeur implicite (grille) vs décodeur convolutif MedVAE.

Complément visuel à `eval_grid_inr_decoder.py`. La question posée est celle du
DÉTAIL à 0.5mm : les métriques globales (nRMSE, SSIM) sont dominées par les
grandes structures et par le fond, et masquent justement ce qui distingue les
deux décodeurs. On montre donc, à partir du MÊME latent 1mm :

    vérité 0.5mm | conv 1mm + trilinéaire | INR interrogé à 0.5mm

avec un agrandissement sur une région corticale (là où le détail se joue) et
les cartes d'erreur associées, à échelle commune pour que la comparaison soit
lisible.

Rappel de ce qui est comparé : le décodeur convolutif de MedVAE a un facteur de
suréchantillonnage FIGÉ à x4 et ne peut pas produire du 0.5mm depuis un latent
1mm — l'interpolation trilinéaire est donc ce que l'évaluateur du challenge lui
applique de toute façon, pas un handicap qu'on lui inflige.

Usage:
    PYTHONPATH=src python src/cfm/figures_grid_inr_decoder.py --env local
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.config import load_yaml_with_include, load_env, resolve_paths
from common.io import SPLIT_MAP
from common.metrics import compute_ssim
from cfm.eval_grid_inr_decoder import ground_truth_pair, to01
from cfm.grid_inr_decoder import GridConditionedINR, GridINRConfig
from cfm.train_grid_inr_decoder import subject_of
from models.tiled_vae import tiled_decode
from models.vae_loader import load_vae

OUT_DIR = Path("results/mmfm/grid_inr_decoder")


def main():
    ap = argparse.ArgumentParser(description="Figure qualitative décodeur grille vs MedVAE")
    ap.add_argument("--config", default="configs/mmfm/grid_inr_decoder.yaml")
    ap.add_argument("--vae-config", default="configs/mmfm/unet.yaml")
    ap.add_argument("--env", default="local")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--weights", choices=["ema", "raw"], default="ema")
    ap.add_argument("--n-volumes", type=int, default=3)
    ap.add_argument("--zoom", type=int, default=112, help="Côté de l'agrandissement (voxels 0.5mm)")
    ap.add_argument("--out", default=str(OUT_DIR / "reconstruction_0p5mm.png"))
    args = ap.parse_args()

    env = load_env(args.env)
    cfg = resolve_paths(load_yaml_with_include(args.config), env)
    d, m = cfg["data"], cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size_1mm = tuple(int(v) for v in d["volume_size"])
    size_05 = tuple(2 * v for v in size_1mm)
    output_dir = Path(d["output_dir"])

    val_subs = set(json.loads((output_dir / "val_subjects.json").read_text())["subjects"])
    index = json.loads((Path(d["latent_cache_dir"]) / "index.json").read_text())
    held = [s for s in index["samples"] if subject_of(s["path"]) in val_subs]

    # Un contraste différent par ligne quand c'est possible : le détail cortical
    # ne se lit pas de la même façon en T1W, T2W et T2FLAIR.
    chosen, seen = [], set()
    for s in sorted(held, key=lambda x: x["path"]):
        mod = Path(s["path"]).parts[-3]
        if mod not in seen:
            chosen.append(s); seen.add(mod)
        if len(chosen) == args.n_volumes:
            break
    chosen += [s for s in sorted(held, key=lambda x: x["path"])
               if s not in chosen][: args.n_volumes - len(chosen)]

    ckpt_path = Path(args.checkpoint or (output_dir / "weights" / "model_final.pth"))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = GridConditionedINR(GridINRConfig(
        latent_channels=int(m.get("latent_channels", 1)), feat_dim=int(m.get("feat_dim", 64)),
        n_res_blocks=int(m.get("n_res_blocks", 2)), hidden_dim=int(m.get("hidden_dim", 256)),
        num_hidden_layers=int(m.get("num_hidden_layers", 5)),
        omega_0=float(m.get("omega_0", 30.0)), omega_hidden=float(m.get("omega_hidden", 30.0)),
    )).to(device)
    state = ckpt["model"]
    if args.weights == "ema" and isinstance(ckpt.get("ema"), dict) and set(ckpt["ema"]) >= set(state):
        state = {k: ckpt["ema"][k] for k in state}
    model.load_state_dict(state); model.eval()

    vae_cfg = resolve_paths(load_yaml_with_include(args.vae_config), env)
    vae = load_vae(vae_cfg, device)
    tile = tuple(int(v) for v in vae_cfg["data"].get("encode_tile", [96, 112, 96]))
    margin = int(vae_cfg["data"].get("encode_tile_margin", 16))

    data_root = Path(d["data_root"]); split_dir = SPLIT_MAP[d.get("split", "retro_train")]
    cache_root = Path(d["latent_cache_root"])

    n = len(chosen)
    fig, axes = plt.subplots(n, 6, figsize=(19, 3.4 * n))
    axes = np.atleast_2d(axes)

    for r, s in enumerate(chosen):
        rel = Path(s["path"]); mod, field, name = rel.parts[-3], rel.parts[-2], rel.stem
        z = torch.load(cache_root / rel, map_location="cpu").float().unsqueeze(0).to(device)
        nii = data_root / split_dir / mod / field / f"{name}.nii.gz"
        _, vol05 = ground_truth_pair(nii, size_1mm, float(d.get("percentile_lower", 0.5)),
                                     float(d.get("percentile_upper", 99.5)))
        with torch.no_grad():
            conv1 = tiled_decode(vae, z, tile=tile, margin=margin)
            conv05 = F.interpolate(conv1, size=size_05, mode="trilinear", align_corners=False)
            inr05 = model.decode_grid(z, size_05)

        gt = to01(vol05)
        cv = to01(conv05[0, 0].cpu().numpy())
        ir = to01(inr05[0, 0].cpu().numpy())
        sl = size_05[2] // 2
        g2, c2, i2 = gt[:, :, sl], cv[:, :, sl], ir[:, :, sl]

        # Agrandissement centré sur le barycentre du tissu, décalé vers le
        # cortex : c'est là que se joue la différence entre les deux décodeurs.
        ys, xs = np.nonzero(g2 > 0.15)
        cy = int(np.clip(ys.mean() - g2.shape[0] * 0.18, args.zoom // 2, g2.shape[0] - args.zoom // 2))
        cx = int(np.clip(xs.mean(), args.zoom // 2, g2.shape[1] - args.zoom // 2))
        h = args.zoom // 2
        box = (slice(cy - h, cy + h), slice(cx - h, cx + h))

        panels = [
            (g2, "vérité 0.5mm", dict(cmap="gray", vmin=0, vmax=1)),
            (c2, f"conv+trilin. (SSIM {compute_ssim(cv, gt):.3f})", dict(cmap="gray", vmin=0, vmax=1)),
            (i2, f"INR @0.5mm (SSIM {compute_ssim(ir, gt):.3f})", dict(cmap="gray", vmin=0, vmax=1)),
            (g2[box], "vérité (zoom)", dict(cmap="gray", vmin=0, vmax=1)),
            (c2[box], "conv+trilin. (zoom)", dict(cmap="gray", vmin=0, vmax=1)),
            (i2[box], "INR (zoom)", dict(cmap="gray", vmin=0, vmax=1)),
        ]
        for c, (img, title, kw) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(np.rot90(img), **kw)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            if c == 0:
                ax.text(-0.06, 0.5, f"{mod} {field}", transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=10)
        # Cadre de l'agrandissement, reporté sur la vue entière.
        axes[r, 0].add_patch(plt.Rectangle((box[0].start, g2.shape[1] - box[1].stop),
                                           args.zoom, args.zoom, fill=False, ec="yellow", lw=1.2))

    fig.suptitle(f"Décodage à 0.5mm depuis un latent 1mm identique — "
                 f"{ckpt_path.name} (iter {ckpt.get('iter', '?')})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Figure : {out}")


if __name__ == "__main__":
    main()
