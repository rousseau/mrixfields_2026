#!/usr/bin/env python3
"""
Benchmark specifique du disentanglement MedVAE v1 (anatomie/modalite).

Mesures principales:
- Acc(modalite | z_m)      : doit etre elevee
- Acc(modalite | z_a)      : doit etre faible (invariance anatomique)
- L1 reconstruction intra  : qualite reconstruction source
- L1 cross-modal (paired)  : transfert de modalite a anatomie fixe
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vae3d.train_medvae_disentangle_v1 import MedVAEDisentanglerV1, _decode_medvae, _encode_medvae_deterministic, load_medvae
from vae3d.train_vqvae import MRIxFieldsHybridDataset


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    ds = MRIxFieldsHybridDataset(
        data_root=Path(args.data_root),
        splits=args.splits,
        modalities=args.modalities,
        fields=args.fields,
        volume_size=tuple(args.volume_size),
        paired_prob=args.paired_prob,
        percentile_lower=args.percentile_lower,
        percentile_upper=args.percentile_upper,
        target_spacing=tuple(args.target_spacing) if args.target_spacing else None,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    medvae = load_medvae(args.medvae_model_name, device)
    with torch.no_grad():
        dummy = torch.zeros(1, 1, *args.volume_size, device=device)
        z = _encode_medvae_deterministic(medvae, dummy)
        latent_channels = int(z.shape[1])

    model = MedVAEDisentanglerV1(
        medvae=medvae,
        latent_channels=latent_channels,
        n_modalities=len(args.modalities),
        anat_channels=int(ckpt_args.get("anat_channels", 8)),
        style_dim=int(ckpt_args.get("style_dim", 32)),
        film_hidden=int(ckpt_args.get("film_hidden", 128)),
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}")
    model.eval()

    total = 0
    correct_zm = 0
    correct_za = 0
    rec_l1_vals = []
    cross_l1_vals = []
    paired_count = 0

    for batch in loader:
        x_src = batch["x_src"].to(device)
        x_tgt = batch["x_tgt"].to(device)
        src_mod = batch["src_mod"].to(device)
        tgt_mod = batch["tgt_mod"].to(device)
        is_paired = batch["is_paired"].to(device)

        out = model(x_src, src_mod, adv_alpha=1.0)

        # z_m classifier
        pred_zm = out["mod_logits"].argmax(dim=1)
        correct_zm += int((pred_zm == src_mod).sum().item())

        # z_a adversarial classifier (forward identique, GRL agit seulement au backward)
        pred_za = out["adv_logits"].argmax(dim=1)
        correct_za += int((pred_za == src_mod).sum().item())

        total += int(src_mod.numel())

        rec_l1 = F.l1_loss(out["x_rec"], x_src, reduction="none").mean(dim=(1, 2, 3, 4))
        rec_l1_vals.extend(rec_l1.detach().cpu().tolist())

        mask = is_paired > 0.5
        if mask.any():
            _, z_a_src, _ = model.encode_parts(x_src[mask])
            _, _, z_m_tgt = model.encode_parts(x_tgt[mask])
            z_cross = model.fuse_to_latent(z_a_src, z_m_tgt, tgt_mod[mask])
            x_cross = _decode_medvae(model.medvae, z_cross, out_shape=x_src.shape[2:])
            c_l1 = F.l1_loss(x_cross, x_tgt[mask], reduction="none").mean(dim=(1, 2, 3, 4))
            cross_l1_vals.extend(c_l1.detach().cpu().tolist())
            paired_count += int(mask.sum().item())

    acc_zm = correct_zm / max(1, total)
    acc_za = correct_za / max(1, total)

    results = {
        "checkpoint": str(args.ckpt),
        "n_samples": total,
        "n_paired": paired_count,
        "acc_mod_from_zm": float(acc_zm),
        "acc_mod_from_za": float(acc_za),
        "rec_l1_mean": float(np.mean(rec_l1_vals)) if rec_l1_vals else float("nan"),
        "rec_l1_std": float(np.std(rec_l1_vals)) if rec_l1_vals else float("nan"),
        "cross_l1_mean": float(np.mean(cross_l1_vals)) if cross_l1_vals else float("nan"),
        "cross_l1_std": float(np.std(cross_l1_vals)) if cross_l1_vals else float("nan"),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "benchmark_disentanglement_v1.json"
    out_csv = out_dir / "benchmark_disentanglement_v1.csv"

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    with open(out_csv, "w") as f:
        f.write("metric,value\n")
        for k, v in results.items():
            f.write(f"{k},{v}\n")

    print("=" * 72)
    print("Benchmark Disentanglement v1")
    print("=" * 72)
    print(f"samples={results['n_samples']} paired={results['n_paired']}")
    print(f"Acc(mod|z_m)={results['acc_mod_from_zm']:.4f}  (plus eleve est mieux)")
    print(f"Acc(mod|z_a)={results['acc_mod_from_za']:.4f}  (plus faible est mieux)")
    print(f"Rec L1={results['rec_l1_mean']:.4f} +/- {results['rec_l1_std']:.4f}")
    print(f"Cross L1={results['cross_l1_mean']:.4f} +/- {results['cross_l1_std']:.4f}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark MedVAE disentanglement v1")
    p.add_argument("--ckpt", type=str, default="outputs/medvae_disentangle_v1/runs/dev_run/weights/model_best.pth")
    p.add_argument("--data-root", type=str, default="/home/rousseau/Data/MRIxFields_20260414")
    p.add_argument("--output-dir", type=str, default="results/benchmark_disentanglement_v1")

    p.add_argument("--splits", nargs="+", default=["retro_train"])
    p.add_argument("--modalities", nargs="+", default=["T1W", "T2W", "T2FLAIR"])
    p.add_argument("--fields", nargs="+", default=["0.1T", "1.5T", "3T", "5T", "7T"])

    p.add_argument("--volume-size", nargs=3, type=int, default=[112, 128, 80])
    p.add_argument("--target-spacing", nargs=3, type=float, default=None)
    p.add_argument("--percentile-lower", type=float, default=0.5)
    p.add_argument("--percentile-upper", type=float, default=99.5)
    p.add_argument("--paired-prob", type=float, default=0.5)
    p.add_argument("--max-samples", type=int, default=64)

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--medvae-model-name", type=str, default="medvae_4_1_3d")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
