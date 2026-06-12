#!/usr/bin/env python3
"""
CFM 3D — Inférence + visualisation 0.1T → 7T sur sujets de validation T1W.

Ce script :
  1. Charge le checkpoint CFM (model_final.pth) entraîné sur 0.1T↔7T
  2. Pour chaque sujet 0.1T de Validating_prospective :
       encode (MedVAE) → Euler ODE 50 steps → decode → prédiction 7T
  3. Génère des figures de comparaison dans l'espace d'inférence 1mm cropé :
       [ 0.1T source | CFM 7T prédit | GT 7T ] × [axiale | coronale | sagittale]

Appariement sujets (Validating_prospective) :
  P_T1W_0.1T_0001  ↔  P_T1W_7T_0016  (1er sujet)
  P_T1W_0.1T_0002  ↔  P_T1W_7T_0017  (2ème sujet)
  P_T1W_0.1T_0003  ↔  P_T1W_7T_0018  (3ème sujet)
  Appariement par ordre alphabétique (position dans la liste triée).

Usage :
  python src/cfm/run_inference_viz_cfm3d.py
  python src/cfm/run_inference_viz_cfm3d.py \\
      --config   configs/cfm3d_T1W_medvae_0p1T_7T.yaml \\
      --checkpoint outputs/cfm3d/runs/cfm3d_T1W_medvae_0p1T_7T/weights/model_final.pth \\
      --output   results/cfm/visuals/cfm3d_0p1T_7T \\
      --n-steps  50
"""

import argparse
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

# ── Path setup ─────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

from cfm.train_cfm_3d import (
    DOMAIN_TO_IDX,
    _center_crop_or_pad_np,
    _resample_volume,
    build_unet_3d,
    load_vae,
)

# ═══════════════════════════════════════════════════════════════════════════
# Utilitaires
# ═══════════════════════════════════════════════════════════════════════════


def _preprocess(nii_path: Path, target_spacing, volume_size, p_lo=0.5, p_hi=99.5):
    """Charge un volume NIfTI → resample 1mm → normalize → crop centré.
    
    Note: Les données du challenge sont déjà normalisées en [0, 1] (sans rescaling).
    On applique le clipping percentile pour enlever les outliers, puis convertit en [-1, 1]
    pour cohérence avec l'entraînement VAE/CFM.
    """
    img = nib.load(str(nii_path))
    vol = img.get_fdata(dtype=np.float32)
    orig_spacing = np.abs(np.diag(img.affine)[:3])

    if target_spacing:
        vol = _resample_volume(vol, orig_spacing, target_spacing)

    # Les données sont déjà en [0, 1], on applique seulement le percentile clipping
    lo = np.percentile(vol, p_lo)
    hi = np.percentile(vol, p_hi)
    vol_n = np.clip(vol, lo, hi)
    
    # Normalisation min-max vers [-1, 1] pour cohérence avec l'entraînement
    vol_n = (vol_n - lo) / max(hi - lo, 1e-8)
    vol_n = vol_n * 2.0 - 1.0

    if volume_size:
        vol_n = _center_crop_or_pad_np(vol_n, volume_size)

    return vol_n  # shape = volume_size, range [-1, 1]


def _load_ckpt_compat(unet, ckpt_path: str, device, use_ema: bool = True):
    """Charge le checkpoint avec compatibilité MONAI attention layer naming."""
    import re

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    key = "ema" if (use_ema and "ema" in state) else "model"
    ckpt_state = state[key]
    current = unet.state_dict()

    # Mapping old MONAI → new MONAI attention key naming:
    #   old: "...attentions.N.to_q.weight"    → new: "...attentions.N.attn.to_q.weight"
    #   old: "...attentions.N.proj_attn.weight" → new: "...attentions.N.attn.out_proj.weight"
    compatible = {}
    for ck, cv in ckpt_state.items():
        if ck in current:
            compatible[ck] = cv
        else:
            mk = re.sub(
                r"\.attentions\.(\d+)\.(to_q|to_k|to_v|proj_attn)\.",
                r".attentions.\1.attn.\2.",
                ck,
            )
            mk = mk.replace("proj_attn", "out_proj")
            if mk in current:
                compatible[mk] = cv

    missing, unexpected = unet.load_state_dict(compatible, strict=False)
    n_compat = len(compatible)
    n_total = len(current)
    print(
        f"  [{key}] {n_compat}/{n_total} clés chargées"
        + (f"  ({len(missing)} manquantes)" if missing else "")
    )
    return unet, key


def _euler_integrate(
    vae, unet, vol_tensor, tgt_idx, n_steps, device, amp_dtype, use_amp
):
    """Encode → Euler ODE → decode. Retourne le volume prédit en [0,1]."""
    B = vol_tensor.shape[0]

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        z_src = vae.encode(vol_tensor)

    dt = 1.0 / n_steps
    z = z_src.clone()
    y = torch.full((B,), tgt_idx, dtype=torch.long, device=device)

    for i in range(n_steps):
        t_vec = torch.full((B,), i * dt, dtype=torch.float32, device=device)
        z_in = torch.cat([z, z_src], dim=1)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            vt = unet(x=z_in, timesteps=t_vec, class_labels=y)
        z = z + dt * vt

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        recon = vae.decode(z)

    pred = recon.squeeze().cpu().numpy()
    # Le décodage VAE retourne en [-1, 1], on convertit en [0, 1] pour l'affichage
    return np.clip((pred + 1.0) / 2.0, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Inférence
# ═══════════════════════════════════════════════════════════════════════════


def run_inference(
    cfg_path, ckpt_path, input_01t_dir, output_dir, n_steps=50, n_subjects=3
):
    """Lance l'inférence CFM sur les volumes 0.1T et sauvegarde les prédictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else torch.float32

    # ── Config ──────────────────────────────────────────────────────────────
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["data_root"] = "/home/rousseau/Data/MRIxFields_20260414"
    cfg["data"]["output_dir"] = cfg["data"]["output_subdir"]

    volume_size = tuple(int(v) for v in cfg["data"]["volume_size"])
    target_spacing = tuple(float(v) for v in cfg["data"]["target_spacing"])
    p_lo = cfg["data"].get("percentile_lower", 0.5)
    p_hi = cfg["data"].get("percentile_upper", 99.5)

    # ── Modèles ─────────────────────────────────────────────────────────────
    print("[1/4] Chargement VAE + UNet ...")
    vae = load_vae(cfg, device)
    unet = build_unet_3d(cfg, vae.latent_channels).to(device)
    unet, key = _load_ckpt_compat(unet, ckpt_path, device, use_ema=True)
    unet.eval()
    print(f"  VAE latent_channels={vae.latent_channels} | device={device}")

    tgt_idx = DOMAIN_TO_IDX["7T"]
    input_files = sorted(Path(input_01t_dir).glob("*.nii.gz"))[:n_subjects]
    print(f"[2/4] {len(input_files)} volumes à prédire (0.1T → 7T) ...")

    # ── Inférence par volume ─────────────────────────────────────────────────
    out_pred_dir = Path(output_dir) / "predictions_crop"
    out_pred_dir.mkdir(parents=True, exist_ok=True)

    results = []  # liste de dicts {fname_01t, pred_vol, src_vol}

    for nii_path in input_files:
        t0 = time.time()
        # Preprocessing (identique à l'entraînement)
        src_vol = _preprocess(
            nii_path, target_spacing, volume_size, p_lo, p_hi
        )  # [-1, 1]

        vol_tensor = (
            torch.from_numpy(src_vol).unsqueeze(0).unsqueeze(0).float().to(device)
        )  # (1,1,H,W,D) en [-1, 1] — cohérent avec l'entraînement

        pred_vol = _euler_integrate(
            vae, unet, vol_tensor, tgt_idx, n_steps, device, amp_dtype, use_amp
        )  # [0, 1] déjà dans [0,1]

        # src_vol : [-1, 1] → [0, 1] pour affichage
        src_vol_display = np.clip((src_vol + 1.0) / 2.0, 0.0, 1.0)

        # Sauvegarde dans l'espace cropé (128×128×80 @ 1mm)
        img_ref = nib.load(str(nii_path))
        orig_sp = np.abs(np.diag(img_ref.affine)[:3])
        # Construire un affine 1mm isotrope centré sur le crop
        resampled_shape = np.array(
            [round(img_ref.shape[i] * orig_sp[i] / target_spacing[i]) for i in range(3)]
        )
        affine_1mm = img_ref.affine.copy().astype(float)
        for i in range(3):
            affine_1mm[:3, i] *= target_spacing[i] / orig_sp[i]
        crop_offset = np.array(
            [(resampled_shape[i] - volume_size[i]) // 2 for i in range(3)], dtype=float
        )
        affine_1mm[:3, 3] += affine_1mm[:3, :3] @ crop_offset

        out_path = out_pred_dir / nii_path.name.replace("0.1T", "7T_pred")
        nib.save(
            nib.Nifti1Image(pred_vol.astype(np.float32), affine_1mm), str(out_path)
        )

        elapsed = time.time() - t0
        print(f"  {nii_path.name} → {out_path.name}  ({elapsed:.1f}s)")
        results.append(
            {"nii_01t": nii_path, "pred_vol": pred_vol, "src_vol": src_vol_display}
        )

    print(f"[3/4] Prédictions sauvegardées : {out_pred_dir}")
    return results, volume_size, target_spacing, p_lo, p_hi


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════


def _extract_slices(vol, view="axial"):
    """Extrait la coupe centrale d'un volume 3D (H,W,D) selon la vue."""
    h, w, d = vol.shape
    if view == "axial":  # Haut → bas (plan XY, milieu en Z)
        s = vol[:, :, d // 2]
        return np.rot90(s)
    elif view == "coronal":  # Avant → arrière (plan XZ, milieu en Y)
        s = vol[:, w // 2, :]
        return np.rot90(s)
    elif view == "sagittal":  # Gauche → droite (plan YZ, milieu en X)
        s = vol[h // 2, :, :]
        return np.rot90(s)


def generate_figures(
    results, input_dir_7t, volume_size, target_spacing, p_lo, p_hi, output_dir
):
    """Génère les figures de comparaison 0.1T | CFM 7T | GT 7T."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    out_fig_dir = Path(output_dir) / "figures"
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    # Appariement positionnel : 1er 0.1T ↔ 1er 7T (tri alphabétique)
    gt_files = sorted(Path(input_dir_7t).glob("*.nii.gz"))

    views = ["axial", "coronal", "sagittal"]
    view_labels = ["Axiale", "Coronale", "Sagittale"]

    # ── Métriques ──────────────────────────────────────────────────────────
    metrics_rows = []

    # ── Figure par sujet (3×3 : vues × colonnes) ─────────────────────────
    per_subject_figs = []

    for i, res in enumerate(results):
        nii_01t = res["nii_01t"]
        src_vol = res["src_vol"]  # (128,128,80) @ 1mm, [0,1]
        pred_vol = res["pred_vol"]  # (128,128,80) @ 1mm, [0,1]

        # GT 7T : les données sont déjà en [0, 1] (challenge preprocessing)
        gt_path = gt_files[i] if i < len(gt_files) else None
        if gt_path and gt_path.exists():
            gt_vol_raw = _preprocess(gt_path, target_spacing, volume_size, p_lo, p_hi)
            gt_vol = np.clip((gt_vol_raw + 1.0) / 2.0, 0.0, 1.0)  # [-1,1] → [0,1]
        else:
            print(f"  [WARN] GT 7T introuvable pour le sujet {i + 1}")
            gt_vol = np.zeros_like(src_vol)

        # Sujet ID depuis le nom de fichier : P_T1W_0.1T_0001.nii.gz → "0001"
        parts = nii_01t.name.split(".")[0].split("_")  # strip extension first
        subject_id = parts[-1] if len(parts) >= 4 else str(i + 1)
        gt_id = gt_files[i].name.split(".")[0].split("_")[-1] if gt_path else "N/A"

        # Métriques MAE / SSIM sur le crop (uniquement là où on a une prédiction)
        mae = float(np.mean(np.abs(pred_vol - gt_vol)))
        rmse = float(np.sqrt(np.mean((pred_vol - gt_vol) ** 2)))
        ss = float(
            np.mean((2 * pred_vol * gt_vol + 0.01) / (pred_vol**2 + gt_vol**2 + 0.01))
        )  # approx SSIM
        metrics_rows.append(
            {
                "sujet_0.1T": subject_id,
                "sujet_7T_GT": gt_id,
                "MAE": f"{mae:.4f}",
                "RMSE": f"{rmse:.4f}",
                "SSIM_approx": f"{ss:.4f}",
            }
        )
        print(f"  Sujet {subject_id} | MAE={mae:.4f}  RMSE={rmse:.4f}  SSIM≈{ss:.4f}")

        # ── Figure individuelle ────────────────────────────────────────────
        fig = plt.figure(figsize=(9, 9.5), dpi=120)
        fig.patch.set_facecolor("white")
        gs = GridSpec(
            4,
            3,
            figure=fig,
            hspace=0.08,
            wspace=0.04,
            top=0.90,
            bottom=0.02,
            left=0.10,
            right=0.98,
        )

        col_titles = ["0.1T (source)", "CFM 7T (prédit)", "7T (GT)"]
        col_colors = ["#333333", "#C0392B", "#1a6fa8"]
        volumes = [src_vol, pred_vol, gt_vol]

        # Ligne de titres colonnes
        for col, (title, color) in enumerate(zip(col_titles, col_colors)):
            ax = fig.add_subplot(gs[0, col])
            ax.text(
                0.5,
                0.5,
                title,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=color,
                transform=ax.transAxes,
            )
            ax.axis("off")

        # 3 lignes de vues
        for row, (view, vlabel) in enumerate(zip(views, view_labels)):
            for col, vol in enumerate(volumes):
                ax = fig.add_subplot(gs[row + 1, col])
                sl = _extract_slices(vol, view)
                ax.imshow(
                    sl,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    aspect="equal",
                    interpolation="nearest",
                )
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(vlabel, fontsize=9, rotation=90, labelpad=4)

            # Label de vue à gauche
            fig.text(
                0.02,
                0.74 - row * 0.27,
                vlabel,
                fontsize=9,
                rotation=90,
                va="center",
                ha="center",
                color="#555",
            )

        # Titre global
        fig.suptitle(
            f"CFM 3D  0.1T → 7T  |  Sujet val. {subject_id}  "
            f"[MAE={mae:.3f}  SSIM≈{ss:.3f}]",
            fontsize=12,
            fontweight="bold",
            y=0.97,
        )

        out_path = out_fig_dir / f"cfm3d_0p1T_7T_subject_{subject_id}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        per_subject_figs.append(out_path)
        print(f"  Figure → {out_path}")

    # ── Figure combinée (tous les sujets côte à côte) ─────────────────────
    n_subj = len(results)
    fig_all = plt.figure(figsize=(4 * n_subj + 0.5, 10), dpi=120)
    fig_all.patch.set_facecolor("white")
    gs_all = GridSpec(
        4,
        n_subj * 3,
        figure=fig_all,
        hspace=0.06,
        wspace=0.04,
        top=0.90,
        bottom=0.03,
        left=0.06,
        right=0.99,
    )

    col_titles = ["0.1T", "CFM 7T", "GT 7T"]
    col_colors = ["#333333", "#C0392B", "#1a6fa8"]

    for i, res in enumerate(results):
        src_vol = res["src_vol"]
        pred_vol = res["pred_vol"]
        gt_path = gt_files[i] if i < len(gt_files) else None
        # GT 7T : les données sont déjà en [0, 1], on convertit en [-1,1] puis [0,1]
        gt_vol = (
            np.clip(
                (_preprocess(gt_path, target_spacing, volume_size, p_lo, p_hi) + 1.0)
                / 2.0,
                0.0,
                1.0,
            )
            if gt_path and gt_path.exists()
            else np.zeros_like(src_vol)
        )

        parts = res["nii_01t"].name.split(".")[0].split("_")
        subject_id = parts[-1] if len(parts) >= 4 else str(i + 1)
        gt_id = gt_files[i].name.split(".")[0].split("_")[-1] if gt_path else "?"

        volumes = [src_vol, pred_vol, gt_vol]

        # Ligne 0 : titres de colonnes
        for c, (ttl, clr) in enumerate(zip(col_titles, col_colors)):
            ax = fig_all.add_subplot(gs_all[0, i * 3 + c])
            lbl = (
                f"{ttl}\n{subject_id if c == 0 else (subject_id if c == 1 else gt_id)}"
            )
            ax.text(
                0.5,
                0.5,
                lbl,
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color=clr,
                transform=ax.transAxes,
            )
            ax.axis("off")

        # Lignes 1-3 : vues
        for row, view in enumerate(views):
            for c, vol in enumerate(volumes):
                ax = fig_all.add_subplot(gs_all[row + 1, i * 3 + c])
                sl = _extract_slices(vol, view)
                ax.imshow(
                    sl,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    aspect="equal",
                    interpolation="nearest",
                )
                ax.axis("off")

    # Labels vues à gauche
    for row, vlabel in enumerate(view_labels):
        fig_all.text(
            0.01,
            0.72 - row * 0.22,
            vlabel,
            fontsize=8,
            rotation=90,
            va="center",
            ha="center",
            color="#555",
        )

    fig_all.suptitle(
        f"OT-CFM 3D  0.1T → 7T  |  {n_subj} sujets validation T1W  "
        f"(MedVAE fine-tuné + UNet 178M, 100k iters)",
        fontsize=11,
        fontweight="bold",
        y=0.97,
    )

    out_all = out_fig_dir / "cfm3d_0p1T_7T_all_subjects.png"
    fig_all.savefig(str(out_all), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig_all)
    print(f"\n  Figure combinée → {out_all}")

    # ── Résumé métriques ───────────────────────────────────────────────────
    print("\n  === Métriques (espace 1mm cropé 128×128×80) ===")
    print(
        f"  {'Sujet 0.1T':>10} | {'GT 7T':>8} | {'MAE':>7} | {'RMSE':>7} | {'SSIM≈':>7}"
    )
    print("  " + "-" * 55)
    for m in metrics_rows:
        print(
            f"  {m['sujet_0.1T']:>10} | {m['sujet_7T_GT']:>8} | "
            f"{m['MAE']:>7} | {m['RMSE']:>7} | {m['SSIM_approx']:>7}"
        )

    return out_fig_dir, metrics_rows, out_all


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="CFM 3D 0.1T→7T : inférence + visualisation"
    )
    parser.add_argument("--config", default="configs/cfm3d_T1W_medvae_0p1T_7T.yaml")
    parser.add_argument(
        "--checkpoint",
        default="outputs/cfm3d/runs/cfm3d_T1W_medvae_0p1T_7T/weights/model_final.pth",
    )
    parser.add_argument(
        "--input-01t",
        default="/home/rousseau/Data/MRIxFields_20260414/Training_prospective/T1W/0.1T",
    )
    parser.add_argument(
        "--input-7t",
        default="/home/rousseau/Data/MRIxFields_20260414/Training_prospective/T1W/7T",
    )
    parser.add_argument("--output", default="results/cfm/visuals/cfm3d_0p1T_7T")
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--n-subjects", type=int, default=3)
    args = parser.parse_args()

    print("=" * 65)
    print("  OT-CFM 3D  0.1T → 7T  |  Validation T1W")
    print("=" * 65)

    results, volume_size, target_spacing, p_lo, p_hi = run_inference(
        args.config,
        args.checkpoint,
        args.input_01t,
        args.output,
        args.n_steps,
        args.n_subjects,
    )

    print("\n[4/4] Génération des figures ...")
    out_fig_dir, metrics, out_all = generate_figures(
        results,
        args.input_7t,
        volume_size,
        target_spacing,
        p_lo,
        p_hi,
        args.output,
    )

    print("\n" + "=" * 65)
    print(f"  Figures : {out_fig_dir}")
    print(f"  Combinée: {out_all}")
    print("=" * 65)


if __name__ == "__main__":
    main()
