#!/usr/bin/env python3
"""Smoke test — Phase 1 of the INR backbone (SIREN + hypernetwork + meta-learning).

Validates the mechanism in src/cfm/inr_backbone.py on a handful of real
retro_train volumes — no MedVAE / flow-matching involved yet (see
docs/MMFM_INR_STATE_OF_THE_ART.md, Plan phase 1 / "porte 1"):

  1. Meta-train the shared backbone (theta, psi) — loss should decrease.
  2. z-discrimination: fit z independently for two DIFFERENT volumes and
     check the resulting reconstructions actually differ. This is the test
     that matters most and is NOT redundant with the others: an early
     version of this backbone passed every other check while `z` stayed
     pinned at its zero-init regardless of which volume was fitted (plain
     voxel-wise MSE let the model reach a deceptively low loss via a
     generic "average brain blob" alone, since after center-crop/pad most
     voxels are flat background shared by every subject — see
     inr_backbone.py's _weighted_mse docstring for the fix). Without this
     check that collapse would silently pass Phase 1.
  3. Fit-and-reconstruct (Algorithm 2) on a training volume: nRMSE/SSIM,
     whole-volume AND foreground-masked (the latter is what actually
     reflects anatomical fidelity, not background correctness) — sane
     ballpark, not SOTA (this is a short smoke run on 40 volumes / a few
     thousand steps, not a real training budget).
  4. Resolution invariance: fit z independently from the SAME volume seen at
     two different grid resolutions, decode both on a shared reference grid,
     compare — the practical analogue of NOIR's epsilon-ReNO check.

Usage:
    PYTHONPATH=src python src/cfm/test_inr_backbone_smoke.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_env
from common.io import SPLIT_MAP, load_nifti_volume
from common.metrics import compute_nrmse, compute_ssim
from cfm.inr_backbone import (
    INRBackbone,
    INRBackboneConfig,
    ModulatedSIREN,
    fit_new_volume,
    make_coord_grid,
    meta_train_step,
)

VOLUME_SIZE = (96, 112, 96)
TARGET_SPACING = (2.0, 2.0, 2.0)
N_VOLUMES = 40
N_META_STEPS = 2000
BATCH_SIZE = 2
POINTS_PER_STEP = 16384
FG_MASK_THRESHOLD_01 = 0.05  # matches INRBackboneConfig.bg_threshold=-0.9 in [-1,1] space
_OPTS = {"modulate_scale": False, "lora_rank": 0, "direct": False,
         "steps": N_META_STEPS, "inner_steps": 10}


def _load_smoke_volumes(device: torch.device) -> torch.Tensor:
    env = load_env("local")
    data_root = Path(env["data_root"])
    src_dir = data_root / SPLIT_MAP["retro_train"] / "T1W" / "3T"
    files = sorted(src_dir.glob("*.nii.gz"))[:N_VOLUMES]
    if len(files) < N_VOLUMES:
        raise RuntimeError(f"Seulement {len(files)} volumes trouvés dans {src_dir} (besoin de {N_VOLUMES}).")
    vols = []
    for p in files:
        vol, _ = load_nifti_volume(
            p, target_spacing=TARGET_SPACING, volume_size=VOLUME_SIZE,
            normalize=True, lo_pct=0.5, hi_pct=99.5,
        )
        vols.append(torch.from_numpy(vol))
    return torch.stack(vols).unsqueeze(1).to(device)  # (N, 1, H, W, D)


def _make_cfg() -> INRBackboneConfig:
    """Config du smoke test, pilotée par les arguments CLI (voir __main__).

    `--modulate-scale` / `--lora-rank` testent l'enrichissement de la
    modulation ; `--direct` supprime le hypernetwork (gamma = z), nécessaire
    dès que modulation_dim est grand sinon hyper_hidden_dim redevient le goulot.
    """
    kw = dict(hidden_dim=256, num_hidden_layers=6,
              inner_steps_train=_OPTS["inner_steps"], inner_steps_eval=2 * _OPTS["inner_steps"],
              modulate_scale=_OPTS["modulate_scale"], lora_rank=_OPTS["lora_rank"])
    if _OPTS["direct"]:
        probe = ModulatedSIREN(hidden_dim=256, num_hidden_layers=6,
                               modulate_scale=_OPTS["modulate_scale"],
                               lora_rank=_OPTS["lora_rank"])
        kw.update(hyper_hidden_dim=0, latent_dim=probe.modulation_dim)
    else:
        kw.update(hyper_hidden_dim=512, latent_dim=4096)
    return INRBackboneConfig(**kw)


def test_meta_training_converges(
    volumes: torch.Tensor, device: torch.device, z_reg_weight: float = 0.0,
) -> INRBackbone:
    n_steps = _OPTS["steps"]
    print(f"\n[1/4] Meta-entraînement ({n_steps} pas, batch={BATCH_SIZE}, "
          f"{POINTS_PER_STEP} points/pas, sur {volumes.shape[0]} volumes, "
          f"z_reg_weight={z_reg_weight:.2e})...")
    cfg = _make_cfg()
    backbone = INRBackbone(cfg).to(device)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-4)
    coords_full = make_coord_grid(VOLUME_SIZE, device=device)

    # Same ramp shape as train_inr_backbone.py (0 -> z_reg_weight over the
    # first 10% of steps) — a short smoke run should still reflect the real
    # schedule, not apply the penalty full-strength from step 0.
    warmup_iters = max(1, int(n_steps * 0.1))

    n = volumes.shape[0]
    losses, jac_penalties = [], []
    t0 = time.time()
    for step in range(n_steps):
        idx = torch.randperm(n, device=device)[:BATCH_SIZE]
        batch = volumes[idx]
        cur_z_reg_weight = z_reg_weight * min(1.0, step / warmup_iters)
        loss, _grad_norm, jac_penalty = meta_train_step(
            backbone, batch, coords_full, optimizer, POINTS_PER_STEP, z_reg_weight=cur_z_reg_weight,
        )
        losses.append(loss)
        jac_penalties.append(jac_penalty)
        if (step + 1) % 200 == 0:
            recent = float(np.mean(losses[-200:]))
            recent_jac = float(np.mean(jac_penalties[-200:]))
            print(f"    [{step + 1:5d}/{n_steps}] loss={recent:.5f} jac_pen={recent_jac:.5f} "
                  f"t={time.time() - t0:.1f}s")

    first_200 = float(np.mean(losses[:200]))
    last_200 = float(np.mean(losses[-200:]))
    assert last_200 < first_200, f"Loss did not decrease: first200={first_200:.5f} last200={last_200:.5f}"
    print(f"✅ Loss décroît : {first_200:.5f} -> {last_200:.5f}")
    return backbone


def test_z_discriminates_between_volumes(backbone: INRBackbone, volumes: torch.Tensor, device: torch.device) -> None:
    print("\n[2/4] Discrimination de z (fit indépendant sur 2 volumes distincts)...")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    coords_full = make_coord_grid(VOLUME_SIZE, device=device)
    z0 = fit_new_volume(backbone, volumes[0:1], coords_full, num_steps=50, lr=1e-2)
    z1 = fit_new_volume(backbone, volumes[1:2], coords_full, num_steps=50, lr=1e-2)
    with torch.no_grad():
        coords_b = coords_full.unsqueeze(0)
        recon0 = backbone.decode(coords_b, z0)
        recon1 = backbone.decode(coords_b, z1)

    gt_rel_diff = ((volumes[0:1] - volumes[1:2]).reshape(-1).norm() / volumes[0:1].reshape(-1).norm()).item()
    rel_diff = ((recon0 - recon1).norm() / recon0.norm().clamp_min(1e-8)).item()
    print(f"    ||z0-z1||={((z0 - z1).norm().item()):.5f}  "
          f"diff relative reconstructions={rel_diff:.5f}  (GT diff relative={gt_rel_diff:.4f}, référence)")
    # Loose, smoke-scale bar: strictly above what a collapsed model produces
    # (empirically ~2e-5, invariant to training length — see module docstring).
    # Not a quality bar — that comes in Phase 3 against the production
    # baselines (nRMSE 0.4288 vectorized / 0.4741 UNet).
    assert rel_diff > 5e-5, (
        f"z ne différencie quasiment pas les volumes (rel_diff={rel_diff:.2e}) — "
        "signe du collapse observé avant le fix de perte pondérée foreground."
    )
    print("✅ z encode un contenu qui dépend du volume (pas de collapse).")

    for p in backbone.parameters():
        p.requires_grad_(True)


def test_reconstruction_quality(backbone: INRBackbone, volumes: torch.Tensor, device: torch.device) -> None:
    print("\n[3/4] Qualité de reconstruction (Algorithme 2, dense)...")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    coords_full = make_coord_grid(VOLUME_SIZE, device=device)
    vol = volumes[0:1]
    z = fit_new_volume(backbone, vol, coords_full, num_steps=50, lr=1e-2)
    with torch.no_grad():
        pred = backbone.decode(coords_full.unsqueeze(0), z).reshape(VOLUME_SIZE)

    pred_np = ((pred.clamp(-1, 1) + 1) / 2).cpu().numpy()
    gt_np = ((vol[0, 0].clamp(-1, 1) + 1) / 2).cpu().numpy()
    nrmse = compute_nrmse(pred_np, gt_np)
    ssim = compute_ssim(pred_np, gt_np)

    fg_mask = gt_np > FG_MASK_THRESHOLD_01
    nrmse_fg = compute_nrmse(pred_np, gt_np, mask=fg_mask)
    print(f"    Volume entier : nRMSE={nrmse:.4f}  SSIM={ssim:.4f}")
    print(f"    Foreground seul (masque > {FG_MASK_THRESHOLD_01}) : nRMSE={nrmse_fg:.4f}")
    assert nrmse_fg < 0.6, (
        f"nRMSE foreground trop élevé pour un fitting Algorithme 2 (mécanisme cassé ?): {nrmse_fg:.4f}"
    )
    print("✅ Reconstruction dans un ordre de grandeur sain (y compris hors fond).")

    for p in backbone.parameters():
        p.requires_grad_(True)


def test_resolution_invariance(backbone: INRBackbone, volumes: torch.Tensor, device: torch.device) -> None:
    print("\n[4/4] Invariance à la résolution (fit indépendant à 2 résolutions)...")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    vol_full = volumes[0:1]  # (1, 1, 96, 112, 96)
    half_size = tuple(s // 2 for s in VOLUME_SIZE)
    vol_half = F.interpolate(vol_full, size=half_size, mode="trilinear", align_corners=True)

    coords_full = make_coord_grid(VOLUME_SIZE, device=device)
    coords_half = make_coord_grid(half_size, device=device)

    z_full = fit_new_volume(backbone, vol_full, coords_full, num_steps=50, lr=1e-2)
    z_half = fit_new_volume(backbone, vol_half, coords_half, num_steps=50, lr=1e-2)

    with torch.no_grad():
        ref_coords = coords_full.unsqueeze(0)
        recon_from_full = backbone.decode(ref_coords, z_full)
        recon_from_half = backbone.decode(ref_coords, z_half)

    rel_err = (
        (recon_from_full - recon_from_half).norm() / recon_from_full.norm().clamp_min(1e-8)
    ).item()
    print(f"    Erreur relative L2 (reconstructions@96x112x96, fit@full vs fit@demi-résolution) : {rel_err:.4f}")
    assert rel_err < 0.5, f"Invariance à la résolution trop faible pour valider le principe : {rel_err:.4f}"
    print("✅ Les deux fits (résolutions différentes) convergent vers des reconstructions voisines.")

    for p in backbone.parameters():
        p.requires_grad_(True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modulate-scale", action="store_true",
                     help="FiLM complet : module aussi l'échelle (capacité x2)")
    ap.add_argument("--lora-rank", type=int, default=0,
                     help="Rang de la modulation bas-rang des poids (0 = désactivée)")
    ap.add_argument("--direct", action="store_true",
                     help="Modulation directe (gamma = z, pas de hypernetwork)")
    ap.add_argument("--steps", type=int, default=N_META_STEPS,
                     help="Nombre de pas de meta-entraînement")
    ap.add_argument("--inner-steps", type=int, default=10,
                     help="Pas de la boucle interne (fit de z). Plus z est grand, "
                          "plus il en faut a priori pour converger.")
    ap.add_argument("--z-reg-weight", type=float, default=0.0,
                     help="Poids de la pénalité de lissage sur z (Hutchinson VJP). 0.0 = désactivée.")
    ap.add_argument("--save-checkpoint", default=None,
                     help="Chemin où sauver le backbone entraîné (format compatible load_inr_backbone), "
                          "pour re-vérifier avec diagnose_inr_latent_smoothness.py.")
    args = ap.parse_args()
    _OPTS.update(modulate_scale=args.modulate_scale, lora_rank=args.lora_rank, direct=args.direct,
                 steps=args.steps, inner_steps=args.inner_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"SMOKE TEST — INR backbone (Phase 1) | device={device} | z_reg_weight={args.z_reg_weight:.2e}")
    print("=" * 70)

    try:
        volumes = _load_smoke_volumes(device)
        print(f"Volumes chargés : {tuple(volumes.shape)}")

        backbone = test_meta_training_converges(volumes, device, z_reg_weight=args.z_reg_weight)
        test_z_discriminates_between_volumes(backbone, volumes, device)
        test_reconstruction_quality(backbone, volumes, device)
        test_resolution_invariance(backbone, volumes, device)

        if args.save_checkpoint:
            out = Path(args.save_checkpoint)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": backbone.state_dict()}, out)
            print(f"\n💾 Checkpoint smoke sauvé : {out}")

        print("\n" + "=" * 70)
        print("✅ TOUS LES SMOKE TESTS SONT PASSÉS")
        print("=" * 70)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ÉCHEC : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
