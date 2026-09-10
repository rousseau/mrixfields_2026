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
  3. Fit-and-reconstruct (Algorithm 2) on a training volume, scored by
     foreground-masked nRMSE (background correctness is free — after
     center-crop/pad most voxels are flat background every subject shares).
     The gate is RELATIVE: the fit must beat the best prediction that has no
     access to z, i.e. the LEAVE-ONE-OUT mean of the other volumes. An
     absolute threshold does not work here, and that is now measured rather
     than argued: a deliberately broken backbone (z=0) scores 0.4443 on the
     production model and 0.3942 here — both comfortably UNDER the old
     `nrmse_fg < 0.6`, which therefore accepted the broken mechanism as
     readily as the healthy one, for months.
  4. Resolution invariance: fit z independently from the SAME volume seen at
     two different grid resolutions, decode both on a shared reference grid,
     compare — the practical analogue of NOIR's epsilon-ReNO check.
  5. Negative control: decode at z=0 (no per-volume information at all) and
     require it to be markedly WORSE than the fit. This is the test OF test
     3 — without it, nothing establishes that gate 3 can separate a healthy
     mechanism from a collapsed one.

Regime: production (1mm, 192x224x192, 8 volumes), matching configs/mmfm/
inr.yaml — not the historical 2mm/96x112x96/40-volume regime. Point budgets
are subsampled (1M points) because a dense decode at 1mm is 8.25M points; the
mechanism is what is under test here, not production-grade fidelity.

Usage:
    PYTHONPATH=src python src/cfm/test_inr_backbone_smoke.py
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_env
from common.io import SPLIT_MAP, load_nifti_volume
from common.metrics import compute_nrmse
from cfm.inr_backbone import (
    INRBackbone,
    INRBackboneConfig,
    diff_inr_configs,
    ModulatedSIREN,
    fit_new_volume,
    make_coord_grid,
    meta_train_step,
)

# Régime de PRODUCTION (2026-09, Phase 0.2) : le smoke test tourne à 1 mm /
# 192×224×192 comme configs/mmfm/inr.yaml, pas à 2 mm / 96×112×96 comme avant.
# Le test est un test de MÉCANISME (z se différencie, la perte décroît, la
# reconstruction est saine), pas une porte de qualité de production — le seuil
# `test_reconstruction_quality` est volontairement large (ordre de grandeur).
# N_VOLUMES=8 (pas 40) parce qu'un stack de 40 volumes 1 mm ≈ 1.3 GB et la pile
# (ollama/desktop) consomme déjà ~44 GB sur ce poste ; 8 volumes suffisent à
# discriminer un backbone saine d'un backbone cassé et tiennent en mémoire.
VOLUME_SIZE = (192, 224, 192)
TARGET_SPACING = (1.0, 1.0, 1.0)
N_VOLUMES = 8
N_META_STEPS = 2000
BATCH_SIZE = 2
POINTS_PER_STEP = 16384
FG_MASK_THRESHOLD_01 = 0.05  # matches INRBackboneConfig.bg_threshold=-0.9 in [-1,1] space
_OPTS = {"modulate_scale": False, "lora_rank": 0, "direct": False,
         "steps": N_META_STEPS, "inner_steps": 10, "lora_inner_lr": None}


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
              modulate_scale=_OPTS["modulate_scale"], lora_rank=_OPTS["lora_rank"],
              lora_inner_lr=_OPTS["lora_inner_lr"])
    if _OPTS["direct"]:
        probe = ModulatedSIREN(hidden_dim=256, num_hidden_layers=6,
                               modulate_scale=_OPTS["modulate_scale"],
                               lora_rank=_OPTS["lora_rank"])
        kw.update(hyper_hidden_dim=0, latent_dim=probe.modulation_dim)
    else:
        kw.update(hyper_hidden_dim=512, latent_dim=4096)
    return INRBackboneConfig(**kw)


def _load_smoke_backbone(path: Path, device: torch.device) -> INRBackbone:
    """Recharge un backbone smoke déjà entraîné : rejoue les portes [2/5]-[5/5]
    sans repayer les ~18 min de méta-entraînement de [1/5].

    Applique la leçon D0 (voir arch_inr.load_inr_backbone) à ce checkpoint-ci :
    la config est comparée à celle du checkpoint quand il la porte, parce
    qu'une divergence de PROCÉDURE (inner_steps, fg_weight...) ne changerait
    aucune forme de poids et passerait donc load_state_dict en silence.
    Checkpoint sans `cfg` (antérieur) → comparaison ignorée, comme pour D0.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = _make_cfg()
    diffs = diff_inr_configs(cfg, state.get("cfg"))
    if diffs:
        lines = "\n".join(f"  {n} : demandé={cur!r}  vs  checkpoint={exp!r}"
                          for n, cur, exp in diffs)
        raise ValueError(
            f"Le backbone de {path} n'a pas été entraîné avec cette config.\n"
            f"{lines}\nRelancez sans --load-checkpoint, ou repassez les mêmes "
            f"options CLI (--modulate-scale / --lora-rank / --direct / "
            f"--inner-steps) qu'à la sauvegarde."
        )
    backbone = INRBackbone(cfg).to(device)
    backbone.load_state_dict(state["model"])
    print(f"\n[1/5] ⏭  Méta-entraînement sauté — backbone rechargé depuis {path}")
    return backbone


def test_meta_training_converges(
    volumes: torch.Tensor, device: torch.device, z_reg_weight: float = 0.0,
) -> INRBackbone:
    n_steps = _OPTS["steps"]
    print(f"\n[1/5] Meta-entraînement ({n_steps} pas, batch={BATCH_SIZE}, "
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
    # TIRAGE DES COORDONNEES — meme correctif que train_inr_backbone.py (2026-09-07).
    # Sans generateur explicite, `sample_points` en fabrique un dont la graine ne
    # depend que de (n_points_grille, num_points) : la boucle verrait alors les MEMES
    # coordonnees a chaque pas. C'est le defaut mesure sur le backbone de production
    # (les 50 000 pas ont vu 0.198 % de la grille, et les poids partages sont restes a
    # 1.01x de leur initialisation). Un smoke test qui tourne dans un autre regime que
    # la production ne valide pas la production.
    sample_gen = torch.Generator(device=device)
    sample_gen.manual_seed(20260906)
    t0 = time.time()
    for step in range(n_steps):
        idx = torch.randperm(n, device=device)[:BATCH_SIZE]
        batch = volumes[idx]
        cur_z_reg_weight = z_reg_weight * min(1.0, step / warmup_iters)
        loss, _grad_norm, jac_penalty = meta_train_step(
            backbone, batch, coords_full, optimizer, POINTS_PER_STEP, z_reg_weight=cur_z_reg_weight,
            generator=sample_gen,
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
    print("\n[2/5] Discrimination de z (fit indépendant sur 2 volumes distincts)...")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Sous-échantillonnage de supervision (1M points) : à 1 mm un fit
    # dense = 8.25M points, la graph d'activation ne tient pas en VRAM
    # sur ce poste (ollama/desktop consomme ~44 GB sur 128 GB). Le budget
    # de points est déjà sous-échantillonné partout dans le projet
    # (points_per_step=16384 dans train_inr_backbone.py, fit_points=1M
    # dans configs/mmfm/inr.yaml) — le smoke-test n'a pas à faire dense
    # pour vérifier le mécanisme.
    coords_all = make_coord_grid(VOLUME_SIZE, device=device)
    n_sub = min(1_000_000, coords_all.shape[0])
    idx = torch.randperm(coords_all.shape[0], device=device)[:n_sub]
    coords_sub = coords_all[idx]
    z0 = fit_new_volume(backbone, volumes[0:1], coords_all, num_steps=50, lr=1e-2, num_points=n_sub)
    z1 = fit_new_volume(backbone, volumes[1:2], coords_all, num_steps=50, lr=1e-2, num_points=n_sub)
    with torch.no_grad():
        coords_b = coords_sub.unsqueeze(0)
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


def _fg_nrmse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """nRMSE sur le foreground (cerveau) : la mesure discriminante.

    Après center_crop_or_pad + percentiles, ~90 % des voxels sont du fond
    quasi-invariant ≈ -1, que TOUTE prédiction (moyenne incluse) atteint à
    coup sûr — il faut donc exclure le fond pour séparer anatomie du reste.
    """
    pred_np = ((pred.clamp(-1, 1) + 1) / 2).cpu().numpy()
    gt_np = ((gt.clamp(-1, 1) + 1) / 2).cpu().numpy()
    m = gt_np > FG_MASK_THRESHOLD_01
    return float(compute_nrmse(pred_np, gt_np, mask=m))


def _resample_idx(coords_full: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    """Sous-échantillon déterministe de n indices sur la grille pleine."""
    gen = torch.Generator(coords_full.device)
    gen.manual_seed(seed)
    return torch.randperm(coords_full.shape[0], device=coords_full.device, generator=gen)[:n]


def test_reconstruction_quality(backbone: INRBackbone, volumes: torch.Tensor, device: torch.device) -> float:
    """[3/5] Le fit de z doit dominer l'« average brain blob » (moyenne LOO).

    C'est le seuil DISCRIMINANT demandé par le CHANGELOG. L'ancien seuil
    absolu (nrmse_fg < 0.6) laissait passer le cassé — mesuré le 2026-09-04 :
    un backbone à z=0 vaut 0.4443 (production) et 0.3942 (smoke), donc SOUS
    0.6, accepté. (L'ancienne justification invoquait le 0.6383 de la
    production : c'est le nRMSE Task 3 du FLOW de bout en bout, pas le
    nrmse_fg de reconstruction du BACKBONE, qui vaut 0.2430 — deux grandeurs
    incommensurables.) Ici on compare le fit à la MEILLEURE prédiction « sans z »
    (la moyenne des autres volumes, le régime exact du collapse que ce test a
    été écrit pour attraper). La moyenne est LEAVE-ONE-OUT : voir le
    commentaire ci-dessous, une baseline qui contient la cible inverse le
    verdict. Retourne le nrmse_fg du fit (utilisé par [5/5]).
    """
    print("\n[3/5] Qualité de reconstruction (fit vs average brain blob)...")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Sous-échantillon (1M points) : un decode DENSE à 1mm (8.25M points)
    # ~33 GB d'activation ne tient pas sur ce poste (ollama/desktop ~44 GB).
    # Le budget de points est déjà sous-échantillonné partout ailleurs
    # (points_per_step=16384, fit_points=1M) — le test de mécanisme n'a pas
    # à faire dense ; la qualité absolue de production est mesurée ailleurs.
    coords_full = make_coord_grid(VOLUME_SIZE, device=device)
    n = min(1_000_000, coords_full.shape[0])
    idx = _resample_idx(coords_full, n)
    coords_sub = coords_full[idx]

    vol = volumes[0:1]
    gt = vol.reshape(1, -1)[0, idx]  # voxels aux mêmes points query
    z = fit_new_volume(backbone, vol, coords_full, num_steps=50, lr=1e-2, num_points=n)
    with torch.no_grad():
        pred_fit = backbone.decode(coords_sub.unsqueeze(0), z).reshape(-1)
    nrmse_fit = _fg_nrmse(pred_fit, gt)

    # La meilleure prédiction « sans z » : la moyenne des AUTRES volumes.
    # LEAVE-ONE-OUT obligatoire. Une moyenne qui contient la cible n'est pas
    # une prédiction, c'est un oracle partiel — et la fuite n'est pas
    # cosmétique : mesurée le 2026-09-03 sur ce jeu, elle vaut +0.0388 de
    # nRMSE_fg (0.2716 cible incluse contre 0.3104 sans, moyenné sur les 8
    # volumes) et INVERSE le verdict de la porte (fit 0.3213 : échec contre
    # 0.3091 avec la cible, succès contre 0.3541 sans). À N=40 (ancien régime
    # 2mm) la cible pesait 1/40 de sa propre référence et la fuite passait
    # inaperçue ; à N=8 (le stack 1mm ne tient pas en mémoire au-delà) elle
    # pèse 1/8 et décide seule du résultat. Voir CHANGELOG.md, 2026-09-03 (nuit).
    mean_vol = volumes[1:].mean(dim=0, keepdim=True)  # volumes[0:1] = la cible
    mean_sub = mean_vol.reshape(1, -1)[0, idx]
    nrmse_mean = _fg_nrmse(mean_sub, gt)

    print(f"    nRMSE_fg avec z (fit)            : {nrmse_fit:.4f}")
    print(f"    nRMSE_fg sans z (moy. LOO, n={volumes.shape[0] - 1}): {nrmse_mean:.4f}")
    assert nrmse_fit < nrmse_mean, (
        f"le fit de z ne bat PAS la moyenne leave-one-out des autres volumes "
        f"(fit={nrmse_fit:.4f} >= moyenne={nrmse_mean:.4f}) — le mécanisme "
        "de fitting est cassé (collapse sur l'average brain blob ; voir "
        "module docstring et inr_backbone.py::_weighted_mse)."
    )
    print(f"✅ le fit de z domine l'average brain blob "
          f"({nrmse_mean:.4f} → {nrmse_fit:.4f}).")

    for p in backbone.parameters():
        p.requires_grad_(True)
    return nrmse_fit


def test_negative_gate(backbone: INRBackbone, volumes: torch.Tensor, device: torch.device, nrmse_fit: float) -> None:
    """[5/5] Contrôle négatif : PROUVE que [3/5] est discriminant.

    Simule un backbone cassé (z=0, modulation nulle → « average brain blob »
    sans aucune information par volume). Le nRMSE_fg à z=0 DOIT être nettement
    PIRE que celui du fit sain fourni par [3/5]. Si les deux sont à peu près
    égaux (ratio < 1.05), le test [3/5] ne peut PAS séparer un mécanisme sain
    d'un mécanisme cassé — c'est exactement le défaut que le CHANGELOG cite
    comme « non corrigé » (threshold qui laisse passer le cassé).
    Ratio = nrmse_z0 / nrmse_fit > 1.05.
    """
    print("\n[5/5] Contrôle négatif : prouver que [3/5] est discriminant...")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    coords_full = make_coord_grid(VOLUME_SIZE, device=device)
    n = min(1_000_000, coords_full.shape[0])
    idx = _resample_idx(coords_full, n)
    coords_sub = coords_full[idx]
    vol = volumes[0:1]
    gt = vol.reshape(1, -1)[0, idx]

    # z=0 : modulation nulle → le réseau produit le même « cerveau moyen »
    # quel que soit le volume (le régime exact du collapse).
    z0 = torch.zeros(1, backbone.cfg.latent_dim, device=device, dtype=torch.float32)
    with torch.no_grad():
        pred_z0 = backbone.decode(coords_sub.unsqueeze(0), z0).reshape(-1)
    nrmse_z0 = _fg_nrmse(pred_z0, gt)

    ratio = nrmse_z0 / max(nrmse_fit, 1e-9)
    print(f"    nRMSE_fg z=0 (cassé)  : {nrmse_z0:.4f}")
    print(f"    nRMSE_fg fit (sain)  : {nrmse_fit:.4f}")
    print(f"    ratio (z0/fit)       : {ratio:.2f}")
    assert ratio >= 1.05, (
        f"le test de reconstruction n'est PAS discriminant : le z=0 (cassé) "
        f"donne nRMSE_fg={nrmse_z0:.4f}, pas nettement pire que le fit sain "
        f"({nrmse_fit:.4f}) — le seuil laisse donc passer le cassé (défaut "
        "noté dans CHANGELOG.md). Revoir le seuil pour forcer ratio > 1.05."
    )
    print("✅ contrôle négatif : un mécanisme cassé (z=0) EST bien détecté, "
          f"le test est discriminant (ratio = {ratio:.2f}).")

    for p in backbone.parameters():
        p.requires_grad_(True)


def test_resolution_invariance(backbone: INRBackbone, volumes: torch.Tensor, device: torch.device) -> None:
    print("\n[4/5] Invariance à la résolution (fit indépendant à 2 résolutions)...")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Sous-échantillon (1M points) pour le fit ET le decode : à 1 mm un fit
    # dense 8.25M × 50 pas ne tient pas en VRAM. Le point du test est la
    # propriété epsilon-ReNO de NOIR (z fitté à 2 résolutions → reconstructions
    # voisines), qui n'exige PAS un fit dense : un budget de points commun et
    # déterministe préserve la comparaison full-vs-demi-résolution.
    n = 1_000_000
    vol_full = volumes[0:1]
    half_size = tuple(s // 2 for s in VOLUME_SIZE)
    vol_half = F.interpolate(vol_full, size=half_size, mode="trilinear", align_corners=True)

    coords_full = make_coord_grid(VOLUME_SIZE, device=device)
    n_f = min(n, coords_full.shape[0])
    idx_f = _resample_idx(coords_full, n_f)
    coords_full_sub = coords_full[idx_f]      # points de comparaison (decode)
    coords_half = make_coord_grid(half_size, device=device)
    n_h = min(n, coords_half.shape[0])

    # Le sous-échantillonnage du FIT passe par `num_points`, pas par une grille
    # déjà réduite : fit_new_volume doit échantillonner coordonnées ET valeurs
    # ENSEMBLE (sample_points). Lui donner une grille de 1M points avec un
    # volume de 8.25M voxels casse l'appariement — RuntimeError de forme dans
    # _weighted_mse (rencontré le 2026-09-04). C'est aussi ce que fait [3/5].
    z_full = fit_new_volume(backbone, vol_full, coords_full, num_steps=50, lr=1e-2, num_points=n_f)
    z_half = fit_new_volume(backbone, vol_half, coords_half, num_steps=50, lr=1e-2, num_points=n_h)

    # decode aux mêmes points query (la grille pleine 1mm, sous-échantillonnée)
    with torch.no_grad():
        ref_coords = coords_full_sub.unsqueeze(0)
        recon_from_full = backbone.decode(ref_coords, z_full)
        recon_from_half = backbone.decode(ref_coords, z_half)
    gt_ref = vol_full.reshape(1, -1)[0, idx_f]

    rel_err = (
        (recon_from_full - recon_from_half).norm() / recon_from_full.norm().clamp_min(1e-8)
    ).item()
    nrmse_full = _fg_nrmse(recon_from_full.reshape(-1), gt_ref)
    nrmse_half = _fg_nrmse(recon_from_half.reshape(-1), gt_ref)
    print(f"    Erreur relative L2 (fit@full vs fit@demi-résolution) : {rel_err:.4f}")
    print(f"    nRMSE_fg recon@full={nrmse_full:.4f}  recon@half={nrmse_half:.4f}")
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
    ap.add_argument("--lora-inner-lr", type=float, default=None,
                     help="Pas de descente distinct pour la portion LoRA de z, requis quand "
                          "--direct et --lora-rank>0 (sinon la LoRA diverge ou reste morte).")
    ap.add_argument("--load-checkpoint", default=None,
                     help="Recharge un backbone smoke déjà entraîné et saute [1/5] : "
                          "rejoue les portes sans repayer le méta-entraînement.")
    ap.add_argument("--save-checkpoint", default=None,
                     help="Chemin où sauver le backbone entraîné (format compatible load_inr_backbone), "
                          "pour re-vérifier avec diagnose_inr_latent_smoothness.py.")
    args = ap.parse_args()
    _OPTS.update(modulate_scale=args.modulate_scale, lora_rank=args.lora_rank, direct=args.direct,
                 steps=args.steps, inner_steps=args.inner_steps, lora_inner_lr=args.lora_inner_lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"SMOKE TEST — INR backbone (Phase 1) | device={device} | z_reg_weight={args.z_reg_weight:.2e}")
    print("=" * 70)

    try:
        volumes = _load_smoke_volumes(device)
        print(f"Volumes chargés : {tuple(volumes.shape)}")

        if args.load_checkpoint:
            backbone = _load_smoke_backbone(Path(args.load_checkpoint), device)
        else:
            backbone = test_meta_training_converges(volumes, device, z_reg_weight=args.z_reg_weight)

        # Sauvegarde AVANT les portes, pas après : le méta-entraînement coûte
        # ~18 min à 1mm, et c'est justement quand une porte ÉCHOUE qu'on veut
        # rejouer les portes sur le même backbone sans le repayer.
        if args.save_checkpoint and not args.load_checkpoint:
            out = Path(args.save_checkpoint)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": backbone.state_dict(),
                        "cfg": asdict(backbone.cfg)}, out)
            print(f"\n💾 Checkpoint smoke sauvé : {out}")

        test_z_discriminates_between_volumes(backbone, volumes, device)
        nrmse_fit = test_reconstruction_quality(backbone, volumes, device)
        test_resolution_invariance(backbone, volumes, device)
        test_negative_gate(backbone, volumes, device, nrmse_fit=nrmse_fit)

        print("\n" + "=" * 70)
        print("✅ TOUS LES SMOKE TESTS SONT PASSÉS")
        print("=" * 70)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ÉCHEC : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
