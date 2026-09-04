#!/usr/bin/env python3
"""Rétro-remplissage de `backbone_cfg` dans le checkpoint INR de production.

EXÉCUTÉ UNE FOIS, le 2026-09-04, sur
`outputs/mmfm/inr_backbone/weights/model_final.pth` (entraîné le 2026-08-11,
donc antérieur au garde-fou D0). Conservé pour l'audit : il montre ce qui a été
inscrit et sur quelle base. Re-lancer est sans risque — l'assert en tête refuse
un checkpoint déjà rempli.

Usage:
    PYTHONPATH=src python src/cfm/backfill_inr_backbone_cfg.py

Règle : n'inscrire que ce qui est SOURCÉ. Les champs d'architecture sont
vérifiés contre les poids eux-mêmes ; les champs de procédure ne sont pas
récupérables et proviennent du `cfg_path` enregistré dans le checkpoint —
la provenance est écrite dans le checkpoint pour que ça reste lisible.
"""
import os, sys
from dataclasses import asdict
from pathlib import Path
import torch

sys.path.insert(0, "src")
from common.config import load_yaml_with_include, load_env, resolve_paths
from cfm.inr_backbone import INRBackbone
from cfm.train_inr_backbone import _backbone_config_from_cfg   # code PARTAGÉ

CKPT = Path("outputs/mmfm/inr_backbone/weights/model_final.pth")
SRC_CFG = "configs/mmfm/inr_backbone.yaml"

state = torch.load(CKPT, map_location="cpu", weights_only=False)   # sans mmap
assert "backbone_cfg" not in state, "déjà rétro-rempli — ne rien faire"
sd = state["model"]

cfg = _backbone_config_from_cfg(resolve_paths(load_yaml_with_include(SRC_CFG), load_env("local")))
print(f"config lue depuis {SRC_CFG} (= cfg_path du checkpoint : {state.get('cfg_path')})")

# --- 1. ce que les POIDS prouvent ---------------------------------------
derived = {
    "latent_dim":        sd["hypernet.net.0.weight"].shape[1],
    "hyper_hidden_dim":  sd["hypernet.net.0.weight"].shape[0],
    "hidden_dim":        sd["siren.first.linear.weight"].shape[0],
    "num_hidden_layers": sd["hypernet.net.2.weight"].shape[0] // sd["siren.first.linear.weight"].shape[0],
    "modulate_scale":    sd["hypernet.net.2.weight"].shape[0] == 2 * 6 * 256,
    "lora_rank":         0 if not any("lora" in k for k in sd) else -1,
}
print("\n--- champs VÉRIFIÉS contre les poids ---")
bad = []
for k, v in derived.items():
    cur = getattr(cfg, k)
    ok = (cur == v)
    print(f"  {k:20s} poids={v!r:10}  config={cur!r:10}  {'✅' if ok else '❌'}")
    if not ok:
        bad.append(k)
if bad:
    raise SystemExit(f"\n❌ ARRÊT : {bad} — la config ne décrit pas ces poids, "
                     "rien n'est écrit.")

# preuve définitive : la config instancie un modèle qui accepte ces poids
INRBackbone(cfg).load_state_dict(sd)   # strict=True par défaut
print("  → load_state_dict(strict=True) accepte la config. Architecture certaine.")

NOT_VERIFIABLE = ["omega_0", "omega_hidden", "inner_lr", "inner_steps_train",
                  "inner_steps_eval", "fg_weight", "bg_threshold"]
print(f"\n--- champs NON récupérables des poids (sourcés du cfg_path) ---")
for k in NOT_VERIFIABLE:
    print(f"  {k:20s} = {getattr(cfg, k)!r}")

# --- 2. écriture, avec sa provenance -------------------------------------
state["backbone_cfg"] = asdict(cfg)
state["backbone_cfg_provenance"] = {
    "backfilled_on": "2026-09-04",
    "source": f"{SRC_CFG} (le cfg_path enregistré dans ce checkpoint), "
              "inchangé depuis son unique commit 223e5f4",
    "verified_against_weights": sorted(derived) + ["(load_state_dict strict)"],
    "not_verifiable_from_weights": NOT_VERIFIABLE,
    "caveat": "Les champs de PROCÉDURE ne sont pas récupérables d'un checkpoint : "
              "ils sont repris du fichier de config pointé par cfg_path, dont les "
              "commentaires datés (<= 2026-08-10) précèdent l'entraînement "
              "(2026-08-10 16:29 -> 2026-08-11 05:54, 48287 s pour 50k iters). "
              "Corroboration : 0.98 s/iter, la valeur annoncée pour latent_dim=129024. "
              "Ce n'est PAS une mesure, c'est la meilleure source disponible.",
    "backup": "model_final.pth.bak_20260904 (md5 e68c642b8454b2175b33c8d62a0b99af)",
}

tmp = CKPT.with_suffix(".pth.tmp")
torch.save(state, tmp)
os.replace(tmp, CKPT)     # atomique
print(f"\n💾 écrit : {CKPT}")
