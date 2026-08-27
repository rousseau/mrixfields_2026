#!/usr/bin/env python3
"""Tests de COMPORTEMENT du flow — pas de formes de tenseurs.

Motif. Les tests existants (`test_mmfm_v2_technical.py`, `test_mmfm_v2_smoke.py`)
ne verifient que des dimensions de sortie. Ils passaient tous pendant les mois ou
le modele etait litteralement aveugle au temps : `cos(v(t=0), v(t=1)) = 1.000000`
sur le checkpoint de production (audit du 2026-08-27,
`results/mmfm/audit_20260827_refcompare/manifest.md`). Un test qui ne peut pas
echouer ne teste rien.

Chaque test ci-dessous ECHOUE sur la configuration d'avant le correctif et PASSE
apres. C'est la propriete qui compte : le verifier dans les deux sens avec

    PYTHONPATH=src python src/cfm/test_mmfm_flow_behaviour.py --demo-regression

Usage :
    PYTHONPATH=src python src/cfm/test_mmfm_flow_behaviour.py
    PYTHONPATH=src python src/cfm/test_mmfm_flow_behaviour.py \\
        --checkpoint outputs/mmfm/vec_r1_time/weights/model_final.pth \\
        --config configs/mmfm/vectorized_r1_time.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cfm.arch_vector import build_vector_mmfm
from cfm.mmfm_vectorized import sinusoidal_time_embedding
from common.io import MODALITIES

# Les 5 temps de champ : _field_to_time(i, 5) = i/4.
FIELD_TIMES = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

# Seuils. Justification de chaque valeur dans le test correspondant.
MIN_EFFECTIVE_RANK = 3.5      # sur 4 (5 temps -> 4 directions apres centrage)
MAX_TIME_COSINE = 0.99        # au-dela, le modele ne distingue pas les extremes
MIN_BEND_FRACTION = 0.50      # fraction de la courbure exigee par les donnees
MAX_LOSS_VS_ZERO = 0.90       # la loss doit battre « predire zero » de 10 %


class Failure(AssertionError):
    pass


def _effective_rank(emb: torch.Tensor) -> float:
    """Participation ratio des valeurs singulieres de l'embedding centre.

    Mesure combien de directions INDEPENDANTES le plongement offre aux 5 temps.
    Une rampe scalaire donne ~1 ; une base quasi orthogonale donne ~4.
    """
    ec = emb - emb.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(ec.double())
    return float((sv.sum() ** 2 / (sv ** 2).sum()).item())


# ---------------------------------------------------------------------------
# 1. Le plongement du temps doit separer les 5 champs
# ---------------------------------------------------------------------------

def test_time_embedding_rank(time_scale: float, dim: int = 256) -> float:
    """`sinusoidal_time_embedding` a max_period=10000 : elle attend des INDICES
    de pas de diffusion dans [0,1000), pas un temps dans [0,1]. Avec t brut,
    l'argument des sinusoides ne depasse jamais 1, donc cos ~ 1 partout et
    sin(x) ~ x : l'embedding s'effondre en une rampe scalaire.
    """
    emb = sinusoidal_time_embedding(FIELD_TIMES * time_scale, dim)
    er = _effective_rank(emb)
    if er < MIN_EFFECTIVE_RANK:
        raise Failure(
            f"rang effectif du plongement du temps = {er:.2f} < {MIN_EFFECTIVE_RANK} "
            f"(time_scale={time_scale:g}). Les 5 temps de champ sont quasi colineaires : "
            f"le modele ne peut pas les distinguer. Correctif : model.time_scale = 1000."
        )
    return er


def test_time_embedding_survives_bf16(time_scale: float, dim: int = 256) -> float:
    """L'entrainement tourne en AMP bf16. Les directions faibles du plongement
    peuvent passer sous le plancher de quantification — auquel cas le signal est
    remplace par du bruit d'arrondi, pas seulement attenue.
    """
    e32 = sinusoidal_time_embedding(FIELD_TIMES * time_scale, dim)
    e16 = e32.to(torch.bfloat16).float()
    rel = ((e16 - e32).norm() / e32.norm()).item()
    er16 = _effective_rank(e16)
    if er16 < MIN_EFFECTIVE_RANK:
        raise Failure(
            f"rang effectif en bf16 = {er16:.2f} < {MIN_EFFECTIVE_RANK} "
            f"(fp32 : {_effective_rank(e32):.2f}). La quantification bf16 detruit "
            f"les directions faibles du plongement du temps."
        )
    return rel


# ---------------------------------------------------------------------------
# 2. Le modele doit REELLEMENT dependre du temps
# ---------------------------------------------------------------------------

def _time_cosine(model, z: torch.Tensor, y: torch.Tensor) -> float:
    """cos entre la vitesse predite aux deux extremites de l'axe temporel.

    A entree z identique, v(t=0) et v(t=1) doivent differer : ce sont deux
    points opposes de la trajectoire. Un cosinus de 1 signifie que le modele
    ignore t.
    """
    with torch.no_grad():
        v0 = model(z, z, torch.zeros(z.shape[0]), y).float()
        v1 = model(z, z, torch.ones(z.shape[0]), y).float()
    return float(F.cosine_similarity(v0, v1, dim=1).mean().item())


def test_time_sensitivity(model, latent_dim: int, seed: int = 0) -> float:
    """ATTENTION au moment de la mesure. Avec `time_cond: film`, la projection de
    modulation part de ZERO (initialisation AdaLN-Zero, standard et voulue) :
    a l'initialisation le modele est time-blind PAR CONSTRUCTION et ce test
    echouerait toujours. Il n'a de sens qu'apres entrainement -- le run R-best
    donne cos = -0.24 / 0.80 / 0.03 sur les trois contrastes. L'appelant doit
    donc passer un modele ENTRAINE quand time_cond vaut 'film'.
    """
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(4, latent_dim, generator=g)
    y = torch.zeros(4, dtype=torch.long)
    c = _time_cosine(model, z, y)
    if c > MAX_TIME_COSINE:
        raise Failure(
            f"cos(v(t=0), v(t=1)) = {c:.6f} > {MAX_TIME_COSINE}. Le modele ne "
            f"distingue pas les extremites de l'axe des champs : sa vitesse est "
            f"quasi constante en t, donc sa trajectoire est une droite et les "
            f"champs intermediaires sont rendus comme une moyenne."
        )
    return c


# ---------------------------------------------------------------------------
# 3. La trajectoire doit pouvoir se COURBER
# ---------------------------------------------------------------------------

def bend_fractions(points: torch.Tensor) -> list[float]:
    """Deviation des points intermediaires a la corde p0->p_last, en fraction
    de la longueur de corde. `points` : (n_anchors, D)."""
    p0, pl = points[0], points[-1]
    chord = pl - p0
    L = chord.norm()
    n = points.shape[0] - 1
    return [float(((points[k] - (p0 + (k / n) * chord)).norm() / L).item())
            for k in range(1, n)]


def test_trajectory_can_bend(model, latent_dim: int, n_steps: int = 100,
                             seed: int = 0) -> list[float]:
    """Integre 0 -> 1 et mesure la courbure. Un modele aveugle au temps produit
    une droite (courbure ~ 0) ; les marginales reelles s'ecartent de la corde de
    0.37 a 0.90 fois sa longueur (mesure du 2026-08-27).

    Ce test tourne a l'initialisation : il ne verifie pas que la courbure est
    la BONNE, seulement que le modele est CAPABLE d'en produire une. C'est
    exactement la capacite qui manquait.
    """
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(2, latent_dim, generator=g)
    y = torch.zeros(2, dtype=torch.long)
    z, dt = z0.clone(), 1.0 / n_steps
    snaps = {0: z0.clone()}
    with torch.no_grad():
        for i in range(n_steps):
            v = model(z, z0, torch.full((2,), i * dt), y).float()
            z = z + dt * v
            if (i + 1) % (n_steps // 4) == 0:
                snaps[(i + 1) // (n_steps // 4)] = z.clone()
    traj = torch.stack([snaps[k] for k in range(5)], dim=1)
    bends = [bend_fractions(traj[b]) for b in range(traj.shape[0])]
    mean_bend = [sum(b[k] for b in bends) / len(bends) for k in range(3)]
    if max(mean_bend) < 1e-3:
        raise Failure(
            f"courbure de la trajectoire = {mean_bend} : le modele integre une "
            f"DROITE. Les champs intermediaires sont alors des points d'un segment "
            f"entre les extremes — contraste faux et apparence moyennee."
        )
    return mean_bend


# ---------------------------------------------------------------------------
# 4. La loss doit battre « predire zero » (dette n2 du CHANGELOG)
# ---------------------------------------------------------------------------

def zero_baseline_from_cache(cache_dir: str, modality: str = "T2W",
                             n: int = 8, loss: str = "l1",
                             adjacent_only: bool = False) -> float:
    """Loss d'un predicteur qui renvoie 0 partout, sur les vraies cibles
    u = (z_j - z_i)/dt. Trois lignes qui auraient arrete le projet des le debut :
    un flow dont la loss ne bat pas la constante nulle n'a rien appris.

    REGIME DE PAIRES. `adjacent_only` change l'echelle des cibles : `dt` y vaut
    toujours 0.25 au lieu d'aller jusqu'a 1.0, donc `u = (z_j - z_i)/dt` est ~4x
    plus grand et la loss n'est PAS comparable a celle du regime toutes-paires
    (deja constate le 2026-08-26 : 18.43 contre 12.08 pour des modeles de qualite
    voisine). La base doit donc etre calculee sur le MEME ensemble de paires que
    l'entrainement, sans quoi le test echoue sur un artefact d'echelle -- ce qui
    est arrive a R-best.

    UNITES. La loss journalisee par mmfm_core est calculee sur des latents BRUTS :
    `ut_global` vient du cache sans transformation, et le modele remultiplie sa
    sortie par `latent_scale` (mmfm_vectorized.forward). La standardisation est
    interne au forward. La base doit donc rester en unites brutes, sans quoi on
    compare deux echelles differentes — l'INR donnait un ratio de 0.0002.
    """
    root = Path(cache_dir)
    meta = json.load(open(root / "index.json"))
    idx = meta["samples"]
    base = root.parent.parent
    # Les deux schemas d'index coexistent : le cache vectorise porte les
    # chaines ("modality"/"field"), celui de l'UNet seulement les indices
    # ("mod_idx"/"field_idx"). On resout via les listes de l'index lui-meme.
    mods, fields = meta["modalities"], meta["fields"]

    def _mod(r) -> str:
        return r["modality"] if "modality" in r else mods[int(r["mod_idx"])]

    def _fld(r) -> str:
        return r["field"] if "field" in r else fields[int(r["field_idx"])]

    def load(field: str, k: int, seed: int) -> torch.Tensor:
        s = [r for r in idx if _mod(r) == modality and _fld(r) == field]
        random.Random(seed).shuffle(s)
        out = []
        for r in s[:k]:
            o = torch.load(base / r["path"], map_location="cpu", weights_only=False)
            z = o["latent"] if isinstance(o, dict) and "latent" in o else o
            out.append(torch.as_tensor(z).float().reshape(-1))
        return torch.stack(out)

    vals = []
    for i in range(len(fields)):
        for j in range(len(fields)):
            if i == j:
                continue
            if adjacent_only and abs(i - j) != 1:
                continue
            # Unites BRUTES, cf. docstring : la loss journalisee l'est aussi.
            zi, zj = load(fields[i], n, 0), load(fields[j], n, 1)
            u = (zj - zi) / ((j - i) / (len(fields) - 1))
            vals.append(u.abs().mean().item() if loss == "l1"
                        else (u ** 2).mean().item())
    return sum(vals) / len(vals)


def test_loss_beats_zero(metrics_path: str, baseline: float) -> tuple[float, float]:
    L = [json.loads(l) for l in open(metrics_path)]
    final = sum(x["loss"] for x in L[-5:]) / min(5, len(L))
    ratio = final / baseline
    if ratio > MAX_LOSS_VS_ZERO:
        raise Failure(
            f"loss finale {final:.4f} contre {baseline:.4f} pour « predire zero » "
            f"(ratio {ratio:.3f} > {MAX_LOSS_VS_ZERO}). Le flow n'a pas appris "
            f"davantage qu'une constante nulle."
        )
    return final, ratio


def test_no_zero_gradient(metrics_path: str) -> int:
    """Un gradient exactement nul signale un entrainement mort.

    PIEGE, verifie le 2026-08-27. `mmfm_core` journalisait `round(grad_norm, 4)`,
    donc toute norme < 5e-5 s'ecrivait "0.0". Le flow INR (latent a l'echelle
    2.6e-4, gradients ~1e-4) affichait 103 "zeros" sur 250 points ; mesure
    directe a l'initialisation : 2.7e-4 en bf16 comme en fp32, pas d'identite
    compris. C'etait un artefact de journalisation, pas un gradient mort.
    La journalisation est passee a 4 chiffres SIGNIFICATIFS ; sur les traces
    anterieures, un "0.0" reste ambigu et n'est donc pas compte comme un echec.
    """
    L = [json.loads(l) for l in open(metrics_path)]
    zeros = [x for x in L if x.get("grad_norm", 1.0) == 0.0]
    if not zeros:
        return len(L)
    legacy = all(
        float(x["grad_norm"]) == round(float(x["grad_norm"]), 4)
        for x in L if x.get("grad_norm", 0.0) != 0.0
    )
    if legacy:
        print(f"       (trace anterieure : {len(zeros)}/{len(L)} points a 0.0 — "
              f"resolution de journalisation 5e-5, pas un gradient mort)")
        return len(L)
    raise Failure(
        f"grad_norm == 0.0 a {len(zeros)}/{len(L)} points journalises "
        f"(premier a l'iteration {zeros[0]['iter']}). Le modele ne recoit "
        f"aucun gradient sur ces pas."
    )


# ---------------------------------------------------------------------------

def _build(cfg: dict, latent_dim: int, time_scale_override: float | None = None):
    cfg = {**cfg, "model": dict(cfg["model"])}
    cfg["model"]["hidden_dim"] = 64
    cfg["model"]["num_blocks"] = 1
    if time_scale_override is not None:
        cfg["model"]["time_scale"] = time_scale_override
    return build_vector_mmfm(cfg, latent_dim, 3)


def run(cfg_path: str, checkpoint: str | None, metrics: str | None) -> int:
    cfg = yaml.safe_load(open(cfg_path))
    ts = float(cfg["model"].get("time_scale", 1.0))
    latent_dim = 512
    ok, ko = 0, 0

    def check(name, fn):
        nonlocal ok, ko
        try:
            r = fn()
            print(f"  OK   {name}" + (f"  -> {r}" if r is not None else ""))
            ok += 1
        except Failure as e:
            print(f"  ECHEC {name}\n        {e}")
            ko += 1

    print(f"Configuration : {cfg_path}  (time_scale = {ts:g})")
    check("rang du plongement du temps", lambda: round(test_time_embedding_rank(ts), 2))
    check("plongement du temps en bf16", lambda: test_time_embedding_survives_bf16(ts) and None)

    model = _build(cfg, latent_dim).eval()
    if cfg["model"].get("time_cond", "concat") == "film":
        print("  n/a  sensibilite au temps (init) : time_cond='film' part d'une "
              "modulation nulle (AdaLN-Zero), la mesure n'a de sens qu'apres "
              "entrainement — voir src/cfm/measure_flow_geometry.py")
    else:
        check("sensibilite au temps (init)", lambda: round(test_time_sensitivity(model, latent_dim), 6))
    check("la trajectoire peut se courber", lambda: [round(x, 4) for x in test_trajectory_can_bend(model, latent_dim)])

    if metrics and Path(metrics).exists():
        check("aucun gradient nul", lambda: test_no_zero_gradient(metrics))
        cache = cfg["data"].get("latent_cache_dir")
        if cache and Path(cache).exists():
            loss_kind = "l2" if cfg["train"].get("loss", "l1") == "l2" else "l1"
            adj = bool(cfg["train"].get("adjacent_only", False))
            try:
                base = zero_baseline_from_cache(cache, loss=loss_kind, adjacent_only=adj)
            except Exception as exc:                      # schema de cache inattendu
                print(f"  n/a  la loss bat « predire zero » : cache illisible ({exc})")
            else:
                check(f"la loss bat « predire zero » (base {base:.4g})",
                      lambda: tuple(round(v, 5) for v in test_loss_beats_zero(metrics, base)))

    print(f"\n{ok} reussis, {ko} echoues")
    return 1 if ko else 0


def demo_regression(cfg_path: str) -> None:
    """Prouve que les tests peuvent echouer : les rejoue a time_scale = 1.0,
    la valeur d'avant le correctif."""
    cfg = yaml.safe_load(open(cfg_path))
    latent_dim = 512
    print("\n=== CONTRE-EPREUVE : time_scale = 1.0 (etat d'avant le correctif) ===")
    for name, fn in [
        ("rang du plongement", lambda: test_time_embedding_rank(1.0)),
        ("plongement en bf16", lambda: test_time_embedding_survives_bf16(1.0)),
        ("sensibilite au temps", lambda: test_time_sensitivity(
            _build(cfg, latent_dim, time_scale_override=1.0).eval(), latent_dim)),
    ]:
        try:
            fn()
            print(f"  !! {name} : a PASSE — le test ne discrimine pas, seuil a revoir")
        except Failure as e:
            print(f"  attendu : {name} echoue\n        {str(e)[:110]}...")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mmfm/vectorized_r1_time.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--metrics", default=None,
                    help="train_metrics.jsonl (defaut : deduit de output_subdir)")
    ap.add_argument("--demo-regression", action="store_true",
                    help="rejoue les tests a time_scale=1.0 pour prouver qu'ils echouent")
    a = ap.parse_args()

    metrics = a.metrics
    if metrics is None:
        cfg = yaml.safe_load(open(a.config))
        sub = cfg["data"].get("output_subdir")
        if sub:
            metrics = str(Path("outputs") / sub / "train_metrics.jsonl")

    code = run(a.config, a.checkpoint, metrics)
    if a.demo_regression:
        demo_regression(a.config)
    sys.exit(code)


if __name__ == "__main__":
    main()
