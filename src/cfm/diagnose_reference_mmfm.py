#!/usr/bin/env python3
"""Le code de REFERENCE Genentech/MMFM, non modifie, s'effondre-t-il pareil ?

CONTEXTE. Trois audits ont deja compare ce projet a des references (torchcfm,
external/MMFM, facebookresearch/flow_matching, NVIDIA/NV-Generate-CTMR), mais en
REIMPLEMENTANT leurs idees dans mmfm_core.py (couplage OT chaine + spline "a la
Genentech", FiLM "a la NVIDIA"...). Le 2026-09-04, deux vrais bugs de code ont ete
trouves et corriges de cette facon (`mmfm_core.py` calculait `z + Dt*c`, une
constante) ; le 2026-09-05, le mecanisme reentraine etait repare
(cos(v(0),v(1)) 1.000000 -> -0.592) mais **le score n'a pas suivi**. Le 2026-09-15,
deux pistes de plus (guidance, conflation temps/champ), toujours par
reimplementation, sont closes NEGATIF/NON ETABLI. Aucun de ces audits n'a fait
tourner le CODE d'un depot de reference tel quel, bout en bout, sur les vraies
donnees du projet -- c'est le test qui manque pour trancher, une bonne fois,
entre "bug residuel dans mmfm_core.py/arch_vector.py" et "limite reelle des
donnees" (0 sujet apparie sur 1056, mesure le 2026-09-02/03).

CE QUE CE SCRIPT FAIT. Fait tourner `external/MMFM` (vendorise, NON MODIFIE) --
`MultiMarginalFlowMatcher.sample_location_and_conditional_flow`
(multi_marginal_fm.py) et `VectorFieldModel` (models.py) -- directement sur les
memes latents reels que la production. Seul le couplage OT chaine est reutilise
depuis `mmfm_core.build_chained_ot_trajectories` (deja valide sur les vraies
donnees : diagnose_coupling_real.py, 180/180) plutot que reimplemente depuis
`external/MMFM/src/mmfm/data.py::couple_samples_no_int` -- la question posee
porte sur le mecanisme de flow + l'architecture, pas sur le couplage, deja
tranche deux fois.

TROIS ETAPES :
  synthetic : porte rapide (minutes) sur le harnais a reponse connue
              (SyntheticMarginals) -- confirme que l'adaptateur est cable avant
              de payer le calcul reel.
  real      : entraine sur le cache reel de production.
  diagnose  : sur un checkpoint (synthetic ou real), calcule cos(v(t=0),v(t=1))
              et la part du deplacement propre au sujet -- la meme mesure que
              `measure_flow_geometry.py`/`diagnose_flow_translation.py`,
              comparable aux chiffres deja connus : production 3.5 %,
              R-best repare 8.25 %, VRAI deplacement 71.7 %
              (`diagnose_true_displacement.py`).

Usage :
    PYTHONPATH=src python src/cfm/diagnose_reference_mmfm.py synthetic --iters 3000
    PYTHONPATH=src python src/cfm/diagnose_reference_mmfm.py real --iters 4000
    PYTHONPATH=src python src/cfm/diagnose_reference_mmfm.py diagnose \\
        --checkpoint outputs/mmfm/reference_genentech/real_step004000.pth
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC.parent / "external" / "MMFM" / "src"))

from cfm.mmfm_core import (  # noqa: E402
    build_chained_ot_trajectories,
    _flat_class,
    _field_to_time,
    euler_integrate,
)
from cfm.synthetic_marginals import (  # noqa: E402
    SyntheticSpec,
    SyntheticLatentDataset,
    SyntheticMarginals,
    evaluate as synthetic_evaluate,
)
from cfm.diagnose_flow_contraction import dispersion, load_cached  # noqa: E402

from mmfm.multi_marginal_fm import MultiMarginalFlowMatcher  # noqa: E402
from mmfm.models import VectorFieldModel  # noqa: E402

FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
MODALITIES = ["T1W", "T2W", "T2FLAIR"]
REAL_CACHE = Path("outputs/mmfm/latent_cache/vectorized/medvae_finetune_c4d1e200/retro_train")
REAL_PAIRED_CACHE = Path("outputs/mmfm/latent_cache/vectorized/medvae_finetune_c4d1e200/pro_train")
OUT_ROOT = Path("outputs/mmfm/reference_genentech")

# Chiffres deja etablis, pour comparaison directe (voir CHANGELOG.md) --
# aucun n'est recalcule ici.
KNOWN_OWN_SHARE = {
    "production (translation)": 0.0348,
    "R-best (mecanisme repare, 2026-09-05)": 0.0825,
    "VRAI deplacement (diagnose_true_displacement.py)": 0.7173,
}


# ---------------------------------------------------------------------------
# Donnees
# ---------------------------------------------------------------------------

class CachedLatentDataset(torch.utils.data.Dataset):
    """Meme contrat que les datasets de mmfm_core (`.samples[i]["class_idx"]`),
    charge depuis le cache precalcule -- meme motif que diagnose_coupling_real.py."""

    def __init__(self, cache_dir: Path, modalities: List[str], fields: List[str]):
        idx = json.load(open(cache_dir / "index.json"))["samples"]
        base = cache_dir.parent.parent
        self.samples: List[dict] = []
        self._paths: List[Path] = []
        for r in idx:
            if r["modality"] not in modalities or r["field"] not in fields:
                continue
            mi, fi = modalities.index(r["modality"]), fields.index(r["field"])
            self.samples.append({"class_idx": _flat_class(mi, fi, len(fields))})
            self._paths.append(base / r["path"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        o = torch.load(self._paths[i], map_location="cpu", weights_only=False)
        z = o["latent"] if isinstance(o, dict) and "latent" in o else o
        return torch.as_tensor(z).float().reshape(-1), None, None, self.samples[i]["class_idx"]


def _build_trajectory_tensor(ds, contrast_fields, usable_contrasts, n_fields,
                              verbose=True) -> Dict[int, torch.Tensor]:
    """`build_chained_ot_trajectories` renvoie des INDICES dans `ds` ; on les
    resout ici en un tenseur (n_traj, K, D) par contraste, format attendu par
    `MultiMarginalFlowMatcher.sample_location_and_conditional_flow`."""
    traj_idx = build_chained_ot_trajectories(ds, contrast_fields, usable_contrasts,
                                              n_fields, verbose=verbose)
    out: Dict[int, torch.Tensor] = {}
    for c, idx in traj_idx.items():
        cols = [torch.stack([torch.as_tensor(ds[i][0]) for i in idx[:, k]])
                for k in range(idx.shape[1])]
        out[c] = torch.stack(cols, dim=1)  # (n_traj, K, D)
    return out


# ---------------------------------------------------------------------------
# Entrainement -- boucle copiee de
# external/MMFM/experiments/synthetic_data/train_mmfm.py:214-238 (perte MSE,
# Adam), adaptee seulement pour lire depuis un tenseur de trajectoires deja
# couplees au lieu de dgp_waves_data.
# ---------------------------------------------------------------------------

def build_reference_model(data_dim: int, n_classes: int, hidden_dim: int,
                           device: torch.device) -> VectorFieldModel:
    return VectorFieldModel(
        data_dim=data_dim,
        x_latent_dim=hidden_dim,
        time_embed_dim=256,
        cond_embed_dim=128,
        conditional_model=True,
        embedding_type="free",
        n_classes=n_classes,
        label_list=list(range(1, n_classes + 1)),
        num_out_layers=3,
        activation="SELU",
    ).to(device)


def make_model_fn(model: VectorFieldModel):
    """Adapte VectorFieldModel (entree = cat[x, classe, t]) a la convention
    `model_fn(z, z_src, t, y) -> v` de `mmfm_core.euler_integrate` et
    `synthetic_marginals.evaluate`. `z_src` est ignore : la reference ne
    conditionne pas sur un point d'ancrage separe, seulement sur (z, t, classe)."""

    def model_fn(z: torch.Tensor, z_src: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cond = (y.float() + 1.0).unsqueeze(1)
        xin = torch.cat([z, cond, t.reshape(-1, 1)], dim=1)
        return model(xin)

    return model_fn


def train(xs_by_contrast: Dict[int, torch.Tensor], n_classes: int, data_dim: int,
          device: torch.device, *, iters: int, batch_size: int, hidden_dim: int,
          lr: float, checkpoint_every: int, out_dir: Path, tag: str,
          print_every: int = 50, grad_clip: float = 1.0) -> VectorFieldModel:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_reference_model(data_dim, n_classes, hidden_dim, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    FM = MultiMarginalFlowMatcher(sigma=0.0, interpolation="cubic")

    usable = list(xs_by_contrast.keys())
    K = next(iter(xs_by_contrast.values())).shape[1]
    t_anchor_row = torch.tensor([_field_to_time(k, K) for k in range(K)], dtype=torch.float32)

    losses: List[float] = []
    t0 = time.time()
    for step in range(1, iters + 1):
        c = random.choice(usable)
        Z = xs_by_contrast[c]
        idx = torch.randint(0, Z.shape[0], (batch_size,))
        xs = Z[idx].to(device)
        timepoints = t_anchor_row.unsqueeze(0).expand(batch_size, -1).to(device)
        cond = torch.full((batch_size, 1), 0.0, device=device)
        cond.fill_(float(c + 1))

        t, xt, ut, _, _ = FM.sample_location_and_conditional_flow(xs=xs, timepoints=timepoints)
        xt_in = torch.cat([xt.squeeze(1), cond, t.reshape(-1, 1).to(device)], dim=1)
        vt = model(xt_in)
        loss = F.mse_loss(vt, ut.squeeze(1).to(device))

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())

        if step % print_every == 0 or step == 1:
            dt = time.time() - t0
            recent = losses[-print_every:]
            print(f"[{tag}] step {step}/{iters}  loss={np.mean(recent):.5f}  "
                  f"({dt/step:.3f} s/step, {dt/60:.1f} min ecoulees, "
                  f"ETA {dt/step*(iters-step)/60:.1f} min)", flush=True)
        if step % checkpoint_every == 0 or step == iters:
            ckpt_path = out_dir / f"{tag}_step{step:06d}.pth"
            torch.save({"step": step, "model": model.state_dict(), "data_dim": data_dim,
                        "hidden_dim": hidden_dim, "n_classes": n_classes, "tag": tag},
                       ckpt_path)
            print(f"[{tag}] checkpoint -> {ckpt_path}", flush=True)

    return model


# ---------------------------------------------------------------------------
# Stage: synthetic -- porte rapide, reponse connue
# ---------------------------------------------------------------------------

def stage_synthetic(a: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = SyntheticSpec(dim=64, n_fields=5, n_contrasts=3, bend=0.6,
                          subject_scale=1.0, noise=0.05, seed=0)
    ds = SyntheticLatentDataset(spec, n_per_class=256, seed=1)
    contrast_fields = {c: list(range(spec.n_fields)) for c in range(spec.n_contrasts)}
    usable = list(range(spec.n_contrasts))

    print(f"[synthetic] couplage OT chaine ({spec.n_contrasts} contrastes x "
          f"{spec.n_fields} champs, dim={spec.dim})...")
    xs_by_contrast = _build_trajectory_tensor(ds, contrast_fields, usable, spec.n_fields)

    model = train(xs_by_contrast, spec.n_contrasts, spec.dim, device,
                  iters=a.iters, batch_size=a.batch_size, hidden_dim=a.hidden_dim,
                  lr=a.lr, checkpoint_every=max(a.iters, 1), out_dir=OUT_ROOT / "synthetic",
                  tag="synthetic", print_every=a.print_every)

    model.eval()
    model_fn = make_model_fn(model)
    dgp = SyntheticMarginals(spec)
    with torch.no_grad():
        res = synthetic_evaluate(model_fn, dgp, device, n=256, n_steps=100)

    print("\n=== Porte synthetique -- reponse connue ===")
    all_nrmse = []
    for c, r in res.items():
        all_nrmse.extend(r["nrmse"])
        print(f"  contraste {c} : nRMSE par champ = "
              f"{['%.4f' % v for v in r['nrmse']]} | courbure/vraie = "
              f"{['%.3f' % v for v in r['bend_ratio']]}")
    mean_nrmse = float(np.mean(all_nrmse))
    print(f"\n  nRMSE moyen : {mean_nrmse:.4f} (plancher de bruit ~0.05 ; "
          f"un conditionnement FAUX donne ~0.086, cf. H1 2026-09-15)")
    if mean_nrmse < 0.07:
        print("  => L'adaptateur est correctement cable : le code de reference "
              "resout le probleme a reponse connue. On peut payer le calcul reel.")
    else:
        print("  => NE PAS lancer 'real' : l'adaptateur a un defaut de cablage "
              "(format des donnees, appel a sample_location_and_conditional_flow, "
              "ou boucle d'entrainement) -- a corriger avant de payer les donnees reelles.")


# ---------------------------------------------------------------------------
# Stage: real -- vrais latents, vrai budget de calcul
# ---------------------------------------------------------------------------

def stage_real(a: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[real] chargement du cache {REAL_CACHE} ...")
    ds = CachedLatentDataset(REAL_CACHE, MODALITIES, FIELDS)
    print(f"[real] {len(ds)} latents indexes.")

    contrast_fields = {m: list(range(len(FIELDS))) for m in range(len(MODALITIES))}
    usable = list(range(len(MODALITIES)))

    print("[real] couplage OT chaine (identique a mmfm_core.py, deja valide "
          "sur les vraies donnees)...")
    xs_by_contrast = _build_trajectory_tensor(ds, contrast_fields, usable, len(FIELDS))
    for c, Z in xs_by_contrast.items():
        print(f"  contraste {MODALITIES[c]} : {Z.shape[0]} trajectoires, "
              f"dim={Z.shape[2]}")
    data_dim = next(iter(xs_by_contrast.values())).shape[2]

    train(xs_by_contrast, len(MODALITIES), data_dim, device,
          iters=a.iters, batch_size=a.batch_size, hidden_dim=a.hidden_dim,
          lr=a.lr, checkpoint_every=a.checkpoint_every, out_dir=OUT_ROOT / "real",
          tag="real", print_every=a.print_every)


# ---------------------------------------------------------------------------
# Stage: diagnose -- meme mesure que measure_flow_geometry.py /
# diagnose_flow_translation.py, sur un checkpoint de reference.
# ---------------------------------------------------------------------------

def stage_diagnose(a: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = build_reference_model(ck["data_dim"], ck["n_classes"], ck["hidden_dim"], device)
    model.load_state_dict(ck["model"])
    model.eval()
    model_fn = make_model_fn(model)

    print(f"=== Diagnostic geometrique -- {a.checkpoint} (step {ck.get('step')}) ===\n")

    base = REAL_CACHE.parent.parent
    idx = json.load(open(REAL_CACHE / "index.json"))["samples"]

    # cos(v(t=0), v(t=1)) : le sondage qui a revele "z + Dt*c" le 2026-09-04.
    for mi, mod in enumerate(MODALITIES):
        N = 4
        z0 = load_cached(idx, base, mod, "0.1T", N, seed=0).to(device)
        y = torch.full((N,), mi, dtype=torch.long, device=device)
        with torch.no_grad():
            v0 = model_fn(z0, z0, torch.zeros(N, device=device), y).float()
            v1 = model_fn(z0, z0, torch.ones(N, device=device), y).float()
        cos01 = F.cosine_similarity(v0, v1, dim=1).mean().item()
        print(f"  {mod:8s} cos(v(t=0),v(t=1)) = {cos01:.6f}  "
              f"(production 1.000000 avant correctif, -0.592 apres ; 1.0 = translation pure)")

    # Part du deplacement propre au sujet -- meme mesure que
    # diagnose_flow_translation.py, avec le model_fn de reference.
    print(f"\n  {'contraste':10s}{'cible':7s}{'part propre au sujet':>22s}")
    owns: List[float] = []
    for mi, mod in enumerate(MODALITIES):
        z_src = load_cached(idx, base, mod, a.src_field, a.n, seed=0).to(device)
        y = torch.full((z_src.shape[0],), mi, dtype=torch.long, device=device)
        t_src = _field_to_time(FIELDS.index(a.src_field), len(FIELDS))
        for f in FIELDS:
            if f == a.src_field:
                continue
            t_tgt = _field_to_time(FIELDS.index(f), len(FIELDS))
            with torch.no_grad():
                z_pred = euler_integrate(model_fn, z_src, y, t_src, t_tgt,
                                          a.n_steps, device, use_amp=False).cpu()
            D = z_pred - z_src.cpu()
            Dm = D.mean(0, keepdim=True)
            own = float(((D - Dm).norm(dim=1).mean() / Dm.norm().clamp_min(1e-12)).item())
            owns.append(own)
            print(f"  {mod:10s}{f:7s}{own:22.4f}")

    mean_own = float(np.mean(owns))
    print(f"\n  part du deplacement propre au sujet (reference, ce checkpoint) : "
          f"{mean_own:.4f} (min {min(owns):.4f}, max {max(owns):.4f})")
    print("\n  Pour comparaison (deja mesure, non recalcule ici) :")
    for label, v in KNOWN_OWN_SHARE.items():
        print(f"    {label:50s} {v:.4f}")

    print()
    if mean_own <= 0.15:
        print("  => Le code de REFERENCE, non modifie, collapse PAREIL (loin des "
              "71.7 % reels). Confirmation independante : la cause est les "
              "donnees (0 sujet apparie), pas un bug residuel de mmfm_core.py/"
              "arch_vector.py. A journaliser et clore.")
    else:
        print("  => Le code de REFERENCE capture significativement plus de "
              "composante propre au sujet que mmfm_core.py. La piste code/"
              "architecture est ROUVERTE : identifier lequel des choix de "
              "VectorFieldModel/MultiMarginalFlowMatcher (embedding du temps, "
              "conditionnement, interpolation) explique l'ecart, puis lancer "
              "scripts/run_task3_eval.sh pour un chiffre Task 3 officiel.")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"checkpoint": a.checkpoint, "step": ck.get("step"),
                   "own_share_mean": mean_own, "owns": owns,
                   "known": KNOWN_OWN_SHARE}, open(a.json_out, "w"), indent=2)
        print(f"\n  ecrit -> {a.json_out}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    common = dict(iters=3000, batch_size=64, hidden_dim=512, lr=1e-3, print_every=50)

    p_syn = sub.add_parser("synthetic", help="porte rapide, harnais a reponse connue")
    p_syn.add_argument("--iters", type=int, default=common["iters"])
    p_syn.add_argument("--batch-size", type=int, default=common["batch_size"])
    p_syn.add_argument("--hidden-dim", type=int, default=common["hidden_dim"])
    p_syn.add_argument("--lr", type=float, default=common["lr"])
    p_syn.add_argument("--print-every", type=int, default=common["print_every"])
    p_syn.set_defaults(func=stage_synthetic)

    p_real = sub.add_parser("real", help="entrainement sur les vrais latents")
    p_real.add_argument("--iters", type=int, default=4000)
    p_real.add_argument("--batch-size", type=int, default=32,
                        help="reduit vs. production : le spline scipy non modifie "
                             "de la reference coute ~1.2 s a batch 64 en dim 129024")
    p_real.add_argument("--hidden-dim", type=int, default=512)
    p_real.add_argument("--lr", type=float, default=1e-3)
    p_real.add_argument("--checkpoint-every", type=int, default=500)
    p_real.add_argument("--print-every", type=int, default=20)
    p_real.set_defaults(func=stage_real)

    p_diag = sub.add_parser("diagnose", help="mesure geometrique sur un checkpoint")
    p_diag.add_argument("--checkpoint", required=True)
    p_diag.add_argument("--n", type=int, default=24)
    p_diag.add_argument("--n-steps", type=int, default=20)
    p_diag.add_argument("--src-field", default="0.1T")
    p_diag.add_argument("--json-out", default=None)
    p_diag.set_defaults(func=stage_diagnose)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
