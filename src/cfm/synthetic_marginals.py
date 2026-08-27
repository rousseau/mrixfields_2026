#!/usr/bin/env python3
"""Probleme multi-marginal SYNTHETIQUE a reponse connue.

Motif. Rien dans ce projet ne verifiait le flow contre une verite terrain. Les
tests existants ne controlaient que des formes de tenseurs, et le modele est
reste litteralement aveugle au temps pendant des mois sans qu'aucun garde-fou
ne bronche. Trois hypotheses ont ete refutees d'affilee (loss L1, interpolation
cubique de sortie, cibles contradictoires) parce qu'elles visaient en aval d'un
defaut d'entree qu'aucun instrument ne pouvait isoler.

Ce module fournit cet instrument : un probleme a 5 marginales dont la reponse
EXACTE est calculable, qui reproduit les trois contraintes reelles du projet —

  1. marginales NON COLINEAIRES (la deviation a la corde est un parametre,
     calee sur les 0.37-0.90 mesures sur nos donnees le 2026-08-27) ;
  2. donnees NON APPARIEES : un echantillon n'existe qu'a UNE marginale, comme
     nos 1056 sujets dont aucun n'existe a deux champs ;
  3. un facteur « sujet » a preserver le long du transport, comme l'anatomie.

Il est branche sur la VRAIE boucle d'entrainement (`mmfm3d_synthetic` dans
mmfm_core._resolve_arch_module) via cfm/arch_synthetic.py : le harnais exerce
donc `_sample_step_plan`, `_compute_flow`, le sampler OT-CFM, la loss, l'EMA et
`euler_integrate` tels quels, sans copie ni reimplementation.

Un echec ici se lit en minutes au lieu de 25 000 iterations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class SyntheticSpec:
    """Parametres du processus generateur.

    bend : deviation du point median a la corde, en fraction de la longueur de
        corde. 0.6 est au milieu de ce qui a ete mesure sur nos marginales
        reelles (0.37 a 0.90). bend=0 rend les marginales colineaires — utile
        comme temoin : un modele aveugle au temps reussit ce cas-la.
    subject_scale : amplitude du facteur individuel a PRESERVER pendant le
        transport (l'analogue de l'anatomie du sujet).
    noise : bruit d'observation, irreductible ; borne inferieure du nRMSE.
    """
    dim: int = 64
    n_fields: int = 5
    n_contrasts: int = 3
    bend: float = 0.6
    subject_scale: float = 1.0
    noise: float = 0.05
    seed: int = 0


class SyntheticMarginals:
    """Trajectoire courbe P_c(t) par contraste, plus un facteur sujet.

        P_c(t) = A_c + t * B_c + sin(pi * t) * C_c

    C_c est construit ORTHOGONAL a B_c, si bien que la deviation du point
    median a la corde A_c -> A_c + B_c vaut exactement ||C_c|| / ||B_c|| = bend.
    La courbure est donc posee, pas subie : on sait ce que le modele doit
    reproduire.

    Un echantillon du champ k, contraste c, sujet s :

        z = P_c(t_k) + s * D_c + noise

    Le transport correct de t=0 vers t est donc : conserver s, suivre P_c.
    """

    def __init__(self, spec: SyntheticSpec = SyntheticSpec()):
        self.spec = spec
        g = torch.Generator().manual_seed(spec.seed)
        d, C = spec.dim, spec.n_contrasts

        self.A = torch.randn(C, d, generator=g)
        B = torch.randn(C, d, generator=g)
        self.B = B / B.norm(dim=1, keepdim=True)          # corde unitaire

        # C_c orthogonal a B_c (Gram-Schmidt), de norme `bend`.
        raw = torch.randn(C, d, generator=g)
        proj = (raw * self.B).sum(1, keepdim=True) * self.B
        Corth = raw - proj
        self.C = spec.bend * Corth / Corth.norm(dim=1, keepdim=True).clamp_min(1e-12)

        # Direction « sujet », orthogonale au plan (B, C) pour que le facteur
        # individuel soit identifiable independamment de la position sur la courbe.
        raw = torch.randn(C, d, generator=g)
        for basis in (self.B, self.C / spec.bend if spec.bend > 0 else self.C):
            nb = basis / basis.norm(dim=1, keepdim=True).clamp_min(1e-12)
            raw = raw - (raw * nb).sum(1, keepdim=True) * nb
        self.D = spec.subject_scale * raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-12)

    # -- geometrie ---------------------------------------------------------

    def field_time(self, k: int) -> float:
        return float(k) / (self.spec.n_fields - 1)

    def curve(self, c: int, t: float | torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.float32).reshape(-1, 1)
        return self.A[c] + t * self.B[c] + torch.sin(math.pi * t) * self.C[c]

    def true_bend(self, c: int = 0) -> list[float]:
        """Deviation a la corde aux 3 ancres intermediaires — ce que le modele
        doit reproduire. Sert de reference au test de courbure."""
        pts = torch.cat([self.curve(c, self.field_time(k))
                         for k in range(self.spec.n_fields)], dim=0)
        p0, pl = pts[0], pts[-1]
        chord = pl - p0
        L = chord.norm()
        n = self.spec.n_fields - 1
        return [float(((pts[k] - (p0 + (k / n) * chord)).norm() / L).item())
                for k in range(1, n)]

    # -- echantillonnage ---------------------------------------------------

    def sample(self, c: int, k: int, n: int, generator: torch.Generator
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """n echantillons du champ k, contraste c. Chaque echantillon a SON
        propre facteur sujet : aucun n'existe a deux champs (non apparie)."""
        s = torch.randn(n, 1, generator=generator)
        base = self.curve(c, self.field_time(k))
        eps = torch.randn(n, self.spec.dim, generator=generator) * self.spec.noise
        return base + s * self.D[c] + eps, s

    def subject_factor(self, z: torch.Tensor, c: int, k: int) -> torch.Tensor:
        """Retrouve s par projection sur D_c. D_c etant orthogonal au plan de la
        courbe, la projection est exacte au bruit pres."""
        d = z - self.curve(c, self.field_time(k))
        Dc = self.D[c]
        return (d @ Dc) / (Dc @ Dc).clamp_min(1e-12)

    def ideal_target(self, z_src: torch.Tensor, c: int, k_src: int, k_tgt: int
                     ) -> torch.Tensor:
        """Ou le transport PARFAIT doit envoyer z_src : meme sujet, position du
        champ cible sur la courbe. C'est la verite terrain par echantillon —
        bien plus informatif qu'une distance entre distributions."""
        s = self.subject_factor(z_src, c, k_src).reshape(-1, 1)
        return self.curve(c, self.field_time(k_tgt)) + s * self.D[c]


class SyntheticLatentDataset(torch.utils.data.Dataset):
    """Expose le probleme synthetique sous le contrat attendu par
    mmfm_core._build_class_loaders : `.samples` porte class_idx, et
    __getitem__ rend (vecteur, mod_idx, field_idx, class_idx).

    Les echantillons sont tires UNE fois et figes : un jeu d'entrainement fini,
    comme les 1939 volumes reels, et non un flux infini qui rendrait le
    probleme trop facile.
    """

    def __init__(self, spec: SyntheticSpec, n_per_class: int = 256, seed: int = 1):
        self.spec = spec
        self.dgp = SyntheticMarginals(spec)
        g = torch.Generator().manual_seed(seed)
        self.vectors: list[torch.Tensor] = []
        self.samples: list[dict] = []
        for c in range(spec.n_contrasts):
            for k in range(spec.n_fields):
                z, _ = self.dgp.sample(c, k, n_per_class, g)
                for i in range(n_per_class):
                    self.vectors.append(z[i])
                    self.samples.append({
                        "mod_idx": c, "field_idx": k,
                        "class_idx": c * spec.n_fields + k,
                    })
        self.flat_dim = spec.dim

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, i: int):
        s = self.samples[i]
        return self.vectors[i], s["mod_idx"], s["field_idx"], s["class_idx"]


# ---------------------------------------------------------------------------
# Evaluation : le modele retrouve-t-il la carte connue ?
# ---------------------------------------------------------------------------

def evaluate(model_fn, dgp: SyntheticMarginals, device, n: int = 256,
             n_steps: int = 100, seed: int = 99) -> dict:
    """Integre 0 -> 1 depuis des echantillons FRAIS du champ 0 et compare aux
    cibles ideales a chaque ancre.

    Renvoie, par contraste :
      - nrmse[k] : erreur relative au transport parfait vers le champ k
      - bend     : courbure de la trajectoire integree
      - true_bend, ratio : la courbure exigee, et la fraction atteinte
    """
    g = torch.Generator().manual_seed(seed)
    nf = dgp.spec.n_fields
    out: dict = {}
    for c in range(dgp.spec.n_contrasts):
        z0, _ = dgp.sample(c, 0, n, g)
        z0 = z0.to(device)
        y = torch.full((n,), c, dtype=torch.long, device=device)
        z, dt = z0.clone(), 1.0 / n_steps
        snaps = {0: z0.clone()}
        every = n_steps // (nf - 1)
        with torch.no_grad():
            for i in range(n_steps):
                t = torch.full((n,), i * dt, device=device)
                z = z + dt * model_fn(z, z0, t, y).float()
                if (i + 1) % every == 0:
                    snaps[(i + 1) // every] = z.clone()

        nrmse = []
        for k in range(nf):
            ideal = dgp.ideal_target(z0.cpu(), c, 0, k)
            pred = snaps[k].cpu()
            nrmse.append(float(((pred - ideal).norm(dim=1)
                                / ideal.norm(dim=1).clamp_min(1e-12)).mean().item()))

        pts = torch.stack([snaps[k].cpu().mean(0) for k in range(nf)])
        p0, pl = pts[0], pts[-1]
        chord = pl - p0
        L = chord.norm().clamp_min(1e-12)
        bend = [float(((pts[k] - (p0 + (k / (nf - 1)) * chord)).norm() / L).item())
                for k in range(1, nf - 1)]
        tb = dgp.true_bend(c)
        out[c] = {
            "nrmse": nrmse,
            "bend": bend,
            "true_bend": tb,
            "bend_ratio": [b / t if t > 0 else float("nan") for b, t in zip(bend, tb)],
        }
    return out
