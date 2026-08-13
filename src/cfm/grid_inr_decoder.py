#!/usr/bin/env python3
"""Décodeur implicite conditionné par GRILLE LATENTE (style LIIF, activations SIREN).

Motivation — ce que les mesures et la littérature imposent
-----------------------------------------------------------
L'INR à latent GLOBAL de ce projet plafonne : le SIREN modulé par décalage ne
reçoit que `num_couches x hidden_dim` = 1536 nombres par volume, quantité
indépendante de `latent_dim` ET de la résolution. Vérifié expérimentalement :
porter latent_dim de 4096 à 129024 (x32) a légèrement DÉGRADÉ Task 3
(0.6223 -> 0.6383), et enrichir la modulation (scale + bas-rang, capacité x60)
n'a rien apporté au smoke test. Voir results/mmfm/comparison_20260807_1mm/.

La littérature dit la même chose : Kim & Fridovich-Keil, NeurIPS 2025, « Grids
Often Outperform Implicit Neural Representations at Compressing Dense Signals »
(arXiv:2506.11139) — à nombre de paramètres égal, une grille régularisée bat
tout INR sur les signaux DENSES ; les INR ne gagnent que sur des signaux
binaires (contours). Une IRM cérébrale est un signal dense à large bande.

Ce module prend donc l'autre voie, celle des décodeurs implicites conditionnés
LOCALEMENT (LIIF, Chen et al. CVPR 2021 ; DDMI arXiv:2401.12517 ; « Continuous
3-D Latent Diffusion for Medical Generation and Reconstruction »
arXiv:2607.16491, qui utilise exactement un décodeur LIIF conditionné par
coordonnées pour CT/IRM afin d'éviter le décodage par sous-volumes recouvrants
— précisément le problème que `models/tiled_vae.py` doit contourner ici).

Ce qu'on garde, ce qu'on abandonne
----------------------------------
GARDÉ : activations sinusoïdales + initialisation de Sitzmann (l'« esprit
SIREN »), et surtout le décodage SANS GRILLE — on peut interroger n'importe
quelle coordonnée, donc décoder à 0.5mm natif depuis un latent 1mm. C'est le
seul argument sérieux pour réintroduire un INR dans ce pipeline, puisque
l'évaluateur ré-interpole de toute façon vers 0.5mm.
ABANDONNÉ : le latent global et la boucle interne d'ajustement de `z` par
descente de gradient (Algorithme 2 de NOIR). Le latent vient désormais de
l'encodeur MedVAE gelé — même latent que les architectures vectorisée/UNet,
donc le modèle de flow reste strictement inchangé et la comparaison reste
équitable.

Architecture
------------
    latent MedVAE (B, C_lat, h, w, d)   [C_lat = 1, h,w,d = 48,56,48 @1mm]
      -> tronc conv 3D léger            -> (B, F, h, w, d)   features locales
      -> grid_sample aux coordonnées    -> (B, N, F)         feature par point
      -> [coord (3) | feature (F)]      -> SIREN             -> (B, N, 1)

La capacité par volume n'est plus bornée par la modulation : elle vaut
h*w*d*F (ex. 48*56*48*64 = 8.3M), et croît avec la résolution du latent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cfm.inr_backbone import SineLayer, _weighted_mse

__all__ = ["GridINRConfig", "GridConditionedINR", "sample_local_features"]


def sample_local_features(feat: Tensor, coords: Tensor, align_corners: bool = True) -> Tensor:
    """Interpole la grille de features aux coordonnées demandées.

    feat   : (B, F, H, W, D) — grille de features (axes dans NOTRE convention).
    coords : (B, N, 3) dans [-1, 1], composantes ordonnées (i, j, k) pour les
             axes (H, W, D) — c'est la convention de `make_coord_grid`.
    -> (B, N, F)

    ATTENTION à l'ordre des axes : `F.grid_sample` attend, pour une entrée 5D
    (N, C, D_in, H_in, W_in), une grille dont la DERNIÈRE dimension est
    (x, y, z) = (index W_in, index H_in, index D_in) — soit l'ORDRE INVERSE de
    nos axes. On retourne donc `coords` sur son dernier axe. Une erreur ici est
    silencieuse (le réseau apprend une anatomie transposée) : voir le test
    d'alignement dans `__main__`.
    """
    b, n, _ = coords.shape
    grid = coords.flip(-1).view(b, n, 1, 1, 3)
    out = F.grid_sample(feat, grid, mode="bilinear", padding_mode="border",
                        align_corners=align_corners)          # (B, F, N, 1, 1)
    return out.squeeze(-1).squeeze(-1).transpose(1, 2)        # (B, N, F)


class _ResBlock3d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.c2(self.act(self.c1(x)))


@dataclass
class GridINRConfig:
    latent_channels: int = 1       # canaux du latent MedVAE
    feat_dim: int = 64             # canaux de la grille de features locales
    n_res_blocks: int = 2          # profondeur du tronc conv
    hidden_dim: int = 256          # largeur du SIREN
    num_hidden_layers: int = 5     # profondeur du SIREN
    omega_0: float = 30.0
    omega_hidden: float = 30.0
    fg_weight: float = 5.0
    bg_threshold: float = -0.9


class GridConditionedINR(nn.Module):
    """Décodeur : (grille latente, coordonnées) -> intensités."""

    def __init__(self, cfg: GridINRConfig):
        super().__init__()
        self.cfg = cfg
        f = cfg.feat_dim
        self.trunk = nn.Sequential(
            nn.Conv3d(cfg.latent_channels, f, 3, padding=1),
            nn.SiLU(),
            *[_ResBlock3d(f) for _ in range(cfg.n_res_blocks)],
        )
        # Le SIREN prend [coordonnée absolue (3) | feature locale (F)].
        # La coordonnée absolue est conservée pour que le réseau puisse encore
        # modéliser une variation globale (position dans le cerveau), la feature
        # locale apportant le détail — c'est la combinaison LIIF.
        self.first = SineLayer(3 + f, cfg.hidden_dim, omega_0=cfg.omega_0, is_first=True)
        self.hidden = nn.ModuleList([
            SineLayer(cfg.hidden_dim, cfg.hidden_dim, omega_0=cfg.omega_hidden, is_first=False)
            for _ in range(cfg.num_hidden_layers - 1)
        ])
        self.final = nn.Linear(cfg.hidden_dim, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / cfg.hidden_dim) / cfg.omega_hidden
            self.final.weight.uniform_(-bound, bound)
            self.final.bias.zero_()

    # -- API ----------------------------------------------------------------

    def encode_features(self, z_grid: Tensor) -> Tensor:
        """(B, C_lat, h, w, d) -> (B, F, h, w, d). À calculer UNE fois par
        volume, puis réutiliser sur tous les blocs de coordonnées."""
        return self.trunk(z_grid)

    def query(self, feat: Tensor, coords: Tensor) -> Tensor:
        """feat: (B, F, h, w, d), coords: (B, N, 3) -> (B, N, 1)."""
        local = sample_local_features(feat, coords)            # (B, N, F)
        h = self.first(torch.cat([coords, local], dim=-1))
        for layer in self.hidden:
            h = layer(h)
        return self.final(h)

    def forward(self, z_grid: Tensor, coords: Tensor) -> Tensor:
        return self.query(self.encode_features(z_grid), coords)

    @torch.no_grad()
    def decode_grid(self, z_grid: Tensor, shape: Tuple[int, int, int],
                    chunk: int = 1_048_576) -> Tensor:
        """Décode un volume complet à la résolution `shape` — ARBITRAIRE et
        indépendante de celle du latent : c'est l'intérêt du décodeur implicite
        (ex. latent 1mm -> sortie 0.5mm native). Découpé en blocs de
        coordonnées pour borner la mémoire."""
        from cfm.inr_backbone import make_coord_grid
        feat = self.encode_features(z_grid)
        coords = make_coord_grid(shape, device=z_grid.device)
        outs = []
        for i in range(0, coords.shape[0], chunk):
            sub = coords[i:i + chunk].unsqueeze(0).expand(z_grid.shape[0], -1, -1)
            outs.append(self.query(feat, sub))
        return torch.cat(outs, dim=1).reshape(z_grid.shape[0], 1, *shape)

    def loss(self, z_grid: Tensor, coords: Tensor, targets: Tensor) -> Tensor:
        """Perte pondérée foreground, identique à celle du backbone global
        (`inr_backbone._weighted_mse`) — le fond plat domine sinon la MSE et le
        réseau converge vers un « cerveau moyen » (piège rencontré en Phase 1)."""
        return _weighted_mse(self.forward(z_grid, coords), targets,
                             self.cfg.fg_weight, self.cfg.bg_threshold)


if __name__ == "__main__":
    # Test d'ALIGNEMENT : une erreur d'ordre d'axes dans grid_sample est
    # silencieuse à l'entraînement (le réseau apprend une anatomie transposée).
    # On vérifie donc que l'interpolation retrouve bien les valeurs d'une grille
    # connue, aux positions attendues.
    from cfm.inr_backbone import make_coord_grid

    H, W, D = 4, 6, 8
    vol = torch.arange(H * W * D, dtype=torch.float32).reshape(1, 1, H, W, D)
    coords = make_coord_grid((H, W, D)).unsqueeze(0)           # (1, H*W*D, 3)
    got = sample_local_features(vol, coords).reshape(H, W, D)
    ref = vol[0, 0]
    err = (got - ref).abs().max().item()
    print(f"alignement grid_sample : erreur max = {err:.6f}")
    assert err < 1e-3, "ORDRE D'AXES INCORRECT dans sample_local_features"

    m = GridConditionedINR(GridINRConfig(feat_dim=16, hidden_dim=64, num_hidden_layers=3))
    z = torch.randn(2, 1, 12, 14, 12)
    print("query   :", tuple(m(z, torch.rand(2, 100, 3) * 2 - 1).shape))
    print("decode  :", tuple(m.decode_grid(z, (24, 28, 24)).shape), "(latent 12x14x12 -> sortie 24x28x24)")
    print("params  :", f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")
    print("✅ OK")
