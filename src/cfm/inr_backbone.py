#!/usr/bin/env python3
"""INR backbone — SIREN + hypernetwork + meta-learning (NOIR/MedFuncta-style).

Standalone module: no MedVAE, no MMFM/flow-matching dependency. Implements
Phase 1 of docs/MMFM_INR_STATE_OF_THE_ART.md — a shared, meta-learned SIREN
coordinate network phi(x; theta, gamma) where the per-volume modulation
`gamma` is predicted by a hypernetwork M_psi from a low-dimensional latent
code `z`. Follows NOIR's Algorithm 1 (meta-training the shared (theta, psi))
and Algorithm 2 (fitting a new `z*` with (theta, psi) frozen) — see the state-
of-the-art doc, section 1.2, for the paper's exact pseudocode this mirrors.

Modulation is a per-hidden-layer additive shift to the pre-sine activation
(the standard "shift modulation" used by Functa/COIN++/NOIR), not a full
weight hypernetwork — keeps the hypernetwork small and the whole backbone
comparable in size to the existing VectorMMFM (~1M params).

The inner loop (`INRBackbone.fit_latent`) is a plain, explicit gradient
descent on `z` (matches the paper's Algorithm 1/2 pseudocode directly, not an
implicit-differentiation trick): `create_graph=True` keeps it differentiable
w.r.t. (theta, psi) for meta-training (MetaSDF-style full unroll — K=5 is
small enough this stays cheap); `create_graph=False` detaches z between steps
for standalone fitting (Algorithm 2 / latent precompute), where no outer
gradient is needed and keeping the graph would only waste memory.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Coordinates processed per block when fitting/decoding a dense grid. Sized so
# one block's SIREN activation graph (~12 tensors of C x hidden_dim) stays near
# 12 GB at hidden_dim=256 — i.e. exactly the footprint a dense 2mm fit already
# had, which is known to run comfortably here. A dense 1mm fit is then 8 blocks.
DEFAULT_FIT_CHUNK = 1_048_576


# ===========================================================================
# Coordinate grid utilities
# ===========================================================================


def make_coord_grid(
    shape: Tuple[int, int, int], device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Normalized voxel-center coordinates in [-1, 1]^3 for a grid of `shape`.

    Resolution-independent by construction: the same physical location maps
    to (approximately) the same coordinate value regardless of `shape` — this
    is what makes INR fitting/evaluation resolution-invariant (see NOIR's
    epsilon-ReNO property, state-of-the-art doc section 1.3).

    Returns: (prod(shape), 3) tensor.
    """
    axes = [torch.linspace(-1.0, 1.0, s, device=device, dtype=dtype) for s in shape]
    grids = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(grids, dim=-1).reshape(-1, 3)


def _weighted_mse(pred: Tensor, target: Tensor, fg_weight: float = 5.0, bg_threshold: float = -0.9) -> Tensor:
    """Foreground-upweighted MSE (mirrors common/metrics.py's
    weighted_pixel_loss). After center-crop/pad + percentile normalization,
    most voxels are flat background (~-1) and nearly identical across every
    subject — plain voxel-wise MSE is dominated by that shared, subject-
    invariant shape, so a fit can reach a deceptively low loss without
    routing any real anatomical detail through `z` (empirically confirmed:
    z stayed pinned near its zero-init, invariant to which volume was
    fitted, even as the hypernetwork's weights grew substantially over
    training — see the state-of-the-art doc's Phase 1 notes). Foreground
    upweighting forces the fit to actually pay for getting brain content
    right, which is what should drive z."""
    with torch.no_grad():
        fg_mask = (target > bg_threshold).float()
        weight = 1.0 + (fg_weight - 1.0) * fg_mask
        weight = weight / weight.mean()
    return ((pred - target) ** 2 * weight).mean()


def sample_points(
    coords: Tensor, values: Tensor, num_points: int, generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    """Random subset of `num_points` shared across the batch (sparse
    supervision, MedFuncta-style — full-volume gradients at every inner step
    are unnecessary and, for 3D volumes, too memory-heavy under create_graph).

    coords: (N, 3). values: (B, N, C). Returns (coords_sub, values_sub).
    """
    n = coords.shape[0]
    if num_points >= n:
        return coords, values
    if generator is None:
        # Le tirage doit être REPRODUCTIBLE : le cache de latents est construit
        # une fois par `precompute_inr_latents.py` et le `z` est re-ajusté à
        # l'inférence. Sans graine, deux ajustements du même volume diffèrent de
        # ~4.4 % en L2 relatif (mesuré, results/mmfm/audit_20260825/latent_drift.csv),
        # ce qui constituait le plancher irréductible de l'audit. La graine est
        # dérivée du contenu pour rester indépendante de l'ordre d'appel.
        #
        # PIÈGE MESURÉ (2026-09-06) — cette graine est CONSTANTE : elle ne dépend
        # que de `n` et `num_points`. C'est ce qu'on veut pour AJUSTER un `z`
        # (Algorithme 2, appel unique par volume), et c'est un défaut grave pour le
        # MÉTA-ENTRAÎNEMENT, qui appelle cette fonction à chaque pas : les 50 000
        # pas du backbone de production ont vu les MÊMES 16 384 coordonnées, soit
        # 0.198 % de la grille (espacement moyen 8.0 voxels). Vérifié : deux appels
        # rendent des indices identiques. Conséquence mesurable sur le checkpoint de
        # production : après 50 000 pas les poids partagés sont à 1.01x / 1.01x /
        # 1.16x de leur initialisation SIREN — la base n'a jamais été apprise.
        # `meta_train_step` passe désormais un générateur qui AVANCE à chaque pas.
        generator = torch.Generator(device=coords.device)
        generator.manual_seed(int(n) * 1_000_003 + int(num_points))
    idx = torch.randperm(n, device=coords.device, generator=generator)[:num_points]
    return coords[idx], values[:, idx]


# ===========================================================================
# Modulated SIREN
# ===========================================================================


class SineLayer(nn.Module):
    """SIREN linear+sine layer (Sitzmann et al. 2020), with the paper's
    principled uniform initialization so pre-activations start in a sensible
    range for sin(). Accepts an optional per-sample additive shift applied to
    the pre-activation (the hypernetwork-predicted modulation)."""

    def __init__(self, in_features: int, out_features: int, omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            bound = (1.0 / in_features) if is_first else (math.sqrt(6.0 / in_features) / omega_0)
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(
        self,
        x: Tensor,
        shift: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
        lora_a: Optional[Tensor] = None,
        lora_b: Optional[Tensor] = None,
    ) -> Tensor:
        """`sin(w0 * (scale ⊙ (W'x + b) + shift))` avec `W' = W + A Bᵀ`.

        - `shift` seul = comportement de référence NOIR. omega_0 multiplie
          (linear(x) + shift) EN BLOC, conformément à `modulated_forward`
          (Sidaty1/NOIR, noir/inrs.py) : `sin(w0*(Wx+b+shift))` et non
          `sin(w0*(Wx+b)) + shift` appliqué après coup. Une version antérieure
          de ce fichier ajoutait le shift APRÈS la mise à l'échelle par omega_0,
          affaiblissant son effet sur la phase d'un facteur omega_0 (=30x).
        - `scale` (FiLM complet) : utilisé comme `1 + scale`, pour qu'une
          modulation nulle (z au zero-init) redonne exactement le réseau non
          modulé — même convention que `shift`.
        - `lora_a`/`lora_b` : mise à jour de rang faible des POIDS eux-mêmes,
          `W + A Bᵀ`, calculée sans jamais matérialiser W' :
          `x @ (W + A Bᵀ)ᵀ = linear(x) + (x @ B) @ Aᵀ`. C'est ce qui fait
          réellement croître la capacité par volume — la modulation
          d'activations seule plafonne à `num_couches x hidden_dim` quelle que
          soit la taille de z (mesuré : latent_dim x32 = résultat légèrement
          PIRE, cf. results/mmfm/comparison_20260807_1mm/manifest.md).
        """
        pre = self.linear(x)
        if lora_a is not None and lora_b is not None:
            pre = pre + torch.matmul(torch.matmul(x, lora_b), lora_a.transpose(-1, -2))
        if scale is not None:
            pre = pre * (1.0 + scale)
        if shift is not None:
            pre = pre + shift
        return torch.sin(self.omega_0 * pre)


class ModulatedSIREN(nn.Module):
    """Shared coordinate network phi(x; theta, gamma). `gamma` is a flat
    per-sample vector (from the hypernetwork), split evenly across every
    hidden layer (including the first) as a per-channel shift."""

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden_dim: int = 256,
        num_hidden_layers: int = 6,
        omega_0: float = 30.0,
        omega_hidden: float = 30.0,
        modulate_scale: bool = False,
        lora_rank: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.in_dim = in_dim
        self.modulate_scale = modulate_scale
        self.lora_rank = int(lora_rank)
        self.first = SineLayer(in_dim, hidden_dim, omega_0=omega_0, is_first=True)
        self.hidden = nn.ModuleList(
            [SineLayer(hidden_dim, hidden_dim, omega_0=omega_hidden, is_first=False)
             for _ in range(num_hidden_layers - 1)]
        )
        self.final = nn.Linear(hidden_dim, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / omega_hidden
            self.final.weight.uniform_(-bound, bound)
            self.final.bias.zero_()

    def _layer_dims(self) -> list:
        """(in_features, out_features) de chaque couche modulée, dans l'ordre."""
        return [(self.in_dim, self.hidden_dim)] + \
               [(self.hidden_dim, self.hidden_dim)] * (self.num_hidden_layers - 1)

    @property
    def modulation_dim(self) -> int:
        """Nombre total de valeurs que `gamma` doit fournir.

        C'est LA quantité qui borne l'information transmissible par volume —
        indépendante de `latent_dim` et de la résolution. Composition par couche :
        shift (hidden) [+ scale (hidden)] [+ A (out x r) + B (in x r)].
        """
        d = 0
        for fin, fout in self._layer_dims():
            d += fout                                  # shift
            if self.modulate_scale:
                d += fout                              # scale
            if self.lora_rank > 0:
                d += fout * self.lora_rank             # A
                d += fin * self.lora_rank              # B
        return d

    def _split(self, gamma: Tensor):
        """Découpe `gamma` (B, modulation_dim) en paramètres par couche."""
        b = gamma.shape[0]
        out, off = [], 0
        r = self.lora_rank
        for fin, fout in self._layer_dims():
            shift = gamma[:, off:off + fout].unsqueeze(1); off += fout
            scale = None
            if self.modulate_scale:
                scale = gamma[:, off:off + fout].unsqueeze(1); off += fout
            a = bmat = None
            if r > 0:
                a = gamma[:, off:off + fout * r].reshape(b, fout, r); off += fout * r
                bmat = gamma[:, off:off + fin * r].reshape(b, fin, r); off += fin * r
            out.append((shift, scale, a, bmat))
        assert off == gamma.shape[1], f"gamma mal découpé : {off} != {gamma.shape[1]}"
        return out

    @torch.no_grad()
    def lora_mask(self) -> Tensor:
        """Masque booléen (modulation_dim,) : True sur les composantes LoRA
        (blocs A+B de chaque couche). Portage de
        `bench_inr_capacity.py::gamma_layout` en méthode d'instance — sert à
        donner à la modulation directe (`hyper_hidden_dim=0`) un traitement
        distinct (init, taux d'apprentissage) pour sa portion LoRA. Masque
        entièrement `False` si `lora_rank=0`."""
        mask = torch.zeros(self.modulation_dim, dtype=torch.bool)
        r = self.lora_rank
        if r == 0:
            return mask
        off = 0
        for fin, fout in self._layer_dims():
            off += fout
            if self.modulate_scale:
                off += fout
            mask[off:off + fout * r + fin * r] = True
            off += fout * r + fin * r
        return mask

    @torch.no_grad()
    def init_direct_modulation(self, device, generator: Optional[torch.Generator] = None,
                               b_scale: float = 1e-2) -> Tensor:
        """Point de départ de `z` quand `z` EST `gamma` directement
        (`hyper_hidden_dim=0`). Portage de `bench_inr_capacity.py::init_gamma`.

        Zéro partout (shift = identité, A = 0) SAUF la portion `B` de chaque
        bloc LoRA, initialisée aléatoire : `A=0 ET B=0` simultanément rend le
        gradient LoRA nul des deux côtés (`dL/dA ∝ x@B=0`, `dL/dB ∝ A=0`),
        mort pour toujours — convention Hu et al. 2021 (LoRA), déjà validée
        par le sondage `bench_inr_capacity.py` sur ce même SIREN gelé."""
        g = torch.zeros(1, self.modulation_dim, device=device)
        r = self.lora_rank
        if r == 0:
            return g
        gen = generator
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(7)
        off = 0
        for fin, fout in self._layer_dims():
            off += fout
            if self.modulate_scale:
                off += fout
            off += fout * r                                   # A reste a zero
            g[0, off:off + fin * r] = torch.randn(fin * r, device=device, generator=gen) * b_scale
            off += fin * r
        return g

    def forward(self, coords: Tensor, gamma: Tensor) -> Tensor:
        """coords: (B, N, in_dim). gamma: (B, modulation_dim). -> (B, N, out_dim)."""
        mods = self._split(gamma)
        h = self.first(coords, *mods[0])
        for i, layer in enumerate(self.hidden):
            h = layer(h, *mods[i + 1])
        return self.final(h)


class Hypernetwork(nn.Module):
    """M_psi: z -> gamma. Single hidden layer, matching the reference NOIR
    implementation's LatentToModulation (Sidaty1/NOIR, noir/inrs.py): SiLU
    activation, plain PyTorch default init (no deliberate output-layer
    dampening — an earlier version of this file added a x0.01 scale-down to
    start near an "unmodulated" SIREN, an idea borrowed from general
    modulated-INR literature, not from NOIR itself; combined with the
    since-fixed missing omega_0 scaling on the shift term, it made the
    initial modulation signal far weaker than intended)."""

    def __init__(self, latent_dim: int, modulation_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, modulation_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


# ===========================================================================
# Backbone: SIREN + hypernetwork + meta-learning
# ===========================================================================


@dataclass
class INRBackboneConfig:
    latent_dim: int = 512
    hidden_dim: int = 256
    num_hidden_layers: int = 6
    omega_0: float = 30.0
    omega_hidden: float = 30.0
    hyper_hidden_dim: int = 64
    inner_lr: float = 1.0e-2
    inner_steps_train: int = 5
    inner_steps_eval: int = 10
    fg_weight: float = 5.0
    bg_threshold: float = -0.9
    # Enrichissement de la modulation (2026-08-11). Défauts = comportement
    # historique strictement inchangé (shift seul, pas de LoRA).
    #   modulate_scale : FiLM complet `sin(w0*(scale*(Wx+b)+shift))` -> capacité x2
    #   lora_rank > 0  : modulation BAS RANG DES POIDS `W + A B^T` -> capacité
    #                    x(2*r) environ. C'est le seul levier qui fasse réellement
    #                    croître l'information par volume : `latent_dim` ne le fait
    #                    pas (mesuré, cf. comparison_20260807_1mm/manifest.md).
    modulate_scale: bool = False
    lora_rank: int = 0
    # Requis quand hyper_hidden_dim=0 ET lora_rank>0 (modulation directe +
    # LoRA) : `z` EST alors `gamma`, et la portion LoRA de `z` a besoin de son
    # propre pas de descente, distinct de celui des décalages (`inner_lr`).
    # Au pas des décalages (échelle ~1e-2), la LoRA diverge dès le premier pas
    # (mesuré par bench_inr_capacity.py::mod_lora* : nRMSE 0.83 sans ce
    # traitement séparé). `None` = comportement historique inchangé (aucun
    # effet tant que ce combo n'est pas activé).
    lora_inner_lr: Optional[float] = None


def diff_inr_configs(current: "INRBackboneConfig", expected: Optional[dict]) -> list:
    """Fichier de cohérence INR — D0.

    Retourne la liste des champs de `current` qui diffèrent de `expected`
    (dict plat des champs de la config du backbone tel qu'entré, sérialisé avec
    `asdict(INRBackboneConfig(...))`). `expected` à `None` = comparaison pas
    demandée → retourne toujours [].

    Ce gardien couvre ce qu'un `load_state_dict` ne peut PAS détecter : les
    champs de la PROCÉDURE de fitting qui déterminent le `z` (inner_lr,
    inner_steps_*, fg_weight, bg_threshold) — une divergence est silencieuse
    (mêmes formes de poids) mais le flow reçoit alors un z source hors
    distribution. Elle couvre aussi les champs d'ARCHITECTURE (hidden_dim,
    num_hidden_layers, latent_dim, hyper_hidden_dim, omega_*, …) dont une
    divergence ferait autrement échouer `load_state_dict` avec un message
    obscur plutôt que le diagnostic précis de ce diff.

    `None` dans `expected` = champ absent (ancien checkpoint) → ignoré.
    """
    if expected is None:
        return []
    diffs = []
    for name, cur_val in asdict(current).items():
        exp_val = expected.get(name, None)
        if exp_val is None:
            continue
        if cur_val != exp_val:
            diffs.append((name, cur_val, exp_val))
    return diffs


class INRBackbone(nn.Module):
    def __init__(self, cfg: INRBackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.siren = ModulatedSIREN(
            in_dim=3, out_dim=1, hidden_dim=cfg.hidden_dim, num_hidden_layers=cfg.num_hidden_layers,
            omega_0=cfg.omega_0, omega_hidden=cfg.omega_hidden,
            modulate_scale=cfg.modulate_scale, lora_rank=cfg.lora_rank,
        )
        # hyper_hidden_dim=0 => MODULATION DIRECTE (pas de hypernetwork) : gamma = z.
        # Nécessaire dès que `modulation_dim` est grand, sinon le hypernetwork
        # recomprime tout par `hyper_hidden_dim` et la capacité gagnée sur la
        # modulation est annulée par ce goulot (l'information transmise est
        # bornée par min(latent_dim, hyper_hidden_dim, modulation_dim)).
        # C'est le schéma de Functa/MedFuncta : z EST la modulation, ajustée
        # directement par descente de gradient.
        if cfg.hyper_hidden_dim == 0:
            if cfg.latent_dim != self.siren.modulation_dim:
                raise ValueError(
                    f"hyper_hidden_dim=0 (modulation directe) impose "
                    f"latent_dim == modulation_dim, or latent_dim={cfg.latent_dim} "
                    f"et modulation_dim={self.siren.modulation_dim}."
                )
            self.hypernet = nn.Identity()
        else:
            self.hypernet = Hypernetwork(cfg.latent_dim, self.siren.modulation_dim, cfg.hyper_hidden_dim)

        # z EST gamma sous modulation directe : si lora_rank>0, sa portion LoRA
        # a besoin d'une init non nulle et d'un pas propre (voir fit_latent) —
        # sans quoi le gradient LoRA est mort des deux côtés, pour toujours.
        self._direct_lora = (cfg.hyper_hidden_dim == 0 and cfg.lora_rank > 0)
        if self._direct_lora:
            if cfg.lora_inner_lr is None:
                raise ValueError(
                    "lora_inner_lr est requis quand hyper_hidden_dim=0 et lora_rank>0 "
                    "(modulation directe + LoRA) : la portion LoRA de z a besoin d'un "
                    "pas distinct de inner_lr, sinon elle diverge ou reste morte."
                )
            self.register_buffer("_lora_mask", self.siren.lora_mask(), persistent=False)

    def decode(self, coords: Tensor, z: Tensor) -> Tensor:
        """coords: (B, N, 3), z: (B, latent_dim) -> (B, N, 1) intensities."""
        gamma = self.hypernet(z)
        return self.siren(coords, gamma)

    def fit_latent(
        self, coords: Tensor, targets: Tensor, num_steps: int, lr: float,
        z_init: Optional[Tensor] = None, create_graph: bool = False,
        chunk_size: Optional[int] = None, lora_init_generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """MetaSDF-style inner loop: gradient descent on `z` only, (theta,
        psi) untouched. coords/targets: (B, N, ·). Returns z: (B, latent_dim).

        `chunk_size` splits each inner step's forward/backward over blocks of
        coordinates, accumulating d(loss)/dz across them. This is EXACT, not an
        approximation: the loss is a sum over points, so the gradient is the sum
        of the per-chunk gradients (the normalizing constants — 1/N and the
        foreground weight's own mean — are computed once over the FULL point set
        before chunking, so each chunk contributes its true share). Needed from
        1mm on: a dense fit there covers 192x224x192 = 8.25M points, whose
        SIREN activation graph would be ~100 GB, against ~12 GB for the 1.03M
        points of a dense 2mm fit. Ignored when `create_graph=True`
        (meta-training), which only ever sees `points_per_step` sampled
        coordinates (16384) and would need the whole unrolled graph live anyway.
        """
        b = coords.shape[0]
        device = coords.device
        n = coords.shape[1]
        if z_init is not None:
            z = z_init.clone()
        elif self._direct_lora:
            z = self.siren.init_direct_modulation(
                device, generator=lora_init_generator
            ).expand(b, -1).clone()
        else:
            z = torch.zeros(b, self.cfg.latent_dim, device=device)
        z.requires_grad_(True)
        chunked = (not create_graph) and chunk_size is not None and n > chunk_size

        # Sous modulation directe+LoRA, la portion LoRA de z a besoin d'un pas
        # distinct de celui des décalages (voir INRBackboneConfig.lora_inner_lr) —
        # sinon, au pas des décalages, elle diverge dès le premier pas (mesuré
        # par bench_inr_capacity.py). `step_lr` reste un simple float dans le
        # cas historique (lora_rank=0 ou hypernetwork actif) : comportement
        # inchangé bit à bit.
        if self._direct_lora:
            step_lr = torch.full((self.cfg.latent_dim,), lr, device=device, dtype=torch.float32)
            step_lr[self._lora_mask.to(device)] = self.cfg.lora_inner_lr
            step_lr = step_lr.unsqueeze(0)
        else:
            step_lr = lr

        # Same normalization as _weighted_mse, but hoisted out of the loop so
        # every chunk divides by the SAME (global) weight mean — computing it
        # per chunk would silently reweight chunks against each other.
        if chunked:
            with torch.no_grad():
                w_full = 1.0 + (self.cfg.fg_weight - 1.0) * (targets > self.cfg.bg_threshold).float()
                w_full = w_full / w_full.mean()

        for _ in range(num_steps):
            if not chunked:
                pred = self.decode(coords, z)
                # *b matches NOIR's inner-loop loss scaling (graph_inner_loop_step:
                # `loss = ((recon-target)**2).mean() * batch_size`): `_weighted_mse`
                # already averages over B*N elements, which dilutes d(loss)/d(z_i)
                # by 1/B relative to fitting each batch item's z independently (the
                # actual intent of the inner loop) — the *b term cancels that
                # dilution so a given inner_lr has the same per-item effect
                # regardless of batch_size.
                loss = _weighted_mse(pred, targets, self.cfg.fg_weight, self.cfg.bg_threshold) * b
                (grad_z,) = torch.autograd.grad(loss, z, create_graph=create_graph)
            else:
                grad_z = torch.zeros_like(z)
                for i in range(0, n, chunk_size):
                    sl = slice(i, min(i + chunk_size, n))
                    pred_c = self.decode(coords[:, sl], z)
                    # `_weighted_mse(...) * b` averages over B*N elements then
                    # multiplies by B, i.e. it is exactly sum/N — so each chunk
                    # contributes its own sum divided by the FULL n.
                    part = ((pred_c - targets[:, sl]) ** 2 * w_full[:, sl]).sum() / n
                    (g,) = torch.autograd.grad(part, z)
                    grad_z = grad_z + g
            z = z - step_lr * grad_z
            if not create_graph:
                z = z.detach().requires_grad_(True)
        return z


# ===========================================================================
# Meta-training (Algorithm 1) / fitting a new volume (Algorithm 2)
# ===========================================================================


def meta_train_step(
    backbone: INRBackbone,
    volumes: Tensor,
    coords_full: Tensor,
    optimizer: torch.optim.Optimizer,
    num_points: int,
    generator: Optional[torch.Generator] = None,
    grad_clip: Optional[float] = 1.0,
    z_reg_weight: float = 0.0,
) -> Tuple[float, float, float]:
    """One outer step of NOIR Algorithm 1 on a batch of volumes.

    volumes: (B, 1, H, W, D) already in [-1, 1]. coords_full: (N, 3) for the
    same (H, W, D) grid (make_coord_grid). The outer loss is evaluated on the
    SAME sampled points used to fit `z` in the inner loop (matches NOIR's
    pseudocode — no separate query set).

    Gradient clipping defaults ON (matches mmfm_core.py's flow training):
    the inner loop backprops through 5 unrolled SGD steps (create_graph=True,
    MetaSDF-style), which can occasionally produce a large gradient spike —
    observed in practice to knock Adam's momentum into a state the run never
    recovers from (loss jumping from ~0.08 to ~0.68 within ~200 steps and
    staying there for the rest of training). Pass grad_clip=None to disable.

    `z_reg_weight` (0.0 = disabled) adds a Hutchinson-estimator gradient
    penalty encouraging decode(z) to be locally smooth in z — motivated by a
    direct measurement (src/cfm/diagnose_inr_latent_smoothness.py) showing
    the INR's cross-field interpolation roughness is ~3x its own same-field
    baseline, vs ~1.3x for MedVAE's latent, on the exact axis the flow model
    must traverse (see /home/rousseau/.claude/plans/temporal-plotting-
    engelbart.md and project memory for the full investigation this
    responds to — flow capacity and representation fidelity were both ruled
    out first). Reuses the single `pred` already computed for the
    reconstruction loss (one extra VJP through the same graph, not a whole
    extra unrolled pass) — WGAN-GP-style gradient penalty, well-posed here
    since SIREN/SiLU are smooth everywhere (no ReLU-style dead second
    derivative).

    Returns (loss, grad_norm, jac_penalty) — grad_norm and jac_penalty are
    logged by callers (e.g. train_inr_backbone.py) specifically so a spike
    like the one above is visible in real time instead of only diagnosable
    after the fact from the loss curve, and so `z_reg_weight` sweeps can
    confirm the penalty term is actually shrinking, not just watch the total
    loss."""
    b = volumes.shape[0]
    values_full = volumes.reshape(b, -1, 1)
    coords, values = sample_points(coords_full, values_full, num_points, generator)
    coords_b = coords.unsqueeze(0).expand(b, -1, -1)

    z = backbone.fit_latent(coords_b, values, backbone.cfg.inner_steps_train, backbone.cfg.inner_lr,
                            create_graph=True, lora_init_generator=generator)
    pred = backbone.decode(coords_b, z)
    recon_loss = _weighted_mse(pred, values, backbone.cfg.fg_weight, backbone.cfg.bg_threshold)

    if z_reg_weight > 0.0:
        # Rademacher probe vector (+-1, not Gaussian) is the standard
        # Hutchinson-estimator choice — minimum-variance for the squared-VJP
        # quantity we want (Hutchinson 1990).
        v = torch.randint(0, 2, pred.shape, device=pred.device, dtype=pred.dtype) * 2 - 1
        (vjp,) = torch.autograd.grad((pred * v).sum(), z, create_graph=True)
        jac_penalty = (vjp ** 2).sum(dim=1).mean()
        loss = recon_loss + z_reg_weight * jac_penalty
    else:
        jac_penalty = torch.zeros((), device=pred.device)
        loss = recon_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(backbone.parameters(), grad_clip)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(backbone.parameters(), float("inf"))
    optimizer.step()
    return float(recon_loss.item()), float(grad_norm), float(jac_penalty.item())


def fit_new_volume(
    backbone: INRBackbone,
    volume: Tensor,
    coords_full: Tensor,
    num_steps: Optional[int] = None,
    lr: Optional[float] = None,
    num_points: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    chunk_size: Optional[int] = DEFAULT_FIT_CHUNK,
) -> Tensor:
    """NOIR Algorithm 2: fit z* for one or more new volumes with (theta,
    psi) frozen. volume: (B, 1, H, W, D). Returns z: (B, latent_dim),
    detached. Caller is expected to have frozen backbone.parameters().

    `chunk_size` bounds the activation memory of each inner step without
    changing the result (see fit_latent); it is a no-op below that many points,
    so 2mm-era callers behave exactly as before.
    """
    num_steps = num_steps or backbone.cfg.inner_steps_eval
    lr = lr if lr is not None else backbone.cfg.inner_lr
    b = volume.shape[0]
    values_full = volume.reshape(b, -1, 1)
    if num_points is not None:
        coords, values = sample_points(coords_full, values_full, num_points, generator)
    else:
        coords, values = coords_full, values_full
    coords_b = coords.unsqueeze(0).expand(b, -1, -1)
    with torch.enable_grad():
        z = backbone.fit_latent(coords_b, values, num_steps, lr, create_graph=False,
                                chunk_size=chunk_size)
    return z.detach()


@torch.no_grad()
def decode_volume(
    backbone: INRBackbone,
    z: Tensor,
    coords_full: Tensor,
    volume_size: Tuple[int, int, int],
    chunk_size: int = DEFAULT_FIT_CHUNK,
) -> Tensor:
    """Decode phi(x; theta, M_psi(z)) over a full grid, in coordinate blocks.

    Same motivation as fit_latent's chunking, one order of magnitude milder
    (no autograd graph to keep): a 1mm grid is 8.25M points, a 0.5mm grid 66M.
    Returns (B, 1, *volume_size).
    """
    b = z.shape[0]
    n = coords_full.shape[0]
    out = torch.empty(b, n, device=z.device, dtype=z.dtype)
    for i in range(0, n, chunk_size):
        sl = slice(i, min(i + chunk_size, n))
        sub = coords_full[sl].unsqueeze(0).expand(b, -1, -1)
        out[:, sl] = backbone.decode(sub, z).squeeze(-1)
    return out.reshape(b, 1, *volume_size)
