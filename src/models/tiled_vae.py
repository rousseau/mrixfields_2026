#!/usr/bin/env python3
"""Encodage/décodage MedVAE par TUILES, pour travailler au-delà de 2mm.

Pourquoi c'est nécessaire (mesuré, voir manifest section « LA RÉSOLUTION EST LE
VRAI GOULOT ») : MedVAE possède une attention « vanilla » à son goulot, donc un
coût mémoire quadratique en nombre de voxels latents. Sur GB10 (121 GB) :
4.7 GB à 96×112×96, 13.8 GB à 128³, 46.1 GB à 160³, puis **OOM à 192×224×192**
(la résolution 1mm cible). Impossible d'encoder un volume 1mm en une passe.

Principe — tuiles NON RECOUVRANTES avec MARGE DE CONTEXTE :
    Un simple découpage en tuiles jointives produit des coutures : le réseau
    convolutif voit un bord artificiel là où la tuile est coupée. On extrait
    donc une tuile ÉLARGIE (tuile + marge de chaque côté), on l'encode, puis on
    ne conserve que la partie centrale du latent correspondant à la tuile
    nominale. Chaque voxel latent est ainsi calculé avec son vrai voisinage, et
    les latents des tuiles s'assemblent exactement (pas de recouvrement, pas de
    pondération, pas d'interpolation).

    Le décodage applique la symétrie exacte : latent élargi -> decode -> crop
    central en espace image.

Contraintes :
    - chaque dimension du volume doit être un multiple de la tuile ;
    - tuile et marge doivent être des multiples du facteur de compression (4)
      pour que le crop du latent tombe sur des frontières entières.

Configuration cible du projet : volume 192×224×192 @1mm = 2×2×2 tuiles de
96×112×96 (taille dont la faisabilité mémoire est mesurée), latent assemblé
48×56×48.
"""

from __future__ import annotations

from typing import Tuple

import torch

__all__ = ["tiled_encode", "tiled_decode", "DOWNSAMPLE"]

DOWNSAMPLE = 4  # medvae_4_1_3d : /4 par dimension


def _check_downsample(vae) -> None:
    """Refuse un modèle dont le facteur de compression n'est pas DOWNSAMPLE.

    `DOWNSAMPLE` est une CONSTANTE de module : avec `medvae_8_1_3d`, le latent d'une
    tuile est 8× plus petit par axe, le crop `[li:li + lh]` avec `lh = tuile // 4`
    déborde, et PyTorch tronque les tranches hors bornes SANS lever d'exception —
    on obtient un latent assemblé à la mauvaise échelle et une reconstruction fausse,
    sans aucun signal. Mesuré : le bras `medvae8` du banc de représentation a dû
    passer par la fenêtre glissante officielle pour cette raison.
    """
    name = getattr(getattr(vae, "medvae", None), "model_name", None)
    if name is None:
        name = getattr(vae, "model_name", None)
    if name is None:
        return                      # objet sans nom de modèle : rien à vérifier
    try:
        factor = int(str(name).split("_")[1])
    except (IndexError, ValueError):
        return
    if factor != DOWNSAMPLE:
        raise ValueError(
            f"tiled_encode/tiled_decode codent DOWNSAMPLE={DOWNSAMPLE} en dur, mais "
            f"'{name}' compresse d'un facteur {factor}. Utiliser la fenêtre glissante "
            f"de MedVAE (MVAE.encode/decode) pour ce modèle, ou paramétrer DOWNSAMPLE."
        )


def _as_latent(z) -> torch.Tensor:
    """Normalise la sortie d'`encode()` en tenseur.

    Selon l'objet passé, `encode()` renvoie soit un tenseur (wrapper `MVAE`,
    `MedVAEFineTuneWrapper`), soit un `DiagonalGaussianDistribution`
    (l'`AutoencoderKL` interne), soit un tuple (mean, logvar).

    ATTENTION — quand l'objet est un `MVAE`/`MedVAEFineTuneWrapper`, le tenseur
    reçu ici est un ÉCHANTILLON de la postérieure, pas le mode : la première
    branche le renvoie tel quel et il n'y a rien à en extraire. Le commentaire
    « on prend toujours le MODE » qui figurait ici jusqu'au 2026-09-06 était faux
    (voir `models/maisi_vae.py::encode`). Effet mesuré sur la reconstruction :
    nul ; effet sur la reproductibilité du cache : réel.

    ATTENTION — tester `hasattr(z, "mode")` en premier est un PIÈGE : un
    `torch.Tensor` possède lui aussi une méthode `.mode()` (le mode STATISTIQUE,
    qui renvoie un `torch.return_types.mode`). Les tenseurs matchaient donc cette
    branche et étaient silencieusement corrompus. On teste le type Tensor
    d'abord.
    """
    if isinstance(z, torch.Tensor):
        return z.float()
    if isinstance(z, (tuple, list)):
        return z[0].float()
    if hasattr(z, "mode"):          # DiagonalGaussianDistribution
        return z.mode().float()
    if hasattr(z, "sample"):
        return z.sample().float()
    raise TypeError(f"Sortie d'encode() non reconnue : {type(z)}")


def _tile_starts(size: int, tile: int) -> list[int]:
    if size % tile != 0:
        raise ValueError(
            f"La dimension {size} n'est pas un multiple de la tuile {tile} — "
            "choisir un volume_size/tuile compatibles (ex. 192×224×192 avec 96×112×96)."
        )
    return list(range(0, size, tile))


def _expand(start: int, tile: int, margin: int, size: int) -> Tuple[int, int, int]:
    """Fenêtre élargie [lo, hi) réellement extraite, et décalage du bloc nominal
    à l'intérieur de cette fenêtre."""
    lo = max(0, start - margin)
    hi = min(size, start + tile + margin)
    return lo, hi, start - lo


@torch.no_grad()
def tiled_encode(
    vae,
    x: torch.Tensor,
    tile: Tuple[int, int, int] = (96, 112, 96),
    margin: int = 16,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """(1, 1, H, W, D) -> (1, C, H/4, W/4, D/4), encodé tuile par tuile.

    `margin` (en voxels image, multiple de 4) est le contexte ajouté de chaque
    côté avant encodage puis retiré du latent. margin=0 reproduit un découpage
    naïf (coutures visibles).
    """
    if margin % DOWNSAMPLE != 0:
        raise ValueError(f"margin={margin} doit être un multiple de {DOWNSAMPLE}")
    _check_downsample(vae)
    device = x.device
    _, _, H, W, D = x.shape
    th, tw, td = tile
    out = None

    for i in _tile_starts(H, th):
        for j in _tile_starts(W, tw):
            for k in _tile_starts(D, td):
                i0, i1, oi = _expand(i, th, margin, H)
                j0, j1, oj = _expand(j, tw, margin, W)
                k0, k1, ok = _expand(k, td, margin, D)

                sub = x[:, :, i0:i1, j0:j1, k0:k1]
                with torch.amp.autocast("cuda", dtype=amp_dtype,
                                        enabled=(use_amp and device.type == "cuda")):
                    z_sub = vae.encode(sub)
                z_sub = _as_latent(z_sub)

                # Crop du latent : on ne garde que la part correspondant à la
                # tuile nominale (le contexte a servi au calcul, pas à la sortie).
                li, lj, lk = oi // DOWNSAMPLE, oj // DOWNSAMPLE, ok // DOWNSAMPLE
                lh, lw, ld = th // DOWNSAMPLE, tw // DOWNSAMPLE, td // DOWNSAMPLE
                z_core = z_sub[:, :, li:li + lh, lj:lj + lw, lk:lk + ld]

                if out is None:
                    out = torch.zeros(
                        1, z_core.shape[1], H // DOWNSAMPLE, W // DOWNSAMPLE, D // DOWNSAMPLE,
                        device=device, dtype=torch.float32,
                    )
                out[:, :, i // DOWNSAMPLE:i // DOWNSAMPLE + lh,
                       j // DOWNSAMPLE:j // DOWNSAMPLE + lw,
                       k // DOWNSAMPLE:k // DOWNSAMPLE + ld] = z_core
    return out


@torch.no_grad()
def tiled_decode(
    vae,
    z: torch.Tensor,
    tile: Tuple[int, int, int] = (96, 112, 96),
    margin: int = 16,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """(1, C, h, w, d) -> (1, 1, 4h, 4w, 4d), décodé tuile par tuile.

    `tile`/`margin` sont exprimés en voxels IMAGE (mêmes valeurs qu'à
    l'encodage) ; la conversion vers l'espace latent est faite ici.
    """
    if margin % DOWNSAMPLE != 0:
        raise ValueError(f"margin={margin} doit être un multiple de {DOWNSAMPLE}")
    _check_downsample(vae)
    device = z.device
    _, _, h, w, d = z.shape
    H, W, D = h * DOWNSAMPLE, w * DOWNSAMPLE, d * DOWNSAMPLE
    lt = tuple(t // DOWNSAMPLE for t in tile)
    lm = margin // DOWNSAMPLE
    out = torch.zeros(1, 1, H, W, D, device=device, dtype=torch.float32)

    for i in _tile_starts(h, lt[0]):
        for j in _tile_starts(w, lt[1]):
            for k in _tile_starts(d, lt[2]):
                i0, i1, oi = _expand(i, lt[0], lm, h)
                j0, j1, oj = _expand(j, lt[1], lm, w)
                k0, k1, ok = _expand(k, lt[2], lm, d)

                sub = z[:, :, i0:i1, j0:j1, k0:k1]
                with torch.amp.autocast("cuda", dtype=amp_dtype,
                                        enabled=(use_amp and device.type == "cuda")):
                    rec = vae.decode(sub)
                if isinstance(rec, (tuple, list)):
                    rec = rec[0]
                rec = rec.float()

                pi, pj, pk = oi * DOWNSAMPLE, oj * DOWNSAMPLE, ok * DOWNSAMPLE
                ph, pw, pd = tile
                core = rec[:, :, pi:pi + ph, pj:pj + pw, pk:pk + pd]
                out[:, :, i * DOWNSAMPLE:i * DOWNSAMPLE + ph,
                       j * DOWNSAMPLE:j * DOWNSAMPLE + pw,
                       k * DOWNSAMPLE:k * DOWNSAMPLE + pd] = core
    return out
