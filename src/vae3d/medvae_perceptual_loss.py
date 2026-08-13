#!/usr/bin/env python3
"""Loss LPIPS + discriminateur PatchGAN pour le fine-tuning MedVAE 3D.

Réimplémentation fidèle de `medvae.losses.LPIPSWithDiscriminator`
(github.com/StanfordMIMI/MedVAE, `medvae/losses/vae_losses.py`) — la classe
assemblée n'est PAS exposée par le package pip `medvae` installé, qui ne
fournit que les briques (`LPIPS`, `NLayerDiscriminator`, `hinge_loss`,
`weights_init` dans `medvae/utils/vae/loss_components.py`). On importe donc
ces briques et on réassemble la loss ici.

Pourquoi c'est nécessaire (voir manifest/mémoire projet) : notre fine-tuning
MedVAE utilisait une L1 pure (perceptuelle et adversariale absentes, KL à 0),
alors que le checkpoint pré-entraîné qu'on affine a lui-même été entraîné
AVEC ces termes. Une perte purement pixel entraîne le réseau à prédire
l'espérance conditionnelle sous incertitude — donc un flou par moyennage
(Larsen et al. 2016 arXiv:1512.09300 ; Isola et al. 2017 pix2pix
arXiv:1611.07004 ; Rombach et al. 2022 LDM arXiv:2112.10752, dont la loss
MedVAE descend directement). Le papier MedVAE (Varma et al., arXiv:2502.14753)
rapporte MS-SSIM 0.983 / PSNR 29.52 sur IRM cérébrale pour exactement
`medvae_4_1_3d` — la capacité est là, c'est la recette qui manquait.

Config officielle de référence pour ce modèle
(`configs/experiment/medvae_4x_1c_3d_finetuning.yaml` +
`configs/criterion/lpips_with_discriminator.yaml`) :
    kl_weight=1e-6, disc_weight=0.5, perceptual_weight=1.0,
    num_channels=1, disc_start=3125, lr=4.5e-6, batch_size=4,
    patches 64^3 aléatoires, volumes normalisés dans [-1, 1].

Écart volontaire vs la référence — VECTORISATION, maths identiques :
leur `discriminator_2d_nets_to_3d` et leur boucle perceptuelle itèrent en
Python sur CHAQUE coupe des 3 axes (pour un patch 64^3 : 192 passes VGG16
séquentielles par appel). On empile les coupes en un seul batch par axe et on
fait une passe — LPIPS/PatchGAN étant appliqués indépendamment par coupe, la
moyenne obtenue est la même, à coût très inférieur. `slice_stride` permet en
plus de sous-échantillonner les coupes (analogue au `fake_3d_ratio` de MONAI)
si le budget de calcul l'exige ; `slice_stride=1` = comportement de référence.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from medvae.utils.vae.loss_components import (
    LPIPS,
    NLayerDiscriminator,
    hinge_loss,
    weights_init,
)

__all__ = ["LPIPSWithDiscriminator3D"]


def _volume_to_slices(x: torch.Tensor, dim: int, stride: int = 1) -> torch.Tensor:
    """(B, C, D, H, W) -> (B*S, C, A, B) : toutes les coupes le long de `dim`.

    `dim` ∈ {2, 3, 4} (profondeur / hauteur / largeur). Équivalent vectorisé
    de la boucle `for j in range(inp.size(dim))` de la référence MedVAE.
    """
    if stride > 1:
        idx = torch.arange(0, x.shape[dim], stride, device=x.device)
        x = x.index_select(dim, idx)
    x = x.movedim(dim, 1)                       # (B, S, C, A, B)
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


class LPIPSWithDiscriminator3D(nn.Module):
    """L1 + LPIPS(2.5D) + KL + PatchGAN(2.5D) à poids adaptatif.

    `forward(..., optimizer_idx=0)` renvoie la loss du générateur (le VAE),
    `optimizer_idx=1` celle du discriminateur — même convention à deux
    optimiseurs que LDM/MedVAE.
    """

    def __init__(
        self,
        disc_start: int,
        kl_weight: float = 1.0e-6,
        disc_weight: float = 0.5,
        perceptual_weight: float = 1.0,
        num_channels: int = 1,
        slice_stride: int = 1,
        slice_chunk: int = 256,
        nll_reduction: str = "sum",
    ):
        super().__init__()
        if nll_reduction not in ("sum", "mean"):
            raise ValueError(f"nll_reduction doit être 'sum' ou 'mean', reçu {nll_reduction!r}")
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.discriminator_weight = disc_weight
        self.discriminator_iter_start = disc_start
        self.slice_stride = max(1, int(slice_stride))
        self.slice_chunk = int(slice_chunk)
        self.nll_reduction = nll_reduction

        self.perceptual_loss = LPIPS().eval()
        for p in self.perceptual_loss.parameters():
            p.requires_grad = False

        # logvar appris (LDM) : pondère automatiquement le terme de
        # reconstruction pendant l'entraînement.
        self.logvar = nn.Parameter(torch.zeros(size=()))

        self.discriminator = NLayerDiscriminator(input_nc=num_channels).apply(weights_init)

    # -- helpers 2.5D -------------------------------------------------------

    def _perceptual_3d(self, inputs: torch.Tensor, recons: torch.Tensor) -> torch.Tensor:
        """LPIPS moyennée sur les coupes des 3 axes (moyenne non pondérée par
        coupe, comme la référence)."""
        total = inputs.new_zeros(())
        count = 0
        for dim in (2, 3, 4):
            a = _volume_to_slices(inputs, dim, self.slice_stride)
            b = _volume_to_slices(recons, dim, self.slice_stride)
            for i in range(0, a.shape[0], self.slice_chunk):
                v = self.perceptual_loss(a[i : i + self.slice_chunk], b[i : i + self.slice_chunk])
                total = total + v.sum()
                count += v.shape[0]
        return total / max(count, 1)

    def _disc_logits_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Logits PatchGAN concaténés (aplatis) sur les coupes des 3 axes."""
        outs = []
        for dim in (2, 3, 4):
            s = _volume_to_slices(x, dim, self.slice_stride)
            for i in range(0, s.shape[0], self.slice_chunk):
                outs.append(self.discriminator(s[i : i + self.slice_chunk]).reshape(-1))
        return torch.cat(outs)

    def calculate_adaptive_weight(
        self, nll_loss: torch.Tensor, g_loss: torch.Tensor, last_layer: torch.Tensor
    ) -> torch.Tensor:
        """Poids adaptatif de la loss adversariale (LDM) : équilibre la norme
        du gradient adversarial sur celle du gradient de reconstruction, mesurés
        sur la dernière couche du décodeur."""
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * self.discriminator_weight

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        posteriors,
        optimizer_idx: int,
        global_step: int,
        last_layer: Optional[torch.Tensor] = None,
        split: str = "train",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        bsz = inputs.shape[0]
        d_valid = 0 if global_step < self.discriminator_iter_start else 1

        if optimizer_idx == 0:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            p_loss = (
                self._perceptual_3d(inputs, reconstructions)
                if self.perceptual_weight > 0
                else inputs.new_zeros(())
            )
            rec_loss = rec_loss + self.perceptual_weight * p_loss

            weighted = rec_loss / torch.exp(self.logvar) + self.logvar
            nll_loss = weighted.sum() / bsz if self.nll_reduction == "sum" else weighted.mean()

            kl_loss = posteriors.kl().sum() / bsz

            d_weight = inputs.new_zeros(())
            g_loss = inputs.new_zeros(())
            if d_valid:
                g_loss = -self._disc_logits_3d(reconstructions).mean()
                if last_layer is not None:
                    try:
                        d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
                    except RuntimeError:
                        d_weight = inputs.new_zeros(())

            loss = nll_loss + self.kl_weight * kl_loss + d_weight * d_valid * g_loss
            log = {
                f"{split}/total": float(loss.detach()),
                f"{split}/nll": float(nll_loss.detach()),
                f"{split}/rec": float(rec_loss.detach().mean()),
                f"{split}/perceptual": float(p_loss.detach()),
                f"{split}/kl": float(kl_loss.detach()),
                f"{split}/d_weight": float(d_weight.detach() if torch.is_tensor(d_weight) else d_weight),
                f"{split}/g_loss": float(g_loss.detach()),
                f"{split}/logvar": float(self.logvar.detach()),
            }
            return loss, log

        # optimizer_idx == 1 : discriminateur
        if not d_valid:
            zero = inputs.new_zeros(())
            return zero, {f"{split}/disc": 0.0}

        logits_real = self._disc_logits_3d(inputs.contiguous().detach())
        logits_fake = self._disc_logits_3d(reconstructions.contiguous().detach())
        d_loss = hinge_loss(logits_real, logits_fake)
        log = {
            f"{split}/disc": float(d_loss.detach()),
            f"{split}/logits_real": float(logits_real.detach().mean()),
            f"{split}/logits_fake": float(logits_fake.detach().mean()),
        }
        return d_loss, log
