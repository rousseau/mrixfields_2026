#!/usr/bin/env python3
"""Conditionnement AdaGN (FiLM sur la group-norm) pour le UNet 3D de MONAI.

Le problème
-----------
`DiffusionUNetResnetBlock` (MONAI 1.5) conditionne ainsi :

    h = conv1(silu(norm1(x)))
    h = h + time_emb_proj(silu(emb))      # décalage ADDITIF seulement
    h = conv2(silu(norm2(h)))             # norm2 NON conditionnée

Le conditionnement ne peut donc que décaler les activations, jamais moduler
leur GAIN. Pire, `norm2` renormalise immédiatement après l'injection : une
group-norm soustrait la moyenne et divise par l'écart-type PAR GROUPE DE
CANAUX, ce qui efface la composante du décalage constante à l'intérieur d'un
groupe — c'est-à-dire une bonne partie de ce qu'on vient d'injecter.

Ce que fait ce module
---------------------
AdaGN, tel que publié par Dhariwal & Nichol, « Diffusion Models Beat GANs »
(arXiv:2105.05233), `guided_diffusion/unet.py::ResBlock` sous
`use_scale_shift_norm`, et repris par Stable Diffusion / LDM :

    scale, shift = emb_proj(silu(emb)).chunk(2)
    h = norm2(h) * (1 + scale) + shift    # FiLM APRÈS la normalisation
    h = conv2(silu(h))

Deux différences qui comptent : la modulation porte sur le GAIN autant que sur
le décalage, et elle s'applique APRÈS `norm2`, donc plus rien ne l'efface.

La convention `(1 + scale)` est celle de la référence : à modulation nulle, le
bloc redevient exactement le bloc non conditionné.

Pour ce projet précisément
--------------------------
Le champ magnétique est l'axe de temps continu et le contraste la classe ; les
deux sont sommés en un seul `emb` par MONAI (`emb = time_embed(t) +
class_embedding(y)`) — c'est aussi ce que fait la référence, on n'y touche pas.
Ce qui manquait était uniquement AdaGN. L'écart documenté du UNet se concentre
sur les transitions ->5T/->7T (0.821 contre 0.676 pour le vectorisé), donc sur
les extrémités de l'axe de champ, là où un conditionnement qui ne sait que
décaler est le plus limité.

Usage :
    unet = convert_unet_to_adagn(unet)          # remplace les blocs en place
    PYTHONPATH=src python src/models/adagn_conditioning.py   # auto-tests
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

try:
    from monai.networks.nets.diffusion_model_unet import DiffusionUNetResnetBlock
except ImportError:  # MONAI <= 1.3 / paquet `generative` séparé
    try:
        from monai.generative.networks.nets.diffusion_model_unet import DiffusionUNetResnetBlock
    except ImportError:
        from generative.networks.nets.diffusion_model_unet import DiffusionUNetResnetBlock

__all__ = ["AdaGNResnetBlock", "convert_unet_to_adagn", "count_resnet_blocks"]


class AdaGNResnetBlock(nn.Module):
    """Bloc résiduel MONAI dont la seconde group-norm est modulée (FiLM).

    On ne reconstruit rien : le bloc REPREND les sous-modules du bloc d'origine
    (`norm1`, `conv1`, `norm2`, `conv2`, `skip_connection`, et les éventuels
    `upsample`/`downsample` de `resblock_updown=True`). Cela garantit une
    structure et une initialisation identiques, et évite de dépendre de la
    signature du constructeur de MONAI, qui a déjà changé entre versions.

    Seul `time_emb_proj` est remplacé : il sortait `out_channels`, il sort
    maintenant `2 * out_channels` (gain et décalage).
    """

    def __init__(self, block: DiffusionUNetResnetBlock, zero_init: bool = True):
        super().__init__()
        self.spatial_dims = block.spatial_dims
        self.channels = block.channels
        self.emb_channels = block.emb_channels
        self.out_channels = block.out_channels
        self.up, self.down = block.up, block.down

        self.norm1 = block.norm1
        self.nonlinearity = block.nonlinearity
        self.conv1 = block.conv1
        self.upsample = block.upsample
        self.downsample = block.downsample
        self.norm2 = block.norm2
        self.conv2 = block.conv2
        self.skip_connection = block.skip_connection

        self.emb_proj = nn.Linear(self.emb_channels, 2 * self.out_channels)
        if zero_init:
            # Modulation nulle à l'initialisation (convention DiT,
            # arXiv:2212.09748) : le réseau démarre exactement comme son
            # équivalent non conditionné et apprend la modulation à partir de
            # là. La référence guided_diffusion laisse l'initialisation par
            # défaut ; DiT montre que l'init nulle converge nettement mieux, ce
            # qui compte ici où le budget est de 25k itérations seulement.
            nn.init.zeros_(self.emb_proj.weight)
            nn.init.zeros_(self.emb_proj.bias)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.norm1(x)
        h = self.nonlinearity(h)

        if self.upsample is not None:
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)

        # (B, 2C) -> deux (B, C, 1, 1, 1) diffusables sur les dims spatiales.
        mod = self.emb_proj(self.nonlinearity(emb))
        shape = (mod.shape[0], 2 * self.out_channels) + (1,) * self.spatial_dims
        scale, shift = mod.view(shape).chunk(2, dim=1)

        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.nonlinearity(h)
        h = self.conv2(h)
        return self.skip_connection(x) + h


def count_resnet_blocks(module: nn.Module) -> int:
    return sum(1 for m in module.modules() if isinstance(m, DiffusionUNetResnetBlock))


def convert_unet_to_adagn(module: nn.Module, zero_init: bool = True) -> int:
    """Remplace, EN PLACE, tous les `DiffusionUNetResnetBlock` par leur version
    AdaGN. Retourne le nombre de blocs convertis.

    Le parcours gère les `nn.ModuleList` (les blocs vivent dans des listes
    `resnets` au sein des blocs descendants/montants) aussi bien que les
    attributs nommés — un simple `setattr` sur le nom pointé par
    `named_modules()` échouerait sur les indices de liste.
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, DiffusionUNetResnetBlock):
            new = AdaGNResnetBlock(child, zero_init=zero_init)
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                module[int(name)] = new
            else:
                setattr(module, name, new)
            n += 1
        else:
            n += convert_unet_to_adagn(child, zero_init=zero_init)
    return n


# ---------------------------------------------------------------------------


def _reference_block_output(block: AdaGNResnetBlock, x: Tensor, emb: Tensor) -> Tensor:
    """Sortie attendue quand la modulation est nulle : le bloc doit se réduire
    EXACTEMENT au chemin non conditionné `conv2(silu(norm2(conv1(...))))`."""
    h = block.nonlinearity(block.norm1(x))
    if block.upsample is not None:
        x = block.upsample(x); h = block.upsample(h)
    elif block.downsample is not None:
        x = block.downsample(x); h = block.downsample(h)
    h = block.conv1(h)
    h = block.conv2(block.nonlinearity(block.norm2(h)))
    return block.skip_connection(x) + h


if __name__ == "__main__":
    import inspect

    from monai.networks.nets import DiffusionModelUNet

    torch.manual_seed(0)
    sig = inspect.signature(DiffusionModelUNet.__init__).parameters
    ch_kwarg = "num_channels" if "num_channels" in sig else "channels"
    unet = DiffusionModelUNet(
        spatial_dims=3, in_channels=2, out_channels=1,
        **{ch_kwarg: (8, 16)}, attention_levels=(False, False),
        num_res_blocks=1, num_head_channels=8, norm_num_groups=4,
        num_class_embeds=5, with_conditioning=False, resblock_updown=True,
    ).eval()

    before = count_resnet_blocks(unet)
    x = torch.randn(2, 2, 8, 8, 8)
    t = torch.tensor([3.0, 7.0])
    y = torch.tensor([0, 3])
    with torch.no_grad():
        out_ref = unet(x=x, timesteps=t, class_labels=y)

    n = convert_unet_to_adagn(unet, zero_init=True)
    after = count_resnet_blocks(unet)
    print(f"1. blocs convertis : {n}/{before} (restants non convertis : {after})")
    assert n == before > 0 and after == 0, "conversion incomplète"

    blk = next(m for m in unet.modules() if isinstance(m, AdaGNResnetBlock))
    xb = torch.randn(2, blk.channels, 8, 8, 8)
    eb = torch.randn(2, blk.emb_channels)

    # ATTENTION — sans la ligne qui suit, les tests 2 à 4 seraient CREUX.
    # MONAI initialise `conv2` à zéro (`zero_module`), donc à l'initialisation
    # la branche résiduelle sort zéro et le bloc vaut exactement `skip(x)`,
    # QUELLE QUE SOIT la modulation. Tout test de conditionnement mené sur un
    # bloc fraîchement initialisé passe donc trivialement — y compris sur une
    # branche morte. On rend la branche vivante avant de tester quoi que ce soit.
    with torch.no_grad():
        for p in blk.conv2.parameters():
            nn.init.normal_(p, std=0.1)

    # 2. Modulation nulle -> AdaGN doit redonner EXACTEMENT le chemin non
    #    conditionné. Ce n'est PAS le réseau d'origine : celui-ci ajoutait un
    #    décalage avant norm2, la version AdaGN ne l'ajoute plus. On compare
    #    donc au chemin sans conditionnement, calculé à la main.
    with torch.no_grad():
        err = (blk(xb, eb) - _reference_block_output(blk, xb, eb)).abs().max().item()
    print(f"2. modulation nulle == bloc non conditionné : erreur max {err:.2e}")
    assert err < 1e-5, "la modulation nulle ne redonne pas le bloc non conditionné"

    # 3. ... et la sortie doit alors être INSENSIBLE à emb.
    with torch.no_grad():
        d0 = (blk(xb, eb) - blk(xb, torch.randn_like(eb))).abs().max().item()
    print(f"3. init nulle -> sortie insensible à emb : écart max {d0:.2e}")
    assert d0 < 1e-6

    # 4. Modulation non nulle -> emb DOIT changer la sortie. Sans ce test, une
    #    branche morte passerait les tests 2 et 3 sans rien signaler.
    with torch.no_grad():
        nn.init.normal_(blk.emb_proj.weight, std=0.05)
        nn.init.normal_(blk.emb_proj.bias, std=0.05)
        d1 = (blk(xb, eb) - blk(xb, torch.randn_like(eb))).abs().max().item()
    print(f"4. modulation active -> emb change la sortie : écart max {d1:.2e}")
    assert d1 > 1e-4, "BRANCHE MORTE : la modulation n'atteint pas la sortie"

    # 4b. Le GAIN doit être modulé, pas seulement le décalage — c'est toute la
    #     différence avec le conditionnement d'origine. On annule la moitié
    #     « shift » et on vérifie que la sortie dépend encore de emb.
    with torch.no_grad():
        blk.emb_proj.weight[blk.out_channels:].zero_()
        blk.emb_proj.bias[blk.out_channels:].zero_()
        d2 = (blk(xb, eb) - blk(xb, torch.randn_like(eb))).abs().max().item()
    print(f"4b. gain seul (décalage annulé) -> emb change la sortie : écart max {d2:.2e}")
    assert d2 > 1e-4, "le gain n'est pas modulé : AdaGN dégénère en décalage additif"

    # 5. Le réseau complet reste appelable avec la même signature, et le gain
    #    comme le décalage sont bien appris (2C paramètres de modulation).
    with torch.no_grad():
        out = unet(x=x, timesteps=t, class_labels=y)
    p_before_mod = sum(p.numel() for m in unet.modules()
                       if isinstance(m, AdaGNResnetBlock) for p in m.emb_proj.parameters())
    print(f"5. forward complet : {tuple(out.shape)} (attendu {tuple(out_ref.shape)}) | "
          f"paramètres de modulation : {p_before_mod / 1e3:.1f}k")
    assert out.shape == out_ref.shape

    print("\n✅ AdaGN vérifié")
