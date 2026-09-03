# Évaluation qualitative — état final

**Date** : 2026-09-03
**Ce qui est nouveau** : les figures portent maintenant les modèles de fin de série
(**R-best**, **LPIPS**) et **les deux plafonds de représentation**, là où la
campagne du 2026-08-28 s'arrêtait à la production d'août.

---

## Ce que les images montrent, et qui corrobore le quantitatif

Sur `panel_T1W_0.1T_to_7T_0006.png`, trois choses se lisent d'un coup :

1. **Les prédictions ressemblent à la SOURCE, pas à la cible.** Les colonnes
   Vectorisé, INR, R-best et LPIPS sont des volumes lisses de la même facture que
   la source 0.1T. La vérité 7T, elle, a une tout autre nature — plis corticaux
   nets, contraste gris/blanc marqué. C'est la lecture visuelle du résultat du
   2026-09-02 : le flow déplace le niveau, il ne synthétise pas l'apparence de la
   cible.

2. **Les plafonds sont NETS.** « Plafond MedVAE » et surtout « Plafond LPIPS »
   restituent le plissement cortical presque comme la vérité, et leurs cartes
   d'erreur sont quasi noires. **La représentation n'est donc pas le facteur
   limitant** — elle sait porter le détail ; c'est le flow qui ne le produit pas.
   C'est la version visuelle du plafond à 0.1048 contre un score de 0.3737.

3. **Le zoom cortical tranche** : vérité = gyri nets, modèles = flou, plafonds =
   gyri nets. Les trois régimes sont visuellement distincts.

Les cartes d'erreur sont à **échelle commune** (indiquée au centre) : les comparer
entre colonnes est légitime. La première est le témoin identité `|source − vérité|`,
saturé — toute méthode dont la carte n'est pas nettement plus sombre ne fait pas
mieux que recopier la source.

## Les figures, par ordre d'intérêt

| fichier | ce qu'il montre |
|---|---|
| `panel_T1W_0.1T_to_7T_0006.png` | **le plus parlant.** Le saut extrême sur le contraste le plus dur, avec les deux plafonds pour référence. |
| `panel_T2W_0.1T_to_7T_0006.png`, `panel_T2FLAIR_0.1T_to_7T_0006.png` | mêmes colonnes, autres contrastes. |
| `panel_*_0.1T_to_3T_0006.png` | champ **intermédiaire** — c'est là que la trajectoire doit passer *par* 3T au lieu de couper droit, donc là où la courbure compte. |
| `translation_*.png` | les quatre figures de déplacement individuel (voir la réserve ci-dessous). |
| `calib_T1W_0.1T_to_7T_0006.png` | l'erreur d'**échelle d'intensité** — 80 % de l'énergie de l'erreur. |
| `spectres_radiaux.png` | netteté : énergie haute fréquence conservée, à lire contre le plafond 1 mm de 0.61–0.64. |

## Réserve sur les figures `translation_*`

Ces quatre figures montrent, pour les 3 sujets, la part du déplacement **propre au
sujet** (déplacement moins sa moyenne sur les sujets), en haut pour le prédit, en
bas pour le vrai, à **échelle de couleur identique**.

**Deux légendes successives ont été retirées** parce qu'elles affirmaient plus que
la figure ne montre : « la rangée du milieu est presque vide » (extrapolation de la
mesure en espace LATENT — 3.5 % contre 71.7 %) puis « contours contre niveau
régional ». Les deux sont fausses sur au moins une cellule. **Le décodeur étant non
linéaire, la mesure en espace latent ne se transpose pas simplement en espace
image.**

Ce qui est mesuré, et qui est hétérogène :

| cellule | corrélation prédit/vrai | niveau vrai / prédit |
|---|---|---|
| T1W 0.1T→7T | **+0.842** | 0.5 |
| T2W 0.1T→7T | +0.482 | 1.9 |
| T2FLAIR 0.1T→7T | +0.875 | 4.4 |
| **T1W 0.1T→3T** | **+0.042** | **8.7** |

Sur les sauts extrêmes le modèle produit une individualité **corrélée** à la
vraie ; sur la paire intermédiaire T1W 0.1T→3T la corrélation s'effondre et le
décalage de niveau manquant est d'un facteur 9. **Consigné sans hypothèse.**

## Comment lire les panneaux

- **Ligne 1** : les volumes, avec nRMSE et SSIM moyennés sur les 3 sujets.
- **Ligne 2** : erreur absolue, **échelle commune**, témoin identité en premier.
- **Ligne 3** : zoom cortical — la région où la netteté se juge.

Les valeurs affichées sont celles de la **paire et du sujet montrés**, pas les
moyennes de la campagne : sur cette cellule R-best (0.738) et LPIPS (0.762) sont
au-dessus de la production (0.698), ce qui est cohérent avec le résultat global
nul et rappelle qu'une cellule n'est pas un classement.

## Reproduction

```
PYTHONPATH=src python src/cfm/eval_qualitative_3arch.py --mode panel \
    --modality T1W --pair 0.1T_to_7T --subject 0006 \
    --with-rbest --with-lpips --with-ceiling --with-ceiling-lpips \
    --only "Vectorise" "INR" "Vectorise R-best" "Vectorise LPIPS" \
           "Plafond MedVAE" "Plafond LPIPS" \
    --outdir results/mmfm/qualitative_20260903

PYTHONPATH=src python src/cfm/figures_translation.py \
    --modality T1W --pair 0.1T_to_7T --outdir results/mmfm/qualitative_20260903
```
