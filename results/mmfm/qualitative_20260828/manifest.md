# Figures qualitatives — R-best et l'auto-reconstruction

**Date** : 2026-08-28
Huit colonnes par panneau : source, vérité, les trois architectures de
production, **R-best** (le run corrigé), puis **l'auto-reconstruction MedVAE et
INR** — la vérité terrain passée dans l'encodeur/décodeur sans aucun transport.

Ces deux dernières colonnes sont les plus instructives : elles séparent à l'œil
ce que la **représentation** peut rendre de ce que le **flow** en fait.

## Quoi regarder

| fichier | ce qu'il montre |
|---|---|
| `panel_T2W_0.1T_to_3T_0006.png` | **le plus parlant.** Colonne « Plafond MedVAE » : le cortex est restitué presque parfaitement, sa carte d'erreur est quasi noire — la représentation MedVAE n'est pas le facteur limitant. Colonne « Plafond INR » : un aplat gris presque sans structure, alors que la colonne « INR » d'à côté montre au moins les ventricules. **L'ajustement INR est plus plat que ce que son propre flow produit.** |
| `panel_T1W_0.1T_to_7T_0006.png` | Le saut extrême sur le contraste le plus dur. C'est là que R-best est le MOINS bon (courbure restaurée à 0.498 seulement, nRMSE +0.0088) — à regarder pour comprendre ce qui résiste. |
| `panel_T2W_0.1T_to_7T_0006.png` | Le cas où R-best gagne le plus (−0.0195, 15/20 paires, courbure 0.965). Comparer sa colonne à « Vectorise ». |
| `panel_T2FLAIR_0.1T_to_3T_0006.png` et `panel_T2FLAIR_0.1T_to_7T_0006.png` | Cas intermédiaire (courbure 0.864). |
| `panel_T1W_0.1T_to_3T_0006.png` | Champ INTERMÉDIAIRE : c'est là que la courbure de la trajectoire compte le plus, puisqu'il faut passer PAR 3T au lieu de couper droit de 0.1T à 7T. |
| `calib_T1W_0.1T_to_7T_0006.png` | Erreur d'ÉCHELLE d'intensité — 80 % de l'erreur totale d'après la mesure du 2026-08-25. Le vrai sujet restant. |
| `spectres_radiaux.png` | Netteté : énergie haute fréquence conservée par rapport à la vérité. Plafond 1 mm de 0.61-0.64, à ne pas confondre avec une limite de modèle. |

## Comment lire les panneaux

- **Ligne 1** : les volumes, avec nRMSE/SSIM moyennés sur les 3 sujets.
- **Ligne 2** : cartes d'erreur absolue, **échelle commune** à toutes les
  colonnes (indiquée au centre) — les comparer entre elles est donc légitime.
  La première carte est le témoin identité `|source − vérité|` : toute méthode
  dont la carte n'est pas nettement plus sombre ne fait pas mieux que recopier
  la source.
- **Ligne 3** : zoom cortical, la région où la netteté se juge.

## Réserve

Les colonnes de plafond sont rangées par paire IDENTITÉ ; elles ne dépendent que
du champ CIBLE, pas de la paire affichée. C'est voulu (aucun transport n'a lieu),
et `pred_path` fait la correspondance. Elles n'ont pas de LPIPS : le script de
scoring de l'auto-reconstruction ne calcule que nRMSE et SSIM.

## Reproduction

```
PYTHONPATH=src python src/cfm/eval_qualitative_3arch.py --mode panel \
    --modality T2W --pair 0.1T_to_3T --subject 0006 \
    --with-rbest --with-ceiling --outdir results/mmfm/qualitative_20260828
```
