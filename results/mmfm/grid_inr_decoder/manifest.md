# Décodeur implicite conditionné par grille — résultat : NÉGATIF

**Date** : 2026-08-12
**Verdict** : l'architecture fonctionne mais n'apporte rien. Le décodeur convolutif
de MedVAE suivi d'une interpolation trilinéaire reste meilleur à 0.5mm, sur les
deux métriques et sur les trois contrastes.

---

## Ce qui était testé, et pourquoi

L'INR à latent GLOBAL de ce projet est clos depuis le 2026-08-07 : sa capacité est
plafonnée par `modulation_dim` = 1536 nombres par volume, indépendamment de
`latent_dim` ET de la résolution (voir `../comparison_20260807_1mm/manifest.md`).

Le conditionnement LOCAL par grille (style LIIF) lève ce plafond : la capacité vaut
`h*w*d*feat_dim` = 48*56*48*64 = **8.3M** au lieu de 1536, et croît avec la
résolution du latent. L'architecture garde l'esprit SIREN (activations
sinusoïdales, initialisation de Sitzmann) et surtout le décodage SANS GRILLE.

**Le seul argument sérieux pour réintroduire un INR ici** était celui-là :
l'évaluateur du challenge ré-interpole les prédictions vers 0.5mm. Le décodeur
convolutif de MedVAE a un facteur de suréchantillonnage FIGÉ à x4 — depuis un
latent 1mm il produit du 1mm, puis subit une trilinéaire qui n'invente rien. Un
décodeur implicite peut être interrogé directement aux coordonnées 0.5mm.

C'est donc la comparaison `conv+trilinéaire @0.5mm` contre `INR @0.5mm`, à latent
IDENTIQUE, qui décide. Tout le reste est secondaire.

## Protocole

- **Latents** : cache existant `outputs/mmfm/latent_cache/unet/medvae_finetune_c4d1e200`
  (1mm, 48x56x48, encodeur MedVAE pré-entraîné, encodage par tuiles). **Inchangé** —
  le modèle de flow n'est pas touché, la comparaison reste équitable.
- **Supervision à 0.5mm** (grille native) alors que le latent reste à 1mm. Sans
  cela le décodeur n'apprendrait rien entre deux voxels 1mm et le test « décoder à
  0.5mm » serait perdu d'avance (voir `src/cfm/train_grid_inr_decoder.py`).
- **Validation par SUJET** : 40 sujets réservés = 76 volumes (3.9%). Un sujet a
  jusqu'à 3 acquisitions ; les disperser aurait fait fuiter son anatomie. Le
  décodeur convolutif MedVAE n'a jamais vu ces données non plus (poids
  pré-entraînés) : les deux sont logés à la même enseigne.
- **Poids EMA** (voir « pièges » ci-dessous).
- 50 000 pas au total, 0.73M paramètres.

## Résultat — 76 volumes hors échantillon, @0.5mm

| | nRMSE (fg) | nRMSE | SSIM |
|---|---|---|---|
| conv + trilinéaire | **0.1346** | **0.1354** | **0.9154** |
| INR @0.5mm         | 0.1511 | 0.1563 | 0.8836 |

L'INR perd sur les deux métriques : **-0.032 de SSIM, +12% de nRMSE**.
(À 1mm : -0.030 de SSIM, +13% de nRMSE.)

### Par contraste — l'INR perd partout

| contraste | n | Δ SSIM | Δ nRMSE |
|---|---|---|---|
| T1W     | 29 | -0.0331 | +0.0083 |
| T2FLAIR | 23 | -0.0216 | +0.0086 |
| T2W     | 24 | -0.0401 | +0.0338 |

### Par champ — LE résultat informatif : l'écart croît avec le champ

| champ | n | Δ SSIM | Δ nRMSE |
|---|---|---|---|
| 0.1T | 23 | -0.0107 | **-0.0108** |
| 1.5T | 28 | -0.0341 | +0.0204 |
| 3T   |  6 | -0.0424 | +0.0421 |
| 5T   |  9 | -0.0439 | +0.0302 |
| 7T   | 10 | **-0.0566** | +0.0404 |

À 0.1T le décodeur implicite fait jeu égal et bat même le convolutif en nRMSE.
L'écart se creuse ensuite RÉGULIÈREMENT avec l'intensité du champ. Interprétation
directe : plus le champ est élevé, plus l'image porte de détail fin, et c'est
exactement ce que ce décodeur lisse (visible dans les agrandissements de
`reconstruction_0p5mm.png`).

## Conclusion

Le décodage à résolution arbitraire ne produit AUCUN détail sous-voxel que
l'interpolation trilinéaire ne fournisse déjà. La promesse tombe, et avec elle le
dernier argument pour réintroduire un INR dans ce pipeline.

C'est la **deuxième** fois que ce constat est mesuré, sur deux conceptions
d'INR très différentes (latent global modulant un SIREN ; grille latente
conditionnant localement un SIREN). Cela rejoint Kim & Fridovich-Keil (NeurIPS
2025, arXiv:2506.11139) : à paramètres égaux, une grille bat tout INR sur les
signaux DENSES — une IRM cérébrale en est un.

**Ne pas rouvrir cette piste sans une idée structurellement différente**, et sans
une raison de penser qu'elle échappe à ce constat.

## Pièges rencontrés — à ne pas repayer

1. **EMA non chauffée.** `decay=0.9999` a une constante de temps de 10 000 pas :
   au checkpoint 5000, l'EMA contient encore **60.7%** de l'initialisation
   ALÉATOIRE et produit du bruit (SSIM 0.031) alors que les poids bruts donnent
   0.794. À 50 000 pas la contamination tombe à 0.67% et le problème disparaît.
   Évaluer une EMA avant ~3x sa constante de temps n'a pas de sens.
2. **L'EMA n'est pas cosmétique ici.** L'entraînement oscille : au même
   checkpoint, poids bruts SSIM 0.799 contre EMA 0.924 (+0.125). Sur un
   entraînement instable, l'EMA est le mécanisme correctif, pas une convention.
3. **Divergence à 15 000 pas** (gradients 3 -> 1618, loss 0.034 -> 0.26). Reprise
   à chaud depuis le checkpoint 15 000 avec lr 1e-4 -> 3e-5, cosinus, et
   **weight_decay 1e-4 -> 0** : appliquer une décroissance de poids à un SIREN
   ronge l'échelle fixée par l'initialisation de Sitzmann et change la fréquence
   effective du réseau. Après correction : gradient max 17.7 sur 35 000 pas.
4. **Sélection d'évaluation non stratifiée.** `sorted(samples)[:24]` ne retenait
   que des T1W (24/24) : T2W et T2FLAIR absents. Corrigé (tirage en tourniquet
   par (contraste, champ)) — et le défaut est passé à « tous les volumes ».
5. **Un volume de contrôle unique induit en erreur.** Sur le volume suivi pendant
   l'entraînement, l'écart de nRMSE paraissait de 1.50x ; sur 76 volumes il est de
   1.12x. Et ce volume donnait une VICTOIRE sur T2FLAIR (0.920 contre 0.898) alors
   que la moyenne des 23 T2FLAIR est une défaite (-0.0216).
6. **La loss d'entraînement, encore une fois, ne dit rien.** Elle est restée entre
   0.023 et 0.047 pendant que le SSIM dense oscillait entre 0.17 et 0.92.

## Fichiers

| chemin | contenu |
|---|---|
| `src/cfm/grid_inr_decoder.py` | architecture (+ test d'alignement `grid_sample` en `__main__`) |
| `src/cfm/train_grid_inr_decoder.py` | entraînement (+ `--self-test` : arithmétique d'indices) |
| `src/cfm/eval_grid_inr_decoder.py` | auto-reconstruction, 4 variantes, ventilation |
| `src/cfm/figures_grid_inr_decoder.py` | figure qualitative |
| `configs/mmfm/grid_inr_decoder.yaml` | configuration |
| `reconstruction_ema_full.csv` | mesures par volume (76) |
| `summary_ema_76.txt` | synthèse brute |
| `reconstruction_0p5mm.png` | figure qualitative (3 contrastes) |
| `train_metrics_run1_diverged.jsonl` | run 1, divergé à 15 000 |
| `train_metrics_run2_resume.jsonl` | run 2, reprise stabilisée |
