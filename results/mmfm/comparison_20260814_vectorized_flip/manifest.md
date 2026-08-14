# Vectorisé réentraîné AVEC l'augmentation par flip — le classement publié est confirmé

**Date** : 2026-08-14
**Verdict** : le flip est neutre sur cette architecture (0.4353 contre 0.4354).
La comparaison vectorisé/UNet, désormais à augmentation STRICTEMENT identique,
donne exactement le même écart qu'avant. **Ce checkpoint devient la référence de
production** — non pour sa performance, égale, mais parce qu'il est le seul
reproductible depuis sa config.

---

## Pourquoi ce run existe

L'audit du 2026-08-13 (`../comparison_20260813_unet_adagn/manifest.md`) a trouvé
un vrai bug d'équité : `flip_lr_prob: 0.5` était déclaré dans les TROIS configs
mais seul l'UNet le recevait. `FlatLatentCacheDataset` n'acceptait pas cet
argument et `arch_vector`/`arch_inr` ignoraient `data_cfg` — le paramètre était
avalé sans erreur. Mesuré : un latent retourné diffère de **26 %** en L2 relatif,
ce n'est pas une quasi-identité.

L'affirmation « configs identiques hors du bloc `model:`, donc seule
l'architecture diffère » était donc FAUSSE. Restait à savoir si le classement en
dépendait.

## Correctif

- `common/dataset.py::FlatLatentCacheDataset` accepte `flip_lr_prob`/`flip_axis`
  et un `latent_shape` OBLIGATOIRE dès que le flip est actif : le vecteur étant
  un latent spatial aplati, il faut le remettre en forme pour le retourner.
- `arch_vector` transmet les trois via une fermeture (le cache plat n'enregistre
  que `flat_dim`, pas la forme spatiale).
- `arch_inr` REFUSE le flip et lève une erreur : son latent est le `z` ajusté par
  l'Algorithme 2 de NOIR, un vecteur de modulation GLOBAL sans structure
  spatiale. Le retourner n'a aucun sens géométrique — il faudrait réajuster `z`
  sur les volumes retournés, donc refaire le precompute (~6 h).
  `configs/mmfm/inr.yaml` passe à `flip_lr_prob: 0.0`, justifié sur place.
- Quatre garde-fous, tous testés : flip sans `latent_shape`, `latent_shape`
  incohérent avec `flat_dim`, `flip_axis` hors bornes, flip demandé sur l'INR.

**Équivalence vérifiée** : à `flip_lr_prob=1.0`, le cache PLAT et le cache
SPATIAL produisent des tenseurs `torch.equal` sur 6 volumes et sur les 3 axes.
Le vectorisé reçoit donc EXACTEMENT l'augmentation de l'UNet, pas une variante.

## Résultat — le flip est neutre

| | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| vectorisé SANS flip (ancienne réf.) | 0.4354 | 0.8995 | 0.0985 |
| vectorisé AVEC flip | **0.4353** | **0.8997** | **0.0983** |

-0.0001 nRMSE, 13/20 paires. Compensation quasi parfaite entre les cibles
faciles (0.1T à 3T : +0.0005 à +0.0013) et les cibles dures (5T -0.0011,
7T -0.0026) — le flip aide marginalement là où c'est difficile, gêne
marginalement ailleurs.

## La comparaison qui compte — à augmentation identique

| | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| UNet | 0.4617 | 0.8955 | 0.1014 |
| **vectorisé + flip** | **0.4353** | **0.8997** | **0.0983** |

-0.0264 nRMSE, **16/20 paires**. À comparer aux -0.0263 et 16/20 mesurés AVANT
correction : l'écart est inchangé au millième près.

**Le classement publié ne dépendait pas du bug.** C'est désormais mesuré, plus
supposé — et c'est le seul intérêt réel de ce run, la performance étant
identique.

Note : le bug handicapait le GAGNANT (le vectorisé s'entraînait sans
augmentation et gagnait quand même), donc la conclusion était conservatrice.
C'est ce qui rendait un renversement improbable — mais improbable n'est pas
mesuré.

## Sur la loss, une cinquième fois

La loss d'entraînement avec flip est restée ~2 points AU-DESSUS de celle sans
flip (13.1 contre 11.3 en médiane sur les premières tranches) pour un résultat
Task 3 identique. C'est attendu — l'augmentation rend l'ajustement plus dur —
mais cela confirme encore que sur ce pipeline la loss ne prédit rien.

## Décision : ce checkpoint devient la référence

`outputs/mmfm/vectorized/weights/model_final.pth` (0.4354, 6.9 Go) est REMPLACÉ
par le nouveau. Les deux sont équivalents en performance ; le nouveau est le seul
**reproductible depuis sa config**, l'ancien ayant été produit sous un
`flip_lr_prob` qui ne s'appliquait pas.

CSV de référence Task 3 : `task3_vectorized_flip_T1W.csv` (ce dossier).
L'ancien reste consultable dans `../comparison_20260807_1mm/`.

## Fichiers

| chemin | contenu |
|---|---|
| `task3_vectorized_flip_T1W.csv` | résultat Task 3 (20 paires x 3 sujets, T1W) |
| `comparaison_flip_vs_sans.txt` | flip contre sans flip, ventilé par champ |
| `comparaison_vs_unet.txt` | contre l'UNet, augmentation identique |
| `chaine_complete.txt` | sortie brute de la chaîne inférence -> évaluation |
