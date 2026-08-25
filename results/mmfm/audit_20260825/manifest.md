# Audit de la chaîne MMFM — phase A (falsifieurs, sans réentraînement)

**Date** : 2026-08-25
**Objet** : l'INR fait pire que ne rien faire (nRMSE 0.6383 contre un témoin
identité à 0.9273 sur T1W, et 0.6470 / 0.5998 contre 0.3859 / 0.5574 sur T2W /
T2FLAIR). Vérifier chaque étage plutôt que chercher une hypothèse de plus.

**Principe** : le vectorisé sert de témoin positif. Les trois architectures
partagent `mmfm_core.py`, le dataloader, le sampler, le couplage OT-CFM, la loss
et l'inférence ; seuls `prep_latent`/`restore_latent` diffèrent. Le premier étage
où elles divergent localise la faute.

---

## Résultat préalable : la représentation n'est PAS en cause

Aucun CSV existant ne mesurait le backbone de **production** à 1 mm — les chiffres
publiés datent de 2 mm avec `latent_dim=4096`, ou d'un budget de fit 8× inférieur.
Mesuré directement, fit puis décodage, **sans flow** :

| T1W, sujet 0006, 1 mm | nRMSE | SSIM | référence z=0 (« cerveau moyen ») |
|---|---|---|---|
| 0.1T | **0.1409** | 0.8551 | 0.3719 / 0.7703 |
| 3T | **0.1824** | 0.8035 | 0.4010 / 0.7528 |
| 7T | **0.2805** | 0.7791 | 0.4989 / 0.7492 |
| volume du cache (`R_T1W_3T_0329`) | 0.2596 | 0.7915 | — |

Décoder le latent **du cache** reproduit un fit frais à 0.0001 près (cos 0.9987) :
le cache est sain et le décodeur produit de vrais cerveaux.

**Budget d'erreur** : plancher de représentation ~0.18–0.28, résultat Task 3
0.6383. **Les ~0.38 restants sont produits en aval.**

## A1 — L'EMA sans correction de biais : le défaut dominant

`common/distributed.py:108-118` initialise les poids fantômes sur
l'initialisation **aléatoire** et n'applique aucune correction de biais.
`0.9999^25000 = 0.082` : **8.2 % du bruit initial survit** dans les poids servis
à l'inférence (`use_ema=True` par défaut, `infer_mmfm_unified.py:391,518`).

Pour le latent INR — écart-type par élément 1.92e-4 — ce résidu vaut **216× le
signal**. Test : la même inférence avec `--no_ema`, aucune autre modification.

| INR, T1W, 20 paires × 3 sujets | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| avec EMA (chiffre publié) | 0.6383 | 0.7966 | 0.2061 |
| **sans EMA** | **0.4430** | **0.8517** | **0.1617** |

**19 paires sur 20 améliorées.** Ventilé par champ cible :

| cible | avec EMA | sans EMA | gain |
|---|---|---|---|
| 0.1T | 0.5926 | **0.2763** | 53.4 % |
| 1.5T | 0.5901 | **0.3541** | 40.0 % |
| 3T | 0.5835 | **0.3477** | 40.4 % |
| 5T | 0.6486 | **0.5029** | 22.5 % |
| 7T | 0.7765 | 0.7342 | 5.4 % |

Le gain décroît avec la difficulté, ce qu'on attend d'une corruption additive
quasi constante : elle domine là où le signal utile est le plus faible.

### Ce que ce seul drapeau change au classement du projet

| T1W | nRMSE |
|---|---|
| Vectorisé | 0.4353 |
| **INR sans EMA** | **0.4430** |
| UNet | 0.4617 |
| *Témoin identité* | *0.9273* |

**L'INR passe de dernière à deuxième, devant l'UNet.** La conclusion publiée
« vectorisé > UNet > INR dans les neuf cellules » mesurait un bug d'EMA, pas une
propriété d'architecture. Et le profil par champ de l'INR corrigée
(0.276 / 0.354 / 0.348 / 0.503 / 0.734) épouse celui du vectorisé
(0.271 / 0.331 / 0.315 / 0.506 / 0.754) — les deux architectures se comportent
désormais pareil.

## A2–A4 — Dérive du latent entre entraînement et inférence

Le flow est entraîné sur un cache et ne rappelle jamais `prep_latent`
(`cache_prebakes_prep=True`) ; l'inférence le rappelle en direct. Rien, côté
entraînement, ne peut détecter la dérive. Mesurée sur 10 volumes du cache
(`src/cfm/audit_inr_latent.py`), L2 relatif contre le `z` du cache :

| variante | L2 relatif | cosinus | lecture |
|---|---|---|---|
| **inférence actuelle** | **0.691** | 0.794 | ce que le flow reçoit aujourd'hui |
| orientation alignée | 0.305 | 0.926 | −56 % |
| + normalisation alignée | **0.069** | 0.997 | −90 % |
| reproduction du precompute (fp32) | 0.044 | 0.999 | plancher |

Le plancher de 0.044 n'est pas nul parce que `inr_backbone.sample_points` tire
ses points avec `generator=None` : **le fit n'est pas déterministe**.

Les deux écarts, à leur source :

1. **Orientation.** Le precompute rééchantillonne avec `scipy.ndimage.zoom` sur le
   tableau brut et garde l'orientation native **LAS** ; l'inférence
   (`infer_mmfm_unified.py:277`, `nibabel.processing.resample_to_output`)
   réoriente en canonique **RAS**, axe 0 inversé. Vérifié sur un volume : écart
   0.343 tel quel, 0.057 après retournement de l'axe 0.
2. **Normalisation.** Le cache de production `inr_932b0b37` porte
   `field_norm_stats_path: None` — percentiles **par volume**. L'inférence tourne
   en `field_fixed`. Les caches vectorisé et UNet portent, eux, bien
   `configs/mmfm/field_norm_stats.json`.

### Pourquoi l'orientation ne tue que l'INR

`flip_lr_prob: 0.5` rend le vectorisé et l'UNet insensibles au miroir : le bug
leur est invisible. `arch_inr.py:105-111` **refuse** le flip — à raison, un
vecteur de modulation global n'a pas de structure spatiale à retourner. L'INR est
donc la seule architecture sans tolérance au miroir, et elle en reçoit un à chaque
inférence. **L'augmentation masquait le bug pour les deux autres depuis le début.**

## A5 — Le témoin : le même défaut ne touche pas le vectorisé

| Vectorisé, T1W | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| avec EMA (production publiée) | 0.4353 | 0.8997 | 0.0983 |
| sans EMA | 0.4359 | 0.8993 | 0.0984 |

**Écart +0.0006 — du bruit.** C'est le contrôle qui valide le mécanisme : mêmes
poids fantômes, même absence de correction de biais, mais un latent d'écart-type
18.1 au lieu de 1.92e-4. Le résidu d'EMA est une corruption d'amplitude
**absolue** : il vaut 216× le signal INR et une fraction négligeable du signal
MedVAE.

Conséquence pratique : **les chiffres publiés du vectorisé et de l'UNet ne sont
pas à revoir de ce fait** — contrairement à ce que le périmètre « les trois
architectures » laissait craindre. Le correctif d'EMA reste à appliquer pour la
correction, mais il ne change pas leurs résultats.

## A7 — Les trois correctifs ensemble, de bout en bout

Drapeau d'audit `--compat_precompute` (`infer_mmfm_unified.py:309,369`) : retourne
l'axe 0 avant le fit et le remet à l'endroit sur la prédiction — ce qui préserve
la comptabilité d'affine de nibabel, le precompute n'en maintenant pas — et
normalise la source par ses propres percentiles. La dénormalisation de sortie
reste celle du champ cible, seule échelle disponible à l'inférence. Combiné à
`--no_ema`.

| T1W, 20 paires × 3 sujets | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| *Témoin identité* | *0.9273* | *0.8699* | — |
| INR publiée (avec EMA) | 0.6383 | 0.7966 | 0.2061 |
| INR sans EMA (A1) | 0.4430 | 0.8517 | 0.1617 |
| **INR + les trois correctifs (A7)** | **0.3787** | 0.8618 | 0.1568 |
| UNet | 0.4617 | 0.8955 | 0.1014 |
| Vectorisé | 0.4353 | **0.8997** | **0.0983** |

**16 paires sur 20 améliorées** par rapport à A1. Par champ cible, l'INR corrigée
passe devant le vectorisé **partout** :

| cible | publiée | A1 | **A7** | vectorisé |
|---|---|---|---|---|
| 0.1T | 0.5926 | 0.2763 | **0.2108** | 0.2709 |
| 1.5T | 0.5901 | 0.3541 | **0.3095** | 0.3306 |
| 3T | 0.5835 | 0.3477 | **0.3018** | 0.3146 |
| 5T | 0.6486 | 0.5029 | **0.4021** | 0.5061 |
| 7T | 0.7765 | 0.7342 | **0.6691** | 0.7542 |

**Y compris à 7T**, le champ dont ce projet a conclu qu'aucune architecture n'y
avait de force propre.

### Vérification : l'INR corrigée fait-elle un vrai travail ?

Le flow INR a été mesuré « 2.7× pire que prédire zéro » sur sa cible de vitesse.
Il fallait donc écarter l'hypothèse que le bon score vienne de la seule
dénormalisation. Corrélation de Pearson, T1W sujet 0006 :

| paire | A7 ~ cible | A7 ~ source | vectorisé ~ cible | vectorisé ~ source |
|---|---|---|---|---|
| `3T_to_7T` | 0.913 | 0.975 | 0.926 | 0.992 |
| `0.1T_to_7T` | 0.921 | 0.959 | 0.923 | 0.997 |
| `7T_to_0.1T` | **0.960** | 0.918 | 0.952 | 0.982 |
| `3T_to_0.1T` | 0.980 | 0.974 | 0.987 | 0.990 |

Même régime que le vectorisé, et sur `7T_to_0.1T` l'INR s'éloigne **davantage** de
sa source que le vectorisé. Elle n'est pas une identité déguisée.

## Conclusions de la phase A

1. **Le classement publié mesurait des bugs, pas des architectures.** L'INR passe
   de 0.6383 (dernière, battue par « ne rien faire » sur deux contrastes) à
   **0.3787**, meilleure des trois en nRMSE sur T1W — sans réentraînement, par
   trois correctifs d'inférence.
2. **Mais le classement est désormais dépendant de la métrique** : l'INR gagne en
   nRMSE (0.3787 contre 0.4353) et perd nettement en SSIM (0.8618 contre 0.8997)
   et en LPIPS (0.1568 contre 0.0983). C'est le compromis perception/distorsion :
   son goulot de 1536 modulations la rend plus lisse, ce qui flatte l'erreur
   quadratique et pénalise la structure. **Aucune des deux n'est « la meilleure »
   sans préciser la métrique** — conclusion bien plus nuancée que « vectorisé >
   UNet > INR dans les neuf cellules ».
3. **Le vectorisé et l'UNet ne sont pas affectés par le défaut d'EMA** (A5 :
   +0.0006, du bruit). Leurs chiffres publiés tiennent.
4. **Ces chiffres sont un plancher pour l'INR** : le flow n'a jamais été
   réentraîné sous le prétraitement corrigé, et il reste numériquement aveugle à
   son entrée (`VectorMMFM` n'applique aucune normalisation d'entrée ; changer de
   sujet ne change la vitesse que de 0.43 %). Les correctifs de fond restent
   entièrement devant.

## Ce qui reste à faire (phase B, non engagée)

| # | correctif | coût |
|---|---|---|
| B1 | un seul chemin de prétraitement partagé, avec assertion d'orientation | 2 h |
| B2 | normaliser le latent en entrée du flow | 1 h + réentraînement ~2 h |
| B3 | correction de biais sur l'EMA (`shadow / (1 - decay^step)`) | 30 min |
| B4 | graîner `sample_points` (le fit n'est pas déterministe, plancher 0.044) | 15 min |
| B5 | régénérer le cache INR sous le prétraitement corrigé | ~6 h GPU |
| B6 | réévaluer l'INR corrigée sur T2W et T2FLAIR | 1 h |

## Fichiers

| chemin | contenu |
|---|---|
| `task3_inr_noema_T1W.csv` | A1 — INR sans EMA |
| `task3_inr_compat_T1W.csv` | A7 — les trois correctifs |
| `task3_vectorized_noema_T1W.csv` | A5 — témoin |
| `latent_drift.csv` | A2–A4 — dérive du latent par variante |
| `src/cfm/audit_inr_latent.py` | l'outil de mesure de la dérive |
