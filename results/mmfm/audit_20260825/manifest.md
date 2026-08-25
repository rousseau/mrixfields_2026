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

---

# Phase B — correctifs de fond (2026-08-25/26)

Quatre correctifs, chacun vérifié séparément.

## B3 — EMA avec échauffement : validé par un contrôle direct

`common/distributed.py::EMAModel.current_decay` : la décroissance démarre à
`(1+t)/(10+t)` et rejoint `decay`. Le résidu d'initialisation aléatoire disparaît
en quelques centaines de pas au lieu de survivre à tout l'entraînement.
`ema_num_updates` est sauvegardé au checkpoint pour qu'une reprise ne
réinitialise pas l'échauffement.

| modèle réentraîné, T1W | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| avec EMA (échauffée) | 0.4070 | 0.8606 | 0.1572 |
| sans EMA | 0.4055 | 0.8603 | 0.1575 |

**Écart 0.0015 — du bruit.** Avant correctif, le même écart valait **0.195**
(0.6383 contre 0.4430). L'EMA est redevenue utilisable.

## B4 — Tirage de points graîné

`inr_backbone.sample_points` tirait avec `generator=None` : deux ajustements du
même volume différaient de 4.4 % en L2 relatif, ce qui constituait le plancher
irréductible de l'audit. La graine est dérivée du contenu (taille du nuage,
budget de points), donc indépendante de l'ordre d'appel. Déterminisme vérifié.

## B1 — Prétraitement aligné, et le garde-fou qui manquait

Le drapeau d'audit est devenu deux clés de config orthogonales,
`inference.compat_orientation` et `inference.compat_source_norm`, activées dans
`configs/mmfm/inr.yaml` avec la mesure en commentaire.

**Décision d'architecture** : aligner l'inférence sur le precompute, et non
l'inverse. Réorienter le precompute en canonique serait plus propre mais imposerait
de régénérer les trois caches et invaliderait le vectorisé et l'UNet.

Surtout, `_check_cache_consistency` lit l'`index.json` du cache et **refuse de
tourner** si la normalisation d'inférence contredit celle qui a servi à le bâtir.
Testé dans les deux sens. C'est le contrôle qui manquait : `cache_prebakes_prep=True`
fait que l'entraînement ne rappelle jamais `prep_latent`, donc les deux chemins ne
se croisent nulle part et rien ne pouvait signaler leur dérive.

### Contrôle : le vectorisé gagne-t-il à ne plus tourner mirroré ?

| Vectorisé, T1W | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| production (miroir toléré) | 0.4353 | 0.8997 | 0.0983 |
| orientation alignée | 0.4351 | 0.8998 | 0.0984 |

**Neutre (−0.0002).** L'augmentation par flip l'avait réellement rendu invariant.
**Les chiffres publiés du vectorisé et de l'UNet n'ont pas à être révisés** — ce
que le périmètre « les trois architectures » laissait craindre.

## B2 — Standardisation de l'entrée du flow

`VectorMMFM` ne normalisait pas ses entrées : les `LayerNorm` sont toutes APRÈS
`input_proj`, donc elles normalisent la somme projetée et ne peuvent pas rattraper
un écart d'échelle. Ajout de `latent_mean`/`latent_scale` en tampons **non
persistants** (les checkpoints existants se chargent inchangés) avec défauts
`(0, 1)` strictement sans effet — vérifié, ainsi que l'équivariance d'échelle.

Mesuré sur le cache : moyenne −1.4e-07, écart-type **2.16e-04**, dispersion par
dimension de 2.9× seulement — un scalaire suffit.

### La loss d'entraînement, enfin

| | prédire zéro | meilleure constante | loss finale | min |
|---|---|---|---|---|
| ancien run | 4.02e-4 | 3.88e-4 | **1.10e-3** (2.7× PIRE) | — |
| **standardisé** | 4.02e-4 | 3.88e-4 | **3.37e-4** | **2.92e-4** |

Le flow passe de « 2.7× pire que prédire zéro » à 16 % sous la meilleure
constante. C'est une reparamétrisation exacte : elle ne change pas la classe de
fonctions, elle remet l'optimisation dans un régime où `lr=2e-5` a un sens.

## B6 — Le tableau final, trois architectures x trois contrastes

### nRMSE

| | T1W | T2W | T2FLAIR | moyenne |
|---|---|---|---|---|
| *Identité* | *0.9273* | *0.3859* | *0.5574* | *0.6235* |
| INR publiée (avant audit) | 0.6383 | 0.6470 | 0.5998 | 0.6284 |
| INR phase A (correctifs d'inférence seuls) | **0.3787** | 0.4296 | 0.3391 | 0.3824 |
| **INR phase B (réentraînée standardisée)** | 0.4070 | 0.3800 | **0.3376** | **0.3749** |
| Vectorisé | 0.4353 | **0.3376** | 0.3654 | 0.3794 |
| UNet | 0.4617 | 0.3676 | 0.3807 | 0.4033 |

### SSIM

| | T1W | T2W | T2FLAIR | moyenne |
|---|---|---|---|---|
| *Identité* | *0.8699* | *0.9084* | *0.8864* | *0.8882* |
| INR publiée | 0.7966 | 0.7576 | 0.8028 | 0.7856 |
| INR phase B | 0.8606 | 0.8579 | 0.8708 | 0.8631 |
| **Vectorisé** | **0.8997** | 0.8980 | **0.8949** | **0.8975** |
| UNet | 0.8955 | 0.8963 | 0.8929 | 0.8949 |

### LPIPS

| | T1W | T2W | T2FLAIR | moyenne |
|---|---|---|---|---|
| INR publiée | 0.2061 | 0.2189 | 0.2034 | 0.2095 |
| INR phase B | 0.1572 | 0.1592 | 0.1507 | 0.1557 |
| **Vectorisé** | **0.0983** | **0.0911** | **0.0927** | **0.0941** |
| UNet | 0.1014 | 0.0919 | 0.0927 | 0.0953 |

## Conclusions de la phase B

1. **L'INR devient la meilleure architecture en nRMSE** (0.3749 contre 0.3794 pour
   le vectorisé) et **reste nettement la dernière en SSIM et LPIPS** (0.8631 contre
   0.8975 ; 0.1557 contre 0.0941). Le classement du projet n'est plus un ordre
   total : il dépend de la métrique. Son goulot de 1536 modulations la rend plus
   lisse, ce qui flatte l'erreur quadratique et pénalise structure et perception.
2. **Le réentraînement aide en moyenne mais pas sur T1W** : phase B 0.3749 contre
   phase A 0.3824, alors que sur T1W seul la phase A gagne (0.3787 contre 0.4070).
   T1W est le contraste atypique, comme déjà établi le 2026-08-14.
3. **Sixième occurrence, la plus nette : la loss d'entraînement ne prédit rien.**
   Elle a été divisée par 3.3 (1.10e-3 → 3.37e-4) et a franchi le seuil du
   prédicteur constant ; le score T1W s'est **dégradé** (0.3787 → 0.4070). Dans un
   espace latent à faible structure commune, un flow qui bouge davantage prend
   plus de risques qu'il n'en gagne.
4. **Le vrai plafond restant est représentationnel, et il est mesuré** :

   | | énergie portée par la moyenne commune | dispersion du nuage |
   |---|---|---|
   | MedVAE | **91.3 %** | 0.416 (amas serré) |
   | INR | **25.1 %** | 1.221 (quasi isotrope) |

   Le latent INR est un nuage presque isotrope autour de zéro : chaque volume
   pointe dans une direction quasi aléatoire, conséquence d'ajuster chaque volume
   indépendamment depuis `z = 0` en 20 pas. Il y a intrinsèquement peu de structure
   commune à exploiter — ce n'est plus un bug, c'est la géométrie du latent.
   **Piste suivante s'il y en a une** : canoniser l'ajustement (initialiser `z`
   depuis un prior appris ou depuis le latent source) pour créer de la structure
   partagée. C'est une modification du precompute (~6 h).

## Décision : quel checkpoint devient la référence INR

`outputs/mmfm/inr_std/weights/model_final.pth` (phase B). Non pour sa marge sur
T1W — la phase A y fait mieux — mais parce qu'il est **meilleur en moyenne sur les
trois contrastes** et surtout **le seul reproductible depuis sa config** : la
phase A dépend d'un ancien checkpoint plus un `--no_ema` qui n'existe que pour
contourner un bug désormais corrigé. Même raisonnement que pour la promotion du
checkpoint vectorisé avec flip le 2026-08-14.

## Ce qui reste ouvert

- **B5, régénérer le cache INR** sous le prétraitement corrigé (~6 h) : non fait.
  Le plancher de dérive résiduel (4.4 %) vient du tirage non graîné de l'ancien
  cache ; il disparaîtra à la prochaine régénération, désormais déterministe.
- **La géométrie du latent** (point 4) : la seule piste à fort levier restante.
- **Le garde-fou permanent** : `test_inr_backbone_smoke.py` assère toujours
  `nrmse_fg < 0.6` à 2 mm. À remplacer par un test à la résolution de production,
  avec un seuil dérivé des mesures, et un test qui compare la loss finale à
  « prédire zéro ». Ce dernier tient en trois lignes et aurait tout arrêté.
