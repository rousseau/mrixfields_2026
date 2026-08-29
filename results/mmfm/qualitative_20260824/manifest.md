# Évaluation qualitative et quantitative des trois architectures — versions de production

> # ⚠ CE DOCUMENT EST PARTIELLEMENT PÉRIMÉ (annoté le 2026-08-26)
> **Tout ce qui concerne l'INR décrit un modèle qui n'existe plus.** L'audit des
> 2026-08-25/26 (`../audit_20260825/manifest.md`) a trouvé que ses chiffres
> mesuraient **trois bugs d'inférence** — EMA sans correction de biais, orientation
> LAS→RAS, normalisation — et non une limite d'architecture. Après correction,
> l'INR passe de 0.6383 à **0.4070** sur T1W et de 0.6284 à **0.3749** en moyenne
> sur les trois contrastes : elle devient **première en nRMSE** au lieu de dernière.
>
> Sont périmés ici : toutes les figures montrant l'INR en blobs, sa colonne dans
> les tableaux, sa part d'erreur d'échelle (20-24 %), son indice de netteté, et
> les tests appariés vectorisé/INR (58/60 paires — cet écart s'est effondré).
> L'évaluation refaite sur le modèle corrigé est dans
> `../qualitative_20260826/manifest.md`.
>
> **Restent valides et non remis en cause** : tout ce qui concerne le vectorisé et
> l'UNet (leurs chiffres ont été explicitement re-contrôlés — orientation alignée :
> 0.4351 contre 0.4353, EMA : 0.4359 contre 0.4353, tous deux neutres), le témoin
> identité, les planchers d'écrêtage et d'interpolation, le constat sur n = 3, et
> la découverte que 80 % de l'erreur du vectorisé est une erreur d'échelle.
>
> Ce document n'est pas effacé : il porte ce qu'on croyait, et l'écart entre les
> deux dates est lui-même le résultat.

**Date** : 2026-08-25
**Objet** : évaluer les trois approches *dans leur dernière version*, sur les trois
contrastes, en regardant enfin les images et pas seulement les moyennes.
**Verdict court** : le classement vectorisé > UNet > INR tient, mais il est bien
plus fragile qu'annoncé, et surtout **l'erreur dominante des deux meilleures
architectures n'est pas une erreur d'anatomie : c'est une erreur d'échelle
d'intensité**, qui représente 80 % de l'énergie de l'erreur en T1W et en T2FLAIR.
Sa part systématique est récupérable sans réentraîner : **0.3794 → 0.3231**.

**Les cinq résultats, dans l'ordre d'importance :**

1. 80 % de l'énergie de l'erreur part avec UN scalaire par volume (§4) ; la part
   systématique en est récupérable par une constante de recalibration par paire,
   estimée sans oracle : score moyen 0.3794 → **0.3231**, −15 %, sans
   réentraînement (§4bis).
2. Face au témoin « ne rien faire », le gain paire-à-paire du vectorisé vaut
   +34 % en T1W, **+5.6 % en T2W et −1.2 % en T2FLAIR** — battu sur 11 paires
   sur 20 (§3).
3. Le « mur du 7T » en T1W vient à **46 %** d'un seul volume aberrant (§5).
4. L'écart vectorisé/UNet est plus petit que le bruit inter-sujet et n'est
   significatif que sur T1W (§2) — et **n = 3 est un plafond imposé par les
   données**, pas une négligence : il n'existe que 3 sujets appariés dans tout le
   jeu (§9).
5. Les deux bonnes architectures ne restituent qu'**un tiers** de la finesse de
   la vérité ; c'est le seul défaut sur lequel une meilleure représentation agit
   (§7).

---

## 0. Ce qui a été évalué, et deux corrections d'inventaire

| approche | checkpoint évalué | prédictions |
|---|---|---|
| Vectorisé | `outputs/mmfm/vectorized/weights/model_final.pth` (2026-08-13, avec flip) | `vectorized_flip/…/T1W`, `vectorized/…/{T2W,T2FLAIR}` |
| UNet | `outputs/mmfm/unet/weights/model_final.pth` (référence, PRE-AdaGN) | `unet/…/{T1W,T2W,T2FLAIR}` |
| INR | `outputs/mmfm/inr/weights/model_final.pth` (`latent_dim = 129024`) | `inr/…/{T1W,T2W,T2FLAIR}` |

Deux points relevés en établissant cet inventaire, tous deux vérifiés sur pièce :

1. **Le chiffre INR/T1W publié dans le tableau 3x3 du 2026-08-15 n'était pas celui
   du checkpoint de production.** Ce tableau annonce 0.6223, valeur du run
   `latent_dim = 4096` ; or `arch_meta` du checkpoint de production donne
   `latent_dim = 129024`, et les prédictions présentes sur disque (2026-08-11)
   redonnent **0.6383** au recalcul. La ligne T1W du tableau mélangeait donc deux
   modèles INR différents, les colonnes T2W/T2FLAIR ayant bien été produites avec
   le checkpoint 129k. Corrigé ici. **Le classement n'en dépend pas** (l'INR reste
   dernier avec plus de 0.19 d'écart), mais la comparaison n'était pas
   celle qu'elle annonçait.
2. **Les prédictions T1W du vectorisé sous `outputs/mmfm/vectorized/` datent du
   2026-08-09**, donc du checkpoint PRE-correction du flip. Celles du modèle de
   production sont sous `outputs/mmfm/vectorized_flip/`. Les deux scripts écrits
   ici déclarent donc leurs racines de prédiction **par modalité**, avec le motif
   documenté sur place.

Sur le UNet : la variante AdaGN (2026-08-13) est plus récente en date, mais elle a
été mesurée neutre à légèrement négative (0.4623 contre 0.4617, 3/20 paires) et
n'a pas été retenue. La « dernière version » du UNet au sens de ce qui est publié
reste le checkpoint de référence, et c'est lui qui est évalué ici. L'AdaGN n'a
jamais été évalué hors T1W ; vu que ses prédictions diffèrent de 0.2-0.3 % en L2
de celles de la référence, l'écart attendu sur T2W/T2FLAIR est du même ordre que
le bruit, mais ce n'est pas mesuré.

## 1. Quantitatif — le tableau, avec le témoin « ne rien faire »

Le témoin identité (recopier le volume source sous le nom de la cible) n'existait
que pour T1W. Il est ici calculé sur les trois contrastes, avec les **formules
exactes** de l'évaluateur officiel — la réimplémentation est vérifiée
bit-à-bit contre le CSV officiel T1W avant tout calcul (écart max 2.2e-16 ; le
script s'arrête au-delà de 5e-4).

### nRMSE (plus bas = mieux)

| | T1W | T2W | T2FLAIR | moyenne |
|---|---|---|---|---|
| *Identité (ne rien faire)* | *0.9273* | *0.3859* | *0.5574* | *0.6235* |
| **Vectorisé** | **0.4353** | **0.3376** | **0.3654** | **0.3794** |
| UNet | 0.4617 | 0.3676 | 0.3807 | 0.4033 |
| INR | 0.6383 | 0.6470 | 0.5998 | 0.6284 |

### SSIM (plus haut = mieux)

| | T1W | T2W | T2FLAIR | moyenne |
|---|---|---|---|---|
| *Identité* | *0.8699* | *0.9084* | *0.8864* | *0.8883* |
| **Vectorisé** | **0.8997** | 0.8980 | **0.8949** | **0.8975** |
| UNet | 0.8955 | 0.8963 | 0.8929 | 0.8949 |
| INR | 0.7966 | 0.7576 | 0.8028 | 0.7856 |

### LPIPS (plus bas = mieux)

| | T1W | T2W | T2FLAIR | moyenne |
|---|---|---|---|---|
| **Vectorisé** | **0.0983** | **0.0911** | **0.0927** | **0.0941** |
| UNet | 0.1014 | 0.0919 | 0.0927 | 0.0953 |
| INR | 0.2061 | 0.2189 | 0.2034 | 0.2095 |

**Premier fait qui saute aux yeux** : en SSIM, sur T2W, le témoin identité
(0.9084) **bat les trois architectures**. Copier la source donne une meilleure
similarité structurelle que n'importe lequel des modèles.

## 2. Le classement vectorisé > UNet est réel mais faible — et il ne tient que sur T1W

Jusqu'ici l'argument était « neuf cellules, neuf fois le même ordre ». C'est vrai
sur les moyennes. Paire par paire, c'est beaucoup moins net :

| | victoires nRMSE | victoires SSIM | victoires LPIPS | écart nRMSE moyen |
|---|---|---|---|---|
| Vectorisé vs UNet, T1W | 16/20 | 17/20 | 17/20 | −0.0264 |
| Vectorisé vs UNet, T2W | 12/20 | 12/20 | 12/20 | −0.0300 |
| Vectorisé vs UNet, T2FLAIR | **10/20** | 12/20 | 12/20 | −0.0153 |

Tests appariés sur les 60 paires (20 paires x 3 contrastes) :

| comparaison | victoires | test des signes | Wilcoxon | écart moyen |
|---|---|---|---|---|
| Vectorisé < UNet | 38/60 | p = 0.026 | p = 1.4e-4 | −0.0239 |
| Vectorisé < INR | 58/60 | p < 1e-9 | p = 1.4e-11 | −0.2489 |
| UNet < INR | 56/60 | p < 1e-9 | p = 3.0e-11 | −0.2250 |

Par contraste, vectorisé contre UNet : T1W p = 0.006 (signes) / 1e-4 (Wilcoxon) ;
T2W p = 0.25 / 0.038 ; **T2FLAIR p = 0.59 / 0.14 — soit rien du tout**.

Et l'écart entre les deux architectures est **plus petit que le bruit inter-sujet** :

| contraste | écart vectorisé/UNet | écart-type inter-sujet moyen (n=3) | rapport |
|---|---|---|---|
| T1W | 0.0264 | 0.1580 | 0.17 |
| T2W | 0.0300 | 0.0615 | 0.49 |
| T2FLAIR | 0.0153 | 0.1174 | 0.13 |

**Ce que cela veut dire** : le vectorisé est bien devant le UNet — le signe est
constant sur les 3 contrastes et les 3 métriques, et Wilcoxon le confirme sur
l'ensemble — mais l'effet vaut un sixième de la variabilité entre sujets, et sur
T2FLAIR seul il est indiscernable. Les p-valeurs ci-dessus sont d'ailleurs
**optimistes** : les 20 paires partagent les mêmes 3 sujets, elles ne sont pas
indépendantes. L'écart vectorisé/INR, lui, ne souffre d'aucune de ces réserves.

## 3. Les modèles n'apportent presque rien là où la source ressemble déjà à la cible

Gain relatif sur le témoin, **calculé paire par paire** puis moyenné
(`1 − nRMSE_modèle / nRMSE_identité`) :

| | T1W | T2W | T2FLAIR |
|---|---|---|---|
| Vectorisé | **+34.1 %** | +5.6 % | **−1.2 %** |
| UNet | +31.3 % | +1.1 % | −0.7 % |
| INR | −20.1 % | −87.4 % | −95.7 % |

Nombre de paires où le vectorisé bat effectivement le témoin :

| contraste | nRMSE | SSIM |
|---|---|---|
| T1W | 18/20 | 13/20 |
| T2W | 12/20 | 7/20 |
| T2FLAIR | **9/20** | 11/20 |

Sur T2FLAIR, **le modèle est battu par la recopie de la source sur plus de la
moitié des paires**. La structure est parfaitement nette : il gagne massivement là
où le témoin est catastrophique (toutes les cibles 7T, témoin à 1.5-1.7) et perd
partout où le témoin est déjà bon (0.1T ↔ 3T, 0.1T ↔ 5T, témoin à 0.21-0.27).

| T2FLAIR, 4 meilleures et 4 pires paires | témoin | vectorisé | écart |
|---|---|---|---|
| `1.5T_to_7T` | 1.605 | 0.520 | −1.086 |
| `0.1T_to_7T` | 1.511 | 0.439 | −1.071 |
| `3T_to_7T` | 1.694 | 0.857 | −0.838 |
| `5T_to_7T` | 1.489 | 0.728 | −0.761 |
| … | | | |
| `0.1T_to_5T` | 0.210 | 0.318 | **+0.108** |
| `5T_to_0.1T` | 0.215 | 0.346 | **+0.131** |
| `0.1T_to_3T` | 0.235 | 0.397 | **+0.162** |
| `3T_to_0.1T` | 0.267 | 0.431 | **+0.164** |

Autrement dit : **la moyenne globale de 0.3794 est portée par les paires où il
suffit de corriger un niveau d'intensité.** C'est le sujet de la section suivante.

## 4. 80 % de l'erreur est une erreur d'échelle d'intensité

On corrige chaque prédiction par le meilleur facteur multiplicatif **global**
possible — un oracle : il connaît la vérité — et on regarde ce qu'il reste.

nRMSE brut → après correction d'échelle oracle :

| contraste | Vectorisé | UNet | INR |
|---|---|---|---|
| T1W | 0.4353 → **0.2181** | 0.4617 → 0.2333 | 0.6383 → 0.5681 |
| T2W | 0.3376 → 0.2717 | 0.3676 → 0.2925 | 0.6470 → 0.6181 |
| T2FLAIR | 0.3654 → **0.1676** | 0.3807 → 0.1726 | 0.5998 → 0.5407 |

Part de l'énergie de l'erreur supprimée par ce seul scalaire :

| contraste | Vectorisé | UNet | INR |
|---|---|---|---|
| T1W | **81.9 %** | 81.0 % | 23.7 % |
| T2W | 39.4 % | 41.4 % | 9.3 % |
| T2FLAIR | **82.9 %** | 83.3 % | 20.0 % |

Le facteur oracle n'est pas proche de 1 et son biais est **systématique par champ
cible** (vectorisé) : 1.46 vers 1.5T, 1.31 vers 3T, mais 0.77 vers 5T et 0.77 vers
7T en T1W ; 1.49 vers 3T et 0.67 vers 7T en T2FLAIR. Les prédictions sont donc
trop sombres vers les champs bas-moyens et trop claires vers les champs hauts.

Deux lectures, toutes deux importantes :

- **Pour le vectorisé et le UNet, l'anatomie est juste et c'est le niveau qui est
  faux.** La figure `calib_T1W_3T_to_7T_0009.png` le montre sans ambiguïté : la
  prédiction passe de nRMSE 1.801 à **0.343** en la multipliant par 0.35, et
  l'image obtenue est un T1W 7T plausible, ventricules et noyaux gris en place.
- **Pour l'INR, non.** Seuls 20-24 % de son erreur partent avec le facteur
  d'échelle ; le reste est structurel, comme les images le montrent. Sa limite
  n'est pas de même nature que celle des deux autres.

### D'où vient ce mauvais niveau — et pourquoi ce n'est pas gratuit à corriger

L'inférence dénormalise avec `configs/mmfm/field_norm_stats.json`, dont le `lo`
vaut 0 partout : la dénormalisation est donc **une multiplication par un `hi`
fixe par (modalité, champ cible)**, calculé sur les 1939 volumes d'entraînement.
Chaque sujet reçoit l'échelle MOYENNE de sa classe.

Un correctif évident se propose : estimer l'écart du sujet à sa classe sur son
volume SOURCE (disponible à l'inférence) et le reporter sur la cible. **Il a été
implémenté et mesuré sur les 540 volumes : il dégrade.** nRMSE du vectorisé
0.4353 → **0.5642** en T1W, 0.3376 → 0.3600 en T2W, 0.3654 → **0.4996** en
T2FLAIR — là où l'oracle donne 0.2181 / 0.2717 / 0.1676.

Le facteur ainsi estimé vaut pourtant 1.00 en moyenne par champ cible : il n'est
pas biaisé, il est **trop bruité**, et appliquer un facteur bruité coûte plus
qu'il ne rapporte. La raison est visible dans les données : le rapport « centile
du sujet / `hi` de population » vaut pour le sujet 0009 en T1W 1.04 à 0.1T, 1.30 à
3T et **0.51 à 7T**. L'écart n'est pas une propriété du sujet transportable d'un
champ à l'autre ; c'est une propriété de l'acquisition cible. **La part
individuelle des 80 % n'est donc pas accessible par ce chemin-là** — mais la part
systématique l'est, et c'est l'objet de la section suivante.

## 4bis. La part systématique EST récupérable — mesurée en laissant le sujet de côté

La question laissée ouverte par la section 4 — ce mauvais facteur est-il propre à
chaque acquisition, ou systématique par cible ? — se tranche sans oracle. On
estime le facteur sur les AUTRES sujets (leave-one-subject-out : avec 3 sujets,
la médiane de 2) et on l'applique au sujet évalué. Aucune information de sa
vérité n'entre dans son propre facteur. `--mode recalib`.

nRMSE moyen, vectorisé :

| règle de recalibration | T1W | T2W | T2FLAIR | moyenne |
|---|---|---|---|---|
| brut (a = 1) — état actuel | 0.4353 | 0.3376 | 0.3654 | 0.3794 |
| constante par contraste | 0.4456 | 0.3371 | 0.3684 | 0.3837 |
| **constante par champ CIBLE** | **0.3717** | 0.3308 | 0.3009 | **0.3345** |
| **constante par PAIRE (source→cible)** | 0.3916 | **0.3085** | **0.2693** | **0.3231** |
| *oracle par volume (borne inatteignable)* | *0.2181* | *0.2717* | *0.1676* | *0.2191* |

Même comportement pour le UNet (0.4033 → 0.3504 par champ cible → 0.3399 par
paire) ; pour l'INR le gain est marginal (0.6284 → 0.6108), cohérent avec le fait
que son erreur n'est pas d'échelle.

**Une constante par paire, estimée sur d'autres sujets, ferait passer le score
moyen de 0.3794 à 0.3231 — 15 % de mieux, sans réentraîner quoi que ce soit.**
Une constante par contraste seul ne suffit pas (elle dégrade) : le biais dépend
bien du couple de champs, pas de la modalité.

Les facteurs oracles médians montrent pourquoi : par champ cible en T1W, 1.09 /
1.35 / 1.20 / 0.69 / 0.81 — un biais net et ordonné (prédictions trop sombres vers
les champs bas-moyens, trop claires vers 5T et 7T), avec un écart-type inter-sujet
de 0.29 à 0.50 autour. **La médiane est le signal, la dispersion est le bruit** :
la première est corrigeable, la seconde ne l'est pas par cette voie.

### Deux réserves, à ne pas contourner

1. **Il n'y a pas de jeu de données sur lequel ajuster proprement ces constantes.**
   Les estimer demande des couples (prédiction, vérité) au même champ ; or il
   n'existe que 3 sujets appariés dans tout le jeu (section 9). Le protocole
   leave-one-out ci-dessus est correct — il démontre que la constante GÉNÉRALISE
   d'un sujet à l'autre — mais il l'établit sur n = 2 par estimation.
2. **Une voie de contournement existe et n'a pas été testée** : le biais est une
   différence entre la distribution d'intensité des volumes PRÉDITS à un champ
   donné et celle des volumes RÉELS au même champ. Comparer ces deux
   distributions ne demande aucun appariement — donc les 1056 sujets
   rétrospectifs suffiraient. C'est la piste à instruire avant d'envisager quoi
   que ce soit sur l'architecture.

## 5. Le « mur du 7T » en T1W tient à UN volume sur trois

Le nRMSE de la cible 7T en T1W (0.7542) a orienté une part importante du travail
de ce projet — l'INR longtemps retenu pour ses victoires apparentes vers 7T, le
conditionnement AdaGN conçu explicitement pour ces cibles. Il se décompose ainsi,
sujet par sujet :

| paire (T1W, vectorisé) | 0006 | 0007 | **0009** | moyenne |
|---|---|---|---|---|
| `3T_to_7T` | 0.382 | 0.353 | **1.801** | 0.845 |
| `1.5T_to_7T` | 0.400 | 0.585 | **1.886** | 0.957 |
| `0.1T_to_7T` | 0.357 | 0.411 | **1.327** | 0.698 |

La cause est dans la vérité terrain, pas dans les modèles. Norme L2 des volumes
T1W bruts :

| champ | 0006 | 0007 | 0009 | max/min |
|---|---|---|---|---|
| 3T | 1950 | 1465 | 1904 | 1.33 |
| 5T | 695 | 648 | 646 | 1.08 |
| **7T** | 845 | 705 | **325** | **2.60** |

Le volume T1W@7T du sujet 0009 est 2.6 fois plus sombre que celui du sujet 0006.
Comme le nRMSE divise par ‖vérité‖, ce seul volume domine la moyenne.

Effet sur les chiffres, cible 7T en T1W :

| | 3 sujets | sans 0009 | part due à un seul sujet |
|---|---|---|---|
| Vectorisé | 0.7542 | **0.4048** | 46 % |
| UNet | 0.7956 | 0.4316 | 46 % |
| INR | 0.7765 | 0.6095 | 22 % |

Sensibilité du tableau complet — en retirant, pour chaque contraste, le sujet le
plus atypique (T1W : 0009 ; T2W : 0006 ; T2FLAIR : 0007) :

| | T1W 3 suj. → 2 suj. | T2W 3 suj. → 2 suj. | T2FLAIR 3 suj. → 2 suj. |
|---|---|---|---|
| Vectorisé | 0.4353 → 0.3460 | 0.3376 → 0.2978 | 0.3654 → 0.3143 |
| UNet | 0.4617 → 0.3690 | 0.3676 → 0.3333 | 0.3807 → 0.3259 |
| INR | 0.6383 → 0.5909 | 0.6470 → 0.6289 | 0.5998 → 0.5886 |

**Le classement survit intégralement**, ce qui est rassurant. Les valeurs
absolues, non : 20 à 25 % du nRMSE publié vient du sujet le plus atypique de
chaque contraste. Avec n = 3, aucune moyenne de ce projet n'est robuste à un
volume près, et le « mur du 7T en T1W » en particulier est **pour moitié un
artefact d'un seul volume aberrant**.

Ce constat ne dit pas que les modèles vont bien vers 7T : même sans 0009, la cible
7T (0.405) reste la plus coûteuse des cinq en T1W. Il dit que son ampleur a été
surestimée du double, et qu'une architecture n'y pouvait rien.

## 6. Qualitatif — ce que les images montrent et que les métriques taisent

Toutes les figures sont produites à partir des volumes de prédiction déjà écrits
(0.5 mm natif, 364x436x364), sans GPU ni ré-inférence : rien ne peut diverger des
chiffres publiés. Les cartes d'erreur d'une même figure partagent une échelle, et
la colonne de gauche de la rangée d'erreur est le **témoin identité** — sans lui,
une carte d'erreur seule paraît toujours mauvaise.

| figure | ce qu'elle établit |
|---|---|
| `panel_T1W_3T_to_7T_0006.png` | Le cas dur « normal ». Vectorisé et UNet rendent une anatomie correcte mais **plate** : le contraste substance blanche / substance grise du 7T n'est pas reproduit. L'erreur se concentre sur le ruban cortical. L'INR rend des blobs. |
| `panel_T2W_3T_to_7T_0006.png` | Le cas facile. Vectorisé et UNet sont visuellement proches de la vérité, texture comprise. Le témoin identité est déjà sombre partout — c'est **pourquoi** T2W→7T est facile : la source ressemble déjà à la cible. |
| `panel_T2FLAIR_3T_to_7T_0006.png` | Échec radiométrique franc : les prédictions saturent en blanc, tout le parenchyme au même niveau, là où la vérité a un contraste net. Anatomie juste, niveaux faux. |
| `panel_T1W_3T_to_0.1T_0006.png` | L'échec **inverse** : la vérité 0.1T est floue (bas champ, faible RSB) et les prédictions sont **plus nettes qu'elle**. Les modèles gardent la finesse de leur source au lieu de prendre celle de leur cible. |
| `panel_T2FLAIR_3T_to_0.1T_0006.png` | Une des paires où le modèle **perd** contre le témoin : la carte d'erreur de l'identité est visiblement plus sombre que celle du modèle. |
| `panel_T1W_3T_to_7T_0009.png` | Le volume aberrant de la section 5. La vérité est très sombre ; les modèles prédisent le niveau 7T habituel, donc tout sature. |
| `calib_T1W_3T_to_7T_0009.png` | La démonstration : la même prédiction multipliée par 0.35 passe de nRMSE 1.801 à 0.343 et redevient un T1W 7T plausible. Sur la même figure, l'INR passe de 1.374 à 0.612 seulement — son erreur n'est pas d'échelle. |

**Trois modes d'échec, et un seul est un problème d'architecture :**

1. *Niveau global faux* — dominant en T1W et T2FLAIR (80 % de l'énergie de
   l'erreur). Ce n'est pas un problème de modèle génératif, c'est un problème de
   calibration d'intensité.
2. *Netteté non transportée* — les prédictions gardent quelque chose de la netteté
   de leur source. Trop nettes vers 0.1T, trop floues vers 7T. Section 7.
3. *Effondrement structurel* — propre à l'INR, et déjà documenté deux fois.

## 7. La netteté, en nombre

Pour mettre un chiffre sur le flou, on compare la répartition de l'énergie entre
détails et structures larges : **indice = (HF/BF)_prédiction / (HF/BF)_vérité**,
avec HF = 0.25-0.75 Nyquist et BF = 0.02-0.15. Le rapport HF/BF est pris avant la
comparaison, et c'est indispensable : une puissance haute fréquence brute est
proportionnelle au carré de l'échelle d'intensité, or c'est précisément
l'échelle que ces modèles ratent (section 4) — un rapport de puissances brutes
mesurerait donc surtout cette erreur-là. Normalisé, l'indice est invariant à tout
facteur multiplicatif.

**Contrôle de l'indice** : la médiane du témoin identité vaut 0.99 / 1.00 / 1.00
sur les trois contrastes. Une vraie image a bien, en médiane, la même balance
spectrale qu'une autre vraie image — l'indice mesure ce qu'il prétend.

Médiane par champ cible (`sharpness_index_cerveau_entier.csv`) :

| T1W | 0.1T | 1.5T | 3T | 5T | 7T |
|---|---|---|---|---|---|
| identité | 11.74 | 2.06 | 0.87 | 0.31 | 0.48 |
| Vectorisé | 2.93 | 0.55 | 0.29 | 0.11 | 0.19 |
| UNet | 3.34 | 0.59 | 0.29 | 0.11 | 0.17 |
| INR | 7.95 | 1.29 | 0.65 | 0.25 | 0.44 |

| T2FLAIR | 0.1T | 1.5T | 3T | 5T | 7T |
|---|---|---|---|---|---|
| identité | 2.31 | 1.23 | 0.88 | 0.76 | 0.88 |
| Vectorisé | 0.70 | 0.40 | 0.33 | 0.30 | 0.40 |
| UNet | 0.90 | 0.48 | 0.34 | 0.29 | 0.35 |
| INR | 1.77 | 1.03 | 0.75 | 0.60 | 0.67 |

Médiane globale sur cerveau entier : vectorisé 0.30 / 0.40 / 0.39
(T1W / T2W / T2FLAIR), UNet 0.30 / 0.44 / 0.41, témoin 0.99 / 1.00 / 1.00.
Restreinte à l'intérieur du cerveau (le chiffre qui fait foi, voir plus bas) :
vectorisé 0.36 / 0.40 / 0.26, UNet 0.36 / 0.42 / 0.26, témoin 1.02 / 1.00 / 1.00.

**Les deux bonnes architectures ne restituent qu'environ un tiers de la finesse de
la vérité.** C'est le prix de la représentation MedVAE plus l'interpolation
1mm → 0.5mm, et c'est cohérent avec l'aspect lissé des zooms corticaux.

Le spectre (`spectres_radiaux.png`) précise **où** ça se joue : vers 7T, les
courbes des modèles passent SOUS la vérité entre 0.25 et 0.55 Nyquist — le détail
anatomique manquant — puis repassent AU-DESSUS après 0.6, ce qui n'est pas du
détail retrouvé mais de la texture ajoutée (décodeur + rééchantillonnage). Vers
0.1T, elles sont au-dessus de la vérité sur toute la bande : la prédiction reste
trop nette pour sa cible.

### Une réserve sur l'INR, et elle inverse sa lecture

Sur le cerveau ENTIER, l'indice de l'INR (médiane 0.66 en T1W, 0.80 en T2FLAIR)
paraît plus proche de 1 que celui du vectorisé (0.30, 0.39) — ce qui
contredirait toutes les images. C'est un artefact de bord : la frontière
cerveau/fond de l'INR est en escalier et injecte de la puissance sur toute la
bande. Recalculé sur l'INTÉRIEUR du cerveau (moitié centrale de la boîte
englobante, `--interior-frac 0.5`), sur les mêmes 180 volumes :

| médiane de l'indice | identité | Vectorisé | UNet | INR |
|---|---|---|---|---|
| T1W — cerveau entier | 0.99 | 0.30 | 0.30 | *0.66* |
| **T1W — intérieur** | **1.02** | **0.36** | **0.36** | **0.24** |
| T2W — cerveau entier | 1.00 | 0.40 | 0.44 | *0.44* |
| **T2W — intérieur** | **1.00** | **0.40** | **0.42** | **0.22** |
| T2FLAIR — cerveau entier | 1.00 | 0.39 | 0.41 | *0.80* |
| **T2FLAIR — intérieur** | **1.00** | **0.26** | **0.26** | **0.14** |

Le témoin reste à 1.00-1.02 sous les deux découpages — l'indice reste valide,
c'est bien l'INR que le cerveau entier flattait. **À l'intérieur du cerveau,
l'INR est la plus floue des trois, d'un facteur 1.5 à 2** par rapport au
vectorisé, et de 4 à 7 par rapport à la vérité. C'est noté dans le code
(`_shrink_bbox`) pour que personne ne relise la première ligne sans la seconde.

Détail par champ cible (intérieur, vectorisé) : T1W 4.16 / 0.74 / 0.28 / 0.11 /
0.22 de 0.1T à 7T ; T2FLAIR 0.72 / 0.28 / 0.24 / 0.18 / 0.27. La netteté est
**toujours** insuffisante sauf vers 0.1T, où elle est très excédentaire — la
prédiction ne sait pas dégrader.

## 8. Fichiers

| chemin | contenu |
|---|---|
| `identity_baseline_{T1W,T2W,T2FLAIR}.csv` | témoin « ne rien faire », formules officielles vérifiées bit-à-bit |
| `calibration_intensite.csv` | par (contraste, paire, sujet, méthode) : nRMSE brut, après correction d'échelle réalisable, oracle d'échelle, oracle affine, et les sommes `p_sq`/`pg`/`g_sq` qui permettent de tester une autre règle de recalibration sans relire les volumes |
| `sharpness_index_cerveau_entier.csv` | indice de netteté relative, cerveau entier |
| `sharpness_index_interieur_0.5.csv` | le même, restreint à l'intérieur du cerveau — c'est celui qui fait foi |
| `spectres_radiaux.png` / `.csv` | puissance spectrale radiale, 3 contrastes x 2 paires |
| `panel_*.png` | panoramas source / vérité / 3 prédictions + cartes d'erreur + témoin + zoom cortical |
| `calib_T1W_3T_to_7T_0009.png` | prédiction brute contre prédiction remise à l'échelle par l'oracle |
| `identity.log`, `calibration.log`, `sharpness*.log` | sorties brutes des balayages — **non versionnées** (`*.log` est dans `.gitignore`), présentes sur la machine uniquement ; tout ce qu'elles contiennent est reproductible depuis les CSV et les commandes ci-dessus |

Scripts (nouveaux) :

| chemin | rôle |
|---|---|
| `src/cfm/eval_qualitative_3arch.py` | figures : `--mode panel`, `--mode spectrum`, `--mode calib` |
| `src/cfm/eval_quantitative_3arch.py` | mesures : `--mode identity`, `--mode calibration`, `--mode sharpness`, `--mode table` |

Aucun des deux n'ouvre le GPU ni ne relance d'inférence : ils lisent les volumes
de prédiction déjà écrits et les CSV de l'évaluateur officiel. Les métriques des
modèles ne sont JAMAIS recalculées maison — sauf dans `--mode calibration`, où le
nRMSE brut recalculé sert justement de contrôle et redonne 0.4353 / 0.4617 /
0.3376 / 0.3654 au millième près.

## 9. Le plafond statistique de ce projet : n = 3, et il n'existe rien de plus

En vérifiant la provenance des données pour écarter toute contamination, un fait
structurel est apparu et il conditionne tout ce qui précède.

| split | sujets | champs distincts par sujet | paires source/cible exploitables |
|---|---|---|---|
| `Training_retrospective` (entraînement) | 1056 | **1 pour tous les 1056** | **0** |
| `Training_prospective` (évaluation) | **3** | 5 | 20 par contraste |
| `Validating_prospective` | 15 | 1 par contraste | **0** |

- **Aucun des 1056 sujets d'entraînement n'existe à deux champs.** Chacun a été
  acquis à UN champ, avec jusqu'à trois contrastes. Les données d'entraînement
  sont donc entièrement **non appariées** en champ — ce qui justifie a posteriori
  le choix du flow matching multi-marginal, mais interdit toute supervision
  appariée.
- **Les 15 sujets de `Validating_prospective` n'ont qu'un champ par contraste** :
  ils servent à soumettre, pas à mesurer.
- **Il ne reste donc que 3 sujets appariés dans tout le jeu de données.** Le n = 3
  de toutes les évaluations de ce projet n'est pas un raccourci qu'on aurait pu
  éviter : c'est le maximum disponible.

**Conséquence directe sur les décisions passées.** Un écart de 0.015 à 0.030 nRMSE
entre deux architectures, mesuré sur 3 sujets dont l'écart-type inter-sujet vaut
0.06 à 0.16, ne peut pas porter un choix d'architecture à lui seul. Ce qui reste
solide, c'est le signe, constant sur 3 contrastes x 3 métriques, plus le Wilcoxon
apparié sur 60 paires — pas l'amplitude, et pas les classements par champ cible.

## 10. Conclusions

### Sur le classement des trois approches

1. **L'INR est distancé sans ambiguïté**, sur les 3 contrastes, les 3 métriques et
   58/60 paires (Wilcoxon p = 1e-11). Les images le confirment : effondrement
   structurel, et à l'intérieur du cerveau c'est la plus floue des trois d'un
   facteur deux. Troisième fermeture cohérente avec les deux précédentes.
2. **Le vectorisé devance le UNet, mais faiblement** : signe constant partout,
   Wilcoxon significatif sur l'ensemble (p = 1.4e-4), mais l'écart vaut un
   sixième de la variabilité inter-sujet et n'est significatif ni sur T2W
   (p = 0.25 au test des signes) ni sur T2FLAIR (p = 0.59). Le vectorisé reste le
   bon choix — il est aussi 2x plus léger — mais **cet écart ne peut pas porter
   une conclusion scientifique**, et n = 3 est un plafond imposé par les données
   (section 9), pas une négligence rattrapable.
3. **Sur T2FLAIR, le vectorisé est battu par la recopie de la source sur 11 paires
   sur 20.** Aucune des trois architectures n'apporte de gain net sur ce contraste
   hors des cibles 7T.

### Sur ce qui limite réellement les deux bonnes architectures

**Ce n'est pas l'anatomie, c'est le niveau.** 82 % de l'énergie de l'erreur en T1W
et 83 % en T2FLAIR partent avec un seul scalaire par volume. La dénormalisation
d'inférence multiplie par un `hi` fixe par (modalité, champ cible), estimé sur
1939 volumes d'entraînement ; chaque sujet reçoit donc l'échelle moyenne de sa
classe, et l'écart réel entre sujets atteint un facteur 2.6.

Ce n'est pas pour autant une correction gratuite : l'estimateur naturel — reporter
sur la cible l'écart mesuré sur le volume source du même sujet — **a été
implémenté et il dégrade** (section 4). L'écart d'échelle n'est pas une propriété
transportable du sujet.

**Mais une part de ce facteur est systématique, et elle est récupérable**
(section 4bis) : une constante par paire (source→cible), estimée en laissant le
sujet évalué de côté, fait passer le score moyen du vectorisé de **0.3794 à
0.3231** — 15 % de mieux, sans réentraîner. C'est de loin le meilleur rapport
gain/coût identifié dans ce projet. Sa mise en œuvre bute sur un point à
instruire : il n'existe pas de données appariées pour ajuster ces constantes hors
des 3 sujets d'évaluation. La voie sans appariement — comparer la distribution
d'intensité des volumes prédits à un champ à celle des volumes réels au même
champ, sur les 1056 sujets rétrospectifs — n'a pas été testée et devrait l'être
avant tout nouveau travail sur l'architecture.

### Sur le second défaut, la netteté

Les deux bonnes architectures ne restituent qu'environ **un tiers** de la finesse
de la vérité (indice médian 0.26-0.40 à l'intérieur du cerveau, contre 1.00 pour
le témoin). Le spectre
montre un déficit dans la bande 0.25-0.55 Nyquist — du détail anatomique perdu —
et un excès au-delà de 0.6 qui est de la texture ajoutée, pas du détail retrouvé.
C'est le seul défaut sur lequel une meilleure représentation agit, et le
fine-tuning LPIPS de MedVAE terminé le même jour (SSIM d'auto-reconstruction
0.9152 → 0.9727 à 1 mm, voir
`results/benchmark_vae/analysis/medvae_lpips_20260825/manifest.md`) est le
candidat direct. Mais il agit sur les 20 % restants de l'erreur, pas sur les 80 %.

### Sur la méthode

Le témoin identité aurait dû exister sur les trois contrastes dès le début : il
change le sens des chiffres (le « meilleur » contraste, T2W, est celui où les
modèles apportent le moins), et sans lui le gain de +5.6 % sur T2W passait pour un
score de 0.3376. Le projet tenait déjà un
compteur sur « la loss d'entraînement ne prédit rien » (cinq occurrences) ; en
voici un second, distinct et au moins aussi coûteux : **une moyenne ne dit pas ce
qu'on croit tant qu'on n'a pas regardé sa décomposition — par sujet, par paire, et
contre un témoin trivial.** Trois des cinq résultats de ce document viennent de
là et d'aucune expérience nouvelle.
