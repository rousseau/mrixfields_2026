# Correction de l'erreur d'échelle d'intensité

**Date** : 2026-08-29 → 2026-08-30
**Verdict final : PISTE FERMÉE, NÉGATIVE.** Elle gagne sur le vectorisé
(0.3737 → 0.3525, p = 0.014) et **dégrade l'UNet (+0.0654) et l'INR (+0.0941)**.
Deux explications de l'échec ont été formulées et **réfutées par la mesure**. La
cause retenue n'est pas réparable par un meilleur estimateur. La clé
`inference.intensity_recalibration` a été **retirée** de la config de production ;
le code et les tables restent en place pour trace.

*La section 3 ci-dessous rapporte le succès sur R-best, qui reste exact — c'est
sa généralisation qui échoue. Lire la section 8 avant d'en faire quoi que ce
soit.*

---

## Le problème

L'inférence dénormalise par un `hi` **fixe** par (modalité, champ cible), calculé
sur les 1939 volumes d'entraînement : chaque sujet reçoit l'échelle **moyenne** de
sa classe. Le facteur multiplicatif global qu'un oracle choisirait enlève **80 %
de l'énergie de l'erreur** en T1W et en T2FLAIR (mesure du 2026-08-25).

Ce qui bloquait : estimer ce facteur demande des couples (prédiction, vérité) au
même champ. Le protocole leave-one-subject-out du 2026-08-25 donnait 0.3794 →
0.3231, mais **sur n = 2 par estimation et en consommant les sujets
d'évaluation** — donc inutilisable comme correctif livrable.

## La voie retenue

Celle que le manifeste du 2026-08-25 laissait ouverte sans l'instruire : le biais
est une différence entre la **distribution** d'intensité des volumes prédits à un
champ donné et celle des volumes **réels** au même champ. Comparer deux
distributions **ne demande aucun appariement**.

- **Volumes réels** : 40 sujets par (modalité, champ) tirés de
  `Training_retrospective`.
- **Volumes prédits** : 316 prédictions issues de sujets d'entraînement
  (`Training_retrospective` et `Validating_prospective`), tous distincts des
  sujets d'évaluation.
- **Facteur** : `a = médiane(‖g‖) / médiane(‖p‖)` par (modalité, champ cible) et
  par paire. La norme L2 est le bon choix : le facteur oracle vaut
  `<p,g>/<p,p> = (‖g‖/‖p‖)·cos(p,g)`, et c'est aussi la norme par laquelle le
  nRMSE divise.

**Aucun sujet d'évaluation n'entre dans l'estimation.**

## Résultat

| métrique | R-best | recalibré | écart | apparié (60 paires) |
|---|---|---|---|---|
| **nRMSE** | 0.3737 | **0.3525** | **−0.0212** | **40/60, signes p = 0.0135, Wilcoxon p = 0.0225** |
| SSIM | 0.9047 | 0.9056 | +0.0009 | 34/60, p = 0.37 |
| LPIPS | 0.0990 | 0.1001 | +0.0011 | 29/60, p = 0.90 |

Par contraste :

| | R-best | recalibré | écart | victoires |
|---|---|---|---|---|
| T1W | 0.4441 | 0.4531 | **+0.0091** | 11/20 |
| T2W | 0.3181 | **0.2939** | −0.0242 | 14/20 |
| T2FLAIR | 0.3589 | **0.3104** | −0.0485 | 15/20 |

**SSIM et LPIPS ne bougent pas** : c'est la signature d'une recalibration d'intensité
pur, qui ne touche pas la structure. C'est le comportement attendu, et c'est
aussi la limite — la correction ne rend pas les images meilleures, elle les
remet au bon niveau.

**T1W résiste**, comme partout dans ce projet. Cohérent avec le fait que la
variance inter-sujet y est énorme : le volume T1W@7T du sujet 0009 est 2.6 fois
plus sombre que celui du sujet 0006 (mesure du 2026-08-25, section 5). Une
constante de classe ne peut rien contre ça.

## Repères

| | nRMSE moyen |
|---|---|
| production (vectorisé) | 0.3794 |
| R-best (géométrie du flow corrigée) | 0.3737 |
| **R-best + recalibration** | **0.3525** |
| *leave-one-out du 2026-08-25 (circulaire, non livrable)* | *0.3231* |
| *oracle par volume (borne inatteignable)* | *0.2191* |

La recalibration récupère donc **environ 40 %** de ce que le protocole circulaire
laissait espérer (0.3737 → 0.3525 contre 0.3737 → ~0.3231), et le fait sans
toucher aux données d'évaluation.

## Deux erreurs commises en chemin, et ce qu'elles coûtent

1. **Hypothèse analytique fausse.** J'ai d'abord suppose que le biais venait d'un
   flow qui ne transporte pas le niveau d'intensite. Mesure : le niveau normalisé
   varie peu entre champs en T1W (0.56 à 0.73) alors que les facteurs oracles y
   vont de 0.68 à 1.29. Écartée avant d'être utilisée.

2. **`Validating_prospective` n'est PAS un split apparié.** J'avais compté
   « 3 fichiers par champ » et conclu « les mêmes 3 sujets aux 5 champs ». Faux :
   les sujets sont 0001-0003 à 0.1T, 0004/0005/0008 à 1.5T, 0010-0012 à 3T,
   0013-0015 à 5T, 0016-0018 à 7T — **chacun n'existe qu'à un seul champ**, comme
   le jeu d'entraînement. Le projet avait déjà établi qu'il n'existe que 3 sujets
   appariés en tout ; il suffisait de lire les identifiants.
   **Coût : 5.2 h de GPU.** Les prédictions ont été récupérées pour la voie sans
   appariement (elles y apportent le troisième contraste), mais la table oracle
   held-out visée est **impossible** : il n'existe aucun jeu apparié en dehors
   des 3 sujets d'évaluation.

## Réserves

- Les facteurs par paire reposent sur 12 à 28 prédictions ; ceux de T2W sur 12
  seulement (une seule campagne couvrait ce contraste).
- La statistique `l2` suppose que seule l'échelle est fausse. Le terme `cos(p,g)`
  manque, et le diagnostic du 2026-08-28 le chiffrait : corrélation 0.441 entre
  facteur estimé et facteur oracle sur un échantillon de prédictions plus mince.
  **La correction sous-corrige, ce qui est la direction sûre** — appliquer un
  facteur bruité coûte plus qu'il ne rapporte (mesuré le 2026-08-25).
- Mesuré sur **R-best** uniquement. Rien ne garantit que les mêmes facteurs
  valent pour la production, l'UNet ou l'INR : la table dépend du modèle.

## Fichiers

| chemin | contenu |
|---|---|
| `task3_rbest_recal_{T1W,T2W,T2FLAIR}.csv` | score après recalibration |
| `../../../configs/mmfm/intensity_recalibration.json` | **la table livrée** (3 contrastes) |
| `../../../configs/mmfm/intensity_recalibration_nopair.json` | table intermédiaire, 2 contrastes, échantillon plus mince |
| `../staircase_20260827/task3_vec_rbest_*.csv` | référence avant recalibration |

## Reproduction

```
# 1. predictions de calibration sur des sujets d'ENTRAINEMENT
PYTHONPATH=src python src/cfm/infer_mmfm_unified.py \
    --config configs/mmfm/vectorized_rbest.yaml \
    --checkpoint outputs/mmfm/vec_rbest/weights/model_final.pth \
    --output_dir outputs/mmfm/calib_rbest/predictions \
    --split Training_retrospective --modalities T1W T2W T2FLAIR --max_subjects 4 \
    --field_norm_stats configs/mmfm/field_norm_stats.json

# 2. table
PYTHONPATH=src python src/cfm/estimate_intensity_recalibration.py \
    --pred-root outputs/mmfm/calib_merged/task3 --split Training_retrospective \
    --limit 40 --out configs/mmfm/intensity_recalibration.json

# 3. en production : la cle inference.intensity_recalibration de
#    configs/mmfm/vectorized_rbest.yaml l'applique automatiquement.
#    Sur un arbre deja ecrit :
PYTHONPATH=src python src/cfm/apply_intensity_recalibration.py \
    --pred-root outputs/mmfm/vec_rbest/predictions/task3 \
    --out-root outputs/mmfm/vec_rbest_recal/predictions/task3 \
    --table configs/mmfm/intensity_recalibration.json
```


---

## 8. Généralisation : la piste est fermée

### 8.1 Les trois architectures

Même méthode, protocole strictement identique, une table **par modèle** (le
facteur vaut `médiane(‖g‖réel) / médiane(‖p‖prédit)` : le numérateur est commun,
le dénominateur est le biais propre à chaque architecture).

| architecture | avant | après | écart | apparié (60 paires) |
|---|---|---|---|---|
| **vectorisé R-best** | 0.3737 | **0.3525** | −0.0212 | 40/60, p = 0.014 |
| **UNet** | 0.4033 | 0.4688 | **+0.0654** | 28/60, p = 0.70 |
| **INR** | 0.3749 | 0.4690 | **+0.0941** | 19/60, p = 0.006 |

Un succès sur trois. Sur l'UNet, la moyenne se dégrade fortement alors que le
test apparié n'est pas significatif : le dégât est **concentré sur quelques
paires**, celles qui reçoivent un facteur aberrant.

Plages des facteurs estimés : R-best **0.741–1.556**, UNet **0.572–3.244**,
INR **jusqu'à 3.401**. Les facteurs *oracles* mesurés vont de **0.63 à 1.45** :
au-delà, un facteur ne reflète plus une erreur d'échelle réelle.

### 8.2 Deux hypothèses formulées, deux réfutées

**Hypothèse 1 — le terme `cos(p,g)` manquant.** Le facteur oracle vaut
`(‖g‖/‖p‖)·cos(p,g)` ; l'estimateur n'en calcule que le rapport de normes. Sur
l'INR l'explication tenait (prédictions plates). **Réfutée sur l'UNet** :
`cos(p,g)` y vaut **0.935 à 0.993**, l'approximation devrait donc être excellente
— or la corrélation à l'oracle n'est que de **0.344** (écart absolu moyen 0.216).

**Hypothèse 2 — la norme L2 conflate intensité et taille du cerveau.**
**Réfutée** : le coefficient de variation inter-sujets de ‖v‖ et celui de la
moyenne sur le cerveau sont quasi identiques (T1W@0.1T 0.213 contre 0.217 ;
T1W@3T 0.140 contre 0.140), et le nombre de voxels cérébraux ne varie que de
8–10 %. Changer de statistique (`--stat brain`) ne changerait rien.

### 8.3 La cause retenue

La même mesure montre autre chose : **la dispersion inter-sujets du niveau
d'intensité atteint 35 %** (T1W@7T, sujets d'évaluation ; 6–21 % ailleurs). Or la
correction est une **constante par classe**, estimée sur les sujets
d'entraînement et appliquée aux sujets d'évaluation.

C'est un problème de **transfert de population**, pas d'estimation : aucun
raffinement de l'estimateur ne fera suivre une constante de classe à une variance
individuelle de cette ampleur. Le manifeste du 2026-08-25 l'avait déjà écrit pour
la voie par volume source — « l'écart n'est pas une propriété du sujet
transportable d'un champ à l'autre » — et le seul protocole qui fonctionnait
(leave-one-out sur les 3 sujets d'évaluation, 0.3794 → 0.3231) **consomme les
sujets d'évaluation** et n'est donc pas déployable.

### 8.4 Le garde-fou : utile mais insuffisant

Un garde-fou a été ajouté puis mesuré : rapport de contraste relatif
(écart-type/moyenne, sans dimension, donc il mesure la forme et non le niveau)
entre prédictions et volumes réels au même champ ; en dessous de 0.65 le facteur
est forcé à 1.000.

| | rapport de contraste | effet |
|---|---|---|
| R-best | 0.696–1.260 | **0 cellule bloquée** — gain intact |
| INR | 0.277–0.820 | **13/15 bloquées**, dont les 5 de T2W |
| **UNet** | *au-dessus du seuil* | **0 bloquée — et l'UNet dégrade quand même** |

Sur l'INR il évite **+0.0891 des +0.0941** de dégât : 52 des 60 paires restent
strictement inchangées, et des 8 modifiées, 4 améliorent et 4 dégradent
(p = 1.00 — ex æquo exclus). **Sur l'UNet il ne bloque rien et laisse passer le
dégât** : le seuil, calibré sur deux architectures, ne discrimine pas la
troisième. Le garde-fou est conservé dans le code, mais il ne rend pas la méthode
sûre.

### 8.5 Ce qui reste vrai

L'erreur d'échelle d'intensité **est** 80 % de l'énergie de l'erreur en T1W et en
T2FLAIR (mesure du 2026-08-25, inchangée) et un oracle par volume ferait passer
le score de 0.3794 à 0.2191. **Cette marche existe ; elle n'est simplement pas
franchissable avec les données disponibles** — 3 sujets appariés dans tout le jeu,
et une variance individuelle qui domine la part systématique.

C'est un résultat, pas un échec de mise en œuvre : il déplace l'effort vers la
marche suivante, **0.2191 → 0.1048, l'erreur structurelle du flow**.
