# Le flow translate, et le vrai transport ne se translate pas

**Date** : 2026-09-02
**Verdict** : **le flow appris est une translation à 3.5 % près, alors que le vrai
déplacement champ→champ est propre au sujet à 71.7 %.** Un facteur **20**. C'est
la conséquence attendue de l'entraînement non apparié, et cela explique d'un coup
pourquoi huit leviers de mécanisme n'ont rien donné.

---

## 1. Ce qui était à expliquer

Le diagnostic du 2026-08-31 avait laissé deux faits sans hypothèse :

1. la dispersion inter-sujets des latents **prédits** est quasi identique quel que
   soit le champ cible — en T2FLAIR : 2651, 2651, 2651, 2650, identique à quatre
   chiffres, alors que les réels varient (T1W : 2208, 2277, 2465, 2171) ;
2. le **rang effectif** était la seule quantité déficitaire (0.862).

## 2. Une seule cause, et le témoin qui manquait

Le diagnostic comparait *prédit* contre *réel-à-la-cible*. Il manquait le nuage
**source**. Ajouté (`src/cfm/diagnose_flow_translation.py`) :

| | valeur | min | max |
|---|---|---|---|
| **part du déplacement propre au sujet** | **0.0348** | 0.0065 | 0.0812 |
| dispersion prédite / **source** | **0.9972** | 0.9910 | 1.0003 |
| rang effectif prédit / **source** | **1.0024** | 0.9996 | 1.0088 |

`D_i = z_pred,i − z_src,i`, part propre = `moyenne_i ‖D_i − D̄‖ / ‖D̄‖`.

**96.5 % du déplacement est commun à tous les sujets.** Le nuage prédit est celui
de la source, transporté en bloc — dispersion et rang préservés à 0.3 % près.

**Le « déficit de rang » n'existait pas.** Le rang prédit égale celui de la source
(T1W 15.2, T2W 17.3, T2FLAIR 18.6) ; le 0.862 comparait ce rang **source** à celui
des nuages **cibles** (19.3, 20.8, 19.6). Les deux observations du 2026-08-31 se
réduisent donc à un seul fait : **le flow translate**.

## 3. Est-ce un défaut ? La mesure qui tranche

Une translation n'est fautive que si la vraie transformation en est une autre.
Attention au piège : Task 3 juge la justesse **par sujet**, pas l'appariement de
distributions — et une translation **préserve exactement l'identité du sujet**. Ce
serait donc le bon comportement si le vrai transport était affine.

Mesuré sur les **3 sujets appariés** (`Training_prospective`, les seuls du jeu à
exister à plusieurs champs ; latents encodés pour l'occasion, cache
`.../pro_train/`), même décomposition, `D_i = z_cible,i − z_source,i` :

| contraste | 0.1T→1.5T | 0.1T→3T | 0.1T→5T | 0.1T→7T |
|---|---|---|---|---|
| T1W | 0.553 | **1.405** | 0.437 | 0.652 |
| T2W | 0.473 | 0.479 | 0.446 | 0.458 |
| T2FLAIR | 0.926 | 0.488 | 0.609 | 0.725 |

Sur les 20 paires × 3 contrastes : **0.7173** en moyenne (min 0.286, max 1.528).

| | part propre au sujet |
|---|---|
| **vrai déplacement** | **0.7173** |
| **flow appris** | **0.0348** |
| rapport | **×20.6** |

Les normes le disent aussi : `‖D̄‖` vaut 1500–3000 et l'écart-type autour, 1000–1800.
**La composante manquée est du même ordre que celle qui est captée.**

## 4. Pourquoi le flow ne peut pas l'apprendre

C'est la conséquence directe de l'entraînement **non apparié**. Le modèle voit une
source `z_i` au champ f et une cible `z_j'` au champ g appartenant à **un autre
sujet** : la composante individuelle de `z_j' − z_i` est, de son point de vue, du
bruit. L'espérance (ou la médiane) conditionnelle est le décalage **commun**.

Le couplage OT existe pour réparer exactement cela — et il est **vacuous à
129 024 dimensions**, mesuré le 2026-08-30 : +0.2 % au lieu des −19 % annoncés par
le harnais, parce que chercher des correspondances entre 8 points dans un tel
espace ne donne rien.

**La chaîne se ferme** : 0 sujet apparié sur 1056 → couplage indépendant → part
individuelle inapprenable → le flow ne peut que translater.

## 5. Ce que cela dit des huit résultats négatifs

Échelle du temps, conditionnement FiLM, `adjacent_only`, recalibration
d'intensité, couplage OT, pas d'intégration, budget d'ajustement INR, MedVAE
perceptuel. **Tous réparaient la MACHINERIE d'un flow qui apprenait
structurellement la mauvaise chose**, pour une raison d'information et non de
mécanisme. Trois étaient de vrais défauts de code — les corriger a réparé la
géométrie (`cos(v(0),v(1))` de 1.000000 à −0.24/0.80/0.03, courbure ×100) sans
toucher à ce qui limite le score.

## 6. La limite de ce qui est affirmé ici

**On ne peut PAS trancher, avec 3 sujets appariés, si ces 71.7 % sont
prédictibles depuis la source.** Deux lectures restent ouvertes :

- **prédictible** — le déplacement dépend de l'anatomie du sujet, qu'un modèle
  pourrait apprendre s'il voyait des couples ; il faudrait alors un couplage qui
  fonctionne en haute dimension ;
- **irréductible** — une part vient de la variation d'acquisition de la cible,
  non transportable depuis la source. On sait déjà que la variance inter-sujets du
  niveau d'intensité (35 %) est de cette nature (manifeste
  `recalib_20260829` §8).

Un ajustement linéaire sur 3 points dans 129 024 dimensions interpolerait
trivialement : la question n'est pas tranchable avec ces données. **Ne pas
conclure au-delà.**

## Fichiers

| chemin | contenu |
|---|---|
| `rbest.json` | mesure sur le flow (3 contrastes × 4 cibles) |
| `../../../src/cfm/diagnose_flow_translation.py` | flow : part propre, dispersion et rang contre la SOURCE |
| `../../../src/cfm/diagnose_true_displacement.py` | vrai déplacement sur les 3 sujets appariés |

## Reproduction

```
PYTHONPATH=src python src/cfm/diagnose_flow_translation.py \
    --config configs/mmfm/vectorized_rbest.yaml \
    --json-out results/mmfm/translation_20260902/rbest.json
# les latents des sujets apparies (45 volumes, 7.3 min) :
PYTHONPATH=src python src/cfm/precompute_mmfm_latents.py \
    --config configs/mmfm/vectorized_rbest.yaml --env local --split pro_train \
    --field-norm-stats configs/mmfm/field_norm_stats.json
PYTHONPATH=src python src/cfm/diagnose_true_displacement.py
```
