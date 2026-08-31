# Le flow contracte-t-il ses latents vers la moyenne ? — NON, l'inverse

**Date** : 2026-08-31
**Verdict** : **hypothèse RÉFUTÉE.** Les latents prédits sont *plus* dispersés que
les réels (115 %), pas moins. La partie D du plan (loss L2) **n'est pas lancée** :
son fondement vient de tomber.

**Mais le diagnostic trouve autre chose** : le **rang effectif** est la seule
quantité systématiquement déficitaire (0.862). Perte de *diversité* à taille
constante, pas effondrement vers la moyenne.

---

## Ce qui était testé, et pourquoi

Le flow est entraîné en L1 — il estime donc la **médiane** conditionnelle — sur un
couplage **indépendant** (l'OT est vacuous à 129 024 dimensions : mesure du
2026-08-30, l'activer a donné +0.2 % au lieu des −19 % annoncés par le harnais).
Un estimateur de tendance centrale sur un couplage aléatoire devrait produire des
latents contractés vers la moyenne de la classe cible.

Cette hypothèse-là aurait expliqué d'un coup les deux symptômes suivis depuis le
début : images trop lisses (une moyenne est lisse) et contraste incomplètement
transporté (une moyenne est à mi-chemin). Et elle aurait rendu le passage à L2
argumenté au lieu de « les 9 scripts de la référence le font ».

## Méthode

Purement en espace latent, sur les caches existants : N latents source passés dans
le flow (`euler_integrate`), N latents prédits au champ cible, comparés en
dispersion à N latents **réels** du même champ. **Aucun appariement** — on compare
deux nuages, ce qui est indispensable ici puisque aucun sujet n'existe à deux
champs. **Aucun décodage MedVAE** — donc quelques minutes.

`src/cfm/diagnose_flow_contraction.py`, checkpoint R-best, 24 latents par nuage,
source 0.1T, 20 pas d'Euler.

**Contrôle de cohérence** : sur un pas d'identité (`t_i = t_j`, donc `dt = 0`, donc
le flow ne déplace rien) l'écart relatif vaut **exactement 0.00e+00**. Sans ce
contrôle, un rapport de 0.8 n'aurait pas été interprétable — il aurait pu venir de
la mesure et non du modèle.

## Résultat

| rapport prédit / réel | valeur | lecture |
|---|---|---|
| amplitude (écart-type par élément) | **1.166** | pas de contraction |
| dispersion inter-sujets | **1.152** | pas de contraction — légère expansion |
| **rang effectif (diversité)** | **0.862** | **déficit systématique** |

Par contraste, le rang effectif prédit contre réel : T1W **15.2 / 19.3**,
T2W **17.4 / 20.8**, T2FLAIR **18.6 / 19.6**.

## L'observation inattendue

**La dispersion prédite est quasi identique quel que soit le champ cible.**

| T1W, dispersion inter-sujets | →1.5T | →3T | →5T | →7T |
|---|---|---|---|---|
| prédit | 2458 | 2459 | 2460 | 2446 |
| réel | 2208 | 2277 | 2465 | 2171 |

Même chose en T2W (2672, 2666, 2655, 2653) et en T2FLAIR (2651, 2651, 2651, 2650).
Le flow déplace bien le centroïde avec `t` — la mesure de géométrie le confirme,
`cos(v(0),v(1))` vaut −0.24/0.80/0.03 et non 1 — mais il applique une
transformation dont la **forme du nuage** ne dépend presque pas du champ visé.

**Aucun correctif n'est proposé à partir de ça.** Je n'ai pas d'hypothèse étayée
sur sa cause, et ce projet a assez payé les hypothèses formulées trop vite : la
piste L1/tendance centrale vient d'être réfutée pour la **troisième** fois
(2026-08-26 par ‖médiane‖/‖moyenne‖ = 0.98–1.01 ; 2026-08-26 par `adjacent_only` ;
ici par la dispersion).

## Conséquence sur le plan

- **Partie D (loss L2 + standardisation) : NON LANCÉE.** 4 h de GPU économisées sur
  un motif dont le fondement est tombé. C'est exactement ce que le diagnostic
  devait décider.
- Parties B (pas d'intégration) et C (budget d'ajustement INR) inchangées.

## Fichiers

| chemin | contenu |
|---|---|
| `rbest.json` | mesures par cellule (contraste × champ cible), prédit / réel / rapport |
| `../../../src/cfm/diagnose_flow_contraction.py` | le diagnostic |

## Reproduction

```
PYTHONPATH=src python src/cfm/diagnose_flow_contraction.py \
    --config configs/mmfm/vectorized_rbest.yaml --n 24 \
    --json-out results/mmfm/contraction_20260831/rbest.json
```
