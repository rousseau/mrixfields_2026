# Couplage en dimension réduite : l'idée est SANS OBJET, et mon explication du 2026-08-30 était fausse

**Date** : 2026-09-03
**Verdict** : **l'OT en dimension pleine apparie parfaitement les bons sujets
(180/180)**. Il n'y a donc rien à « restaurer » par une réduction de dimension.
Le couplage échoue en production pour une raison entièrement différente : **il n'y
a aucun appariement correct à trouver dans les données d'entraînement**.

---

## 1. D'où venait l'idée

Le 2026-09-02 a établi que le flow appris est une translation (3.5 % du
déplacement propre au sujet) quand le vrai transport l'est à 71.7 %. La cause
identifiée était l'entraînement non apparié, et le couplage OT — vacuous en
production (+0.2 %, 2026-08-30) — devait être ce qui répare cela.

**L'explication que j'avais écrite le 2026-08-30 était** : « chercher des
correspondances entre 8 points dans un espace à 129 024 dimensions ne donne rien,
la structure de plus proche voisin y étant essentiellement aléatoire ». D'où
l'idée de coupler dans un espace réduit.

## 2. Le test synthétique, et pourquoi il ne conclut pas

`src/cfm/diagnose_coupling_dimension.py` — qualité du couplage mesurée comme le
Spearman entre le facteur sujet de la source et celui du partenaire choisi (la
vraie carte préserve `s`, donc 1.0 = parfait, 0 = hasard) :

| dim | OT plein | ACP 4 | ACP 16 | ACP 64 | aléa 4 | aléa 16 | aléa 64 |
|---|---|---|---|---|---|---|---|
| 64 | 0.987 | 0.992 | 0.989 | 0.987 | 0.931 | 0.966 | 0.982 |
| 4096 | 0.952 | 0.972 | 0.926 | 0.852 | 0.254 | 0.379 | 0.641 |
| 32768 | 0.873 | 0.927 | 0.723 | 0.503 | −0.008 | 0.054 | 0.195 |
| **129024** | **0.767** | **0.897** | 0.602 | 0.250 | −0.045 | −0.044 | 0.126 |

Deux enseignements réels : **l'ACP-4 bat l'OT plein partout**, et l'écart croît
avec la dimension (0.897 contre 0.767 à 129 024) ; et la projection **aléatoire
détruit tout** (−0.045) — le témoin confirme que ce qui compte est la *direction*
du signal, pas la préservation des distances.

**Mais l'OT plein ne s'effondre PAS** (0.767, très loin du hasard). Le problème
synthétique ne reproduit donc pas le régime réel, et la raison est identifiable :
**son facteur sujet est unidimensionnel** — une seule direction commune à tous les
sujets — alors que l'anatomie réelle est de haute dimension. Conclure depuis ce
régime serait conclure depuis un problème plus facile que le nôtre.

## 3. Le test direct sur nos latents, qui tranche

`src/cfm/diagnose_coupling_real.py`. Les 3 sujets de `Training_prospective` sont
les seuls du jeu à exister à plusieurs champs : pour chaque (contraste, paire de
champs) on connaît donc l'appariement véritable. On demande à l'OT d'apparier et
on compte. Affectation 1-1 par l'algorithme hongrois sur le plan OT — un `argmax`
par ligne pourrait choisir deux fois la même cible.

Hasard : pour une permutation uniforme de 3 éléments, l'espérance du nombre de
points fixes vaut exactement 1, soit **1/3**.

| méthode | justes | taux |
|---|---|---|
| **OT plein** (129 024 dim) | **180/180** | **1.000** |
| ACP 4 | 180/180 | 1.000 |
| ACP 2 | 176/180 | 0.978 |
| aléatoire 4 | 168/180 | 0.933 |
| aléatoire 2 | 134/180 | 0.744 |
| *hasard* | | *0.333* |

Taux de 1.000 sur les trois contrastes séparément.

## 4. Ce que cela change

**Mon explication du 2026-08-30 est fausse.** L'OT ne calcule pas mal
l'appariement en haute dimension : il l'apparie parfaitement dès que des
correspondances existent.

**La vraie raison de son inutilité en production** : il n'y a **aucune
correspondance à trouver**. Zéro sujet sur 1056 n'existe à deux champs, donc l'OT
ne peut que coupler le sujet A au champ f avec le sujet **B** au champ g. Le
déplacement `A@f → B@g` contient (anatomie de B − anatomie de A), qui
n'appartient pas à la transformation de champ.

**Aucune méthode de couplage, en aucune dimension, ne peut retrouver une
correspondance absente des données.** L'idée du couplage en dimension réduite est
donc **sans objet** — elle reposait sur une explication erronée.

Cela recadre aussi le rôle de l'OT en flow matching non apparié : c'est un
dispositif de **réduction de variance** pour apprendre la carte de transport
*marginale*, pas un moyen de retrouver des correspondances individuelles. Task 3
demande la carte individuelle.

## 5. Réserve sur ce qui est établi

Avec **3** sujets, l'affectation est un problème 3×3 et l'anatomie d'un sujet est
beaucoup plus distinctive que l'écart entre deux champs : l'OT y réussit
trivialement. « L'OT apparie correctement » n'est donc démontré que dans ce
régime facile, et rien ne dit qu'il tiendrait avec 100 sujets.

**Le point logique, lui, n'en dépend pas** : il porte sur l'*absence* de
correspondances dans les données d'entraînement, pas sur la capacité de l'OT.

## 6. Ce qui reste, et ce qui est clos

**Clos** : le couplage (toutes dimensions), et avec lui la dernière piste que le
diagnostic du 2026-09-02 avait désignée.

**Ouvert, et seul** : la composante individuelle de 71.7 % est-elle *prédictible
depuis le volume source* ? Si oui, un modèle pourrait l'apprendre sans couplage —
il suffirait qu'elle soit une fonction de `z_src`. Si non, elle est irréductible.
**3 sujets appariés ne permettent pas de trancher** : un ajustement linéaire sur
3 points dans 129 024 dimensions interpolerait trivialement.

## Fichiers

| chemin | contenu |
|---|---|
| `synthetic.json` | qualité du couplage par dimension, harnais |
| `real.json` | taux d'appariement correct sur les 3 sujets appariés |
| `../../../src/cfm/diagnose_coupling_dimension.py` | test synthétique (ACP contre aléatoire contre plein) |
| `../../../src/cfm/diagnose_coupling_real.py` | test direct sur nos latents |
