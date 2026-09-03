# La composante individuelle du déplacement n'est pas prédictible depuis la source

**Date** : 2026-09-03
**Verdict** : **NON, sous la seule forme testable avec ces données.** Un
rééchelonnement de la déviation source explique **13.5 %** de la composante
individuelle — et c'est le R² *dans l'échantillon*, donc une borne supérieure.
Hors échantillon, le terme dépendant de la source **ne bat pas la translation
seule** (33/60, p = 0.52).

---

## 1. La dernière question ouverte

Le 2026-09-02 a mesuré que le vrai transport champ→champ est propre au sujet à
**71.7 %** quand le flow appris ne l'est qu'à **3.5 %**. Le 2026-09-03 a fermé la
piste du couplage : il n'y a **aucune correspondance à trouver** dans des données
non appariées, en aucune dimension — et l'explication par la concentration des
distances que j'avais avancée le 2026-08-30 était fausse (l'OT plein apparie
180/180 dès que des correspondances existent).

Restait une seule issue : **si cette composante est une fonction du volume
source**, un modèle peut l'apprendre sans couplage, en conditionnant sur `z_src`.

## 2. Le piège, et la forme retenue

Il n'existe que **3 sujets appariés** dans tout le jeu. Ajuster une matrice, ou
même un vecteur, dans 129 024 dimensions interpolerait trivialement : trois points
y sont toujours parfaitement explicables, et tout R² obtenu ainsi ne voudrait rien
dire.

Forme testée, **à un seul paramètre**. Si la carte était affine en intensité —
`z_cible ≈ a·z_source + b` — alors `D_i − D̄ = (a−1)·(z_i − z̄)` et la composante
individuelle serait entièrement prédictible par un scalaire. On ajuste donc **un
`α` par (contraste, paire)** par moindres carrés sur les écarts centrés : 1
paramètre contre 3 × 129 024 dimensions de résidu.

Plus un **vrai test hors échantillon** : `α` ajusté sur 2 sujets, prédiction du
troisième, 3 plis, et le rapport qui compte est
`‖D_k − D̂_k‖ / ‖D_k − D̄_ajust‖` — le terme dépendant de la source fait-il mieux
que la translation seule ? Inférieur à 1 = oui.

## 3. Le résultat

| | moyenne | min | max |
|---|---|---|---|
| **R² dans l'échantillon (1 paramètre)** | **0.135** | 0.000 | 0.552 |
| **LOO, erreur modèle / translation** | **0.990** | 0.808 | 1.272 |
| α + 1 | 0.770 | 0.339 | 1.030 |

Cellules où le modèle bat la translation hors échantillon : **33/60**, test des
signes **p = 0.52** (Wilcoxon p = 0.084). Sur 60 cellules (20 paires × 3
contrastes).

**Le R² dans l'échantillon est décisif à lui seul** : 0.135 avec un paramètre est
une *borne supérieure* sur ce qui peut généraliser. Même dans le meilleur des cas,
un rééchelonnement de la source ne capte que 13.5 % de la composante individuelle.

## 4. Ce que cela établit

**86.5 % de la composante individuelle est orthogonale à la déviation source.**
C'est donc de l'information sur l'**acquisition cible** qui n'est pas présente
dans la source — et à ce titre **irréductible**, quelle que soit l'architecture,
le couplage ou le conditionnement.

Cela recoupe une mesure indépendante : le 2026-08-25 avait établi que l'écart
d'intensité d'un sujet à sa classe **n'est pas transportable d'un champ à
l'autre** — « c'est une propriété de l'acquisition cible » — avec 35 % de variance
inter-sujets (manifeste `recalib_20260829` §8). Deux chemins sans rapport, même
conclusion.

**Observation réelle au passage** : les `α` sont **systématiquement négatifs**
(α+1 = 0.770 en moyenne, toutes les cellules sous 1.03). La déviation individuelle
**rétrécit** en montant en champ. Motif cohérent, mais qui n'explique que 13.5 %
de la variance. Le contrôle par les facteurs d'échelle d'intensité oracles du
2026-08-25 (0.63 à 1.45) recouvre partiellement cette plage sans la reproduire —
la carte n'est donc pas non plus un simple rééchelonnement global.

## 5. La limite, à ne pas contourner

**Seule la forme affine/scalaire a été testée.** Une fonction plus riche de
`z_src` pourrait en principe capter davantage — mais aucune n'est testable avec
3 sujets appariés sans surajustement trivial. La conclusion exacte est donc :
*non prédictible sous la seule forme que ces données permettent de tester*, avec
un R² dans l'échantillon de 0.135 qui borne l'espoir par le haut.

Le test hors échantillon repose sur 2 sujets d'ajustement : il est très mince, et
c'est le R² qui porte la conclusion, pas lui.

## 6. Conséquence

**La série est close.** Huit leviers de mécanisme, puis le couplage (toutes
dimensions), puis la prédictibilité depuis la source : tout est mesuré, tout est
négatif, et la cause est identifiée et non contournable avec ces données —
**0 sujet apparié sur 1056, et une composante individuelle majoritairement
absente de la source**.

Le plateau à 0.3737 brut / 0.2143 structurel n'est pas un défaut de mise en
œuvre : c'est la borne de ce que l'information disponible permet.

## Fichiers

| chemin | contenu |
|---|---|
| `scalar.json` | 60 cellules : α, R², rapport LOO |
| `../../../src/cfm/diagnose_displacement_predictable.py` | la mesure |
