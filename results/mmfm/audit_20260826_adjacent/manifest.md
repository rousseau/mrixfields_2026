# `adjacent_only: true` — résultat NÉGATIF, et il réfute le mécanisme proposé

**Date** : 2026-08-26
**Verdict** : neutre dans l'ensemble (34 victoires sur 60 paires, p = 0.18), et
**pire là où l'hypothèse prédisait le gain** : sur les sauts les plus longs
(0.1T ↔ 7T), +0.0153 de nRMSE et 2 victoires sur 6.

---

## Ce qui était testé, et pourquoi

L'audit du code (`../audit_20260826_order3/manifest.md`) a établi un fait mesuré :
l'entraînement tire uniformément parmi les 20 paires ordonnées de champs et pose
une **droite** entre chacune, alors que **les marginales ne sont pas alignées** —
l'écart du point intermédiaire à la corde vaut 0.72 à 1.32 fois la longueur de la
corde, soit 2.3 à 5.6 fois le bruit d'estimation.

Hypothèse : ces droites se contredisent (à un même `(z_t, t)`, un échantillon dit
« va vers 3T », un autre « tu devrais avoir dépassé 3T en route vers 7T »), le
modèle n'apprend qu'un compromis, et c'est ce qui produit le contraste
incomplètement transporté qu'on voit sur les figures.

`adjacent_only: true` n'entraîne que les sauts entre champs consécutifs : la
chaîne devient cohérente, et les transitions longues s'obtiennent en intégrant à
travers les marginales intermédiaires.

**Vérifications avant de lancer** (le run coûte 2 h) :
- `_build_contrast_fields` trie bien la liste des champs, donc « adjacent dans la
  liste » = adjacent en intensité de champ ;
- l'échantillonneur produit 90 % de sauts adjacents et 10 % d'identité, aucune
  paire longue — contre 54 % de sauts longs sans l'option ;
- le booléen YAML est parsé comme `True` malgré le commentaire en ligne.

Config dédiée `configs/mmfm/vectorized_adjacent.yaml` : trois clés changées
(`adjacent_only`, `output_subdir`, `task_name`), le fichier de production reste
intact pour que le checkpoint de référence demeure reproductible.

## Résultat

| | T1W | T2W | T2FLAIR | moyenne |
|---|---|---|---|---|
| **nRMSE** — toutes paires (production) | 0.4353 | **0.3376** | **0.3654** | **0.3794** |
| **nRMSE** — adjacent seulement | **0.4326** | 0.3419 | 0.3691 | 0.3812 |
| SSIM — toutes paires | 0.8997 | 0.8980 | **0.8949** | **0.8975** |
| SSIM — adjacent seulement | 0.8995 | **0.8981** | 0.8944 | 0.8973 |
| LPIPS — toutes paires | 0.0983 | 0.0911 | 0.0927 | 0.0941 |
| LPIPS — adjacent seulement | **0.0982** | 0.0911 | **0.0924** | **0.0939** |

Test apparié sur les 60 paires : **34/60**, test des signes p = 0.18, Wilcoxon
p = 0.47, écart moyen +0.0018. **Aucune différence soutenue.**

## La ventilation réfute le mécanisme

C'est sur les sauts longs que l'incohérence des droites devait mordre le plus.
C'est exactement là que la variante fait le **moins bien** :

| longueur de saut | n | toutes paires | adjacent | écart | victoires |
|---|---|---|---|---|---|
| 1 (adjacent) | 24 | 0.3211 | 0.3211 | +0.0000 | 13/24 |
| 2 | 18 | 0.4161 | 0.4156 | −0.0005 | 12/18 |
| 3 | 12 | 0.4298 | 0.4317 | +0.0019 | 7/12 |
| **4 (0.1T ↔ 7T)** | 6 | **0.4022** | **0.4175** | **+0.0153** | **2/6** |

L'ordre est monotone et inverse de la prédiction : plus le saut est long, plus la
variante cohérente est pénalisée. Entraîner directement la transition longue vaut
mieux que l'obtenir en intégrant à travers la chaîne — l'erreur accumulée sur
quatre hops dépasse ce que coûtait l'incohérence.

## Ce que cela apprend

1. **Le fait mesuré reste vrai** : les marginales ne sont pas alignées, et les
   cibles d'entraînement se contredisent. **La conséquence que j'en tirais est
   fausse** : rendre les cibles cohérentes n'améliore rien.
2. **Le modèle absorbait déjà l'incohérence.** Elle n'était pas la contrainte
   active sur le transport de contraste.
3. **Ordre de grandeur utile** : deux runs de 25 000 itérations de la même
   architecture, ne différant que par ce drapeau, terminent à 0.3794 et 0.3812.
   Cela borne ce qu'on peut lire dans les micro-écarts de ce projet — tout ce qui
   est sous ~0.002 est indistinguable du bruit de run.
4. **`adjacent_only` reste à `false`** en production ; la config de variante est
   conservée pour trace.

## Fichiers

| chemin | contenu |
|---|---|
| `task3_vectorized_adjacent_{T1W,T2W,T2FLAIR}.csv` | Task 3, 20 paires × 3 sujets |
| `../../../configs/mmfm/vectorized_adjacent.yaml` | config de la variante |
| `outputs/mmfm/vectorized_adjacent/` | checkpoint, log, métriques (non versionné) |

Loss d'entraînement : 18.43 finale pour un seuil « prédire zéro » de **22.18**
(83 % du seuil). La référence est à 12.08 pour un seuil de **14.55** (83 %
également). **Les deux runs n'ont pas des loss comparables** — le régime adjacent
divise par un `dt` de 0.25 au lieu de jusqu'à 1.0, ses cibles sont ~4× plus
grandes — mais leur position relative est identique.
