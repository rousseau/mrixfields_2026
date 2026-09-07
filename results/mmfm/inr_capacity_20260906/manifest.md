# Capacité de la représentation INR — l'INR n'est pas mal réglé, il est affamé

**Date** : 2026-09-06
**Script** : `src/cfm/bench_inr_capacity.py`
**Données** : `Training_prospective`, sujet 0006, T1W × {0.1T, 3T, 7T} = 3 volumes par variante
**Espace de mesure** : identique à `bench_representation.py` — grille 1 mm 192×224×192,
normalisation `field_fixed`, volume normalisé dans [−1, 1]. **Les chiffres MedVAE et INR
de cette page sont donc directement comparables : même volume, même entrée, même métrique.**
**CSV** : `cells.csv`, `summary.csv`

---

## La question, et pourquoi elle n'était pas tranchée

L'auto-reconstruction INR valait nRMSE ~0.35 contre ~0.08 pour MedVAE. Trois causes
possibles appellent des correctifs opposés :

1. **la famille** — un INR ne peut pas représenter ces volumes ;
2. **la capacité** — il pourrait, mais sa modulation par volume est trop étroite ;
3. **l'optimisation** — la capacité suffirait, mais les 20 pas de descente à pas fixe
   qui ajustent `z` ne convergent pas.

Le dépôt ne les séparait pas. Ce banc les sépare, et il ajoute le témoin qui manquait
à toute interprétation : **ce que vaut un budget de N nombres dépensé de la façon la
plus bête possible** — une grille basse résolution ré-interpolée linéairement.

---

## Le tableau, sur les mêmes 3 volumes

| représentation | budget par volume | nRMSE | SSIM | PSNR |
|---|---|---|---|---|
| *témoin trivial : grille 11×13×11* | *1 536* | *0.2381* | *0.6902* | *18.91* |
| **INR production** (`fit_new_volume`, 20 pas SGD) | **512 effectifs** | 0.1448 | 0.8057 | **23.41** |
| INR, 200 pas de SGD | 512 | 0.1417 | 0.8085 | 23.61 |
| INR, Adam sur `z` via le hypernetwork | 512 | *0.4341* | *0.4809* | *13.83* (diverge) |
| INR, modulation DIRECTE (shift), Adam | 1 536 | 0.1396 | 0.7796 | 23.72 |
| + `modulate_scale` | 3 072 | 0.1268 | 0.7962 | 24.61 |
| + LoRA rang 4 | ~13 k | 0.1019 | 0.8525 | 26.61 |
| + LoRA rang 16 | ~51 k | 0.0737 | 0.9027 | 29.45 |
| **+ LoRA rang 64** | **~200 k** | **0.0531** | **0.9351** | **32.32** |
| *témoin trivial : grille 48×56×48* | *129 024* | *0.0954* | *0.8889* | *27.21* |
| **MedVAE pré-entraîné** | 129 024 | 0.0587 | 0.9551 | 31.82 |
| **MedVAE affiné LPIPS** | 129 024 | 0.0503 | 0.9673 | 33.14 |
| **SIREN LIBRE 256×6** (tous poids, un volume) | ~330 k | **0.0306** | 0.9710 | **37.17** |
| **SIREN LIBRE 512×8** (tous poids, un volume) | ~1.8 M | **0.0157** | **0.9905** | **42.86** |

Et la même échelle en **erreur restreinte au premier plan** — la seule honnête ici,
puisque 82 % du volume est du fond plat trivialement reconstruit :

| représentation | budget | nRMSE premier plan |
|---|---|---|
| *témoin trivial 11×13×11* | *1 536* | *0.6641* |
| INR production | 512 | 0.4367 |
| + LoRA rang 64 | ~200 k | 0.1548 |
| *témoin trivial 48×56×48* | *129 024* | *0.2835* |
| MedVAE pré-entraîné | 129 024 | 0.1832 |
| MedVAE affiné LPIPS | 129 024 | 0.1576 |
| **SIREN libre 256×6** | ~330 k | **0.0866** |
| **SIREN libre 512×8** | ~1.8 M | **0.0428** |

---

## Le plafond de la famille INR, mesuré pour la première fois

Un SIREN **neuf**, tous poids libres, ajusté sur UN volume (3 000 pas d'Adam) :

| | nRMSE | premier plan | SSIM | PSNR |
|---|---|---|---|---|
| taille de production (256×6, ~330 k) | 0.0306 | 0.0866 | 0.9710 | **37.17 dB** |
| plus grand (512×8, ~1.8 M) | 0.0157 | **0.0428** | 0.9905 | **42.86 dB** |

**La famille INR représente ces volumes quasi sans perte** — 42.9 dB, deux fois moins
d'erreur en premier plan que le meilleur MedVAE (0.0428 contre 0.1576). La prédiction
de la littérature (>35–40 dB pour un sur-apprentissage mono-volume) est vérifiée sur
ces données précises.

**Cela ferme définitivement la question posée** : ce n'est ni la famille de modèles, ni
l'architecture SIREN, ni `omega_0`, ni la profondeur. Un SIREN à la taille EXACTE de la
production atteint 37.17 dB quand la production plafonne à 23.41 dB. **Tout l'écart est
dans ce que l'on autorise à varier par volume** — 512 nombres en production, 330 000
pour le SIREN libre.

*(Réserve : le SIREN libre n'est pas une représentation utilisable telle quelle pour la
tâche — il faudrait stocker 330 k poids par volume et le flow devrait opérer dessus.
C'est une BORNE SUPÉRIEURE de la famille, pas une proposition. Ce qu'elle établit, c'est
que le plafond n'est pas là où le projet le croyait.)*

## Trois lectures, dans l'ordre d'importance

### 1. Les deux représentations valent exactement la même chose par unité de budget

- L'INR de production bat son témoin trivial de **+4.50 dB** (23.41 contre 18.91).
- MedVAE bat le sien de **+4.61 dB** (31.82 contre 27.21).

**À budget égal, l'INR et le VAE extraient la même quantité d'information.** L'écart de
8.4 dB entre eux en production n'est pas un écart de qualité de représentation : c'est
l'écart entre 512 nombres et 129 024, soit un facteur **252**.

Conséquence directe sur une phrase répétée dans le dépôt depuis des mois — « le handicap
de l'INR est représentationnel » : c'est vrai au sens du budget, **faux au sens de la
famille de modèles**. Aucun correctif de mécanisme INR ne pouvait combler 252×.

### 2. La capacité est le levier, et l'échelle est monotone et raide

De 1 536 à 200 000 valeurs de modulation : **+8.6 dB** (23.72 → 32.32), sans toucher aux
poids du SIREN, sans réentraîner quoi que ce soit. À ~200 k valeurs, l'INR **dépasse
MedVAE pré-entraîné** (32.32 contre 31.82) et approche le MedVAE perceptuel (33.14).

Le code disait déjà où était le levier (`inr_backbone.py:150`, « le seul levier qui fasse
réellement croître l'information par volume ») ; il n'avait jamais été mesuré.

### 3. Ce n'est PAS un problème d'optimisation, et le hypernetwork ne sert à rien

- 20 pas de SGD contre 200 pas : 23.41 → 23.61 dB. **+0.2 dB pour 10× le budget.**
- La modulation directe optimisée par Adam (1 536 DDL, 1 000 pas) donne 23.72 dB, soit
  **le même résultat** que les 20 pas de SGD de production.

Autrement dit : **`latent_dim = 129 024` et son hypernetwork de 66 M paramètres
n'apportent rien** par rapport à l'optimisation directe de 1 536 nombres. Ils coûtent
1 Go de checkpoint et 0.98 s/itération pour un résultat identique.

`prod_adam` (Adam sur `z` à travers le hypernetwork) **diverge** franchement (13.83 dB) :
le chemin `z → hypernetwork → gamma` est mal conditionné, ce que corrobore la mesure de
l'auditeur (valeurs propres de `J Jᵀ` de 19.5 à 0.0076, atténuation jusqu'à 1300×).

---

## Deux défauts confirmés par mesure indépendante, dans le code d'entraînement

**A. Le méta-entraînement a vu les MÊMES 16 384 coordonnées à chacun des 50 000 pas.**
`meta_train_step` est appelé sans générateur (`train_inr_backbone.py:219`) ; `sample_points`
en fabrique alors un, **graine dérivée du contenu** (`inr_backbone.py:104` :
`int(n) * 1_000_003 + int(num_points)`), donc constante. Vérifié : deux appels rendent des
indices identiques. Les poids partagés `θ` n'ont donc jamais été supervisés ailleurs que
sur **0.198 %** de la grille (espacement moyen 8.0 voxels).

**B. Après 50 000 pas, la base SIREN est restée son initialisation aléatoire.**
RMS des poids EMA de production contre la valeur exacte de l'initialisation SIREN :

| tenseur | appris | init | rapport |
|---|---|---|---|
| `siren.first.linear.weight` | 0.194692 | 0.192450 | **1.01×** |
| `siren.hidden.0.linear.weight` | 0.002984 | 0.002946 | **1.01×** |
| `siren.hidden.4.linear.weight` | 0.003408 | 0.002946 | 1.16× |
| `siren.final.weight` | 0.008841 | 0.002946 | 3.00× |

Seules la couche de sortie et le hypernetwork ont bougé. **La « base de fonctions
méta-apprise » est un SIREN aléatoire**, et A explique pourquoi.

Ces deux défauts sont réels et gratuits à corriger, mais **ils n'expliquent pas l'écart
de 8.4 dB** : le banc les contourne (il optimise la modulation directement sur ce même
SIREN) et retrouve 32.32 dB dès que le budget monte. C'est bien la capacité qui borne.

---

## La correction, entièrement dans les options existantes

Le schéma Functa/MedFuncta est déjà implémenté (`inr_backbone.py:341` : `hyper_hidden_dim=0`
⇒ `gamma = z`, modulation directe, le hypernetwork disparaît) :

```yaml
inr_backbone:
  hyper_hidden_dim: 0        # supprime le goulot de rang 512 ET les 66 M paramètres
  lora_rank: 16              # modulation_dim ~51k  (ou 64 -> ~200k)
  latent_dim: 51456          # doit égaler modulation_dim quand hyper_hidden_dim=0
```

et, dans `train_inr_backbone.py`, passer un générateur qui **avance à chaque pas** pour
que `θ` voie autre chose que 0.198 % du volume.

**Réserves à ne pas perdre.**
- Les chiffres `mod_*` sont obtenus sur le SIREN **gelé** de production, dont `θ` a été
  méta-entraîné pour une modulation par décalage seul. Un backbone réentraîné pour la
  LoRA ferait vraisemblablement mieux — ces valeurs sont un **plancher**, pas un plafond.
- L'initialisation LoRA est un piège armé : `fit_latent` initialise `z` à zéro, donc en
  modulation directe `A` et `B` partiraient tous deux à zéro et **le gradient serait nul
  des deux côtés** — la capacité annoncée resterait morte. Le banc initialise `B` à
  l'aléatoire et `A` à zéro (convention Hu et al. 2021, `bench_inr_capacity.py::init_gamma`).
- Le pas d'apprentissage de la LoRA doit être séparé de celui des décalages : au pas des
  décalages (1e-2), la LoRA de rang 16 **fait diverger** le réseau (nRMSE 0.83, SSIM 0.09).
  Le banc balaie ce pas et retient le meilleur ; sans cette précaution on aurait conclu à
  tort que la capacité supplémentaire ne sert à rien.
- 3 volumes, un seul contraste (T1W), un seul sujet. L'échelle est monotone sur les trois
  cellules sans exception, mais le niveau absolu n'est pas garanti sur T2W/T2FLAIR.
- `prod_sgd20` ajuste ici sur la grille DENSE alors que la production échantillonne
  1 048 576 points (`inr.yaml:62`) : la variante mesurée est donc **au moins aussi bonne**
  que la production. L'écart mesuré est un plancher.
