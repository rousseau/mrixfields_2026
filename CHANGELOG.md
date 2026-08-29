# Journal des expériences — MRIxFields 2026

Chaque étape et chaque test mesuré, du plus récent au plus ancien. Une entrée par
expérience, **y compris et surtout les résultats négatifs** : ils sont la majorité,
et ce sont eux qu'on refait par inadvertance quand ils ne sont écrits nulle part.

**Convention.** Toute expérience qui produit un chiffre entre ici, avec :
sa date, ce qui était testé, **le chiffre**, le verdict, et le chemin du manifeste
détaillé. Le manifeste porte la méthode et les réserves ; cette page porte
l'historique et permet de retrouver quoi que ce soit en une lecture. Une
expérience non écrite ici est réputée ne pas avoir eu lieu.

---

## État de référence au 2026-08-26

Task 3, 20 paires × 3 sujets, évaluateur officiel, 1 mm, prédictions écrites à
0.5 mm natif.

| nRMSE | T1W | T2W | T2FLAIR | moyenne | SSIM moy. | LPIPS moy. |
|---|---|---|---|---|---|---|
| **INR** (`outputs/mmfm/inr_std`) | 0.4070 | 0.3800 | **0.3376** | **0.3749** | 0.8631 | 0.1557 |
| **Vectorisé** (`outputs/mmfm/vectorized`) | **0.4353** | **0.3376** | 0.3654 | 0.3794 | **0.8975** | **0.0941** |
| UNet (`outputs/mmfm/unet`) | 0.4617 | 0.3676 | 0.3807 | 0.4033 | 0.8949 | 0.0953 |
| *Témoin identité (recopier la source)* | *0.9273* | *0.3859* | *0.5574* | *0.6235* | *0.8882* | — |

**Le classement n'est pas un ordre total.** En moyenne agrégée l'INR devance le
vectorisé en nRMSE (0.3749 contre 0.3794) — **mais cet avantage n'est pas soutenu
paire par paire : 29 victoires sur 60, p = 0.65 au test des signes, Wilcoxon
p = 0.38.** Les deux sont statistiquement indiscernables en nRMSE. Le vectorisé,
lui, est **nettement devant en SSIM et en LPIPS**. Ne jamais écrire « la
meilleure » sans la métrique — ni sans le test apparié.

**Planchers connus, à soustraire de toute conclusion** : écrêtage à `hi`
(0.10–0.13 de nRMSE pour un prédicteur parfait), interpolation 1 mm → 0.5 mm
(0.0156). **n = 3 est un plafond imposé par les données**, pas une négligence.

---

## Leçons récurrentes, avec leur compteur

**La loss d'entraînement ne prédit rien sur ce pipeline — 6 occurrences.**
2026-08-13 (AdaGN : loss +2.4, résultat identique) · 2026-08-14 (flip : loss +1.8,
résultat identique) · et la plus nette, 2026-08-26 : la loss du flow INR divisée
par 3.3 et passée sous le seuil du prédicteur constant, **score T1W dégradé**
(0.3787 → 0.4070). Ne jamais conclure sur une courbe de loss.

**Une moyenne ne dit pas ce qu'on croit — 2 occurrences majeures.**
2026-08-14 (T1W n'était pas représentatif : 2/3 du problème non mesuré) ·
2026-08-25 (sans témoin identité, un gain de +5.6 % passait pour un score de
0.3376). Toujours décomposer : par sujet, par paire, et contre un témoin trivial.

**Un test dont le seuil accepte le cassé ne teste rien.**
`test_inr_backbone_smoke.py:200` assère `nrmse_fg < 0.6` quand la production
valait 0.6383, à 2 mm alors que la production est à 1 mm. Il a laissé passer
trois bugs pendant des mois. **Non corrigé à ce jour.**

---

## 2026-08-27 (soir) — Le correctif de temps SEUL ne change rien : le mécanisme de conditionnement domine

**Verdict : R1 (`time_scale: 1000` seul, 25 000 itérations, 2 h) est TOUJOURS
aveugle au temps sur données réelles.** Détail :
`results/mmfm/synthetic_20260827/manifest.md`.

| mesure sur latents réels | production | **R1 (`ts=1000`)** |
|---|---|---|
| cos(v(t=0), v(t=1)) | 1.000000 | **1.000000** |
| courbure / courbure exigée, T1W | 0.006 | **0.003** |
| courbure / courbure exigée, T2W | 0.051 | **0.054** |
| courbure / courbure exigée, T2FLAIR | 0.031 | **0.027** |
| part de variance du temps à la 1ʳᵉ couche | 0.00000 % | **0.00037 %** |

Le correctif multiplie bien le signal temporel par ~3700, et le réseau *essaie*
de s'en servir (poids par canal sur le temps ×1.34 → ×2.27 relativement au
latent). **Cela reste 0.0004 % de l'entrée.** Le défaut trouvé le matin (rang
effectif 1.22 sur 4) était réel et mesurable, mais ce n'était pas la contrainte
active.

**L'instrument qui manquait l'avait prédit avant la fin du run.** Un harnais
synthétique à réponse connue (`src/cfm/synthetic_marginals.py`, branché sur la
VRAIE boucle d'entraînement, 2 minutes par run) donnait en régime fidèle
`ts=1` → 0.001 et `ts=1000` → 0.016 : ×16, et toujours 50× sous la porte.

**Ce qui domine : concaténer au lieu de moduler.** Le temps pèse 256 canaux
contre 258 048 de latent. Le harnais montre qu'il faut ~6 % de la variance
d'entrée pour que la trajectoire se courbe ; y parvenir par concaténation
demanderait **~65 000 dimensions de temps**. En modulation (FiLM : `scale, shift`
appris par bloc résiduel), la force du conditionnement **ne dépend plus de la
dimension du latent** — 256 dimensions suffisent là où la concaténation en exige
2048.

C'est le mécanisme des deux références (MONAI additionne l'embedding dans chaque
resblock ; Genentech/MMFM a `sum_time_embed`), et **cela explique la mesure du
matin** : l'UNet variait de 1.7–2.6 % selon `t` quand le vectorisé était à 0.00 %.

**`adjacent_only` réhabilité.** Testé seul le 2026-08-26 et mesuré neutre
(34/60, p = 0.18) — sur un modèle aveugle au temps, incapable de courber sa
trajectoire quoi qu'on fasse de ses cibles. Dans le harnais : FiLM **sans**
adjacent échoue (0.516), FiLM **avec** adjacent franchit la porte (1.031).
**Les deux sont complémentaires, aucun ne suffit seul.** Le test du 26 était
valide ; sa conclusion ne l'était pas. Deuxième relecture de cette entrée.

**Balayage complet, régime fidèle (dim 4096, bf16, bruit à SNR constant) :**

| variante | courbure atteinte | porte ≥ 0.80 |
|---|---|---|
| `time_scale=1` | 0.001 | non |
| `time_scale=1000` | 0.016 | non |
| + `adjacent_only` | 0.053 | non |
| + `adjacent_only`, fp32 | 0.316 | non |
| + `adjacent_only`, 16 000 itérations | 0.584 | non |
| + `adjacent_only`, `time_embed_dim` 2048 | 1.019 | **oui** |
| + `adjacent_only`, **FiLM** (`time_embed_dim` 256) | **1.031** | **oui** |
| FiLM mais TOUTES les paires | 0.516 | non |

L'AMP bf16 coûte un facteur 6 ; ce n'est pas un problème de budget
d'optimisation (16 000 itérations ne suffisent pas).

**DETTE n°2 SOLDÉE** — loss finale contre « prédire zéro », unités brutes :
vectorisé 11.94 / 15.07 = **0.792** · UNet 7.89 / 15.07 = **0.524** · INR
3.50e-4 / 4.94e-4 = **0.713**. Les trois battent la constante nulle. L'UNet est
le meilleur sur cet axe et le pire au score final — **septième occurrence de
« la loss ne prédit rien »**.

**CORRECTION** — le flow INR n'avait pas 103 gradients nuls sur 250 :
`mmfm_core` journalisait `round(grad_norm, 4)`, donc toute norme < 5e-5
s'écrivait `0.0`, et le latent INR est à l'échelle 2.6e-4. Mesure directe à
l'initialisation : 2.7e-4 en bf16 comme en fp32. Journalisation passée à 4
chiffres significatifs.

**RÉSERVE sur la métrique.** En dimension 4096, le nRMSE du harnais est **plat**
(0.0114 à 0.0119) sur des variantes dont la courbure varie d'un facteur **53**.
Si la courbure s'améliore sur données réelles, **le nRMSE peut à peine bouger**.
Cohérent avec les 80 % d'erreur d'échelle d'intensité du 2026-08-25. Ne pas
conclure à l'échec sur le seul nRMSE.

**En cours** : R-best (`time_scale` 1000 + FiLM + `adjacent_only`) sur le
vectorisé ; évaluation Task 3 de R1.

## 2026-08-27 — CAUSE RACINE : le temps n'atteint pas le modèle

**Verdict : défaut trouvé, mesuré à chaque maillon, commun aux TROIS
architectures.** Détail : `results/mmfm/audit_20260827_refcompare/manifest.md`.

Comparaison demandée à `Genentech/MMFM` (le papier lui-même) et
`NVIDIA-Medtech/NV-Generate-CTMR`. Les trois codes utilisent *la même* fonction
`timestep_embedding` (origine `atong01/conditional-flow-matching`), avec
`max_period = 10000`. **Les deux références multiplient `t` par 1000 avant de
l'appeler ; nous passons `t ∈ [0,1]` brut.** Introduit au premier commit du flow
vectorisé (`2f96c13`), jamais revisité depuis.

Avec `t ∈ [0,1]`, l'argument des sinusoïdes ne dépasse jamais 1 : `cos ≈ 1`
partout, `sin(x) ≈ x`. L'embedding multi-échelle s'effondre en une rampe scalaire.

**La chaîne, mesurée maillon par maillon :**

| # | mesure | actuel | référence (×1000) |
|---|---|---|---|
| 1 | rang effectif de l'embedding aux 5 temps (max 4) | **1.22** | 3.98 |
| 1 | distance entre champs adjacents | 0.682 | 13.024 (**×19**) |
| 1 | σ3, σ4 en fp32 | 8.3e-3, **2.3e-4** | 9.00, 9.00 |
| 1 | σ3, σ4 en **bf16** (l'AMP de l'entraînement) | 1.2e-2, 8.1e-3 — *remontées* : le signal est remplacé par du bruit de quantification | inchangées |
| 2 | part de variance du temps à la 1ʳᵉ couche, EMA de production | **0.0000 %** (1.2e-7) | — |
| 3 | variation de `v` selon `t`, vectorisé | **0.00 %**, cos(v(0),v(1)) = **1.000000** | — |
| 3 | *idem selon le SUJET* (témoin) | *10.96 / 14.25 / 29.80 %* | — |
| 3 | variation de `v` selon `t`, UNet MONAI | 1.71 % / 2.59 % (sujet : 14.65 / 22.30 %) | — |
| 4 | courbure de la trajectoire / courbure exigée par les données | **1 à 5 %** | — |

**Le vectorisé de production est littéralement aveugle au temps** : sa vitesse
est numériquement identique à `t = 0` et à `t = 1`. Une vitesse constante intègre
une droite — et c'est ce qu'on observe : déviation à la corde 0.1T→7T de
0.005–0.023 aux ancres intermédiaires, quand les marginales réelles s'en écartent
de **0.37 à 0.90**. Les champs intermédiaires sont donc rendus comme des points
d'un segment entre 0.1T et 7T : **contraste faux et apparence moyennée, donc
lissée**. Les deux symptômes signalés, un seul défaut, les trois architectures.

Portée : `mmfm_vectorized.py` sert le vectorisé **et** l'INR (`arch_inr.py`
importe `build_vector_mmfm`) ; l'UNet passe le même `t` à MONAI, dont
`get_timestep_embedding` est la même formule. L'UNet s'en tire un peu mieux
(MONAI additionne l'embedding dans chaque resblock via un MLP appris, au lieu de
le noyer dans une concaténation à 258 432 canaux) mais reste ~9× plus sensible
au sujet qu'au champ visé.

**Second déséquilibre, même famille** : `latent_scale` n'existe que dans
`inr.yaml`. Le vectorisé et l'UNet tournent avec latent d'écart-type **21.98**
contre un embedding de temps à **0.027**. C'est l'image en miroir du bug INR
corrigé en phase B ; ce côté-ci n'avait jamais été regardé.

**Correctif** : une ligne dans `mmfm_vectorized.py`, une dans `arch_unet.py`.
**Mais il invalide les trois checkpoints** (le sens de l'entrée temporelle
change) et impose ~25 000 itérations de réentraînement chacun. **Non appliqué,
décision à prendre.**

> **Relecture du 2026-08-27 (soir).** Le correctif a été appliqué et mesuré :
> **il ne change rien à lui seul** (R1 : cos = 1.000000 inchangé, courbure
> 0.003–0.054). Le défaut décrit ici est réel — le rang effectif de 1.22 est
> mesuré — mais il n'était pas la contrainte active. Ce qui domine est le
> MÉCANISME de conditionnement (concaténer 256 canaux face à 258 048 au lieu de
> moduler). Voir l'entrée du soir.

**Écarts secondaires relevés, non testés** : loss L1 alors que les **9** scripts
d'entraînement de la référence utilisent MSE (dette n°7, corroborée de
l'extérieur) · **aucune guidance sans classifieur** chez nous, présente et
*sélectionnée* dans les deux références · interpolant : la référence enchaîne des
transports OT pour former une trajectoire couplée et y ajuste **une spline
cubique** (cible continue en `t`), nous tirons des droites par paire (cible
discontinue aux ancres) — en linéaire les deux coïncident, l'ingrédient non
testé est donc le **cubique** · `identity_prob: 0.1` apprend `v = 0` à un `t`
uniforme, sans équivalent dans aucune des deux références · NVIDIA dénormalise
l'IRM sur une plage globale **fixe** `[0, 1000]`, nous par percentiles.

## 2026-08-26 — `adjacent_only: true` : NÉGATIF, et le mécanisme est réfuté

**Verdict : neutre (34/60 paires, p = 0.18), et PIRE là où le gain était prédit.**
Détail : `results/mmfm/audit_20260826_adjacent/manifest.md`.

| nRMSE | T1W | T2W | T2FLAIR | moyenne |
|---|---|---|---|---|
| toutes paires (production) | 0.4353 | **0.3376** | **0.3654** | **0.3794** |
| adjacent seulement | **0.4326** | 0.3419 | 0.3691 | 0.3812 |

SSIM et LPIPS : identiques à 0.0002 près.

**La ventilation par longueur de saut inverse la prédiction** — c'est sur les
sauts longs que l'incohérence des droites devait mordre :

| longueur | n | toutes paires | adjacent | écart | victoires |
|---|---|---|---|---|---|
| 1 | 24 | 0.3211 | 0.3211 | +0.0000 | 13/24 |
| 2 | 18 | 0.4161 | 0.4156 | −0.0005 | 12/18 |
| 3 | 12 | 0.4298 | 0.4317 | +0.0019 | 7/12 |
| **4 (0.1T↔7T)** | 6 | **0.4022** | **0.4175** | **+0.0153** | **2/6** |

Monotone et inverse : plus le saut est long, plus la variante cohérente est
pénalisée. **Entraîner directement la transition longue vaut mieux que l'obtenir
en intégrant à travers la chaîne.** Le fait mesuré (marginales non alignées,
écart 0.72-1.32× la corde) reste vrai ; la conséquence que j'en tirais est fausse.

> **Relecture du 2026-08-27 — « le mécanisme est réfuté » était une erreur de
> lecture.** `adjacent_only` oblige le saut long à s'obtenir en intégrant un
> chemin *courbé* à travers les intermédiaires : précisément ce qu'un modèle
> aveugle au temps ne peut pas faire (mesuré depuis : cos(v(0),v(1)) = 1.000000).
> Entraîner la paire directement lui donne une cible en ligne droite qu'il *peut*
> représenter. L'échec sur le saut 4 est donc un **second symptôme** du défaut
> d'échelle du temps, pas une réfutation de la non-colinéarité. Le remède était
> bloqué par un défaut situé en amont de lui. À vitesse constante, « adjacent » et
> « toutes paires » convergent d'ailleurs vers la même droite — d'où l'écart de
> 0.0018, cohérent avec du bruit de run. Voir
> `results/mmfm/audit_20260827_refcompare/manifest.md` §6.

**Ordre de grandeur à retenir** : deux runs de 25 000 itérations de la même
architecture, ne différant que par ce drapeau, terminent à 0.3794 et 0.3812.
**Tout écart sous ~0.002 dans ce projet est indistinguable du bruit de run.**

## 2026-08-26 — Audit du code : le flow est entraîné sur des cibles contradictoires

**Verdict : un défaut structurel trouvé, une attribution obtenue, deux hypothèses
réfutées par la mesure.** Détail :
`results/mmfm/audit_20260826_order3/manifest.md`.

| # | piste | résultat |
|---|---|---|
| 1 | Marginales non alignées vs droites entre TOUTES les paires | **DÉFAUT** — le point intermédiaire s'écarte de la corde de **0.72 à 1.32 fois sa longueur** (2.3 à 5.6× le bruit). Les cibles d'entraînement se contredisent. `adjacent_only: true` existe et **n'a jamais été testé**. |
| 2 | D'où vient le flou de l'INR ? | **ATTRIBUÉ** — auto-reconstruction sans flow 0.036-0.090, pipeline complet 0.047-0.090, plafond 0.61. **Le flow n'ajoute aucun flou** ; aucun correctif de flow ne la rendra nette. |
| 3 | Loss L1 au lieu de L2 | **RÉFUTÉ** — écart théorique réel (le flow matching exige une espérance, donc un coût quadratique), mais mesuré : ‖médiane‖/‖moyenne‖ = 0.98-1.01 et la médiane conserve **106-144 %** des hautes fréquences. N'explique pas le lissage. |
| 4 | Interpolation finale `order=1` → `order=3` | **RÉFUTÉ** — le plafond monte fort (netteté 0.615 → 0.910, nRMSE 0.0382 → 0.0251 sur un volume parfait) mais **de bout en bout c'est neutre** : vectorisé 0.4353 → 0.4374 en nRMSE, 0.0983 → 0.0969 en LPIPS. Le modèle n'a rien à mettre dans la bande gagnée. Défaut laissé à 1. |

**Correction d'une conclusion antérieure** : « les modèles ne restituent qu'un
tiers de la finesse » confondait la **résolution de travail** et un défaut de
modèle. Rapporté au plafond atteignable à 1 mm, le vectorisé est à **66 % / 108 %
/ 50 %** — il le dépasse sur T2W. L'INR est à 12 %. Conséquence pour la décision
MedVAE-LPIPS : **un meilleur VAE à 1 mm ne peut pas franchir 0.61**.

**Vérifié sans défaut** : intégration d'Euler, convention d'intervalle
décodeur→dénormalisation, remplissage `reflect`, masque par le support de la
source (0.01-0.09 % d'énergie), tuiles disjointes sans moyennage.

## 2026-08-26 (soir) — Évaluation refaite sur l'INR corrigée

**Verdict : INR et vectorisé sont indiscernables en nRMSE ; le vectorisé reste
nettement devant en SSIM et LPIPS.** Détail :
`results/mmfm/qualitative_20260826/manifest.md`.

| mesure | résultat |
|---|---|
| Tests appariés, 60 paires | INR < vectorisé **29/60** (p = 0.65) ; INR < UNet 35/60 (p = 0.02 Wilcoxon) ; vectorisé < UNet 38/60 (p = 1.4e-4) |
| Gain sur le témoin identité | INR **+34.9 % T1W, −7.7 % T2W, +3.1 % T2FLAIR** — battue par la recopie sur T2W |
| Correction d'échelle oracle, INR | **70.4 % T1W / 30.4 % T2W / 66.4 % T2FLAIR** de l'énergie de l'erreur |
| Netteté, intérieur du cerveau (médiane) | INR **0.09 / 0.05 / 0.06** contre vectorisé 0.36 / 0.40 / 0.26, témoin 1.00 |

**Deux conclusions du 2026-08-25 sont renversées** — toutes deux parce qu'elles
portaient sur le modèle bogué :
- « Seuls 20-24 % de l'erreur INR sont d'échelle, sa limite est structurelle » →
  **66-70 % après correction**. Les trois architectures partagent la même limite
  dominante : la calibration d'intensité.
- « L'INR est la plus floue d'un facteur 1.5 à 2 » → **d'un facteur 4 à 8**. Le
  goulot de 1536 modulations, isolé sans confusion avec les artefacts de bord.

**Correction de framing** : le commit `d8db590` écrivait « l'INR devient première
en nRMSE ». Vrai de la moyenne agrégée, faux au sens statistique. Corrigé dans le
manifeste d'audit et dans l'état de référence de ce journal.

Le manifeste du 2026-08-25 (`results/mmfm/qualitative_20260824/manifest.md`) est
**annoté, pas effacé** : il porte ce qu'on croyait, et l'écart entre les deux
dates est lui-même un résultat.

## 2026-08-26 — Audit phase B : correctifs de fond `d8db590`

**Verdict : l'INR rejoint le vectorisé en nRMSE (0.3749 contre 0.3794) et reste
nettement dernière en SSIM et LPIPS.** Détail :
`results/mmfm/audit_20260825/manifest.md`.

> **Correction du 2026-08-26 (soir)** — le message de commit et le manifeste
> écrivaient « l'INR devient première en nRMSE ». C'est vrai de la moyenne agrégée
> et **faux au sens statistique** : 29 victoires sur 60 paires, p = 0.65 au test
> des signes. Les deux architectures sont indiscernables en nRMSE ; la moyenne est
> portée par quelques paires à fort écart. Mesuré dans
> `results/mmfm/qualitative_20260826/manifest.md`.

| correctif | test | résultat |
|---|---|---|
| EMA avec échauffement (`(1+t)/(10+t)`) | même modèle avec/sans EMA | 0.4070 vs 0.4055 — **du bruit** (l'écart valait 0.195 avant) |
| Standardisation de l'entrée du flow (`latent_scale: 2.16e-4`) | loss contre « prédire zéro » (4.02e-4) | 1.10e-3 → **3.37e-4**, soit 16 % sous la meilleure constante |
| Deux clés de config + garde-fou `_check_cache_consistency` | refus de tourner sur cache incohérent | testé dans les deux sens |
| `sample_points` graîné | deux ajustements du même volume | déterminisme vérifié (plancher de 4.4 % éliminé) |
| **Contrôle** : vectorisé avec orientation alignée | 20 paires × 3 sujets | 0.4351 vs 0.4353 — **neutre, les chiffres publiés tiennent** |

**Résultat négatif notable** : le réentraînement dégrade T1W (0.3787 → 0.4070)
tout en améliorant la moyenne (0.3824 → 0.3749). Dans un latent à faible structure
commune, un flow qui bouge davantage prend plus de risques qu'il n'en gagne.

**Plafond restant, mesuré et géométrique** : part de l'énergie portée par la
moyenne commune du nuage de latents — MedVAE **91.3 %** (dispersion 0.416), INR
**25.1 %** (dispersion 1.221, quasi isotrope). Ce n'est plus un bug.

**Non fait** : régénérer le cache INR sous le prétraitement corrigé (~6 h) ;
remplacer le test dont le seuil accepte le cassé.

## 2026-08-25 — Audit phase A : trois bugs d'inférence `d6ade93`

**Verdict : l'INR n'était pas limitée, elle était cassée. 0.6383 → 0.3787 sur
T1W, sans réentraîner.** Détail : `results/mmfm/audit_20260825/manifest.md`.

| test | résultat |
|---|---|
| A1 — INR sans EMA | 0.6383 → **0.4430**, 19/20 paires améliorées |
| A2–A4 — dérive du latent (10 volumes) | 0.691 → 0.305 (orientation) → 0.069 (normalisation) → 0.044 (plancher) |
| A5 — **témoin** : vectorisé sans EMA | 0.4353 → 0.4359, **insensible** |
| A6 — aller-retour de représentation à 1 mm | nRMSE 0.14 / 0.18 / 0.28 à 0.1T / 3T / 7T — **la représentation n'était pas en cause** |
| A7 — les trois correctifs ensemble | **0.3787**, devant le vectorisé sur T1W |

**Les trois bugs** : ① EMA sans correction de biais (`0.9999^25000 = 0.082`, soit
8.2 % du bruit initial servi à l'inférence — 216× le signal INR, négligeable pour
MedVAE) ; ② orientation **LAS→RAS** (chaque volume arrivait mirroré ;
`flip_lr_prob: 0.5` rendait le vectorisé et l'UNet insensibles, `arch_inr.py`
refuse le flip donc l'INR prenait tout) ; ③ normalisation (cache bâti sans
`field_norm_stats`, inférence en `field_fixed`).

**L'augmentation par flip masquait le bug d'orientation pour deux architectures
sur trois.**

## 2026-08-25 — Évaluation qualitative et quantitative des trois approches `45f1cf1`

**Verdict : 80 % de l'énergie de l'erreur est une erreur d'échelle d'intensité,
pas d'anatomie.** Détail : `results/mmfm/qualitative_20260824/manifest.md`.

| mesure | résultat |
|---|---|
| Correction d'échelle oracle (un scalaire par volume) | supprime **82 %** (T1W) et **83 %** (T2FLAIR) de l'énergie de l'erreur ; 39 % en T2W ; 20–24 % pour l'INR |
| Constante de recalibration par paire, estimée sans oracle (leave-one-subject-out) | 0.3794 → **0.3231**, −15 %, **sans réentraîner** |
| Témoin identité, gain paire-à-paire | +34.1 % T1W, **+5.6 % T2W, −1.2 % T2FLAIR** (battu sur 11 paires sur 20) |
| Sujet 0009, T1W@7T | norme L2 de 325 contre 845 et 705 → **46 % du « mur du 7T » vient d'un seul volume** |
| Indice de netteté (intérieur du cerveau) | vectorisé 0.36 / UNet 0.36 / INR 0.24, témoin 1.00 — **un tiers de la finesse réelle** |
| Estimateur d'échelle depuis le volume source | **dégrade** (0.4353 → 0.5642) : non biaisé mais trop bruité |

**Découverte structurelle** : les données d'entraînement sont **non appariées** —
aucun des 1056 sujets rétrospectifs n'existe à deux champs, et il n'y a que
**3 sujets appariés** dans tout le jeu. `n = 3` est un plafond, pas un raccourci.

## 2026-08-25 — MedVAE avec perte perceptuelle : la barre est franchie

**Verdict : LPIPS + PatchGAN bat le pré-entraîné sur les deux métriques et les
deux résolutions.** Détail : `results/benchmark_vae/analysis/medvae_lpips_20260825/manifest.md`.

| auto-reconstruction | @1 mm nRMSE / SSIM | @0.5 mm nRMSE / SSIM |
|---|---|---|
| pré-entraîné | 0.0819 / 0.9152 | 0.0488 / 0.9481 |
| fine-tuné L1 (écarté) | **0.0592** / 0.8790 | **0.0350** / 0.8837 |
| **fine-tuné LPIPS** | 0.0682 / **0.9727** | 0.0381 / **0.9825** |

**Le diagnostic du 2026-08-07 (« la recette de loss n'est pas la cause
dominante ») était faux** : même recette, LPIPS à la place de L1, +0.057 de SSIM
là où L1 en perdait 0.036. **Adoption NON décidée** — impose de régénérer les
caches et de réentraîner les deux flows (>1 jour), et n'agit que sur les 20 % de
l'erreur qui ne sont pas d'échelle.

## 2026-08-15 — Les trois architectures sur les trois contrastes `27f1153`

**Verdict de l'époque : le classement vectorisé > UNet > INR tient dans les neuf
cellules.** Détail : `results/mmfm/comparison_20260814_all_contrasts/manifest.md`.

> **⚠ INVALIDÉ le 2026-08-25** : ce classement mesurait trois bugs d'inférence de
> l'INR. La case INR/T1W mélangeait de plus deux modèles (0.6223 pour `z=4096`
> alors que le checkpoint de production est `z=129024`, soit 0.6383).
> L'écart vectorisé/UNet, lui, tient — mais il vaut un sixième de l'écart-type
> inter-sujet et n'est significatif que sur T1W (16/20, p=0.006 ; T2W 12/20 ;
> T2FLAIR 10/20).

## 2026-08-14 — Task 3 sur les trois contrastes : T1W n'était pas représentatif `2f00b5b`

**Verdict : deux tiers du problème n'étaient pas mesurés.** Détail :
`results/mmfm/comparison_20260814_all_contrasts/manifest.md`.

Vectorisé : T1W 0.4353, **T2W 0.3376**, T2FLAIR 0.3654 → moyenne **0.3794**.
**T1W est le contraste le plus difficile** ; le projet avait optimisé sur son pire
cas. Le « mur du 7T » (0.7542 en T1W) **n'existe pas en T2W** (0.3620).

**Prédiction réfutée** : la rareté de T2W@5T (43 volumes, la plus petite classe)
avait été annoncée comme cause probable de dégradation. Score obtenu : 0.3323, son
deuxième meilleur. La corrélation avec le volume de données est **inverse** —
T1W@7T est la classe la mieux dotée (235 volumes) et la plus mauvaise.

## 2026-08-14 — Bug d'équité du flip corrigé ; le classement est confirmé `b6efe83`

**Verdict : le flip est neutre (0.4353 contre 0.4354), le classement publié n'en
dépendait pas.** Détail : `results/mmfm/comparison_20260814_vectorized_flip/manifest.md`.

`flip_lr_prob: 0.5` était déclaré dans les trois configs mais **seul l'UNet le
recevait** — `FlatLatentCacheDataset` n'acceptait pas l'argument et l'avalait sans
erreur. Un latent retourné diffère de **26 %** en L2. Le bug handicapait le
gagnant, la conclusion était donc conservatrice — mais ce n'était pas mesuré.
Ce checkpoint devient la référence : seul reproductible depuis sa config.

## 2026-08-14 — Piège de l'encodeur sans cache `ea6e990`

L'entraînement sans cache encodait le volume entier là où le precompute encode par
tuiles : **1.5 à 3.7 %** d'écart sur le latent. Aligné via `_encode_like_cache`.

> Erreur de méthode commise en vérifiant : j'ai d'abord utilisé une normalisation
> par percentiles là où le cache utilise `field_norm_stats`, ce qui a donné un
> faux 10.6 % et la fausse conclusion que le correctif ne marchait pas.

## 2026-08-13 — UNet + conditionnement AdaGN : NÉGATIF `223e5f4`

**Verdict : aucun effet. 0.4623 contre 0.4617.** Détail :
`results/mmfm/comparison_20260813_unet_adagn/manifest.md`.

3/20 paires gagnées, écarts uniformément entre +0.0003 et +0.0010 y compris sur
5T et 7T que le conditionnement visait. **Les prédictions des deux modèles ne
diffèrent que de 0.2–0.3 % en L2** malgré 3.4 M paramètres d'écart et 2.4 points
de loss. Le conditionnement n'était pas le goulot.

**Témoin identité établi pour la première fois** : 0.9273. Aucun modèle du projet
n'avait jamais été comparé à « ne rien faire ».

> Piège rencontré : MONAI initialise `conv2` et la conv de sortie à zéro, donc le
> réseau sort exactement zéro à l'initialisation et **tout test de conditionnement
> passe trivialement**. Corrigé en réveillant les 140 tenseurs nuls avant mesure.

## 2026-08-12 — Décodeur implicite conditionné par grille : NÉGATIF

**Verdict : l'architecture fonctionne mais n'apporte rien.** Détail :
`results/mmfm/grid_inr_decoder/manifest.md`.

Capacité portée de 1536 à 8.3 M, et pourtant sur 76 volumes tenus à l'écart à
0.5 mm : conv+trilinéaire **0.1346 / 0.9154** contre INR **0.1511 / 0.8836**.
L'écart croît avec le champ. **Deuxième fermeture de la piste INR**, sur un design
très différent du premier.

**Leçon de méthode** : `infer_mmfm_unified.py` interpole 1 mm → 0.5 mm à l'ordre 1,
ce qui coûte 0.0156 de nRMSE. Le décodeur à grille était donc plafonné à ce gain —
**calculable en dix minutes avant d'implémenter quoi que ce soit.**

## 2026-08-10/11 — Capacité du latent INR : 4096 → 129024 : NÉGATIF

**Verdict : légèrement pire. 0.6223 → 0.6383, pour ~20 h de calcul.** Détail :
`results/mmfm/comparison_20260807_1mm/manifest.md`.

> **Explication trouvée le 2026-08-25** : le goulot n'est pas `latent_dim`.
> `hypernet.net.0.weight` est (512, 129024) et `hypernet.net.2.weight` (1536, 512) :
> tout est écrasé en aval sur **1536 modulations** pour 8.26 M voxels. L'expérience
> ne pouvait rien changer par construction. `latent_dim` nomme la taille vue par le
> flow, pas la capacité de la représentation — confusion qui a coûté une expérience
> entière.

## 2026-08-07/10 — Migration 2 mm → 1 mm et comparaison à trois

**Verdict : la résolution était le vrai goulot pour vectorisé et UNet, et l'INR
est la seule que 1 mm dégrade.** Détail : `results/mmfm/comparison_20260807_1mm/manifest.md`.

| @1 mm, T1W | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| Vectorisé | **0.4354** | **0.8995** | **0.0985** |
| UNet | 0.4617 | 0.8955 | 0.1014 |
| INR (`z=4096`) | 0.6223 | 0.8002 | 0.2051 |

L'INR passe de 0.4869 (@2 mm) à 0.6223 (@1 mm) : **+0.135**.

## 2026-08-01 — Comparaison finale à 2 mm, trois architectures

**Verdict : vectorisé 0.4288, UNet 0.4741, INR 0.4869.** Détail :
`results/mmfm/comparison_20260801_final/manifest.md`.

Trois candidats par architecture (50k, prolongation à plateau, prolongation à
décroissance immédiate). La prolongation à **schedule corrigé** gagne pour le
vectorisé (0.4361 → 0.4288) et **dégrade** l'UNet (0.4741 → 0.4766). Section
« la résolution est le vrai goulot » : c'est ce diagnostic qui a motivé la
migration à 1 mm.

## 2026-07-30 — Comparaison harmonisée vectorisé vs UNet

**Verdict : 0.4361 contre 0.4741, à code strictement identique hors architecture.**
Détail : `results/mmfm/analysis/protocol_20260730_harmonized_comparison/`.

Suite à la demande explicite d'harmoniser le code pour comparer équitablement.
L'écart se concentre sur les cibles hautes : →5T 0.4997 contre 0.5435, →7T 0.7220
contre 0.8209.

## 2026-07-29 — Baseline UNet multi-marginal, et le correctif de dénormalisation

**Verdict : la dénormalisation par champ cible divise l'erreur par plus de deux
sur les paires extrêmes.** Détail : `results/mmfm/analysis/protocol_20260729_unet_mm_baseline/`.

`0.1T→7T` : nRMSE **1.506 → 0.666**, SSIM 0.770 → 0.878. Aucune évaluation fiable
n'existait pour cette lignée avant cette porte. Triage de 4 modèles avant
évaluation complète (run2_local retenu, 2.456 contre 3.07–3.63).

## 2026-07-22 — MMFM v2 vectorisé, et clamp de la vitesse `5ef9bca` `37da6f4`

Première version vectorisée du flow multi-marginal. Le clamp `±1e6` ajouté pour
l'instabilité numérique.

> **Vérifié le 2026-08-25 : ce clamp est inerte pour les trois architectures**
> (vitesses de l'ordre de 4e-4 pour l'INR, 15 pour MedVAE — cinq à neuf ordres de
> grandeur sous le seuil). L'epsilon `max(dt, 1e-4)` du même commit a disparu dans
> l'unification, correctement : `|dt| ≥ 0.25` par construction.

## 2026-06 à 2026-07-20 — Fondations

Benchmark des VAE 3D (AEKL, VQ-VAE, RHVAE, MedVAE, NV-Generate), réorganisation de
`results/`, MMFM-UNet v2 à attention factorisée, scripts Jean Zay (DDP 4×H100),
inférence pleine résolution par patches, intégration de l'évaluateur officiel.
Détail dans l'historique git — aucun manifeste dédié pour cette période, et c'est
une lacune que ce journal existe pour ne plus reproduire.

---

## Dette connue, non traitée

| # | sujet | coût estimé |
|---|---|---|
| 1 | `test_inr_backbone_smoke.py` : seuil `nrmse_fg < 0.6` qui accepte le cassé, à 2 mm | 1 h |
| 2 | Aucun test qui compare la loss finale à « prédire zéro » — trois lignes, aurait tout arrêté | 15 min |
| 3 | Régénérer le cache INR sous le prétraitement corrigé | ~6 h GPU |
| 4 | Constante de recalibration d'intensité par paire (−15 % mesuré, sans réentraîner) : pas de données appariées pour l'ajuster hors des 3 sujets d'évaluation ; voie non testée = comparer les distributions d'intensité prédites et réelles, sans appariement | à instruire |
| 5 | Adoption du MedVAE perceptuel : régénérer les caches + réentraîner les deux flows | >1 jour |
| 6 | Géométrie du latent INR (25 % de structure commune contre 91 %) : canoniser l'ajustement | ~6 h |
| 7 | Loss L1 au lieu de L2 : écart à la dérivation du flow matching, effet mesuré nul sur les symptômes, à corriger par correction | 15 min + réentraînement |
| 8 | ~~Échelle du temps~~ **FAIT** (`model.time_scale`, défaut 1.0) — mesuré SANS effet seul | — |
| 8b | **Conditionnement par modulation (`time_cond: film`) — le facteur dominant.** Fait pour le vectorisé/INR ; reste à porter sur l'UNet (MONAI le fait déjà nativement) | fait + réentraînements |
| 9 | Standardisation du latent pour le vectorisé et l'UNet (`latent_scale` n'existe que dans `inr.yaml` ; latent σ=21.98 contre temps σ=0.027) | 10 min + réentraînement |
| 10 | Guidance sans classifieur : absente chez nous, présente dans les deux références | ~2 h + réentraînement |
| 11 | Interpolant cubique sur trajectoire couplée par chaînage OT (méthode réelle du papier MMFM) au lieu de droites par paire | ~1 jour |
