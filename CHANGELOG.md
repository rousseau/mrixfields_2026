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

## 2026-09-19/20 — Fine-tuning LOO, budget étendu (5000 itérations) : gain nRMSE réel, mais compromis SSIM/LPIPS sur T1W en `8win`

**Verdict : MIXTE, pas une nouvelle référence.** Détail :
`results/mmfm/finetune_loo_5k_20260919/manifest.md`.

Suite du fil `pro_pretrained` : le point d'arrêt à 1500 itérations était choisi
sur un proxy (nRMSE latent agrégé), jamais sur le score Task 3 par contraste.
Rallongé à 5000 itérations pour les 3 replis, checkpoint choisi PAR (repli,
contraste) sur le proxy latent (T2W inchangé, toujours production).

| | `cc` | `8win` (officielle) |
|---|---|---|
| ΔnRMSE vs best-of-both | **-0.0305** (p=0.0034) | **-0.0237** (p=0.0161 Wilcoxon, NS aux signes) |
| ΔSSIM | -0.0020 (NS) | **-0.0035 (p=0.029, PIRE)** |
| ΔLPIPS | +0.0024 (NS) | **+0.0055 (p=0.0015, PIRE)** |

**Le signal encourageant en `cc` ne tient pas en `8win`** : SSIM et LPIPS
deviennent significativement pires, alors qu'ils étaient neutres en `cc`.
Décomposition par contraste (`8win`) : **T2FLAIR est un gain propre** sur les
trois métriques (nRMSE -0.039, SSIM/LPIPS neutres) ; **T1W est un compromis**
(nRMSE -0.032, mais SSIM -0.0112 et LPIPS +0.0129, tous deux significatifs) —
probable sur-apprentissage sur les 2 sujets d'entraînement au-delà d'un
certain budget pour ce contraste, invisible en `cc` et au proxy latent.

**Décision** : ne pas remplacer la référence best-of-both par cette variante
« 5k partout ».

**Combinaison "mixte" testée dans la foulée (2026-09-20), sans coût GPU
supplémentaire** (recombinaison de prédictions déjà écrites) : T1W original
du best-of-both (1500 itérations) + T2FLAIR étendu (gain propre ci-dessus) +
T2W production.

| `8win`, 60 cellules | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **mixte** | **0.2962** | 0.8424 | 0.1971 |
| best-of-both (référence) | 0.3092 | 0.8421 | 0.1958 |

Seul T2FLAIR diffère (T1W/T2W identiques, 40/60 cellules égales) : isolé sur
ses 20 cellules, nRMSE **-0.0390 (Wilcoxon p=0.0107)**, SSIM neutre (p=0.67),
LPIPS pas significatif mais dans le mauvais sens (+0.0038, p=0.15, à
surveiller). **Gain net et propre, sans le compromis T1W. Nouvelle
référence pour le fine-tuning LOO**, remplace le best-of-both par repli.

**Recherche d'un compromis T1W intermédiaire (2026-09-20) : NÉGATIF.**
Hypothèse : le compromis nRMSE/SSIM-LPIPS de T1W pourrait s'atténuer à un
point plus tôt (testé : iter 3250-3500 au lieu de 3750-4750). Résultat : vs
T1W original, nRMSE toujours pas significatif (p=0.33) et SSIM/LPIPS toujours
significativement pires (p=0.0023/0.0017) ; vs T1W étendu, indiscernable sur
les 3 métriques. **Pas de zone intermédiaire exploitable** — le compromis
apparaît déjà pleinement avant iter 3250 et ne se dose pas en reculant.
Confirme le choix de garder T1W à son checkpoint original dans la référence.

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
`test_inr_backbone_smoke.py` assérait `nrmse_fg < 0.6` à 2 mm alors que la
production est à 1 mm. Il a laissé passer trois bugs pendant des mois.
**CORRIGÉ et MESURÉ le 2026-09-04** : le seuil est désormais relatif (battre la
moyenne leave-one-out) et un contrôle négatif prouve qu'il discrimine. Et la
preuve que l'ancien seuil acceptait le cassé est maintenant chiffrée, pas
supposée : un backbone à z=0 vaut 0.4443 (production) et 0.3942 (smoke), tous
deux **sous** 0.6, donc tous deux acceptés par l'ancienne porte. *(L'ancienne
justification comparait 0.6 au 0.6383 de Task 3 — deux grandeurs
incommensurables, voir l'entrée du 2026-09-04.)*

---

## 2026-09-18 (nuit, suite) — Recalibration d'intensité réessayée sur le checkpoint actuel : NÉGATIF, plus net qu'avant

**Verdict : dégradation significative, pas juste "sans effet".** Détail :
section « Suite — recalibration d'intensité... » de
`results/mmfm/vectorized_trajectory_srclevel_task3_20260918/manifest.md`.

Suite directe de la piste retrouvée ci-dessous (recalibration par paire de
champs, gain de −0.0212 sur l'ancien `vec_rbest`, jamais réadoptée). Réessayée
proprement sur le checkpoint de production ACTUEL (post-audit du 2026-09-04) :
table réestimée sur ce checkpoint (4 sujets `retro_train`/classe, 240
volumes), appliquée aux prédictions d'évaluation déjà écrites.

| protocole officiel `cc`, 60 cellules | nRMSE | SSIM |
|---|---|---|
| recalibré | 0.3893 | 0.8061 |
| production | 0.3549 | 0.8098 |
| Δ | **+0.0344** (p=0.014 Wilcoxon) | **-0.0038** (p=0.007) |

**Cause identifiée** : les facteurs de la nouvelle table sont tous proches de
1 (1.02-1.11 — ce checkpoint est déjà mieux calibré que l'ancien `vec_rbest`,
qui allait de 0.74 à 1.56). Comparés à l'oracle par cellule (calculé dans
l'entrée précédente), **6 des 15 cellules (contraste × champ cible) vont dans
le sens OPPOSÉ à l'oracle** — 3 sur 5 pour T2W, qui est aussi le contraste le
plus dégradé (+0.0676).

**Ce que ça ajoute à la clôture du 2026-08-30** : pas seulement "le remède est
insuffisant" mais "sa DIRECTION n'est plus stable d'un checkpoint à
l'autre" — un correctif de mécanisme a suffisamment déplacé le biais résiduel
pour qu'une table réestimée proprement pointe à l'envers de ce que les 3
sujets d'évaluation demandent sur près de la moitié des cellules. Piste
refermée plus fermement : aucun biais de classe stable à corriger sans
appariement, sur aucun checkpoint testé à ce jour.

---

## 2026-09-18 (nuit) — Décomposition de l'erreur restante : `level_cond` neutre sur la calibration, T2W est structurel, une piste de recalibration oubliée

**Verdict : deux découvertes, un rappel.** Détail : section « Suite — décomposition
de l'erreur restante » de `results/mmfm/vectorized_trajectory_srclevel_task3_20260918/manifest.md`.

Suite immédiate de l'entrée ci-dessous : au lieu de retenter un levier, décomposition
(`eval_quantitative_3arch.py --mode calibration`, sur les prédictions déjà écrites,
production `cc` et `level_cond`, 3 sujets × 20 paires) de brut → correction
"réalisable" (côté source seul) → oracle d'échelle.

| | brut | oracle d'échelle | % d'énergie retirée |
|---|---|---|---|
| T1W | 0.4578 → 0.4569 | 0.2535 → 0.2537 | **76.0 %** |
| T2W | 0.3043 → 0.3047 | 0.2663 → 0.2676 | **25.0 %** |
| T2FLAIR | 0.3627 → 0.3619 | 0.2176 → 0.2157 | **68.1 %** |

**1. `level_cond` ne change presque rien à la calibration finale** : les
facteurs d'échelle oracle par champ cible sont quasi identiques avant/après
(ex. T1W→7T : 0.78 les deux fois). Le réseau route le signal (mesuré dans
l'entrée précédente) sans que la sortie décodée n'en bouge la calibration
globale — cohérent avec le score de bout en bout inchangé.

**2. T2W est structurellement différent de T1W/T2FLAIR** : seulement 25 % de
son erreur est retirable par une correction d'échelle globale, contre 68-76 %
pour les deux autres — son facteur oracle par champ reste proche de 1
(0.92-1.23) là où T1W/T2FLAIR vont de 0.76 à 1.8. **Tout futur progrès
structurel a plus de chances d'être visible sur T2W** ; sur T1W/T2FLAIR il
serait noyé sous 68-76 % de bruit d'échelle.

**3. Rappel confirmé** : la correction "réalisable" (`subject_scale`, sans
oracle) est activement NUISIBLE sur les trois contrastes (-62 %, -36 %, -97 %
d'énergie *ajoutée*) — la voie sans appariement reste fermée (item de dette
#4, 2026-08-30).

**Piste ouverte retrouvée, jamais réadoptée** : `estimate_intensity_recalibration.py`
(constante PAR PAIRE, pas par sujet, estimée sans appariement sur 143 sujets
d'entraînement par classe) donne un gain SIGNIFICATIF et propre au vectorisé
— **0.3737 → 0.3525, p=0.014, 40/60** (entrée du 2026-08-30) — retiré de la
config de production uniquement parce qu'il dégrade l'UNet (+0.0654) et
l'INR (+0.0941), pour garder un réglage commun aux trois architectures. Cette
contrainde ne s'applique plus si l'effort se concentre sur le vectorisé seul.

---

## 2026-09-18 (soir) — Conditionnement du flow par le niveau de la source (`level_cond`) : signal appris, score inchangé

**Verdict : NÉGATIF.** Détail : `results/mmfm/vectorized_trajectory_srclevel_task3_20260918/manifest.md`.

Suite du fil ouvert plus haut (« le témoin qui manquait », 2026-09-04/05) :
`normalize_volume` détruit le niveau d'intensité observé de la source avant
que le flow ne le voie — un témoin trivial qui le lit directement bat les
trois architectures (nRMSE 0.2561 contre 0.3549 pour la production). Le
niveau est réinjecté par le même mécanisme FiLM déjà validé pour le temps
(`nn.Linear(1,32)` rejoignant le vecteur `cond` partagé), sur un cache
enrichi in-place (1939 échantillons, `cache_id` inchangé) et un entraînement
identique à la production sur tout le reste (25 000 itérations, même
couplage OT, même loss).

| protocole officiel `cc`, 60 cellules | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **avec `level_cond`** | 0.3547 | 0.8101 | 0.1928 |
| production (identique sinon) | 0.3549 | 0.8098 | 0.1928 |
| témoin `scaled_identity` | 0.2561 | 0.8707 | — |

Comparaison appariée : nRMSE 30/60 victoires (p=1.00), SSIM 28/60 (p=0.699),
LPIPS 31/60 (p=0.897) — **un pile ou face**, aucune métrique distinguable de
la production.

**Diagnostic qui va au-delà du chiffre nul** : contrairement au bug
« aveugle au temps » (cos(v(t=0),v(t=1))=1.000000 avant correctif), le réseau
n'ignore PAS ce signal. `cond_proj.weight` route `level_feat` avec une
magnitude du même ordre que le temps (0.0016-0.0018 contre 0.0031-0.0033), et
faire varier `level` de 0.10 à 0.49 à tout le reste fixé change le champ de
vitesse prédit de façon non triviale (cos = 0.949-0.9999, diff. relative de
norme 1.7-31.7 % sur 8 tirages). **Le signal est appris et utilisé — l'utiliser
ne referme simplement pas l'écart avec le témoin.** Troisième confirmation
indépendante, après le MedVAE perceptuel (2026-09-02, gain de représentation
absorbé) et l'audit du mécanisme (2026-09-04, mécanisme réparé, score
inchangé), que ce pipeline n'additionne pas les gains de ses composants — voir
la section « Plafond, mesuré et géométrique » et `project-plateau-huit-leviers`.

**Bug d'implémentation trouvé et corrigé en cours de route** : le pipeline
d'inférence Task 3 (`infer_mmfm_unified.py::process_volume_unified` /
`_infer_patch_unified`) est un chemin SÉPARÉ de `mmfm_core.py::infer()` — le
premier run a crashé (`ValueError: level_cond=True exige level`) parce que
seul le second avait été mis à jour. Corrigé (niveau calculé sur le volume
rééchantillonné non normalisé, avant `field_fixed`, transmis à
`euler_integrate`), vérifié sur un cas réel avant de relancer les 180 volumes.

**Non payé** : confirmation `8win` (~5h) — sans objet, Δ≈0 sous `cc` ne
laisse rien à confirmer.

---

## 2026-09-18 — Sélection du checkpoint par (repli × contraste) plutôt que par repli seul : gain marginal, non confirmé en `8win`

**Question posée** : le choix d'UN checkpoint par repli (sur le nRMSE latent
MOYENNÉ sur les 3 contrastes, cf. entrées précédentes) est-il optimal pour
T1W et T2FLAIR séparément, maintenant que T2W est de toute façon écarté au
profit de la production ? Réponse via les mêmes JSON de diagnostic
(`diagnose_finetune_checkpoints.py`, relancé), décomposés par contraste :

| repli | T1W retenu (agrégé) | T1W meilleur propre | T2FLAIR retenu (agrégé) | T2FLAIR meilleur propre |
|---|---|---|---|---|
| 0006 | iter 1050 : 0.2145 | iter 750 : **0.2021** | iter 1050 : 0.2122 | iter 1500 : **0.2046** |
| 0007 | iter 450 : 0.2062 | iter 450 (déjà optimal) | iter 450 : 0.3367 | iter 1500 : **0.2692** |
| 0009 | iter 900 : 0.2493 | iter 900 (déjà optimal) | iter 900 : 0.1945 | iter 900 (déjà optimal) |

Écart notable sur le repli 0007/T2FLAIR (0.0675 de nRMSE latent) — le choix
agrégé y est franchement sous-optimal.

**Recombinaison testée (`cc`, sans réentraînement)** : pour chaque (repli,
contraste), utiliser le checkpoint propre à ce contraste au lieu du
checkpoint agrégé du repli ; T2W toujours depuis la production (entrée
précédente).

| | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **sélection par (repli × contraste)** | **0.3090** | **0.8251** | 0.1804 |
| best-of-both (sélection par repli seul) | 0.3207 | 0.8238 | 0.1802 |
| production | 0.3549 | 0.8098 | 0.1928 |

**Vs best-of-both (repli seul)** : nRMSE 23/60, p=0.068 (à la limite) ; SSIM
25/60, p=0.036 (marginal) ; LPIPS 20/60, p=0.72 (aucun effet). **Vs
production** : les trois toujours significatifs, et le p de nRMSE se resserre
nettement (9.7e-05 contre 0.0012 pour la sélection par repli seul).

**Verdict : gain réel mais modeste**, essentiellement tiré par le repli
0007/T2FLAIR ; pas assez net pour justifier, à lui seul, le coût d'une
confirmation `8win` (~5h de calcul) — **décision explicite de ne pas payer
cette confirmation et de clore ce fil ici**, `cc` seulement. Le chiffre
`best-of-both` par repli (entrée du 2026-09-17) reste la version confirmée
sous les deux géométries et donc la référence à citer.

**Diagnostic.** Rechargé les 10 checkpoints par repli (déjà entraînés,
`diagnose_finetune_checkpoints.py` relancé pour la décomposition par
contraste, absente du premier passage) : **production est déjà excellente en
T2W** (nRMSE latent 0.177–0.199, bien meilleur que ses propres T1W/T2FLAIR) et
**AUCUN checkpoint, dans AUCUN des 3 replis, ne bat jamais production sur
T2W** — la dégradation apparaît dès la première sauvegarde (150 itérations) et
ne se résorbe jamais. Ce n'est donc pas un problème de point d'arrêt mal
choisi : n'importe quelle quantité de fine-tuning dégrade T2W.

**Piste explicative, cohérente mais pas prouvée au niveau de la paire** :
`diagnose_true_displacement.py` (déjà utilisé) montre que la part du VRAI
déplacement propre au sujet en T2W (0.452–0.480 sur les 4 paires depuis 0.1T)
est nettement plus basse ET plus resserrée que T1W (0.437–1.440) ou T2FLAIR
(0.477–0.918) — le transport T2W ressemble le plus à un décalage commun de
population, cohérent avec le fait que la production (entraînée sur 1053
sujets non appariés) l'ait déjà bien capturé, et que 30 volumes de fine-tuning
n'apportent alors que du bruit. Corrélation directe testée sur les 12 paires
depuis 0.1T (part propre vs delta nRMSE du fine-tuning) : **Pearson r=-0.217,
p=0.50 — direction cohérente, non significative** (n=12, sous-alimenté). Piste
plausible, pas établie au niveau de la paire individuelle.

**Correctif, sans réentraînement** : à l'inférence, choisir par CONTRASTE
plutôt qu'un seul modèle pour les trois — fine-tuning LOO pour T1W/T2FLAIR
(où il gagne), production pour T2W (où il perd toujours). Recombinaison pure
des prédictions déjà calculées (`cp` entre dossiers existants), zéro calcul
GPU supplémentaire.

| | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **best-of-both** (`8win`) | **0.3092** | 0.8421 | 0.1958 |
| fine-tuning LOO pur (`8win`) | 0.3320 | **0.8468** | **0.1945** |
| production (`8win`) | 0.3394 | 0.8313 | 0.2060 |

**best-of-both vs production, `8win`** : nRMSE p=0.0020, SSIM p=3.7e-05, LPIPS
p=3.3e-05 — **les TROIS métriques significatives**, contrairement au
fine-tuning pur (nRMSE non significatif, 32/60, p=0.51). Confirmé en `cc` :
nRMSE 0.3207 (p=0.0012), SSIM 0.8238 (p=2.0e-06), LPIPS 0.1802 (p=4.0e-06).

**Bilan** : la sélection par contraste n'est pas seulement un correctif de la
régression T2W, c'est un résultat strictement meilleur que les deux options
pures sur toutes les métriques testées — la première fois dans cette
investigation qu'une amélioration significative est obtenue simultanément sur
nRMSE, SSIM et LPIPS. Reste ouvert : la piste explicative (part propre au
sujet) mérite un test à plus grande échelle avant d'être considérée établie ;
et rien n'indique que T1W/T2FLAIR ne bénéficieraient pas eux aussi d'un arrêt
plus fin, spécifique à leur propre trajectoire de checkpoints plutôt qu'au
choix agrégé actuel.

---

## 2026-09-17 — Confirmation en géométrie `8win` (protocole officiel) : le résultat du fine-tuning LOO tient

**Suite immédiate de l'entrée précédente**, à la demande explicite de
confirmer sous la géométrie des chiffres officiellement publiés (`8win`, 8
fenêtres décalées fusionnées par Hann — pas `cc`). Chaîne complète relancée :
3 replis LOO (mêmes checkpoints : excl0006/checkpoint_1050,
excl0007/checkpoint_450, excl0009/checkpoint_900) + baseline production
fraîchement recalculée sur `model_final.pth` — 360 volumes, 5.2h pour la seule
baseline production (104 s/volume × 180, coût `8win` déjà documenté le
2026-09-07). Intégrité vérifiée : 0 fichier corrompu sur 360 (leçon du
2026-09-16 appliquée d'emblée).

| métrique | fine-tuning LOO | production (`8win`, frais) | delta | gagne sur | Wilcoxon p |
|---|---|---|---|---|---|
| nRMSE | 0.3320 | 0.3394 | −0.0074 | 32/60 | 0.508 (NS) |
| **SSIM** | **0.8468** | 0.8313 | **+0.0155** | **39/60** | **0.0031** |
| **LPIPS** | **0.1945** | 0.2060 | **−0.0116** | **44/60** | **1.2e-05** |

**Le motif tient sous le protocole officiel** : SSIM et LPIPS s'améliorent de
façon significative (p=0.0031 et p=1.2e-5), nRMSE progresse dans la même
direction sans atteindre la significativité (32/60, p=0.51) — légèrement plus
faible qu'en `cc` (34/60, p=0.13) mais qualitativement identique. Aucune des
trois métriques ne s'inverse de sens entre les deux géométries.

**Hétérogénéité par contraste, notable** :

| contraste | nRMSE fine-tune vs production | verdict |
|---|---|---|
| T1W | 0.3508 vs **0.4285** | net gain (le contraste historiquement le plus dur) |
| T2W | 0.3399 vs **0.2714** | **régression** — production déjà meilleure ici |
| T2FLAIR | 0.3053 vs 0.3184 | léger gain |

Le gain global en nRMSE est tiré par T1W et partiellement annulé par une
régression sur T2W — cohérent avec la non-significativité de la moyenne
agrégée, et avec le constat déjà ancien du projet que les trois contrastes ne
répondent jamais de façon homogène aux mêmes leviers (voir l'entrée du
2026-08-14, « T1W n'était pas représentatif »).

**Bilan de la piste `pro_pretrained`** : confirmé sous les deux géométries,
c'est le premier levier qui améliore SSIM/LPIPS significativement sans
dégrader le nRMSE (contrairement à H1/H2/MedVAE-LPIPS). Reste non tranché :
la régression T2W, le choix du point d'arrêt par repli (nRMSE latent, pas
Task 3), et si un budget de fine-tuning plus long ou une perte de contenu
(LPIPS/SSIM directe, comme le fait le baseline officiel) améliorerait encore
la marge — pistes pour une suite, non engagées ici.

---

## 2026-09-16 (soir) — Fine-tuning supervisé LOO sur `pro_train` (`pro_pretrained`) : premier résultat Task 3 positif de l'investigation — significativité en attente

**Motivation.** Suite directe de l'audit de référence du 2026-09-15 et de la
lecture du code officiel du challenge (`~/Code/MRIxFields2026/Baseline/`) :
jamais testé, la recette `pro_pretrained` recommandée pour les baselines GAN
(pré-entraînement non apparié + fine-tuning supervisé sur les 3 sujets
prospectifs réellement appariés). Voir le plan
`~/.claude/plans/actuellement-l-tape-de-repr-sentation-radiant-moonbeam.md`.

**Prérequis** : cache `pro_train` corrigé (entrée du jour, ci-dessous).

**Méthode** : nouveau `src/cfm/make_pro_train_loo_indexes.py` (3 index LOO,
30 entrées chacun) ; 3 configs `configs/mmfm/vectorized_finetune_loo_excl{0006,0007,0009}.yaml`
(copies de `vectorized_trajectory.yaml`, `train.resume`/`resume_weights_only`
depuis le checkpoint de production, `total_iters: 1500`, `ema_decay: 0.99`).
**Zéro nouvelle ligne dans `mmfm_core.py`** : le couplage OT chaîné existant,
sur `pro_train` (2 sujets par repli, vraie correspondance), retrouve
directement la VRAIE trajectoire de chaque sujet — déjà validé 180/180.
**Vérifié avant de lancer** : `lambda_edge_fwd`/`lambda_cycle` sont des
no-op garantis en `marginal_mode=trajectory` (`t_i=t_j=t_anchor`, rollout sur
`dt=0`) — retirés des configs, Phase 1 teste donc uniquement l'effet de la
vraie trajectoire sur la perte de flow matching elle-même.

**Porte rapide avant Task 3** (`src/cfm/diagnose_finetune_checkpoints.py`,
nRMSE latent DIRECT contre la vraie cible du sujet tenu à l'écart — pas un
proxy, contrairement au 2026-09-15) :

| repli (sujet exclu) | production | meilleur fine-tuné | itération |
|---|---|---|---|
| 0006 | 0.2723 | 0.2653 | 1050 |
| 0007 | 0.2717 | 0.2552 | 450 |
| 0009 | 0.2363 | 0.2126 | 900 |

Amélioration dans les 3 replis, non monotone dans le temps (attendu, budget de
30 volumes) — porte franchie, passage à l'éval Task 3 officielle.

**Éval Task 3 officielle (géométrie `cc`, agrégée sur les 3 replis — chaque
sujet noté par le modèle qui NE l'a PAS vu)** : nouveaux
`scripts/run_task3_eval_loo.sh` (inférence, écrit dans un dossier partagé —
`evaluate.py` exige les 3 sujets prospectifs présents et ne sait pas noter un
sous-ensemble) et `scripts/eval_task3_loo_aggregate.sh` (évaluation, une fois
les 3 replis déposés).

**Incident de données rencontré et corrigé** : le tout premier essai
d'inférence (repli 0006) a été interrompu par une extinction mémoire du
système (hors de notre contrôle) pendant l'écriture d'un volume ; `--skip_existing`
n'a pas détecté ce fichier `.nii.gz` tronqué comme invalide au réessai.
Repéré par le crash de `evaluate.py` (`EOFError` gzip), confirmé par une
vérification systématique des 180 fichiers (1 seul corrompu), régénéré. Leçon :
`--skip_existing` vérifie l'existence, pas l'intégrité — à surveiller après
toute interruption forcée d'un job d'inférence.

| | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **fine-tuning LOO** (T1W/T2W/T2FLAIR) | 0.3567 / 0.3450 / 0.3096 → **0.3371** | 0.8414 / 0.8237 / 0.8299 → **0.8317** | 0.1883 / 0.1758 / 0.1768 → **0.1803** |

**Comparaison rigoureuse, appariée** : le chiffre `cc` de production archivé
(0.3581/0.8379, entrée du 2026-09-05) n'est PAS strictement comparable — il
date de plusieurs semaines et sa provenance exacte (checkpoint, cache) n'est
pas ré-auditée ici. Baseline `cc` **fraîche**, recalculée aujourd'hui même sur
`outputs/mmfm/vectorized_trajectory/weights/model_final.pth` (même géométrie,
même 180 volumes, même `evaluate.py`) :
`outputs/mmfm/vectorized_trajectory/predictions_cc_baseline_20260916/` — 180
volumes vérifiés sans corruption. Test apparié (signes + Wilcoxon) sur les 60
cellules (contraste × paire) communes :

| métrique | fine-tuning LOO | production (frais, `cc`) | delta | gagne sur | Wilcoxon p |
|---|---|---|---|---|---|
| nRMSE | 0.3371 | 0.3549 | **−0.0178** (−5.0 %) | 34/60 | 0.133 (NS) |
| **SSIM** | **0.8317** | 0.8098 | **+0.0218** | **46/60** | **3.5e-06** |
| **LPIPS** | **0.1803** | 0.1928 | **−0.0126** | **41/60** | **6.4e-05** |

**Verdict : SSIM et LPIPS s'améliorent de façon hautement significative ; le
nRMSE s'améliore dans la même direction mais n'atteint pas la significativité**
(34/60, p=0.133 — même schéma qualitatif que H1 le 2026-09-15, mais dans le
bon sens). **C'est un renversement du motif établi trois fois de suite
(2026-09-02 LPIPS/MedVAE, 2026-09-05 mécanisme réparé, 2026-09-15 H1/H2)** :
« toucher le conditionnement/la représentation ne fait jamais progresser le
nRMSE de façon significative ET dégrade systématiquement SSIM/LPIPS ». Ici,
nRMSE ne progresse pas significativement (cohérent avec le motif), mais
SSIM/LPIPS **progressent** au lieu de se dégrader — la première fois dans
toute cette investigation qu'un levier améliore la qualité perceptuelle sans
la payer en distorsion.

**Incident de comparaison, corrigé en direct** : la première rédaction de
cette entrée comparait au chiffre archivé (0.3581/0.8379) sans le recalculer —
erreur de méthode signalée et réparée avant publication, pas après : toujours
recalculer la baseline sous EXACTEMENT le même run que la comparaison,
jamais réutiliser un chiffre historique par confiance.

**Limites, à instruire avant toute conclusion définitive** : (1) fine-tuning
sur 2 sujets par repli, budget 1500 itérations non poussé jusqu'à un plateau
identifié — le point d'arrêt optimal par repli reste approximatif (choisi sur
le nRMSE latent, pas sur le Task 3 lui-même, pour des raisons de coût) ; (2)
n=3 sujets pour le LOO — la significativité SSIM/LPIPS tient sur 60 cellules
mais seulement 3 sujets réels, donc 3 modèles fine-tunés indépendants ; (3)
geometrie `cc`, pas encore confirmé en `8win` (protocole des chiffres
officiellement publiés).

---

## 2026-09-16 — Cache `pro_train` mal encodé, corrigé ; les chiffres fondateurs (71.7 %, 180/180) tiennent

**Découvert en préparant un fine-tuning supervisé** (recette officielle
`pro_pretrained` du challenge, voir plus bas) : le cache
`medvae_finetune_c4d1e200/pro_train/` (utilisé par `diagnose_coupling_real.py`
et `diagnose_true_displacement.py`) a été écrit par `precompute_mmfm_latents.py`
(fenêtre glissante, `encode_scheme: medvae_sliding_window`), alors que
`retro_train` — sous le MÊME identifiant de cache — a été écrit par
`precompute_unet_latents.py` + conversion (**tuilé**, `encode_scheme: tiled`).
C'est exactement la collision documentée dans `flat_latent_cache_id`
(`src/common/dataset.py`) pour `medvae_finetune_1989e9d1`/`ff550d64` — sauf que
`pro_train` semble être resté sur l'ancien chemin sans jamais être régénéré,
malgré le correctif du 2026-09-07 qui a ajouté `encode_scheme` à l'index
(champ *backfillé*, donc déduit, pas re-mesuré — voir son propre avertissement).

**Corrigé** : `precompute_unet_latents.py --split pro_train` (tuilé, 45
volumes, 3.8 min) + `convert_unet_cache_to_flat.py`, puis copie dans
`medvae_finetune_c4d1e200/pro_train/` (ancien cache préservé dans
`pro_train.bak_sliding_window_20260916/`). `index.json` porte désormais
`encode_scheme: tiled`, cohérent avec `retro_train`.

**Impact sur les chiffres déjà publiés — mesuré, pas supposé** : les deux
scripts qui lisent `pro_train` ont été relancés sur le cache corrigé.

| mesure | ancien cache (sliding window) | cache corrigé (tuilé) |
|---|---|---|
| OT plein, appariement exact | 180/180 (1.000) | **180/180 (1.000), inchangé** |
| part du VRAI déplacement propre au sujet | 0.7173 | **0.7247** |

**Écart de 0.0074, dans le bruit.** Les deux mesures qui portent tout le
diagnostic du plateau (couplage parfait quand une correspondance existe ;
71–72 % du vrai déplacement propre au sujet) sont donc confirmées
indépendamment du schéma d'encodage — attendu, puisque les deux comparent des
champs entre eux AU SEIN du même cache, jamais contre `retro_train`. Le défaut
ne remettait rien en cause de fond, mais aurait contaminé toute comparaison
future entre `pro_train` et un modèle entraîné sur `retro_train` — exactement
ce qu'un fine-tuning supervisé sur `pro_train` s'apprête à faire, d'où la
correction avant de s'en servir.

---

## 2026-09-15 (soir) — Code de référence Genentech/MMFM, NON MODIFIÉ, sur les vrais latents : capture la composante propre au sujet que `mmfm_core.py` rate. Piste code/architecture ROUVERTE.

**Motivation.** Trois audits précédents (2026-09-04, 2026-09-14/15) comparaient à
des références en **réimplémentant** leurs idées dans `mmfm_core.py` (couplage OT
chaîné « à la Genentech », FiLM « à la NVIDIA »…). Aucun n'avait fait tourner le
**code d'un dépôt de référence tel quel**, bout en bout, sur les données réelles du
projet. Nouveau script : `src/cfm/diagnose_reference_mmfm.py`.

**Méthode.** `external/MMFM` (Genentech, vendorisé, **zéro ligne modifiée**) —
`MultiMarginalFlowMatcher.sample_location_and_conditional_flow`
(`multi_marginal_fm.py`) + `VectorFieldModel` (`models.py`) — entraîné sur les
MÊMES latents réels que la production (`medvae_finetune_c4d1e200/retro_train`),
via le MÊME couplage OT chaîné que `mmfm_core.py`
(`build_chained_ot_trajectories`, déjà validé : `diagnose_coupling_real.py`,
180/180). Seule dépendance manquante installée : `addict` (`pip install --user
addict`). Boucle d'entraînement copiée de
`external/MMFM/experiments/synthetic_data/train_mmfm.py:214-238` (MSE, Adam).

**Porte synthétique (harnais à réponse connue, avant tout calcul réel)** :
8000 itérations, `hidden_dim=128`, `lr=3e-4` → nRMSE **0.0579** (plancher 0.05),
courbure/vraie **0.92-1.06** (contre 0.02-0.21 à `lr=1e-3`/`hidden_dim=512` — la
première tentative, mal réglée, ne passait PAS la porte). L'adaptateur est
correctement câblé : le code de référence résout le problème à réponse connue.

**Entraînement réel** : `hidden_dim=512` (budget comparable au vectorisé de
production), `lr=3e-4`, `batch_size=32`, `grad_clip=1.0`, 4000 itérations,
24.6 min (`0.369 s/pas` — le spline scipy non modifié de la référence coûte
~1.2 s/pas à `batch_size=64` en dimension 129 024, d'où le budget réduit vs. les
25 000 itérations/2 h07 de production — limite mesurée et assumée, pas cachée).
Perte stabilisée ~1500-1650 dès ~1500 itérations.

**Résultat — part du déplacement propre au sujet** (même mesure que
`diagnose_flow_translation.py`, sur le modèle de référence, checkpoints tous les
500 pas) :

| itération | part propre au sujet |
|---|---|
| 500 | 0.6853 |
| 1000 | 0.6694 |
| 2000 | 0.7456 |
| 3000 | 0.5395 |
| 4000 | 0.6483 |
| **production (`mmfm_core.py`)** | **0.0348** |
| **R-best, mécanisme réparé (2026-09-05)** | **0.0825** |
| **VRAI déplacement** (`diagnose_true_displacement.py`) | **0.7173** |

**Stable dans la bande 0.54-0.75 dès le premier checkpoint (500 pas) et à
travers les 8 checkpoints** — ce n'est PAS un transitoire comme celui qui avait
trompé l'analyse du 2026-09-05 (« trois points pris dans un transitoire ne font
pas une tendance »). `cos(v(t=0),v(t=1))` reste pourtant très proche de 1
(0.9968-1.000 selon le contraste/checkpoint) : contrairement au mécanisme
« réparé » de production (`cos = -0.592`), ce n'est donc PAS une dépendance
temporelle forte qui explique l'écart, mais une dépendance à l'état COURANT `z`
que l'architecture de référence (MLP simple, concaténation brute
`[z, classe, t]`, sans FiLM ni `time_scale`) préserve et que `arch_vector.py` ne
préserve pas.

**Témoin de sanité** (magnitude/dispersion, pas de divergence) : `‖z_pred‖`
proche de `‖z_réel‖` sur toutes les cellules testées (ex. T1W→7T : 9090.6 contre
9168.7 ; T2FLAIR→3T : 9585.4 contre 9376.8), rang effectif prédit du même ordre
que le rang réel (8.6-13.4 contre 13.2-14.1). Le modèle ne diverge pas et ne
produit pas un nuage dégénéré — la composante propre au sujet mesurée n'est pas
un artefact d'explosion numérique.

**Ce que ceci établit** : sur des données et un couplage IDENTIQUES à la
production, un code de référence indépendant et non modifié capture une part du
déplacement propre au sujet du même ordre que le VRAI déplacement (65 % contre
71.7 %), là où `mmfm_core.py`/`arch_vector.py` n'en capture que 3.5-8.25 %. **La
thèse « 0 sujet apparié → composante individuelle inapprenable → limite des
données irréductible » ne peut donc plus être la cause UNIQUE** : la même
donnée, le même couplage, permettent visiblement d'apprendre davantage avec une
architecture différente.

**Ce que ceci n'établit PAS encore** : `own_share` élevé prouve que le flow de
référence n'est pas une translation pure, PAS qu'il transporte juste (vers la
bonne cible). Deux réserves explicites : (1) 4000 itérations est nettement moins
que les 25 000 de production en TERMES ABSOLUS (même si `own_share` est stable
sur les 4000 disponibles) — un entraînement plus long pourrait en principe
converger vers le même effondrement que la production a fini par atteindre à
25 000 itérations, comme celle-ci l'a elle-même fait de façon non monotone
(5.26 %→1.01 %→1.40 %→8.25 % aux itérations 2500/5000/7500/25000) ; (2) aucun
score Task 3 (nRMSE/SSIM/LPIPS officiel, image réelle après décodage MedVAE)
n'a encore été calculé pour ce modèle de référence — c'est la mesure qui
tranchera réellement, pas ce proxy géométrique.

**Suite immédiate — chiffre Task 3 officiel obtenu, et il RENVERSE la lecture
optimiste ci-dessus.** Nouveau script `src/cfm/infer_reference_mmfm.py` :
réutilise `process_volume_unified`/`_make_flow_spec`/`load_vae` de
`infer_mmfm_unified.py` tels quels (VAE, dénormalisation par champ, garde-fous),
seul `model`/`adapter` change (`VectorFieldModel` de référence + `LatentVectorizer`,
la MÊME classe de flatten que `arch_vector.py`). Géométrie `cc` (un seul crop
centré, pas `8win`) pour le coût — 6.5x plus rapide, biais connu et petit
(+0.0034 sur la région notée, mesuré le 2026-09-07). Task 3 complet : 3
contrastes × 20 paires × 3 sujets = 180 volumes, 49.4 min, checkpoint à 4000
itérations.

| | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **référence Genentech/MMFM (4000 itér.)** | T1W 0.3922 / T2W 0.3538 / T2FLAIR 0.3697 → **0.3719** | 0.8147 / 0.7724 / 0.7987 → **0.7953** | 0.2064/0.1990/0.2024 → **0.2026** |
| production `mmfm_core.py` (`vectorized_trajectory`, 25 000 itér., même recette d'inférence, 2026-09-05) | **0.3581** | **0.8379** | — |

**Verdict : NÉGATIF sur le score réel.** La référence, malgré une part du
déplacement propre au sujet 8 à 18× supérieure à `mmfm_core.py` en espace
latent (0.65 contre 0.035-0.08), est **légèrement pire en nRMSE** (+0.0138) et
**nettement pire en SSIM** (−0.0426) que la production. C'est la TROISIÈME
occurrence documentée dans ce projet du même motif (après le MedVAE perceptuel
du 2026-09-02 et le mécanisme réparé du 2026-09-05) : **un indicateur interne
amélioré ne se traduit pas en score amélioré, et peut même coexister avec un
score dégradé**. `own_share` mesure que le flow n'est pas une pure translation,
PAS que le déplacement propre au sujet qu'il ajoute est le BON — une variance
supplémentaire mal dirigée dégrade la SSIM (sensible au bruit structuré) sans
forcément aggraver le nRMSE global.

**Réserve majeure, non tranchée** : cette comparaison n'est PAS à budget
d'entraînement égal — 4000 itérations à `lr=3e-4` sans warmup/schedule, contre
25 000 itérations à `lr=2e-5` avec 2000 pas de warmup pour la production.
Au rythme mesuré ici (0.369 s/pas à `batch_size=32`), 25 000 itérations de la
référence prendraient ~2.6 h — un budget du même ordre que celui de la
production (2 h 07) et donc ATTEIGNABLE. Tant que ce budget n'est pas égalisé,
la question « l'architecture de référence transporterait-elle correctement si
elle était entraînée aussi longtemps ? » reste ouverte, et ce résultat négatif
ne peut pas encore être lu comme une clôture définitive de la piste
architecture.

**Suite — budget égalisé à 25 000 itérations (153.3 min, comparable aux 2h07 de
production), pour trancher la réserve ci-dessus.** Perte encore en baisse à la
fin (~1000, contre ~1500-1650 à 4000 itérations) — le modèle continue
d'apprendre, ce n'est pas un plateau. `own_share` **ne s'effondre PAS comme
production l'avait fait de façon non monotone** ; au contraire il MONTE et
DÉPASSE la vraie valeur :

| itération | part propre au sujet (référence) |
|---|---|
| 4 000 | 0.6483 |
| 5 000 | 0.7338 |
| 10 000 | 0.8443 |
| 15 000 | 1.0773 |
| 20 000 | 1.0935 |
| 25 000 | 1.0515 |
| **VRAI déplacement** | **0.7173** |

**Chiffre Task 3 officiel à 25 000 itérations (même protocole `cc`, 180
volumes, 49.5 min)** :

| | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| référence, 4 000 itér. | 0.3719 | 0.7953 | 0.2026 |
| **référence, 25 000 itér. (budget égalisé)** | **0.4295** | **0.7339** | **0.2349** |
| production (même recette d'inférence) | 0.3581 | 0.8379 | — |

**Verdict définitif sur cette piste : le score EMPIRE avec plus
d'entraînement**, alors même que `own_share` grimpe au-delà de la vraie
valeur (1.05-1.09 contre 0.7173). La réserve du budget est levée dans le sens
le PLUS défavorable à l'hypothèse architecture : ce n'est pas que la référence
avait besoin de plus de temps pour bien transporter — c'est qu'elle
**SUR-APPREND** les ~100-235 trajectoires par contraste disponibles (couplées
par OT chaîné, non i.i.d.), au point de produire, sur des sujets tenus à
l'écart (`Training_prospective`), une variance propre au sujet qui EXCÈDE la
réalité sans lui correspondre — du bruit dirigé par sujet, pas du signal.
`own_share` élevé n'a donc jamais mesuré un transport correct, seulement une
architecture prête à mémoriser plus finement un jeu de couplages OT restreint.

**Ce que l'ensemble de ce diagnostic établit** : deux architectures
indépendantes (celle du projet et celle de Genentech/MMFM, code non modifié),
sur les MÊMES données et le MÊME couplage, à budget d'entraînement sous- ET
sur-alloué, ne font JAMAIS mieux que la production — la référence sous-alloué
est proche (légèrement pire), sur-alloué elle est nettement pire. Une
architecture strictement plus expressive et sans le goulot FiLM/`time_scale`
de `arch_vector.py` ne débloque donc PAS le score ; elle expose au contraire à
quel point le signal individuel disponible (71.7 % du vrai déplacement, 86.5 %
orthogonal à la source, 0 sujet apparié sur 1056) ressemble à du bruit du
point de vue d'un modèle assez flexible pour l'ajuster. **La limite des
données reste, après cette contre-épreuve à deux architectures et deux
budgets, l'explication la mieux soutenue par l'ensemble des preuves
accumulées dans ce projet** — et cette fois avec un témoin d'architecture
totalement indépendant, testé aux deux extrêmes du budget, pas seulement une
réimplémentation locale des mêmes idées.

Scripts et artefacts : `src/cfm/diagnose_reference_mmfm.py`,
`src/cfm/infer_reference_mmfm.py` ; checkpoints dans
`outputs/mmfm/reference_genentech/real/` (`real_step004000.pth` et
`real_step025000.pth`, budgets bas/égalisé) ; logs
`outputs/mmfm/reference_genentech/{real_train,real_train_25k,infer_task3,infer_task3_25k,eval_25k}.log` ;
prédictions `outputs/mmfm/reference_genentech/predictions{,_25k}/task3/` ; CSV
d'évaluation officielle `results/mmfm/reference_genentech_task3_20260915/` et
`results/mmfm/reference_genentech_task3_25k_20260915/`.

---

## 2026-09-15 — BILAN : comparaison NVIDIA/NV-Generate-CTMR, H1 et H2 closes — deux pistes mesurées, ni l'une ni l'autre ne bat la production

**Point de départ.** La couche de représentation (MedVAE) ayant été validée
comme performante et bien réglée (voir les entrées de représentation
2026-09-06/07), le soupçon s'est porté sur l'étage de synthèse (flow
matching, `mmfm`). Une comparaison avec le dépôt NVIDIA/NV-Generate-CTMR
(rectified flow via MONAI, torchcfm) a identifié deux écarts structurels
réels avec notre implémentation :

1. **Absence de guidance conditionnelle** (dropout de label à l'entraînement
   + amplification `scale` à l'inférence) — piste **H2**.
2. **Conflation temps/sémantique** : `t` porte à la fois l'intégration ODE
   et l'identité du champ (`_field_to_time`), alors que dans torchcfm et
   NVIDIA `t` est une interpolation pure, toute sémantique passant par le
   conditionnement — piste **H1**.

Un troisième écart (loss L1 chez NVIDIA contre notre MSE) n'a pas été
retenu : déjà tranché par nos propres ablations antérieures.

## Ce que les deux pistes ont donné

| piste | mécanisme testé | nRMSE vs production (R-best, 0.3516) | SSIM | LPIPS | verdict |
|---|---|---|---|---|---|
| **H2** — `cond_dropout=0.1` seul (`gs=1.0`) | dropout de conditionnement à l'entraînement | 0.3367, indiscernable (Wilcoxon p=0.84) | **pire** (p=0.0027) | **pire** (p<1e-8) | NÉGATIF |
| **H2** — `+ guidance_scale=2.0` | amplification à l'inférence, même modèle | 0.3566, **pire que gs=1.0** (p=0.037) | pire | pire | NÉGATIF (la guidance aggrave) |
| **H1** — `marginal_mode: pairwise_ot` | `t` local + champ en conditionnement | 0.3209, amélioration NON significative (p=0.057) | **pire** (p=0.0002) | **pire** (p<1e-9) | NON ÉTABLI |

Manifestes complets : `results/mmfm/guidance_h2_task3_official_20260914/manifest.md`
et `results/mmfm/h1_pairwise_cond_task3_official_20260915/manifest.md`.

**Un motif se répète sur les trois lignes** : toucher le conditionnement ou
la représentation de classe du flow (dropout, guidance, ou champ en
conditionnement plutôt qu'en temps) ne fait JAMAIS progresser le nRMSE de
façon statistiquement établie, et dégrade systématiquement et
significativement SSIM et LPIPS. Ce n'est plus une coïncidence isolée mais
un troisième et quatrième point du même phénomène déjà noté ailleurs dans ce
projet (« améliorer un axe peut en dégrader un autre » — voir Volet 2,
2026-09-08).

**Leçon méthodologique acquise en cours de route (H2)** : un balayage
exploratoire sur un sous-ensemble réduit (3 paires × 1 sujet) avait suggéré
un optimum à `guidance_scale≈2.0` — ce signal s'est **inversé** une fois
mesuré sur les 60 paires officielles. Un sous-ensemble de cette taille ne
doit plus servir à choisir un hyperparamètre sans validation à l'échelle
complète.

## Ce que ce bilan établit, et ce qu'il n'établit pas

- **Établi** : ni l'ajout de guidance conditionnelle, ni la correction de la
  conflation temps/champ, ne suffisent à eux seuls à faire progresser le
  score de traduction dans les implémentations testées ici. Les deux
  mécanismes sont mécaniquement sains (validés séparément — H2 sur le
  harnais réel, H1 sur le harnais synthétique à réponse connue) : l'échec
  n'est pas un bug d'implémentation, c'est un résultat de mesure.
- **Non établi** : que la conflation temps/champ ou l'absence de guidance
  soient sans rapport avec la performance du flow — H1 en particulier montre
  un signal directionnellement cohérent (nRMSE meilleur sur 60% des paires)
  qui n'atteint simplement pas la significativité à budget d'entraînement
  inchangé. Les deux causes avancées (table de classe diluée en H1, absence
  de contrôle de cohérence perceptuelle en H1 et H2) restent non
  départagées.
- **Hors de portée de ce bilan** : le plafond de représentation reste bon
  (MedVAE validé) ; le goulot documenté ailleurs (`n=3` sujets appariés,
  écrêtage, plancher protocolaire) demeure la limite structurelle la mieux
  établie du projet.

## Clôture et suite

**H1 et H2 sont closes.** Le code des deux mécanismes reste dans le dépôt,
rétrocompatible et documenté, réutilisable pour toute reprise future
(budget d'itérations plus grand pour H1, régularisation de cohérence
perceptuelle pour l'un ou l'autre). Aucune reprise n'est engagée sans un
nouveau feu vert explicite — ni l'une ni l'autre piste n'a produit
d'argument suffisant pour justifier d'elle-même la dépense de calcul d'un
second cycle.

---

## 2026-09-15 — H1 : refonte pairwise (`t` local, champ en conditionnement) : NON ÉTABLI

Suite de la comparaison NVIDIA/NV-Generate-CTMR (voir l'entrée H2 ci-dessous) :
dans torchcfm et NVIDIA, `t` ne porte jamais de sémantique — interpolation
pure entre 2 points fixes, tout le reste en conditionnement. Notre design
(`marginal_mode: trajectory`) fait l'inverse : `_field_to_time` encode la
position du champ DANS `t`, et une spline cubique intègre en continu à
travers les 5 marginales. Testé : `marginal_mode: pairwise_ot` (nouveau,
rétrocompatible) — réutilise le MÊME couplage OT chaîné que `trajectory`,
mais `t` redevient local dans [0,1] entre exactement 2 marginales
(`_compute_flow(..., 0.0, 1.0, ...)`, sans rescale), et l'identité (champ
source, champ cible) devient conditionnement via `_flat_triple`
(num_classes 3×5×5=75, contre 3). Config :
`configs/mmfm/vectorized_pairwise_cond.yaml`, copie exacte de
`vectorized_trajectory.yaml` (production/R-best), une seule clé change.
Manifeste complet :
`results/mmfm/h1_pairwise_cond_task3_official_20260915/manifest.md`.

**Validé au préalable sur le harnais synthétique à réponse connue**
(`configs/mmfm/synthetic_pairwise_ot.yaml`, `scripts/eval_synthetic_pairwise_ot.py`) :
nRMSE 0.053 au bon conditionnement de paire de champs (plancher 0.05) contre
0.086 à un conditionnement délibérément FAUX (ratio 1.62×) — le mécanisme est
mécaniquement sain avant tout calcul sur données réelles.

**Évaluation officielle (20 paires × 3 sujets × 3 contrastes, 8win, appariée
contre R-best, nRMSE référence 0.3516)** :

| métrique | H1 | production | H1 meilleur sur | Wilcoxon p |
|---|---|---|---|---|
| nRMSE | 0.3209 | 0.3516 | 36/60 | **0.057 (à la limite, NON significatif)** |
| SSIM | 0.8272 | 0.8499 | 16/60 | **0.0002 (significativement pire)** |
| LPIPS | 0.2119 | 0.1880 | 6/60 | **<1e-9 (significativement pire)** |

**Verdict : NON ÉTABLI.** L'amélioration apparente du nRMSE agrégé (−8.7 %)
ne résiste PAS au test apparié (p=0.057, juste au-dessus du seuil ; par
contraste, H1 ne gagne que 11-13 paires sur 20 — direction cohérente, effet
trop faible et diffus pour conclure). SSIM et LPIPS se dégradent
significativement — même profil que H2 (`cond_dropout`) : un changement du
conditionnement/de la représentation de classe n'améliore pas clairement le
nRMSE tout en dégradant nettement le perceptuel. Deux causes non
départagées : la table de classe 25× plus grande (75 contre 3) pourrait
diluer l'apprentissage FiLM par entrée à budget d'itérations égal, et le CFM
par paire indépendante perd la contrainte de cohérence globale qu'imposait
la spline à travers les 5 marginales.

**Ce résultat ne tranche PAS l'hypothèse H1** (« `t` ne devrait pas porter de
sémantique ») — il montre seulement que cette implémentation précise (75
classes, budget d'itérations inchangé, sans régularisation de cohérence
inter-paires) ne bat pas la production sur l'ensemble des 3 métriques.
Mécanisme conservé dans le code (rétrocompatible, validé synthétiquement,
réutilisable). Reprendre supposerait un budget d'itérations plus grand ou une
régularisation de cohérence, ni l'un ni l'autre testés ici — en attente d'un
nouveau feu vert explicite.

---

## 2026-09-14/15 — Guidance conditionnelle (style CFG) sur le flow mmfm : NÉGATIF, et le balayage réduit a mal orienté

Suite à l'analyse comparative du repo NVIDIA/NV-Generate-CTMR (rectified flow
via MONAI) : leur conditionnement utilise un dropout de label à l'entraînement
(10 %) et une guidance `scale=15` à l'inférence, mécanisme absent de notre
flow. Testé sur le vectorisé de production (`vectorized_trajectory.yaml`,
copie exacte + `model.cond_dropout_prob: 0.1`). Code rétrocompatible
(`cond_dropout_prob=0.0`/`guidance_scale=1.0` = comportement historique
inchangé). Manifeste complet :
`results/mmfm/guidance_h2_task3_official_20260914/manifest.md`.

**Balayage exploratoire sur sous-ensemble réduit (3 paires × 1 sujet, NON
officiel)** : optimum apparent à `guidance_scale≈2.0` (nRMSE −1.4 % vs pas de
guidance). **Ce signal ne survit PAS à l'échelle officielle.**

**Évaluation officielle (20 paires × 3 sujets × 3 contrastes, 8win, appariée
contre R-best/production, nRMSE référence 0.3516 — la valeur
géométrie-vérifiée de `results/mmfm/geometry_20260907/`, pas l'ancien 0.3794
périmé)** :

| variante | nRMSE moyen | vs production | vs production |
|---|---|---|---|
| **production (R-best)** | 0.3516 | — SSIM 0.8499 | — LPIPS 0.1880 |
| **H2 `gs=1.0`** (dropout seul) | 0.3367 | indiscernable (p=0.84) | SSIM **pire** (p=0.0027), LPIPS **pire** (p<1e-8) |
| **H2 `gs=2.0`** (dropout + guidance) | 0.3566 | indiscernable (p=0.49) | SSIM **pire** (p=0.0004), LPIPS **pire** (p<1e-10) |

Et surtout : **`gs=2.0` contre `gs=1.0` (même modèle, seule la guidance
change)** — nRMSE **significativement PIRE** (23/60 paires, Wilcoxon
p=0.037), SSIM et LPIPS pires aussi (p<1e-6). La guidance n'aide sur AUCUN
axe une fois mesurée à l'échelle officielle.

**Verdict : NÉGATIF sur toute la ligne, et une leçon méthodologique.** Le
dropout de conditionnement seul ne touche pas le nRMSE mais dégrade
significativement SSIM/LPIPS ; ajouter de la guidance dégrade en plus le
nRMSE lui-même — l'inverse du signal observé sur le sous-ensemble réduit
(n=3 paires). **Un balayage exploratoire sous-alimenté a activement mal
orienté le choix d'hyperparamètre** : son optimum apparent s'est inversé en
dégradation nette une fois mesuré sur les 60 paires officielles. Ne plus
choisir un hyperparamètre sur un sous-ensemble réduit sans test apparié à
l'échelle complète avant adoption.

**Clôture de la piste H2.** Mécanisme conservé dans le code (rétrocompatible,
réutilisable), aucune configuration testée ne bat la production. H1 (refonte
pairwise, champ en conditionnement plutôt qu'en temps) reste documentée et
non engagée, en attente d'un feu vert explicite séparé — ce résultat ne se
prononce pas sur elle.

**Réserves** : un seul entraînement par variante (pas de répétition, variance
run-to-run non quantifiée) ; entraînement ralenti par partage GPU avec une
tâche locale indépendante (sans effet attendu sur la qualité, seulement la
durée) ; guidance implémentée seulement pour l'architecture vectorisée.

---

## 2026-09-14 — L'hétérogénéité d'échelle est ÉCARTÉE comme cause du désastre Task 3 de l'INR direct+LoRA16

Suite immédiate de l'entrée précédente. Deux causes plausibles y étaient
identifiées pour l'effondrement Task 3 : (1) hétérogénéité d'échelle par
dimension (×49.9), (2) rugosité de l'espace latent. Ce test isole (1) :
`VectorMMFM`/`arch_vector.py` acceptent désormais un vecteur `(latent_dim,)`
en plus d'un scalaire pour `latent_mean`/`latent_scale`
(`model.latent_mean_path`/`latent_scale_path`, rétrocompatible). Vecteur =
constante par groupe (shift vs LoRA), mesurée sur le MÊME cache — backbone et
precompute réutilisés tels quels, seul le flow réentraîné (25000 pas, 80 min).
Manifeste : `results/mmfm/inr_direct_lora16_pergroup_task3_20260913/manifest.md`.

**Résultat : AUCUN changement mesurable.** nRMSE 0.4586 contre 0.4587
(scalaire), SSIM 0.6906 contre 0.6939, duel paire à paire 33/60 — dans le
bruit. **L'hétérogénéité d'échelle n'est pas la cause.**

Reste la rugosité de l'espace latent, non testée directement : le diagnostic
dédié (`diagnose_inr_latent_smoothness.py`) s'est montré anormalement lent
sur cette architecture (>2h30 sans terminer, probablement le surcoût LoRA par
appel `decode`) et a été interrompu sans résultat exploitable.

**CLÔTURE (décision utilisateur, 2026-09-14) : piste fermée ici.** Ni la
rugosité (diagnostic à écrire), ni un rang LoRA plus petit (4 ou 8, nouveau
cycle complet ~50h) ne seront poursuivis pour l'instant. Bilan de la tentative
« modulation directe + LoRA rang 16 » sur ses trois volets (2026-09-10,
2026-09-13, 2026-09-14) : ~70h de calcul GPU sur 4 jours ; un bug de code réel
trouvé et corrigé (init LoRA morte) et conservé, réutilisable pour toute
reprise future ; deux hypothèses de cause posées, une écartée proprement
(échelle), une non tranchée (rugosité) faute d'outil de diagnostic adapté à
cette échelle ; le score Task 3 reste négatif et sévère dans toutes les
variantes testées. Rouvrir cette piste supposerait soit un diagnostic de
rugosité correctement dimensionné, soit un budget de calcul pour un nouveau
rang.

---

## 2026-09-13 — Suite du 2026-09-10 : extension Adam au precompute lancée, Task 3 NÉGATIF et SÉVÈRE (la traduction s'effondre malgré une représentation meilleure)

Reprise de la pause du 2026-09-10 : extension d'Adam (taux d'apprentissage
séparés shift/LoRA) au precompute ET à l'inférence (`fit_new_volume_adam`,
`src/cfm/inr_backbone.py`), backbone `hyper_hidden_dim=0`+`lora_rank=16`
entraîné en entier (50000 pas, 14h46), precompute complet (1939 volumes,
38.3h), flow entraîné (25000 pas, 53.5min). Un bug réel trouvé et corrigé en
cours de route : `fit_new_volume_adam` ne forçait pas `torch.enable_grad()`,
`RuntimeError` sous le `no_grad()` de l'inférence. Manifeste complet :
`results/mmfm/inr_direct_lora16_task3_20260913/manifest.md`.

**Porte de qualité (auto-reconstruction, 8 volumes) : EXCELLENTE.** Adam
(300 pas) : nRMSE_fg **0.1734**, gate 8/8, ratio z0/fit 2.5-2.7 — contre 0.2387
en SGD (mécanisme historique). Ce backbone représente MIEUX les volumes que la
production actuelle.

**Task 3 (traduction, géométrie `cc`) : NÉGATIF et sévère.**

| | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| Vectorisé (production) | 0.3794 | 0.8975 | 0.0941 |
| UNet (production) | 0.4033 | 0.8949 | 0.0953 |
| INR (production, `inr_std`) | 0.3749 | 0.8631 | 0.1557 |
| **INR direct LoRA16 (ce run)** | **0.4587** | **0.6939** | **0.3499** |

**SSIM sous le témoin identité (0.6939 contre 0.8883)** : dégradation
perceptuelle, pas seulement absence de gain. Gain vs identité **NÉGATIF** sur
T2W (−31.6%) et T2FLAIR (−38.7%) — pire que recopier la source. INR
(production) bat ce nouveau backbone sur 45/60 paires.

**Troisième instance du phénomène « Volet 2 »** (2026-09-08) : améliorer la
fidélité de représentation ne se transmet pas au score, et peut le dégrader
activement — ici de façon bien plus sévère, sur un axe différent (capacité du
latent, pas écrêtage). Deux causes plausibles, non départagées : (1)
hétérogénéité d'échelle par dimension mesurée à ×49.9 (contre ×2.9 pour
l'ancien cache MedVAE) mal servie par un scalaire unique de normalisation ;
(2) rugosité de l'espace latent (déjà mesurée ×2.2-2.9 sur l'ancien latent
1536-d, jamais retestée à cette échelle 30× plus grande, 46640-d) — chaque `z`
est ajusté indépendamment pour reconstruire SON volume, sans contrainte de
continuité entre volumes voisins que le flow doit pourtant traverser.

**Verdict : cette tentative (rang 16, normalisation scalaire, sans
régularisation) est close, NÉGATIVE.** Le correctif de code (bug LoRA du
2026-09-10 + bug `enable_grad` du 2026-09-13) reste acquis, rétrocompatible et
réutilisable. Pistes non testées : normalisation par sous-groupe shift/LoRA,
régularisation de lissage sur `z`, rang LoRA plus petit.

---

## 2026-09-10 — INR `hyper_hidden_dim=0`+`lora_rank=16` : bug d'init LoRA corrigé, mécanisme confirmé, PAUSE avant la production (l'optimiseur, pas la modulation, est le facteur limitant)

Tentative d'entraîner un backbone INR nativement en modulation directe
(bypass du hypernetwork, goulot de rang 512 déjà diagnostiqué le 2026-09-07)
+ LoRA rang 16 (~46640 valeurs/volume, même ordre de grandeur que le latent
MedVAE — le rang 64 du 2026-09-07, ~182k, avait été jugé trop grand pour le
flow matching). Manifeste complet : `results/mmfm/inr_direct_lora16_smoke_20260910/manifest.md`.

**Bug trouvé et corrigé AVANT tout calcul de production** : `INRBackbone.fit_latent`
initialisait toujours `z` à zéro ; avec `hyper_hidden_dim=0`, `z` EST `gamma`,
donc les tranches LoRA `A` ET `B` démarraient à zéro — gradient LoRA mort des
deux côtés, pour toujours. Confirmé empiriquement : `lora_rank=16` sans
correctif (nRMSE_fg=0.2082) est indiscernable de `lora_rank=0` (0.2070).
Corrigé dans `src/cfm/inr_backbone.py` (init B aléatoire + taux d'apprentissage
LoRA séparé, convention déjà validée par `bench_inr_capacity.py`) —
**rétrocompatible, vérifié bit-à-bit pour `lora_rank=0`** (tout le code de
production actuel).

**Après correctif, le gain reste marginal avec la SGD de production**
(0.2053-0.2132 selon `lora_inner_lr`, ≤0.8% vs témoin) et n'a **pas plateauné**
à 500 pas (0.2002) — mais **Adam avec taux d'apprentissage séparés bat la
meilleure SGD de 30.7% relatif en 6.3x moins de temps** (300 pas, nRMSE_fg
0.1387 contre 0.2002, 70s contre 442s). **Le mécanisme LoRA est vivant et
capable ; c'est l'optimiseur à pas fixe de la boucle interne de production
(`fit_latent`) qui est mal adapté à un espace de modulation aussi grand,
pas la modulation directe+LoRA elle-même.**

**Décision (avec l'utilisateur) : PAUSE.** Basculer le precompute vers Adam
est une extension de périmètre non approuvée (coût recalibré à ~13-38h de
precompute contre 6-7h prévues) — le correctif de code est conservé (utile
pour toute future tentative), mais l'entraînement de production (50000 pas),
le precompute, le flow et l'évaluation Task 3 n'ont **pas** été exécutés.

---

## 2026-09-09/10 — Figures qualitatives INR vs MedVAE, checkpoints ACTUELS (les anciennes dataient d'avant l'audit EMA/orientation)

Les seules figures INR vs MedVAE du dépôt (`comparison_20260801_final/
figures_representation_capacity/`) montraient un INR périmé : `z=4096`
(remplacé par 129024 le 2026-08-10), **antérieur** à l'audit du 2026-08-25
(EMA non chauffée + orientation LAS/RAS). Script :
`src/cfm/figures_representation_capacity_current.py` (aucun chemin
parallèle — appelle directement `bench_representation.roundtrip`/
`bench_inr_capacity.load_production_backbone`). Manifeste complet :
`results/mmfm/qualitative_representation_20260909/manifest.md`.

**Jeu A** (9 cellules, T1W × {0006,0007,0009} × {0.1T,3T,7T}, auto-reconstruction
1mm normalisée) : moyenne MedVAE nRMSE **0.0518**/SSIM **0.9528** contre INR
déployé (20 pas SGD, procédure de production) nRMSE **0.1292**/SSIM **0.8046**.
Confirme visuellement le lissage de l'INR déployé sur les 9 cellules.

**Jeu B** (3 cellules, sujet 0006, même SIREN gelé, modulation LoRA r=64
optimisée par Adam à la place des 20 pas SGD de production) :

| | MedVAE | INR déployé (512 eff.) | INR plafond (LoRA r=64, ~182k eff.) |
|---|---|---|---|
| nRMSE moyen | 0.0587 | 0.1448 | **0.0531** |
| SSIM moyen | 0.9551 | 0.8057 | 0.9351 |

**À budget comparable, l'INR plafond bat MedVAE pré-entraîné en nRMSE** —
confirme visuellement (pas seulement en CSV) la conclusion du 2026-09-07 :
le handicap de l'INR déployé est un budget de modulation (512 valeurs/volume),
pas un défaut de la famille SIREN. Le plafond LoRA n'est pas déployable tel
quel (~182k valeurs/volume, ~5-10 min d'Adam par volume ici) — c'est une borne
de capacité, pas une proposition d'architecture.

---

## 2026-09-08 — **Volet 2 : NÉGATIF, et pour la première fois le mécanisme est nommé. Améliorer la représentation a rendu le travail du flow PLUS DUR.**

**Verdict : −21 % de plafond de représentation ont produit +7.4 % de score.** Ce n'est
plus une non-transmission comme les 2026-09-01 et 2026-09-02 : c'est une
**anti-corrélation**, et la décomposition la localise. CSV :
`results/mmfm/hi125_20260907/`. Config : `configs/mmfm/vectorized_l1_hi125.yaml`.
Cache : `medvae_finetune_6109b0b0` (1939 volumes, chemin TUILÉ, 2 h 50).

Recette d'inférence partagée avec les repères : crop centré, tranche notée [150, 180),
180 volumes, table `field_norm_stats_hi125.json` à la dénormalisation.

| tranche notée | T1W | T2W | T2FLAIR | nRMSE | SSIM |
|---|---|---|---|---|---|
| **LPIPS + `hi` ×1.25** | 0.4579 | 0.3147 | 0.3499 | **0.3742** | **0.8334** |
| l1 + 4 correctifs (repère) | 0.4249 | 0.3018 | 0.3189 | **0.3485** | 0.8220 |
| production | 0.4034 | 0.3215 | 0.3493 | 0.3581 | 0.8379 |
| *témoin srclevel* | | | | *0.2561* | — |

**+0.0257 de nRMSE (+7.4 %), payé par un gain de SSIM de +0.0114** contre le repère.
Le SSIM reste sous la production (−0.0045). Les trois contrastes reculent en nRMSE.

### Test apparié, et le contrôle qui le rend lisible

La référence a été **ré-évaluée aujourd'hui** sur ses prédictions stockées, par le même
évaluateur et la même région : **0.3485**, exactement la valeur archivée. Le contrôle
passe, la comparaison est propre.

| | valeur |
|---|---|
| nouveau meilleur sur | **21/60 paires**, signes **p = 0.027** |
| écart moyen | **+0.0256** |
| écart **médian** | **+0.0094** |
| min / max | −0.0632 / **+0.2194** |
| SSIM : nouveau meilleur sur | **46/60 paires**, +0.0115 |

La dégradation est significative, mais la moyenne vaut près de trois fois la médiane :
**elle n'est pas diffuse, elle est portée par quelques paires.**

### 79 % de la dégradation vient d'UN SEUL champ source : 3T

| champ SOURCE | écart moyen | dégradé sur |
|---|---|---|
| 0.1T | +0.0167 | 7/12 |
| 1.5T | −0.0021 | 7/12 |
| **3T** | **+0.1008** | **10/12** |
| 5T | +0.0136 | 9/12 |
| 7T | −0.0009 | 6/12 |

Les 12 paires de source 3T pèsent **+1.2100 sur +1.5378**, soit **79 %**. Sans elles,
l'écart moyen tombe à **+0.0068** — trois fois le plancher de bruit, mais quatre fois
moins que le chiffre global. Les sept pires paires sur huit ont 3T pour source
(T2FLAIR 3T→7T +0.2194, T1W 3T→5T +0.2023, T1W 3T→7T +0.1961…).

**Et 3T est exactement là où l'ancienne table écrêtait le plus** : la mesure du
2026-09-07 donne T2FLAIR@3T **50.0 %** et T1W@3T **37.1 %** de voxels de premier plan
saturés — les deux plus fortes valeurs des 15 cellules. **Les cellules que le correctif
change le plus sont celles qui cassent le plus.** Le mécanisme n'est pas établi, mais la
corrélation est nette et elle désigne l'expérience suivante : un `hi_scale` **par
cellule** plutôt qu'un facteur uniforme — les cellules peu écrêtées n'avaient rien à
gagner et ont quand même payé le changement de distribution.

### Ce n'est PAS un entraînement raté, et ce n'est PAS l'amplification d'échelle

Deux contrôles écartent les explications faciles.

**Le mécanisme du flow est intact, et à un cheveu du repère.** La loss n'est pas
comparable entre caches (échelles de latent différentes) mais son RATIO à la base
« prédire zéro » l'est : **0.621 contre 0.623** pour le repère. Porte mécanistique
5/5. Entraînement 22.45 → 14.87 en 120.9 min à 3.49 it/s, identique au repère.

**L'amplification par `hi` ×1.25 est déjà comptée dans le plafond.** Le bras
`lpips_hi125` du banc fait l'aller-retour COMPLET, dénormalisation incluse, et il
s'améliore (0.1140 → 0.0901). Le facteur 1.25 sur la sortie n'explique donc pas la
dégradation.

### La décomposition, qui localise le défaut

Convention du projet, `score² ≈ plafond² + flow²` :

| | score | plafond | terme de flow |
|---|---|---|---|
| l1 + 4 correctifs | 0.3485 | 0.1140 | **0.3293** |
| LPIPS + `hi` ×1.25 | 0.3742 | 0.0901 | **0.3632** |
| écart | +7.4 % | **−21.0 %** | **+10.3 %** |

> **Le plafond baisse de 21 %, le terme de flow monte de 10 %, et le second l'emporte.**

C'est un résultat neuf. Les deux nulls précédents laissaient ouverte l'idée que le
gain de représentation était simplement « absorbé ». Ici il est **plus que compensé** :
les latents améliorés sont plus difficiles à transporter. Deux causes candidates, non
départagées :

1. **Occupation de la dynamique.** `hi` ×1.25 fait passer l'écart-type du foreground de
   0.406 à 0.347 et `latent_scale` de 18.125 à 15.924 (−12 %). Le déplacement entre
   champs que le flow doit produire rétrécit avec le signal, alors que son erreur
   propre, elle, ne rétrécit pas forcément.
2. **Géométrie des latents perceptuels.** Le MedVAE-LPIPS produit un latent d'une autre
   distribution ; rien ne dit qu'il soit aussi linéairement transportable. Le 2026-09-02
   avait déjà noté que « la loss n'est pas comparable entre VAE ».

**L'expérience qui les sépare, et elle est bon marché** : réentraîner avec le
checkpoint LPIPS SEUL (table `hi` ×1.0). Le cache LPIPS existe déjà (`1989e9d1`) mais il
est encodé par la fenêtre glissante et non tuilé — donc inutilisable pour une
attribution propre. Coût : une régénération de cache (2 h 50) plus 2 h d'entraînement.

### Ce que cela dit du plafond comme guide

**Troisième fois, et la plus nette : le plafond de représentation ne prédit pas le
score.** 2026-09-01 : −6.9 % de plafond annoncés → +0.0004. 2026-09-02 : −0.0072 →
+0.0004. 2026-09-08 : **−21 % → +7.4 %, de signe opposé.**

Le plafond mesure ce qu'un flow PARFAIT obtiendrait. Il est donc muet sur la difficulté
que la représentation impose au flow réel — et cette campagne montre que les deux
peuvent bouger en sens contraire. **Ne plus arbitrer une adoption sur le plafond seul.**

### Ce qui reste acquis de la validation du 2026-09-07

Rien de ce résultat n'invalide la campagne de validation : MedVAE est bien paramétré,
la table sature bien 25–50 % des voxels de cerveau, le plancher protocolaire tombe bien
de 60 à 71 % quand on desserre l'écrêtage, et l'INR est bien affamé et non mal réglé.
Ce qui tombe, c'est **l'inférence de ces faits vers le score**.

### Réserves

- **Deux variables à la fois** (checkpoint LPIPS et `hi` ×1.25), par choix assumé pour
  tenir dans une journée. L'attribution entre les deux n'est pas faite. La localisation
  sur 3T pointe plutôt vers `hi` ×1.25 que vers le checkpoint, mais ne le démontre pas.
- Le plafond 0.0901 est mesuré sur 15 volumes / 1 sujet, le score sur 180 volumes /
  3 sujets. La décomposition structurelle mélange donc deux échantillons.
- L'orthogonalité `score² ≈ plafond² + flow²` est une convention du projet, pas un
  théorème.
- Incident de conduite : la chaîne est morte d'une `erreur de syntaxe ligne 157` parce
  que j'ai édité `run_volet2_chain.sh` PENDANT son exécution — bash lit ses scripts par
  décalage d'octets. Les 180 prédictions étaient intactes, l'évaluation a été relancée à
  la main. **Ne jamais éditer un script en cours d'exécution.**
- `measure_flow_geometry.py` (porte [5b], géométrie des poids réels) n'a pas tourné,
  emportée par le même incident. À lancer.

---

## 2026-09-07 (nuit) — **Les 8 fenêtres : l'alarme est levée sur le nRMSE, et un LEVIER apparaît là où on ne le cherchait pas**

**Verdict en deux temps.** Le tableau de référence n'est pas corrompu — l'écart entre
les deux géométries vaut **+0.0005 en région `full`** (26/60 paires, p = 0.37), quatre
fois sous le plancher de bruit. Mais sur la région **réellement notée**, moyenner 8
générations décalées vaut **0.0034 de nRMSE** (12/60 paires, p = 3.2e−06) — plus que la
plupart des effets arbitrés ce dernier mois. Détail :
`results/mmfm/geometry_20260907/manifest.md`.

**Une seule variable** : même checkpoint R-best, même config, même table de
normalisation, mêmes 180 volumes. Seul `--center_crop_only` change. Ré-inférence
complète en 0.80 h à **16.0 s/volume**, contre 104.1 pour le run d'origine — le rapport
6.5× attendu de 8 passes, ce qui confirme le diagnostic par construction.

**Contrôle passé avant toute comparaison** : les prédictions STOCKÉES rejouées sous
l'évaluateur d'aujourd'hui, région `full`, redonnent **0.4441 / 0.9042** en T1W, soit
exactement le chiffre publié le 2026-08-27. L'évaluateur n'a pas bougé.

### Les deux régions

| | 8 fenêtres | crop centré | écart | signes |
|---|---|---|---|---|
| **nRMSE, région `full`** | **0.3737** | **0.3742** | **+0.0005** | 26/60, p = 0.37 |
| SSIM, région `full` | 0.9047 | 0.8987 | −0.0060 | |
| **nRMSE, région `slab`** | **0.3516** | **0.3550** | **+0.0034** | **12/60, p = 3.2e−06** |
| SSIM, région `slab` | 0.8499 | 0.8394 | −0.0105 | |

### Ce que ça règle

**L'avance de R-best sur la production (0.0057) tient** : la géométrie n'en explique que
9 %. « R-best est le meilleur flow » (`AGENTS.md`) reste établi. Et le plafond de
représentation 0.1048 reste comparable aux scores en région `full`.

Reste que le SSIM et le LPIPS se déplacent de 0.006 et 0.006 en `full`, de 0.010 et
0.010 en `slab` : **ces deux métriques-là ne sont pas comparables entre les deux
familles de runs**, et l'inventaire est de 15 runs sur 30 dans chaque camp.

### Ce que ça ouvre, et qui n'était pas la question

> Moyenner 8 générations décalées est une **réduction de variance par ensemble** qui
> paie 0.0034 sur la région notée, avec p = 3.2e−06.

À situer : flip 0.0001, AdaGN 0.0006, plancher de bruit 0.002, ordre 3 0.0031, les 4
correctifs du flow 0.0031, **les 8 fenêtres 0.0034**, l1 contre mse 0.0064.

C'était utilisé par accident depuis le 2026-08-27. **C'est maintenant un choix** :
`scripts/run_task3_eval.sh` prend la géométrie en 6ᵉ argument, défaut `8win`, et
l'imprime en tête de run. Coût 6.5× le temps machine.

**Réserves.** Un seul checkpoint ; le mécanisme est générique mais son ampleur dépend de
la corrélation des erreurs entre fenêtres. Le nombre de fenêtres (8) et les décalages
(−16, +7) viennent de la grille par défaut, jamais balayés. Et le LPIPS va en sens
inverse du nRMSE et du SSIM — c'est un lissage, pas un gain de netteté.

---

## 2026-09-07 (soir) — **Le tableau de référence compare DEUX géométries d'inférence. Vérifié par la durée des runs.**

> **SUITE MESURÉE LE MÊME JOUR (entrée « nuit » ci-dessus) : l'alarme portée ici est
> LEVÉE sur le nRMSE.** L'écart entre les deux géométries vaut +0.0005 en région
> `full`, celle des chiffres publiés (26/60 paires, p = 0.37) — l'avance de R-best sur
> la production tient. Elle reste fondée sur le SSIM et le LPIPS (0.006), et sur la
> région notée où les 8 fenêtres sont MEILLEURES de 0.0034 (p = 3.2e−06). Lire les deux
> entrées ensemble.

**Verdict : à partir du 2026-08-27, tous les chiffres passés par
`scripts/run_task3_eval.sh` ont été produits en 8 fenêtres décalées fusionnées
par Hann, pendant que les chiffres antérieurs l'étaient en crop centré unique.
Les deux familles sont comparées dans le même tableau.**

`--center_crop_only` est déclaré `action="store_true"` sans `default=None`
(`infer_mmfm_unified.py:767`) et n'est jamais lu par `_flag()` : **aucune clé de
config ne peut l'activer**, il faut le taper. `scripts/run_task3_eval.sh:37-43` ne
le tape pas. La branche par défaut (`infer_mmfm_unified.py:355-372`) découpe 8
fenêtres décalées de −16 à +7 voxels et les fusionne par produit de Hann, alors
que **tout le cache de latents a été encodé sur un crop centré unique**
(`precompute_*_latents.py:154`).

### La signature est binaire, et je l'ai mesurée moi-même

Écart médian entre fichiers de prédiction consécutifs, robuste aux reprises :

| run | s/volume | géométrie |
|---|---|---|
| `compare_20260905/production` (`--center_crop_only` documenté) | **16.0** | crop centré |
| `compare_20260905/trajectory_l1` (idem) | **16.0** | crop centré |
| `mmfm/vectorized` (production, 0.3794) | **17.0** | crop centré |
| `mmfm/inr_std` (0.3749) | **9.0** | crop centré |
| **`mmfm/vec_rbest`** (R-best, **0.3737**) | **104.0** | **8 fenêtres** |
| **`mmfm/ceiling_vectorized`** (plafond **0.1048**) | **104.0** | **8 fenêtres** |
| `mmfm/vec_lpips`, `vec_batch8`, `ceiling_lpips` | 104.0 | 8 fenêtres |

Rapport 6.1–6.5× entre les deux familles, exactement 8 passes plus le coût fixe
d'E/S. La coupure est chronologique et sans exception : elle tombe le 2026-08-27,
date de création du pilote.

### Ce que cela touche

- **R-best 0.3737**, désigné meilleur flow dans `AGENTS.md`, est comparé à
  « production 0.3794 » qui est en crop centré. **L'écart de 0.0057 n'est pas
  attribuable** : il mélange le réglage et la géométrie.
- **Le plafond de représentation 0.1048** (et son successeur LPIPS 0.0976), cités
  partout depuis le 2026-08-27, sont en 8 fenêtres.
- Cela **explique l'écart resté ouvert hier** : le banc
  `results/mmfm/representation_20260906/` donnait 0.1184 contre 0.1048 publié,
  « +0.0135, même signe sur 13 cellules sur 15 », avec l'hypothèse écrite au
  conditionnel « le chemin publié passe par `infer_mmfm_unified` (recomposition
  par patches pondérés possible) ». **L'hypothèse est confirmée.**

### Ce que cela ne touche PAS

L'entrée du 2026-09-05 (« la recette d'inférence ne comptait pas : 0.3581 contre
0.3584, soit 0.0003 ») comparait **deux runs en crop centré** — les fichiers
stockés de `mmfm/vectorized` sont à 17.0 s/volume. Elle ne mesure donc **pas**
l'effet des 8 fenêtres, qui reste inconnu.

### Correctif

Une ligne dans `scripts/run_task3_eval.sh`. Mais la question ouverte n'est pas le
correctif : c'est **de combien** les 8 fenêtres déplacent le score, et donc si le
classement des variantes tient. Une seule ré-inférence de `vec_rbest` avec le
drapeau (49 min mesurées) le dit.

---

## 2026-09-07 — **Validation de la couche de représentation : le VAE est bon, l'INR est affamé, et le « plafond » mesurait surtout le protocole**

**Verdict : les deux briques de représentation fonctionnent correctement pour ce
qu'on leur donne. Le déficit de l'INR est un déficit de BUDGET (facteur 252), pas
de famille de modèles. Et 60 % du « plafond de représentation » n'est pas
imputable au VAE.** Manifestes :
`results/mmfm/representation_20260906/manifest.md` et
`results/mmfm/inr_capacity_20260906/manifest.md`. Scripts :
`src/cfm/bench_representation.py`, `src/cfm/bench_inr_capacity.py`.

### 1. Le plafond 0.1048 n'était pas une propriété du VAE

La chaîne complète a été relancée **avec l'identité à la place du VAE**. Sujet 0006,
3 contrastes × 5 champs, tranche notée [150, 180).

| | nRMSE tranche notée |
|---|---|
| `noop` (contrôle, doit valoir 0) | **0.0000** |
| **chaîne complète SANS VAE** | **0.0685** |
| chaîne complète AVEC MedVAE (`prod`) | 0.1140 |

Décomposition du plancher protocolaire : **écrêtage au percentile 99.5 = 0.0359**,
rééchantillonnage 0.5 → 1 → 0.5 mm = 0.0326, **crop 192×224×192 = 0.0000** (mesuré,
`protocol` et `protocol_nocrop` identiques à la 4ᵉ décimale).

Et la part protocolaire varie de **11 %** (T2FLAIR@3T) à **71 %** (T2FLAIR@0.1T)
selon la cellule : **comparer deux cellules du plafond compare surtout deux
quantités d'écrêtage.** Le chiffre « 0.10–0.13 pour un prédicteur parfait », cité
depuis le 2026-08-26, est confirmé en ordre de grandeur et mal attribué.

### 2. Les écarts à la recette officielle MedVAE sont neutres ou favorables

| écart au standard amont | VAE seul nRMSE | chaîne notée | verdict |
|---|---|---|---|
| **production** | 0.0597 | 0.1140 | référence |
| échantillon → **mode** de la postérieure | 0.0597 | 0.1139 | **neutre** |
| bfloat16 → **float32** | 0.0597 | 0.1140 | **neutre** |
| marge de contexte 16 → **32** | 0.0594 | 0.1138 | **neutre** (−0.0003, +0.043 dB) |
| tuilage maison → **fenêtre glissante officielle** | 0.0599 | 0.1168 | **le maison gagne**, et 1.7× plus vite |
| **recette officielle complète** (min-max + CropForeground) | 0.0737 | 0.1509 | **+32 %, nettement pire** |
| percentile + CropForeground | 0.1167 | 0.1215 | `CropForeground` coûte **4.7 dB** |
| `medvae_8_1_3d` au lieu du 4× | 0.1384 | 0.2195 | **bien pire** — le 4× est le bon choix |

**Aucun écart au standard n'est un défaut de performance.** `medvae_4_1_3d` est en
outre le seul autre poids 3D disponible en dehors du 8× (`factory.py`
::`FILE_DICT_ASSOCIATIONS`), que les auteurs mesurent eux-mêmes à −3.29 dB
(26.23 contre 29.52 PSNR, Table 4 du papier). **La question « a-t-on choisi la bonne
brique 3D ? » se ferme : oui, et il n'y a pas de second choix.**

### 3. Un vrai défaut, sans effet sur la fidélité

`MVAE.encode()` en 3D renvoie `posterior.sample()`, **pas le mode** — trois
commentaires du dépôt affirmaient le contraire (`maisi_vae.py:139`,
`tiled_vae.py:50`, `finetune_medvae.py:287`). Mesuré : deux appels diffèrent,
bruit/signal du latent **2.3e-3**. Effet sur la reconstruction : **nul à la 4ᵉ
décimale**. C'est un défaut de **reproductibilité du cache**, pas un levier de score.
Commentaires corrigés, `encode_mode()` ajouté.

### 4. Le seul levier réel côté VAE : l'écrêtage

L'erreur ABSOLUE du VAE est constante (PSNR 31.3–31.5 dB quelle que soit la
normalisation) ; seule l'occupation de la dynamique change l'erreur relative.
Balayage du `hi` fixe par (contraste, champ) — **forme transposable à la traduction**,
contrairement aux percentiles par volume :

| `hi` × | 0.5 | 0.75 | **1.0 (prod)** | **1.25** | 1.5 | 2.0 |
|---|---|---|---|---|---|---|
| nRMSE tranche notée | 0.3900 | 0.2028 | **0.1140** | **0.0989** | 0.1001 | 0.1053 |

**Optimum plat entre ×1.25 et ×1.5 : −13.2 %.** Le mécanisme est mesuré : la table
`field_norm_stats.json` sature 25–50 % des voxels de cerveau selon la cellule
(T2FLAIR@3T 50.0 %, T1W@3T 37.1 %, T1W@7T 25.1 %), et l'écrêtage n'est pas inversible.

**Et les deux leviers se composent** (`part propre au VAE` = `sqrt(total² − plancher²)`) :

| variante | nRMSE notée | SSIM | vs production | part propre au VAE |
|---|---|---|---|---|
| *protocole seul (plancher)* | *0.0685* | *0.9717* | — | *0.0000* |
| **production** | 0.1140 | 0.8984 | — | 0.0911 |
| MedVAE LPIPS | 0.1063 | 0.9129 | −6.7 % | 0.0813 |
| production + `hi` ×1.25 | 0.0989 | 0.9077 | −13.2 % | 0.0714 |
| **LPIPS + `hi` ×1.25** | **0.0901** | **0.9227** | **−21.0 %** | **0.0585** |
| LPIPS + `hi` ×1.5 | 0.0904 | 0.9221 | −20.7 % | 0.0589 |

**−21.0 % sur la chaîne, −36 % sur la part propre au VAE, sans rien réentraîner** :
un checkpoint déjà dans le dépôt et un scalaire par cellule dans un JSON. Le SSIM monte
aussi (0.8984 → 0.9227), donc ce n'est pas un arbitrage perception/distorsion. À noter
que le gain LPIPS seul (−6.7 %) vaut un cinquième de ce qui avait été annoncé le
2026-08-25 sous la géométrie du fine-tuning (patchs 64³).

### 5. L'INR : ce n'est pas la famille, c'est le budget

Même volume, même entrée normalisée, même métrique que le banc VAE. T1W × {0.1T, 3T,
7T}. Le témoin qui manquait à toute interprétation : **ce que vaut un budget de N
nombres dépensé de la façon la plus bête possible** (grille basse résolution
ré-interpolée).

| représentation | budget/volume | nRMSE | nRMSE premier plan | PSNR |
|---|---|---|---|---|
| *témoin trivial 11×13×11* | *1 536* | *0.2381* | *0.6641* | *18.91* |
| **INR production** (20 pas SGD) | **512 effectifs** | 0.1448 | 0.4367 | 23.41 |
| INR, 200 pas de SGD | 512 | 0.1417 | 0.4276 | 23.61 |
| INR, modulation directe (Adam) | 1 536 | 0.1396 | 0.4062 | 23.72 |
| + `modulate_scale` | 3 072 | 0.1268 | 0.3748 | 24.61 |
| + LoRA rang 4 | 12 812 | 0.1019 | 0.3086 | 26.61 |
| + LoRA rang 16 | 46 640 | 0.0737 | 0.2203 | 29.45 |
| **+ LoRA rang 64** | **181 952** | **0.0531** | **0.1548** | **32.32** |
| *témoin trivial 48×56×48* | *129 024* | *0.0954* | *0.2835* | *27.21* |
| **MedVAE pré-entraîné** | 129 024 | 0.0587 | 0.1832 | 31.82 |
| **MedVAE affiné LPIPS** | 129 024 | 0.0503 | 0.1576 | 33.14 |
| **SIREN LIBRE 256×6** (tous poids, un volume) | ~330 k | 0.0306 | **0.0866** | **37.17** |
| **SIREN LIBRE 512×8** (tous poids, un volume) | ~1.8 M | 0.0157 | **0.0428** | **42.86** |

**Le plafond de la famille INR est mesuré pour la première fois, et il est très haut** :
un SIREN neuf tous poids libres atteint **42.86 dB** sur ces volumes, deux fois moins
d'erreur en premier plan que le meilleur MedVAE. Et à la taille EXACTE de la production
(256×6), il atteint 37.17 dB là où la production plafonne à 23.41. **Ce n'est donc ni la
famille, ni l'architecture SIREN, ni `omega_0`, ni la profondeur : tout l'écart est dans
ce qu'on autorise à varier par volume.** *(Le SIREN libre n'est pas une proposition
utilisable — 330 k poids par volume à stocker et à faire transporter par le flow — c'est
une borne supérieure, et elle dit que le plafond n'était pas où le projet le croyait.)*

**Les deux représentations valent exactement la même chose par unité de budget** :
l'INR bat son témoin trivial de −34 % en premier plan (0.4367 contre 0.6641),
MedVAE bat le sien de −35 % (0.1832 contre 0.2835). L'écart entre eux n'est pas un
écart de qualité : c'est **512 nombres contre 129 024, un facteur 252**. Et l'échelle
est monotone sur cinq ordres de grandeur de budget, du témoin trivial au SIREN libre.

**À budget comparable, l'INR dépasse MedVAE pré-entraîné** (0.1548 contre 0.1832 en
premier plan) et rejoint le MedVAE perceptuel (0.1576) — **sans réentraîner le SIREN**,
en optimisant seulement la modulation.

**Ce n'est pas un problème d'optimisation** : 20 pas → 200 pas de SGD ne gagne que
+0.2 dB, et l'optimisation directe des 1 536 décalages par Adam donne le même
résultat que la production. **Le hypernetwork de 66 M paramètres n'apporte donc
rien** (1 Go de checkpoint, 0.98 s/itération) — et sa jacobienne est de rang 512,
inférieur aux 1 536 décalages : **la capacité réelle par volume est 512**, pas
129 024 comme l'annonce `inr.yaml:48`.

Conséquence sur une phrase répétée depuis des mois — « le handicap de l'INR est
représentationnel » : **vrai au sens du budget, faux au sens de la famille**. Aucun
correctif de mécanisme ne pouvait combler un facteur 252.

### 6. Deux défauts d'entraînement de l'INR, vérifiés et corrigés

**A. Les 50 000 pas de méta-entraînement ont vu les MÊMES 16 384 coordonnées.**
`meta_train_step` était appelé sans générateur (`train_inr_backbone.py:219`) et
`sample_points` en fabriquait un dont la graine ne dépend que de `(n, num_points)`
(`inr_backbone.py:104`) — donc constante. Vérifié : deux appels rendent des indices
identiques. `θ` n'a jamais été supervisé ailleurs que sur **0.198 %** de la grille
(espacement moyen 8.0 voxels). **CORRIGÉ** : un générateur qui avance à chaque pas
(recouvrement entre deux tirages : 30/16384, conforme au hasard), tandis que
l'ajustement de `z` reste figé, comme il doit l'être.

**B. Après 50 000 pas, la base SIREN est restée son initialisation aléatoire.**
RMS des poids EMA contre la valeur exacte de l'init : `first` **1.01×**,
`hidden.0` **1.01×**, `hidden.4` 1.16×, seul `final` à 3.00×. A explique B.

Ces deux défauts sont réels et gratuits à corriger, mais **ils n'expliquent pas
l'écart de 8.4 dB** : le banc les contourne et retrouve 32.32 dB dès que le budget
monte. C'est bien la capacité qui borne.

### 7. Le banc VAE historique (`results/benchmark_vae/`) est invalide

Deux causes vérifiées dans le code, chacune suffisante :
1. **il alimente MedVAE en [0, 1]** (`benchmark_vae.py:77`) là où la recette amont et
   la production utilisent [−1, 1] ;
2. **`PatchedVAE` laisse 14.89 % des voxels sans aucun patch** (`patched_vae.py:106` :
   `range(0, h - ph + 1, sh)` ne couvre pas la queue) — simulé exactement sur la
   géométrie du banc : 81 patches, 8 601 152 voxels jamais écrits, qui sortent à zéro
   et entrent quand même dans les métriques.

Aucun des deux ne touche la production, mais **ces CSV ne peuvent pas départager deux
architectures**. Le présent banc les remplace pour cette question.

### Ce qui est actionnable, par ordre de rapport qualité/prix

1. **`hi` × 1.25 dans `field_norm_stats.json` + checkpoint LPIPS** — −21.0 % de
   l'erreur de représentation, SSIM +0.024, **coût : régénérer le cache de latents**
   (~5 h 30) et réentraîner le flow (~2 h). Aucun code à écrire.
2. **INR : `hyper_hidden_dim: 0` + `lora_rank: 16` (`latent_dim: 46640`) ou `64` (`latent_dim: 181952`)**
   — options déjà présentes, jamais activées. Supprime d'un coup le goulot de rang 512
   ET les 66 M paramètres du hypernetwork (1 Go de checkpoint, 0.98 s/itération).
   Attention à l'initialisation LoRA : `fit_latent` part de `z = 0`, donc `A` et `B`
   seraient tous deux nuls et **le gradient serait nul des deux côtés** — il faut
   initialiser `B` à l'aléatoire (convention Hu et al. 2021, voir
   `bench_inr_capacity.py::init_gamma`), et donner à la LoRA un pas d'apprentissage
   distinct de celui des décalages (au pas des décalages, la LoRA de rang 16 fait
   **diverger** le réseau : nRMSE 0.83).
3. **Réentraîner le backbone INR** avec le tirage de coordonnées corrigé — les 50 000 pas
   précédents ont vu 0.198 % de la grille.

**Ce qui N'EST PAS actionnable, et pourquoi** : passer au 8×, s'aligner sur la recette
officielle MedVAE, prendre le mode plutôt que l'échantillon, passer en float32, changer
de VAE 3D. Tous mesurés, tous neutres ou négatifs.

### Réserves, à ne pas perdre

- **Un seul sujet** (0006). Les écarts entre variantes portent sur 15 volumes appariés
  et sont fiables ; le NIVEAU absolu ne l'est qu'à ±13 % (le banc reproduit le plafond
  publié à +0.0135 près, même signe sur 13 cellules sur 15).
- **Le PSNR de ce banc n'est PAS comparable au 29.52 dB publié** : 82 % de nos volumes
  est du fond plat trivialement reconstruit, là où la recette amont recadre sur le
  cerveau. Restreinte au premier plan, notre erreur relative vaut 0.2642 et non 0.0597.
  Ne pas écrire « nous faisons mieux que le papier ».
- Le bras `prod_margin32` **ne mesurait pas la marge** : à marge 32 la tuile élargie
  fait 176 sur l'axe W et MedVAE la re-découpe en fenêtres de 88 (`roi_size_calc`,
  `gpu_dim=160`). Vérifié, puis **repris proprement** avec `gpu_dim=256` (une seule
  fenêtre des deux côtés) : marge 32 − marge 16 = **−0.0003 nRMSE, +0.043 dB**. La
  conclusion tient, elle est maintenant établie. Le piège, lui, reste armé pour toute
  future expérience qui dépasserait 160 sur un axe.
- Les chiffres `mod_*` sont obtenus sur le SIREN **gelé** de production, méta-entraîné
  pour une modulation par décalage seul : ce sont des **planchers**, pas des plafonds.
- **`scripts/run_task3_eval.sh` ne passe pas `--center_crop_only`** alors que tous les
  chiffres publiés l'ont utilisé : toute mesure passée par ce pilote sortirait sous une
  autre géométrie. Vérifié, non corrigé.
- Rien ici ne dit ce que le score de bout en bout deviendrait. Le 2026-09-02 a établi
  qu'un gain de représentation de −0.0072 s'était transmis en **+0.0004** sur le score.

---

## 2026-09-06 (soir) — **Cache SANS normalisation : NÉGATIF. Le niveau atteint le réseau, mais MedVAE perd sa dynamique.**

**Verdict : le défaut 04 est réel, et sa correction naïve est pire que le mal.**
Config : `configs/mmfm/vectorized_identity.yaml`, cache `medvae_finetune_ff550d64`
(1939 volumes ré-encodés, 5 h 30). CSV : `results/mmfm/compare_20260906/`.

Recette d'inférence partagée, tranche notée, 180 volumes chacun :

| run | T1W | T2W | T2FLAIR | nRMSE | SSIM |
|---|---|---|---|---|---|
| **identity (sans normalisation)** | **0.5278** | 0.3166 | 0.4297 | **0.4247** | 0.8176 |
| l1 + 4 correctifs | 0.4249 | 0.3018 | 0.3189 | **0.3485** | 0.8220 |
| mse + 4 correctifs | 0.4332 | 0.2959 | 0.3357 | 0.3549 | 0.8099 |
| production | 0.4034 | 0.3215 | 0.3493 | 0.3581 | 0.8379 |
| *témoin srclevel* | *0.2901* | *0.2612* | *0.2171* | ***0.2561*** | — |

**+0.076 de nRMSE, la pire des quatre variantes.** T1W s'effondre (0.4249 →
0.5278). Le correctif visait 0.3485 → 0.2561 ; il fait l'inverse.

### La cause, mesurée

Poser `lo=0, hi=1` rend bien la chaîne exactement inversible (aller-retour à
1.5e-8, contre un plancher de 0.0549 en `field_fixed` pour un prédicteur
PARFAIT) et fait bien arriver le niveau absolu au réseau. Mais le signal
n'occupe alors plus qu'une fraction de la dynamique :

| | écart-type du foreground | p99.5 après normalisation |
|---|---|---|
| T1W@7T, identité | **0.226** | **−0.20** |
| T1W@7T, field_fixed | **0.615** | +1.00 (par construction) |
| T1W@0.1T, identité | 0.384 | +0.64 |

99.5 % des voxels de T1W@7T vivent sous −0.20, dans le bas de [−1,1]. MedVAE est
entraîné sur des entrées qui remplissent leur plage : il reconstruit mal ce
régime. Et la correspondance est nette — **T1W@7T est le cas le plus comprimé
(std 0.226) et T1W le contraste le plus dégradé**. Les statistiques du latent le
confirment : `latent_scale` passe de 18.125 à **13.551** (−25 %), le latent est
lui aussi moins varié.

### Ce que ça apprend

Deux objectifs entrent en conflit sous un unique couple (lo, hi) : **préserver le
niveau relatif entre volumes** exige un diviseur COMMUN, **remplir la dynamique
du VAE** exige un diviseur proche du maximum de CHAQUE volume. `(0, 1)` satisfait
le premier et sacrifie le second ; les percentiles par volume faisaient
l'inverse ; `field_fixed` est un compromis par classe.

**Piste non testée qui résout le conflit** : un diviseur global UNIQUE, identique
pour tous les volumes et tous les champs, calé sur le p99.5 de POPULATION
(~0.5 au lieu de 1.0). Il préserve exactement les écarts de niveau entre volumes
(un seul diviseur) tout en doublant l'occupation de la dynamique. Une ligne dans
`field_norm_stats_identity.json`.

**Réserve honnête** : rien ne garantit que le flow saurait exploiter le niveau
même correctement présenté. Ce run montre que la présentation naïve casse le
VAE ; il ne montre pas que la présentation corrigée marcherait.

---

## 2026-09-06 — **`l1` contre `mse`, variable unique : la perte théoriquement fausse gagne sur les DEUX métriques.**

**Verdict : l'attribution est confirmée, et elle va plus loin que prévu.** CSV :
`outputs/mmfm/compare_20260905/`. Config : `configs/mmfm/vectorized_trajectory_l1.yaml`
(diff avec la version mse : `loss`, `task_name`, `output_subdir` — rien d'autre).

Recette d'inférence partagée, 180 volumes chacun, tranche notée :

| run | T1W | T2W | T2FLAIR | nRMSE | SSIM |
|---|---|---|---|---|---|
| **l1 + 4 correctifs** | 0.4249 | 0.3018 | 0.3189 | **0.3485** | **0.8220** |
| mse + 4 correctifs | 0.4332 | 0.2959 | 0.3357 | 0.3549 | 0.8099 |
| production (l1, sans correctifs) | 0.4034 | 0.3215 | 0.3493 | 0.3581 | 0.8379 |
| *témoin srclevel* | *0.2901* | *0.2612* | *0.2171* | ***0.2561*** | — |

**Attribution (l1 − mse, seule la perte diffère) : nRMSE −0.0064, SSIM +0.0121.**
`l1` gagne sur les DEUX métriques. L'hypothèse « mse coûte du SSIM » est
vérifiée (43 % de l'écart récupéré), mais l'effet ne s'y limite pas : la perte
quadratique dégrade aussi le nRMSE.

**Le mécanisme est réparé dans les deux runs**, donc l'écart est bien imputable à
la perte : `l1` donne cos(v0,v1)=0.035, sujet 7.34 %, colinéarité 0.26/0.69/0.83 ;
`mse` donne −0.592, 8.25 %, 0.32/0.74/0.74 ; production 1.000000, 0.765 %, 1.000000.

**Ce que ça dit du théorème.** Le flow matching identifie le champ de vitesse
marginal à une ESPÉRANCE conditionnelle — le coût quadratique est le bon
estimateur de cet objet-là. Mais on n'évalue pas le champ de vitesse : on évalue
le POINT D'ARRIVÉE après intégration. Sur des cibles au bruit lourd (couplage OT
synthétique entre sujets non appariés), l'estimateur robuste transporte mieux.
Le correctif « théoriquement correct » était empiriquement le mauvais.

**Netteté (`hf_ratio`, médiane, 36 cellules) — l'attribution est fermée par la
grandeur qu'elle concerne** : production **0.2754**, l1 **0.2640**, mse
**0.2436**. L'ordre des SSIM suit exactement l'ordre des netteté : `mse` est
8.4 % plus flou que `l1`, et c'est ce flou que le SSIM sanctionne.
Deux observations qui dépassent la question posée : les trois variantes ne
restituent que **~25 %** du rapport HF/BF de la vérité (plancher commun, très
au-dessus des écarts entre elles) ; et les 4 correctifs DÉGRADENT la netteté par
rapport à la production (0.264 / 0.244 contre 0.275) — la spline moyenne sur
cinq marginales là où les droites par paires n'en moyennaient que deux.

**Meilleure configuration à ce jour : 0.3485** contre 0.3581 pour la production,
soit −0.0095 (2.7 % relatif, au-dessus du plancher de bruit de 0.002). Premier
gain réel de la série — mais le SSIM reste sous la production (−0.0160), et le
témoin `srclevel` garde **27 %** d'avance.

---

## 2026-09-05 — **Les quatre correctifs : mécanisme RÉPARÉ, score NÉGATIF.**

**Verdict : le mécanisme du flow n'est pas le levier.** Les trois dégénérescences
sont levées et vérifiées, et le score ne suit pas. CSV :
`outputs/mmfm/compare_20260905/compare_shared_recipe.csv`. Config :
`configs/mmfm/vectorized_trajectory.yaml`. 25 000 itérations, 2 h 07, 3.49 it/s.

### Le score, à recette d'inférence STRICTEMENT partagée

Les deux jeux de 180 volumes ont été produits par le même chemin et les mêmes
drapeaux (`--center_crop_only`, mêmes `field_norm_stats`), chaque modèle étant
construit avec sa propre config puisque c'est requis pour le charger.

| tranche [150,180) | T1W | T2W | T2FLAIR | nRMSE | SSIM |
|---|---|---|---|---|---|
| trajectory (4 correctifs) | 0.4332 | 0.2959 | 0.3357 | **0.3549** | **0.8098** |
| production (même recette) | 0.4034 | 0.3215 | 0.3493 | **0.3581** | **0.8379** |
| écart | +0.030 | −0.026 | −0.014 | **−0.0031** | **−0.0281** |
| *témoin srclevel* | | | | *0.2561* | — |

**Gain de 0.0031 en nRMSE — sous le plancher de bruit de run (0.002) — payé par
0.0281 de SSIM, neuf fois plus grand.** T2W et T2FLAIR progressent, T1W recule
nettement. Et le modèle reste **28 % derrière un scalaire par volume**.

**La recette d'inférence ne comptait pas** : production à recette partagée
= 0.3581 contre 0.3584 pour les fichiers stockés, soit **0.0003**. La réserve
posée la veille (2.7e-2 d'écart en espace prédiction, contre un plancher bf16 de
7.6e-4) était justifiée dans son principe et négligeable dans son effet.

### Le mécanisme, lui, est bien réparé

Sondes sur les poids EMA, latents réels du cache c4d1e200 :

| sonde | production | trajectory |
|---|---|---|
| `cos(v(t=0), v(t=1))` | 1.00000000 | **−0.592** (137 % de variation) |
| variation de v entre deux sujets | 0.765 % | **8.247 %** |
| colinéarité des 4 déplacements | 1.000000 | **0.32 / 0.74 / 0.74** |
| normes des déplacements | 1.91/3.82/5.74/7.65 (rampe 1:2:3:4) | 17.1/14.2/15.6/18.2 |

**Piège de lecture, à noter** : la sensibilité au sujet fait 5.26 % (2500 pas) →
1.010 % (5000) → 1.397 % (7500) → **8.247 % (25000)**. J'ai conclu à un plateau
sur les trois premiers points et annoncé que la composante individuelle ne
serait pas au rendez-vous. **C'était un creux, pas un plateau.** Trois points
dans un transitoire ne font pas une tendance — même erreur que « une moyenne ne
dit pas ce qu'on croit », dans sa version temporelle.

### Hypothèse d'attribution, NON testée

**La perte MSE explique probablement la chute de SSIM.** Le 2026-08-26 avait
mesuré que la médiane conserve 106-144 % des hautes fréquences par rapport à la
moyenne. Passer de `l1` à `mse` remplace un estimateur de médiane par un
estimateur de moyenne : théoriquement correct pour le flow matching, plus flou en
pratique. La L1 était un mauvais estimateur qui rendait service au SSIM.
Confondu avec les trois autres changements — se teste en relançant `loss: l1`,
tout le reste inchangé.

### Ce que ça établit

Deuxième occurrence du même schéma après le MedVAE perceptuel (2026-09-02) :
**une propriété interne mesurablement améliorée ne se transmet pas à la
métrique.** Le compteur passe à 2. Le levier restant est celui que le témoin
désigne depuis le début — l'échelle d'intensité câblée à une constante par
classe (défaut 04 de l'audit), que le flow ne touche pas quel que soit son degré
de correction.

---

## 2026-09-04 (soir) — **Audit de code : la métrique était la mauvaise, et le flow calculait une constante. Quatre correctifs posés.**

**Verdict : la clôture de série du 2026-09-03 est renversée. Le plateau n'était
pas la borne de l'information disponible.** Rapport complet :
https://claude.ai/code/artifact/cacb9435-8746-42b6-bb6f-139c25657b5a
CSV : `results/mmfm/region_20260904/`.

Harnais validé d'abord : il retrouve le témoin identité (0.9273 / 0.3859 /
0.5574) et le vectorisé (0.3795 / SSIM 0.8975) **à la quatrième décimale**, et la
conformité des formules à l'évaluateur officiel est exacte (écart 2.2e-16).

### 1. La métrique locale n'était pas celle du classement

`Submission/build_submission/README.md:114` : « Both pred and seg are sliced
along axial **[150, 180)** … the GT is fixed at [150, 180), so other ranges will
fail evaluation. » Le classement note un pavé **(364, 436, 30)**.
`build_task3_submission.py` appliquait déjà ce clip — l'arbre de soumission était
juste — mais **aucun script d'évaluation ne le faisait**, et le mot n'apparaît
nulle part dans ce journal. Tout ce qui précède est mesuré sur 364 coupes, une
région 12x plus grande (78.8 % de fond contre 54.8 %).

**État de référence recalculé, mêmes fichiers, mêmes formules :**

| | T1W | T2W | T2FLAIR | moy. nRMSE | moy. SSIM |
|---|---|---|---|---|---|
| Vectorisé — volume entier | 0.4354 | 0.3376 | 0.3654 | 0.3795 | 0.8975 |
| **Vectorisé — tranche notée** | 0.4043 | 0.3215 | 0.3493 | **0.3584** | **0.8377** |
| **UNet — tranche notée** | 0.4217 | 0.3271 | 0.3646 | **0.3711** | 0.8360 |
| **INR — tranche notée** | 0.3708 | 0.3555 | 0.3175 | **0.3479** | 0.7835 |

L'écart vaut 0.021 en nRMSE et **0.059 en SSIM** — dix fois la taille des effets
arbitrés depuis un mois (AdaGN 0.0006, flip 0.0001, ordre 3 0.0031). **Le
classement des trois architectures ne s'inverse PAS** (prédiction que j'avais
posée comme possible : elle est fausse, et l'ordre non total INR/vectorisé est
préservé — INR première en nRMSE, nettement dernière en SSIM).

### 2. Le flow de production EST une constante — mesuré, pas déduit

Poids EMA de `outputs/mmfm/vectorized/weights/model_final.pth`, latents réels du
cache c4d1e200 :

| sonde | valeur |
|---|---|
| `cos(v(t=0), v(t=1))` | **1.00000000** (‖v‖ 841.3135 → 841.3134) |
| `cos(v(sujet A), v(sujet B))` | **0.99997336** (latents distants de 34 %) |
| variation de v quand z_t → z_cible | **1.37 %** |
| déplacements vers 1.5/3/5/7T | 1.91 / 3.82 / 5.74 / 7.65 % — rapport **1:2:3:4** |
| colinéarité des 4 déplacements | **1.000000** |

C'est `z ↦ z + Δt·c` pour un unique vecteur constant : **un degré de liberté**
pour les 20 cellules. Et il est sous-dimensionné : `cos(c, E[z_7T]−E[z_0.1T])`
= 0.74, ‖c‖ = 37 % du déplacement inter-classes, si bien que
‖pred − E[z_7T]‖ = 41.8 % contre ‖z_src − E[z_7T]‖ = 44.2 % — **le modèle
parcourt 5 % du chemin**. Les huit leviers négatifs étaient des ablations
d'architecture autour d'une fonction constante.

**Trois dégénérescences distinctes, jamais combinées.** Vectorisé : aveugle au
temps ET au sujet. INR (`latent_scale: 2.16e-4`) : voit le sujet (cos A/B 0.917),
pas le temps (0.99999881). R-best (`time_scale: 1000` + `film`) : voit le temps
(cos(v0,v1) = −0.09), pas le sujet (0.99993843, `latent_scale: 1.0` sur un latent
d'écart-type 18.1). Aucune des 16 configs ne pose les deux à la fois.

### 3. La cause, chiffrée

`mmfm_core.py:728-729` tirait z_src et z_tgt de **deux DataLoader indépendants**,
et à `batch_size: 1` l'OT exact est dégénéré : aucun couplage. **R² de la
prédiction du latent cible depuis le latent source, T1W 0.1T→7T, prédicteur
affine à un paramètre :**

| régime | R² au-delà d'une constante |
|---|---|
| pairwise (production) | **+0.0005** |
| trajectoire (OT chaîné) | **+0.0335** |

**+0.0005 : il n'y avait rien à apprendre.** Le meilleur prédicteur était
E[z_cible], une constante — exactement ce que le modèle a appris. Ce n'était pas
un défaut d'optimisation, c'était l'optimum d'un objectif vide.

### 4. Le témoin qui manquait — et il bat les trois architectures

Sur la tranche notée, 180 cellules :

| témoin | nRMSE | déployable ? |
|---|---|---|
| recopier la source | 0.5996 | — |
| source × scalaire, LOO sur les 2 autres sujets | **0.3179** | oui |
| **source × scalaire, depuis le NIVEAU OBSERVÉ dans la source** | **0.2561** | oui |
| source × scalaire oracle | 0.1740 | non (borne) |
| meilleure architecture (INR) | 0.3479 | — |

`corr(niveau du foreground de la source, gain oracle) = −0.76 / −0.70 / −0.88`.
Le gain requis **est lisible dans la source**, et `normalize_volume` le divise
avant que le réseau ne le voie. « La composante individuelle n'est pas
prédictible depuis la source » (R² = 0.135, 2026-09-03) avait été testé dans
l'espace normalisé, d'où l'information avait déjà été retirée.

**Réserve, n = 3.** Le LOO nu est très bruité : le sujet 0009 a un 7T environ
deux fois plus sombre que les autres (gains `7T_to_*` = [2.19, 2.66, **5.31**]),
et le LOO lui attribue 2.43. Sur T1W il vaut 0.4088, soit à égalité avec le
vectorisé (0.4043) et derrière l'INR (0.3708) — **la moyenne de 0.3179 est portée
par T2W et T2FLAIR**. C'est `srclevel`, qui lit le niveau de la source, qui gagne
partout (T1W 0.2901). Ces témoins sont des BORNES, pas des méthodes.

### Correctifs posés (code)

| # | correctif | fichier |
|---|---|---|
| 1 | `Z_CLIP_RANGE = (150, 180)` + `apply_z_clip`, `--region {slab,full}` (défaut `slab`), région écrite dans chaque CSV | `common/io.py`, `eval_quantitative_3arch.py`, `evaluation/evaluate.py` |
| 2 | `--mode scaled_identity` : témoins LOO / srclevel / oracle | `eval_quantitative_3arch.py` |
| 3a | `train.loss` — défaut **`mse`** (le FM exige une espérance, la L1 donne la médiane) | `mmfm_core.py` |
| 3b | `train.flip_per_step` — **un seul tirage par pas** au lieu d'un par appel à `__getitem__` : une transition sur deux demandait un cerveau miroir | `mmfm_core.py`, testé sur les 4 formes de latent |
| 3c | `build_chained_ot_trajectories` — couplage OT chaîné hors boucle (`data.py:123`) | `mmfm_core.py` |
| 3d | `_compute_flow_trajectory` — spline cubique à travers les K marginales (`multi_marginal_fm.py:334`) | `mmfm_core.py` |
| 4 | `configs/mmfm/vectorized_trajectory.yaml` — les 4 correctifs + `time_scale: 1000` + `time_cond: film` + `latent_mean: 19.913378` / `latent_scale: 18.125440` (mesurés sur 240 volumes du cache) | config |

La spline est implémentée comme un opérateur linéaire fixe sur les 5 ancres
(temps de champ fixes), obtenu en ajustant `scipy.CubicSpline` sur la base
canonique : sémantique scipy identique à la référence, vérifiée à **6e-16**
(nu=0) et **7e-15** (nu=1), sans aller-retour CPU par pas.

**Smoke de bout en bout** : couplage des 3 contrastes en **2.4 s**
(100/100/99 trajectoires ; sujets distincts par champ [100,100,100,43,81] pour
T2W, l'OT réutilise les 43 volumes de 5T — cas rectangulaire assumé), puis
entraînement à 3.8 it/s, 11.6 GB, gradients ~150. Le chemin `pairwise` reste
fonctionnel (non-régression vérifiée).

**Non fait, et il faut le dire** : aucun ré-entraînement complet. Les correctifs
sont posés et vérifiés MÉCANIQUEMENT (le couplage produit du signal apprenable,
la spline est exacte, le flip est cohérent, les deux chemins tournent) ; leur
effet sur le score reste à mesurer par un run de 25 000 itérations.

---

## 2026-09-04 — **Le backbone INR de production est sain, et on peut enfin le prouver**

**Verdict : les cinq portes passent, et le contrôle négatif établit pour la première
fois qu'elles savent séparer un mécanisme sain d'un cassé.** Logs :
`outputs/gate_inr_production_20260904.log`, `outputs/smoke_inr_1mm_20260904_gates.log`,
`outputs/smoke_inr_1mm_20260904_loo.log`. Table :
`results/mmfm/gate_inr_production_20260904/gate_production_20steps.csv`.

### Le backbone de PRODUCTION, mesuré pour la première fois

`outputs/mmfm/inr_backbone/weights/model_final.pth` (129024-d, 2026-08-11), réglage de
fitting de production (`inner_steps_eval=20`, `lr=0.01`, 1 048 576 points), 8 volumes
T1W/3T `retro_train`, nRMSE **foreground** :

| | moy. | min | max |
|---|---|---|---|
| **fit de z** | **0.2430** | 0.2301 | 0.2585 |
| moyenne leave-one-out (« sans z ») | 0.3100 | 0.2881 | 0.3533 |
| **z = 0** (cassé simulé) | 0.4443 | 0.4048 | 0.4870 |

**8/8 volumes gagnés**, sans exception, 22 % de marge. Ratio z0/fit **1.83** contre un
seuil à 1.05. Le mécanisme de fitting de production ne s'effondre pas sur le cerveau
moyen — affirmation jamais vérifiée jusqu'ici.

### Le smoke backbone : faible, pas cassé

4096-d, 2000 pas, 8 volumes, 1 mm (loss 0.06441 → 0.00021 en 1191 s sur GB10) :

| porte | chiffre | verdict |
|---|---|---|
| `[2/5]` discrimination de z | diff relative des recons **0.2103** (vérité terrain 0.2588), ‖z0−z1‖ = 0.0975 | ✅ |
| `[3/5]` fit vs moyenne LOO | **0.3175** contre **0.3533** | ✅ |
| `[4/5]` invariance à la résolution | erreur relative L2 **0.0056** ; recon@full 0.3175, recon@demi 0.3178 | ✅ |
| `[5/5]` contrôle négatif z=0 | **0.3942** contre 0.3175 → ratio **1.24** (seuil 1.05) | ✅ |

Ses 0.3175 tombent **entre** le sain de production (0.2430) et le cassé (0.4443) : le
budget de 2000 pas, calibré pour 40 volumes à 2 mm, est simplement insuffisant à 1 mm.
Faible, pas défaillant — distinction que l'ancienne porte absolue ne permettait pas.

### Confirmation de bout en bout, et une réserve sur `[5/5]`

Premier passage complet `[1/5]`→`[5/5]` en une exécution (les portes n'avaient jusque-là
été vues qu'en rejeu sur backbone rechargé) : **les cinq passent**, code de sortie 0.
Log : `outputs/smoke_inr_1mm_20260904_full.log`.

| | fit | LOO | z=0 | ratio `[5/5]` |
|---|---|---|---|---|
| smoke, backbone A | 0.3175 | 0.3533 | 0.3942 | 1.24 |
| smoke, backbone B | 0.3185 | 0.3533 | **0.3517** | **1.10** |
| production | 0.2430 | 0.3100 | 0.4443 | 1.83 |

**Le contrôle négatif a peu de marge sur un backbone smoke : 1.10 contre un seuil de
1.05.** Le fit est reproductible d'un entraînement à l'autre (0.3175/0.3185) et la
baseline LOO est identique au chiffre près (0.3533, elle est déterministe) — c'est **z=0
qui varie**, de 0.3942 à 0.3517 : le « cerveau moyen » produit sans information de volume
dépend du tirage d'entraînement. **`[5/5]` peut donc échouer un jour sans que rien ne soit
cassé** ; ce serait un faux positif sur le test, pas un verdict sur le backbone. Cause
connue et déjà dite par `[3/5]` : 2000 pas ne suffisent pas à 1 mm. En production la marge
est de 1.83, sans ambiguïté. À surveiller ; ne pas relever le seuil sans avoir d'abord
augmenté le budget du smoke.

### Trois défauts trouvés dans le correctif lui-même

1. **La baseline trichait** (découvert au premier lancement, entrée du 2026-09-03 nuit).
   La moyenne « sans z » incluait le volume à reconstruire. Fuite mesurée : **+0.0388**
   de nRMSE_fg (0.2716 cible incluse contre 0.3104 en leave-one-out, moyenné sur les 8).
   Verdict inversé : le même fit à 0.3213 échouait contre 0.3091 et passait contre 0.3541.
   **Cause : `N_VOLUMES` 40 → 8**, imposé par la mémoire à 1 mm, qui fait passer le poids
   de la cible dans sa propre référence de 1/40 à 1/8. Le changement de régime a converti
   un biais inoffensif en biais décisif, dans la modification même qui créait la baseline.
2. **`[4/5]` plantait** : `RuntimeError: The size of tensor a (1000000) must match the size
   of tensor b (8257536)`. Le test sous-échantillonnait les coordonnées (1M) mais passait
   le volume entier (8.25M voxels) à `fit_new_volume` — coordonnées et valeurs
   désappariées. `[3/5]` y échappait en passant la grille pleine + `num_points`, ce qui
   laisse `sample_points` échantillonner les deux **ensemble**. Corrigé de la même façon.
3. **La justification écrite était fausse.** Le commentaire « l'ancien seuil `nrmse_fg <
   0.6` laissait passer le cassé, la production 1 mm est à 0.6383 » compare le nRMSE
   **Task 3 du flow de bout en bout** au nRMSE_fg de **reconstruction du backbone**, qui
   vaut 0.2430. Deux grandeurs incommensurables. La conclusion tient, mais la vraie
   démonstration est celle d'aujourd'hui : **z=0 vaut 0.4443, donc l'ancien seuil de 0.6
   l'acceptait**.

### Garde-fou D0 : d'abord muet, puis armé par rétro-remplissage

Au premier chargement le garde-fou n'imprimait rien : `backbone_cfg` était absent du
checkpoint de production (clés réelles
`['iter','model','ema','optimizer','scheduler','cfg_path']`), donc `diff_inr_configs`
renvoyait `[]` — muet exactement là où il compte.

**Rétro-rempli le 2026-09-04**, avec la règle de n'inscrire que ce qui est sourcé :

| | champs | statut |
|---|---|---|
| **Prouvés par les poids** | `latent_dim` 129024, `hyper_hidden_dim` 512, `hidden_dim` 256, `num_hidden_layers` 6, `modulate_scale` False, `lora_rank` 0 | lus dans les formes des 18 tenseurs, puis `load_state_dict(strict=True)` accepté |
| **Non récupérables d'un checkpoint** | `omega_0`/`omega_hidden` 30.0, `inner_lr` 0.01, `inner_steps_train` 10, `inner_steps_eval` 20, `fg_weight` 5.0, `bg_threshold` −0.9 | repris du `cfg_path` enregistré (`configs/mmfm/inr_backbone.yaml`, inchangé depuis son unique commit 223e5f4) |

**Ce n'est pas une mesure et le checkpoint le dit** : une clé
`backbone_cfg_provenance` y consigne la date, la source, la liste de ce qui est vérifié,
la liste de ce qui ne l'est pas, et le chemin de la sauvegarde. Corroboration pour les
champs non vérifiables : le run a duré 48 287 s pour 50 000 itérations, soit 0.97 s/iter
contre les 0.98 s/iter annoncés par la config pour `latent_dim=129024`, et tous les
commentaires datés de ce fichier (≤ 2026-08-10) précèdent l'entraînement
(2026-08-10 16:29 → 2026-08-11 05:54).

**Intégrité vérifiée** : les 36 tenseurs (`model` + `ema`) sont identiques au bit près à
la sauvegarde `model_final.pth.bak_20260904` (md5 `e68c642b8454b2175b33c8d62a0b99af`),
`iter=49999` et `cfg_path` préservés, écriture atomique via `os.replace`.

**Le garde-fou est maintenant prouvé dans les deux sens** — ce qui n'avait jamais été
fait : cas nominal, il imprime « cohérence INR : 13 champs » ; cas de divergence de
PROCÉDURE (`inner_lr` 0.01 → 0.05, `inner_steps_eval` 20 → 10, deux changements qui ne
touchent aucune forme de poids et passeraient `load_state_dict` en silence), il lève une
`ValueError` nommant les deux champs et leurs deux valeurs.

### Parité de calcul

`src/cfm/gate_inr_production.py` **importe** `_fg_nrmse`, `_resample_idx` et
`_load_smoke_volumes` de `test_inr_backbone_smoke` au lieu de les réimplémenter : les deux
backbones sont jugés par le même code, aux mêmes points query, sur les mêmes volumes. Seul
le backbone change. Contrôle croisé : la baseline LOO calculée par ce script (0.3100)
retrouve celle mesurée séparément avant correction (0.3104).

### Outillage

`--load-checkpoint` rejoue les portes `[2/5]`–`[5/5]` sur un backbone déjà entraîné
(~4 min) au lieu de repayer les ~20 min de méta-entraînement ; la sauvegarde a été
déplacée **avant** les portes, puisque c'est précisément quand une porte échoue qu'on veut
rejouer sans réentraîner. Le checkpoint smoke embarque désormais sa config, et le
rechargement applique la même vérification que D0.

---

## 2026-09-03 (nuit) — **Le nouveau garde-fou du smoke test échoue, mais sa baseline triche**

**Verdict : la porte `[3/5]` est INVALIDE en l'état — sa référence « sans z » inclut le
volume qu'elle sert à reconstruire.** Log : `outputs/smoke_inr_1mm_20260903.log`.

Premier lancement du smoke test INR remis au régime de production (1 mm, 192x224x192,
8 volumes) avec sa nouvelle porte relative « le fit de z doit battre la moyenne du
dataset ». Méta-entraînement sain (loss 0.06677 -> 0.00034 en 2000 pas, 18 min sur GB10),
`[2/4]` discrimination de z passée (diff relative des reconstructions 0.2063 contre
0.2588 pour la vérité terrain). Puis :

| | nRMSE_fg |
|---|---|
| fit de z (backbone smoke, 2000 pas) | 0.3213 |
| moyenne du dataset, **cible incluse** (baseline du test) | 0.3091 |
| moyenne du dataset, **leave-one-out** (baseline honnête) | 0.3541 |

**Le verdict s'inverse selon la baseline** : échec contre la moyenne qui a vu la cible,
succès de 9.4 % contre la moyenne leave-one-out. Mesuré sur les 8 volumes, la fuite vaut
**+0.0388 de nRMSE_fg** (0.2716 avec la cible contre 0.3104 sans).

**La cause est dans le même changement.** `N_VOLUMES` est passé de 40 à 8 (le stack 1 mm
ne tient pas en mémoire autrement). À 40 volumes la cible pesait 1/40 de sa propre
référence — négligeable. À 8 volumes elle en pèse 1/8, et domine la décision. Le passage
au régime de production a converti un biais inoffensif en biais décisif, dans la
modification même qui introduisait la baseline.

**Ce qui reste non mesuré** : l'assertion arrête la série avant `[4/5]` (invariance à la
résolution) et surtout avant `[5/5]`, le contrôle négatif à z=0 — c'est-à-dire avant la
porte qui devait *prouver* que `[3/5]` sait séparer un mécanisme sain d'un cassé. La
raison d'être de la Phase 0.2 reste donc à établir.

**Réserve, à ne pas confondre** : ce backbone est celui du smoke test (4096-d, 2000 pas,
8 volumes), **pas** le backbone de production (`outputs/mmfm/inr_backbone/weights/
model_final.pth`, 129024-d, 2026-08-11). Rien ici ne dit quoi que ce soit sur l'INR de
production — cette mesure-là n'a pas encore été faite.

**Défauts annexes constatés au passage** : le test `[2/4]` n'a pas été renuméroté `[2/5]` ;
le docstring du module annonce toujours « 4 tests / 40 volumes / nRMSE et SSIM » alors que
`compute_ssim` n'est plus importé ; le garde-fou D0 (`backbone_cfg` dans le checkpoint)
est muet sur le backbone de production, dont les clés sont
`['iter','model','ema','optimizer','scheduler','cfg_path']` — vérifié.

**→ Résolu le 2026-09-04 (entrée ci-dessus).** Baseline corrigée en leave-one-out,
porte repassée, et le backbone de PRODUCTION mesuré pour la première fois.

---

## 2026-09-03 (soir) — **La composante individuelle n'est pas prédictible depuis la source. La série est close.**

**Verdict : NON, sous la seule forme testable avec 3 sujets appariés.** Détail :
`results/mmfm/predictable_20260903/manifest.md`.

| | moyenne | min | max |
|---|---|---|---|
| **R² dans l'échantillon (1 paramètre)** | **0.135** | 0.000 | 0.552 |
| **LOO, erreur modèle / translation** | **0.990** | 0.808 | 1.272 |
| α + 1 | 0.770 | 0.339 | 1.030 |

33/60 cellules battent la translation hors échantillon — signes **p = 0.52**.

**Forme testée, à UN paramètre.** Si la carte était affine en intensité,
`D_i − D̄ = (a−1)(z_i − z̄)` et la composante individuelle serait entièrement
prédictible par un scalaire. Un paramètre contre 3 × 129 024 dimensions de résidu.
Ajuster plus riche aurait interpolé trivialement — trois points dans un tel espace
sont toujours parfaitement explicables.

**Le R² dans l'échantillon suffit à conclure** : 0.135 avec un paramètre est une
*borne supérieure* sur ce qui généralise. **86.5 % de la composante individuelle
est orthogonale à la déviation source** — c'est de l'information sur l'acquisition
CIBLE, absente de la source, donc **irréductible** quelle que soit l'architecture.

**Recoupement indépendant** : le 2026-08-25 avait établi que l'écart d'intensité
d'un sujet à sa classe n'est pas transportable d'un champ à l'autre — « c'est une
propriété de l'acquisition cible » — avec 35 % de variance inter-sujets. Deux
chemins sans rapport, même conclusion.

**Observation réelle** : les α sont **systématiquement négatifs** (α+1 = 0.770 en
moyenne, toutes les cellules sous 1.03) — la déviation individuelle *rétrécit* en
montant en champ. Cohérent, mais 13.5 % de la variance seulement.

### La série est close, et la cause est nommée

Huit leviers de mécanisme, puis le couplage (toutes dimensions, 2026-09-03), puis
la prédictibilité depuis la source. Tout est mesuré, tout est négatif, et la cause
n'est pas contournable avec ces données :

> **0 sujet apparié sur 1056**, et une composante individuelle du transport qui
> représente 71.7 % du déplacement et dont 86.5 % est absente du volume source.

Le plateau à **0.3737** brut / **0.2143** structurel n'est pas un défaut de mise
en œuvre : **c'est la borne de ce que l'information disponible permet.** Le
plafond de représentation à 0.1048 suppose un flow parfait, ce que l'absence
d'appariement rend inatteignable.

**Limite assumée** : seule la forme affine/scalaire est testable ici. Une fonction
plus riche de `z_src` pourrait en principe capter davantage, mais aucune n'est
vérifiable avec 3 sujets appariés.

## 2026-09-03 — Couplage en dimension réduite : SANS OBJET, et mon explication du 30/08 était fausse

**Verdict : l'OT en dimension pleine apparie parfaitement (180/180).** Il n'y a
rien à restaurer. Détail : `results/mmfm/coupling_20260903/manifest.md`.

### Le test direct, sur nos latents

Les 3 sujets de `Training_prospective` sont les seuls à exister à plusieurs
champs : on connaît donc l'appariement véritable. Hasard = 1/3 (espérance du
nombre de points fixes d'une permutation de 3 éléments).

| méthode | justes | taux |
|---|---|---|
| **OT plein** (129 024 dim) | **180/180** | **1.000** |
| ACP 4 | 180/180 | 1.000 |
| aléatoire 4 | 168/180 | 0.933 |
| *hasard* | | *0.333* |

### Ce que cela corrige

**Mon explication du 2026-08-30 — « la concentration des distances rend la
structure de plus proche voisin essentiellement aléatoire » — est FAUSSE.** L'OT
apparie parfaitement dès que des correspondances existent.

**La vraie raison de son inutilité** : il n'y a aucune correspondance à trouver.
0 sujet sur 1056 n'existe à deux champs, donc l'OT ne peut coupler que le sujet A
au champ f avec le sujet **B** au champ g — et `A@f → B@g` contient (anatomie de
B − anatomie de A), qui n'appartient pas à la transformation de champ. **Aucune
méthode de couplage, en aucune dimension, ne peut retrouver une correspondance
absente des données.**

Cela recadre le rôle de l'OT en flow matching non apparié : **réduction de
variance** pour la carte *marginale*, pas récupération de correspondances
individuelles. Task 3 demande la carte individuelle.

### Le harnais, une fois de plus non représentatif

Il ne reproduit pas l'effondrement (0.767 à dim 129 024, très loin du hasard),
parce que **son facteur sujet est unidimensionnel** alors que l'anatomie réelle est
de haute dimension. Deux enseignements réels tout de même : l'ACP-4 bat l'OT plein
partout et l'écart croît avec la dimension (0.897 contre 0.767) ; la projection
aléatoire détruit tout (−0.045), donc ce qui compte est la *direction* du signal.
Troisième fois que le harnais s'avère non représentatif du régime réel.

### Réserve

Avec 3 sujets l'affectation est 3×3 et l'anatomie est bien plus distinctive que
l'écart entre champs : l'OT y réussit trivialement. « L'OT apparie correctement »
n'est établi que dans ce régime facile. **Le point logique n'en dépend pas** : il
porte sur l'absence de correspondances, pas sur la capacité de l'OT.

### État

**Le couplage est clos, toutes dimensions.** Reste une seule question ouverte :
la composante individuelle de 71.7 % est-elle **prédictible depuis le volume
source** ? Si oui, un modèle pourrait l'apprendre sans couplage. **3 sujets
appariés ne permettent pas de trancher.**

## 2026-09-02 (soir) — **La piste du rang effectif aboutit : le flow translate, le vrai transport non.**

**Verdict : le flow appris est une translation à 3.5 % près, alors que le vrai
déplacement champ→champ est propre au sujet à 71.7 %. Un facteur 20.** Détail :
`results/mmfm/translation_20260902/manifest.md`.

### Les deux observations du 2026-08-31 se réduisent à UN fait

Il manquait le témoin **source**. Le diagnostic comparait prédit contre
réel-à-la-cible ; ajouté le nuage source :

| | valeur |
|---|---|
| **part du déplacement propre au sujet** | **0.0348** |
| dispersion prédite / **source** | **0.9972** |
| rang effectif prédit / **source** | **1.0024** |

Le nuage prédit **est** celui de la source, transporté en bloc. **Le « déficit de
rang de 0.862 » n'existait pas** : le rang prédit égale celui de la source (15.2 /
17.3 / 18.6) et le 0.862 le comparait à celui des **cibles** (19.3 / 20.8 / 19.6).

### La mesure qui en fait un défaut

Une translation n'est fautive que si le vrai transport en est un autre — et
attention, Task 3 juge par SUJET, or une translation préserve exactement
l'identité du sujet. Mesuré sur les **3 sujets appariés** (les seuls du jeu ;
45 latents encodés pour l'occasion) :

| | part propre au sujet |
|---|---|
| **vrai déplacement** | **0.7173** (min 0.286, max 1.528) |
| **flow appris** | **0.0348** |

`‖D̄‖` vaut 1500–3000, l'écart-type autour 1000–1800 : **la composante manquée est
du même ordre que celle qui est captée.**

### Pourquoi le flow ne PEUT pas l'apprendre

Conséquence directe du non-appariement. Le modèle voit une source `z_i` au champ f
et une cible `z_j'` au champ g appartenant à **un autre sujet** : la composante
individuelle de `z_j' − z_i` est, de son point de vue, du bruit. L'espérance
conditionnelle est le décalage **commun**. Le couplage OT existe pour réparer
cela, et il est **vacuous à 129 024 dimensions** (mesuré le 2026-08-30 : +0.2 %).

**0 sujet apparié sur 1056 → couplage indépendant → part individuelle
inapprenable → le flow ne peut que translater.**

### Ce que cela dit des HUIT résultats négatifs

Tous réparaient la **machinerie** d'un flow qui apprenait structurellement la
mauvaise chose, pour une raison d'**information** et non de mécanisme. Trois
étaient de vrais défauts de code, et les corriger a réparé la géométrie
(`cos(v(0),v(1))` de 1.000000 à −0.24/0.80/0.03, courbure ×100) **sans toucher à
ce qui limite le score**. Le plateau cesse d'être un mystère.

### La limite de ce qui est affirmé

**Avec 3 sujets appariés on ne peut PAS trancher si ces 71.7 % sont prédictibles
depuis la source.** Un ajustement linéaire sur 3 points dans 129 024 dimensions
interpolerait trivialement. Deux lectures restent ouvertes : déplacement dépendant
de l'anatomie (apprenable avec un couplage qui marcherait en haute dimension), ou
variation d'acquisition de la cible non transportable — on sait déjà que les 35 %
de variance inter-sujets du niveau d'intensité sont de cette seconde nature.
**Ne pas conclure au-delà.**

## 2026-09-02 — MedVAE perceptuel adopté : NÉGATIF. Le gain de représentation ne se transmet pas.

**Verdict : le plafond descend de 6.9 %, le score ne bouge pas d'un iota.**
Détail : `results/mmfm/structural_20260902_lpips/`.

| | brut | structurel |
|---|---|---|
| production | 0.3794 | 0.2191 |
| **R-best** | 0.3737 | **0.2143** |
| batch8 | 0.3713 | 0.2147 |
| **LPIPS** | 0.3739 | **0.2147** |

- brut contre R-best : **+0.0002**, 29/60, signes p = 0.90, Wilcoxon p = 0.92
- **structurel contre R-best : +0.0004**, 97/180, signes p = 0.33, Wilcoxon p = 0.41

**La porte annonçait −0.0072. On mesure +0.0004.**

Contrôle mécanistique passé avant l'évaluation (5/5) : `cos(v(0),v(1))` =
−0.235/0.461/0.276, courbure moyenne 0.850 contre 0.776 pour R-best. Ce n'est donc
pas un entraînement raté — le modèle voit le temps et courbe sa trajectoire.

### Ce que ce résultat apprend, et qui est NEUF

C'est le premier cas où l'on a **à la fois** une amélioration de plafond mesurée
et certaine — **−0.0072, les 15 cellules sans exception** — **et** un effet nul de
bout en bout. Les sept nulls précédents portaient sur des mécanismes dont l'effet
réel était inconnu ; celui-ci a un gain établi **qui ne se transmet pas**.

L'attente avait été conditionnée explicitement : « au mieux et **si les erreurs
s'additionnent** ». **Elles ne s'additionnent pas.** Le gain de représentation est
intégralement absorbé par le flow, qui produit la même erreur totale dans un
espace latent meilleur.

**Conséquence sur un raisonnement tenu depuis le 2026-08-27** : la décomposition
« 72 % de l'erreur est le flow, 28 % la représentation », déduite du plafond,
**n'est actionnable dans aucun des deux sens**. Elle décrit une borne, pas un
budget d'erreur qu'on pourrait réduire terme par terme. Sept leviers de flow n'ont
rien donné ; améliorer la représentation ne donne rien non plus.

### Défauts de mon propre protocole, rencontrés en chemin

1. **Le `cache_id` n'est pas résolu pareil par les deux points d'entrée.** C'est un
   hash de (checkpoint VAE, spacing, volume_size, percentiles,
   `field_norm_stats_path`) ; `precompute_mmfm_latents.py` reçoit
   `--field-norm-stats` et produit `1989e9d1`, `train_mmfm_unified.py` ne le reçoit
   pas et cherche `4b1bb0a5`. Les configs de production **masquaient** ce défaut en
   fixant `latent_cache_dir` explicitement. En le retirant pour laisser la
   résolution automatique opérer, je l'ai exposé — et l'entraînement a échoué
   immédiatement.
2. **Mes chaînes n'arrêtaient pas sur échec.** `chain13` a enchaîné le contrôle
   mécanistique sur un checkpoint inexistant au lieu de s'arrêter : **~14 h de GPU
   inactif**. Les chaînes suivantes portent `set -e`.
3. **La loss n'est pas comparable entre VAE.** Base « prédire zéro » 21.06 ici
   contre 22.32 pour R-best : les latents perceptuels n'ont pas la même échelle.
   Seul le ratio l'est (0.819 contre 0.829). Même piège que le régime
   `adjacent_only` du 2026-08-26.

### État : plateau, sur les deux côtés

**Huit leviers, huit fois rien** au-delà du plancher de bruit de 0.002 : échelle du
temps, conditionnement FiLM, `adjacent_only`, recalibration d'intensité, couplage
OT, pas d'intégration, budget d'ajustement INR, MedVAE perceptuel. **Trois étaient
des défauts de code réels** — la géométrie du flow est passée de
`cos(v(0),v(1)) = 1.000000` à −0.24/0.80/0.03, courbure ×100.

Le seul progrès mesurable de toute la série reste **T2W structurel 0.2717 → 0.2598**
(R-best), six fois le plancher de bruit, invisible en nRMSE brut.

**Le plateau à ~0.3737 brut / ~0.2143 structurel est robuste à tout ce qui a été
essayé, des deux côtés de la chaîne.** Ce n'est pas un échec de mise en œuvre :
c'est une propriété mesurée du problème avec ces données — 3 sujets appariés, et
une variance inter-sujets de 35 % sur le niveau d'intensité.

**Fait non expliqué, à ne pas perdre** : le rang effectif des latents prédits est
la seule quantité déficitaire (0.862), et leur dispersion est **quasi identique
quel que soit le champ visé** — le flow déplace le centroïde avec `t` mais la forme
du nuage ne dépend presque pas de la cible. C'est la seule piste que la série n'a
pas refermée, et elle reste sans hypothèse.

## 2026-09-01 — Plafond du MedVAE perceptuel : porte franchie, mais de 6.9 % seulement

**Verdict : le plafond passe de 0.1048 à 0.0976 de nRMSE, les 15 cellules
s'améliorent sans exception.** Détail : `results/mmfm/ceiling_20260901/manifest.md`.

| | pré-entraîné | **LPIPS** | écart |
|---|---|---|---|
| T1W | 0.1081 | 0.0999 | −0.0082 |
| T2W | 0.1228 | 0.1157 | −0.0071 |
| T2FLAIR | 0.0836 | 0.0772 | −0.0063 |
| **moyenne** | **0.1048** | **0.0976** | **−0.0072** |
| SSIM | 0.9518 | **0.9579** | +0.0061 |

**Pourquoi il fallait mesurer avant d'adopter.** Les chiffres du 2026-08-25
(SSIM 0.9727 contre 0.9152) donnaient un écart **quatre fois plus grand**. Ils
proviennent d'une auto-reconstruction par **patches 64³** à 1 mm, sans le
rééchantillonnage 1 mm → 0.5 mm ni les sujets d'évaluation. Adopter sur cette base
aurait été payer une journée sur une comparaison de protocoles différents. La
mesure sous notre protocole coûte 1.3 h — paires identité, `dt = 0`, donc le flow
ne déplace rien et son entraînement sur les anciens latents est sans conséquence.

**Ce que cela borne.** Le vectorisé est à **0.2143** structurel pour un plafond de
0.1048 : l'écart de 0.1095 est celui que le critère d'arrêt vient de déclarer
irréductible par les leviers de flow. Abaisser le plafond ne rapporte, au mieux et
si les erreurs s'additionnent, que ces **0.0072** — 3.4 % du score structurel.
Au-dessus du plancher de bruit, donc mesurable ; pas de quoi franchir un palier.

**Adoption lancée** malgré cela, le gain étant réel et le levier étant le dernier
ouvert : cache régénéré (`medvae_finetune_1989e9d1`, 10.0 s/volume × 1939 ≈ 5.4 h,
554 Mo), puis entraînement (~2 h) et contrôle mécanistique. L'évaluation complète
(~6 h) n'est **pas** enchaînée automatiquement : la porte mécanistique se lit
d'abord.

## 2026-09-01 — Pas d'intégration et budget d'ajustement INR : les deux NÉGATIFS. **Critère d'arrêt atteint.**

### B — nombre de pas d'intégration : NÉGATIF

4 paires de saut maximal × 3 sujets × 3 contrastes = 36 volumes par réglage, là
où l'erreur de discrétisation doit être la plus forte.

| `n_steps` | brut | **structurel** |
|---|---|---|
| 20 (production) | 0.4587 | **0.2788** |
| 50 | 0.4583 | 0.2795 |
| 100 | 0.4577 | 0.2795 |

Contre 20 pas : **+0.0007**, **18/36**, signes p = 1.00, Wilcoxon p = 0.46 — pour
50 comme pour 100. **Rien.** C'était pourtant le seul levier dont le mécanisme
était *renforcé* par les résultats précédents (une droite s'intègre en un pas,
une courbe non). La trajectoire est bel et bien courbée maintenant, et
l'intégration à 20 pas la suit déjà assez bien.

### C — budget d'ajustement de l'INR : gain réel mais marginal, la limite est STRUCTURELLE

Auto-reconstruction T1W, `inner_steps_eval` ∈ {20, 60, 200} :

| pas | nRMSE | SSIM | contraste restitué |
|---|---|---|---|
| 20 | 0.3582 | 0.8452 | 0.632 |
| 60 | 0.3515 | 0.8500 | 0.680 |
| 200 | 0.3471 | 0.8527 | 0.696 |
| *MedVAE, pour comparaison* | *0.1048* | *0.9517* | *0.873* |

Monotone et réel, mais **−3.1 % de nRMSE pour 10× le budget**, et **5T/7T ne
bougent pas** (0.424 → 0.426 ; 0.555 → 0.547). La « limite de capacité » de l'INR
n'est donc **pas** un budget d'optimisation : c'est structurel. Question ouverte
depuis des mois, tranchée.

### CORRECTION — le « 20.6 % de contraste restitué » était une généralisation abusive

Ce chiffre, cité depuis le 2026-08-28 dans plusieurs manifestes et dans le plan,
provenait d'**un seul volume** : T2W@3T, sujet 0006 — la pire cellule. Mesuré
proprement sur les 3 contrastes × 5 champs × 3 sujets, l'ajustement INR à 20 pas
restitue **0.551** du contraste en moyenne :

| | 0.1T | 1.5T | 3T | 5T | 7T | moyenne |
|---|---|---|---|---|---|---|
| T1W | 0.650 | 0.489 | 0.592 | 0.714 | 0.714 | 0.632 |
| T2W | 0.693 | 0.282 | **0.245** | 0.259 | 0.516 | 0.399 |
| T2FLAIR | 0.887 | 0.631 | 0.437 | 0.540 | 0.618 | 0.623 |

La mesure d'origine n'était pas fausse ; **son extrapolation l'était**. La
conclusion qualitative tient (le handicap de l'INR est représentationnel : 0.551
contre 0.873 pour MedVAE) mais sa sévérité était surestimée d'un facteur ~2.7.
**Ne jamais généraliser une cellule à un tableau.**

### CRITÈRE D'ARRÊT ATTEINT

> **L'erreur structurelle de 0.2143 n'est pas réductible par les leviers de flow
> disponibles.** Le plafond de représentation à 0.1048 reste hors d'atteinte, et
> l'écart n'est ni un défaut de conditionnement, ni de couplage, ni de
> discrétisation, ni de loss.

Sept leviers, sept fois rien au-delà du plancher de bruit : échelle du temps,
conditionnement FiLM, `adjacent_only`, recalibration d'intensité, couplage OT,
pas d'intégration, budget d'ajustement INR. Trois étaient des **défauts de code
réels** — la géométrie du flow est passée de `cos(v(0),v(1)) = 1.000000` à
−0.24/0.80/0.03, courbure ×100. Le seul progrès mesurable reste **T2W structurel
0.2717 → 0.2598**.

**C'est un résultat, pas un échec.** Il ferme le côté flow. Ce qui reste ouvert :
la représentation (MedVAE perceptuel, dette n°5 — la barre avait été franchie le
2026-08-25 : SSIM 0.9727 contre 0.9152), ou l'acceptation du plateau.

**Fait non expliqué, à ne pas perdre** : le rang effectif des latents prédits est
la seule quantité déficitaire (0.862), et leur dispersion est quasi identique
quel que soit le champ visé. Le flow déplace le centroïde avec `t` mais la forme
du nuage ne dépend presque pas de la cible.

## 2026-08-31 — Contraction des latents : hypothèse RÉFUTÉE (c'est l'inverse)

**Verdict : les latents prédits sont PLUS dispersés que les réels, pas moins.**
Détail : `results/mmfm/contraction_20260831/manifest.md`.

| rapport prédit / réel | valeur |
|---|---|
| amplitude (écart-type par élément) | **1.166** |
| dispersion inter-sujets | **1.152** |
| **rang effectif (diversité)** | **0.862** |

Contrôle de cohérence : sur un pas d'identité (`dt = 0`), écart relatif
**exactement 0.00e+00** — la mesure est fiable.

L'hypothèse testée : L1 estime la médiane conditionnelle, sur un couplage
indépendant (l'OT étant vacuous à 129 024 dimensions), donc le flow devrait rendre
une tendance centrale. **Faux.** C'est la **troisième** réfutation de la piste
L1/tendance centrale — après ‖médiane‖/‖moyenne‖ = 0.98–1.01 et l'échec
d'`adjacent_only`, tous deux le 2026-08-26.

**La partie D du plan (loss L2 + standardisation, 4 h de GPU) n'est donc PAS
lancée.** C'est précisément ce que le diagnostic devait décider, et il l'a fait en
quelques minutes au lieu de deux réentraînements.

**Observation neuve, sans correctif proposé.** Le rang effectif est la seule
quantité déficitaire (T1W 15.2/19.3, T2W 17.4/20.8, T2FLAIR 18.6/19.6) : perte de
**diversité à taille constante**, pas effondrement vers la moyenne. Et la
dispersion prédite est **quasi identique quel que soit le champ cible** (T1W :
2458, 2459, 2460, 2446 contre 2208, 2277, 2465, 2171 pour les vrais) — le flow
déplace le centroïde avec `t` mais la forme du nuage ne dépend presque pas du
champ visé. Consigné sans hypothèse : ce projet a assez payé celles formulées trop
vite.

## 2026-08-30/31 — `batch_size: 8` (OT enfin actif) : NÉGATIF, et le harnais s'est trompé de 20 points

**Verdict : activer le couplage OT ne change rien de mesurable.** Détail :
`results/mmfm/structural_20260830_batch8/`.

**Le défaut de code était réel.** Toutes les configs de production sont à
`batch_size: 1`. Le couplage OT permute les cibles *à l'intérieur d'un lot* :
avec un seul échantillon de chaque côté il n'a rien à permuter. `ot_method:
exact` n'a donc jamais rien fait — vérifié directement, `ot_sampler.sample_plan`
ne réordonne rien à batch 1 et réordonne à batch 8. Le flow était entraîné en
couplage **indépendant** depuis le début.

**Le corriger ne rapporte rien.**

| | production | R-best | batch8 | écart / R-best | apparié |
|---|---|---|---|---|---|
| nRMSE officiel | 0.3794 | 0.3737 | 0.3713 | −0.0024 | 34/60, signes p = 0.37, Wilcoxon p = 0.53 |
| **nRMSE structurel** (échelle oracle retirée) | 0.2191 | **0.2143** | **0.2147** | **+0.0004** | **95/180, p = 0.50** |

Config : `configs/mmfm/vectorized_batch8.yaml`, 2.12 h d'entraînement. |

Sur la métrique de travail — celle construite exprès pour voir les progrès
structurels — l'écart est de **+0.2 %**. Le harnais annonçait **−19 %**.

**Pourquoi le harnais s'est trompé, et ce que ça coûte à sa crédibilité.** L'OT
par mini-lot cherche des correspondances entre 8 points dans un espace à
**129 024 dimensions** : à cette échelle, la structure de plus proche voisin est
essentiellement aléatoire, et le « couplage » n'apporte aucune information.

> **CORRECTION du 2026-09-03 — cette explication est FAUSSE.** Mesuré
> directement : l'OT sur les latents pleins (129 024 dimensions) apparie le bon
> sujet **180 fois sur 180** dès que des correspondances existent. La
> concentration des distances ne l'empêche pas. La vraie raison est qu'il n'y a
> **aucune correspondance à trouver** — 0 sujet sur 1056 n'existe à deux champs,
> donc l'OT ne peut coupler que des sujets DIFFÉRENTS. Voir
> `results/mmfm/coupling_20260903/manifest.md`. Le
harnais tournait en dimension 4096 avec 64 points — plus dense, donc l'OT y avait
un sens. **Ce n'est pas `batch_size: 1` qui rendait l'OT vacuous, c'est la
dimension.** L'augmenter à 8 ne change pas ça.

Le harnais a maintenant fait deux prédictions vérifiables sur données réelles :
le *mécanisme* de R-best (juste) et le gain structurel du couplage OT (**faux de
20 points**). **Il élimine ce qui ne peut pas marcher ; il ne prédit pas les
amplitudes.** Toute décision fondée sur une amplitude annoncée par le harnais
doit désormais être traitée comme une hypothèse, pas comme une estimation.

**Coût quasi nul en temps** : 3.31 it/s à batch 8 contre 3.48 à batch 1, 11.6 GB.
Le couplage OT était presque gratuit — il n'apporte simplement rien.

**Conséquence sur la suite** : l'interpolant cubique du papier MMFM repose sur un
**chaînage OT** des 5 marginales. Cette prémisse vient d'être invalidée à notre
échelle. La partie *cubique* (cible lisse en `t`) reste testable indépendamment
du couplage, mais l'ingrédient « trajectoire couplée » est à considérer comme
inopérant ici.

**Le motif à regarder en face : cinq correctifs de mécanisme, zéro gain de score.**
Échelle du temps, conditionnement FiLM, `adjacent_only`, recalibration
d'intensité, couplage OT. Trois étaient des défauts de code réels et mesurés — la
géométrie du flow est passée de `cos(v(0),v(1)) = 1.000000` à −0.24/0.80/0.03,
courbure ×100. Le seul progrès mesurable reste **local et structurel** : T2W
0.2717 → 0.2598 avec R-best, six fois le plancher de bruit, invisible en brut.

**Ce que cela impose** : ne plus enchaîner des réentraînements de 2 h sur la foi
d'un raisonnement mécanistique, et commencer par un diagnostic qui décide si le
levier existe. Plan révisé en conséquence, avec critère d'arrêt explicite.

*(Cette entrée fusionne deux rédactions indépendantes de la MÊME expérience, qui
coexistaient aux dates du 30 et du 31 — violation de la règle « une entrée par
expérience » de ce fichier.)*

## 2026-08-30 — Recalibration d'intensité : PISTE FERMÉE, négative

**Verdict : un succès sur trois architectures, deux explications réfutées, et une
cause qui n'est pas réparable par un meilleur estimateur.** Détail :
`results/mmfm/recalib_20260829/manifest.md` §8.

| architecture | avant | après | écart | apparié (60 paires) |
|---|---|---|---|---|
| **vectorisé R-best** | 0.3737 | **0.3525** | −0.0212 | 40/60, p = 0.014 |
| **UNet** | 0.4033 | 0.4688 | **+0.0654** | 28/60, p = 0.70 |
| **INR** | 0.3749 | 0.4690 | **+0.0941** | 19/60, p = 0.006 |

La clé `inference.intensity_recalibration` est **retirée** de la config de
production. Le code et les tables restent en place pour trace. Garder un réglage
qui marche sur 1 architecture sur 3 sans explication serait un réglage non
justifié.

**Deux hypothèses formulées, deux réfutées par la mesure.**

1. *Le terme `cos(p,g)` manquant* — le facteur oracle vaut `(‖g‖/‖p‖)·cos(p,g)`
   et l'estimateur n'en calcule que le rapport de normes. Vrai pour l'INR
   (prédictions plates). **Faux pour l'UNet** : `cos` y vaut **0.935–0.993**, et
   la corrélation à l'oracle n'est pourtant que de **0.344**.
2. *La norme L2 conflate intensité et taille du cerveau* — **faux** :
   CV(‖v‖) ≈ CV(moyenne cerveau) partout (0.213 contre 0.217 ; 0.140 contre
   0.140), le nombre de voxels ne varie que de 8–10 %.

**Cause retenue** : la **dispersion inter-sujets du niveau atteint 35 %**
(T1W@7T) face à une correction qui est une **constante par classe**, estimée sur
une population et appliquée à une autre. Problème de transfert de population, pas
d'estimation. Le seul protocole qui fonctionnait (leave-one-out, 0.3794 → 0.3231)
consomme les sujets d'évaluation et n'est pas déployable.

**Garde-fou : utile, insuffisant.** Rapport de contraste relatif prédictions/réel,
seuil 0.65. Sur l'INR il évite **+0.0891 des +0.0941** de dégât (52/60 paires
strictement inchangées ; des 8 modifiées, 4 gagnent et 4 perdent, p = 1.00 ex
æquo exclus). Sur R-best il ne bloque rien et le gain reste intact. **Sur l'UNet
il ne bloque rien non plus — et le dégât passe.** Calibré sur deux architectures,
il ne discrimine pas la troisième.

**Ce qui reste vrai et déplace la suite** : l'erreur d'échelle EST 80 % de
l'énergie de l'erreur, et un oracle par volume donnerait 0.2191 au lieu de
0.3794. **La marche existe mais n'est pas franchissable avec 3 sujets appariés.**
L'effort passe à la marche suivante, **0.2191 → 0.1048 : l'erreur STRUCTURELLE
du flow.**

**Piège de statistique, corrigé en cours de route** : j'avais annoncé « 4/60
victoires, p = 0.0000 » pour l'INR gardé, en comptant **52 égalités exactes**
(facteur 1.000) comme des défaites. Les ex æquo doivent être exclus d'un test des
signes — le bon calcul est 4 contre 4 sur 8, p = 1.00. À vérifier
systématiquement dès qu'une correction laisse des paires inchangées.

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
| 1 | ~~`test_inr_backbone_smoke.py` : seuil `nrmse_fg < 0.6` qui accepte le cassé, à 2 mm~~ **CORRIGÉ ET MESURÉ (2026-09-04)** — seuil désormais relatif (bat la moyenne leave-one-out), contrôle négatif prouvé | — |
| 2 | Aucun test qui compare la loss finale à « prédire zéro » — trois lignes, aurait tout arrêté | 15 min |
| 3 | Régénérer le cache INR sous le prétraitement corrigé | ~6 h GPU |
| 4 | ~~Constante de recalibration d'intensité par paire~~ **FERMÉE, NÉGATIVE (2026-08-30, RECONFIRMÉE 2026-09-18)** — voie sans appariement instruite deux fois sur deux checkpoints différents : gagnait sur le vectorisé en 2026-08-30 (−0.0212) mais dégrade significativement (+0.0344) le checkpoint actuel, post-audit — la direction même de la correction n'est pas stable d'un checkpoint à l'autre. Non réparable par un meilleur estimateur | — |
| 5 | ~~Adoption du MedVAE perceptuel : régénérer les caches + réentraîner les deux flows~~ **FAITE, NÉGATIVE (2026-09-02)** — cache régénéré, flow réentraîné, score inchangé (structurel 0.2147 contre 0.2143, p=0.33) : le gain de représentation est absorbé par le flow | — |
| 6 | ~~Géométrie du latent INR (25 % de structure commune contre 91 %) : canoniser l'ajustement~~ **DIAGNOSTIC RENVERSÉ (2026-09-07) puis FERMÉ (2026-09-14)** — ce n'est pas la géométrie mais le budget (facteur 252) ; correctif LoRA rang 16 tenté et clos négatif | — |
| 7 | Loss L1 au lieu de L2 : **mesuré le 2026-09-06, `l1` gagne sur nRMSE ET SSIM contre `mse`** (avec les 4 correctifs) — décision explicite de NE PAS adopter en production, pour ne pas introduire une deuxième inconnue sur des valeurs déjà validées (`vectorized_trajectory.yaml` garde `loss: mse`) | 15 min + réentraînement |
| 8 | ~~Échelle du temps~~ **FAIT** (`model.time_scale`, défaut 1.0) — mesuré SANS effet seul | — |
| 8b | **Conditionnement par modulation (`time_cond: film`) — le facteur dominant.** Fait pour le vectorisé/INR ; pour l'UNet, `unet_ts_adj.yaml` argumente que MONAI le fait déjà nativement (AdaGN) et prépare l'expérience, mais aucun run réel n'est retrouvé dans ce journal — gatée "après R-best", jamais relancée | fait + réentraînements |
| 9 | Standardisation du latent : **fait pour le vectorisé** (`latent_mean`/`latent_scale` dans `vectorized_trajectory.yaml`, 2026-09-04) ; toujours absent des configs `unet*.yaml` | 10 min + réentraînement |
| 10 | ~~Guidance sans classifieur : absente chez nous, présente dans les deux références~~ **IMPLÉMENTÉE ET TESTÉE, NÉGATIVE (2026-09-14/15)** — `cond_dropout` + `guidance_scale` sur 60 paires officielles : aggrave SSIM/LPIPS | — |
| 11 | ~~Interpolant cubique sur trajectoire couplée par chaînage OT (méthode réelle du papier MMFM) au lieu de droites par paire~~ **IMPLÉMENTÉ ET MESURÉ, NÉGATIF NET (2026-09-04/05)** — gain nRMSE sous le plancher de bruit (−0.0031), SSIM −0.0281 | — |
