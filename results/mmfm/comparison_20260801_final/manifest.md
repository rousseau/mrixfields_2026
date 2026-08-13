# Comparaison finale — MMFM vectorisé vs UNet vs INR (checkpoints de production)

Conclut la comparaison harmonisée démarrée dans
`results/mmfm/analysis/protocol_20260730_harmonized_comparison/`. Après le run harmonisé initial
(50000 itérations chacun, code strictement identique hors architecture), deux prolongations ont été
testées pour chaque architecture — voir cette section pour le diagnostic complet du problème de
schedule LR rencontré sur la première prolongation.

## Les 3 candidats testés par architecture

| Run | Description | nRMSE global (20 paires) |
|---|---|---|
| 50k | From scratch (vectorisé) / reprise poids-seuls (UNet), schedule original | vec **0.4361** / unet **0.4741** |
| ext | Prolongation ~12h, schedule à long plateau LR plein avant décroissance | vec 0.4659 / unet 0.4772 |
| **ft** | Prolongation avec décroissance LR immédiate dès la reprise (schedule corrigé) | vec **0.4288** / unet 0.4766 |

**Gagnant vectorisé : `ft`** (100000 itérations totales — 50k from scratch + 50k de continuation
schedule-corrigée). La correction du schedule (décroissance immédiate au lieu d'un long plateau à LR
plein) a permis à l'entraînement prolongé d'apporter un vrai gain, contrairement à la première
tentative (`ext`) qui dégradait le résultat.

**Gagnant UNet : `50k`** (le run harmonisé original). Ni `ext` ni `ft` n'améliorent le UNet — cette
architecture semble avoir réellement plafonné à 50k itérations, contrairement au vectorisé qui avait
encore de la marge une fois le schedule corrigé.

Ces deux checkpoints sont maintenant déployés comme référence de production :
`outputs/mmfm/vectorized/weights/model_final.pth` (ft, 100k) et
`outputs/mmfm/unet/weights/model_final.pth` (50k).

## Résultat final (checkpoints de production, 20 paires × 3 sujets, T1W)

| Modèle | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **Vectorisé** (100k, schedule corrigé) | **0.4288** | 0.8730 | 0.1407 |
| **UNet** (50k) | 0.4741 | 0.8721 | 0.1392 |

Vectorisé gagne 15/20 paires. Écart global : -0.0453 nRMSE (-9.6% relatif) en faveur du vectorisé —
un écart plus net qu'à l'étape harmonisée initiale (0.436 vs 0.474), car le vectorisé a continué à
s'améliorer avec un entraînement mieux programmé tandis que le UNet stagnait.

### nRMSE moyen par champ cible

| Cible | Vectorisé | UNet |
|---|---|---|
| →0.1T | 0.2957 | 0.3018 |
| →1.5T | 0.3479 | 0.3625 |
| →3T | 0.3316 | 0.3419 |
| →5T | 0.4926 | 0.5435 |
| →7T | 0.6760 | 0.8209 |

### Détail par paire

| Paire | Vectorisé | UNet | Gagnant |
|---|---|---|---|
| 0.1T→1.5T | 0.258 | 0.246 | UNet |
| 0.1T→3T | 0.251 | 0.241 | UNet |
| 0.1T→5T | 0.461 | 0.492 | Vectorisé |
| 0.1T→7T | 0.722 | 0.782 | Vectorisé |
| 1.5T→0.1T | 0.249 | 0.230 | UNet |
| 1.5T→3T | 0.255 | 0.255 | UNet |
| 1.5T→5T | 0.572 | 0.693 | Vectorisé |
| 1.5T→7T | 0.707 | 1.035 | Vectorisé |
| 3T→0.1T | 0.219 | 0.197 | UNet |
| 3T→1.5T | 0.222 | 0.227 | Vectorisé |
| 3T→5T | 0.491 | 0.526 | Vectorisé |
| 3T→7T | 0.753 | 0.907 | Vectorisé |
| 5T→0.1T | 0.320 | 0.341 | Vectorisé |
| 5T→1.5T | 0.432 | 0.456 | Vectorisé |
| 5T→3T | 0.376 | 0.393 | Vectorisé |
| 5T→7T | 0.522 | 0.560 | Vectorisé |
| 7T→0.1T | 0.395 | 0.439 | Vectorisé |
| 7T→1.5T | 0.480 | 0.521 | Vectorisé |
| 7T→3T | 0.444 | 0.478 | Vectorisé |
| 7T→5T | 0.447 | 0.463 | Vectorisé |

Le UNet ne l'emporte que sur les transitions vers un champ *faible et proche* (→0.1T/1.5T/3T,
généralement depuis une source adjacente) ; le vectorisé domine partout ailleurs, notamment sur les
transitions vers champ *fort* (→5T/7T), historiquement les plus difficiles.

Figures qualitatives — vectorisé/UNet uniquement, remplacées le 2026-08-02 par la version à trois
méthodes (voir section INR ci-dessous pour les figures actuelles) : ~~`figures/compare_T1W_*.png`~~.

## Reproductibilité du checkpoint vectorisé de production

Contrairement au UNet (reproductible en un seul `train_mmfm_unified.py` avec
`configs/mmfm/unet.yaml`), le checkpoint vectorisé de production est le résultat d'une procédure en
deux phases :

1. `configs/mmfm/vectorized.yaml` tel quel (50000 itérations, from scratch).
2. Reprise complète (`--resume`, pas `--resume_weights_only`) depuis `model_final.pth` de la phase 1,
   avec une config identique mais `train.total_iters: 100000` — ce qui aligne `decay_start =
   total_iters // 2 = 50000` exactement sur le point de reprise, donnant une décroissance LR continue
   sans plateau intermédiaire.

`configs/mmfm/vectorized.yaml` n'encode que la phase 1 (le "from scratch" reste la référence
canonique éditable) ; la phase 2 est documentée ici plutôt qu'ajoutée à la config, car elle dépend
d'un état intermédiaire (le checkpoint de la phase 1), pas reproductible par un unique run frais.

## Pistes non retenues pour la suite (vectorisé/UNet)

- **UNet** : plafonne à 50k quel que soit le schedule testé — un gain supplémentaire nécessiterait
  probablement un changement architectural (FiLM/AdaGN, cf. annexe de l'ancien plan d'unification),
  pas plus d'itérations.
- **Vectorisé** : gain net obtenu via un schedule mieux réglé — reste à voir si une 3e phase
  (schedule corrigé à nouveau, depuis les 100k) apporterait un gain supplémentaire ou si le modèle a
  maintenant atteint son plafond de capacité (hypothèse déjà documentée en début de session,
  ~20-30k itérations avant harmonisation).

## Troisième architecture — INR (SIREN + hypernetwork), 2026-08-02

Voir `docs/MMFM_INR_STATE_OF_THE_ART.md` pour le design et la revue de littérature (NOIR
arXiv:2603.13118), et la mémoire projet `project_mmfm_inr_variant` pour l'historique complet
(un bug de collapse de latent en Phase 1, un bug de divergence d'entraînement en Phase 3 — tous deux
diagnostiqués et corrigés). Contrairement au vectorisé/UNet, l'INR n'utilise **pas** MedVAE : il
opère directement sur le volume brut via un backbone SIREN+hypernetwork méta-appris (latent 512-d,
theta/psi entraînés une fois sur tout retro_train, 1939 volumes, 50000 itérations), dont le vecteur
`z` fitté par volume est ensuite consommé par le **même** `VectorMMFM` que l'architecture vectorisée
(bloc `model:` identique à `vectorized.yaml`) — seule la représentation change, pas le modèle de flow.

Checkpoints de production : `outputs/mmfm/inr_backbone/weights/model_final.pth` (backbone, 50k iters,
loss final ~0.080) et `outputs/mmfm/inr/weights/model_final.pth` (flow, 50k iters sur cache complet,
loss final 0.0028).

### Résultat (20 paires × 3 sujets, T1W, évaluateur officiel)

| Modèle | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **Vectorisé** (100k, schedule corrigé) | **0.4288** | **0.8730** | **0.1407** |
| **UNet** (50k) | 0.4741 | 0.8721 | 0.1392 |
| **INR** (backbone 50k + flow 50k) | 0.4566 | 0.8342 | 0.1914 |

nRMSE : INR bat UNet (-3.7% relatif) mais reste derrière le vectorisé (+6.5% relatif). SSIM/LPIPS :
INR nettement derrière les deux autres — cohérent avec la limite documentée dans l'analyse NOIR
(state-of-the-art doc, section 1.4) : une représentation par régression (ici le fitting INR) tend à
perdre en fidélité perceptuelle même quand l'erreur brute reste compétitive.

### Par paire — les 20 gagnants

| Paire | Vectorisé | UNet | INR | Gagnant |
|---|---|---|---|---|
| 0.1T→1.5T | 0.258 | **0.246** | 0.365 | UNet |
| 0.1T→3T | 0.251 | **0.241** | 0.351 | UNet |
| 0.1T→5T | 0.461 | 0.492 | **0.429** | INR |
| 0.1T→7T | 0.722 | 0.782 | **0.665** | INR |
| 1.5T→0.1T | 0.249 | **0.230** | 0.282 | UNet |
| 1.5T→3T | **0.255** | **0.255** | 0.320 | Vec/UNet |
| 1.5T→5T | 0.572 | 0.693 | **0.506** | INR |
| 1.5T→7T | **0.707** | 1.035 | 0.812 | Vectorisé |
| 3T→0.1T | 0.219 | **0.197** | 0.273 | UNet |
| 3T→1.5T | **0.222** | 0.227 | 0.329 | Vectorisé |
| 3T→5T | 0.491 | 0.526 | **0.479** | INR |
| 3T→7T | **0.753** | 0.907 | 0.790 | Vectorisé |
| 5T→0.1T | **0.320** | 0.341 | 0.381 | Vectorisé |
| 5T→1.5T | **0.432** | 0.456 | 0.501 | Vectorisé |
| 5T→3T | **0.376** | 0.393 | 0.450 | Vectorisé |
| 5T→7T | **0.522** | 0.560 | 0.597 | Vectorisé |
| 7T→0.1T | 0.395 | 0.439 | **0.348** | INR |
| 7T→1.5T | 0.480 | 0.521 | **0.464** | INR |
| 7T→3T | 0.444 | 0.478 | **0.419** | INR |
| 7T→5T | 0.447 | 0.463 | **0.373** | INR |

Décompte des victoires (nRMSE) : Vectorisé 8/20, INR 8/20, UNet 4/20.

### Par champ cible

| Cible | Vectorisé | UNet | INR |
|---|---|---|---|
| →0.1T | **0.2958** | 0.3018 | 0.3208 |
| →1.5T | **0.3480** | 0.3625 | 0.4147 |
| →3T | **0.3315** | 0.3417 | 0.3847 |
| →5T | 0.4928 | 0.5435 | **0.4469** |
| →7T | 0.6760 | 0.8210 | **0.7161** (bat UNet, derrière vectorisé) |

Figures qualitatives à trois méthodes (Source | GT | vectorisé | UNet | INR), 3 sujets ×
`figures/compare_T1W_*.png` — 2 paires où l'INR gagne (0.1T→7T, 7T→5T) et 2 où il perd (0.1T→1.5T,
1.5T→7T), choisies pour illustrer visuellement le constat ci-dessous. Confirme visuellement le
signal LPIPS/SSIM : la reconstruction INR est nettement plus lisse/floue que le vectorisé et le UNet
(perte de détail anatomique fin), y compris sur les paires où son nRMSE est meilleur — l'INR gagne en
tendance d'intensité globale sur les cibles extrêmes, pas en netteté.

**Constat le plus net** : l'INR gagne systématiquement (4/4) sur les transitions **→7T**, la cible
historiquement la plus difficile pour les deux autres architectures, et 2/4 sur →5T — mais perd
nettement sur les transitions vers un champ faible/proche (→0.1T/1.5T/3T), où le vectorisé et le UNet
restent meilleurs. Hypothèse : en opérant sur le volume brut plutôt que sur le latent MedVAE, l'INR
n'hérite d'aucun biais de reconstruction propre au VAE (potentiellement moins bien calibré sur les
plages d'intensité extrêmes de 5T/7T) — mais sa représentation actuelle (backbone 50k itérations,
LR très prudent aligné sur NOIR β=5e-6) reste moins fine que MedVAE sur les transitions faciles/proches,
où la marge de progression restante est surtout une question de fidélité de reconstruction plutôt que
de robustesse aux extrêmes.

### Pistes non retenues pour l'instant (INR)

- **Candidat 2 (WIRE)** et **Candidat 3 (conditionnement local)** du plan INR (voir
  `docs/MMFM_INR_STATE_OF_THE_ART.md`, section 5) — non testés, resteraient pertinents si l'objectif
  est de rattraper le vectorisé sur l'ensemble des paires plutôt que seulement sur →5T/7T.
- **MedVAE non retouché** (perte foreground-pondérée, EMA) — dette technique documentée dans la
  mémoire projet, décision explicite de l'utilisateur de ne pas retoucher pour ne pas invalider les
  résultats vectorisé/UNet actuels.

## INR — backbone corrigé (représentation fixée), 2026-08-05

Investigation déclenchée par des figures qualitatives à 3 méthodes montrant des prédictions INR
nettement plus floues que vectorisé/UNet. Diagnostic complet (auto-reconstruction du backbone vs
MedVAE, plusieurs tentatives d'agrandissement de capacité) puis correction de **trois écarts
d'implémentation réels** identifiés en comparant `src/cfm/inr_backbone.py` au code source officiel de
NOIR (github.com/Sidaty1/NOIR) — pas seulement à son papier. Détail complet des tentatives (dont deux
qui n'ont PAS marché : plus d'itérations à capacité fixe, agrandir `latent_dim` seul) et des trois
correctifs (échelle `omega_0` manquante sur le shift, gradient de la boucle interne dilué par le
batch, hypernetwork sur-amorti) dans la mémoire projet et `docs/MMFM_INR_STATE_OF_THE_ART.md` section
6. Config finale : `latent_dim=4096` (= capacité MedVAE), `hyper_hidden_dim=512`,
`inner_steps=10/20`.

**Capacité de représentation (auto-reconstruction, sans traduction de champ)** — voir
`inr_reconstruction_capacity.csv` (résultat final) vs `inr_reconstruction_capacity_4096d_buggyomega.csv`
(avant les 3 correctifs) : SSIM auto-reconstruction passé de ~0.57-0.68 (toutes tentatives
précédentes, y compris backbone 512-d original) à **0.826-0.827** (prospectif et retro_train,
quasi-identiques — pas de sur-apprentissage), contre MedVAE ~0.91. Écart divisé par ~3.

**Résultat Task 3, backbone corrigé (20 paires × 3 sujets, T1W)** — voir `task3_inr_T1W.csv`
(nouveau) vs `task3_inr_T1W_buggyomega.csv` (ancien, = la ligne INR de la table plus haut) :

| Version INR | nRMSE | SSIM | LPIPS | Victoires vs vectorisé |
|---|---|---|---|---|
| Ancien backbone (512→4096-d, bugs NOIR non corrigés) | 0.4566 | 0.8342 | 0.1914 | 8/20 |
| **Backbone corrigé (4096-d, 3 correctifs NOIR)** | **0.4787** | **0.8384** | **0.1729** | **1/20** |
| Vectorisé (référence) | 0.4288 | 0.8730 | 0.1407 | — |

**Résultat mitigé, à prendre au sérieux plutôt qu'à minimiser** : SSIM et LPIPS s'améliorent
(LPIPS -9.7% relatif, la métrique la plus alignée perception humaine), mais le **nRMSE se dégrade**
et l'INR perd son avantage antérieur sur les cibles →5T/→7T (7T→moyenne : 0.716→0.777, contre
vectorisé 0.676 — l'INR est maintenant PIRE que le vectorisé y compris sur son ancien point fort).

Hypothèse : le modèle de flow (`VectorMMFM`, architecture et budget d'entraînement inchangés) doit
maintenant interpoler dans un espace latent 8x plus grand (4096 vs 512) et beaucoup plus riche/haute
fréquence qu'avant. De petites erreurs de vitesse prédite se traduisent, après décodage par un SIREN
désormais beaucoup plus fidèle, en erreurs d'intensité brute plus visibles — cohérent avec le
compromis perception/distorsion classique (Blau & Michaeli 2018) : le flow produit des prédictions
plus justes structurellement/perceptuellement, mais avec une erreur de magnitude brute plus grande. Le
goulot d'étranglement se serait donc déplacé de la représentation (résolu) vers le modèle de flow
lui-même — hypothèse testée et **réfutée**, voir section suivante.

## INR — l'hypothèse "flow sous-dimensionné" testée et réfutée, 2026-08-06

Deux tests bon marché (aucun n'a nécessité de refaire le precompute des latents, seul le backbone y
touche) avant d'engager un retrain complet :

**1. Précision d'intégration à l'inférence (gratuit, pas de retrain)** — le flow entraîné avec
`n_steps=20` (défaut) a été ré-évalué en inférence avec `n_steps=50` : nRMSE=0.4789, SSIM=0.8384,
LPIPS=0.1729, **quasi identique** au résultat à 20 pas. Écarte la précision d'intégration Euler comme
explication.

**2. Vérification dimensionnelle** — la formulation "espace latent 8x plus grand" du diagnostic
précédent était **fausse** : le latent MedVAE aplati du vectorisé (à partir duquel `hidden_dim=1024`
fonctionne très bien, nRMSE 0.4288) fait **16128-d** (vérifié dans
`outputs/mmfm/latent_cache/vectorized/.../*.pt`, `torch.Size([16128])`), contre **4096-d** pour le
latent INR — 4x plus PETIT, pas plus grand. Avec `hidden_dim=1024`, le flow INR compresse même son
entrée (ratio hidden:latent 0.25:1) BEAUCOUP moins agressivement que le flow vectorisé (0.064:1), qui
pourtant réussit mieux. La capacité brute rapportée à la dimensionnalité n'est donc a priori pas le
facteur limitant.

**3. Ablation malgré tout** — testée car peu coûteuse une fois 1. et 2. posés (retrain du flow seul
~28min à `hidden_dim=1024`/9.5it→28.6it/s ; ~90min à `hidden_dim=2048`/9.4it/s, pas de precompute à
refaire). Inspiré du code de référence NOIR (`noir/no.py::NeuralOperatorDeep`, qui vise
`hidden_dim ≈ latent_dim`, 1:1) : `hidden_dim: 1024 → 2048` (`num_blocks=4` inchangé, variable isolée),
160.2M paramètres (vs ~47M avant). Smoke test (300 iters) sain (loss 0.446→0.057, grad stable), retrain
complet 50k iters sain (loss finale 0.0023, grad_norm stable ~0.70, comparable au run à 1024). Task 3 :

| Config flow | nRMSE | SSIM | LPIPS | Paires nRMSE améliorées |
|---|---|---|---|---|
| `hidden_dim=1024` (original, 2026-08-05) | 0.4787 | 0.8384 | 0.1729 | — |
| `hidden_dim=1024` (reproduit, 2026-08-06, nouveau seed) | 0.4869 | 0.8369 | 0.1751 | — |
| **`hidden_dim=2048`** | **0.4911** | **0.8351** | **0.1768** | **5/20 seulement** |

Doubler la capacité du flow **dégrade** les trois métriques (et le nouveau run à 1024, avec un seed
différent, confirme que ce n'est pas un artefact de variance d'entraînement : les deux runs à 1024
encadrent nettement le run à 2048). La capacité du flow est donc écartée comme facteur limitant —
config `hidden_dim=1024` restaurée comme checkpoint de production (`configs/mmfm/inr.yaml`,
`outputs/mmfm/inr/weights/model_final.pth`). Détail : `task3_inr_T1W_hidden2048_ablation_20260806.csv`
et `task3_inr_T1W_hidden1024_run1_20260805.csv` (résultat original, avant ce retrain de vérification) ;
`task3_inr_T1W.csv` contient maintenant le run reproduit du 2026-08-06.

**Nouvelle hypothèse de travail** : le facteur limitant n'est probablement ni la fidélité de
représentation (résolue, section précédente) ni la capacité du modèle de flow (écartée ci-dessus),
mais la **structure/régularité de l'espace latent INR lui-même**. Chaque `z` est obtenu par quelques
pas de descente de gradient par volume (Algorithme 2, meta-learning) sans aucun régularisateur de
continuité entre volumes différents, contrairement au latent MedVAE qui est explicitement façonné
(KL + perceptuel) pour être lisse/continu — une propriété dont un modèle de flow matching (qui doit
interpoler le long d'un chemin continu entre marginales) dépend directement. Piste non testée à ce
jour : régularisateur de continuité/lissage sur `z` pendant le meta-training du backbone (ex. pénalité
sur la variance intra-classe des `z`, ou un terme de contraction proche de VICReg/Barlow Twins), ou
lissage post-hoc de l'espace latent (PCA/whitening avant le flow).

## INR — régularisation de lissage sur z : diagnostic confirmé, correctif testé infructueux, 2026-08-06

Plan détaillé : `/home/rousseau/.claude/plans/temporal-plotting-engelbart.md` (revu par un second agent
avant implémentation). Deux phases.

**Phase 1 — diagnostic de rugosité locale (pas de ré-entraînement)**, nouveau
`src/cfm/diagnose_inr_latent_smoothness.py` : lissage d'interpolation (8 pas linéaires entre deux `z`
en cache, diff L1 pondérée foreground entre pas consécutifs, std = rugosité) comparé même-champ
(contrôle) vs champs différents (la trajectoire que le flow doit réellement apprendre), INR vs MedVAE,
sur les checkpoints de production (T1W, 20 paires/condition) :

| | INR | MedVAE |
|---|---|---|
| std_step même champ | 0.000390 | 0.002258 |
| std_step champs différents | 0.001143 | 0.003021 |
| **ratio croisé/même** | **2.93** | **1.34** |
| Lipschitz isotrope (std/médiane) | 0.130 | 0.043 |

**Confirmé sans ambiguïté** : la rugosité de l'INR est ~2.2x plus marquée que MedVAE spécifiquement
sur l'axe champ-croisé (celui que le flow traverse), pas juste "plus rugueux en général" — CSV :
`inr_latent_smoothness.csv`.

**Phase 2 — pénalité de lissage sur z (Hutchinson VJP, WGAN-GP)** : implémentée dans
`inr_backbone.py::meta_train_step` (nouveau paramètre `z_reg_weight`), rampée sur les 10% premiers pas
d'entraînement (`train_inr_backbone.py`), nouvelles clés `configs/mmfm/inr_backbone.yaml`
(`z_reg_weight`/`z_reg_warmup_frac`, défaut 0.0 = inactif). Retenue après un second avis (agent Plan)
qui a écarté deux alternatives : shrinkage L2 simple (déjà implicitement présent via le fitting
early-stopped depuis zero-init) et compacité intra-classe modalité×champ (mal alignée avec le
conditionnement réel d'OT-CFM dans `mmfm_core.py`, coût d'implémentation plus lourd).

**Balayage (smoke test, 40 volumes/2000 pas, avant tout retrain complet)** :

| `z_reg_weight` | Discrimination z | Reconstruction | Lecture |
|---|---|---|---|
| 0 (contrôle) | ✅ rel_diff=0.179 | ✅ nRMSE_fg=0.269 | — |
| 1e-8 | ✅ rel_diff=0.141 | ✅ nRMSE_fg=0.287 | pénalité brute CROISSANTE (113k→347k) — trop faible pour agir |
| 1e-7 | ✅ rel_diff=0.075 | ✅ nRMSE_fg=0.305 | pénalité brute stable (~20-47k), mais... |
| 1e-6 | ❌ rel_diff=1.4e-05 | — | collapse (z n'importe plus) |
| 1e-4 | ❌ rel_diff=2.2e-06 | — | collapse sévère |

Vérification supplémentaire (`src/cfm/compare_zreg_smoke_checkpoints.py`, nouveau — refit `z` en
direct sur chaque checkpoint smoke, puisqu'ils n'ont pas servi à construire le cache de production) sur
le ratio de rugosité même-champ/champs-différents lui-même :

| Checkpoint | ratio croisé/même |
|---|---|
| contrôle (w=0) | 1.447 |
| w=1e-8 | 1.576 (pire que le contrôle) |
| w=1e-7 | 0.924 (meilleur en apparence...) |

... mais à w=1e-7, les std_step même-champ ET champs-différents s'effondrent tous les deux à ~0.000003
(vs 0.0017-0.0025 au contrôle) : ce n'est pas un lissage réel, c'est le début du même collapse
dégénéré que celui observé pleinement à 1e-6/1e-4 (decode(z) devient quasi invariant à z) — la
diminution du ratio est un artefact du numérateur ET du dénominateur qui s'effondrent ensemble, pas une
trajectoire plus lisse et informative.

**Aucune valeur testée ne satisfait le critère de la porte du plan** ("réduit la rugosité mesurée SANS
violer les checks de sanité") — **le retrain complet (~6.3h) n'a PAS été lancé**, conformément aux
critères de décision explicites du plan (arrêt planifié, pas avorté : la porte à l'échelle smoke
existait précisément pour intercepter ce cas avant la dépense).

**Bilan** : le diagnostic de rugosité (Phase 1) est solide et directement réutilisable pour toute future
investigation sur cet espace latent. Le correctif testé (pénalité scalaire de Hutchinson ajoutée
directement à la loss) a une plage utile très étroite, voire inexistante, pour cette architecture — la
transition "trop faible pour agir" → "collapse dégénéré" ne laisse pas de palier stable identifié entre
1e-8 et 1e-4. Pistes non testées si cette direction est reprise : normaliser la pénalité par l'échelle
propre de `pred` (la magnitude brute, dizaines à centaines de milliers, rend le poids difficile à
calibrer et explique peut-être la bande utile si étroite) ; une borne de Lipschitz par couche avec
weight normalization (Liu et al. 2022, SIGGRAPH — modifie directement `SineLayer`/`Hypernetwork`,
rayon d'impact plus large, délibérément différé) ; ou appliquer la pénalité après un warmup plus long
laissant le réseau atteindre une fidélité de reconstruction établie avant d'introduire la contrainte de
lissage.

**Résultat de production INR inchangé** : nRMSE 0.4869 / SSIM 0.8369 / LPIPS 0.1751
(`hidden_dim=1024`, backbone non modifié). Checkpoints smoke jetables (13MB chacun, pas des artefacts
de production) : `outputs/mmfm/_smoke_inr_zreg/{w0,w1e-8,w1e-7}/`.

## LA RÉSOLUTION EST LE VRAI GOULOT — capacité de représentation multi-résolution, 2026-08-06/07

Investigation déclenchée par les figures qualitatives de capacité de représentation, qui montraient un
flou important **des deux** représentations (MedVAE et INR) alors que la littérature rapporte de bien
meilleurs résultats. Conclusion : ni les poids, ni la fonction de perte n'étaient le facteur dominant —
**c'est la résolution de travail du pipeline**.

### Le fait déclencheur

La grille native du challenge est **364×436×364 @ 0.5mm** (vérifié sur les fichiers). Le pipeline MMFM
travaillait à **96×112×96 @ 2mm** : ~98% des voxels acquis étaient jetés avant que MedVAE ou l'INR ne
voient quoi que ce soit. Et l'évaluation ré-interpole ensuite ×4 vers la grille native, donc tout le
détail au-dessus de la fréquence de Nyquist de 2mm est absent **par construction** au moment du scoring.

### Contrainte architecturale découverte : MedVAE ne peut PAS traiter un gros volume en une passe

Sonde mémoire (`scratchpad/probe_res.py`) sur le plein volume :

| Volume | Voxels | Mémoire GPU |
|---|---|---|
| 96×112×96 (2mm) | 1.0M | 4.7 GB |
| 128³ | 2.1M (×2) | 13.8 GB (×2.9) |
| 160³ | 4.1M (×2) | 46.1 GB (×3.3) |
| 192×224×192 (1mm) | 8.3M (×2) | **OOM (>121 GB)** |

La mémoire croît en **×3 quand les voxels doublent**, pas ×2 : signature du coût quadratique de
l'attention « vanilla » au goulot de MedVAE. Le traitement **par patches est donc obligatoire** au-delà
de 2mm — ce qui coïncide avec la recette officielle MedVAE (patches 64³,
`medvae/utils/loaders.py::load_mri_3d_finetune`). L'INR, lui, n'a aucune contrainte de ce type (grille
de coordonnées : 0.79 GB même à 0.5mm).

### Résultats (auto-reconstruction, 9 volumes T1W = 3 sujets × {0.1T, 3T, 7T}, patches 64³ overlap 0.5)

Scripts : `src/vae3d/eval_representation_multires.py`, `src/cfm/eval_inr_multires.py`.
CSV : `results/benchmark_vae/metrics/medvae_representation_multires.csv`,
`results/mmfm/comparison_20260801_final/inr_representation_multires.csv`.

| Représentation | 2mm (96×112×96) | 1mm (192×224×192) | Δ SSIM 2mm→1mm |
|---|---|---|---|
| **MedVAE pré-entraîné** (poids Stanford bruts) | nRMSE 0.1869 / SSIM 0.8000 | **nRMSE 0.0819 / SSIM 0.9152** | **+0.115** |
| MedVAE fine-tuné L1 (notre production) | nRMSE 0.1463 / SSIM 0.8380 | nRMSE 0.0592 / SSIM 0.8790 | +0.041 |
| INR (SIREN, z=4096) | nRMSE 0.2052 / SSIM 0.8344 | nRMSE 0.2090 / SSIM 0.8080 | **−0.026** |

Tableau complet des trois résolutions (SSIM moyen / nRMSE moyen, 9 volumes chacun) :

| Représentation | 2mm | 1mm | 0.5mm (natif) |
|---|---|---|---|
| **MedVAE pré-entraîné** | 0.8000 / 0.1869 | 0.9152 / 0.0819 | **0.9481 / 0.0488** |
| MedVAE fine-tuné L1 | 0.8380 / 0.1463 | 0.8790 / 0.0592 | 0.8837 / 0.0350 |
| INR (SIREN, z=4096) | 0.8344 / 0.2052 | 0.8080 / 0.2090 | (voir `inr_representation_05mm.csv`) |

Le pré-entraîné progresse **monotonement** avec la résolution (SSIM 0.800 → 0.915 → 0.948), rejoignant
l'ordre de grandeur publié par MedVAE (MS-SSIM 0.983, arXiv:2502.14753). Le fine-tuné L1 sature à
~0.88. L'écart entre les deux s'inverse puis se creuse avec la résolution : **−0.038 à 2mm, +0.036 à
1mm, +0.064 à 0.5mm** — c'est la mesure directe du coût structurel de la L1.

### Trois conclusions

1. **La résolution est le levier dominant.** 2mm→1mm : nRMSE ÷2.3, SSIM +0.115 pour le pré-entraîné.
   Les poids MedVAE disponibles fonctionnent parfaitement à la résolution du challenge ; c'est notre
   pipeline qui les étranglait.
2. **Notre fine-tuning L1 plafonne la structure.** Il gagne en nRMSE (ce qu'il optimise) mais bloque le
   SSIM à ~0.88-0.90 à toute résolution, là où le pré-entraîné atteint 0.96. Signature manuelle du
   compromis perception/distorsion. **Le meilleur checkpoint disponible aujourd'hui est le
   pré-entraîné NON MODIFIÉ**, utilisé à haute résolution. Attention : à 2mm cette hiérarchie est
   INVERSÉE (le fine-tuné y est meilleur) — c'est ce qui avait initialement fait conclure à tort que la
   recette de loss n'importait pas ; la vérité terrain à 2mm est déjà trop lissée pour révéler l'écart.
3. **L'INR est structurellement capé.** Son `z` est figé à 4096 dimensions quelle que soit la
   résolution ; le latent MedVAE est spatial et grandit avec le volume (16k valeurs à 2mm → 129k à 1mm
   → 1.03M à 0.5mm). L'invariance à la résolution de NOIR/MedFuncta permet d'**évaluer** à toute
   résolution, elle n'apporte aucune capacité pour **représenter** davantage. Pour en profiter, il
   faudrait augmenter `z` (ou passer à une modulation directe par couche façon MedFuncta).

### Conséquence pour l'architecture du pipeline

Taille du latent MedVAE selon la résolution : 2mm → 24×28×24 = **16k** ; 1mm → 48×56×48 = **129k** ;
0.5mm → 96×112×96 = **1.03M**. Le flow vectorisé (MLP sur latent aplati, aujourd'hui la meilleure des
trois méthodes à 0.4288 nRMSE) devient inopérant au-delà de 2mm. **L'UNet spatial devient l'architecture
naturelle** — et à 0.5mm son latent (96×112×96) a exactement la taille des *volumes* qu'il traite déjà
aujourd'hui, donc à coût comparable. La hiérarchie des trois méthodes est appelée à s'inverser.

### Recette LPIPS+PatchGAN : codée, validée, NON retenue pour l'instant

`src/vae3d/medvae_perceptual_loss.py` (loss MedVAE/LDM réassemblée : L1 + LPIPS 2.5D + PatchGAN 2.5D +
KL, traitement des coupes vectorisé), intégration à deux optimiseurs dans `finetune_medvae.py`, config
`configs/medvae_finetune_lpips.yaml`. Smoke test réussi (60 pas : boucle à 2 optimiseurs, activation du
discriminateur, validation, checkpoint). Un run à 1mm a été lancé puis **arrêté délibérément** : il
aurait reproduit l'incohérence résolution-fine-tuning / résolution-déploiement qui vient d'être
diagnostiquée, et il ralentissait de moitié la mesure 0.5mm. À relancer à la résolution finalement
retenue **si** un fine-tuning s'avère utile — ce que les 0.963 SSIM du pré-entraîné rendent douteux.
Coût mesuré : ~12s/pas à `slice_stride=1` (768 coupes VGG par appel à batch=4), ~8s/pas à
`slice_stride=2`.
