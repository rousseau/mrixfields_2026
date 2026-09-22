# GAN latent (style UNSB) — Paliers 0 et 1, 2026-09-22

Voir le plan complet : `.claude/plans/` (session du 2026-09-21/22, revue de
littérature sur le conditionnement non apparié). Contexte : ce projet a
mesuré à trois reprises indépendantes (`level_cond`, recalibration
d'intensité, `src_cond` — voir CHANGELOG.md) un flow qui APPREND le signal
ajouté sans que le score s'améliore. Diagnostic théorique : sous perte L2
contre des cibles issues d'un couplage OT chaîné non informatif (aucun sujet
n'existe à 2 champs), l'estimateur optimal est l'espérance conditionnelle,
qui s'effondre vers une constante de population. Une perte adversariale a un
optimum en DISTRIBUTION (Goodfellow), pas point-à-point — mécanisme jamais
essayé sur le flow lui-même (tout le code GAN existant était confiné au
fine-tuning du VAE, `vae3d/medvae_perceptual_loss.py`).

Implémentation : `LatentDiscriminator` (mmfm_vectorized.py, discriminateur en
espace latent, jamais conditionné sur `z_src` — raison structurelle détaillée
dans le docstring de la classe), câblée dans `mmfm_core.py::train()` via
`ArchAdapter.build_discriminator` (nouveau champ optionnel, `None` par défaut
pour `arch_unet.py`/`arch_inr.py`). `train.lambda_gan` (défaut `0.0`) pilote
l'alternance G/D, sur le patron déjà établi par `lambda_cycle`/`lambda_lpips`
(garde-fous explicites, paliers de coût croissant avant tout entraînement sur
données réelles).

## Palier 0 (secondes, CPU)

- Deux bugs préexistants corrigés (prérequis indépendants du GAN, découverts
  en relisant `arch_synthetic.py` pour le brancher) :
  1. `arch_synthetic.py::make_model_fn` n'acceptait pas `level=...`, passé
     inconditionnellement par `mmfm_core.py` depuis l'ajout de `level_cond`
     (2026-09-18) — même oubli que celui déjà corrigé pour
     `arch_unet.py`/`arch_inr.py` à l'époque, manqué ici car ce fichier
     n'appelle pas `euler_integrate` directement.
  2. `SyntheticLatentDataset.__getitem__` renvoyait un 4-tuple, alors que
     `mmfm_core.py` (branche trajectoire) lit `ds[i][-1]` en attendant un
     `level` (Tensor). Ajouté comme constante `0.0` (aucun équivalent
     synthétique au niveau d'intensité).
- `LatentDiscriminator` : test unitaire de forme/gradient (sortie `(B,)`,
  pas de NaN, gradient jusqu'à `input_proj`/`blocks[0].cond_proj`).
- **Non-régression** (`test_mmfm_flow_behaviour.py::test_gan_inactive_is_noop`,
  nouveau) : `train()` lancé deux fois sur le harnais synthétique (24 iters,
  RNG globale remise au même état avant chaque run) — une fois sans aucune
  clé `lambda_gan*`, une fois avec `lambda_gan: 0.0` **plus** toutes les
  autres clés `gan_*`/`disc_*` à des valeurs actives-si-le-garde-fou-était-
  perce (`gan_start=0, gan_every=1, disc_lr=5e-3`≠`lr`). **Résultat : les 24
  points de la colonne `loss` sont identiques bit à bit.** Preuve directe
  (pas un proxy) : si le discriminateur était construit par erreur, son
  initialisation consommerait la RNG globale AVANT la première itération et
  décalerait l'init de G + tout shuffle de DataLoader qui suit.

  ```
  PYTHONPATH=src python src/cfm/test_mmfm_flow_behaviour.py --gan-noop-check
  ```

## Palier 1 (minutes, harnais synthétique, `configs/mmfm/synthetic_gan.yaml`)

4000 itérations (budget inchangé vs le témoin `synthetic.yaml`),
`lambda_gan=0.05` (calibration du plan : subordonné à la MSE, pas 1.0 —
défaut UNSB, qui n'a aucune perte de régression directe), `gan_start=500`.

**Comparaison `eval_synthetic.py` (vérité terrain EXACTE, la seule du
projet)** :

| | témoin (`synthetic.yaml`) | GAN (`synthetic_gan.yaml`) |
|---|---|---|
| fraction de courbure moyenne (porte ≥ 0.80) | 1.018 | 1.025 |
| pire nRMSE intermédiaire (porte ≤ 0.100) | 0.0672 | **0.0668** |
| porte | FRANCHIE | FRANCHIE |

**Aucune dégradation de la fidélité à la vérité terrain — légère
amélioration sur les deux mesures.** C'est le check anti-hallucination le
plus fort du protocole (le seul endroit où la vérité est connue exactement,
échantillon par échantillon) : si `lambda_gan` avait poussé G vers du
plausible-pour-la-classe déconnecté de la source, ce couple de métriques
l'aurait révélé directement, PAS un gain de netteté superficiel.

**Santé de l'entraînement adversarial** (`train_log_gan_palier1_synthetic.txt`, colonnes
`gan_g`/`gan_d`/`d_acc`/`grad_d`) :
- `d_acc` sur la fenêtre finale (step 3200→4000) : 0.67 → 0.67 → 0.70 → 0.72
  → 0.74 — dans la borne du plan [0.55, 0.90], dynamique saine (D légèrement
  en avance sur G, sans l'écraser).
- `gan_g` croît progressivement (0.29 → 0.54) au fur et à mesure que D
  s'améliore : pas de collapse de G, pas de stagnation de D à 0.5 (qui
  signalerait un D mort).
- `grad_d` (norme AVANT clip à `disc_grad_clip=1.0`) : borné sur toute la
  fenêtre active (0.06 à 12.50, sans tendance monotone), aucun blow-up.
- `nan` aux impressions à 200/400 : attendu, PAS un défaut — `gan_start=500`,
  les listes glissantes `recent_gan_*` sont vides avant ce pas (même
  convention d'affichage que `lambda_lpips`/`lambda_cycle` avant leur
  première activation).
- `test_mmfm_flow_behaviour.py` (suite standard) : 6/6 passent sur le
  checkpoint GAN, y compris `aucun gradient nul`.

**Verdict Palier 0-1 : POSITIF, aucun signal d'alarme.** Le mécanisme est
correctement isolé (no-op prouvé à `lambda_gan=0`), s'entraîne de façon
stable sur le problème à réponse connue, et n'a dégradé la fidélité sur
aucune des deux mesures qui comptent. Ne prouve PAS encore que le mécanisme
apporte un gain sur les données réelles (le problème synthétique n'a pas le
défaut structurel de `retro_train` — 0 sujet apparié sur 1056 — qui motive
l'expérience) : c'est précisément ce que le Palier 2 doit tester.

## Palier 2 (2026-09-22, `retro_train` réel, reprise poids-seuls)

Confirmation utilisateur obtenue avant lancement (question posée explicitement
— un GAN peut diverger, contrairement aux 4 leviers précédents qui ne
pouvaient qu'être neutres/négatifs). Deux runs, mêmes poids de départ
(`outputs/mmfm/vectorized_trajectory/weights/model_final.pth`,
`--resume_weights_only`), mêmes 2000 itérations, même `step_rng` (déterministe
par pas — les deux runs tirent donc EXACTEMENT les mêmes trajectoires/positions
d'ancrage/tirages de flip) :

- **GAN** : `configs/mmfm/vectorized_trajectory_gan.yaml` (`lambda_gan=0.05`,
  `gan_start=0` — test plus dur que le Palier 1, un modèle déjà bon se
  déstabilise-t-il dès l'activation ?). Log complet :
  `train_log_gan_palier2.txt`.
- **Témoin** : `configs/mmfm/vectorized_trajectory_gan_control.yaml` (copie
  stricte, `lambda_gan=0.0`). Log complet : `train_log_control_palier2.txt`.

**Les deux runs se sont terminés sans erreur** (`model_final.pth` écrit dans
les deux cas, aucun NaN/Inf, aucun crash).

**Loss de flow (G), GAN vs témoin — pas de dégradation** (`train_metrics.jsonl`
des deux runs, 40 points chacun, `print_every=50`) :

| | GAN | Témoin |
|---|---|---|
| moyenne | 1634.01 | 1677.55 |
| médiane | 1641.64 | 1639.77 |
| écart-type | 223.00 | 228.54 |
| moyenne des 5 derniers points | 1825.85 | 1767.92 |

Mann-Whitney U (deux distributions indépendantes de loss) : **p = 0.554** —
pas de différence statistiquement significative. La médiane est quasi
identique (1641.64 vs 1639.77) ; la moyenne du run GAN est même légèrement
INFÉRIEURE (ratio 0.974). Le critère du plan (« loss de flow pas dégradée vs
témoin ») est rempli sans ambiguïté.

`grad_norm` (G, norme avant clip) plus élevé sous GAN (moyenne 512.9 vs 417.1,
+23 %) : attendu — G reçoit désormais aussi le gradient adversarial en plus de
la MSE — sans que cela se traduise par une loss dégradée ni une instabilité
(`grad_clip=1.0` borne l'amplitude de la mise à jour effective dans les deux
cas).

**Vitesse** : les deux runs montrent le même ralentissement transitoire
(2.6→1.2 it/s puis retour à 3.5+ it/s) à des moments qui coïncident avec mes
propres commandes concurrentes (diagnostic per-cellule notamment) — PAS un
effet du mécanisme GAN : le témoin (aucun code GAN actif) montre exactement le
même profil.

**Diagnostic de mémorisation par cellule** (`gan_percell_diagnostic.py`,
`percell_diagnostic_palier2.json` — D et G-EMA du checkpoint GAN final,
16 échantillons réels + 16 générés par cellule, 12 cellules = 3 contrastes ×
4 champs cibles depuis 0.1T) :

| contraste | champ cible | n classe cible | d_acc |
|---|---|---|---|
| T1W | 7T | 235 | 0.844 |
| T1W | 5T | 121 | 0.781 |
| T2FLAIR | 7T | 84 ⚠ | 0.781 |
| T1W | 1.5T | 104 | 0.719 |
| T2W | 7T | 81 ⚠ | 0.719 |
| T1W | 3T | 143 | 0.625 |
| T2FLAIR | 5T | 102 | 0.562 |
| T2FLAIR | 1.5T | 221 | 0.500 |
| T2W | 1.5T | 221 | 0.469 |
| T2FLAIR | 3T | 143 | 0.438 |
| T2W | 3T | 142 | 0.375 |
| **T2W** | **5T** | **43 ⚠ (la plus petite)** | **0.344** |

Moyenne des 12 cellules : 0.596 (min 0.344, max 0.844) — cohérent avec le
`d_acc` global de fin d'entraînement (~0.62-0.69).

**Aucune cellule ne montre le signal de mémorisation redouté** (`d_acc→1.0`,
D triomphant sur une classe qu'il aurait appris par cœur) — le maximum observé
(0.844, T1W@7T) reste sous la borne haute du plan (0.90). **La seule anomalie
est dans l'AUTRE sens** : T2W@5T (43 volumes, la plus petite classe du jeu de
données) a `d_acc=0.344`, EN DESSOUS du hasard (0.5), avec `D(réel)=+0.464`
quasi identique à `D(faux)=+0.466` — D ne sépare quasiment pas réel/faux sur
cette cellule. C'est de la SOUS-discrimination par manque d'exemples, pas de
la mémorisation : D n'a pas eu assez de vrais T2W@5T pour apprendre une
frontière de décision utile, ce qui signifie qu'il ne peut PAS pousser G vers
un raccourci class-plausible-mais-décorrélé sur cette cellule précise — plutôt
rassurant pour le risque d'hallucination que le plan redoutait, pas un signal
d'alarme.

**Verdict Palier 2 : POSITIF, tous les critères du plan remplis.** Pas de
crash/NaN, `d_acc` global et par cellule dans les bornes (l'unique anomalie va
dans le sens sous-discrimination, pas mémorisation), `grad_d` sans blow-up
(déjà vérifié en continu dans `train_log_gan_palier2.txt`), loss de flow
statistiquement indiscernable du témoin (p=0.554) voire légèrement meilleure
en moyenne. Le mécanisme ne déstabilise pas un modèle déjà entraîné, même avec
`gan_start=0` (le test le plus dur du protocole).

## Palier 3 (2026-09-22, continuation à budget étendu + inspection)

Confirmation utilisateur obtenue avant lancement. Continuation RÉELLE (pas
`--resume_weights_only` cette fois : reprise COMPLÈTE — G, EMA, optimizer,
scheduler, ET D/disc_optimizer/disc_scheduler) depuis le `model_final.pth`
du Palier 2 (iter 2000), `total_iters` porté de 2000 à 5000 dans
`configs/mmfm/vectorized_trajectory_gan.yaml` — donc 3000 itérations
supplémentaires, budget cumulé Palier 2+3 = 5000 itérations GAN-actives
(20% de la production), `gan_start=0` inchangé. Terminé sans erreur, aucun
NaN/Inf (`train_log_gan_palier3.txt`). Santé sur la fenêtre finale
(step 4700→5000) : `d_acc` 0.67→0.76→0.79→0.73→0.76→0.69→0.75 — légèrement
plus haut que sur le Palier 2 (tendance à surveiller si le budget devait
encore s'étendre) mais toujours sous la borne 0.90 ; `gan_g` continue de
croître (0.47→0.61) sans collapse ; `grad_d` borné (2.03-10.93).

**Inférence + évaluation ciblées** (sujet 0006 — un des 3 seuls sujets
réellement appariés du jeu de données, voir
[[project-dataset-unpaired-n3]] —, T1W, 3 paires couvrant un cas facile
(0.1T→1.5T), un cas dur connu (X→7T) et le cas dur avec la source la plus
dégradée (0.1T→7T) ; géométrie `--center_crop_only`, ~32s/volume — PAS le
protocole officiel 20 paires × 3 sujets en 8win, qui est réservé au
Palier 4) : `infer_mmfm_unified.py` sur le checkpoint GAN final, comparé
aux prédictions de production déjà calculées
(`outputs/mmfm/vectorized_trajectory/predictions_cc_baseline_20260916/`,
MÊME géométrie cc — comparaison valide, voir la note de
`feedback-zclip-metrique-officielle` sur cc vs 8win non comparables entre
eux). Évaluateur OFFICIEL du challenge
(`~/Code/MRIxFields2026/Evaluation/evaluate.py`), sujet 0006 isolé des CSV
témoins (qui portent aussi 0007/0009) pour une comparaison strictement
appariée :

| Paire | Métrique | Témoin (0006) | GAN (0006) | Δ (GAN − témoin) |
|---|---|---|---|---|
| 3T→7T | nRMSE | 0.38322 | 0.37700 | **−0.0062** (mieux) |
| 3T→7T | SSIM | 0.88408 | 0.88572 | **+0.0016** (mieux) |
| 3T→7T | LPIPS | 0.10186 | 0.10016 | **−0.0017** (mieux) |
| 0.1T→7T | nRMSE | 0.36118 | 0.36347 | +0.0023 (légèrement pire) |
| 0.1T→7T | SSIM | 0.88207 | 0.88185 | −0.0002 (≈ égal) |
| 0.1T→7T | LPIPS | 0.13405 | 0.13407 | +0.0000 (≈ égal) |
| 0.1T→1.5T | nRMSE | 0.17423 | 0.17495 | +0.0007 (légèrement pire) |
| 0.1T→1.5T | SSIM | 0.90670 | 0.90612 | −0.0006 (légèrement pire) |
| 0.1T→1.5T | LPIPS | 0.10689 | 0.10669 | −0.0002 (légèrement mieux) |

Cross-vérifié par un second code indépendant
(`eval_quantitative_3arch.py --mode calibration --region slab`, la région
TRACKÉE par le classement officiel) : `nrmse_raw` identique aux 5 décimales
sur les 3 paires (0.38322/0.36118/0.17423 témoin ; 0.377/0.363/0.175 GAN) —
les deux pipelines d'évaluation du projet s'accordent.

**PAS de signature d'hallucination.** Le risque nommé par le plan est
SSIM/LPIPS qui s'améliore PENDANT QUE nRMSE stagne/se dégrade — signature
absente ici : sur 3T→7T les trois métriques s'améliorent ENSEMBLE (cohérent,
pas un découplage netteté/fidélité) ; sur les deux autres paires, les trois
métriques bougent de conserve dans des écarts de 0.0002 à 0.0007 — sous le
niveau de bruit d'une mesure à un seul sujet, pas un signal. Rappel :
n=1 sujet × 3 paires est un SMOKE TEST (« bon marché » comme le demande le
plan), pas une preuve statistique — voir Palier 4 pour ça.

**Inspection qualitative** (`qualitative_palier3_0006_T1W.png`, script
autonome `gan_palier3_qualitative.py` — pas d'ajout au registre `METHODS`
de `eval_qualitative_3arch.py`, cet outil partagé sert la comparaison
DURABLE vectorisé/UNet/INR, pas un diagnostic ponctuel de palier ; source,
vérité terrain, témoin, GAN, cartes d'erreur témoin/GAN, 3 paires × 6
colonnes) : témoin et GAN sont visuellement quasi indiscernables sur les 3
paires — même position/forme des ventricules, même ruban cortical, même
niveau de flou sur les paires dures (0.1T→7T), AUCUNE structure
anatomiquement implausible dans la colonne GAN qui serait absente du
témoin. Les cartes d'erreur ont le MÊME motif spatial (bord cortical plus
lumineux, artefact de registration/interpolation déjà connu, pas propre au
GAN) sur les deux colonnes — pas de zone localisée où le GAN « invente »
une structure fausse avec une confiance trompeuse.

**Verdict Palier 3 : POSITIF, aucun signal d'alarme, ni quantitatif ni
qualitatif.** Le mécanisme reste stable à budget étendu (5000 itérations
cumulées, 20% de la production), sans dégrader la fidélité et sans produire
la signature d'hallucination redoutée. `d_acc` en légère hausse sur la
fenêtre finale — seul point à surveiller si le budget devait encore
s'étendre au-delà de ce palier.

**Prochaine étape (Palier 4, évaluation Task 3 OFFICIELLE — 20 paires × 3
sujets, géométrie 8win, ~104s/volume × 60 prédictions × 2 (GAN + témoin) =
plusieurs heures GPU, comparaison appariée test des signes + Wilcoxon par
contraste contre la référence actuelle 0.2843 nRMSE — EN ATTENTE DE
CONFIRMATION, engagement le plus lourd du protocole)**.

## Fichiers modifiés/créés

- `src/cfm/mmfm_vectorized.py` : + `LatentDiscriminator`.
- `src/cfm/arch_vector.py`, `src/cfm/arch_synthetic.py` : +
  `build_latent_discriminator`, câblé via `ArchAdapter.build_discriminator`.
  `arch_synthetic.py` : + correctif `level=None` (prérequis, indépendant du
  GAN).
- `src/cfm/synthetic_marginals.py` : + `level` constant dans
  `SyntheticLatentDataset.__getitem__` (prérequis, indépendant du GAN).
- `src/cfm/mmfm_core.py` : `ArchAdapter.build_discriminator` (champ optionnel) ;
  clés `train.lambda_gan*`/`gan_*`/`disc_*` ; garde-fous
  (`marginal_mode != "trajectory"`, `accumulation_steps != 1`,
  `adapter.build_discriminator is None`) ; construction/reprise/checkpoint de
  `disc`/`disc_optimizer`/`disc_scheduler` (tous optionnels, `None` par
  défaut — schéma de checkpoint additif, aucune clé existante modifiée) ;
  `did_flip` nommé (branche trajectoire) ; blocs G (après `lambda_lpips`,
  réutilise `euler_integrate` — pas `cycle_rollout_vector`, qui aurait
  doublé le coût avec un aller-retour inutile pour ce besoin) et D (après
  `ema.update`, réutilise `class_loaders` + `did_flip` pour le "réel") ;
  logging étendu (`loss_gan_g`, `loss_gan_d`, `d_acc`, `grad_norm_d`,
  `gan_contrast`, `gan_tgt_field`).
- `src/cfm/test_mmfm_flow_behaviour.py` : +
  `test_gan_inactive_is_noop`, flag `--gan-noop-check`.
- `configs/mmfm/synthetic_gan.yaml`, `configs/mmfm/vectorized_trajectory_gan.yaml`,
  `configs/mmfm/vectorized_trajectory_gan_control.yaml` (témoin Palier 2,
  copie stricte + `lambda_gan=0.0`) : nouvelles configs.
- `gan_percell_diagnostic.py` (dans ce manifeste) : script de mesure
  post-hoc, per-cellule, du `d_acc` sur un checkpoint entraîné — pas ajouté
  au dépôt (diagnostic ponctuel du Palier 2, réutilise `arch_vector`/
  `euler_integrate` sans réimplémentation).
- `gan_palier3_qualitative.py` (dans ce manifeste) : panel qualitatif
  autonome (source/vérité/témoin/GAN/erreurs), réutilise
  `visualization/common.py::load_mid_slice_aligned`/`normalize` — pas
  ajouté au registre `METHODS` de `eval_qualitative_3arch.py` (diagnostic
  ponctuel, pas une architecture à comparer durablement).

## Écart au plan initial

Le plan mentionnait réutiliser `cycle_rollout_vector` pour générer
l'échantillon "faux" du bloc G. En relisant son code en détail au moment de
l'implémenter, `euler_integrate` (déjà dans `mmfm_core.py`, utilisé par
l'inférence) s'est avéré strictement suffisant : c'est un aller SIMPLE,
alors que `cycle_rollout_vector` calcule un aller-retour complet (le retour
n'étant nécessaire que pour une perte de cycle-consistency, pas pour générer
un échantillon à montrer à D) — deux fois moins d'appels réseau pour le même
résultat utile. Décision prise sur place, pas re-soumise au plan : c'est une
optimisation qui ne change ni le design ni le comportement du mécanisme, et
le plan lui-même n'exigeait que « réutiliser la machinerie de rollout
différentiable existante plutôt que d'en écrire une nouvelle », satisfait
par les deux fonctions.
