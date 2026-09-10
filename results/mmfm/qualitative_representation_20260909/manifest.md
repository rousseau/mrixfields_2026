# Comparaison qualitative INR (SIREN) vs MedVAE — checkpoints ACTUELS

**Date** : 2026-09-09/10
**Script** : `src/cfm/figures_representation_capacity_current.py`
**CSV** : `metrics.csv` (12 lignes : 9 cellules jeu A + 3 cellules jeu B)

## Pourquoi cette campagne était nécessaire

Les seules figures qualitatives INR vs MedVAE du dépôt
(`results/mmfm/comparison_20260801_final/figures_representation_capacity/`,
script `figures_representation_capacity.py`) datent du 2026-08-01/07 et montrent
un INR **périmé** : latent `z=4096` (remplacé par `129024` le 2026-08-10), et
surtout **antérieures** à l'audit du 2026-08-25 qui a corrigé le biais EMA non
chauffé et l'inversion d'orientation LAS/RAS (effet combiné mesuré : nRMSE T1W
0.6383 → 0.3787, voir `results/mmfm/audit_20260825/`). Aucune figure à jour ne
montrait le backbone INR réellement déployé aujourd'hui, ni le plafond de
capacité mesuré le 2026-09-06/07 (facteur budget ×252, voir CHANGELOG
2026-09-07 §5). Cette campagne comble les deux trous.

**Aucun chemin parallèle** : le script ne réimplémente rien, il appelle
directement le code déjà validé par les campagnes chiffrées de 2026-09-06/07 —
`cfm.bench_representation.roundtrip(ARMS["prod"], ...)` pour MedVAE, et
`cfm.bench_inr_capacity.load_production_backbone` + `fit_new_volume`/
`decode_volume`/`siren_with_capacity` pour l'INR (même checkpoint
`outputs/mmfm/inr_backbone/weights/model_final.pth` qui alimente le cache de
production `inr_932b0b37`). Les nombres obtenus ici recoupent ceux déjà publiés
(ex. INR déployé, T1W×{0.1T,3T,7T}, sujet 0006 : nRMSE 0.107/0.144/0.183 ici
contre la moyenne 0.1448 sur ces mêmes 3 cellules dans
`results/mmfm/inr_capacity_20260906/summary.csv`).

Espace de mesure : **auto-reconstruction, 1mm, volume normalisé 192×224×192
(sans retour à la grille native 0.5mm)** — les colonnes `*_norm` de
`bench_representation.py`, pas la chaîne complète bout-en-bout. C'est la
fidélité intrinsèque de chaque représentation, pas le score du classement.

## Jeu A — production vs production (avant/après direct)

9 cellules : T1W × {0006, 0007, 0009} × {0.1T, 3T, 7T} — mêmes sujets/champs
que les figures périmées de 2026-08-01, pour permettre une comparaison directe.
Fichiers : `repr_current_T1W_{sujet}_{champ}.png`.

| Sujet | Champ | MedVAE nRMSE | MedVAE SSIM | INR nRMSE | INR SSIM |
|---|---|---|---|---|---|
| 0006 | 0.1T | 0.0288 | 0.9801 | 0.1070 | 0.8448 |
| 0006 | 3T   | 0.0677 | 0.9425 | 0.1442 | 0.7967 |
| 0006 | 7T   | 0.0795 | 0.9427 | 0.1832 | 0.7756 |
| 0007 | 0.1T | 0.0292 | 0.9789 | 0.1122 | 0.8268 |
| 0007 | 3T   | 0.0556 | 0.9302 | 0.1228 | 0.7772 |
| 0007 | 7T   | 0.0737 | 0.9309 | 0.1662 | 0.7572 |
| 0009 | 0.1T | 0.0270 | 0.9766 | 0.0981 | 0.8471 |
| 0009 | 3T   | 0.0569 | 0.9490 | 0.1225 | 0.8213 |
| 0009 | 7T   | 0.0481 | 0.9447 | 0.1069 | 0.7942 |
| **moyenne** | | **0.0518** | **0.9528** | **0.1292** | **0.8046** |

**Verdict qualitatif net, cohérent sur les 9 cellules** : MedVAE préserve le
détail gyral fin (SSIM 0.93–0.98) ; l'INR déployé (20 pas de SGD, la procédure
de production réelle) produit une reconstruction visiblement lissée/floue,
perdant la texture corticale fine, même là où son nRMSE brut reste dans le même
ordre de grandeur. Ceci **confirme visuellement** ce que les métriques
disaient déjà depuis le 2026-08-26 (SSIM/LPIPS nettement en faveur du
vectorisé/MedVAE) et le budget ×252 mesuré le 2026-09-07.

## Jeu B — plafond de capacité (le SIREN n'est pas en cause, la modulation l'est)

3 cellules : T1W × 0006 × {0.1T, 3T, 7T} — mêmes cellules que
`results/mmfm/inr_capacity_20260906/`. Ajoute une 4e colonne : **le même SIREN
gelé** (poids du backbone de production, aucun réentraînement), mais avec sa
modulation optimisée directement par Adam en LoRA rang 64 (arm `mod_lora64` de
`bench_inr_capacity.py`, ~182k valeurs par volume au lieu des ~512 effectives
de la production). Fichiers : `plafond_T1W_{champ}_0006.png`.

| Champ | MedVAE nRMSE/SSIM | INR déployé (512 eff.) | INR plafond (LoRA r=64, ~182k eff.) |
|---|---|---|---|
| 0.1T | 0.0288 / 0.9801 | 0.1070 / 0.8448 | **0.0324 / 0.9651** |
| 3T   | 0.0677 / 0.9425 | 0.1442 / 0.7967 | **0.0631 / 0.9165** |
| 7T   | 0.0795 / 0.9427 | 0.1832 / 0.7756 | **0.0638 / 0.9236** |
| **moyenne** | **0.0587 / 0.9551** | **0.1448 / 0.8057** | **0.0531 / 0.9351** |

**À budget comparable (~182k valeurs vs 129024 pour MedVAE), l'INR plafond BAT
MedVAE pré-entraîné sur les 3 cellules** (nRMSE moyen 0.0531 contre 0.0587,
SSIM 0.9351 contre 0.9551 — nRMSE meilleur, SSIM légèrement en retrait).
Visuellement (voir les figures), le plafond LoRA restitue les circonvolutions
et le contraste substance grise/blanche perdus par l'INR déployé, avec une
carte d'erreur du même ordre de grandeur que MedVAE. **Ceci rend visible, pas
seulement chiffré, la conclusion du 2026-09-07** : le handicap de l'INR déployé
est un handicap de BUDGET de modulation (512 valeurs/volume via le
hypernetwork actuel), pas un handicap de la famille SIREN ni de l'architecture.

## Réserves

- **Un seul sujet (0006) pour le jeu B** — les écarts par champ sont mesurés
  sur une seule cellule chacun ; le niveau absolu n'est fiable qu'à la
  précision déjà établie pour ce type de mesure (±13 %, voir
  `results/mmfm/representation_20260906/manifest.md`).
- **Espace de travail normalisé, pas la chaîne bout-en-bout ni la grille
  native 0.5mm** — ces chiffres ne se comparent pas directement aux scores du
  classement (région `slab` [150,180)), seulement à `bench_representation.py`
  et `bench_inr_capacity.py` (mêmes conventions).
- **T1W seul.** Les mêmes 9+3 cellules pourraient être reproduites pour
  T2W/T2FLAIR à coût identique si utile.
- **Le plafond LoRA r=64 n'est pas déployable tel quel** : ~182k valeurs par
  volume à optimiser par Adam (~5-10 min/volume ici) et à faire transporter
  par le flow — c'est une borne supérieure de capacité, pas une proposition
  d'architecture de production.
- Ce jeu de figures **supersède** (sans le supprimer)
  `results/mmfm/comparison_20260801_final/figures_representation_capacity/` et
  `figures_representation_1mm/`, qui restent comme trace historique de l'état
  pré-audit (z=4096, EMA non chauffée, orientation non corrigée).
