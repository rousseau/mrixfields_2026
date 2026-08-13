# Evaluation Script — MRIxFields2026

## Résumé

`src/evaluation/evaluate.py` est un **wrapper d'orchestration**, pas un ré-implémenteur
de métriques : pour un `--method` et un `--task` donnés, il découvre les prédictions
déjà générées, les apparie avec le ground truth, et appelle **en sous-processus**
l'évaluateur officiel du challenge (`~/Code/MRIxFields2026/Evaluation/evaluate.py`)
une fois par paire source→cible, puis agrège les résultats.

✅ Métriques disponibles (calculées par le script officiel appelé en sous-processus) :
- `nrmse`, `ssim`, `lpips` — voxel-level, disponibles pour task1/task2/task3.

❌ Non disponibles pour l'instant :
- `dice`, `volume` — le wrapper ne transmet pas encore `--pred_seg_dir`/`--target_seg_dir`
  au script officiel. Task 1 et Task 2 sont donc évalués ici sur 3 métriques
  seulement, alors que le classement officiel de ces deux tâches se fait sur 5.
  Pour un score Dice/Volume, utiliser directement le script officiel avec des
  segmentations SynthSeg déjà produites (`Evaluation/segment.py`).

✅ Méthodes supportées (auto-découverte de `--pred-dir` si non fourni) :
- `stargan2d`, `cfm`, `mmfm` (v1), `mmfm_v2`, `mmfm_unet`

## Portée

- **Sujets** : uniquement `0006`, `0007`, `0009` (`Training_prospective`) — les seuls
  sujets disposant d'un ground truth complet aux 5 champs. Ce n'est **pas** le jeu
  d'évaluation officiel Synapse (`Validating_prospective`), qui n'a pas de GT
  disponible localement. Ce script sert à un contrôle de développement local,
  pas à prédire le score leaderboard.
- **Tâches** : `task1` (Any→7T), `task2` (0.1T→Higher), `task3` (Any→Any, 20 paires)
  — pairs identiques à celles du challenge (`Submission/README.md`).

## ⚠️ Rééchantillonnage — les métriques locales ne sont pas le score officiel

Nos modèles s'entraînent/infèrent à une résolution réduite (ex. `volume_size:
[128,128,80]`), alors que la grille native du challenge est `364x436x364` à 0.5mm.
`prepare_pair_dir` rééchantillonne donc les prédictions (interpolation cubique)
vers la grille du GT avant de les passer au script officiel. Le pipeline
officiel de soumission (`Submission/build_submission/build_submission.py`), lui,
ne rééchantillonne jamais — il attend des prédictions déjà en résolution native.

Le script affiche un avertissement dès qu'un rééchantillonnage a lieu. **Les
métriques produites ici sont une estimation approximative pour itérer
rapidement en local, pas un prédicteur fiable du score sur le leaderboard.**

## Usage

```bash
# Task 2 (0.1T → 1.5T/3T/5T/7T) — auto-découverte de --pred-dir
python src/evaluation/evaluate.py --method mmfm_unet --task task2

# Task 1 (Any → 7T) — avec dossier de prédiction spécifique
python src/evaluation/evaluate.py --method mmfm_unet --task task1 --pred-dir results/task1/...

# Task 3, méthode mmfm v2
python src/evaluation/evaluate.py --method mmfm_v2 --task task3
```

### Options

```
--method       stargan2d | cfm | mmfm | mmfm_v2 | mmfm_unet
--task         task1 | task2 | task3
--modality     T1W | T2W | T2FLAIR (défaut: T1W)
--pred-dir     Dossier de prédictions (sinon auto-découverte via discover_pred_dir)
--target-dir   Dossier GT (défaut: {data_root}/Training_prospective/{modality})
--data-root    Racine du dataset (défaut: $MRIXFIELDS_DATA ou /home/rousseau/Data/MRIxFields_20260414)
--metrics      Sous-ensemble de nrmse,ssim,lpips (dice/volume non câblés, voir ci-dessus)
--device       cuda | cpu
--output-csv   Défaut: results/{task}_{method}_{modality}.csv
```

## Format de sortie

CSV (`results/{task}_{method}_{modality}.csv`), une ligne par paire source→cible :

| method | task | pair | n_subjects | nrmse_mean | nrmse_std | ssim_mean | ssim_std | lpips_mean | lpips_std |
|--------|------|------|-----------|------------|-----------|-----------|----------|------------|-----------|
| mmfm_v2 | task3 | 0.1T_to_7T | 3 | 0.123 | 0.012 | 0.945 | 0.003 | 0.087 | 0.004 |

Le script exige une cohorte **complète** (les 3 sujets × toutes les paires de la
tâche) avant de lancer l'évaluation ; sinon il affiche la matrice de complétude
et s'arrête sans rien calculer.

## Gestion des doublons de nommage

Un même dossier de prédictions peut contenir, pour un même (source, sujet, cible),
un fichier au nommage officiel (`P_{MOD}_{TGT}_{ID}.nii.gz`) et un fichier au
nommage legacy (`P_{MOD}_{SRC}_{ID}_{MOD}_{TGT}_{méthode}.nii.gz`) — typiquement
après plusieurs runs d'inférence successifs dans le même dossier de sortie.
`build_prediction_matrix` préfère systématiquement le nommage officiel ; en cas
d'ambiguïté (deux fichiers de même statut), il garde le premier par ordre
alphabétique et affiche un avertissement explicite plutôt que de choisir
silencieusement selon l'ordre du système de fichiers.

## Scripts associés (hors périmètre de ce fichier)

- `src/submission/build_task3_submission.py` — construit l'arborescence de
  soumission Task 3 (clip axial `[150:180]`, renommage officiel, zip). Aucun
  script équivalent n'existe encore ici pour Task 1/Task 2 (qui nécessitent en
  plus les segmentations `seg/`) ; utiliser directement les outils officiels
  `Baseline/scripts/segment_predictions.py` + `Submission/build_submission/build_submission.py`
  pour ces deux tâches.

## Références

- Script officiel appelé en sous-processus : `~/Code/MRIxFields2026/Evaluation/evaluate.py`
- Scoreur Synapse réel (référence de vérité pour le format de soumission) :
  `~/Code/MRIxFields2026/Submission/evaluation-2026/score.py`
- Dataset : `~/Data/MRIxFields_20260414/`
- SynthSeg (requis pour Dice/Volume, non installé sur cette machine) :
  https://github.com/BBillot/SynthSeg
