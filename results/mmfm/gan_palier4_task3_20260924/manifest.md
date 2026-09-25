# Palier 4 — perte adversariale latente, Task 3 officielle (2026-09-23 → 25)

Suite de `results/mmfm/gan_latent_paliers01_20260922/manifest.md` (Paliers 0-3).

## Verdict

**POSITIF MAIS MINUSCULE.** Contre un témoin au MÊME budget (5000 itérations,
`lambda_gan=0`, mêmes reprises), le GAN améliore les trois métriques
**ensemble** et de façon significative sur les 60 cellules : nRMSE −0.0029
(Wilcoxon p=0.005), SSIM +0.0009 (p=0.0006), LPIPS −0.0008 (p=0.002). Pas de
signature d'hallucination (le perceptuel ne progresse pas au détriment du
nRMSE). Le gain est porté par **T1W** (−0.0060 nRMSE, 15/20 paires) ; T2W et
T2FLAIR ne bougent pas de façon détectable en nRMSE.

**Ordre de grandeur :** 0.0029, c'est à peine au-dessus du plancher de bruit du
protocole (~0.002, `scripts/run_task3_eval.sh`), et **19 fois plus petit** que
ce qu'apporte le fine-tuning LOO (0.3394 → 0.2843). Par rapport à la production
(25k itérations, sans GAN), l'écart n'est plus que de −0.0015 (Wilcoxon
p=0.048, signes p=0.25) : **la moitié de l'avantage sur le témoin vient de ce que
le témoin se DÉGRADE** en prolongeant l'entraînement (0.3409 contre 0.3394 pour la
production), et que le GAN évite cette dégradation.

La référence 0.2843 n'est pas battue (+0.0537) — comparaison attendue et NON
équitable : la référence contient le fine-tuning LOO sur pro_train, le GAN non.

## Protocole

- **GAN** : `outputs/mmfm/vectorized_trajectory_gan/weights/model_final.pth`.
  Production → 2000 itérations en reprise des poids seuls (Palier 2), puis
  continuation complète jusqu'à 5000 (Palier 3). `lambda_gan=0.05`, `gan_start=0`.
- **Témoin** : `outputs/mmfm/vectorized_trajectory_gan_control/weights/model_final.pth`.
  Même chemin, `lambda_gan=0`. La continuation 2000 → 5000 a été AJOUTÉE pour ce
  palier (`configs/mmfm/vectorized_trajectory_gan_control.yaml`, `total_iters` passé
  à 5000, `--resume` complet). Le log ne contient ni NaN ni erreur, 3.4 it/s.
  Sans ce témoin, un écart GAN/production aurait confondu l'effet du GAN avec
  celui de 5000 itérations de plus.
- Inférence : `infer_mmfm_unified.py`, géométrie 8win (sans `--center_crop_only`),
  `--split Training_prospective`, `--field_norm_stats configs/mmfm/field_norm_stats.json`,
  EMA. Au total 180 volumes par modèle (20 paires × 3 sujets × 3 contrastes).
  L'inférence du GAN, lancée le 2026-09-23, s'était interrompue sans erreur
  après 156 volumes (l'arrêt est tombé dans T2FLAIR 5T→0.1T). Elle a été complétée
  le 2026-09-24 avec `--skip_existing`.
- Évaluation : `src/evaluation/evaluate.py`, région `slab` par défaut (celle du
  classement). Test apparié par paire : test des signes + Wilcoxon. Voir
  `compare_palier4.py`.
- Scripts : `palier4_chain.sh` (inférence + témoin), `eval_palier4.sh`,
  `compare_palier4.py`. La sortie complète est dans `compare_palier4.txt`.

## Chiffres (8win, slab, 60 cellules)

| | nRMSE | SSIM | LPIPS | T1W | T2W | T2FLAIR |
|---|---|---|---|---|---|---|
| **GAN (5000)** | **0.3380** | 0.8313 | 0.2055 | 0.4250 | 0.2715 | 0.3174 |
| témoin (5000) | 0.3409 | 0.8304 | 0.2063 | 0.4311 | 0.2725 | 0.3190 |
| production (25k) | 0.3394 | 0.8313 | 0.2060 | 0.4285 | 0.2714 | 0.3184 |
| référence LOO | 0.2843 | 0.8418 | 0.1991 | 0.3152 | 0.2714 | 0.2663 |

GAN − témoin (test principal) :

| | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| T1W | **−0.0060** (15/20, p_s=0.041, p_W=0.004) | **+0.0017** (16/20, p_W=0.002) | −0.0017 (NS) |
| T2W | −0.0011 (NS) | +0.0002 (NS) | −0.0005 (p_W=0.03) |
| T2FLAIR | −0.0016 (NS) | +0.0008 (NS) | −0.0004 (NS) |
| **Tous** | **−0.0029** (38/60, p_s=0.052, p_W=0.005) | **+0.0009** (43/60, p_W=0.0006) | **−0.0008** (41/60, p_W=0.002) |

## Réserves

- Un seul tirage par run (une graine) : la variance entre graines n'est pas
  mesurée. Pour un effet de 0.003, c'est la réserve principale.
- Le GAN n'a pas été combiné au fine-tuning LOO. Rien ne dit que son gain
  (surtout T1W) s'ajoute à celui de la recette 0.2843, qui porte déjà T1W de
  0.4285 à 0.3152.
- n=3 sujets appariés (plafond des données, voir la mémoire projet) : les 20
  paires d'un contraste partagent les mêmes 3 sujets, elles ne sont pas indépendantes.
