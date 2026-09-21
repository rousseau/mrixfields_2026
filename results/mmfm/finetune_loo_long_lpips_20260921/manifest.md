# Fine-tuning LOO T1W, budget étendu + perte de contenu LPIPS(2.5D) — NOUVELLE RÉFÉRENCE

**Date** : 2026-09-21
**Verdict : POSITIF.** La perte LPIPS auxiliaire récupère le gain nRMSE du
budget étendu pour T1W sans le compromis SSIM/LPIPS qui l'accompagnait sous
MSE seul. Nouvelle référence : **nRMSE 0.2843 (`8win`)**, contre 0.2962 pour
la précédente.

## Motivation

Le budget étendu MSE-seul (5000 itérations, entrées du 2026-09-19/20)
améliore le nRMSE de T1W mais dégrade significativement SSIM (-0.0112,
p=0.0023) et LPIPS (+0.0129, p=0.0017) en `8win` — confirmé ne pas être un
curseur dosable entre 3250 et 4750 itérations (entrée du 2026-09-20). La
perte de contenu LPIPS auxiliaire, déjà codée et validée sur T2W (négative
là pour une raison différente — signal individuel insuffisant, pas un
problème de recette), est réessayée ici sur T1W, où l'hypothèse est
différente : elle pourrait retenir la dérive perceptuelle que le MSE seul
laisse filer à budget étendu.

## Méthode

- 3 replis réentraînés à 3750 itérations (point où le compromis MSE-seul
  était déjà pleinement formé sur le repli 0006) avec `lambda_lpips: 1.0`,
  `lpips_every: 40` (~94 déclenchements/repli, coût mesuré ~87 min/repli).
  Configs : `configs/mmfm/vectorized_finetune_loo_excl{0006,0007,0009}_long_lpips.yaml`
  — copie de `..._5k.yaml` (budget étendu MSE) + le mécanisme LPIPS déjà
  utilisé pour T2W.
- Diagnostic latent par (repli, contraste) : T1W s'améliore nettement sur
  les 3 replis (meilleur checkpoint bat la production de référence sur
  chacun). **Risque identifié** : sur le repli 0009, T2FLAIR se dégrade
  fortement aux itérations profondes (nRMSE latent 0.18 → 0.44) — un
  compromis croisé entre contrastes au sein du MÊME modèle fine-tuné.
  Décision : ne réutiliser CE run que pour T1W, garder T2FLAIR et T2W tels
  qu'établis par ailleurs (runs séparés, pas de conflit — chaque repli
  produit un modèle indépendant par run).
- Candidats retenus (proxy latent) : excl0006 iter 3750, excl0007 iter 250,
  excl0009 iter 2250.
- Évaluation directement en `8win` (la géométrie qui avait révélé le
  compromis MSE-seul) — pas de détour par `cc`.

## Résultat — T1W isolé (20 paires, `8win`)

| | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **T1W long+LPIPS (nouveau)** | 0.3152 | 0.8543 | 0.2075 |
| T1W original (best-of-both, 1500 iters) | 0.3508 | 0.8562 | 0.2015 |
| *repère : T1W étendu MSE-seul (compromis)* | *0.3186* | *0.8450* | *0.2144* |
| Δ vs original | **-0.0357** (p=0.041 signes, **p=0.005 Wilcoxon**) | -0.0019 (p=0.041, minuscule) | +0.0060 (NS, p=0.06-0.26) |

Comparé au compromis MSE-seul (ΔSSIM=-0.0112, ΔLPIPS=+0.0129, tous deux
fortement significatifs), la perte LPIPS auxiliaire **réduit la dégradation
SSIM d'un facteur ~6 et LPIPS d'un facteur ~2**, tout en conservant (et même
en améliorant légèrement) le gain nRMSE, qui devient significatif sur les
deux tests statistiques cette fois (contre Wilcoxon seul pour la variante
MSE-seule).

## Résultat — combinaison complète (T1W long+LPIPS + T2FLAIR étendu + T2W production)

| `8win`, 60 cellules | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **nouvelle référence (v2)** | **0.2843** | 0.8418 | 0.1991 |
| référence précédente (v1, T1W original) | 0.2962 | 0.8424 | 0.1971 |
| production | 0.3394 | 0.8313 | 0.2060 |
| Δ (v2 - v1) | **-0.0119** (p=0.041 signes, **p=0.006 Wilcoxon**) | -0.0006 (NS) | +0.0020 (NS) |

**Gain net et significatif en nRMSE, sans coût significatif sur SSIM ni
LPIPS.** Nouvelle référence pour ce fil — remplace v1.

## Conclusion

Contrairement à T2W (où changer la perte n'a rien changé — le problème est
la quantité/nature de l'information disponible, pas la recette), T1W répond
positivement à la perte de contenu : elle permet d'exploiter un budget
d'entraînement plus long SANS payer le prix perceptuel que le MSE seul
imposait. Ceci suggère que le compromis MSE-seul de T1W n'était PAS un
signe de sur-apprentissage sur des données insuffisantes (comme pour T2W),
mais un artefact du choix de perte lui-même — cohérent avec le fait que T1W
a, parmi les 3 contrastes, le plus de vrai déplacement propre au sujet
(0.437-1.440, `diagnose_true_displacement.py`, mémoire projet), donc le plus
à gagner d'un entraînement plus long ET les moyens statistiques d'en
profiter, à condition que la perte ne le pousse pas vers un optimum flatté.

## Réserves

- Le compromis croisé T2FLAIR/0009 (dégradation à itérations profondes du
  MÊME run) mérite d'être compris avant de réutiliser ce mécanisme sur
  T2FLAIR lui-même — non fait ici, T2FLAIR reste sur son propre run MSE-seul
  établi.
- `lambda_lpips=1.0` non calibré finement (poids de départ, repris de
  l'expérience T2W). Un balayage de poids n'a pas été fait — le gain actuel
  pourrait être amélioré ou au contraire s'avérer sensible au réglage exact.
- 3750 itérations choisies par analogie avec le point de compromis MSE-seul
  sur UN repli (0006) — pas nécessairement optimal pour les 2 autres (le
  meilleur T1W du repli 0007 est atteint dès l'itération 250, suggérant que
  ce repli n'avait pas besoin d'un budget aussi long).
