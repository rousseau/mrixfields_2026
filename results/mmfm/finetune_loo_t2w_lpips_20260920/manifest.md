# Fine-tuning LOO T2W avec perte de contenu LPIPS(2.5D) — PISTE FERMÉE, NÉGATIVE

**Date** : 2026-09-20/21
**Verdict : NÉGATIF, sans ambiguïté.** Changer la recette (perte de contenu
décodée en plus du MSE latent) ne corrige pas le problème de T2W — même
verdict que le fine-tuning MSE pur (aucun checkpoint ne bat jamais
production).

## Motivation

Sur les 3 replis, aucun fine-tuning MSE latent (budget 1500 ou 5000
itérations) n'a jamais battu la production sur T2W — dégradation dès la
première sauvegarde, jamais résorbée (entrées du 2026-09-17/19). Piste
explicative non prouvée : la part du vrai déplacement propre au sujet en T2W
est la plus faible des 3 contrastes, donc 30 exemples de supervision latente
n'y ajoutent que du bruit. Hypothèse testée ici : le problème vient peut-être
de la RECETTE (MSE latent seul) plutôt que des données — un fine-tuning
supervisé sur l'image décodée (plus proche du baseline GAN officiel du
challenge) pourrait capter un signal que le MSE latent rate.

## Implémentation

Perte auxiliaire ajoutée à `mmfm_core.py::train` (`train.lambda_lpips`),
compatible avec `marginal_mode: trajectory` (pas de changement de
conditionnement de classe, donc compatible avec la reprise des poids de
production — contrairement à `pairwise_ot`) :

- Une SECONDE position de la trajectoire déjà chargée (`Z`), différente de
  l'ancre, sert de vraie cible — disponible uniquement parce que ce
  fine-tuning tourne sur `pro_train` (sujets appariés), jamais sur
  `retro_train`.
- Intégration RÉELLE du flow (dt ≠ 0, contrairement à la supervision spline
  de ce mode) via `cycle_rollout_vector` (différentiable), vers cette
  seconde position.
- Décodage PAR TUILES (`tiled_decode_grad`, nouvelle fonction différentiable
  ajoutée à `models/tiled_vae.py` — un décodage direct d'un volume
  192×224×192 entier fait OOM, mesuré au premier essai) du latent prédit et
  du vrai latent cible.
- LPIPS(2.5D) entre les deux volumes décodés (`vae3d/medvae_perceptual_loss.py::lpips_2p5d`,
  réutilise le module LPIPS déjà validé pour le fine-tuning MedVAE perceptuel).
- Gaté par `lpips_every`/`lpips_batch_subset` (coût dominé par 2 décodages
  VAE + LPIPS sur toutes les coupes des 3 axes) : mesuré ~35s/pas déclenché
  contre ~0.05s/pas normal.

Bug trouvé et corrigé EN COURS DE ROUTE (avant tout run réel) : le premier
smoke test faisait OOM (`vae.decode()` direct sur volume entier) — corrigé en
utilisant le décodage par tuiles déjà nécessaire à cette résolution partout
ailleurs dans le pipeline.

## Résultat (diagnostic latent, nRMSE contre la VRAIE cible du sujet exclu)

| repli | production (repère) | meilleur checkpoint fine-tuné (T2W+LPIPS) |
|---|---|---|
| 0006 | **0.1991** | 0.3086 (iter 300) — jamais en dessous de la production |
| 0007 | **0.1910** | 0.2371 (iter 300) — jamais en dessous de la production |
| 0009 | **0.1768** | 0.1890 (iter 1350) — jamais en dessous de la production |

Sur les 3 replis, **AUCUN checkpoint ne bat jamais la production sur T2W**.
La dégradation apparaît dès la première sauvegarde (iter 150) et, sur 2 des
3 replis (0006, 0007), s'aggrave avec l'entraînement avant de se stabiliser
à un niveau toujours pire que la production (0006 : 0.32 → pic 0.41 vers
iter 750 → stabilise à 0.35 ; 0007 : motif similaire). Seul le repli 0009
reste dans une plage proche de la production (0.18-0.21) sans jamais la
dépasser.

**Pas d'évaluation Task 3 payée** : l'écart au repère latent est large et
sans ambiguïté sur les 3 replis — mesurer la porte avant de payer
l'inférence complète (discipline déjà établie dans ce projet).

## Conclusion

Changer la RECETE (perte de contenu décodée au lieu du MSE latent seul) ne
corrige pas le problème de T2W. Ceci **renforce** l'hypothèse déjà avancée
(le vrai déplacement propre au sujet en T2W est trop faible pour qu'un
fine-tuning sur 30 exemples y ajoute quoi que ce soit d'utile, quelle que
soit la perte utilisée) plutôt que de l'infirmer : ce n'est pas un problème
de signal d'apprentissage (MSE latent vs perceptuel), c'est un problème de
**quantité et de nature de l'information disponible** pour ce contraste
spécifique. La production reste, et doit rester, le choix pour T2W dans
toute combinaison future.

## Réserves

- `lambda_lpips=1.0` non calibré contre le MSE latent (échelles différentes) —
  mais l'écart mesuré est assez large pour qu'un simple recalibrage du poids
  ne semble pas être la explication : même à poids réduit, il est peu
  probable que le signal MSE latent (déjà dominant, poids implicite 1.0 dans
  `loss = loss_fn(...) + cur_lambda_lpips * loss_lpips`) change de
  comportement qualitatif sur ce contraste précis.
- `lpips_every=25` (pas 10, réduit pour le coût) — un signal plus fréquent
  n'aurait probablement pas inversé une tendance aussi nette et déjà présente
  dès iter 150.
- Le mécanisme lui-même (perte auxiliaire LPIPS en `marginal_mode:
  trajectory`, décodage tuilé différentiable) est validé et RÉUTILISABLE
  pour d'autres expériences futures — c'est la piste "corriger T2W par une
  meilleure recette" qui est fermée, pas le mécanisme.
