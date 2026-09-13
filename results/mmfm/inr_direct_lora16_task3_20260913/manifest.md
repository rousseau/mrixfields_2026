# INR modulation directe + LoRA rang 16 — Task 3 : NÉGATIF, et sévèrement

**Date** : 2026-09-11/13
**Verdict : NÉGATIF.** Malgré un mécanisme validé et une capacité de
représentation mesurée EXCELLENTE (voir §1-2), le score de traduction Task 3
est **pire que les trois architectures existantes, et pire que le témoin
identité sur SSIM/LPIPS**. C'est la troisième instance du même phénomène que
« Volet 2 » (2026-09-08) : améliorer la fidélité de représentation ne se
transmet pas au score de traduction, et peut activement le dégrader — ici de
façon beaucoup plus sévère.

---

## 1. Ce qui a été construit (rappel, tout validé en cours de route)

- Correctif du bug d'initialisation LoRA morte dans `INRBackbone.fit_latent`
  (`src/cfm/inr_backbone.py`) — vérifié rétrocompatible, mécanisme confirmé
  vivant par sondage.
- Backbone entraîné nativement en modulation directe (`hyper_hidden_dim=0`) +
  LoRA rang 16 (`configs/mmfm/inr_backbone_direct_lora16.yaml`) : 50000 pas,
  14h46, convergence saine (loss 0.51→0.028).
- Extension `fit_new_volume_adam` (Adam, taux d'apprentissage séparés
  shift/LoRA) pour le fitting de `z` au precompute ET à l'inférence — un bug
  réel trouvé et corrigé en cours de route (`torch.enable_grad()` manquant,
  `RuntimeError` sous le `no_grad()` de l'inférence).
- Precompute complet : 1939 volumes, 0 erreur, 38.3h.
- Flow entraîné : 25000 pas, 53.5 min, loss stable ~0.006.

## 2. La porte de qualité (auto-reconstruction) — EXCELLENTE

`results/mmfm/gate_inr_direct_lora16_20260911/` :

| Fitting | nRMSE_fg moyen (8 volumes) | Gate 3 | Ratio z0/fit |
|---|---|---|---|
| SGD (mécanisme historique) | 0.2387 | 8/8 ✅ | 1.89 |
| **Adam (mécanisme retenu)** | **0.1734** | 8/8 ✅ | **2.5-2.7** |

Confirmé cohérent avec le sondage gelé du 2026-09-07 (~0.14-0.29 fg). **Sur ce
critère seul, ce backbone représente mieux les volumes que le backbone de
production actuel.**

## 3. Task 3 — le score de traduction s'effondre

Pipeline standard (`scripts/run_task3_eval.sh`, géométrie `cc`, comparable à
`vectorized`/`inr_std`). CSV : `task3_inr_direct_lora16_{T1W,T2W,T2FLAIR}.csv`.

| | T1W | T2W | T2FLAIR | moyenne |
|---|---|---|---|---|
| Identité (témoin) | 0.9273 / 0.8699 | 0.3859 / 0.9084 | 0.5574 / 0.8864 | 0.6235 / 0.8883 |
| Vectorisé | 0.4353 / 0.8997 | 0.3376 / 0.8980 | 0.3654 / 0.8949 | 0.3794 / 0.8975 |
| UNet | 0.4617 / 0.8955 | 0.3676 / 0.8963 | 0.3807 / 0.8929 | 0.4033 / 0.8949 |
| INR (production, `inr_std`) | 0.4070 / 0.8606 | 0.3800 / 0.8579 | 0.3376 / 0.8708 | 0.3749 / 0.8631 |
| **INR direct LoRA16 (ce run)** | **0.5190 / 0.6808** | **0.4584 / 0.6733** | **0.3988 / 0.7277** | **0.4587 / 0.6939** |

(format : nRMSE / SSIM)

**LPIPS moyen : 0.3499**, contre 0.1557 (INR production) et 0.0941
(vectorisé) — plus du double de la LPIPS du pire des trois baselines.

**Duels paire à paire (60 duels par comparaison)** :
- INR (production) bat INR direct LoRA16 : **45/60**
- Vectorisé bat INR direct LoRA16 : 43/60
- UNet bat INR direct LoRA16 : 39/60

**Gain relatif sur le témoin identité (nRMSE)** : **NÉGATIF sur T1W (-7.7%),
T2W (-31.6%) et T2FLAIR (-38.7%)** — ce modèle est pire que ne rien faire
(recopier la source) sur 2 des 3 contrastes. Le SSIM (0.6939) est également
**sous le témoin identité (0.8883)** — un signal de dégradation perceptuelle
sévère, pas seulement d'absence de gain.

## 4. Interprétation — pourquoi un gain de représentation aussi net produit un score aussi mauvais

L'écart entre §2 (auto-reconstruction excellente) et §3 (traduction très
dégradée) n'est pas contradictoire : ce sont deux propriétés différentes du
même espace latent. Deux causes plausibles, toutes deux déjà identifiées comme
risques AVANT ce run (voir commentaires de `configs/mmfm/inr_direct_lora16.yaml`) :

1. **Hétérogénéité d'échelle par dimension** : mesurée à ×49.9 (min/max de
   l'écart-type par dimension) contre ×2.9 pour l'ancien cache MedVAE — la
   portion shift (1536 dims) a un écart-type ~3× celui de la portion LoRA
   (45104 dims). Un seul scalaire (`latent_mean`/`latent_scale`) normalise mal
   ce mélange : le flow voit probablement la portion LoRA (là où vit
   l'essentiel du gain de fidélité mesuré en §2) à une échelle très inadaptée
   à son initialisation, et ne peut pas bien la piloter.
2. **Rugosité de l'espace latent** : le projet a déjà mesuré (2026-08-06,
   `diagnose_inr_latent_smoothness.py`) que le latent INR (alors 1536-d) est
   ~2.2-2.9× plus rugueux que MedVAE sur l'axe champ-croisé que le flow doit
   traverser, sans aucun régularisateur de continuité entre volumes. Chaque
   `z` ici est ajusté INDÉPENDAMMENT par Adam pour reconstruire SON PROPRE
   volume du mieux possible (c'est justement ce qui rend §2 excellent) — rien
   ne contraint deux `z` de volumes voisins (même sujet, champs différents) à
   être proches. Avec 30× plus de dimensions qu'avant (46640 contre 1536),
   cette rugosité non contrainte est probablement bien pire, précisément là où
   le flow doit interpoler.

Ces deux causes ne sont PAS testées séparément ici — ce run ne permet pas de
trancher laquelle domine, seulement de confirmer que leur effet combiné est
sévère.

## 5. Ce que ça n'invalide PAS

- Le correctif du bug LoRA (§1) reste correct et rétrocompatible — un vrai
  bug de code, indépendant du verdict Task 3.
- La conclusion du 2026-09-07 (« le handicap de l'INR déployé est un budget,
  pas un défaut de famille ») reste vraie pour l'auto-reconstruction (§2 la
  reconfirme). Ce qui est nouveau ici : un meilleur budget de représentation
  ne suffit PAS à améliorer la traduction si l'espace résultant n'est pas
  exploitable par le flow — la même leçon que « Volet 2 », mesurée maintenant
  sur un axe différent (capacité du latent, pas écrêtage/normalisation) et de
  façon plus sévère.

## 6. Prochaines pistes (non testées, hors périmètre de ce run)

- Normalisation par sous-groupe (shift vs LoRA) au lieu d'un scalaire unique.
- Régularisation de lissage sur `z` pendant le fitting Adam (pénalité de
  continuité entre volumes voisins) — piste déjà tentée à petite échelle le
  2026-08-06 sur l'ancien latent (1536-d), sans succès net, mais jamais
  testée à cette échelle (46640-d) ni avec Adam.
- Rang LoRA intermédiaire (4 ou 8) : moins de capacité, potentiellement moins
  de rugosité ajoutée.

Aucune de ces pistes n'a été engagée — ce résultat NÉGATIF clôt cette
tentative précise (rang 16, normalisation scalaire, pas de régularisation).
