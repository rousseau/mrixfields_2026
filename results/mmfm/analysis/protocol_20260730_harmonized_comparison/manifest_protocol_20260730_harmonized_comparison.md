# Manifest — Comparaison harmonisée vectorisé vs UNet, post-unification (2026-07-30)

## Contexte

Suite à la demande explicite de l'utilisateur ("Il faut harmoniser le code, afin de comparer de
façon équitable" puis "Dans l'idéal, il ne faudrait avoir qu'une version mmfm vectorisée et une
seule version mmfm unet... Utilisant le même code, sauf pour l'architecture"), le code
d'entraînement a été entièrement unifié (`src/cfm/mmfm_core.py` + `arch_vector.py` + `arch_unet.py`
+ `train_mmfm_unified.py`, remplaçant `train_mmfm_3d.py`/`train_mmfm_unet_3d.py`, supprimés). Les
deux architectures partagent désormais : le même dataloader, le même sampler multi-marginal
(conscience de disponibilité par contraste, respecte `num_targets_per_step`), le même couplage
OT-CFM réel (`FM.sample_location_and_conditional_flow`, jamais d'interpolation linéaire à couplage
indépendant), la même loss (L1), les mêmes régularisateurs (cycle/edge-consistency, désactivés ici
pour isoler la comparaison architecture-only), le même schéma de checkpoint, et le même pipeline
d'inférence/évaluation. Seule l'architecture du modèle de flow diffère (MLP vectoriel résiduel vs
UNet 3D spatial MONAI).

**Découverte importante en cours d'unification** : le vectorisé n'utilisait *jamais* de vrai
couplage OT (interpolation linéaire à couplage indépendant malgré un objet
`ExactOptimalTransportConditionalFlowMatcher` construit mais jamais appelé) et un sampler naïf sans
garde de disponibilité — deux divergences non-architecturales significatives par rapport au UNet.
Un smoke test croisé a confirmé que faire fonctionner chaque architecture sous le régime de l'autre
ne cassait rien, avant de basculer le vectorisé vers le régime partagé (sampler
availability-aware + OT-CFM réel) et de supprimer les chemins de code temporaires.

## Runs harmonisés (Phase 5)

Deux configs strictement identiques hors du bloc `model:` (mêmes `data:`/`train:`/`inference:` —
96×112×96 @ 2mm, 50000 itérations, batch_size=1, lr=2e-5, warmup_steps=2000, identity_prob=0.1,
adjacent_only=false, n_steps inférence=20, `norm_mode=field_fixed`) :

- `configs/mmfm3d_vectorized_harmonized.yaml` — **from scratch** (pas de reprise). Nécessaire car le
  passage à un couplage OT réel change la distribution de vitesses cible apprise — pas une simple
  continuation d'entraînement.
- `configs/mmfm3d_unet_harmonized.yaml` — reprise poids-seuls (`--resume_weights_only`) depuis
  `mmfm3d_multimarginal_medvae_run4_fieldnorm_ext/model_final.pth` (30k itérations à 128×128×80@1mm) —
  légitime car l'algorithme UNet n'a pas changé lors de l'unification (sampler et OT-CFM réels déjà
  utilisés avant), le modèle purement convolutionnel s'adapte au changement de forme spatiale.

Caches latents dédiés construits à 96×112×96@2mm avec `field_norm_stats.json` (1939 échantillons
chacun) : `outputs/latent_cache_vec/medvae_finetune_f210b265/` (vectorisé) et
`outputs/latent_cache_unet_spatial/medvae_finetune_f210b265/` (UNet).

### Résultats d'entraînement

| Run | Itérations | Durée | Vitesse | Loss finale |
|---|---|---|---|---|
| Vectorisé harmonisé (from scratch) | 50000/50000 | 115.8 min | ~7.2 it/s | ~11 (stable, pas de divergence) |
| UNet harmonisé (reprise poids-seuls) | 50000/50000 | 234.5 min | ~4.9 it/s | ~1.5 (stable, pas de divergence) |

(Les magnitudes de loss ne sont pas comparables entre architectures — L1 sur un vecteur aplati
16128-d vs un tenseur spatial (1,24,28,24) à échelle/normalisation différente — seule la tendance
et l'absence de divergence sont significatives ici.)

## Éval comparative (Phase 6)

Protocole identique aux Phases 0/1 : 20 paires × 3 sujets (0006, 0007, 0009) × T1W, générées via
`infer_mmfm_unified.py` (`--norm_mode field_fixed --center_crop_only`, single-shot puisque le FOV
d'entraînement 96×112×96@2mm dépasse le FOV natif reséchantillonné), évaluées via
`src/evaluation/evaluate.py --task task3`.

Pred dirs : `outputs/predictions/mmfm_vectorized_harmonized/task3/T1W`,
`outputs/predictions/mmfm_unet_harmonized/task3/T1W` — 60/60 prédictions chacun (gate PASS).

CSV : `task3_vectorized_harmonized_T1W.csv`, `task3_unet_harmonized_T1W.csv`.

### Résultat global (moyenne des 20 paires)

| Modèle | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **Vectorisé harmonisé** | **0.4361** | **0.8740** | 0.1391 |
| UNet harmonisé | 0.4741 | 0.8721 | 0.1392 |

### nRMSE moyen par champ CIBLE

| Cible | Vectorisé | UNet |
|---|---|---|
| →0.1T | 0.2853 | 0.3018 |
| →1.5T | 0.3440 | 0.3625 |
| →3T | 0.3294 | 0.3419 |
| →5T | 0.4997 | 0.5435 |
| →7T | 0.7220 | 0.8209 |

### Détail par paire (nRMSE)

| Paire | Vectorisé | UNet | Gagnant |
|---|---|---|---|
| 0.1T→1.5T | 0.258 | 0.246 | UNet |
| 0.1T→3T | 0.247 | 0.241 | UNet |
| 0.1T→5T | 0.426 | 0.492 | Vectorisé |
| 0.1T→7T | 0.682 | 0.782 | Vectorisé |
| 1.5T→0.1T | 0.244 | 0.230 | UNet |
| 1.5T→3T | 0.253 | 0.255 | Vectorisé |
| 1.5T→5T | 0.617 | 0.693 | Vectorisé |
| 1.5T→7T | 0.861 | 1.035 | Vectorisé |
| 3T→0.1T | 0.213 | 0.197 | UNet |
| 3T→1.5T | 0.220 | 0.227 | Vectorisé |
| 3T→5T | 0.495 | 0.526 | Vectorisé |
| 3T→7T | 0.817 | 0.907 | Vectorisé |
| 5T→0.1T | 0.291 | 0.341 | Vectorisé |
| 5T→1.5T | 0.421 | 0.456 | Vectorisé |
| 5T→3T | 0.373 | 0.393 | Vectorisé |
| 5T→7T | 0.528 | 0.560 | Vectorisé |
| 7T→0.1T | 0.393 | 0.439 | Vectorisé |
| 7T→1.5T | 0.477 | 0.521 | Vectorisé |
| 7T→3T | 0.446 | 0.478 | Vectorisé |
| 7T→5T | 0.460 | 0.463 | Vectorisé |

**Bilan** : vectorisé gagne 16/20 paires, UNet gagne 4/20 (uniquement les transitions vers un champ
*faible* — →0.1T/1.5T/3T depuis une source adjacente). Le vectorisé l'emporte plus nettement sur les
transitions vers champ *fort* (→5T/7T), historiquement les plus difficiles.

Figures qualitatives (4 paires clés × 3 sujets) : `figures/compare_T1W_*.png` — anatomie plausible
des deux côtés, sans artefact ni collapse de mode ; légèrement moins de flou côté vectorisé sur les
cas 0.1T→7T, cohérent avec l'écart quantitatif.

## Comparaison à l'historique pré-unification

| Modèle | nRMSE global (20 paires) |
|---|---|
| Vectorisé régularisé v2 (référence session, pré-unification, confondue par résolution/loss/sampler/OT-CFM) | ≈0.530 |
| UNet run4_fieldnorm (pré-unification, 128×128×80@1mm, 15k iters) | 0.438 |
| **Vectorisé harmonisé (post-unification, from scratch)** | **0.436** |
| **UNet harmonisé (post-unification, reprise)** | **0.474** |

## Conclusion

**Le résultat inverse la conclusion provisoire de la première moitié de cette session.** La
comparaison initiale (non-harmonisée) suggérait que le UNet dépassait nettement le vectorisé
(0.438 vs 0.530). Une fois le code réellement unifié — même dataloader, même sampler, même
couplage OT-CFM, même loss, même pipeline d'inférence, seule l'architecture différant — le
**vectorisé fait en réalité légèrement mieux que le UNet** (0.436 vs 0.474, et gagne 16/20 paires).
L'essentiel de l'avantage apparent du UNet dans la comparaison initiale provenait donc de
confondants non-architecturaux (résolution/FOV d'entraînement, loss MSE vs L1, sampler naïf vs
availability-aware, et surtout l'absence de couplage OT réel côté vectorisé) — pas d'un avantage
architectural intrinsèque. Ceci valide directement la demande de l'utilisateur : sans harmonisation
du code, la comparaison précédente n'était pas fiable.

Bonus méthodologique : le simple passage au sampler availability-aware + OT-CFM réel (aucun
changement d'architecture) a fait passer le vectorisé de nRMSE≈0.530 à 0.436 — la correction de
bug la plus rentable de toute la lignée vectorisée cette session, plus que tout ajustement de
régularisation (cycle/edge-consistency) essayé précédemment.

## Extension nocturne (2026-07-30/31) — plus d'itérations n'aide pas

À la demande de l'utilisateur, les deux modèles harmonisés ont été prolongés (reprise complète —
poids+EMA+optimizer+scheduler, nouveau `total_iters` dimensionné pour occuper ~12h de calcul GPU
partagé chacun) : `configs/mmfm3d_vectorized_harmonized_ext.yaml` (50k→375k itérations, +325k,
~11.6h) et `configs/mmfm3d_unet_harmonized_ext.yaml` (50k→270k itérations, +220k, ~18.9h — le run
UNet a dépassé la fenêtre de 12h prévue et a été laissé terminer sur confirmation explicite).

### Résultat — la prolongation n'améliore ni l'un ni l'autre modèle

| Modèle | 50k / 50k iters | Étendu | Δ |
|---|---|---|---|
| Vectorisé | nRMSE 0.4361 | nRMSE 0.4659 (375k iters) | **+0.030 (pire)** |
| UNet | nRMSE 0.4741 | nRMSE 0.4772 (270k iters) | +0.003 (quasi stable, légèrement pire) |

CSV : `task3_vectorized_harmonized_ext_T1W.csv`, `task3_unet_harmonized_ext_T1W.csv`. Figures :
`figures_ext/compare_T1W_*.png`.

**Interprétation** : les deux modèles avaient déjà convergé/plafonné vers 50k itérations — cohérent
avec le plateau de validation observé plus tôt cette session (≈20-30k itérations) pour la lignée
vectorisée. Le schedule LR des runs étendus maintient un LR plein (2e-5, pas de décroissance) sur
une très longue portion (jusqu'à decay_start=187500 pour le vectorisé, 135000 pour le UNet) avant de
redécroître — cette longue phase à LR élevé après convergence semble avoir légèrement dérivé le
modèle vectorisé (le plus sensible des deux) hors de son bassin optimal, sans que la décroissance
finale suffise à revenir aussi bas qu'avant. Observation qualitative cohérente : le modèle vectorisé
étendu (375k) montre un léger artefact de bandes sur la coupe sagittale zoomée du cas 0.1T→7T
(`figures_ext/compare_T1W_0006_0.1T_to_7T.png`), absent de la version 50k.

**La conclusion architecturale reste robuste** : même après prolongation, le vectorisé bat toujours
le UNet (0.466 vs 0.477, gagne 14/20 paires vs 16/20 à 50k) — l'écart s'est simplement resserré, pas
inversé. **Recommandation** : conserver les checkpoints à 50k itérations
(`mmfm3d_vectorized_harmonized/model_final.pth`, `mmfm3d_unet_harmonized/model_final.pth`) comme
référence de production — ce sont les meilleurs résultats obtenus des deux côtés, et prolonger
l'entraînement au-delà n'apporte aucun gain (au contraire pour le vectorisé).

**Prochaine étape suggérée** : si un travail architectural (FiLM/AdaGN, cf. l'annexe du plan
d'unification) est encore souhaité, il peut maintenant être évalué sur une base de comparaison
réellement équitable — mais le UNet standard n'a plus d'avantage établi à battre ; l'écart à
combler serait plutôt de rattraper le vectorisé harmonisé (0.436).
