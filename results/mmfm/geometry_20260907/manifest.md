# Les deux géométries d'inférence — combien elles séparent, et qui est concerné

**Date** : 2026-09-07
**Question** : `scripts/run_task3_eval.sh` n'a jamais passé `--center_crop_only`. De
combien les 8 fenêtres décalées fusionnées par Hann déplacent-elles le score, et le
classement des variantes tient-il ?
**Une seule variable** : même checkpoint (`outputs/mmfm/vec_rbest/weights/model_final.pth`),
même config (`configs/mmfm/vectorized_rbest.yaml`), même table de normalisation
(`configs/mmfm/field_norm_stats.json`), mêmes 180 volumes. Seul le drapeau change.
**CSV** : `task3_rbest_{8win,cc}_{full,slab}_{T1W,T2W,T2FLAIR}.csv`

---

## Le défaut

`--center_crop_only` était déclaré `action="store_true"` **sans `default=None`**
(`src/cfm/infer_mmfm_unified.py:767`) et n'était jamais lu par `_flag()` : **aucune clé
de config ne pouvait l'activer**, il fallait le taper sur la ligne de commande. Le
pilote canonique, créé le 2026-08-27, ne le tapait pas.

Les deux branches de `process_volume_unified` :

- **crop centré** (`:337-352`) : `resample_to_output` → 183×219×183, puis
  `center_crop_or_pad_np` → 192×224×192, **une seule passe** VAE → flow → VAE.
- **par défaut** (`:355-372`) : pad reflect 16 → 215×251×215, puis 8 fenêtres aux
  origines −16 et +7 en coordonnées natives, fusionnées par produit de Hann.
  **Aucune des 8 n'est le crop sur lequel le cache a été encodé**
  (`precompute_*_latents.py:154`, crop toujours centré).

## La signature est binaire, et elle a été vérifiée deux fois

Écart médian entre fichiers de prédiction consécutifs, robuste aux reprises :

| famille | s/volume | runs |
|---|---|---|
| crop centré | 9.2 – 34.1 | `inr_std`, `inr`, `inr_compat`, `vectorized`, `vectorized_adjacent`, `vectorized_flip`, `unet`, `unet_adagn`, `order3_*` |
| **8 fenêtres** | **51.6 – 241.9** | **`vec_rbest`, `vec_r1_time`, `vec_batch8`, `vec_lpips`, `valid_rbest`, `ceiling_vectorized`, `ceiling_lpips`, `ceiling_inr`, `calib_rbest`, `calib_inr`, `calib_unet`, `vec_rbest_n{50,100}`, `vec_rbest_ns{50,100}`** |

**15 runs sur 30**, sans zone grise : 34.1 s/vol d'un côté, 51.6 de l'autre. La coupure
est chronologique et tombe exactement à la création du pilote.

*(Cinq répertoires à ~1–2 s/volume — `inr_recal`, `unet_recal`, `vec_rbest_recal`,
`inr_recal_guard`, `_identity_baseline` — ne font pas d'inférence : ils appliquent un
scalaire à des prédictions existantes. Ils **héritent** de la géométrie de leur source ;
`vec_rbest_recal` est donc en 8 fenêtres.)*

**Confirmation par construction.** La ré-inférence de ce manifeste, avec le drapeau et
rien d'autre de changé, tourne à **16.0 s/volume** — contre 104.1 pour le run R-best
original, soit le rapport 6.5× attendu de 8 passes. 180 volumes en 0.80 h. La signature
temporelle n'était donc pas une coïncidence de charge machine.

## Deux variables, pas une — le piège rencontré en préparant la mesure

Les CSV publiés de R-best (`results/mmfm/staircase_20260827/`) **n'ont pas de colonne
`region`** : ils précèdent l'option, donc ils portent sur le volume entier. Or
`src/evaluation/evaluate.py:472` note la tranche `[150, 180)` par défaut depuis le
2026-09-04. Comparer directement une ré-inférence d'aujourd'hui au 0.3737 publié aurait
superposé la géométrie et la région.

D'où les **quatre** combinaisons de cette page : 2 géométries × 2 régions, avec un
contrôle qui rejoue les prédictions stockées sous l'évaluateur d'aujourd'hui.

---

## Le contrôle passe exactement

Avant de comparer quoi que ce soit : les prédictions **stockées** de R-best rejouées
sous l'évaluateur d'aujourd'hui, région `full`, doivent redonner le chiffre publié le
2026-08-27.

| | nRMSE | SSIM |
|---|---|---|
| publié le 2026-08-27 (T1W) | 0.4441 | 0.9042 |
| rejoué le 2026-09-07 (T1W, `8win`/`full`) | **0.4441** | **0.9042** |

Identique à la quatrième décimale. L'évaluateur n'a pas changé de comportement en
région `full`, les prédictions stockées sont intactes, et **la géométrie est bien la
seule variable qui reste**.

## Résultats

### Région `full` — celle des chiffres publiés : INDISCERNABLE

| | 8 fenêtres | crop centré | écart |
|---|---|---|---|
| nRMSE T1W | 0.4441 | 0.4439 | −0.0002 |
| nRMSE T2W | 0.3181 | 0.3189 | +0.0008 |
| nRMSE T2FLAIR | 0.3589 | 0.3599 | +0.0010 |
| **nRMSE moyen** | **0.3737** | **0.3742** | **+0.0005** |
| SSIM moyen | 0.9047 | 0.8987 | −0.0060 |
| LPIPS moyen | 0.0990 | 0.0934 | −0.0056 |

**26/60 paires, test des signes p = 0.37.** L'écart de 0.0005 est quatre fois sous le
plancher de bruit de run du projet (0.002).

**Le tableau de référence historique n'est donc PAS corrompu**, et l'avance de R-best
sur la production (0.0057) tient : la géométrie n'en explique que 9 %. L'alarme posée
au CHANGELOG le 2026-09-07 (soir) est **levée sur le nRMSE**. Elle reste fondée sur le
SSIM (−0.0060) et le LPIPS (−0.0056), dont l'ordre de grandeur dépasse plusieurs effets
arbitrés par le projet.

### Région `slab` — celle que le classement note : les 8 fenêtres GAGNENT

| | 8 fenêtres | crop centré | écart |
|---|---|---|---|
| nRMSE T1W | 0.4096 | 0.4134 | +0.0038 |
| nRMSE T2W | 0.2988 | 0.3035 | +0.0047 |
| nRMSE T2FLAIR | 0.3464 | 0.3482 | +0.0018 |
| **nRMSE moyen** | **0.3516** | **0.3550** | **+0.0034** |
| SSIM moyen | 0.8499 | 0.8394 | −0.0105 |
| LPIPS moyen | 0.1880 | 0.1777 | −0.0102 |

**12/60 paires seulement pour le crop centré, test des signes p = 3.2e−06.** L'effet est
consistant, pas moyen : les 8 fenêtres gagnent sur 48 paires sur 60.

---

## Ce que cela change, et qui n'était pas la question posée

La question était « de combien les 8 fenêtres faussent-elles le tableau ». La réponse
est **presque pas** sur le nRMSE en région `full`. Mais la mesure en a répondu une autre :

> **Moyenner 8 générations décalées vaut 0.0034 de nRMSE sur la région notée**, avec
> p = 3.2e−06. C'est plus que le plancher de bruit (0.002) et plus que la plupart des
> effets sur lesquels le projet a tranché ce dernier mois.

| repère | effet |
|---|---|
| flip | 0.0001 |
| AdaGN | 0.0006 |
| *plancher de bruit de run* | *0.0020* |
| interpolation ordre 3 | 0.0031 |
| les 4 correctifs du flow | 0.0031 |
| **les 8 fenêtres, région notée** | **0.0034** |
| `l1` contre `mse` | 0.0064 |
| `hi` ×1.25 + LPIPS (représentation, plafond) | 0.0239 |

Le mécanisme est explicable et n'a rien de suspect : c'est une **réduction de variance
par ensemble**. Huit générations issues de huit décalages différents sont moyennées ; la
part d'erreur non corrélée entre elles diminue. Que les 8 fenêtres soient hors de la
distribution du cache (encodé sur un crop centré) ne l'empêche pas — l'ensemble paie
plus que le désalignement ne coûte.

**Décision** : le défaut du pilote reste `8win`, mais il est désormais **explicite,
paramétré et imprimé** (`scripts/run_task3_eval.sh`, 6ᵉ argument). Ce n'était pas un
choix, c'en est un. Coût : 6.5× le temps machine (104 contre 16 s/volume). `cc` reste
disponible pour comparer aux chiffres antérieurs au 2026-08-27, et pour toute mesure où
la vitesse compte plus que le dernier millième.

## Réserves

- **Un seul checkpoint** (R-best). Rien ne garantit que les 0.0034 se retrouvent sur une
  autre variante ; le mécanisme (réduction de variance) est générique, mais son ampleur
  dépend de la corrélation des erreurs entre fenêtres, qui peut varier d'un modèle à
  l'autre.
- **Le nombre de fenêtres n'est pas optimisé.** 8 vient de la grille par défaut de
  `_grid_positions`, pas d'un balayage. Rien ne dit que 8 soit le bon nombre, ni que les
  décalages (−16 et +7) soient les bons.
- Le SSIM et le LPIPS vont en sens **opposés** au nRMSE dans les deux régions : les 8
  fenêtres gagnent en nRMSE et en SSIM, perdent en LPIPS. Un lissage par moyennage, donc,
  et pas un gain de netteté — ce qui est cohérent avec un ensemble.
