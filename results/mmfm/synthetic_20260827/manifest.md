# Harnais synthétique — le flow est-il correctement estimé ?

**Date** : 2026-08-27
**Demande** : « être sûr de la capacité de représentation de chaque algorithme et
que le flow est correctement estimé. »

**Verdict** : le défaut d'échelle du temps trouvé le matin est **réel mais
minoritaire**. Deux facteurs le dominent, dont un jamais identifié jusqu'ici, et
un troisième que j'avais rapporté comme réfuté à tort.

---

## L'instrument

Rien ne vérifiait le flow contre une vérité terrain — c'est pourquoi le modèle a
pu rester aveugle au temps pendant des mois sans qu'aucun test ne bronche, et
pourquoi trois hypothèses ont été réfutées d'affilée en visant en aval du défaut.

`src/cfm/synthetic_marginals.py` définit un problème à 5 marginales dont la carte
exacte est calculable, avec nos trois contraintes réelles :

- marginales **non colinéaires** — `P_c(t) = A + tB + sin(πt)C`, avec `C ⊥ B` :
  la déviation à la corde vaut exactement `‖C‖/‖B‖`, donc la courbure est
  **posée**, pas subie. Réglée à 0.6, au milieu des 0.37–0.90 mesurés sur nos
  marginales réelles ;
- données **non appariées** — chaque échantillon n'existe qu'à une marginale,
  comme nos 1056 sujets ;
- un facteur « sujet » à **préserver** pendant le transport, l'analogue de
  l'anatomie.

`src/cfm/arch_synthetic.py` le branche sur `mmfm_core.train()` **sans modifier une
seule ligne du code partagé** : le harnais exerce `_sample_step_plan`,
`_compute_flow`, le sampler OT-CFM, la loss, l'EMA et `euler_integrate` réels.
Un run coûte **2 minutes** au lieu de 2 heures.

Deux mesures, dont le transport parfait échantillon par échantillon (bien plus
informatif qu'une distance entre distributions) :

| mesure | porte |
|---|---|
| courbure de la trajectoire intégrée / courbure exigée | ≥ 0.80 |
| pire nRMSE aux ancres intermédiaires | ≤ 2 × le bruit |

## Régime nominal (dim 64, fp32)

| variante | courbure | nRMSE interm. | porte |
|---|---|---|---|
| `time_scale=1` (état d'avant) | 0.480 | 0.0793 | non |
| `time_scale=1000` | 0.567 | 0.0807 | non |
| + `identity_prob=0` | 0.595 | 0.0716 | non |
| **+ `adjacent_only=true`** | **1.042** | **0.0650** | **oui** |
| témoin `bend=0` (marginales colinéaires) | *dégénéré* | 0.0628 | *sans objet* |

En dimension 64 le temps pèse 384 canaux de conditionnement contre 128 de latent :
il n'est pas noyé, et le correctif d'échelle ne rapporte que +18 %. **Ce régime ne
reproduit donc pas notre défaut principal** — d'où le second.

## Régime fidèle (dim 4096, bf16, bruit à SNR constant)

| variante | courbure | nRMSE interm. | porte |
|---|---|---|---|
| `time_scale=1` | 0.001 | 0.0115 | non |
| `time_scale=1000` | 0.016 (**×16**) | 0.0114 | non |
| + `adjacent_only` | 0.053 (**×53**) | 0.0116 | non |
| + `adjacent_only`, **fp32** | 0.316 (**×6** sur bf16) | 0.0103 | non |
| + `adjacent_only`, **16 000 itérations** | 0.584 | 0.0119 | non |
| + `adjacent_only`, `time_embed_dim` **2048** | **1.019** | **0.0070** | **oui** |
| + `adjacent_only`, **FiLM** (`time_embed_dim` 256) | **1.031** | **0.0075** | **oui** |
| **FiLM mais TOUTES les paires** | 0.516 | 0.0083 | **non** |

Ce régime reproduit fidèlement l'échec réel : courbure 0.001–0.05 contre les
0.005–0.023 mesurés sur les checkpoints de production.

## Ce que cela établit

**1. Le mécanisme de conditionnement est le facteur dominant, pas l'échelle du
temps.** Par concaténation, le temps pèse 256 canaux contre 258 048 de latent —
soit **0.026 %** de la variance d'entrée même après le correctif `time_scale`
(contre 0.0001 % avant). Le harnais montre qu'il en faut ~6 %. Y parvenir par
concaténation demanderait **~65 000 dimensions de temps** : non viable.

En modulation (FiLM : `scale, shift` appris par bloc résiduel), la force du
conditionnement **ne dépend plus de la dimension du latent** — 256 dimensions
suffisent alors là où la concaténation en exige 2048. C'est le mécanisme des deux
références : MONAI additionne l'embedding dans chaque resblock via une projection
apprise, Genentech/MMFM a l'option `sum_time_embed`.

**Cela explique la mesure du matin** : l'UNet (injection additive) variait de
1.7–2.6 % selon `t` là où le vectorisé (concaténation) était à **0.00 %**. Ce
n'était pas une différence d'architecture au sens large, mais précisément celle-là.

**2. `adjacent_only` était un vrai correctif, testé dans des conditions qui ne
pouvaient pas le révéler.** Le 2026-08-26 il a été mesuré neutre (34/60, p = 0.18)
— sur un modèle **aveugle au temps**, incapable de courber sa trajectoire quoi
qu'on fasse de ses cibles. Ici : FiLM + toutes paires = 0.516 (échec), FiLM +
adjacent = 1.031 (porte franchie). **Les deux correctifs sont complémentaires ;
aucun ne suffit seul.** Le test du 26 était valide, sa conclusion ne l'était pas.

**3. Ce n'est pas un problème de budget d'optimisation.** 16 000 itérations au
lieu de 4 000 font passer la courbure de 0.053 à 0.584 : mieux, mais toujours
sous la porte, alors que FiLM la franchit en 4 000.

**4. L'AMP bf16 coûte un facteur 6** sur la courbure (0.053 → 0.316 en fp32),
cohérent avec la mesure du matin : deux des quatre directions du plongement du
temps passent sous le plancher de quantification bf16.

## Réserve importante sur la métrique

En dimension 4096, **le nRMSE est plat** (0.0114 à 0.0119) sur des variantes dont
la courbure varie d'un facteur **53**. L'erreur y est dominée par la direction
« sujet » et le bruit, pas par la géométrie des marginales.

Conséquence directe pour l'évaluation réelle : **si la courbure s'améliore sur
données réelles, le nRMSE peut à peine bouger.** C'est cohérent avec le constat du
2026-08-25 — 80 % de l'erreur est une erreur d'échelle d'intensité — et avec le
fait que nRMSE et SSIM récompensent le flou. Ne pas conclure à l'échec du
correctif sur le seul nRMSE ; regarder aussi LPIPS, la netteté et les figures.

## Une erreur de calibration, corrigée

Le premier régime « fidèle » était **mal configuré** : `noise: 0.05` par élément à
4096 dimensions donne une norme de bruit de **3.2** pour une corde de **1.0** — le
signal était noyé, ce n'est pas notre problème. Les trois runs concernés
(`fidele_*.json`) donnaient une courbure de 0.001–0.026 pour de mauvaises raisons.
Le bruit a été renormalisé à SNR constant (`0.4/√dim = 0.00625`) et les runs
refaits (`hd_*.json`). Les conclusions ci-dessus portent sur les runs corrigés.

## Fichiers

| chemin | contenu |
|---|---|
| `ts1.json`, `ts1000.json`, `identity0.json`, `adjacent.json`, `colineaire.json` | régime nominal (dim 64) |
| `fidele_*.json` | **premier régime fidèle, MAL CALIBRÉ** — conservés pour trace |
| `hd_*.json` | régime fidèle corrigé (dim 4096, bruit à SNR constant) |
| `../../../src/cfm/synthetic_marginals.py` | le processus générateur et l'évaluation |
| `../../../src/cfm/arch_synthetic.py` | l'adaptateur (purement additif) |
| `../../../configs/mmfm/synthetic.yaml` | config de référence du harnais |

## Reproduction

```
PYTHONPATH=src python src/cfm/train_mmfm_unified.py \
    --method mmfm3d_synthetic --config configs/mmfm/synthetic.yaml --env local
PYTHONPATH=src python src/cfm/eval_synthetic.py --config configs/mmfm/synthetic.yaml
```

Toute variante s'obtient en surchargeant une clé de ce YAML — `model.time_scale`,
`model.time_cond`, `train.adjacent_only`, `synthetic.dim`, `synthetic.noise`.
