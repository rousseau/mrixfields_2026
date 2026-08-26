# Évaluation refaite sur l'INR corrigée — qualitatif et quantitatif

**Date** : 2026-08-26
**Pourquoi** : l'évaluation du 2026-08-25
(`../qualitative_20260824/manifest.md`) décrivait un INR qui n'existe plus. Ses
figures, sa décomposition d'erreur et ses tests appariés portaient sur un
checkpoint affecté par trois bugs d'inférence, corrigés depuis
(`../audit_20260825/manifest.md`). Le manifeste du 2026-08-25 a été annoté en
conséquence ; il n'est pas effacé, il porte ce qu'on croyait.

**Ce qui change** : uniquement l'INR. Les chiffres du vectorisé et de l'UNet ont
été explicitement re-contrôlés pendant l'audit (orientation alignée : 0.4351
contre 0.4353 ; sans EMA : 0.4359 contre 0.4353 — les deux neutres) et sont
inchangés.

---

## 1. Le résultat qui corrige le précédent

Le 2026-08-26 au matin, le manifeste d'audit et le message de commit écrivaient
« l'INR devient la meilleure architecture en nRMSE ». **C'est vrai de la moyenne
agrégée et faux au sens statistique.**

Tests appariés sur les 60 paires (20 paires × 3 contrastes), nRMSE :

| comparaison | victoires | test des signes | Wilcoxon | écart moyen |
|---|---|---|---|---|
| **INR < Vectorisé** | **29/60** | **p = 0.65** | **p = 0.38** | −0.0046 |
| INR < UNet | 35/60 | p = 0.12 | p = 0.02 | −0.0285 |
| Vectorisé < UNet | 38/60 | p = 0.026 | p = 1.4e-4 | −0.0239 |

**L'INR et le vectorisé sont indiscernables en nRMSE.** La moyenne agrégée
(0.3749 contre 0.3794) est portée par quelques paires à fort écart, pas par une
supériorité systématique. L'INR bat en revanche l'UNet avec un soutien modéré, et
le vectorisé bat l'UNet avec le soutien le plus net.

Sur SSIM et LPIPS il n'y a pas d'ambiguïté : vectorisé 0.8975 / 0.0941, INR
0.8631 / 0.1557.

## 2. Face au témoin « ne rien faire »

Gain paire-à-paire, `1 − nRMSE_modèle / nRMSE_identité` :

| | T1W | T2W | T2FLAIR |
|---|---|---|---|
| Vectorisé | +34.1 % | +5.6 % | −1.2 % |
| UNet | +31.3 % | +1.1 % | −0.7 % |
| **INR corrigée** | **+34.9 %** | **−7.7 %** | +3.1 % |

Nombre de paires où le modèle bat le témoin (nRMSE | SSIM), sur 20 :

| | T1W | T2W | T2FLAIR |
|---|---|---|---|
| Vectorisé | 18 \| 13 | 12 \| 7 | 9 \| 11 |
| UNet | 18 \| 13 | 11 \| 7 | 9 \| 11 |
| **INR corrigée** | 16 \| 9 | **9 \| 2** | 10 \| 10 |

**Sur T2W, l'INR corrigée fait moins bien que recopier la source** (−7.7 %, battue
sur 11 paires sur 20, et sur 18 en SSIM). C'est sa faiblesse restante, et elle
n'apparaît pas dans la moyenne des trois contrastes. Le correctif a supprimé les
bugs ; il n'a pas rendu l'architecture bonne partout.

## 3. Qualitatif — ce que les images montrent maintenant

Les panneaux portent une colonne supplémentaire, `--with-before`, qui affiche
l'INR d'AVANT l'audit sur la même figure et à la même échelle d'erreur : l'effet
des trois bugs devient visible d'un coup d'œil.

| figure | ce qu'elle établit |
|---|---|
| `panel_T1W_3T_to_7T_0006.png` | L'INR corrigée rend une **anatomie reconnaissable** — ventricules, noyaux gris, ruban cortical — là où l'ancienne donnait des blobs. Mais elle est **visiblement plus lisse** que le vectorisé : le ruban cortical est émoussé, les détails fins ont disparu. |
| `panel_T2W_3T_to_7T_0006.png` | Le contraste où l'INR reste faible ; à comparer au témoin identité, déjà sombre. |
| `panel_T2FLAIR_3T_to_7T_0006.png` | Même lecture ; l'INR corrigée ne sature plus. |
| `panel_T1W_3T_to_0.1T_0006.png` | L'échec inverse, commun aux trois : la vérité 0.1T est floue et les prédictions sont plus nettes qu'elle. |
| `panel_T1W_3T_to_7T_0009.png` | Le volume aberrant (norme L2 2.6× plus faible) — le sujet qui portait 46 % du « mur du 7T ». |
| `panel_T2FLAIR_3T_to_0.1T_0006.png` | Une paire où les modèles perdent contre le témoin. |
| `calib_T1W_3T_to_7T_0009.png` | Prédiction brute contre prédiction remise à l'échelle par l'oracle. |
| `spectres_radiaux.png` | Puissance spectrale radiale, 3 contrastes × 2 paires. |

**Le défaut visuel de l'INR a changé de nature** : ce n'était pas du flou, c'était
un effondrement structurel dû aux bugs. Maintenant c'est du flou — le vrai plafond
des 1536 modulations.

## 4. L'erreur de l'INR est, elle aussi, d'abord une erreur d'échelle

Correction par le meilleur facteur multiplicatif **global** possible (un oracle :
il connaît la vérité). nRMSE brut → après correction :

| contraste | Vectorisé | UNet | **INR corrigée** |
|---|---|---|---|
| T1W | 0.4353 → 0.2181 | 0.4617 → 0.2333 | 0.4070 → **0.2607** |
| T2W | 0.3376 → 0.2717 | 0.3676 → 0.2925 | 0.3800 → **0.3213** |
| T2FLAIR | 0.3654 → 0.1676 | 0.3807 → 0.1726 | 0.3376 → **0.2134** |

Part de l'énergie de l'erreur supprimée par ce seul scalaire :

| contraste | Vectorisé | UNet | INR **corrigée** | *INR avant audit* |
|---|---|---|---|---|
| T1W | 81.9 % | 81.0 % | **70.4 %** | *23.7 %* |
| T2W | 39.4 % | 41.4 % | **30.4 %** | *9.3 %* |
| T2FLAIR | 82.9 % | 83.3 % | **66.4 %** | *20.0 %* |

> **Une conclusion du 2026-08-25 est renversée.** Ce manifeste écrivait :
> « pour l'INR, seuls 20-24 % de son erreur partent avec le facteur d'échelle ; le
> reste est structurel — sa limite n'est pas de même nature que celle des deux
> autres. » **C'était un artefact des bugs.** Une fois ceux-ci corrigés, l'INR
> passe à 66-70 % sur T1W et T2FLAIR : **sa limite est de la même nature que celle
> des deux autres architectures.** L'erreur dominante de tout le pipeline est une
> erreur de calibration d'intensité, pour les trois.

L'estimateur réalisable (reporter sur la cible l'écart mesuré sur le volume
source) **dégrade toujours**, pour l'INR comme pour les autres — sauf sur T2W où
il gagne marginalement (+4.6 %). Non biaisé mais trop bruité, conclusion inchangée.

## 5. La netteté : le vrai plafond de l'INR, enfin isolé

Indice de netteté relative `(HF/BF)_prédiction / (HF/BF)_vérité`, **médiane sur
l'intérieur du cerveau** (`--interior-frac 0.5`, qui écarte les artefacts de bord).
Le témoin identité à 1.00 valide la mesure.

| contraste | identité | Vectorisé | UNet | **INR corrigée** |
|---|---|---|---|---|
| T1W | 1.02 | 0.36 | 0.36 | **0.09** |
| T2W | 1.00 | 0.40 | 0.42 | **0.05** |
| T2FLAIR | 1.00 | 0.26 | 0.26 | **0.06** |

**L'INR corrigée ne restitue que 5 à 9 % de la finesse de la vérité, soit 4 à 8
fois moins que le vectorisé.** C'est le goulot de 1536 modulations pour 8.26 M
voxels, mesuré sans le confondre avec autre chose.

Voilà l'explication complète du compromis : une image extrêmement lisse minimise
l'erreur quadratique (nRMSE au niveau du vectorisé) et détruit la structure (SSIM
et LPIPS nettement en retrait). Ce n'est pas un défaut à corriger par un réglage,
c'est ce que 1536 nombres permettent de décrire.

## 6. Conclusions

1. **INR et vectorisé sont indiscernables en nRMSE** (29/60 paires, p = 0.65). La
   moyenne agrégée qui donnait l'INR devant est portée par quelques paires. Le
   vectorisé reste **nettement devant en SSIM et LPIPS**, et il est deux fois plus
   léger. **Il reste le choix de production.**
2. **L'INR corrigée reste faible sur T2W** : −7.7 % contre le témoin identité,
   battue sur 11 paires sur 20 en nRMSE et 18 sur 20 en SSIM. Corriger les bugs ne
   l'a pas rendue bonne partout.
3. **Les trois architectures partagent la même limite dominante** : 66 à 83 % de
   l'énergie de leur erreur part avec un seul scalaire d'intensité par volume. La
   distinction « l'INR est structurellement différente » n'était qu'un effet des
   bugs.
4. **Le plafond propre à l'INR est la netteté**, et il est chiffré : 5 à 9 % de la
   finesse réelle. Seule piste à fort levier connue — élargir la modulation
   (per-couche directe, `w0` croissant en profondeur) ou canoniser l'ajustement du
   latent, dont la géométrie reste le second problème (25 % de structure commune
   contre 91 % pour MedVAE).

## 7. Fichiers

| chemin | contenu |
|---|---|
| `panel_*.png` | panoramas 6 colonnes, avec la colonne « INR avant audit » |
| `calib_T1W_3T_to_7T_0009.png` | brut contre remis à l'échelle par l'oracle |
| `spectres_radiaux.png` / `.csv` | puissance spectrale radiale |
| `calibration_intensite.csv` | décomposition d'échelle, 540 volumes |
| `sharpness_index_interieur_0.5.csv` | indice de netteté, intérieur du cerveau |
| `calibration.log`, `sharpness_interieur.log` | sorties brutes |

Le témoin identité (`../qualitative_20260824/identity_baseline_*.csv`) est
inchangé : il ne dépend d'aucune architecture.
