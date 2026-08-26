# Audit du code : quatre pistes, deux réfutées par la mesure

**Date** : 2026-08-26
**Objet** : deux symptômes persistants sur les figures — contraste mal transporté,
images trop lisses. Chercher dans le code ce qui les produit.

**Résumé** : un vrai défaut structurel trouvé (les marginales ne sont pas
alignées), une attribution nette obtenue (le flou de l'INR est entièrement
représentationnel), et **deux hypothèses que j'ai formulées puis que la mesure a
réfutées**. Les deux réfutations sont consignées avec le même soin que les
découvertes : ce sont elles qui empêchent de repartir sur une fausse piste.

---

## 1. Le problème d'estimation du flow — TROUVÉ

`_sample_step_plan` (`mmfm_core.py:255`) tire uniformément parmi les **20 paires
ordonnées** de champs et `_compute_flow` pose une **droite** entre chacune
(interpolation linéaire de torchcfm, `sigma: 0`). Ces droites ne sont mutuellement
compatibles que si les marginales sont alignées. Elles ne le sont pas :

Écart du point intermédiaire à la corde, rapporté à la longueur de la corde
(moyennes de classe, T1W, cache MedVAE, 40 volumes par champ) :

| corde | point intermédiaire | écart / corde | vs bruit d'estimation |
|---|---|---|---|
| 0.1T → 3T | 1.5T | **1.054** | 5.5× |
| 0.1T → 7T | 3T | 0.757 | 4.0× |
| 1.5T → 5T | 3T | **1.321** | 5.6× |
| 3T → 7T | 5T | 0.928 | 2.3× |
| 0.1T → 7T | 5T | 0.715 | — |

Le point intermédiaire s'écarte de la corde **d'autant que la corde est longue**,
et l'écart vaut 2.3 à 5.6 fois le bruit d'estimation de la moyenne. Même constat
sur le cache INR (0.77 à 1.36).

**Conséquence** : à un même `(z_t, t)`, un échantillon d'entraînement dit
« dirige-toi vers 3T », un autre « tu devrais déjà avoir dépassé 3T en route vers
7T ». Les cibles se contredisent et le modèle ne peut apprendre qu'un compromis —
c'est exactement le contraste incomplètement transporté qu'on voit sur les images.

**`adjacent_only: true` existe** (`mmfm_core.py:253`, `train.adjacent_only`) et
rend la chaîne cohérente : on n'entraîne que les sauts adjacents, les transitions
longues s'obtenant par intégration. **Jamais testé.** C'est le levier à instruire.

## 2. Le flou de l'INR est entièrement représentationnel — ATTRIBUÉ

Netteté (indice HF/BF relatif à la vérité, intérieur du cerveau, sujet 0006) :

| | auto-reconstruction (fit+décodage, **sans flow**) | pipeline complet | plafond 1 mm |
|---|---|---|---|
| T1W → 7T | 0.090 | 0.074 | 0.615 |
| T1W → 3T | 0.036 | 0.090 | 0.612 |
| T2FLAIR → 7T | 0.043 | 0.047 | 0.644 |

**Le flow n'ajoute aucun flou** : sans lui, l'INR est déjà aussi lisse. Les 1536
modulations sont toute l'histoire. **Aucun correctif de flow — ni `adjacent_only`,
ni le passage à une loss quadratique — ne rendra l'INR plus nette.**

## 3. RÉFUTÉ — la loss L1 n'explique pas le lissage

`mmfm_core.py:753` : `loss = F.l1_loss(v_t, ut_global)`, codé en dur.

C'est un **écart réel à la dérivation du flow matching** : le champ marginal qui
transporte p₀ vers p₁ est `u_t(x) = E[u_t(x|z) | x_t = x]`, une **espérance**, et
on ne la récupère qu'avec un coût **quadratique**. La L1 donne la médiane
conditionnelle composante par composante, un autre champ, sans garantie de
transport.

**J'ai annoncé que cela expliquerait le lissage. C'est faux.** Sur les vraies
cibles de vitesse (source 3T, T1W, toutes paires de 40 volumes par champ) :

| cible | ‖moyenne‖ | ‖médiane‖ | rapport | HF conservée par la médiane |
|---|---|---|---|---|
| 0.1T | 3960.7 | 4007.2 | 1.012 | 129 % |
| 1.5T | 9602.3 | 9450.5 | 0.984 | 144 % |
| 5T | 6148.5 | 6073.8 | 0.988 | 130 % |
| 7T | 2488.0 | 2488.4 | 1.000 | 106 % |

Médiane et moyenne ont la même norme, et la médiane conserve **plus** de hautes
fréquences, pas moins. La distribution conditionnelle est trop symétrique pour que
l'écart compte. **Le correctif reste souhaitable par correction théorique, mais il
ne faut pas en attendre d'effet sur les symptômes observés.**

## 4. RÉFUTÉ — l'interpolation cubique ne gagne rien de bout en bout

`infer_mmfm_unified.py` remonte la prédiction de 1 mm à 0.5 mm avec `order=1`.
Sur un volume **parfait** faisant seulement cet aller-retour — ce qui borne ce
qu'un modèle à 1 mm peut atteindre :

| | netteté | nRMSE | SSIM |
|---|---|---|---|
| ordre 1 | 0.615 (T1W) / 0.644 (T2FLAIR) | 0.0382 / 0.0326 | 0.9952 / 0.9957 |
| **ordre 3** | **0.910 / 0.922** | **0.0251 / 0.0199** | **0.9977 / 0.9984** |

Le plafond monte nettement. **De bout en bout, il ne se passe presque rien** :

| T1W | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| Vectorisé, ordre 1 | **0.4353** | **0.8997** | 0.0983 |
| Vectorisé, ordre 3 | 0.4374 | 0.8966 | **0.0969** |
| INR, ordre 1 | **0.4070** | **0.8606** | 0.1572 |
| INR, ordre 3 | 0.4075 | 0.8602 | **0.1561** |

**Le plafond n'est pas contraignant** : le modèle n'a rien à mettre dans la bande
que l'ordre 3 laisse passer. Un peu de LPIPS gagné, un peu de nRMSE perdu — un
arbitrage perception/distorsion, pas un gain. `upsample_order` est conservé comme
paramètre, **avec 1 pour défaut** : changer pour un résultat neutre invaliderait
tous les chiffres publiés.

> Le UNet n'a pas pu être évalué ici : `configs/mmfm/unet.yaml` porte
> `use_adagn: true` alors que le checkpoint de référence est PRE-AdaGN, et le
> chargement échoue bruyamment — comportement attendu et documenté le 2026-08-13.

## 5. Une correction à mes propres conclusions

Le manifeste du 2026-08-25 et celui du 2026-08-26 écrivent que « les deux bonnes
architectures ne restituent qu'un tiers de la finesse de la vérité ». **C'était une
confusion entre la résolution de travail et un défaut de modèle.** Rapporté au
plafond réellement atteignable à 1 mm (0.61-0.64), le vectorisé est à **66 %
(T1W), 108 % (T2W), 50 % (T2FLAIR)** — sur T2W il le dépasse en ajoutant de la
texture. L'INR, elle, est à **12 %** : son flou est bien le sien.

Conséquence pour la décision MedVAE-LPIPS en attente : **un meilleur VAE à 1 mm ne
peut pas franchir le plafond de 0.61.** Le levier de netteté pour le vectorisé et
l'UNet est la résolution de travail, pas le VAE.

## 6. Zones vérifiées sans défaut

| zone | vérification |
|---|---|
| `euler_integrate` | `dt = (t_end−t_start)/n_steps`, boucle exacte, `dt` signé — pas de décalage |
| Convention d'intervalle | le décodeur sort en [−1,1], `(clip+1)/2` ramène en [0,1], `denormalize_from_01` mappe vers [lo,hi] — cohérent |
| Remplissage avant encodage | `center_crop_or_pad_np(mode="reflect")` — pas de constante parasite |
| Masque par le support de la source | 0.01 à 0.09 % de l'énergie de la cible perdue — négligeable |
| Encodage/décodage par tuiles | tuiles **disjointes** en espace latent avec contexte, aucun moyennage — pas de lissage |

## 7. Fichiers

| chemin | contenu |
|---|---|
| `task3_vectorized_order3_T1W.csv` | vectorisé, interpolation cubique |
| `task3_inr_order3_T1W.csv` | INR, interpolation cubique |
