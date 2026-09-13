# Hypothèse « hétérogénéité d'échelle » testée et ÉCARTÉE

**Date** : 2026-09-13/14
**Contexte** : suite de `results/mmfm/inr_direct_lora16_task3_20260913/manifest.md`
(Task 3 NÉGATIF et sévère pour l'INR direct+LoRA rang 16). Deux causes plausibles
y étaient identifiées, non départagées : (1) hétérogénéité d'échelle par
dimension (×49.9, contre ×2.9 pour l'ancien cache MedVAE) mal servie par un
scalaire unique de normalisation ; (2) rugosité de l'espace latent.

## Ce qui a été testé

**Uniquement la cause (1).** Extension de `VectorMMFM`/`arch_vector.py`
(`src/cfm/mmfm_vectorized.py`) pour accepter un vecteur `(latent_dim,)` en plus
d'un scalaire pour `latent_mean`/`latent_scale` (`model.latent_mean_path`/
`model.latent_scale_path`, chargés depuis un `.pt`) — rétrocompatible, vérifié
sur les configs scalaires existantes (buffer 0-dim inchangé).

Vecteur construit : constante = moyenne/écart-type mesurés séparément sur la
portion shift (1536 dims, std=4.27e-02) et la portion LoRA (45104 dims,
std=1.41e-02) du MÊME cache `inr_e124f7c5` — aucun re-precompute, seul le flow
est réentraîné (25000 pas, 80 min, backbone et cache identiques à l'expérience
précédente). Config : `configs/mmfm/inr_direct_lora16_pergroup.yaml`.

## Résultat : AUCUN changement mesurable

| | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| Scalaire unique (2026-09-13) | 0.4587 | 0.6939 | 0.3499 |
| **Par groupe (ce test)** | **0.4586** | **0.6906** | **0.3572** |

Écart dans le bruit (0.0001 de nRMSE). **Duel paire à paire : 33/60** — la
variante par groupe ne bat même pas la variante scalaire de façon significative
(quasi 50/50). Les deux variantes restent aussi négatives l'une que l'autre par
rapport aux trois architectures existantes et au témoin identité.

## Verdict

**L'hétérogénéité d'échelle par dimension N'EST PAS la cause du désastre Task 3.**
Elle est écartée avec un test propre (backbone et cache identiques, seule
variable changée). Reste la cause (2), non testée directement ici : la
rugosité de l'espace latent (le diagnostic `diagnose_inr_latent_smoothness.py`
lancé pour la mesurer directement s'est avéré anormalement lent sur cette
architecture — >2h30 sans terminer, probablement le surcoût LoRA par appel
`decode` — et a été interrompu sans résultat exploitable).

Aucune piste supplémentaire n'a été engagée après ce résultat. Voir le
manifeste parent pour la liste complète des pistes restantes (régularisation
de lissage sur `z`, rang LoRA plus petit).
