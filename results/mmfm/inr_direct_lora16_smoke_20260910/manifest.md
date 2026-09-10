# INR `hyper_hidden_dim=0` + `lora_rank=16` — bug corrigé, mécanisme confirmé, PAUSE avant la production

**Date** : 2026-09-10
**Code** : `src/cfm/inr_backbone.py` (correctif principal), `src/cfm/arch_inr.py`,
`src/cfm/train_inr_backbone.py`, `src/cfm/test_inr_backbone_smoke.py` (plomberie config)
**Checkpoints smoke** : `outputs/smoke/inr_direct_{rank0,rank16_prebugfix,lora16_lr1e-2,lora16_lr1e-3,lora16_lr1e-4}.pth`
**Décision** : pause délibérée avant tout entraînement de production (voir §5) — le
correctif de code est conservé (backward-compatible, vérifié), mais la suite du plan
(configs de production, backbone 50000 pas, precompute, flow, Task 3) n'a **pas**
été exécutée.

---

## 1. Pourquoi cette investigation

Un sondage gelé antérieur (`bench_inr_capacity.py::mod_lora16/64`, 2026-09-06/07)
avait montré qu'en bypassant le hypernetwork (goulot mesuré à rang 512) et en
optimisant directement une modulation shift+LoRA sur le SIREN de production
**gelé**, la fidélité rejoint/dépasse MedVAE. La question posée ici : peut-on
**entraîner** un nouveau backbone nativement avec `hyper_hidden_dim=0` (modulation
directe, `gamma=z`) et `lora_rank=16` (~46640 valeurs/volume), pour un latent du
même ordre de grandeur que celui de MedVAE (129024) — donc plausiblement
flow-matchable, contrairement au rang 64 (~182k, plus grand que MedVAE lui-même).

## 2. Bug trouvé et corrigé (avant tout calcul de production)

`INRBackbone.fit_latent` initialisait **toujours** `z` à zéro. Avec
`hyper_hidden_dim=0`, `z` **est** `gamma` directement (le hypernetwork devient
`nn.Identity()`) — donc les tranches LoRA `A` et `B` de `z` démarraient toutes les
deux à zéro : `dL/dA ∝ x@B=0` et `dL/dB ∝ A=0`, gradient LoRA mort des deux côtés,
pour toujours. Confirmé **empiriquement** avant tout correctif (§3) : `--lora-rank 16`
sans correctif (nRMSE_fg=0.2082) est indiscernable de `--lora-rank 0` (0.2070).

**Correctif** (`src/cfm/inr_backbone.py`) :
- `ModulatedSIREN.lora_mask()` / `init_direct_modulation()` — portage des
  fonctions déjà validées de `bench_inr_capacity.py::gamma_layout`/`init_gamma`
  (B aléatoire, A=0 — convention Hu et al. 2021) en méthodes d'instance.
- `INRBackboneConfig.lora_inner_lr` (nouveau champ, défaut `None`).
- `INRBackbone.__init__` : garde-fou `_direct_lora`, lève `ValueError` explicite
  si `hyper_hidden_dim=0`+`lora_rank>0` sans `lora_inner_lr`.
- `INRBackbone.fit_latent` : init de `z` via `init_direct_modulation` (au lieu de
  zéros) et pas de descente par-dimension (`step_lr`, `lora_inner_lr` sur le
  masque LoRA, `lr` partout ailleurs) quand `_direct_lora`.
- `meta_train_step` : passe le générateur de `sample_points` à l'init LoRA
  (évite de rejouer la même direction `B` à chaque pas de méta-entraînement).

**Vérifié rétrocompatible** : pour `lora_rank=0` (tout le code de production
actuel), `_direct_lora=False`, `step_lr` reste un simple float — comportement
inchangé bit à bit (testé directement, `fit_latent` produit un `z` identique
à l'ancien code sur un cas `hyper_hidden_dim=512, lora_rank=0`).

## 3. Diagnostic pré/post-correctif (smoke test, 8 volumes, 2000 pas méta)

| Variante | nRMSE_fg (porte [3/5]) | Ratio contrôle négatif [5/5] |
|---|---|---|
| `--lora-rank 0` (shift seul, témoin) | 0.2070 | 1.68 |
| `--lora-rank 16`, **pré-correctif** (LoRA morte) | 0.2082 | 1.83 |
| `--lora-rank 16`, post-correctif, `lora_inner_lr=1e-2` | 0.2055 | 1.81 |
| `--lora-rank 16`, post-correctif, `lora_inner_lr=1e-3` | 0.2132 | 1.64 |
| `--lora-rank 16`, post-correctif, `lora_inner_lr=1e-4` | 0.2053 | 1.81 |

Toutes les portes du smoke test PASSENT dans tous les cas (mécanisme sain,
z discrimine, pas de collapse) — mais le gain post-correctif reste **marginal**
(≤0.8%) et dans le bruit de variance d'un méta-entraînement à cette échelle
réduite (chaque run repart d'une initialisation de poids θ différente).

## 4. Diagnostic complémentaire — pourquoi le gain reste marginal

Deux sondages supplémentaires sur le checkpoint `lora16_lr1e-4` (le meilleur
des trois), θ gelé :

**a) Plus de pas de SGD** (mécanisme de production, `fit_latent`, pas fixe) :

| pas SGD | nRMSE_fg | temps |
|---|---|---|
| 50 (référence smoke) | 0.2053 | 45s |
| 200 | 0.2019 | 177s |
| 500 | 0.2002 | 442s |

**Toujours en amélioration, pas de plateau atteint à 500 pas.** Le budget de
production actuel (`inner_steps_eval=20`) est donc nettement insuffisant pour
exploiter cet espace de modulation — mais rien ne dit combien de pas
suffiraient, ni si la SGD y arriverait un jour à un coût raisonnable.

**b) Adam direct sur z** (comme `bench_inr_capacity.py::fit_by_adam`), d'abord
testé avec une init naïve à zéro (**erreur méthodologique de ce diagnostic** —
reproduit par inadvertance le bug du §2 pour ce test-ci seulement, sans toucher
au code de production) : 1000 pas → nRMSE_fg=0.2642, **pire que tout** —
résultat invalide, pas une preuve contre Adam.

Reproduit **correctement** (init via `init_direct_modulation`, groupes de taux
d'apprentissage séparés shift/LoRA, comme `bench_inr_capacity.py::gamma_layout`) :

| pas Adam | lr_lora | nRMSE_fg | temps |
|---|---|---|---|
| 100 | 1e-3 | **0.1813** | 24s |
| 100 | 1e-4 | 0.2568 | 24s |
| 300 | 1e-3 | **0.1387** | 70s |
| 300 | 1e-4 | 0.1917 | 70s |

**Adam à 300 pas (lr_lora=1e-3) bat la meilleure SGD à 500 pas de 30.7%
relatif, en 6.3x moins de temps.** Le mécanisme LoRA est bien vivant et
capable — c'est l'optimiseur de la boucle interne de production (SGD à pas
fixe) qui est mal adapté à un espace de modulation aussi grand, pas le
principe de la modulation directe+LoRA lui-même.

## 5. Pourquoi une pause, et ce qui reste ouvert

Ce résultat déplace le problème plutôt que de le résoudre dans le périmètre
approuvé : le plan initial ne prévoyait qu'un correctif d'initialisation dans
`fit_latent` (SGD inchangée), pas un changement d'optimiseur. Basculer le
*precompute* (`fit_new_volume`/`precompute_inr_latents.py`, θ gelé, sans
contrainte de différentiabilité) vers Adam est faisable et prometteur, mais
c'est une extension de périmètre non approuvée, avec un coût de calcul à
revalider (100-300 pas Adam × 1939 volumes ≈ 13-38h de precompute, contre
6-7h prévues initialement). Le méta-entraînement (boucle différentiable,
`create_graph=True`) resterait a priori en SGD — ce diagnostic montre qu'un θ
méta-entraîné via SGD répond déjà bien à un fit Adam a posteriori, donc rien
n'indique qu'il faille aussi changer l'optimiseur du méta-entraînement.

**Décision prise avec l'utilisateur (2026-09-10) : marquer une pause ici.**
Le correctif de code (§2) est conservé — c'est une amélioration générale,
rétrocompatible, réutilisable pour toute future tentative de modulation
directe+LoRA, indépendamment de la suite donnée à ce rang 16 précis. **Non
exécuté** : `configs/mmfm/inr_backbone_direct_lora16.yaml`/`inr_direct_lora16.yaml`,
l'entraînement complet du backbone (50000 pas, ~13-14h), le precompute, le
flow, l'évaluation Task 3.

## Réserves

- Ce diagnostic tourne sur un backbone **smoke** (2000 pas méta, 8 volumes),
  pas sur un backbone de production (50000 pas, 1939 volumes) — le
  comportement à l'échelle réelle (θ beaucoup mieux entraîné) pourrait différer,
  probablement en mieux.
- Les 3 runs de calibration (§3) partent chacun d'une initialisation aléatoire
  différente des poids θ — la comparaison entre `lora_inner_lr` est bruitée
  par cette variance, pas seulement par le taux d'apprentissage lui-même.
- Le sondage Adam (§4b) n'a pas balayé `lr_lora` au-delà de {1e-3,1e-4} ni de
  pas au-delà de 300 — la courbe n'est pas nécessairement saturée non plus.
