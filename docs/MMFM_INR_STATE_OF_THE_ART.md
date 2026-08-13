# MMFM — Variante INR : analyse NOIR + état de l'art + design

Ce document analyse l'article NOIR (arXiv:2603.13118) et situe une future variante MMFM basée sur
une implicit neural representation (INR) par rapport à l'état de l'art, en vue de la Task 3 du
challenge MRIxFields 2026 (traduction de champ magnétique any-to-any, 5 champs × 3 contrastes,
entraînement non-appairé sur données rétrospectives). Complète `docs/MMFM_ARCHITECTURE.md` (design
unifié vectorisé/UNet) — voir ce document pour le contexte du pipeline `mmfm_core.py`/`ArchAdapter`
sur lequel s'appuie ce qui suit.

**Décision de scope actée** : l'INR opérera sur le **volume brut**, remplaçant MedVAE pour cette
variante (pas une reparamétrisation du latent MedVAE) — voir section 5.

## 1. Analyse détaillée : NOIR (El Hadramy et al., arXiv:2603.13118, 13 mars 2026)

*Univ. de Bâle (Dépt. Biomedical Engineering) + Harvard Medical School / Brigham and Women's
Hospital.*

### 1.1 Problème et thèse

NOIR conteste l'hypothèse implicite de la quasi-totalité du deep learning médical : que l'image
médicale doit être représentée et traitée comme une grille discrète pixel/voxel. Les auteurs
reformulent quatre tâches usuelles (segmentation, complétion de forme, synthèse d'image, traduction
d'image) comme un problème unique de **Neural Operator (NO) learning entre espaces de fonctions
continues**, en s'appuyant sur des implicit neural representations (INR) pour matérialiser ces
espaces de fonctions.

### 1.2 Architecture

**Bloc 1 — Représentation continue par INR.** Pour un signal discret `f_i^d` (image/volume), on
cherche une fonction continue `f_i ≈ φ(x; θ, γ_i)` où :
- `φ` est un MLP SIREN (activations sinusoïdales, Sitzmann et al. 2020), 6-10 couches cachées de
  256 unités, dernière couche à activation sigmoïde (régression) ou softmax (segmentation
  multi-classe).
- `θ` : paramètres **partagés** au niveau du dataset (capturent la structure commune : anatomie,
  caractéristiques d'acquisition).
- `γ_i` : paramètres de **modulation spécifiques à l'échantillon** — au lieu d'être optimisés
  directement, ils sont **prédits par un hypernetwork `M_ψ`** à partir d'un code latent bas-dimensionnel
  `z_i ∈ ℝ^{p}` (`p` = 64 pour Shenzhen/segmentation binaire, jusqu'à 4096 pour fastMRI/traduction).
  L'hypernetwork a 1 couche cachée de 64 unités.

Entraînement en **meta-learning** (Algorithme 1, dans la lignée de MetaSDF/DeepSDF) :
```
pour chaque minibatch d'échantillons i :
    z_i ← 0                                   # init à zéro
    pour k = 1..K (K=5 en train) :
        z_i ← z_i - α ∇_z L(f_i^d, φ(·; θ, M_ψ(z_i)))   # boucle interne, LR α=1e-2
    (θ, ψ) ← (θ, ψ) - β ∇_(θ,ψ) Σ L_i                    # UN pas de boucle externe, LR β=5e-6
```
À l'inférence sur un **nouveau** volume (Algorithme 2) : `(θ, ψ)` restent gelés, seul un nouveau
`z*` est optimisé (K=10 pas) — pas de ré-entraînement du réseau partagé, fitting rapide par
descente de gradient depuis une init apprise.

**Bloc 2 — Neural Operator (NO).** Un MLP à connexions résiduelles (1-3 couches, 512-2048 unités
selon la tâche, activation SiLU/ReLU) apprend `T: z_in → z_out`, où `z_in`/`z_out` sont les codes
latents des INR d'entrée et de sortie (obtenus séparément via Algorithme 2). Entraînement par
**simple régression MSE** : `L_NO = (1/N) Σ ‖T(z_in^i) − z_out^i‖²` — **pas de composante
générative/stochastique**.

### 1.3 Propriété théorique : ε-ReNO

NOIR est montré empiriquement satisfaire la propriété d'**ε-Representation-equivalent Neural
Operator** (Bartolucci et al. 2023) : l'erreur d'aliasing entre discrétisations différentes du même
signal reste bornée par une petite constante ε. Vérifié en pratique sur Shenzhen à 5 résolutions
(32² à 200²) : `ε_zin = 4×10⁻⁶`, `ε_zout = 2.2×10⁻⁵`, `ε_seg = 1.8×10⁻²` — les latents (et donc les
prédictions) restent quasi-identiques quelle que soit la résolution d'entrée. C'est la propriété la
plus directement transposable à notre problème : les 5 champs du challenge n'ont pas le même FOV
natif après reséchantillonnage, ce qui nous a coûté un travail non négligeable cette session
(`center_crop_or_pad_np`, padding `_pad_to_multiple` pour le UNet, alignement des configs
vectorisé/UNet sur 96×112×96@2mm). Une architecture native-résolution-invariante supprimerait une
classe entière de ces problèmes.

### 1.4 Résultats

| Tâche | Dataset | NOIR vs meilleure baseline |
|---|---|---|
| Segmentation binaire | Shenzhen (CXR) | DSC 0.94, à égalité avec FNO/ViT (0.95) |
| Segmentation multi-classe | OASIS-4 (cerveau) | DSC 0.85, **meilleur que tous** à basse résolution (≤6% d'écart au meilleur, partout) |
| Complétion de forme 3D | SkullBreak | DSC 0.86, 3-4% derrière FNO/ViT — détail fin (région orbitale) sous-représenté |
| **Synthèse d'image** | Ultrasound (label→image) | **PSNR 30.94, meilleur que tous** (AttU-Net 28.55) — attribué à l'absence de lissage implicite conv/attention |
| **Traduction d'image** | fastMRI (PD→T2, genou 2D) | PSNR 22.87 **identique aux 3 résolutions**, mais **derrière DDPM** (24.55-25.59) et ≈ViT |

**Point le plus important pour nous** : sur la tâche la plus proche de la nôtre (traduction
d'image), NOIR n'est **pas** l'état de l'art en fidélité brute — DDPM (diffusion) le bat nettement.
Le NO de NOIR est un régresseur MSE déterministe ; ce n'est probablement pas un hasard qu'il perde
face à un modèle génératif sur une tâche où la distribution cible a une vraie composante stochastique
(texture, bruit, détail non déterministe à partir de la seule source). NOIR l'emporte en revanche
clairement là où (a) la sortie est peu ambiguë depuis l'entrée (segmentation) ou (b) l'absence de
lissage conv/attention aide (synthèse depuis un label map, où le détail fin importe plus que la
cohérence stochastique).

### 1.5 Limites déclarées par les auteurs
- Dépendance critique à la dimension du latent `p` : doit croître avec la complexité/variabilité du
  signal, ce qui complique le co-design avec l'architecture du NO (plus `p` grandit, plus `T` doit
  être expressif, plus il faut de données pour l'entraîner sans surapprentissage).
  Nécessite des données appariées (`f_i^d`, `g_i^d`) — pas utilisable en non-supervisé tel quel
  (les auteurs le notent comme limite explicite, avec la registration comme exemple d'usage bloqué).
  **Notre Task 3 est non-appariée par sujet** (retro_train), mais notre design multi-marginal actuel
  contourne déjà ce problème via l'échantillonnage indépendant par classe (mod, champ) — le NO/flow
  matching de la variante INR devra suivre le même schéma que le vectorisé/UNet actuels, pas
  reproduire le schéma d'entraînement appairé de NOIR tel quel.

## 2. État de l'art — par thème

### 2.1 Représentation continue (fonction d'activation / encodage de coordonnées)

| Méthode | Principe | Points forts | Points faibles |
|---|---|---|---|
| **SIREN** (Sitzmann et al. 2020) | MLP à activations `sin(ω₀·Wx+b)` | Standard de facto (NOIR, MedFuncta), bien étudié, dérivées analytiques stables | Sensible à l'initialisation/`ω₀` ; artefacts de "ringing" (haute fréquence parasite) |
| **Fourier features** (Tancik et al. 2020) | Encodage positionnel (`sin`/`cos` à fréquences multiples) + MLP ReLU classique | Simple, ancien, bien compris | Moins expressif que SIREN/WIRE à budget de paramètres égal |
| **WIRE** (Saragadam et al., CVPR 2023) | Activation en ondelette de Gabor complexe | Capacité de représentation la plus haute des trois, convergence plus rapide, erreur plus compacte spatialement (moins de ringing que SIREN) ; **utilisé par HULFSynth pour la synthèse IRM champ bas→haut** | Complexité de calcul un peu plus élevée (nombres complexes) |
| **Hash encoding multi-résolution** (Müller et al., Instant-NGP 2022) | Table de hachage apprise + interpolation, très rapide à fitter par instance | Fitting quasi-instantané | Conçu pour l'overfit per-instance (NeRF temps réel) — moins naturel pour la généralisation dataset-level dont on a besoin (le principe même — mémoriser via une table de hash dense — s'oppose à la compacité recherchée pour un vecteur latent flowable) |

**Choix retenu pour le Candidat 1** : SIREN (le plus documenté, code de référence disponible pour
NOIR et MedFuncta). **WIRE retenu comme Candidat 2** (swap d'activation à faible coût si SIREN
déçoit en fidélité ou vitesse de fitting).

### 2.2 Généralisation et conditionnement (un INR par instance ne généralise pas)

Un INR seul (`φ(x;θ)` sans modulation) doit être re-fitté à zéro pour chaque nouveau signal — pas
un problème pour NeRF/reconstruction 3D isolée, rédhibitoire pour un dataset médical de milliers de
volumes. Trois familles de solutions, non exclusives :

1. **Hypernetworks** (Ha et al. 2016) : un réseau auxiliaire prédit les poids (ou une modulation des
   poids) du réseau cible à partir d'une entrée conditionnante. C'est le mécanisme utilisé par NOIR
   (`M_ψ: z → γ`).
2. **Auto-decoder** (DeepSDF, Park et al. 2019) : code latent par instance appris conjointement avec
   un décodeur partagé — pas d'encodeur explicite, le latent est directement optimisé (ou, comme
   NOIR/MetaSDF, initialisé à zéro puis affiné en quelques pas).
3. **Meta-learning** (MetaSDF, Sitzmann et al. 2020) : apprend une **initialisation** partagée qui
   permet un fitting rapide (quelques pas de gradient) sur un nouveau signal — c'est la technique
   combinée avec l'auto-decoder dans NOIR et MedFuncta (Algorithme 1/2 ci-dessus).
4. **Modulation locale/spatiale** — au lieu d'un vecteur latent global unique par volume, une
   fonction de modulation *spatialement variable* : **3D MTransINR** (CIABiomed 2025) génère un
   champ de modulation voxel-wise via un petit 3D U-Net, appliqué à un MLP de coordonnées partagé
   (bias modulation). Résultat : garde une partie de l'inductive bias spatiale d'un UNet classique
   tout en restant évaluable à résolution arbitraire — hybride entre notre `arch_unet.py` actuel et
   un INR pur. Bat un baseline 3D Pix2Pix sur ProstateX (PSNR/SSIM, toutes résolutions testées).
   **SeCo-INR** (semantic conditioning) explore une idée voisine pour la super-résolution médicale
   (conditionnement par features sémantiques plutôt que par un simple vecteur latent).

**Implication pour nous** : les options 1-3 (latent global, vecteur plat) sont directement
compatibles avec notre `VectorMMFM` existant comme modèle de flow — c'est le chemin le moins
coûteux et le premier à tester (Candidat 1). L'option 4 (modulation locale) est plus proche en
esprit de notre `arch_unet.py`, mais casserait la réutilisation directe de `VectorMMFM` — réservée en
Candidat 3 (stretch) si la perte de détail anatomique fin s'avère un problème réel avec un latent
global.

### 2.3 Frameworks médicaux directement comparables

- **NOIR** (2026) — voir section 1.
- **MedFuncta** (Friedrich, Bieder, McGinnis, Wolleb, Rueckert, Cattin — MIDL 2026 oral,
  arXiv:2502.14401) — **même laboratoire que NOIR (Cattin lab, Bâle)**, publié quelques mois avant.
  Framework de neural fields médicaux **à grande échelle** : encode des signaux médicaux divers en
  un vecteur latent 1D unifié qui module un neural field partagé méta-appris. Contributions
  techniques : (a) fréquence `ω` **non-constante** dans les activations SIREN, avec un lien établi
  entre le schedule de `ω` et les learning rates par couche (justification théorique de la dynamique
  d'apprentissage) ; (b) stratégie de meta-learning **scalable via supervision creuse**
  (sparse supervision — sous-échantillonner les points de supervision pendant l'entraînement plutôt
  que d'utiliser tous les voxels) pour réduire mémoire/calcul, condition nécessaire pour passer à
  l'échelle sur de gros volumes 3D (notre cas : volumes bien plus gros que les images 2D 200×200 de
  Shenzhen). Publie code, poids pré-entraînés, et **MedNF** — un dataset de >500k vecteurs latents
  pour neural fields médicaux multi-instances. **Candidat naturel comme point de départ/backbone de
  fitting** pour notre variante INR — probablement plus robuste et mieux dimensionné pour nos
  volumes 3D complets que la recette « telle quelle » de NOIR (pensée pour des images 2D/petits
  volumes 3D dans leurs expériences).
- **Local Implicit Neural Representations for Multi-Sequence MRI Translation**
  (arXiv:2302.01031, 2023) — précurseur direct sur la traduction de séquence IRM via INR **locaux**
  (pas de vecteur latent global) — antérieur à NOIR/MedFuncta/3D MTransINR, probablement la
  référence historique qui a motivé le choix du conditionnement local dans les travaux plus récents.
  À creuser en détail avant de trancher entre latent global (Candidat 1) et conditionnement local
  (Candidat 3) — si ce papier montre déjà que le global perd trop de détail sur une tâche de
  traduction IRM, cela accélérerait la décision de tester le Candidat 3 plus tôt.
- **HULFSynth** (arXiv:2511.14897, 2025) — synthèse bidirectionnelle **ultra-low-field
  (B0<0.1T) ↔ high-field (B0≥1T)** — quasiment notre paire de champs la plus extrême (0.1T↔7T).
  Design radicalement différent de NOIR/MedFuncta : **non-supervisé et physique**, pas un modèle
  génératif appris de bout en bout. Le forward model estime un facteur de contraste par type de
  tissu (`m ∈ ℝ³`, matière blanche/grise/LCR) via un système linéaire explicite, puis un INR à
  activation **WIRE** (pas SIREN) prédit conjointement l'intensité haut-champ et une segmentation
  tissulaire, optimisé par une loss composite (MAE + Dice/CE + variation totale) — sans jamais voir
  de paire réelle basse-champ/haute-champ pour l'entraînement du réseau lui-même. Résultats forts :
  amélioration du contraste substance blanche/grise ×4 (synthétique) et ×2.5 (données réelles 64mT),
  Dice 0.67 vs 0.33 pour SynthSeg sur les données réelles ; les baselines supervisées
  (LoHiResGAN, SynthSR) échouent carrément hors distribution. **Non directement réutilisable** pour
  notre cadre (nous avons besoin d'un mapping appris multi-marginal/multi-contraste, pas d'un modèle
  physique par tissu), mais instructif comme alternative paradigmatique : si notre approche
  générative peine sur les transitions vers 0.1T/7T (nos paires historiquement les plus difficiles),
  une piste physique-informée reste une option de repli à garder en tête.
- **3D MTransINR** (CIABiomed 2025) — voir section 2.2 (conditionnement local).

### 2.4 Neural Operators (cadre théorique de NOIR)

NOIR se positionne comme une instance de Neural Operator (Kovachki et al. 2021) — modèles qui
apprennent des correspondances entre espaces de fonctions de dimension infinie plutôt qu'entre
discrétisations fixes. Familles connues : DeepONet (Lu et al. 2019), Fourier Neural Operator (FNO,
Li et al. 2020 — restreint aux grilles régulières, convolutions spectrales), Convolutional Neural
Operator (CNO), Graph Neural Operators. Le critère d'**ε-ReNO** (Bartolucci et al. 2023) — que NOIR
satisfait empiriquement et que FNO ne satisfait pas selon les auteurs — donne un protocole concret et
réutilisable pour **vérifier**, pas seulement affirmer, qu'une architecture candidate est
véritablement robuste au changement de résolution (voir Phase 1 du plan : reproduire ce test sur nos
propres données).

### 2.5 Génératif en espace latent — le pont avec notre pipeline MMFM existant

Le point le plus directement exploitable pour notre projet : une lignée de travaux fait déjà du
**génératif** (pas de la simple régression) directement sur les latents de neural fields. **Functa**
(Dupont et al., "From data to functa", 2022) apprend un GAN/modèle génératif sur l'espace des
modulations d'un neural field partagé. **COIN++** (Dupont et al. 2022) applique une idée voisine à la
compression cross-modale. Plus largement, la **"Latent Flow Matching"** (flow matching ODE
simulation-free dans un espace latent compact, ex. VinAI LFM) est un pattern déjà établi et validé en
dehors du contexte médical pour la génération d'images haute résolution, en tirant parti de la
compacité de l'espace latent pour réduire le coût de résolution de l'ODE.

**C'est la thèse centrale de ce document** : NOIR démontre la partie "représentation resolution-
invariante" mais utilise le maillon le plus faible possible pour la partie "mapping" (régression MSE
déterministe), ce qui explique probablement son retard face à DDPM sur la traduction d'image. Notre
projet dispose déjà, indépendamment, d'un flow matching multi-marginal validé et compétitif
(nRMSE 0.4288, meilleur que UNet 0.4741 — voir `results/mmfm/comparison_20260801_final/manifest.md`).
Combiner les deux — **INR resolution-invariant comme représentation + flow matching multi-marginal
comme mapping génératif** — n'est pas une extrapolation hasardeuse mais la composition de deux
briques individuellement prouvées, chacune dans la littérature qui lui est propre.

## 3. Tableau comparatif synthétique

| Référence | Conditionnement | Activation | Génératif ? | Coût de fitting | Résultat clé pertinent |
|---|---|---|---|---|---|
| NOIR (2026) | Hypernetwork + meta-learning, latent global | SIREN | Non (régression MSE) | K=5-10 pas/volume | ε-ReNO validé ; perd face à DDPM sur traduction |
| MedFuncta (2025/26) | Idem NOIR, `ω` variable, supervision creuse | SIREN (variante) | Non | Optimisé pour l'échelle | Scalable à >500k volumes, poids publics |
| 3D MTransINR (2025) | Modulation locale (3D U-Net) | — | Adversarial (GAN) | — | Bat Pix2Pix 3D sur ProstateX/BraTS |
| Local INR MRI Translation (2023) | Local (précurseur) | — | — | — | À approfondir (Phase 0 suite) |
| HULFSynth (2025) | Physique (facteur de contraste), pas de latent appris | WIRE | Non (physique) | Optimisation par volume | Fonctionne zero-shot, hors distribution incluse |
| **Notre proposition** | Hypernetwork + meta-learning, latent global | SIREN (WIRE non testé) | **Oui — flow matching multi-marginal** | Précompute one-shot (cache) | Validé (Phase 3, 2026-08-02) : nRMSE 0.4566, entre vectorisé 0.4288 et UNet 0.4741 — meilleur sur les cibles 5T/7T, en retard sur SSIM/LPIPS (backbone encore sous-convergé, voir `results/mmfm/comparison_20260801_final/manifest.md`) |

## 4. Design : intégration dans `ArchAdapter`

Voir `docs/MMFM_ARCHITECTURE.md` pour le contrat `ArchAdapter` complet. Mapping proposé pour
`src/cfm/arch_inr.py` :

- **Backbone INR** (`src/cfm/inr_backbone.py`) : SIREN + hypernetwork + meta-learning (Algorithmes
  1/2 de NOIR), un seul backbone partagé entraîné sur tout `retro_train` — remplace MedVAE comme
  dépendance frozen partagée pour cette variante.
- **`prep_latent`** : fitting meta-appris (boucle interne K pas) → `z`. Coût dominant mais one-shot
  par volume, **précalculable et cachable** exactement comme `precompute_mmfm_latents.py`/
  `precompute_unet_latents.py` le font déjà pour MedVAE — même pattern, nouveau
  `precompute_inr_latents.py`.
- **`restore_latent`** : évalue `φ(x; θ, M_ψ(z))` sur la grille de coordonnées voulue — élimine le
  besoin de `_pad_to_multiple`/`center_crop_or_pad_np` puisque l'évaluation se fait à n'importe quelle
  résolution/grille directement.
- **`make_model_fn`** : `z` est un vecteur plat → **`arch_vector.build_vector_mmfm` /
  `VectorMMFM` est directement réutilisable** comme modèle de flow. Le sampler multi-marginal, le
  couplage OT-CFM réel, les régularisateurs cycle/edge-consistency et l'EMA de `mmfm_core.py` restent
  identiques à l'existant sans aucune modification.
- **`checkpoint_key_remap`** : identité.

## 5. Ordre de test des architectures candidates

1. **Candidat 1 (prioritaire)** — SIREN + hypernetwork + meta-learning (style NOIR/MedFuncta), latent
   global, `VectorMMFM` comme flow. Le plus documenté, code public disponible pour les deux papiers de
   référence (NOIR : https://github.com/Sidaty1/NOIR ; MedFuncta :
   https://github.com/pfriedri/medfuncta), risque le plus faible.
2. **Candidat 2** — remplacer SIREN par WIRE (Gabor wavelet) dans le même backbone, si le Candidat 1
   est instable ou trop lent à fitter/converger. Swap d'activation, pas de nouvelle architecture.
3. **Candidat 3 (optionnel, stretch)** — conditionnement local façon 3D MTransINR (modulation
   voxel-wise générée par un petit réseau conv), si le latent global perd trop de détail anatomique
   fin. Coût d'implémentation nettement plus élevé (le flow ne peut plus être `VectorMMFM` seul) — à
   ne déclencher que si 1 et 2 plafonnent.

## 6. État de l'art au niveau code (2026-08-03)

Après un premier round d'entraînement production montrant un écart de fidélité important avec MedVAE
(SSIM auto-reconstruction ~0.68 vs ~0.91 — voir `results/mmfm/comparison_20260801_final/
inr_reconstruction_capacity.csv`), inspection directe du code source des deux implémentations de
référence les plus proches — pas seulement de leurs papiers — pour identifier d'éventuels écarts
d'implémentation, pas seulement d'hyperparamètres.

### 6.1 NOIR (github.com/Sidaty1/NOIR) — trois écarts trouvés et corrigés

Comparaison ligne-à-ligne de `noir/inrs.py` (`ModulatedSiren`, `LatentToModulation`) et
`noir/metalearning.py` (`graph_inner_loop_step`) avec `src/cfm/inr_backbone.py` :

1. **`omega_0` non appliqué au shift.** Référence : `x = scale*linear(x)+shift; x = sin(w0*x)`,
   c.-à-d. `sin(w0·(Wx+b+shift))`. Notre code faisait `sin(w0·(Wx+b)) + shift` — le shift échappait à
   la mise à l'échelle par `w0` (=30), rendant son effet sur la phase du sinus **~30x plus faible**
   que prévu pour une même sortie brute du hypernetwork. Corrigé dans `SineLayer.forward`.
2. **Gradient de la boucle interne dilué par le batch.** Référence :
   `loss = ((recon-target)**2).mean() * batch_size` avant `torch.autograd.grad(loss, modulations, ...)`
   — cette multiplication annule la dilution par `1/batch_size` qu'introduit `.mean()` sur un tenseur
   `(batch_size, ...)`, pour que `z` de chaque élément du batch soit ajusté comme s'il était fitté
   indépendamment. Notre `fit_latent` n'avait pas cette correction (dilution 1/B, B=2 en
   méta-entraînement). Corrigé (`loss * b` avant le calcul du gradient de `z`).
3. **Hypernetwork : pas d'amortissement délibéré.** `LatentToModulation` utilise l'init PyTorch par
   défaut, sans réduction d'échelle sur la dernière couche. Notre implémentation appliquait un
   `×0.01` sur les poids de sortie (idée empruntée à la littérature générale des INR modulés, pas à
   NOIR) — combiné au bug (1), cela rendait le signal de modulation initial extrêmement faible.
   Corrigé (suppression de l'amortissement).

Un test smoke combinant les trois correctifs a montré une perte de méta-entraînement ~35 % plus basse
et le meilleur nRMSE observé à ce jour (voir mémoire projet pour le détail chiffré) — signal nettement
plus net qu'un simple changement d'hyperparamètre (`hyper_hidden_dim`) testé isolément.

### 6.2 MedFuncta (github.com/pfriedri/medfuncta) — confirmation + orientations alternatives

MedFuncta est le candidat le plus directement comparable : même laboratoire que NOIR, mais entraîné
sur des données médicales 3D à l'échelle (dont de l'IRM cérébrale, BraTS). Son
`models/layers.py::LatentModulatedSIRENLayer` confirme indépendamment le point 6.1.1 ci-dessus
(`x = scale*linear(x)+shift; x = sin(w0*x)`) — deuxième implémentation de référence, non liée à NOIR,
qui valide le correctif.

Sa config 3D (`configs/experiments/3d_imgs/default.yaml`) révèle plusieurs choix structurellement
différents des nôtres, dont certains restent à évaluer si les correctifs de la section 7.1 ne
suffisent pas à combler l'écart avec MedVAE :

| Paramètre | MedFuncta (3D, BraTS) | Notre config actuelle |
|---|---|---|
| `num_layers` (SIREN) | **15** | 6 |
| `latent_modulation_dim` | 8192 | 4096 |
| `hidden_dim` | 256 | 256 |
| `inner_steps` (train / test) | **10 / 20** | 10 / 20 (déjà aligné, validé indépendamment) |
| `max_iter` | **250 000** | 50 000 |
| Résolution du volume fitté | **32³** (BraTS downsamplé) | 96×112×96 (~31x plus de voxels) |
| Mécanisme de modulation | **Un `nn.Linear(latent_dim, hidden_dim)` DIRECT par couche**, sans hypernetwork partagé ni goulot d'étranglement caché | Un hypernetwork partagé unique (`Linear→SiLU→Linear`, `hyper_hidden_dim` caché) diffusant vers toutes les couches |
| Fréquence SIREN (`w0`) | **Variable par couche** (schedule linéaire 20→300 sur la profondeur, `common/w0_utils.py`) — basses fréquences dans les premières couches, hautes fréquences (donc plus de détail fin représentable) dans les dernières | Fixe (`omega_0=omega_hidden=30`) pour toutes les couches |

**Deux orientations concrètes, non encore testées, si les correctifs de 6.1 seuls ne suffisent pas :**
- **Modulation directe par couche, sans hypernetwork partagé** — remplacer notre
  `Hypernetwork` (bottleneck partagé `hyper_hidden_dim`) par un `nn.Linear(latent_dim, hidden_dim)`
  indépendant par couche modulée, éliminant complètement l'hypothèse de goulot d'étranglement
  (celle qui a motivé, sans succès isolé, l'essai `hyper_hidden_dim=512` — voir mémoire projet)
  puisqu'il n'y a alors plus de compression intermédiaire du tout.
- **Fréquence `w0` variable par profondeur** — vise directement le symptôme observé (flou / SSIM
  faible malgré un nRMSE correct) : une fréquence croissante avec la profondeur du réseau permettrait
  de représenter des détails plus fins dans les couches finales sans sacrifier la stabilité des
  couches précoces (basse fréquence). Non testé — nécessiterait de généraliser `ModulatedSIREN` pour
  accepter un `omega_0` par couche plutôt qu'un scalaire unique.

Also worth noting: MedFuncta entraîne 5x plus longtemps (250k vs 50k itérations) sur un volume
**31x plus petit** (32³ vs 96×112×96) — leur ratio itérations/complexité-de-la-tâche est très
supérieur au nôtre, ce qui reste une explication plausible et non exclusive du reste de l'écart avec
MedVAE, indépendamment des choix architecturaux ci-dessus.

### 6.3 SIREN original (github.com/vsitzmann/siren) — alternative écartée

Le papier SIREN original (Sitzmann et al.) propose lui aussi une variante hypernetwork
(`meta_modules.py::HyperNetwork`), mais d'un genre différent : elle prédit l'intégralité des **poids**
du réseau cible (pas seulement une modulation shift/scale), avec une tête de hypernetwork séparée par
paramètre. Plus lourd et plus complexe que l'approche modulation (NOIR/MedFuncta/notre implémentation)
— confirme que notre choix de rester sur une modulation shift-only était le bon compromis
capacité/simplicité, pas une simplification à corriger.

## 7. Références principales

1. El Hadramy, S., Haouchine, N., Wehrli, M., Cattin, P. C. *NOIR: Neural Operator mapping for
   Implicit Representations.* arXiv:2603.13118 (2026). Univ. Basel (Dept. Biomedical Engineering) +
   Harvard Medical School / Brigham and Women's Hospital. Code : https://github.com/Sidaty1/NOIR
   (implémentation officielle — voir section 7 pour la comparaison ligne-à-ligne avec notre code).
2. Friedrich, P. et al. *MedFuncta: A Unified Framework for Learning Efficient Medical Neural Fields.*
   arXiv:2502.14401 (2025), MIDL 2026 oral. Code : https://github.com/pfriedri/medfuncta.
3. Sitzmann, V. et al. *Implicit Neural Representations with Periodic Activation Functions (SIREN).*
   arXiv:2006.09661 (2020). Code : https://github.com/vsitzmann/siren.
4. Sitzmann, V. et al. *MetaSDF: Meta-learning Signed Distance Functions.* (2020).
5. Park, J. et al. *DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation.* (2019).
6. Ha, D. et al. *HyperNetworks.* arXiv:1609.09106 (2016).
7. Saragadam, V. et al. *WIRE: Wavelet Implicit Neural Representations.* CVPR 2023, arXiv:2301.05187.
8. Tancik, M. et al. *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.* (2020).
9. Müller, T. et al. *Instant Neural Graphics Primitives with a Multiresolution Hash Encoding.* (2022).
10. Bartolucci, F. et al. *Representation Equivalent Neural Operators: A Framework for Alias-free Operator Learning.* NeurIPS 2023.
11. Kovachki, N. et al. *Neural Operator: Learning Maps Between Function Spaces.* arXiv:2108.08481 (2021).
12. Dupont, E. et al. *From Data to Functa: Your Data Point is a Function and You Can Treat It Like One.* (2022).
13. Dupont, E. et al. *COIN++: Neural Compression Across Modalities.* (2022).
14. HULFSynth. *An INR based Super-Resolution and Ultra Low-Field MRI Synthesis via Contrast Factor Estimation.* arXiv:2511.14897 (2025).
15. *3D MTransINR: a 3D Modality Translation Model Based on Implicit Neural Representations.* CIABiomed 2025 (Springer LNCS).
16. *Local Implicit Neural Representations for Multi-Sequence MRI Translation.* arXiv:2302.01031 (2023).
17. Molaei, A. et al. *Implicit Neural Representation in Medical Imaging: A Comparative Survey.* ICCV Workshops 2023.
18. Ho, J. et al. *Denoising Diffusion Probabilistic Models (DDPM).* (2020) — baseline de référence battant NOIR sur la traduction fastMRI, contexte pour la section 2.5.

*Awesome-list utile pour approfondir : [xmindflow/Awesome-Implicit-Neural-Representations-in-Medical-imaging](https://github.com/xmindflow/Awesome-Implicit-Neural-Representations-in-Medical-imaging).*
