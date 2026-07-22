# MMFM v2 Vectorized — Plan d'implementation

Ce document decrit le plan d'implementation d'une version **MMFM vectorisee v2** avec:
- **classe discrete** = modalite (T1W, T2W, T2FLAIR)
- **champ continu** = temps global sur [0,1]

L'objectif est d'aligner la formulation de la version vectorisee avec la logique multi-marginale deja utilisee en UNet, tout en conservant un backbone MLP vectoriel.

---

## 1. Objectifs fonctionnels

1. Remplacer le conditionnement "15 classes (modalite x champ)" par:
   - classe = modalite (3 classes)
   - champ = temps continu
2. Conserver l'API d'entrainement/inference existante autant que possible.
3. Preserver la compatibilite de la version v1 (baseline historique) via un mode explicite v2.
4. Permettre une comparaison propre:
   - v1 (discret domaine)
   - v2 (continu champ)
   - UNet multi-marginal

---

## 2. Specification de la formulation v2

### 2.1 Variables de conditionnement

- **Classe conditionnante**: `mod_idx in {0,1,2}`
- **Temps global du champ**:

\[
 t = \frac{\text{field\_idx}}{N_{fields}-1}
\]

Avec 5 champs `[0.1T, 1.5T, 3T, 5T, 7T]`:
- `0.1T -> 0.00`
- `1.5T -> 0.25`
- `3T   -> 0.50`
- `5T   -> 0.75`
- `7T   -> 1.00`

### 2.2 Transition source -> cible

Pour une modalite donnee:
1. choisir un champ source `fi`
2. choisir un champ cible `fj`
3. calculer `t_i`, `t_j`, `dt_field = t_j - t_i`

### 2.3 Cas identite

Si `fi == fj`:
- vitesse cible nulle
- utile pour contraindre la preservation anatomique

---

## 3. Plan d'implementation par phase

## Phase A — Cadrage et compatibilite

### Taches
1. Figer l'ordre des champs en config (invariant absolu).
2. Introduire un flag methode explicite (`mmfm3d_vectorized_v2`).
3. Definir les options de sampling v2:
   - `adjacent_only` (demarrage recommande: `true`)
   - `identity_prob`

### Livrables
- Convention de mapping champ->temps documentee
- Nommage clair des runs/checkpoints v2

---

## Phase B — Dataset et contrat de donnees

### Taches
1. Verifier que le dataset expose deja:
   - volume
   - `mod_idx`
   - `field_idx`
   - `class_idx` (legacy, optionnel)
2. Si necessaire, ajuster le retour pour garantir un contrat stable.
3. Ajouter une verification de coherence:
   - index modalite
   - index champ
   - cardinalite par classe/champ

### Compatibilite UNet
Cette phase doit **rester compatible** avec la version UNet (meme contrat de donnees).

### Livrables
- Contrat de donnees commun v2/UNet
- Test de coherence dataset

---

## Phase C — Modele vectorise v2

### Fichier cible
- `src/cfm/mmfm_vectorized.py`

### Taches
1. Conserver l'architecture MLP residuelle.
2. Changer la semantique du conditionnement:
   - `class_labels` = modalite (3 classes)
   - `timesteps` = temps global continu
3. Verifier la forme d'entree:
   - `concat(z_t_vec, z_src_vec, time_embed(t), class_embed(mod))`
4. Rendre `num_classes` configurable (attendu v2 = 3).

### Livrables
- Modele v2 capable de conditionnement mixte (discret + continu)

---

## Phase D — Entrainement v2

### Fichier cible
- `src/cfm/train_mmfm_3d.py`

### Taches
1. Ajouter un chemin logique dedie a `mmfm3d_vectorized_v2`.
2. Remplacer le sampling "classe source/cible" par:
   - choix modalite
   - choix transition champ source->cible
3. OT-CFM:
   - echantillonner `t_local`
   - projeter en `t_global = t_i + t_local * (t_j - t_i)`
   - re-echelle de la vitesse en temps global (`ut_global`).
4. Cas identite:
   - vitesse cible nulle
5. DDP:
   - synchroniser plan d'echantillonnage (modalite + transitions) entre ranks.

### Livrables
- Boucle d'entrainement v2 stable mono-GPU et DDP

---

## Phase E — Inference v2

### Fichiers cibles
- `src/cfm/train_mmfm_3d.py` (integrateur Euler vectoriel)
- `src/cfm/infer_mmfm_unified.py`

### Taches
1. Integrer de `t_start` vers `t_end` (pas uniquement [0,1] complet).
2. Supporter `dt` signe (montee/descente de champ).
3. Construire un `flow_spec` v2:
   - `contrast`
   - `t_start`
   - `t_end`
4. Preserver la voie legacy v1 via switch methode explicite.

### Livrables
- Inference v2 any-to-any (meme modalite, champ source->cible)

---

## Phase F — Configs, runs, documentation

### Taches
1. Creer une config dediee, par ex:
   - `configs/mmfm3d_medvae_multimodal_v2.yaml`
2. Parametres recommandes (demarrage):
   - `num_classes=3`
   - `adjacent_only=true`
   - `identity_prob` modere
3. Nommage run explicite v2 (`task_name`, `output_subdir`).
4. Ajouter section v2 dans la doc d'experiences.

### Livrables
- Config v2 reproductible
- Documentation utilisateur minimaliste (train + infer)

---

## Phase G — Validation et comparaison

### Tests techniques
1. Test unitaire mapping champ->temps
2. Test shapes modele v2
3. Test infer Euler `t_start -> t_end`
4. Smoke train court (quelques centaines d'iterations)

### Validation experimentale
1. Evaluer avec le pipeline unifie
2. Comparer:
   - v1 vs v2 (meme VAE, meme split)
   - v2 vs UNet multi-marginal
3. Verifier qualite sur transitions:
   - adjacentes
   - longue distance (ex: 0.1T -> 7T)

### Livrables
- Tableau de comparaison (nRMSE / SSIM / LPIPS)
- Conclusion sur gain de formulation v2

---

## 4. Criteres d'acceptation

1. `mmfm3d_vectorized_v2` s'entraine sans erreur en local.
2. Le mode DDP v2 fonctionne avec echantillonnage synchronise.
3. L'inference v2 genere des volumes valides pour toute paire de champs.
4. La version UNet n'est pas regressée.
5. Les artefacts v2 sont separes des artefacts v1 (runs/checkpoints/results).

---

## 5. Risques et mitigations

1. **Instabilite numerique** si `dt_field` faible
   - mitigation: cas identite explicite + clipping + monitoring gradient
2. **Confusion v1/v2** au niveau checkpoints
   - mitigation: naming strict + validation metadata checkpoint
3. **Regressions inference** (flow_spec legacy vs v2)
   - mitigation: switch methode explicite + tests smoke dedies

---

## 6. Strategie de rollout recommandee

1. Implementer v2 minimal (adjacent_only=true, identity_prob modere).
2. Valider techniquement (smoke train/infer).
3. Lancer premier run de reference v2.
4. Ouvrir ensuite les transitions full-pairs (adjacent_only=false) pour etude.

---

## 7. Checklist implementation

- [ ] Convention champ->temps gelee et documentee
- [ ] Mode methode `mmfm3d_vectorized_v2` ajoute
- [ ] Contrat dataset verifie (mod_idx, field_idx)
- [ ] Modele vectorise conditionne sur modalite + temps global
- [ ] Boucle train v2 avec sampling transitions champ
- [ ] Cas identite implemente et teste
- [ ] DDP synchro plan d'echantillonnage verifiee
- [ ] Inference v2 `t_start -> t_end` validee
- [ ] Config v2 creee
- [ ] Evaluation v1/v2/UNet executee
- [ ] Resultats documentes
