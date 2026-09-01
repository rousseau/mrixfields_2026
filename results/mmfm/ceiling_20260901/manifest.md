# Plafond de représentation du MedVAE perceptuel (LPIPS)

**Date** : 2026-09-01
**Verdict** : **porte franchie, mais modestement** — le plafond passe de 0.1048 à
**0.0976** de nRMSE (−6.9 %), avec les 15 cellules qui s'améliorent sans
exception. Réel, mais pas un changement d'ordre de grandeur.

---

## Pourquoi cette mesure était nécessaire

Le critère d'arrêt du 2026-09-01 a fermé le côté flow : sept leviers, sept fois
rien au-delà du plancher de bruit. Ce qui reste ouvert est la **représentation**.

Le fine-tuning LPIPS avait été mesuré le 2026-08-25 et paraissait très supérieur
(SSIM 0.9727 contre 0.9152 à 1 mm). **Mais ces chiffres ne sont pas comparables
aux nôtres** : ils viennent d'une auto-reconstruction par **patches 64³** à 1 mm,
sans le rééchantillonnage final 1 mm → 0.5 mm ni les sujets d'évaluation. Adopter
sur cette base aurait été payer une journée sur une comparaison de protocoles
différents.

## Le protocole, identique à celui du 2026-08-27

Paires **identité** (`--pairs 0.1T_to_0.1T,...`) : `dt = 0`, donc `euler_integrate`
laisse le latent intact et le fait que le flow ait été entraîné sur les ANCIENS
latents est sans conséquence. Il reste encode → decode avec le VAE perceptuel, par
le **même chemin de code** que les prédictions — même normalisation, même tuilage,
même rééchantillonnage. 3 contrastes × 5 champs × 3 sujets = 45 volumes.

## Le résultat

| | pré-entraîné | **LPIPS** | écart |
|---|---|---|---|
| T1W | 0.1081 | 0.0999 | −0.0082 |
| T2W | 0.1228 | 0.1157 | −0.0071 |
| T2FLAIR | 0.0836 | 0.0772 | −0.0063 |
| **moyenne nRMSE** | **0.1048** | **0.0976** | **−0.0072** |
| **SSIM** | 0.9518 | **0.9579** | +0.0061 |

**Les 15 cellules s'améliorent**, de −2.4e-3 (T2W@0.1T) à −1.26e-2 (T1W@7T). La
cohérence parfaite du signe écarte le bruit ; l'amplitude reste modeste.

Détail par cellule dans `ceiling_lpips_{T1W,T2W,T2FLAIR}.csv`, référence
pré-entraînée dans `ceiling_vectorized_*.csv` (copiée de `../ceiling_20260827/`).

## Ce que cela borne

L'état actuel du vectorisé est **0.2143** en erreur structurelle pour un plafond
de **0.1048** : l'écart de 0.1095 est celui que le critère d'arrêt a déclaré
irréductible par les leviers de flow. Abaisser le plafond à 0.0976 ne rapporte,
**au mieux et si les erreurs s'additionnent**, que ces 0.0072 — soit ~3.4 % du
score structurel.

C'est au-dessus du plancher de bruit (0.002), donc mesurable. Ce n'est pas ce qui
fera franchir un palier.

## Note sur le contrôle d'équité

Le script émet « ATTENTION : les deux architectures ne partagent pas le même
encodage ». **C'est attendu ici** : ce contrôle existe pour vérifier que le
vectorisé et l'UNet, qui partagent MedVAE, ont bien le même plafond. On compare
ici deux encodeurs délibérément différents, et l'écart signalé (max 1.26e-2) est
précisément la quantité mesurée.

## Reproduction

```
PYTHONPATH=src python src/cfm/infer_mmfm_unified.py \
    --config configs/mmfm/ceiling_lpips.yaml \
    --checkpoint outputs/mmfm/vec_rbest/weights/model_final.pth \
    --output_dir outputs/mmfm/ceiling_lpips/predictions \
    --pairs 0.1T_to_0.1T,1.5T_to_1.5T,3T_to_3T,5T_to_5T,7T_to_7T \
    --modalities T1W T2W T2FLAIR --field_norm_stats configs/mmfm/field_norm_stats.json
PYTHONPATH=src python src/cfm/eval_representation_ceiling.py \
    --pred-root outputs/mmfm/ceiling_lpips/predictions/task3 --name lpips \
    --outdir results/mmfm/ceiling_20260901 --compare vectorized
```
