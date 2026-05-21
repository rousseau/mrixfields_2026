# MMFM v1 vectorized baseline

Cette baseline implémente une version simple et fidèle à l’idée MMFM: le cœur
du modèle opère sur des vecteurs latents, tandis que MedVAE reste inchangé.

## Principe

Le pipeline est volontairement minimal:

1. volume 3D NIfTI brut
2. encodage MedVAE
3. aplatissement du latent en un vecteur unique
4. MMFM vectoriel conditionné sur le domaine cible
5. intégration du champ de vitesse dans l’espace vectoriel
6. remise en forme du vecteur latent
7. décodage MedVAE

La baseline conserve le découpage des domaines en 15 classes discrètes
`3 modalités x 5 champs`. Cela permet de comparer cette version avec le
prototype précédent sans changer la structure du dataset.

## Mapping exact des shapes

Pour la config courante, l’entrée est un volume MRI prétraité de forme:

```text
x : (B, 1, 128, 128, 80)
```

Après encodage MedVAE:

```text
z : (B, 1, 32, 32, 20)
```

La vectorisation transforme ce latent en:

```text
z_vec : (B, 20480)
```

avec `20480 = 1 x 32 x 32 x 20`.

Pendant l’entraînement CFM/MMFM:

```text
z_src_vec : (B, 20480)
z_t_vec   : (B, 20480)
t         : (B,) ou (B, 1)
y         : (B,)
v_t       : (B, 20480)
u_t       : (B, 20480)
```

Le modèle prédit un champ de vitesse dans le même espace vectoriel que le
latent aplati.

Après intégration:

```text
z_hat_vec : (B, 20480)
z_hat     : (B, 1, 32, 32, 20)
x_hat     : (B, 1, 128, 128, 80)
```

## Ce qui change par rapport au prototype précédent

Le prototype initial utilisait encore un `DiffusionModelUNet` 3D dans le latent.
Cette v1 remplace ce bloc par un MLP vectoriel:

- entrée = concaténation de `z_t_vec`, `z_src_vec`, embedding de temps et
  embedding de classe
- backbone = blocs résiduels MLP
- sortie = champ vectoriel de même dimension que le latent aplati

Le VAE n’est pas modifié, ni fine-tuné pendant cette baseline.

## Fichiers clés

- [src/cfm/train_mmfm_3d.py](../src/cfm/train_mmfm_3d.py)
- [src/cfm/mmfm_vectorized.py](../src/cfm/mmfm_vectorized.py)
- [configs/mmfm3d_medvae_multimodal.yaml](../configs/mmfm3d_medvae_multimodal.yaml)

## Lancement local

```bash
PYTHONPATH=src python src/cfm/train_mmfm_3d.py \
  --config configs/mmfm3d_medvae_multimodal.yaml \
  --env local
```

## Lancement Jean Zay 4xH100

Le script SLURM principal est [src/slurm/cfm_3d_jeanzay.slurm](../src/slurm/cfm_3d_jeanzay.slurm).
Pour la v1 vectorisée, il suffit d’utiliser la phase `mmfm`:

```bash
sbatch src/slurm/cfm_3d_jeanzay.slurm mmfm T1W
```

## Smoke test local

Un test léger vérifie la vectorisation et la sortie du MLP:

```bash
PYTHONPATH=src python src/cfm/test_mmfm_v1_smoke.py
```
