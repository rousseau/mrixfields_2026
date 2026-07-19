#!/usr/bin/env bash
# Synchronise le code, les configs et les poids nécessaires vers Jean Zay.
# Utilise l'alias SSH "jeanzay" configuré dans ~/.ssh/config
# (proxy Telecom Bretagne + identifiant ulq73oz). Voir src/slurm/setup_ssh_keys.sh.
#
# Usage :
#   bash src/slurm/sync_to_jeanzay.sh
#
# Prérequis :
#   - ~/.ssh/config contient un Host "jeanzay" (ProxyCommand + IdentityFile)
#   - La clé ~/.ssh/id_mrixfields est chargée (ou via ssh-agent)

set -e

JZ_HOST="jeanzay"
JZ_WORK='$(ssh jeanzay echo $WORK)'
JZ_DIR='$WORK/MRIX/mrixfields_2026'

echo "=== Sync vers Jean Zay (alias SSH : ${JZ_HOST}) ==="
echo "Répertoire cible : \$WORK/MRIX/mrixfields_2026"

# 1. Code source, configs, scripts SLURM (sans outputs/logs ni git)
rsync -avz --delete \
  --exclude='.git' \
  --exclude='outputs' \
  --exclude='logs' \
  --exclude='results/benchmark_vae/visuals' \
  --exclude='results/benchmark_vae/comparisons' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.egg-info' \
  --exclude='.pytest_cache' \
  --exclude='.sync_env' \
  ./ "${JZ_HOST}:MRIX/mrixfields_2026/"

# 2. Poids du modèle MMFM-UNet multi-marginal (run 1) — 2.7 GB
echo "=== Sync poids MMFM-UNet run1 ==="
rsync -avz --progress \
  outputs/cfm3d/runs/mmfm3d_multimarginal_medvae_run1/weights/ \
  "${JZ_HOST}:MRIX/mrixfields_2026/outputs/cfm3d/runs/mmfm3d_multimarginal_medvae_run1/weights/"

# 3. Poids MedVAE fine-tuné (requis par load_vae)
echo "=== Sync poids MedVAE fine-tuné ==="
rsync -avz --progress \
  outputs/medvae/runs/medvae_finetune_all/weights/ \
  "${JZ_HOST}:MRIX/mrixfields_2026/outputs/medvae/runs/medvae_finetune_all/weights/"

cat <<EOF

=== Sync terminée ===

Pour se connecter :  ssh jeanzay

Sur Jean Zay, lancer l'inférence Task 3 :
  cd \$WORK/MRIX/mrixfields_2026
  sbatch src/slurm/infer_mmfm_jeanzay.slurm \\
    configs/mmfm3d_multimarginal_medvae_run1.yaml \\
    outputs/cfm3d/runs/mmfm3d_multimarginal_medvae_run1/weights/model_final.pth \\
    outputs/submission_candidates/mmfm_multimarginal_val \\
    Validating_prospective
EOF
