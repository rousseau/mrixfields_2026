#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Script SLURM — Benchmark 3 VAE sur images complètes (364x436x364)
#
# Compare AEKL, VQ-VAE, et MedVAE sur full-resolution volumes MRIxFields
# avec patch-based processing (112×128×80 patches, overlap 25%)
#
# Lancement :
#   sbatch src/slurm/benchmark_vae_jeanzay.slurm T1W 0.1T
#   sbatch src/slurm/benchmark_vae_jeanzay.slurm T2W 3T
#
# Arguments positionnels :
#   $1 : MODALITY  (T1W, T2W, T2FLAIR)
#   $2 : FIELD     (0.1T, 1.5T, 3T, 5T, 7T)
#
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=benchmark_vae
# SBATCH_ACCOUNT doit être défini dans l'environnement avant sbatch :
#   export SBATCH_ACCOUNT=<projet>@h100   (ex: crp@h100)
# SLURM lit automatiquement cette variable — ne pas hardcoder ici.
#SBATCH -A crp@h100
#SBATCH -C h100

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=10:00:00

#SBATCH --output=logs/benchmark_vae_%x_%j.out
#SBATCH --error=logs/benchmark_vae_%x_%j.err

set -e

MODALITY="${1:-T1W}"
FIELD="${2:-0.1T}"

CODE_DIR="$WORK/MRIX/mrixfields_2026"
DATA_ROOT="$WORK/MRIX/data"
OUTPUT_DIR="$WORK/MRIX/mrixfields_2026/outputs/benchmark/runs"

mkdir -p "$CODE_DIR/logs"
mkdir -p "$OUTPUT_DIR"

echo "======================================================================="
echo " VAE Benchmark — Full-resolution volumes (364×436×364)"
echo "======================================================================="
echo " Job ID    : $SLURM_JOB_ID"
echo " Modality  : $MODALITY"
echo " Field     : $FIELD"
echo " Code      : $CODE_DIR"
echo " Data      : $DATA_ROOT"
echo " Output    : $OUTPUT_DIR"
echo "======================================================================="

# ─── Modules Jean Zay ────────────────────────────────────────────────────────
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

# ─── Dépendances utilisateur ─────────────────────────────────────────────────
export PYTHONUSERBASE="$WORK/.local"
export PATH="$WORK/.local/bin:$PATH"

_install_if_missing() {
    python3 -c "import $1" 2>/dev/null || pip install --user --quiet "$2"
}
_install_if_missing nibabel nibabel
_install_if_missing scipy scipy
_install_if_missing medvae medvae

# ─── Cache PyTorch ───────────────────────────────────────────────────────────
export TORCH_HOME="$WORK/.cache/torch"

# ─── Vérification des données ─────────────────────────────────────────────────
if [ ! -d "$DATA_ROOT" ]; then
    echo "[ERREUR] DATA_ROOT introuvable : $DATA_ROOT"
    exit 1
fi

N_VOLS=$(find "$DATA_ROOT/Training_retrospective/$MODALITY/$FIELD" -name "*.nii.gz" 2>/dev/null | wc -l)
echo "Volumes disponibles : $N_VOLS ($MODALITY/$FIELD)"
echo "GPU : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ─── Lancement benchmark ──────────────────────────────────────────────────────
cd "$CODE_DIR"

echo ""
echo "Démarrage benchmark..."
echo ""

python3 src/vae3d/benchmark_vae.py \
    --data-root "$DATA_ROOT" \
    --modality "$MODALITY" \
    --field "$FIELD" \
    --max-samples 3 \
    --aekl-ckpt  "outputs/vae3d/runs/vae3d_T1W/weights/model_best.pth" \
    --vqvae-ckpt "outputs/vqvae3d/runs/vqvae_full/weights/vqvae_final.pth" \
    --medvae-finetuned-ckpt "outputs/medvae/runs/medvae_T1W/weights/model_final.pth" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda

echo ""
echo "======================================================================="
echo " Benchmark terminé"
echo "======================================================================="
