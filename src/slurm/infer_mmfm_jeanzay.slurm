#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Script SLURM — Inférence MMFM-UNet multi-GPU sur Jean Zay (4×H100 SXM5 80GB)
#
# Génère les prédictions Task 3 (any-to-any) pour un split donné
# (Validating_prospective ou Testing_prospective) en parallélisant les 25 paires
# de champs sur 4 GPUs.
#
# Lancement :
#   sbatch src/slurm/infer_mmfm_jeanzay.slurm \
#       configs/mmfm3d_multimarginal_medvae_run1.yaml \
#       outputs/cfm3d/runs/mmfm3d_multimarginal_medvae_run1/weights/model_final.pth \
#       outputs/submission_candidates/mmfm_multimarginal_val \
#       Validating_prospective
#
# Prérequis (via setup_jeanzay.sh) :
#   $WORK/MRIX/mrixfields_2026/   ← ce repo (à jour)
#   $WORK/MRIX/data/              ← données NIfTI
#   $WORK/MRIX/challenge/         ← code officiel du challenge (pip install)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=mmfm_infer
#SBATCH -A crp@h100
#SBATCH -C h100

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=24:00:00

#SBATCH --output=logs/mmfm_infer_%x_%j.out
#SBATCH --error=logs/mmfm_infer_%x_%j.err

set -e

CONFIG="${1}"
CHECKPOINT="${2}"
OUTPUT_DIR="${3}"
SPLIT="${4:-Validating_prospective}"
MODALITIES="${5:-T1W T2W T2FLAIR}"

CODE_DIR="$WORK/MRIX/mrixfields_2026"
mkdir -p "$CODE_DIR/logs"

echo "======================================================================="
echo " Job ID      : $SLURM_JOB_ID"
echo " Nœud        : $SLURM_JOB_NODELIST"
echo " Split       : $SPLIT"
echo " Config      : $CONFIG"
echo " Checkpoint  : $CHECKPOINT"
echo " Output dir  : $OUTPUT_DIR"
echo " Modalités   : $MODALITIES"
echo "======================================================================="

# ─── Modules Jean Zay ────────────────────────────────────────────────────────
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

# ─── Dépendances utilisateur ─────────────────────────────────────────────────
export PYTHONUSERBASE="$WORK/.local"
export PATH="$WORK/.local/bin:$PATH"

cd "$CODE_DIR"

# ─── Vérifications ───────────────────────────────────────────────────────────
if [ ! -f "$CONFIG" ]; then
    echo "[ERREUR] Config introuvable : $CONFIG"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERREUR] Checkpoint introuvable : $CHECKPOINT"
    exit 1
fi

# Toutes les paires any-to-any (Task 3)
PAIRS=(
    "0.1T_to_0.1T"  "0.1T_to_1.5T"  "0.1T_to_3T"  "0.1T_to_5T"  "0.1T_to_7T"
    "1.5T_to_0.1T"  "1.5T_to_1.5T"  "1.5T_to_3T"  "1.5T_to_5T"  "1.5T_to_7T"
    "3T_to_0.1T"    "3T_to_1.5T"    "3T_to_3T"    "3T_to_5T"    "3T_to_7T"
    "5T_to_0.1T"    "5T_to_1.5T"    "5T_to_3T"    "5T_to_5T"    "5T_to_7T"
    "7T_to_0.1T"    "7T_to_1.5T"    "7T_to_3T"    "7T_to_5T"    "7T_to_7T"
)
N_PAIRS=${#PAIRS[@]}
N_GPUS=4

echo "GPU           : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "Nombre paires : $N_PAIRS"
echo "GPUs          : $N_GPUS"
echo "======================================================================="

# ─── Lancement de 4 processus indépendants, un par GPU ───────────────────────
PIDS=()
for GPU_ID in $(seq 0 $((N_GPUS - 1))); do
    # Répartition des paires sur les GPUs
    PAIRS_SUBSET=""
    for IDX in $(seq 0 $((N_PAIRS - 1))); do
        if [ $((IDX % N_GPUS)) -eq $GPU_ID ]; then
            PAIRS_SUBSET="${PAIRS_SUBSET},${PAIRS[$IDX]}"
        fi
    done
    # Supprimer la première virgule
    PAIRS_SUBSET="${PAIRS_SUBSET#,}"

    N_SUBSET=$(echo "$PAIRS_SUBSET" | tr ',' '\n' | wc -l)
    echo "[GPU $GPU_ID] $N_SUBSET paires : $PAIRS_SUBSET"

    export CUDA_VISIBLE_DEVICES=$GPU_ID
    PYTHONPATH=src python src/cfm/infer_mmfm_unified.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --output_dir "${OUTPUT_DIR}/gpu${GPU_ID}" \
        --split "$SPLIT" \
        --modalities $MODALITIES \
        --pairs "$PAIRS_SUBSET" \
        --env remote \
        --skip_existing \
        > "logs/mmfm_infer_gpu${GPU_ID}_${SLURM_JOB_ID}.out" 2>&1 &

    PIDS+=($!)
    unset CUDA_VISIBLE_DEVICES
    sleep 5
done

# ─── Attente de la fin de tous les processus ─────────────────────────────────
echo "======================================================================="
echo " Attente de la fin des 4 processus (PIDs: ${PIDS[*]})"
echo "======================================================================="

EXIT_CODES=()
for PID in "${PIDS[@]}"; do
    wait "$PID"
    EXIT_CODES+=($?)
done

# ─── Résumé ─────────────────────────────────────────────────────────────────
echo "======================================================================="
GPU_OK=0
GPU_FAIL=0
for i in "${!EXIT_CODES[@]}"; do
    if [ "${EXIT_CODES[$i]}" -eq 0 ]; then
        echo " GPU $i : OK"
        GPU_OK=$((GPU_OK + 1))
    else
        echo " GPU $i : ÉCHEC (exit ${EXIT_CODES[$i]})"
        GPU_FAIL=$((GPU_FAIL + 1))
    fi
done
echo "======================================================================="

# Sortie avec erreur si au moins un GPU a échoué
if [ "$GPU_FAIL" -gt 0 ]; then
    echo "[ERREUR] $GPU_FAIL / $N_GPUS processus ont échoué."
    exit 1
fi

# ─── Fusion des résultats GPU dans un seul dossier de soumission ─────────────
# infer_mmfm_unified.py crée un sous-dossier 'task3' ; on le fusionne ici.
echo "======================================================================="
echo " Fusion des prédictions GPU dans : ${OUTPUT_DIR}/task3"
echo "======================================================================="

mkdir -p "${OUTPUT_DIR}/task3"
for GPU_ID in $(seq 0 $((N_GPUS - 1))); do
    GPU_OUT="${OUTPUT_DIR}/gpu${GPU_ID}/task3"
    if [ -d "$GPU_OUT" ]; then
        find "$GPU_OUT" -type f -name "*.nii.gz" | while read -r f; do
            rel="${f#${GPU_OUT}/}"
            dest="${OUTPUT_DIR}/task3/${rel}"
            mkdir -p "$(dirname "$dest")"
            mv "$f" "$dest"
        done
    fi
done

# Comptage final
N_PRED=$(find "${OUTPUT_DIR}/task3" -name "*.nii.gz" | wc -l)
echo "Nombre total de prédictions : $N_PRED"
echo "Prédictions dans : ${OUTPUT_DIR}/task3"
