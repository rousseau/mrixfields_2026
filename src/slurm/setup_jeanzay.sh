#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Setup Jean Zay — première installation (nœud de login)
#
# À exécuter UNE SEULE FOIS depuis le nœud de login Jean Zay :
#   bash src/slurm/setup_jeanzay.sh
#
# Ce script :
#   1. Crée la structure de répertoires sous $WORK/MRIX/
#   2. Installe les dépendances Python dans $WORK/.local
#   3. Affiche les commandes rsync pour transférer les données et les .npz
# ─────────────────────────────────────────────────────────────────────────────

set -e

MRIX_ROOT="$WORK/MRIX"
PROJECT_DIR="$MRIX_ROOT/mrixfields_2026"
DATA_DIR="$MRIX_ROOT/data"
CHALLENGE_DIR="$MRIX_ROOT/challenge"
# Le package mrixfields est dans le sous-répertoire Baseline/
CHALLENGE_BASELINE="$CHALLENGE_DIR/Baseline"

echo "======================================================================="
echo " Setup Jean Zay — MRIxFields 2026"
echo " MRIX_ROOT : $MRIX_ROOT"
echo "======================================================================="

# ─── 1. Répertoires ───────────────────────────────────────────────────────────
mkdir -p "$DATA_DIR"
mkdir -p "$PROJECT_DIR/outputs/stargan2d/preprocessed"
mkdir -p "$PROJECT_DIR/outputs/cfm2d"
mkdir -p "$PROJECT_DIR/logs"
echo "[1/4] Répertoires créés"

# ─── 2. Cloner le repo du challenge (pour le package mrixfields) ──────────────
if [ ! -d "$CHALLENGE_DIR" ]; then
    echo ""
    echo "[2/4] Clonage du code officiel du challenge..."
    echo "      [ACTION REQUISE] Lancez manuellement :"
    echo "      git clone <URL_DU_CHALLENGE> $CHALLENGE_DIR"
elif [ ! -f "$CHALLENGE_BASELINE/setup.py" ]; then
    echo "[2/4] Challenge repo présent mais Baseline/ introuvable : $CHALLENGE_BASELINE"
    echo "      Vérifiez la structure du repo."
else
    echo "[2/4] Challenge repo OK : $CHALLENGE_BASELINE"
fi

# ─── 3. Installer les dépendances Python ─────────────────────────────────────
echo ""
echo "[3/4] Installation des dépendances Python..."

module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

export PYTHONUSERBASE="$WORK/.local"
export PATH="$WORK/.local/bin:$PATH"

pip install --user --quiet torchcfm pot torchdiffeq nibabel

# Package mrixfields
if [ -f "$CHALLENGE_BASELINE/setup.py" ]; then
    # --no-deps : évite tensorflow<2.16 incompatible avec Python 3.12
    # On n'utilise que mrixfields.data.transforms (aucune dépendance TF)
    pip install --user --quiet --no-deps "$CHALLENGE_BASELINE"
    echo "  → mrixfields installé (sans dépendances) depuis $CHALLENGE_BASELINE"
elif [ -d "$CHALLENGE_DIR" ]; then
    echo "  [ATTENTION] setup.py introuvable dans $CHALLENGE_BASELINE"
    echo "  Vérifiez que le repo challenge est bien structuré avec un Baseline/."
else
    echo "  [ATTENTION] mrixfields non installé (challenge repo manquant)"
    echo "  Relancez ce script après avoir cloné le challenge."
fi

echo "[3/4] Dépendances installées dans \$WORK/.local"

# ─── 4. Transfert des données ─────────────────────────────────────────────────
echo ""
echo "[4/4] Transfert des données (à lancer depuis la machine locale) :"
echo ""
echo "  # Données brutes NIfTI (structure challenge) :"
echo "  rsync -avP ~/Data/MRIxFields_20260414/ \\"
echo "    <login>@jean-zay.idris.fr:$DATA_DIR/"
echo ""
echo "  # Données préprocessées .npz (StarGAN + CFM partagent ces fichiers) :"
echo "  rsync -avP ~/Exp/mrixfields_2026/outputs/stargan2d/preprocessed/ \\"
echo "    <login>@jean-zay.idris.fr:$PROJECT_DIR/outputs/stargan2d/preprocessed/"
echo ""
echo "  # Optionnel — transférer des checkpoints StarGAN existants :"
echo "  rsync -avP ~/Exp/mrixfields_2026/outputs/stargan2d/runs/ \\"
echo "    <login>@jean-zay.idris.fr:$PROJECT_DIR/outputs/stargan2d/runs/"
echo ""
echo "======================================================================="
echo " Setup terminé."
echo ""
echo " Lancer un entraînement :"
echo "   cd $PROJECT_DIR"
echo "   sbatch src/slurm/cfm_jeanzay.slurm T1W"
echo "======================================================================="
