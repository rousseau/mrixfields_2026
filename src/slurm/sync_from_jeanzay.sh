#!/usr/bin/env bash
# =============================================================================
# sync_from_jeanzay.sh — Synchronisation des résultats depuis Jean Zay
#
# Découvre automatiquement les runs depuis configs/*.yaml (output_subdir),
# puis rapatrie pour chaque run :
#   - outputs/<subdir>/weights/        (checkpoints .pth)
#   - outputs/<subdir>/train_metrics.jsonl
#   - logs/*.{out,err}                 (logs SLURM)
#
# Principe de connexion (ControlMaster) :
#   Le script ouvre une connexion SSH maître au début du script.
#   Les 2 mots de passe (proxy + Jean Zay) sont demandés une seule fois.
#   Tous les scp suivants réutilisent ce tunnel sans redemander de mot de passe.
#   Le tunnel est fermé proprement à la fin.
#
# Usage :
#   bash src/slurm/sync_from_jeanzay.sh [FILTRE]
#   bash src/slurm/sync_from_jeanzay.sh --all
#
#   FILTRE : sous-chaîne du nom de run (ex: "mmfm3d_unet", "cfm3d")
#   --all  : sync tous les runs sans demander de confirmation
#
# Exemples :
#   bash src/slurm/sync_from_jeanzay.sh mmfm3d_unet   # un run spécifique
#   bash src/slurm/sync_from_jeanzay.sh cfm3d         # tous les cfm3d
#   bash src/slurm/sync_from_jeanzay.sh --all         # tout sans confirmation
#
# Configuration requise :
#   Copier src/slurm/sync_from_jeanzay.sh.env.example → .sync_env
#   et y renseigner les variables. Ce fichier est ignoré par git.
# =============================================================================
set -euo pipefail

FILTER="${1:-}"
FORCE_ALL=0
if [[ "$FILTER" == "--all" ]]; then
    FORCE_ALL=1
    FILTER=""
fi

# ── Résolution des chemins ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Chargement de .sync_env ──────────────────────────────────────────────────
ENV_FILE="$PROJECT_ROOT/.sync_env"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

# Vérification des variables obligatoires
_check_var() {
    local var_name="$1"
    if [[ -z "${!var_name:-}" ]]; then
        echo "[ERREUR] Variable '$var_name' non définie."
        echo ""
        echo "Créer le fichier .sync_env à la racine du projet :"
        echo "  cp src/slurm/sync_from_jeanzay.sh.env.example .sync_env"
        echo "  # Puis éditer .sync_env avec vos identifiants"
        exit 1
    fi
}

_check_var JEANZAY_USER
_check_var JEANZAY_HOST
_check_var PROXY_USER
_check_var PROXY_HOST
_check_var REMOTE_BASE

# ── Découverte automatique des runs depuis configs/*.yaml ───────────────────
mapfile -t ALL_SUBDIRS < <(python3 - <<'PYEOF'
import yaml, glob, sys

configs_dir = "configs"
subdirs = set()
for path in sorted(glob.glob(f"{configs_dir}/*.yaml")):
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            continue
        subdir = cfg.get("data", {}).get("output_subdir", "")
        if subdir and "{{" not in subdir:  # ignorer les templates non résolus
            subdirs.add(subdir)
    except Exception:
        pass  # ignorer les configs malformées

for s in sorted(subdirs):
    print(s)
PYEOF
)

if [[ ${#ALL_SUBDIRS[@]} -eq 0 ]]; then
    echo "[WARN] Aucun output_subdir trouvé dans configs/*.yaml"
    exit 0
fi

# ── Filtrage ─────────────────────────────────────────────────────────────────
if [[ -n "$FILTER" ]]; then
    mapfile -t SUBDIRS < <(printf '%s\n' "${ALL_SUBDIRS[@]}" | grep -i "$FILTER" || true)
    if [[ ${#SUBDIRS[@]} -eq 0 ]]; then
        echo "[ERREUR] Aucun run ne correspond au filtre '$FILTER'"
        echo ""
        echo "Runs disponibles (${#ALL_SUBDIRS[@]}) — noms utilisables comme filtre :"
        for _s in "${ALL_SUBDIRS[@]}"; do
            printf '  %-45s  (%s)\n' "$(basename "$_s")" "$_s"
        done
        exit 1
    fi
else
    SUBDIRS=("${ALL_SUBDIRS[@]}")
fi

# ── Affichage du plan ────────────────────────────────────────────────────────
echo "======================================================================="
echo " sync_from_jeanzay.sh — Synchronisation depuis Jean Zay"
echo "======================================================================="
echo "  Jean Zay  : ${JEANZAY_USER}@${JEANZAY_HOST}"
echo "  Proxy     : ${PROXY_USER}@${PROXY_HOST}"
echo "  Remote    : ${REMOTE_BASE}"
if [[ -n "$FILTER" ]]; then
    echo "  Filtre    : $FILTER"
fi
echo "  Runs      : ${#SUBDIRS[@]}"
printf '    - %s\n' "${SUBDIRS[@]}"
echo "======================================================================="
echo ""

# ── Confirmation si aucun filtre et pas de --all ─────────────────────────────
if [[ -z "$FILTER" && $FORCE_ALL -eq 0 ]]; then
    echo "[ATTENTION] Aucun filtre — ${#SUBDIRS[@]} runs seront tentés."
    echo "Cela peut prendre plusieurs minutes."
    echo ""
    read -r -p "Continuer ? [o/N] " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[oOyY]$ ]]; then
        echo "Annulé."
        echo ""
        echo "Conseil : utiliser un filtre pour cibler un run :"
        echo "  bash src/slurm/sync_from_jeanzay.sh mmfm3d_unet"
        exit 0
    fi
    echo ""
fi

# ── Ouverture du tunnel SSH maître (ControlMaster) ───────────────────────────
# Les 2 mots de passe (proxy + Jean Zay) sont demandés ici, une seule fois.
# Tous les scp suivants réutilisent ce tunnel sans redemander de mot de passe.
SOCK="/tmp/jz_sync_$$.sock"

_close_tunnel() {
    ssh -S "$SOCK" -O exit "${JEANZAY_USER}@${JEANZAY_HOST}" 2>/dev/null || true
    rm -f "$SOCK"
}
trap _close_tunnel EXIT

PROXY_CMD="ssh -i ~/.ssh/id_mrixfields -o StrictHostKeyChecking=no ${PROXY_USER}@${PROXY_HOST} nc %h %p"

echo "Ouverture du tunnel SSH (2 mots de passe demandés) ..."
echo ""
ssh -fNM \
    -S "$SOCK" \
    -o "ProxyCommand=${PROXY_CMD}" \
    -o "StrictHostKeyChecking=no" \
    -o "ConnectTimeout=30" \
    -o "ServerAliveInterval=60" \
    "${JEANZAY_USER}@${JEANZAY_HOST}"

echo ""
echo "Tunnel ouvert. Début de la synchronisation."
echo ""

# Options scp réutilisant le tunnel maître
SCP_CTRL="-o ControlPath=${SOCK} -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=15"

# ── Fonction scp via tunnel ──────────────────────────────────────────────────
_scp() {
    local remote_path="$1"
    local local_dest="$2"
    # shellcheck disable=SC2086
    scp -rp $SCP_CTRL \
        "${JEANZAY_USER}@${JEANZAY_HOST}:${remote_path}" \
        "${local_dest}" 2>/dev/null
}

# ── Synchronisation des runs ─────────────────────────────────────────────────
N_OK=0
N_FAIL=0

for SUBDIR in "${SUBDIRS[@]}"; do
    RUN_NAME=$(basename "$SUBDIR")
    LOCAL_DIR="$PROJECT_ROOT/outputs/$SUBDIR"
    REMOTE_DIR="$REMOTE_BASE/outputs/$SUBDIR"

    echo "── $RUN_NAME"

    mkdir -p "$LOCAL_DIR/weights"

    # 1. Checkpoints
    echo -n "   weights/   ... "
    if _scp "${REMOTE_DIR}/weights/" "${LOCAL_DIR}/"; then
        N_NEW=$(find "$LOCAL_DIR/weights" -name "*.pth" | wc -l)
        echo "OK  ($N_NEW fichiers .pth locaux)"
        N_OK=$((N_OK + 1))
    else
        echo "SKIP (run absent ou erreur réseau)"
        N_FAIL=$((N_FAIL + 1))
    fi

    # 2. train_metrics.jsonl
    echo -n "   metrics    ... "
    if _scp "${REMOTE_DIR}/train_metrics.jsonl" "${LOCAL_DIR}/train_metrics.jsonl"; then
        NLINES=$(wc -l < "$LOCAL_DIR/train_metrics.jsonl" 2>/dev/null || echo "0")
        echo "OK  ($NLINES entrées)"
    else
        echo "absent"
    fi

    echo ""
done

# ── Synchronisation des logs SLURM ──────────────────────────────────────────
echo "── Logs SLURM (logs/*.out + logs/*.err)"
mkdir -p "$PROJECT_ROOT/logs"
echo -n "   logs/      ... "
# shellcheck disable=SC2086
if scp -p $SCP_CTRL \
    "${JEANZAY_USER}@${JEANZAY_HOST}:${REMOTE_BASE}/logs/*.out" \
    "${JEANZAY_USER}@${JEANZAY_HOST}:${REMOTE_BASE}/logs/*.err" \
    "$PROJECT_ROOT/logs/" 2>/dev/null; then
    N_LOGS=$(find "$PROJECT_ROOT/logs" -name "*.out" -o -name "*.err" | wc -l)
    echo "OK  ($N_LOGS fichiers)"
else
    echo "SKIP (aucun log ou erreur réseau)"
fi

# ── État des jobs SLURM ──────────────────────────────────────────────────────
echo ""
echo "── État des jobs SLURM (squeue)"
# shellcheck disable=SC2086
ssh $SCP_CTRL "${JEANZAY_USER}@${JEANZAY_HOST}" \
    "squeue -u ${JEANZAY_USER} --format='%.10i %.20j %.8T %.12M %.5D %R' 2>/dev/null || echo '  (aucun job en cours)'"

# ── Résumé ───────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================="
echo " Sync terminé : $N_OK runs copiés, $N_FAIL ignorés (absents/erreur)"
echo ""
echo "Visualiser les courbes d'apprentissage :"
echo "  python src/analysis/plot_training_curves.py"
echo "  # → results/training_curves.png"
echo "======================================================================="
