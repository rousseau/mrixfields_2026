#!/usr/bin/env bash
# =============================================================================
# sync_from_jeanzay.sh — Synchronisation des résultats depuis Jean Zay
#
# Découvre automatiquement les runs depuis configs/*.yaml (output_subdir),
# puis rapatrie pour chaque run :
#   - outputs/<subdir>/weights/        (checkpoints .pth)
#   - outputs/<subdir>/train_metrics.jsonl
#   - logs/<job_name>_*.{out,err}      (logs SLURM)
#
# Usage :
#   bash src/slurm/sync_from_jeanzay.sh [FILTRE]
#   bash src/slurm/sync_from_jeanzay.sh --all
#
#   FILTRE : sous-chaîne du nom de run (ex: "mmfm_unet", "cfm3d")
#   --all  : sync tous les runs sans demander de confirmation
#
# Exemples :
#   bash src/slurm/sync_from_jeanzay.sh mmfm_unet    # seulement mmfm_unet
#   bash src/slurm/sync_from_jeanzay.sh cfm3d        # tous les cfm3d
#   bash src/slurm/sync_from_jeanzay.sh --all        # tout syncer sans confirmation
#
# Configuration requise :
#   Copier src/slurm/sync_from_jeanzay.sh.env.example → .sync_env
#   et y renseigner JEANZAY_USER, PROXY_USER, PROXY_HOST, REMOTE_BASE
#   Ce fichier est ignoré par git.
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

# ── Configuration de connexion (requiert .sync_env) ──────────────────────────
ENV_FILE="$PROJECT_ROOT/.sync_env"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

# Vérification que les variables obligatoires sont définies
_check_var() {
    local var_name="$1"
    if [[ -z "${!var_name:-}" ]]; then
        echo "[ERREUR] Variable '$var_name' non définie."
        echo ""
        echo "Créer le fichier .sync_env à la racine du projet :"
        echo "  cp src/slurm/sync_from_jeanzay.sh.env.example .sync_env"
        echo "  # Puis éditer .sync_env avec vos identifiants"
        echo ""
        echo "Ce fichier est ignoré par git (.gitignore)."
        exit 1
    fi
}

_check_var JEANZAY_USER
_check_var JEANZAY_HOST
_check_var PROXY_USER
_check_var PROXY_HOST
_check_var REMOTE_BASE

SCP_OPTS="-o ProxyCommand=\"ssh ${PROXY_USER}@${PROXY_HOST} nc %h %p\" -o StrictHostKeyChecking=no -o ConnectTimeout=15"
REMOTE="${JEANZAY_USER}@${JEANZAY_HOST}"

# ── Découverte automatique des runs depuis configs/*.yaml ───────────────────
mapfile -t ALL_SUBDIRS < <(python3 - <<'PYEOF'
import yaml, glob, os, re, sys

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
        for _s in "${ALL_SUBDIRS[@]}"; do printf '  %-45s  (%s)\n' "$(basename "$_s")" "$_s"; done
        exit 1
    fi
else
    SUBDIRS=("${ALL_SUBDIRS[@]}")
fi

# ── Affichage du plan ────────────────────────────────────────────────────────
echo "======================================================================="
echo " sync_from_jeanzay.sh — Synchronisation depuis Jean Zay"
echo "======================================================================="
echo "  Remote  : ${REMOTE}:${REMOTE_BASE}"
echo "  Proxy   : ${PROXY_USER}@${PROXY_HOST}"
if [[ -n "$FILTER" ]]; then
    echo "  Filtre  : $FILTER"
fi
echo "  Runs    : ${#SUBDIRS[@]}"
printf '    - %s\n' "${SUBDIRS[@]}"
echo "======================================================================="
echo ""

# ── Confirmation si aucun filtre et pas de --all ─────────────────────────────
if [[ -z "$FILTER" && $FORCE_ALL -eq 0 ]]; then
    echo "[ATTENTION] Aucun filtre spécifié — ${#SUBDIRS[@]} runs seront tentés."
    echo "Cela peut prendre plusieurs minutes (une tentative SSH par run)."
    echo ""
    read -r -p "Continuer ? [o/N] " CONFIRM
    if [[ ! "$CONFIRM" =~ ^[oOyY]$ ]]; then
        echo "Annulé."
        echo ""
        echo "Conseil : utiliser un filtre pour cibler un run spécifique :"
        echo "  bash src/slurm/sync_from_jeanzay.sh mmfm_unet"
        exit 0
    fi
    echo ""
fi

# ── Fonction scp avec gestion d'erreur silencieuse ──────────────────────────
_scp() {
    local remote_path="$1"
    local local_dest="$2"
    eval scp -rp $SCP_OPTS \
        "${REMOTE}:${remote_path}" \
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
if eval scp -p $SCP_OPTS \
    "${REMOTE}:${REMOTE_BASE}/logs/*.out" \
    "${REMOTE}:${REMOTE_BASE}/logs/*.err" \
    "$PROJECT_ROOT/logs/" 2>/dev/null; then
    N_LOGS=$(find "$PROJECT_ROOT/logs" -name "*.out" -o -name "*.err" | wc -l)
    echo "OK  ($N_LOGS fichiers)"
else
    echo "SKIP (aucun log ou erreur réseau)"
fi

# ── Résumé ───────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================="
echo " Sync terminé : $N_OK runs copiés, $N_FAIL ignorés (absents/erreur)"
echo ""
echo "Visualiser les courbes d'apprentissage :"
echo "  python src/analysis/plot_training_curves.py"
echo "  # → results/training_curves.png"
echo "======================================================================="
