#!/usr/bin/env bash
# =============================================================================
# setup_ssh_keys.sh — Configuration clés SSH pour accès Jean Zay sans mot de passe
#
# Ce script guide la mise en place d'une clé SSH dédiée permettant d'accéder
# à Jean Zay via proxy sans saisir de mot de passe à chaque commande.
#
# Prérequis : avoir renseigné .sync_env (cp .env.example .sync_env)
#
# Usage :
#   bash src/slurm/setup_ssh_keys.sh
#
# Ce script effectue dans l'ordre :
#   1. Génération de la clé SSH dédiée ~/.ssh/id_mrixfields (si absente)
#   2. Dépôt de la clé sur le proxy (1 mot de passe proxy demandé)
#   3. Dépôt de la clé sur Jean Zay via proxy (1 mot de passe Jean Zay demandé)
#   4. Ajout du bloc Host dans ~/.ssh/config
#   5. Test de connexion sans mot de passe
#
# Après exécution : ssh jeanzay  →  connexion directe sans mot de passe
# =============================================================================
set -euo pipefail

# ── Résolution des chemins ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Chargement de .sync_env ──────────────────────────────────────────────────
ENV_FILE="$PROJECT_ROOT/.sync_env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "[ERREUR] Fichier .sync_env introuvable."
    echo ""
    echo "Créer le fichier .sync_env à la racine du projet :"
    echo "  cp src/slurm/sync_from_jeanzay.sh.env.example .sync_env"
    echo "  # Puis éditer .sync_env avec vos identifiants"
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

# Vérification variables obligatoires
for var in JEANZAY_USER JEANZAY_HOST PROXY_USER PROXY_HOST; do
    if [[ -z "${!var:-}" ]]; then
        echo "[ERREUR] Variable '$var' non définie dans .sync_env"
        exit 1
    fi
done

KEY_FILE="$HOME/.ssh/id_mrixfields"
SSH_CONFIG="$HOME/.ssh/config"

echo "======================================================================="
echo " setup_ssh_keys.sh — Configuration SSH pour Jean Zay"
echo "======================================================================="
echo "  Proxy    : ${PROXY_USER}@${PROXY_HOST}"
echo "  Jean Zay : ${JEANZAY_USER}@${JEANZAY_HOST}"
echo "  Clé SSH  : ${KEY_FILE}"
echo "======================================================================="
echo ""

# ── Étape 1 : Génération de la clé ──────────────────────────────────────────
echo "── Étape 1/4 : Clé SSH"
if [[ -f "$KEY_FILE" ]]; then
    echo "   Clé déjà présente : $KEY_FILE  [OK]"
else
    echo "   Génération de la clé ed25519 (sans passphrase)..."
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"
    ssh-keygen -t ed25519 -f "$KEY_FILE" -C "mrixfields_sync" -N ""
    echo "   Clé générée : $KEY_FILE  [OK]"
fi
echo ""

# ── Étape 2 : Dépôt sur le proxy ────────────────────────────────────────────
echo "── Étape 2/4 : Dépôt de la clé sur le proxy"
echo "   Commande : ssh-copy-id -i ${KEY_FILE}.pub ${PROXY_USER}@${PROXY_HOST}"
echo "   → Saisir le mot de passe du proxy (${PROXY_HOST}) :"
echo ""
ssh-copy-id -i "${KEY_FILE}.pub" "${PROXY_USER}@${PROXY_HOST}"
echo ""
echo "   Dépôt proxy  [OK]"
echo ""

# ── Étape 3 : Dépôt sur Jean Zay via proxy ──────────────────────────────────
echo "── Étape 3/4 : Dépôt de la clé sur Jean Zay (via proxy)"
echo "   → Saisir le mot de passe Jean Zay (${JEANZAY_HOST}) :"
echo ""
ssh-copy-id -i "${KEY_FILE}.pub" \
    -o "ProxyCommand=ssh -i ${KEY_FILE} ${PROXY_USER}@${PROXY_HOST} nc %h %p" \
    -o "StrictHostKeyChecking=no" \
    "${JEANZAY_USER}@${JEANZAY_HOST}"
echo ""
echo "   Dépôt Jean Zay  [OK]"
echo ""

# ── Étape 4 : Bloc Host dans ~/.ssh/config ───────────────────────────────────
echo "── Étape 4/4 : Configuration ~/.ssh/config"

SSH_CONFIG_BLOCK="
# ── MRIxFields — Jean Zay (généré par setup_ssh_keys.sh) ──────────────────
Host jz-proxy
    HostName ${PROXY_HOST}
    User ${PROXY_USER}
    IdentityFile ${KEY_FILE}
    ServerAliveInterval 60

Host jeanzay
    HostName ${JEANZAY_HOST}
    User ${JEANZAY_USER}
    IdentityFile ${KEY_FILE}
    ProxyCommand ssh jz-proxy nc %h %p
    ServerAliveInterval 60
    StrictHostKeyChecking no
# ── fin MRIxFields ─────────────────────────────────────────────────────────
"

if grep -q "Host jeanzay" "$SSH_CONFIG" 2>/dev/null; then
    echo "   Bloc 'jeanzay' déjà présent dans $SSH_CONFIG  [OK]"
    echo "   (Pour le mettre à jour, éditer manuellement $SSH_CONFIG)"
else
    touch "$SSH_CONFIG"
    chmod 600 "$SSH_CONFIG"
    echo "$SSH_CONFIG_BLOCK" >> "$SSH_CONFIG"
    echo "   Bloc ajouté dans $SSH_CONFIG  [OK]"
fi
echo ""

# ── Étape 5 : Test de connexion ──────────────────────────────────────────────
echo "── Test de connexion sans mot de passe"
echo "   ssh jeanzay hostname ..."
echo ""
if ssh -o BatchMode=yes -o ConnectTimeout=15 jeanzay hostname 2>/dev/null; then
    echo ""
    echo "   Connexion sans mot de passe  [OK]"
else
    echo ""
    echo "   [WARN] Le test a échoué. Vérifier :"
    echo "   - Que le proxy est accessible : ssh jz-proxy"
    echo "   - Que Jean Zay est accessible : ssh jeanzay"
    echo "   - Que les clés sont bien déposées (étapes 2 et 3)"
fi

echo ""
echo "======================================================================="
echo " Configuration terminée."
echo ""
echo " Commandes disponibles :"
echo "   ssh jeanzay                    # connexion interactive"
echo "   ssh jeanzay squeue -u \$USER   # état des jobs"
echo "   scp jeanzay:<path> <local>     # copie sans mot de passe"
echo ""
echo " Synchroniser les résultats :"
echo "   bash src/slurm/sync_from_jeanzay.sh mmfm3d_unet"
echo "======================================================================="
