#!/bin/bash
#############################################################################
# Alpha Forge - Git Sync Script (runs on VPS)
# Pulls latest code and rebuilds
#############################################################################

set -e

PROJECT_DIR="/opt/alpha-forge/quant_b3_backtest"
LOG_FILE="/opt/alpha-forge/logs/sync.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"; }

cd "$PROJECT_DIR"

log ">>> Starting sync..."

# Check for changes
git fetch origin main
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    log ">>> Already up to date"
    exit 0
fi

log ">>> Updating from $LOCAL to $REMOTE"

# Pull changes
git reset --hard origin/main

# Check what changed
CHANGED_FILES=$(git diff --name-only "$LOCAL" "$REMOTE")
log ">>> Changed files: $CHANGED_FILES"

# Rebuild dashboard if needed
if echo "$CHANGED_FILES" | grep -q "^dashboard/"; then
    log ">>> Dashboard changed, rebuilding..."
    cd dashboard
    npm ci --silent
    NODE_OPTIONS='--max-old-space-size=1024' npm run build
    cd ..
fi

# Restart services
log ">>> Restarting services..."
cd dashboard
pm2 restart ecosystem.config.cjs --update-env
pm2 status

log ">>> Sync complete!"



















