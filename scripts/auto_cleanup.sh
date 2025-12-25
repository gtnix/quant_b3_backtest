#!/bin/bash
# Auto-cleanup script - runs cargo clean when target/ exceeds threshold
# Usage: ./scripts/auto_cleanup.sh [threshold_gb]
# Can be added to .bashrc or called before builds

THRESHOLD_GB=${1:-5}
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$PROJECT_DIR/target"

if [ ! -d "$TARGET_DIR" ]; then
    echo "[auto_cleanup] target/ não existe, nada a limpar"
    exit 0
fi

# Get size in GB
SIZE_KB=$(du -sk "$TARGET_DIR" 2>/dev/null | cut -f1)
SIZE_GB=$((SIZE_KB / 1024 / 1024))

echo "[auto_cleanup] target/ = ${SIZE_GB}GB (threshold: ${THRESHOLD_GB}GB)"

if [ "$SIZE_GB" -ge "$THRESHOLD_GB" ]; then
    echo "[auto_cleanup] Limpando target/ (${SIZE_GB}GB >= ${THRESHOLD_GB}GB)..."
    rm -rf "$TARGET_DIR"
    
    # Also clean logs
    rm -f "$PROJECT_DIR"/*.log 2>/dev/null
    
    echo "[auto_cleanup] Limpeza concluída!"
else
    echo "[auto_cleanup] OK, abaixo do threshold"
fi

