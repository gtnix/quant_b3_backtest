#!/bin/bash
# Auto-cleanup script - runs cargo clean when target/ exceeds threshold
# 
# Usage: 
#   ./scripts/auto_cleanup.sh [threshold_gb]     # Run cleanup if needed
#   ./scripts/auto_cleanup.sh --dry-run          # Show what would be deleted
#   ./scripts/auto_cleanup.sh --force            # Force cleanup even if CI
#
# Safety:
#   - Validates paths before deletion (prevents rm -rf /)
#   - Requires --force flag in CI environments
#   - Dry-run mode shows what would be deleted

set -euo pipefail

# Parse arguments
DRY_RUN=false
FORCE=false
THRESHOLD_GB=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        [0-9]*)
            THRESHOLD_GB=$1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [threshold_gb] [--dry-run] [--force]"
            exit 1
            ;;
    esac
done

# Determine project directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="$PROJECT_DIR/target"

# SAFETY: Validate paths
validate_path() {
    local path="$1"
    local expected_parent="$2"
    
    # Check path starts with expected parent
    if [[ ! "$path" == "$expected_parent"* ]]; then
        echo "[SAFETY] Path validation failed: $path does not start with $expected_parent"
        exit 1
    fi
    
    # Check path is not root or home
    if [[ "$path" == "/" || "$path" == "$HOME" || "$path" == "/home" ]]; then
        echo "[SAFETY] Refusing to delete critical path: $path"
        exit 1
    fi
    
    # Check path contains "target" for additional safety
    if [[ ! "$path" == *"target"* ]]; then
        echo "[SAFETY] Path does not contain 'target': $path"
        exit 1
    fi
}

# SAFETY: Check CI environment
if [[ "${CI:-}" == "true" && "$FORCE" != "true" ]]; then
    echo "[auto_cleanup] Running in CI without --force flag, skipping"
    exit 0
fi

# Check if target exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "[auto_cleanup] target/ não existe, nada a limpar"
    exit 0
fi

# Validate target path before any operations
validate_path "$TARGET_DIR" "$PROJECT_DIR"

# Get size in GB
SIZE_KB=$(du -sk "$TARGET_DIR" 2>/dev/null | cut -f1)
SIZE_GB=$((SIZE_KB / 1024 / 1024))

echo "[auto_cleanup] target/ = ${SIZE_GB}GB (threshold: ${THRESHOLD_GB}GB)"

if [ "$SIZE_GB" -ge "$THRESHOLD_GB" ]; then
    if [ "$DRY_RUN" = true ]; then
        echo "[auto_cleanup] DRY-RUN: Would delete:"
        echo "  - $TARGET_DIR (${SIZE_GB}GB)"
        echo "  - $PROJECT_DIR/*.log"
        echo ""
        echo "Run without --dry-run to actually delete."
    else
        echo "[auto_cleanup] Limpando target/ (${SIZE_GB}GB >= ${THRESHOLD_GB}GB)..."
        
        # Final safety check
        validate_path "$TARGET_DIR" "$PROJECT_DIR"
        
        # Log what we're about to delete
        echo "[auto_cleanup] Deleting: $TARGET_DIR"
        rm -rf "$TARGET_DIR"
        
        # Also clean logs
        if ls "$PROJECT_DIR"/*.log 1>/dev/null 2>&1; then
            echo "[auto_cleanup] Deleting: $PROJECT_DIR/*.log"
            rm -f "$PROJECT_DIR"/*.log 2>/dev/null
        fi
        
        echo "[auto_cleanup] Limpeza concluída!"
    fi
else
    echo "[auto_cleanup] OK, abaixo do threshold"
fi
