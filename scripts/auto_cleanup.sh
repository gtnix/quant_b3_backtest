#!/bin/bash
# Auto-cleanup script for SCG runs and build artifacts
# 
# Usage:
#   ./scripts/auto_cleanup.sh                    # Status only
#   ./scripts/auto_cleanup.sh --runs             # Clean old SCG runs (>7 days)
#   ./scripts/auto_cleanup.sh --runs --days 3    # Clean runs older than 3 days
#   ./scripts/auto_cleanup.sh --target           # Clean target/ folder
#   ./scripts/auto_cleanup.sh --all              # Clean runs + cache + cockpit
#   ./scripts/auto_cleanup.sh --nuke             # NUCLEAR: delete ALL output + target
#   ./scripts/auto_cleanup.sh --dry-run          # Show what would be deleted

set -euo pipefail

DRY_RUN=false
CLEAN_RUNS=false
CLEAN_TARGET=false
CLEAN_ALL=false
NUKE=false
DAYS=7

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --runs) CLEAN_RUNS=true; shift ;;
        --target) CLEAN_TARGET=true; shift ;;
        --all) CLEAN_ALL=true; shift ;;
        --nuke) NUKE=true; shift ;;
        --days) DAYS=$2; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

du_safe() { du -sh "$1" 2>/dev/null | cut -f1 || echo "0"; }

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     STORAGE STATUS                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
printf "  %-30s %s\n" "target/" "$(du_safe "$PROJECT_DIR/target")"
printf "  %-30s %s\n" "output/scg/" "$(du_safe "$PROJECT_DIR/output/scg")"
printf "  %-30s %s\n" "cache/" "$(du_safe "$PROJECT_DIR/cache")"
printf "  %-30s %s\n" "artifacts/cockpit_runs/" "$(du_safe "$PROJECT_DIR/artifacts/cockpit_runs")"
printf "  %-30s %s\n" ".tmp/" "$(du_safe "$PROJECT_DIR/.tmp")"
echo ""

if [[ "$CLEAN_RUNS" == "true" || "$CLEAN_ALL" == "true" ]]; then
    echo "Cleaning SCG runs older than ${DAYS} days..."
    if [[ -d "$PROJECT_DIR/output/scg" ]]; then
        FOUND=$(find "$PROJECT_DIR/output/scg" -maxdepth 1 -type d -name "run_*" -mtime +${DAYS} 2>/dev/null | wc -l)
        if [[ "$FOUND" -gt 0 ]]; then
            if [[ "$DRY_RUN" == "true" ]]; then
                echo "  [DRY-RUN] Would delete $FOUND runs:"
                find "$PROJECT_DIR/output/scg" -maxdepth 1 -type d -name "run_*" -mtime +${DAYS} -exec basename {} \;
            else
                find "$PROJECT_DIR/output/scg" -maxdepth 1 -type d -name "run_*" -mtime +${DAYS} -exec rm -rf {} \;
                echo "  ✓ Deleted $FOUND old runs"
            fi
        else
            echo "  No runs older than ${DAYS} days"
        fi
    fi
fi

if [[ "$CLEAN_ALL" == "true" ]]; then
    echo ""
    echo "Cleaning cache and cockpit runs..."
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] Would delete:"
        echo "    - cache/"
        echo "    - artifacts/cockpit_runs/"
        echo "    - .tmp/"
    else
        rm -rf "$PROJECT_DIR/cache" "$PROJECT_DIR/artifacts/cockpit_runs" "$PROJECT_DIR/.tmp" 2>/dev/null || true
        mkdir -p "$PROJECT_DIR/cache" "$PROJECT_DIR/artifacts/cockpit_runs"
        echo "  ✓ Cleared cache and cockpit runs"
    fi
fi

if [[ "$CLEAN_TARGET" == "true" ]]; then
    echo ""
    echo "Cleaning target/..."
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] Would delete target/"
    else
        rm -rf "$PROJECT_DIR/target"
        echo "  ✓ Deleted target/"
    fi
fi

if [[ "$NUKE" == "true" ]]; then
    echo ""
    echo "🔥 NUCLEAR CLEANUP..."
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] Would delete:"
        echo "    - output/ (ALL)"
        echo "    - target/"
        echo "    - dashboard/src-tauri/target/"
        echo "    - cache/"
        echo "    - .tmp/"
        echo "    - artifacts/cockpit_runs/"
    else
        rm -rf "$PROJECT_DIR/output" "$PROJECT_DIR/target" "$PROJECT_DIR/cache" "$PROJECT_DIR/.tmp" 2>/dev/null || true
        rm -rf "$PROJECT_DIR/dashboard/src-tauri/target" 2>/dev/null || true
        rm -rf "$PROJECT_DIR/artifacts/cockpit_runs" 2>/dev/null || true
        mkdir -p "$PROJECT_DIR/cache" "$PROJECT_DIR/artifacts/cockpit_runs"
        git gc --prune=now --quiet 2>/dev/null || true
        echo "  ✓ NUKED everything. Run 'cargo build --release' to rebuild."
    fi
fi

if [[ "$CLEAN_RUNS" == "false" && "$CLEAN_TARGET" == "false" && "$CLEAN_ALL" == "false" && "$NUKE" == "false" ]]; then
    echo "Use --runs, --target, --all, or --nuke to clean"
fi
