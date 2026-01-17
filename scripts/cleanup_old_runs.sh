#!/bin/bash
# Auto-cleanup old SCG runs - keep only N most recent
# Runs every hour via cron or manually

OUTPUT_DIR="${1:-/home/bahuan/Documents/GitHub/quant_b3_backtest/output/scg}"
KEEP_RUNS="${2:-5}"
MIN_FREE_GB="${3:-2}"

# Check disk space
FREE_GB=$(df --output=avail -BG "$OUTPUT_DIR" 2>/dev/null | tail -1 | tr -d 'G ')

echo "[$(date)] Disk cleanup check: ${FREE_GB}GB free, threshold: ${MIN_FREE_GB}GB"

if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
    echo "[$(date)] Low disk space detected! Cleaning old runs..."
    
    cd "$OUTPUT_DIR" || exit 1
    
    # Count runs
    TOTAL_RUNS=$(ls -d run_* 2>/dev/null | wc -l)
    
    if [ "$TOTAL_RUNS" -gt "$KEEP_RUNS" ]; then
        # Remove oldest runs, keep KEEP_RUNS most recent
        TO_REMOVE=$((TOTAL_RUNS - KEEP_RUNS))
        echo "[$(date)] Removing $TO_REMOVE old runs (keeping $KEEP_RUNS)"
        
        ls -t | tail -n +$((KEEP_RUNS + 1)) | xargs -r rm -rf
        
        NEW_FREE=$(df --output=avail -BG "$OUTPUT_DIR" | tail -1 | tr -d 'G ')
        echo "[$(date)] Cleanup complete. Now ${NEW_FREE}GB free"
    else
        echo "[$(date)] Only $TOTAL_RUNS runs, nothing to clean"
    fi
else
    echo "[$(date)] Disk space OK"
fi
