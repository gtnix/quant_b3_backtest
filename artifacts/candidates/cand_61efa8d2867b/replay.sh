#!/bin/bash
# Replay script for candidate cand_61efa8d2867b
# Generated: 2025-12-28T22:45:46.909716859+00:00
#
# This script re-runs the backtest with the exact same configuration
# to verify reproducibility.

set -e

# Navigate to project root (works from any directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

SEED=44
RUN_ID="run_cf69a9b76073"
CAMPAIGN_ID="camp_539cb2beeaa9"
STRATEGY_CONFIG="$SCRIPT_DIR/strategy.toml"

echo "Replaying candidate cand_61efa8d2867b"
echo "  Seed: $SEED"
echo "  Run ID: $RUN_ID"
echo "  Project root: $PROJECT_ROOT"
echo "  Strategy config: $STRATEGY_CONFIG"

# Check if strategy.toml exists
if [ ! -f "$STRATEGY_CONFIG" ]; then
    echo "ERROR: strategy.toml not found in $SCRIPT_DIR"
    exit 1
fi

# Run the backtest using pre-compiled binary
./target/release/combiner run \
    --config "$STRATEGY_CONFIG" \
    --seed $SEED \
    --output output/replay/$RUN_ID

echo "Replay complete. Compare results in output/replay/$RUN_ID"
