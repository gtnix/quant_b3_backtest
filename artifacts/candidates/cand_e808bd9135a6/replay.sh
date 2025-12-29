#!/bin/bash
# Replay script for candidate cand_e808bd9135a6
# Generated: 2025-12-28T22:22:46.058009432+00:00
#
# This script re-runs the backtest with the exact same configuration
# to verify reproducibility.

set -e

# Navigate to project root (works from any directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

SEED=42
RUN_ID="run_104478a3c91a"
CAMPAIGN_ID="camp_f45b07858429"

echo "Replaying candidate cand_e808bd9135a6"
echo "  Seed: $SEED"
echo "  Run ID: $RUN_ID"
echo "  Project root: $PROJECT_ROOT"

# Run the backtest using pre-compiled binary
./target/release/combiner run \
    --seed $SEED \
    --output output/replay/$RUN_ID

echo "Replay complete. Compare results in output/replay/$RUN_ID"
