#!/bin/bash
# Replay script for candidate cand_ba87603b992f
# Generated: 2025-12-28T21:30:30.232461572+00:00
#
# This script re-runs the backtest with the exact same configuration
# to verify reproducibility.

set -e

SEED=42
RUN_ID="run_5f15de092579"
CAMPAIGN_ID="camp_8e196a7d7db2"

echo "Replaying candidate cand_ba87603b992f"
echo "  Seed: $SEED"
echo "  Run ID: $RUN_ID"

# Run the backtest
cargo run --release --bin combiner -- run \
    --config configs/campaigns/$(basename "$0" .sh).toml \
    --seed $SEED \
    --output output/replay/$RUN_ID

echo "Replay complete. Compare results in output/replay/$RUN_ID"
