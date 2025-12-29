#!/bin/bash
# Replay script for candidate cand_242c293846b4
# Generated: 2025-12-28T21:22:21.425770291+00:00
#
# This script re-runs the backtest with the exact same configuration
# to verify reproducibility.

set -e

SEED=42
RUN_ID="run_c880705adaab"
CAMPAIGN_ID="camp_59ef4ea5dd46"

echo "Replaying candidate cand_242c293846b4"
echo "  Seed: $SEED"
echo "  Run ID: $RUN_ID"

# Run the backtest
cargo run --release --bin combiner -- run \
    --config configs/campaigns/$(basename "$0" .sh).toml \
    --seed $SEED \
    --output output/replay/$RUN_ID

echo "Replay complete. Compare results in output/replay/$RUN_ID"
