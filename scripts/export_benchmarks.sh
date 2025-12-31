#!/bin/bash
# Export Criterion benchmark results to consolidated JSON

set -e

OUTPUT_DIR="${1:-benches/results}"
CRITERION_DIR="target/criterion"

mkdir -p "$OUTPUT_DIR"

# Generate consolidated benchmark.json
cat > "$OUTPUT_DIR/benchmark.json" << 'EOF'
{
  "version": "1.0.0",
  "exported_at": "$(date -Iseconds)",
  "scenarios": {}
}
EOF

# Read actual values from Criterion
python3 << 'PYTHON'
import json
import os
from pathlib import Path

criterion_dir = Path("target/criterion")
output = {
    "version": "1.0.0",
    "exported_at": "",
    "scenarios": {}
}

scenarios = [
    ("intraday_net_zero/mean_reversion_10k_events", 10000),
    ("intraday_net_zero/noop_baseline_10k_events", 10000),
    ("daily_swing/trend_200_assets", 50400),
    ("stress_universe/noop_1000_assets", 252000),
]

for scenario_path, events in scenarios:
    est_file = criterion_dir / scenario_path / "new" / "estimates.json"
    if est_file.exists():
        with open(est_file) as f:
            est = json.load(f)
        mean_ns = est["mean"]["point_estimate"]
        std_ns = est["std_dev"]["point_estimate"]
        
        throughput = events / (mean_ns / 1e9)
        
        parts = scenario_path.split("/")
        group = parts[0]
        name = parts[1]
        
        if group not in output["scenarios"]:
            output["scenarios"][group] = {}
        
        output["scenarios"][group][name] = {
            "events": events,
            "mean_ns": mean_ns,
            "std_dev_ns": std_ns,
            "throughput_events_per_sec": throughput
        }

with open("benches/results/benchmark.json", "w") as f:
    json.dump(output, f, indent=2)

print("Exported to benches/results/benchmark.json")
PYTHON

echo "Benchmark export complete"




























