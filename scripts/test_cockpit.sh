#!/bin/bash
# Test Cockpit Config Generation
# 
# This script simulates what the Cockpit UI does when you click START,
# allowing you to debug the config generation without running the full Tauri app.
#
# Usage:
#   ./scripts/test_cockpit.sh                    # Generate and validate config
#   ./scripts/test_cockpit.sh --run              # Actually execute the run
#   ./scripts/test_cockpit.sh --preset rapid     # Use a specific preset
#   ./scripts/test_cockpit.sh --timeout 60       # Override timeout

set -euo pipefail

# Parse arguments
PRESET="rapid"
TIMEOUT=180
POPULATION=50
RUN_IT=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --run)
            RUN_IT=true
            shift
            ;;
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --population)
            POPULATION="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--run] [--preset rapid|institutional|exhaustive] [--timeout N] [--verbose]"
            exit 1
            ;;
    esac
done

# Determine project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Use local temp folder (not /tmp)
export TMPDIR="$PROJECT_DIR/.tmp"
mkdir -p "$TMPDIR"

# Generate run ID (same format as Tauri)
RUN_ID="cockpit_$(date +%Y%m%d_%H%M%S)"

# Create output directory
COCKPIT_DIR="$PROJECT_DIR/artifacts/cockpit_runs/$RUN_ID"
mkdir -p "$COCKPIT_DIR"

# Set preset-specific values
case "$PRESET" in
    rapid)
        MAX_GENS=1000000000
        CONVERGENCE=999
        STRESS_ENABLED=false
        MIN_SHARPE=0.3
        MAX_PBO=0.25
        ;;
    institutional)
        MAX_GENS=50
        CONVERGENCE=10
        STRESS_ENABLED=true
        MIN_SHARPE=0.5
        MAX_PBO=0.15
        ;;
    exhaustive)
        MAX_GENS=100
        CONVERGENCE=15
        STRESS_ENABLED=true
        MIN_SHARPE=0.5
        MAX_PBO=0.15
        ;;
    *)
        echo "Unknown preset: $PRESET"
        exit 1
        ;;
esac

# Generate campaign TOML (exactly like Tauri does)
cat > "$COCKPIT_DIR/campaign.toml" << EOF
# Cockpit-generated Campaign Configuration
# Run ID: $RUN_ID
# Preset: $PRESET
# Generated at: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

[campaign]
name = "$RUN_ID"
tag = "$PRESET"
owner = "cockpit"
notes = "Auto-generated from test_cockpit.sh with preset: $PRESET"

[dataset]
market = "BR"
start_date = "2018-01-01"
end_date = "2024-12-01"
universe = "ibov"

[evolution]
population_size = $POPULATION
max_generations = $MAX_GENS
convergence_generations = $CONVERGENCE

[execution]
config_path = "configs/execution_institutional.toml"
delay_bars = 1

[seeds]
count = 1
base_seed = 42

[budget]
max_runs = 1
top_k = 50
persist_stage_a_top_n = 100
timeout_per_run_secs = $TIMEOUT
stress_enabled = $STRESS_ENABLED

[promotion]
min_oos_sharpe_net = $MIN_SHARPE
max_pbo = $MAX_PBO
min_stress_passed = 0
gates_required = true

[data_integrity]
mode = "fast"
max_gap_days = 10
jump_threshold_pct = 30.0
price_adjustment = "adjusted"
universe_type = "static"
enabled = true
EOF

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              COCKPIT TEST - CONFIG GENERATED                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Run ID:   $RUN_ID"
echo "Preset:   $PRESET"
echo "Timeout:  ${TIMEOUT}s"
echo "Config:   $COCKPIT_DIR/campaign.toml"
echo ""

# Check if combiner binary exists
COMBINER="$PROJECT_DIR/target/release/combiner"
if [ ! -f "$COMBINER" ]; then
    echo "⚠ Combiner binary not found at: $COMBINER"
    echo "  Build with: cargo build --release -p combiner_cli"
    echo ""
    echo "Showing generated config:"
    echo "════════════════════════════════════════════════════════════════"
    cat "$COCKPIT_DIR/campaign.toml"
    exit 1
fi

# Validate the config
echo "Validating config..."
echo ""

if [ "$VERBOSE" = true ]; then
    "$COMBINER" factory validate --campaign "$COCKPIT_DIR/campaign.toml" --verbose
else
    "$COMBINER" factory validate --campaign "$COCKPIT_DIR/campaign.toml"
fi

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Validation FAILED"
    exit 1
fi

echo ""

# Actually run if requested
if [ "$RUN_IT" = true ]; then
    echo "════════════════════════════════════════════════════════════════"
    echo "EXECUTING RUN..."
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    # Load .env file
    if [ -f "$PROJECT_DIR/.env" ]; then
        set -a
        source "$PROJECT_DIR/.env"
        set +a
        echo "Loaded .env file"
    fi
    
    # Map DATABASE_URL to NEON_DATABASE_URL if needed
    if [ -z "${NEON_DATABASE_URL:-}" ] && [ -n "${DATABASE_URL:-}" ]; then
        export NEON_DATABASE_URL="$DATABASE_URL"
    fi
    
    # Run the campaign
    cd "$PROJECT_DIR"
    "$COMBINER" factory run --campaign "$COCKPIT_DIR/campaign.toml"
else
    echo "To actually execute, run with --run flag:"
    echo "  $0 --preset $PRESET --run"
fi

