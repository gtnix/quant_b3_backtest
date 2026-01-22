#!/bin/bash
# =============================================================================
# MINE - Single Entry Point for Alpha Forge (API-controlled)
# =============================================================================
# Usage: ./scripts/mine.sh [day|night]
# 
# This script:
#   1. Starts the dashboard backend (API)
#   2. Starts mining via POST /api/omp/quick-start
#   3. Dashboard is the canonical controller for start/stop/status
# =============================================================================
set -e
cd /projetos/quant_b3_backtest

MODE="${1:-day}"
API_PORT=3001
API_URL="http://localhost:$API_PORT/api"

echo "==========================================="
echo "  ALPHA FORGE"
echo "==========================================="
echo "  Mode: $MODE"
echo ""

# ============ CLEANUP ============
pkill -f "combiner run" 2>/dev/null || true
pkill -f "node server" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
sleep 1

mkdir -p logs output/scg

# ============ PREFLIGHT ============
echo "[1/4] Preflight validation..."

if ! ./target/release/combiner preflight --config configs/default.toml > /dev/null 2>&1; then
    echo "  BR: FAILED - run preflight manually for details"
    exit 1
fi
echo "  BR: OK"

if ! ./target/release/combiner preflight --config configs/default_us.toml > /dev/null 2>&1; then
    echo "  US: FAILED - run preflight manually for details"
    exit 1
fi
echo "  US: OK"

# ============ DASHBOARD ============
echo ""
echo "[2/4] Starting Dashboard backend..."
cd dashboard
nohup node server/index.js > ../logs/api.log 2>&1 &
cd ..

# Wait for API to be ready
echo -n "  Waiting for API..."
for i in {1..30}; do
    if curl -s "$API_URL/omp/status" > /dev/null 2>&1; then
        echo " OK"
        break
    fi
    sleep 1
    echo -n "."
done

if ! curl -s "$API_URL/omp/status" > /dev/null 2>&1; then
    echo " FAILED (API not responding)"
    exit 1
fi

# ============ START FRONTEND ============
echo ""
echo "[3/4] Starting Dashboard frontend..."
cd dashboard
nohup npm run dev > ../logs/frontend.log 2>&1 &
cd ..
sleep 2

# ============ START MINING VIA API ============
echo ""
echo "[4/4] Starting Mining via API..."

# Map mode to API parameters
if [ "$MODE" = "night" ]; then
    API_MODE="full"
else
    API_MODE="quick"
fi

# Call the canonical API endpoint
RESPONSE=$(curl -s -X POST "$API_URL/omp/quick-start" \
    -H "Content-Type: application/json" \
    -d "{\"mode\": \"$API_MODE\", \"indefinite\": true, \"markets\": [\"BR\", \"US\"]}")

# Check response
if echo "$RESPONSE" | grep -q '"status":"started"'; then
    EXPERIMENT_ID=$(echo "$RESPONSE" | grep -o '"experimentId":"[^"]*"' | cut -d'"' -f4)
    echo "  Mining started: $EXPERIMENT_ID"
else
    echo "  FAILED to start mining:"
    echo "  $RESPONSE"
    exit 1
fi

# ============ STATUS ============
echo ""
echo "==========================================="
echo "  RUNNING"
echo "==========================================="
echo "  Dashboard: http://localhost:5173"
echo "  API: $API_URL/omp/status"
echo "  Experiment: $EXPERIMENT_ID"
echo "  Logs: tail -f logs/br.log logs/us.log"
echo "  Stop: curl -X POST $API_URL/omp/stop"
echo "        or ./scripts/stop.sh"

# Open browser
command -v xdg-open &>/dev/null && xdg-open http://localhost:5173 2>/dev/null &
