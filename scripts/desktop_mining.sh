#!/bin/bash
# =============================================================================
# Desktop Mining - All-in-one script for long-running strategy mining
# Uses 50% CPU (8 workers) with Tauri desktop visualization
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DASHBOARD_DIR="$PROJECT_ROOT/dashboard"
WORKERS=$(($(nproc) / 2))  # 50% of CPUs

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       🚀 Desktop Mining - Quant B3 Strategy Discovery        ║${NC}"
echo -e "${BLUE}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║  Workers: ${YELLOW}${WORKERS} CPUs${BLUE} (50% of $(nproc))                              ║${NC}"
echo -e "${BLUE}║  Mode: ${GREEN}ephemeral_artifacts${BLUE} (low disk usage)                 ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if binaries exist
if [ ! -f "$PROJECT_ROOT/target/release/combiner" ]; then
    echo -e "${YELLOW}Building combiner...${NC}"
    cd "$PROJECT_ROOT" && cargo build --release --bin combiner
fi

if [ ! -f "$PROJECT_ROOT/target/release/backtest" ]; then
    echo -e "${YELLOW}Building backtest...${NC}"
    cd "$PROJECT_ROOT" && cargo build --release --bin backtest
fi

# Kill any existing processes
pkill -f "node.*server.js" 2>/dev/null || true
pkill -f "quant-b3-dashboard" 2>/dev/null || true

# Start API server in background
echo -e "${GREEN}[1/2] Starting API server...${NC}"
cd "$DASHBOARD_DIR"
node server.js &
API_PID=$!
sleep 2

# Check if API is running
if curl -s http://localhost:3001/api/health > /dev/null 2>&1; then
    echo -e "      ${GREEN}✓ API running on http://localhost:3001${NC}"
else
    echo -e "      ${YELLOW}⚠ API may take a moment to start${NC}"
fi

# Start Tauri app
echo -e "${GREEN}[2/2] Starting Tauri desktop app...${NC}"
cd "$DASHBOARD_DIR"

# Check if Tauri binary exists, otherwise use dev mode
TAURI_BIN="$DASHBOARD_DIR/src-tauri/target/release/quant-dashboard"
if [ -f "$TAURI_BIN" ]; then
    echo -e "      ${GREEN}✓ Using native Tauri app (lightweight)${NC}"
    "$TAURI_BIN" &
    TAURI_PID=$!
else
    echo -e "      ${YELLOW}Tauri binary not found, using browser mode${NC}"
    echo -e "      ${YELLOW}To build Tauri: cd dashboard && CI=false cargo tauri build${NC}"
    echo ""
    echo -e "      Opening browser at ${GREEN}http://localhost:5173${NC}"
    npm run dev &
    TAURI_PID=$!
    sleep 3
    xdg-open "http://localhost:5173" 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Desktop Mining ready! Navigate to Mining tab to start.${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Press ${YELLOW}Ctrl+C${NC} to stop all services"
echo ""

# Trap Ctrl+C to cleanup
cleanup() {
    echo ""
    echo -e "${YELLOW}Stopping services...${NC}"
    kill $API_PID 2>/dev/null || true
    kill $TAURI_PID 2>/dev/null || true
    pkill -f "node.*server.js" 2>/dev/null || true
    pkill -f "vite" 2>/dev/null || true
    echo -e "${GREEN}Done.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes
wait
