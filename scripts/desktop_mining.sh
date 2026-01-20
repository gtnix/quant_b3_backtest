#!/bin/bash
# =============================================================================
# Desktop Mining - All-in-one launcher for strategy mining
# Double-click from Ubuntu desktop to start everything
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DASHBOARD_DIR="$PROJECT_ROOT/dashboard"
LOG_DIR="$PROJECT_ROOT/logs"
TAURI_BIN="$DASHBOARD_DIR/src-tauri/target/release/quant-dashboard"

mkdir -p "$LOG_DIR"

# Function to show notification
notify() {
    notify-send "Quant Dashboard" "$1" -i "$DASHBOARD_DIR/src-tauri/icons/128x128.png" 2>/dev/null || true
}

# Kill any existing processes
pkill -f "node.*server.js" 2>/dev/null || true
sleep 1

# Start API server in background (silent)
cd "$DASHBOARD_DIR"
nohup node server.js > "$LOG_DIR/api-server.log" 2>&1 &
API_PID=$!

# Wait for API to be ready
for i in {1..10}; do
    if curl -s http://localhost:3001/api/health > /dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

# Start Tauri app
if [ -f "$TAURI_BIN" ]; then
    notify "Iniciando Dashboard..."
    "$TAURI_BIN" 2>> "$LOG_DIR/tauri.log"
else
    notify "Tauri não encontrado, abrindo browser..."
    xdg-open "http://localhost:5173" 2>/dev/null &
    cd "$DASHBOARD_DIR" && npm run dev >> "$LOG_DIR/vite.log" 2>&1
fi
