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

# Function to check if API is ready
wait_for_api() {
    echo "Aguardando API..." >> "$LOG_DIR/startup.log"
    for i in {1..30}; do
        if curl -s http://localhost:3001/api/health > /dev/null 2>&1; then
            echo "API pronta após ${i}s" >> "$LOG_DIR/startup.log"
            return 0
        fi
        sleep 1
    done
    echo "API timeout após 30s" >> "$LOG_DIR/startup.log"
    return 1
}

# Kill any existing processes
pkill -f "node.*server.js" 2>/dev/null || true
pkill -f "quant-dashboard" 2>/dev/null || true
sleep 1

echo "=== Iniciando Quant Dashboard $(date) ===" > "$LOG_DIR/startup.log"

# Start API server in background (silent)
cd "$DASHBOARD_DIR"
echo "Iniciando API server..." >> "$LOG_DIR/startup.log"
nohup node server.js >> "$LOG_DIR/api-server.log" 2>&1 &
API_PID=$!
echo "API PID: $API_PID" >> "$LOG_DIR/startup.log"

# Wait for API to be ready
if ! wait_for_api; then
    notify "Erro: API não iniciou. Verifique logs/api-server.log"
    exit 1
fi

notify "API pronta. Abrindo Dashboard..."

# Start Tauri app
if [ -f "$TAURI_BIN" ]; then
    echo "Iniciando Tauri..." >> "$LOG_DIR/startup.log"
    "$TAURI_BIN" >> "$LOG_DIR/tauri.log" 2>&1
else
    echo "Tauri não encontrado, abrindo browser..." >> "$LOG_DIR/startup.log"
    notify "Tauri não encontrado, abrindo browser..."
    xdg-open "http://localhost:5173" 2>/dev/null &
    cd "$DASHBOARD_DIR" && npm run dev >> "$LOG_DIR/vite.log" 2>&1
fi

# When Tauri closes, stop API
echo "Tauri fechado, parando API..." >> "$LOG_DIR/startup.log"
kill $API_PID 2>/dev/null || true
