#!/bin/bash
# =============================================================================
# Start Alpha Forge Dashboard
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

mkdir -p logs

echo "========================================="
echo "   Alpha Forge - Strategy Miner"
echo "========================================="
echo ""

# Parar processos anteriores
pkill -f "node server.js" 2>/dev/null
pkill -f "vite" 2>/dev/null
sleep 1

# Iniciar API Server
echo "Iniciando API Server..."
cd "$PROJECT_DIR/dashboard"
nohup node server.js > "$PROJECT_DIR/logs/server.log" 2>&1 &
API_PID=$!
echo "  API PID: $API_PID"

sleep 3

# Iniciar Frontend
echo "Iniciando Frontend..."
nohup npm run dev > "$PROJECT_DIR/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID"

sleep 3

echo ""
echo "========================================="
echo "  Dashboard: http://localhost:5173"
echo "  API:       http://localhost:3001"
echo "========================================="
echo ""
echo "Para parar: ./scripts/stop.sh"
echo ""

# Abrir no navegador (opcional)
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:5173 2>/dev/null &
fi

