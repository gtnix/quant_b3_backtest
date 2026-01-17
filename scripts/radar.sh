#!/bin/bash
# =============================================================================
# Radar 15 Dias - BR + US em Paralelo
# =============================================================================
# Uso: ./scripts/radar.sh
# Mercados: Brasil (B3) + Estados Unidos (S&P 500)
# CPU: 2 workers cada (4 total = 25% de 16 cores)
# Runtime: 8 horas cada
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Carregar variáveis de ambiente
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

mkdir -p logs

echo "========================================="
echo "   Radar 15 Dias - BR + US"
echo "========================================="
echo "  Capital: R$ 100.000 / USD 100.000"
echo "  Mercados: Brasil + EUA (paralelo)"
echo "  CPU: 2 workers BR + 2 workers US"
echo "  Runtime: 8 horas cada"
echo "  Long + Short habilitados"
echo "========================================="
echo ""

# Compilar em release se necessário
if [ ! -f ./target/release/combiner ]; then
    echo "Compilando combiner em release..."
    cargo build -p combiner_cli --release
fi

echo "Iniciando campanhas em background..."
echo ""

# BR Campaign
nohup ./target/release/combiner factory run \
    --campaign configs/campaigns/radar_15d.toml \
    > logs/radar_br.log 2>&1 &
BR_PID=$!
echo "  BR PID: $BR_PID (logs/radar_br.log)"

# US Campaign
nohup ./target/release/combiner factory run \
    --campaign configs/campaigns/radar_15d_us.toml \
    > logs/radar_us.log 2>&1 &
US_PID=$!
echo "  US PID: $US_PID (logs/radar_us.log)"

echo ""
echo "========================================="
echo "  Campanhas rodando em background!"
echo "========================================="
echo ""
echo "Monitorar:"
echo "  tail -f logs/radar_br.log"
echo "  tail -f logs/radar_us.log"
echo ""
echo "Verificar processos:"
echo "  ps aux | grep combiner"
echo ""
echo "Parar:"
echo "  kill $BR_PID $US_PID"
echo ""
echo "Dashboard: http://localhost:5173"
echo "========================================="
