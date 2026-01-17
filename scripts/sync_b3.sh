#!/bin/bash
# =============================================================================
# SYNC B3 - Sincronização completa de dados do mercado brasileiro
# =============================================================================
# Provider: BRAPI (suporta daily + intraday)
# Tabelas: b3_index_composition, ohlcv_daily, ohlcv_intraday_br
# =============================================================================

set -e

cd "$(dirname "$0")/.."

# Carregar variáveis de ambiente
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep BRAPI | xargs)
    # Mapear BRAPI_TOKEN para BRAPI_API_KEY se necessário
    if [ -z "$BRAPI_API_KEY" ] && [ -n "$BRAPI_TOKEN" ]; then
        export BRAPI_API_KEY="$BRAPI_TOKEN"
    fi
fi

if [ -z "$BRAPI_API_KEY" ]; then
    echo "❌ BRAPI_API_KEY não configurada"
    exit 1
fi

echo "=========================================="
echo "  SYNC B3 - Mercado Brasileiro"
echo "=========================================="
echo ""

# Parâmetros configuráveis
DAILY_RANGE="${DAILY_RANGE:-1mo}"
INTRADAY_INTERVAL="${INTRADAY_INTERVAL:-30m}"
INTRADAY_RANGE="${INTRADAY_RANGE:-5d}"

# -----------------------------------------------------------------------------
# 1. Sincronizar índices B3 (IBOV, IBRA, SMLL, IDIV, IFIX)
# -----------------------------------------------------------------------------
echo "📊 [1/3] Sincronizando índices B3..."
python3 -m datahub_b3 sync

# -----------------------------------------------------------------------------
# 2. Sincronizar dados diários (OHLCV)
# -----------------------------------------------------------------------------
echo ""
echo "📈 [2/3] Sincronizando daily OHLCV (${DAILY_RANGE})..."
python3 -m datahub_b3 daily-sync --range "$DAILY_RANGE"

# -----------------------------------------------------------------------------
# 3. Sincronizar dados intraday via BRAPI
# -----------------------------------------------------------------------------
echo ""
echo "⏱️  [3/3] Sincronizando intraday ${INTRADAY_INTERVAL} (${INTRADAY_RANGE})..."
python3 -m datahub_b3 intraday-sync --interval "$INTRADAY_INTERVAL" --range "$INTRADAY_RANGE"

echo ""
echo "=========================================="
echo "  ✅ SYNC B3 COMPLETO"
echo "=========================================="
