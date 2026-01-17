#!/bin/bash
# =============================================================================
# SYNC US - Sincronização completa de dados do mercado americano
# =============================================================================
# Provider: yfinance (daily + intraday)
# Tabelas: us_index_composition, ohlcv_us, ohlcv_intraday_us, dividends_us
# =============================================================================

set -e

cd "$(dirname "$0")/.."

# Carregar variáveis de ambiente
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "=========================================="
echo "  SYNC US - Mercado Americano"
echo "=========================================="
echo ""

# Parâmetros configuráveis
DAILY_PERIOD="${DAILY_PERIOD:-1mo}"
INTRADAY_INTERVAL="${INTRADAY_INTERVAL:-30m}"
INTRADAY_PERIOD="${INTRADAY_PERIOD:-5d}"
SKIP_DIVIDENDS="${SKIP_DIVIDENDS:-false}"

# -----------------------------------------------------------------------------
# 1. Sincronizar índices US (SPX, NDX, DJI)
# -----------------------------------------------------------------------------
echo "📊 [1/5] Sincronizando índices US..."
python3 -m datahub_us indices-sync

# -----------------------------------------------------------------------------
# 2. Update incremental (busca novos dados desde última data)
# -----------------------------------------------------------------------------
echo ""
echo "🔄 [2/5] Update incremental..."
python3 -m datahub_us update

# -----------------------------------------------------------------------------
# 3. Sincronizar dados diários (OHLCV)
# -----------------------------------------------------------------------------
echo ""
echo "📈 [3/5] Sincronizando daily OHLCV (${DAILY_PERIOD})..."
python3 -m datahub_us daily-sync --period "$DAILY_PERIOD"

# -----------------------------------------------------------------------------
# 4. Sincronizar dados intraday via yfinance
# -----------------------------------------------------------------------------
echo ""
echo "⏱️  [4/5] Sincronizando intraday ${INTRADAY_INTERVAL} (${INTRADAY_PERIOD})..."
python3 -m datahub_us intraday-sync --interval "$INTRADAY_INTERVAL" --period "$INTRADAY_PERIOD"

# -----------------------------------------------------------------------------
# 5. Sincronizar dividendos (opcional)
# -----------------------------------------------------------------------------
if [ "$SKIP_DIVIDENDS" != "true" ]; then
    echo ""
    echo "💰 [5/5] Sincronizando dividendos..."
    python3 -m datahub_us dividends-sync
else
    echo ""
    echo "⏭️  [5/5] Dividendos pulados (SKIP_DIVIDENDS=true)"
fi

echo ""
echo "=========================================="
echo "  ✅ SYNC US COMPLETO"
echo "=========================================="
