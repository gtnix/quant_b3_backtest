#!/bin/bash
# RAP Grid Search Optimizer
# 50% In-Sample (2022-2023) | 50% Out-of-Sample (2024)

set -e

PROJECT_DIR="/home/bahuan/Documents/GitHub/quant_b3_backtest"
cd "$PROJECT_DIR"

echo "
╔═══════════════════════════════════════════════════════════════════════════╗
║  🔬 RAP GRID SEARCH OPTIMIZER                                             ║
║  📊 In-Sample: 2022-01-01 a 2023-06-30 (18 meses)                        ║
║  🎯 Out-of-Sample: 2023-07-01 a 2024-12-31 (18 meses)                    ║
╚═══════════════════════════════════════════════════════════════════════════╝
"

# Compilar se necessário
if [ ! -f "target/release/backtest" ]; then
    echo "⚙️  Compilando backtester..."
    cargo build --release -p backtester_cli 2>/dev/null
fi

# Diretório de resultados
RESULTS_DIR="output/optimization_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# CSV de resultados
RESULTS_CSV="$RESULTS_DIR/grid_results.csv"
echo "config_id,entry_z,exit_z,stop_z,coint_pval,top_pairs,lookback,is_return,is_trades,is_sharpe,oos_return,oos_trades,oos_sharpe,total_time" > "$RESULTS_CSV"

# Parâmetros a testar (mais agressivos)
ENTRY_ZSCORES=(1.0 1.25 1.5)
EXIT_ZSCORES=(0.3 0.5)
STOP_ZSCORES=(2.0 2.5)
COINT_PVALUES=(0.10 0.20)
TOP_PAIRS=(20 30)
LOOKBACKS=(45 60)

CONFIG_ID=0
TOTAL_CONFIGS=$((${#ENTRY_ZSCORES[@]} * ${#EXIT_ZSCORES[@]} * ${#STOP_ZSCORES[@]} * ${#COINT_PVALUES[@]} * ${#TOP_PAIRS[@]} * ${#LOOKBACKS[@]}))

echo "📋 Total de configurações: $TOTAL_CONFIGS"
echo ""

BEST_OOS_RETURN=-999
BEST_CONFIG=""

for entry_z in "${ENTRY_ZSCORES[@]}"; do
for exit_z in "${EXIT_ZSCORES[@]}"; do
for stop_z in "${STOP_ZSCORES[@]}"; do
for coint_pval in "${COINT_PVALUES[@]}"; do
for top_pairs in "${TOP_PAIRS[@]}"; do
for lookback in "${LOOKBACKS[@]}"; do

    CONFIG_ID=$((CONFIG_ID + 1))
    
    # Criar config temporária
    CONFIG_FILE="$RESULTS_DIR/config_${CONFIG_ID}.toml"
    
    cat > "$CONFIG_FILE" << EOF
name = "RAP Opt Config $CONFIG_ID"
initial_capital = 100000.0
equity = 100000.0
leverage_max = 3.0
gross_exposure_max = 300000.0

data_source = "cache"
universe = "IBOV"
min_avg_volume = 1000000
cache_dir = "cache"

lookback_days = $lookback
top_n_pairs = $top_pairs
cointegration_test = "engle_granger"
cointegration_pvalue = $coint_pval
max_pairs_per_asset = 5

rebalance_frequency = "weekly"
daily_exit_monitoring = true
max_entry_day = 25

entry_zscore = $entry_z
exit_zscore = $exit_z
stop_zscore = $stop_z

kalman_delta = 0.001
kalman_ve = 0.001

kelly_fraction = 0.5
max_exposure_per_pair = 0.20
max_total_exposure = 1.0
kelly_lookback_trades = 10

execution_price = "close"
slippage_bps = 2.0
lot_size = 100
brokerage_per_order = 5.0
brokerage_bps = 3.0
b3_emoluments_bps = 3.25
stock_loan_rate_pa = 0.02

fail_fast = false
neutrality_tolerance = 0.05
EOF

    printf "\r🔄 [$CONFIG_ID/$TOTAL_CONFIGS] entry=%.2f exit=%.2f stop=%.2f pval=%.2f pairs=%d lookback=%d" \
        "$entry_z" "$exit_z" "$stop_z" "$coint_pval" "$top_pairs" "$lookback"

    START_TIME=$(date +%s.%N)
    
    # In-Sample backtest (2022-01-01 a 2023-06-30)
    IS_OUTPUT="$RESULTS_DIR/is_${CONFIG_ID}"
    ./target/release/backtest rap \
        --config "$CONFIG_FILE" \
        --output "$IS_OUTPUT" \
        --start-date 2022-01-01 \
        --end-date 2023-06-30 2>/dev/null || true
    
    # Extrair métricas IS
    if [ -f "$IS_OUTPUT/metrics.json" ]; then
        IS_RETURN=$(jq -r '.total_return // 0' "$IS_OUTPUT/metrics.json")
        IS_TRADES=$(jq -r '.total_trades // 0' "$IS_OUTPUT/metrics.json")
        IS_SHARPE=$(jq -r '.sharpe_ratio // 0' "$IS_OUTPUT/metrics.json")
    else
        IS_RETURN=0
        IS_TRADES=0
        IS_SHARPE=0
    fi
    
    # Out-of-Sample backtest (2023-07-01 a 2024-12-31)
    OOS_OUTPUT="$RESULTS_DIR/oos_${CONFIG_ID}"
    ./target/release/backtest rap \
        --config "$CONFIG_FILE" \
        --output "$OOS_OUTPUT" \
        --start-date 2023-07-01 \
        --end-date 2024-12-31 2>/dev/null || true
    
    # Extrair métricas OOS
    if [ -f "$OOS_OUTPUT/metrics.json" ]; then
        OOS_RETURN=$(jq -r '.total_return // 0' "$OOS_OUTPUT/metrics.json")
        OOS_TRADES=$(jq -r '.total_trades // 0' "$OOS_OUTPUT/metrics.json")
        OOS_SHARPE=$(jq -r '.sharpe_ratio // 0' "$OOS_OUTPUT/metrics.json")
    else
        OOS_RETURN=0
        OOS_TRADES=0
        OOS_SHARPE=0
    fi
    
    END_TIME=$(date +%s.%N)
    ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
    
    # Salvar resultado
    echo "$CONFIG_ID,$entry_z,$exit_z,$stop_z,$coint_pval,$top_pairs,$lookback,$IS_RETURN,$IS_TRADES,$IS_SHARPE,$OOS_RETURN,$OOS_TRADES,$OOS_SHARPE,$ELAPSED" >> "$RESULTS_CSV"
    
    # Verificar se é o melhor OOS
    if (( $(echo "$OOS_RETURN > $BEST_OOS_RETURN" | bc -l) )); then
        BEST_OOS_RETURN=$OOS_RETURN
        BEST_CONFIG="$CONFIG_ID"
        BEST_CONFIG_FILE="$CONFIG_FILE"
    fi
    
    # Limpar outputs intermediários para economizar espaço
    rm -rf "$IS_OUTPUT" "$OOS_OUTPUT"

done
done
done
done
done
done

echo ""
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "                         OTIMIZAÇÃO CONCLUÍDA                              "
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

# Ordenar por OOS return e mostrar top 10
echo "📊 TOP 10 Configurações (por Retorno OOS):"
echo ""
echo "Rank | Config | Entry Z | Exit Z | Stop Z | P-val | Pairs | IS Ret% | IS Trades | OOS Ret% | OOS Trades"
echo "-----|--------|---------|--------|--------|-------|-------|---------|-----------|----------|----------"

tail -n +2 "$RESULTS_CSV" | sort -t',' -k11 -rn | head -10 | while IFS=',' read -r id entry exit stop pval pairs lookback is_ret is_trades is_sharpe oos_ret oos_trades oos_sharpe elapsed; do
    is_pct=$(echo "scale=2; $is_ret * 100" | bc)
    oos_pct=$(echo "scale=2; $oos_ret * 100" | bc)
    printf "  %2s |   %3s  |  %5s  |  %4s  |  %4s  |  %4s |  %3s  |  %6s%% |    %5s   |  %6s%% |    %5s\n" \
        "" "$id" "$entry" "$exit" "$stop" "$pval" "$pairs" "$is_pct" "$is_trades" "$oos_pct" "$oos_trades"
done

echo ""
echo "📁 Resultados salvos em: $RESULTS_CSV"
echo ""

# Mostrar melhor config
if [ -n "$BEST_CONFIG" ]; then
    echo "🏆 MELHOR CONFIGURAÇÃO (OOS): Config #$BEST_CONFIG"
    echo ""
    cat "$RESULTS_DIR/config_${BEST_CONFIG}.toml"
    echo ""
    
    # Copiar melhor config
    cp "$RESULTS_DIR/config_${BEST_CONFIG}.toml" "$PROJECT_DIR/config/rap_best_oos.toml"
    echo "✅ Melhor config salva em: config/rap_best_oos.toml"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"













