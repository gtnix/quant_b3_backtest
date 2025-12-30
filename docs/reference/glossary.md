# Glossário

**Versão**: 2.1.0  
**Última Atualização**: 2025-12-29

## Termos Técnicos

### A

**Adjusted Price**: Preço de fechamento ajustado para dividendos e splits. Usado para cálculo de indicadores e sinais.

**Anti-Double-Count**: Política que previne contagem dupla de dividendos ao usar preços raw para valuation e adjusted para signals.

**Artifact**: Arquivo de output gerado por um run de experimento (metadata.json, metrics.json, timeseries.csv, trace.jsonl).

**AoS (Array of Structures)**: Layout de dados onde cada elemento é uma struct completa. Menos eficiente para cache que SoA.

### B

**Bar**: Candle OHLCV (Open, High, Low, Close, Volume) representando um período de trading.

**Block**: Unidade modular de lógica de estratégia (selection, entry, exit, sizing).

**BlockRegistry**: Registro que mapeia block_id para implementações de StrategyBlock.

**Browser Mode**: Modo de execução do Dashboard via navegador web, usando API Server (Express) + Neon PostgreSQL. Alternativa ao Desktop Mode (Tauri).

### C

**CAGR (Compound Annual Growth Rate)**: Retorno anualizado composto. Fórmula: `(end/start)^(1/years) - 1`.

**Calmar Ratio**: CAGR dividido pelo máximo drawdown (absoluto).

**Cockpit**: Painel de controle do Dashboard para orquestração de runs SCG. Permite configurar presets, gates, ranking methods e monitorar progresso em tempo real.

**Compositor**: Executor de pipeline de estratégia que encadeia blocks.

**CompiledStrategy**: Estratégia pré-compilada com params tipados para performance.

### D

**Determinism**: Propriedade de produzir outputs idênticos para inputs idênticos.

**Dividend**: Pagamento de proventos por ação. Creditado no ex-date.

**Drawdown**: Queda do equity em relação ao pico histórico (high-water mark).

**DSR (Deflated Sharpe Ratio)**: Sharpe Ratio ajustado para múltiplas comparações. Penaliza estratégias testadas muitas vezes. Valores >1.0 indicam edge estatístico genuíno.

**DualPriceBar**: Estrutura com preços adjusted e raw para a mesma data.

### E

**Eligibility**: Condição de um ativo estar disponível para trading em uma data específica.

**Entry Engine**: Componente que aplica gating filters e gera sinais de entrada.

**Ex-Date**: Data a partir da qual o comprador não tem direito ao dividendo.

**Exit Engine**: Componente que verifica condições de saída (stop-loss, take-profit).

### F

**Fast Mode**: Modo de execução otimizado com SoA e zero alocações (93-124x speedup).

**FX**: Foreign Exchange - taxa de câmbio entre moedas.

**FxPair**: Par de moedas (base/quote). USD/BRL = 5.50 significa 1 USD = 5.50 BRL.

### G

**Gates**: Thresholds de validação configuráveis no Cockpit (minOosSharpeNet, maxPbo, minStressPassed). Estratégias que não passam nos gates são filtradas.

**Gating**: Filtros aplicados antes de um ativo ser considerado candidato.

**Golden Strategy**: Estratégia baseline para testes de regressão.

### H

**HHI (Herfindahl-Hirschman Index)**: Métrica de concentração. `Σ(weight_i)²`.

**Hot Path**: Caminho de código crítico para performance, executado frequentemente.

### I

**Invariant**: Condição que deve sempre ser verdadeira no sistema.

### L

**LOCF (Last Observation Carried Forward)**: Usar último valor disponível quando dado está faltando.

### M

**Mark-to-Market**: Avaliar posições a preço de mercado atual.

**Max Drawdown**: Maior drawdown observado durante o período.

### N

**NAV (Net Asset Value)**: Valor líquido do portfólio (posições + cash).

**Netting**: Consolidação de ordens opostas (buy - sell).

### O

**Orchestrator**: Componente que coordena EntryEngine, ExitEngine e netting.

### P

**PBO (Probability of Backtest Overfitting)**: Probabilidade de uma estratégia ter performado bem por sorte. Valores ≤0.15 indicam baixo risco de overfitting. Calculado via Combinatorially Symmetric Cross-Validation.

**Pipeline**: Sequência de blocks executados em ordem.

**PolicyViolation**: Erro retornado quando configuração viola política (ex: anti-double-count).

**Prealloc**: Buffers pré-alocados para evitar alocações no hot path.

**Preset**: Perfil de configuração pré-definido no Cockpit. Tipos: Rapid (3min debug), Institutional (15min produção), Exhaustive (1h exploração máxima).

**Profit Factor**: Lucro bruto dividido por perda bruta.

### R

**Raw Price**: Preço de fechamento não ajustado. Usado para valuation quando dividends são cashflow.

**Rebalance**: Ajuste de pesos do portfólio para targets desejados.

**Regression**: Degradação de performance em comparação com baseline.

**Risk Parity**: Alocação inversamente proporcional à volatilidade.

### S

**Sharpe Ratio**: `(return - risk_free) / volatility`. Mede retorno ajustado a risco.

**Signal**: Indicação de ação (buy, sell, hold) gerada por um block.

**SoA (Structure of Arrays)**: Layout de dados onde cada campo é um array separado. Eficiente para cache.

**Sortino Ratio**: Similar a Sharpe, mas usa apenas downside volatility.

**SSE (Server-Sent Events)**: Protocolo HTTP para streaming unidirecional server→client. Usado no Browser Mode para atualizações em tempo real de progresso do SCG.

**Strict Mode**: Modo que falha o run se invariantes forem violadas.

**Survivorship Bias**: Viés de incluir apenas ativos que "sobreviveram" até o presente.

**SymbolTable**: Mapeamento O(1) symbol ↔ u16 ID para performance.

### T

**Trace**: Log de execução do pipeline (trace.jsonl).

**Trailing Stop**: Stop que move junto com o preço conforme sobe.

**Turnover**: Volume de trading relativo ao tamanho do portfólio.

### U

**UnifiedEngine**: Engine canônico de simulação com precisão decimal e suporte a dividendos.

**Universe**: Conjunto de ativos disponíveis para seleção.

**Universe Eligibility**: Verificação se ativo existia na data de rebalance.

### V

**Valuation**: Avaliação de posições a preço de mercado.

**Volatility**: Desvio padrão de retornos, tipicamente anualizado (×√252).

**Vol Targeting**: Ajuste de posições para atingir volatilidade alvo do portfólio.

### W

**Weight**: Proporção do portfólio alocada a um ativo (0.0 - 1.0).

### Z

**Zero-Alloc**: Execução sem alocações de memória no hot path.

---

## Constantes

| Constante | Valor | Descrição |
|-----------|-------|-----------|
| `TRADING_DAYS_PER_YEAR` | 252 | Dias de trading por ano |
| `DEFAULT_RISK_FREE_RATE` | 0.05 | Taxa livre de risco (5% a.a.) |
| `WEIGHT_SUM_TOLERANCE` | 0.001 | Tolerância para soma de pesos |
| `MIN_VOLATILITY_THRESHOLD` | 0.0001 | Vol mínima para evitar divisão por zero |
| `MAX_RATIO_VALUE` | 999.99 | Cap para ratios infinitos |

---

## Acrônimos

| Acrônimo | Significado |
|----------|-------------|
| ADR | Architecture Decision Record |
| AoS | Array of Structures |
| B3 | Brasil, Bolsa, Balcão |
| CAGR | Compound Annual Growth Rate |
| CLI | Command Line Interface |
| DSL | Domain Specific Language |
| DSR | Deflated Sharpe Ratio |
| EOD | End of Day |
| FX | Foreign Exchange |
| HHI | Herfindahl-Hirschman Index |
| LOCF | Last Observation Carried Forward |
| NAV | Net Asset Value |
| OHLCV | Open, High, Low, Close, Volume |
| OOS | Out-of-Sample |
| PBO | Probability of Backtest Overfitting |
| PnL | Profit and Loss |
| SCG | Sistema Combinador Generativo |
| SoA | Structure of Arrays |
| SSE | Server-Sent Events |
| WFA | Walk-Forward Analysis |






