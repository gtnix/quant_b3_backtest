# Módulo de Validação — Especificação Técnica

**Versão**: 1.0.0  
**Data**: 2026-01-01  
**Status**: A implementar

---

## 1. Visão Geral

O Módulo de Validação é um crate Rust (`backtester_validation`) que:

1. Valida outputs de backtest
2. Gera artefatos legíveis
3. Bloqueia resultados absurdos
4. Fornece diagnóstico para humanos

---

## 2. Estrutura do Crate

```
crates/backtester_validation/
├── Cargo.toml
├── src/
│   ├── lib.rs           # Exports públicos
│   ├── pipeline.rs      # Orquestração do pipeline
│   ├── schema.rs        # Validação de schema strict
│   ├── sanity.rs        # Sanity checks (Sharpe>20, etc)
│   ├── crosscheck.rs    # Recompute e comparação de métricas
│   ├── attribution.rs   # Asset attribution (PnL por ativo)
│   ├── report.rs        # Geração de backtest_report.md
│   └── summary.rs       # validation_summary.json
└── tests/
    ├── schema_tests.rs
    ├── sanity_tests.rs
    └── golden_tests.rs
```

---

## 3. Pipeline de Validação

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE VALIDAÇÃO                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: BacktestArtifacts                                       │
│  ├── metrics.json                                               │
│  ├── nav_history.csv                                            │
│  ├── trades.csv                                                 │
│  └── manifest.json                                              │
│                                                                 │
│  Etapa A: Validação Estrutural ────────────────────────────────│
│  ├── Arquivos existem?                                          │
│  ├── JSON/CSV parseáveis?                                       │
│  └── Campos obrigatórios presentes e não-null?                  │
│       │                                                         │
│       └── Se FAIL → Parar com erro claro                        │
│                                                                 │
│  Etapa B: Invariantes Numéricos ───────────────────────────────│
│  ├── Nenhum NaN/Inf em equity/returns                           │
│  ├── final_nav == nav_history[-1]                               │
│  ├── drawdown ∈ [-1, 0]                                         │
│  └── total_trades == len(trades.csv)                            │
│       │                                                         │
│       └── Se FAIL → Parar com erro claro                        │
│                                                                 │
│  Etapa C: Sanity Checks ───────────────────────────────────────│
│  ├── Sharpe > 10 → WARN                                         │
│  ├── Sharpe > 20 → FAIL                                         │
│  ├── Vol < 1% com retornos altos → WARN/FAIL                    │
│  ├── CAGR > 200% em equities → WARN                             │
│  └── num_trades < 30 → WARN                                     │
│       │                                                         │
│       └── Flags em sanity.json                                  │
│                                                                 │
│  Etapa D: Cross-check de Métricas ─────────────────────────────│
│  ├── Recomputar CAGR, Vol, Sharpe, MaxDD de nav_history         │
│  ├── Comparar com metrics.json                                  │
│  └── Divergência > 0.1% → FAIL                                  │
│                                                                 │
│  Etapa E: Asset Attribution ───────────────────────────────────│
│  ├── Calcular PnL por ativo de trades.csv                       │
│  ├── Gerar asset_attribution.csv                                │
│  └── Alertar se concentração > 80% em 1 ativo                   │
│                                                                 │
│  Output: ValidationResult                                       │
│  ├── validation_summary.json                                    │
│  ├── sanity.json                                                │
│  ├── asset_attribution.csv                                      │
│  └── backtest_report.md                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Tipos Principais

### 4.1 BacktestArtifacts (Input)

```rust
pub struct BacktestArtifacts {
    pub run_id: String,
    pub metrics_path: PathBuf,
    pub nav_history_path: PathBuf,
    pub trades_path: PathBuf,
    pub manifest_path: PathBuf,
}
```

### 4.2 ValidationResult (Output)

```rust
pub struct ValidationResult {
    pub verdict: Verdict,
    pub schema_check: SchemaCheckResult,
    pub sanity_check: SanityCheckResult,
    pub crosscheck: CrosscheckResult,
    pub attribution: AttributionResult,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
}

pub enum Verdict {
    Pass,
    Warn,
    Fail,
}
```

### 4.3 SanityCheckResult

```rust
pub struct SanityCheckResult {
    pub sharpe_suspicious: bool,      // Sharpe > 10
    pub sharpe_absurd: bool,          // Sharpe > 20
    pub vol_too_low: bool,            // Vol < 1%
    pub trades_too_few: bool,         // Trades < 30
    pub cagr_unrealistic: bool,       // CAGR > 200%
    pub equity_monotonic: bool,       // Muitos dias sem variação
    pub verdict: Verdict,
    pub message: String,
}
```

### 4.4 AssetAttribution

```rust
pub struct AssetAttribution {
    pub symbol: String,
    pub net_pnl: f64,
    pub gross_pnl: f64,
    pub total_costs: f64,
    pub num_trades: u32,
    pub win_rate: f64,
    pub avg_trade_pnl: f64,
    pub contribution_pct: f64,
}
```

---

## 5. Artefatos Gerados

### 5.1 validation_summary.json

```json
{
  "run_id": "exp_12345",
  "verdict": "WARN",
  "checks": {
    "schema": "PASS",
    "invariants": "PASS",
    "sanity": "WARN",
    "crosscheck": "PASS"
  },
  "warnings": [
    "Sharpe ratio 12.5 > 10 (suspicious)"
  ],
  "errors": [],
  "generated_at": "2026-01-01T12:00:00Z"
}
```

### 5.2 sanity.json

```json
{
  "sharpe_ratio": 12.5,
  "annual_volatility": 0.08,
  "cagr": 0.45,
  "num_trades": 85,
  "flags": {
    "sharpe_suspicious": true,
    "sharpe_absurd": false,
    "vol_too_low": false,
    "trades_too_few": false
  },
  "verdict": "WARN",
  "message": "Sharpe > 10: investigar volatilidade baixa"
}
```

### 5.3 asset_attribution.csv

```csv
symbol,net_pnl,gross_pnl,total_costs,num_trades,win_rate,avg_trade_pnl,contribution_pct
PETR4,12500.00,13200.00,700.00,45,0.62,277.78,35.2
VALE3,8200.00,8800.00,600.00,32,0.58,256.25,23.1
ITUB4,-3500.00,-3100.00,400.00,28,0.42,-125.00,-9.9
...
```

### 5.4 backtest_report.md

```markdown
# Backtest Report

**Run ID**: exp_12345
**Período**: 2020-01-02 a 2024-12-31
**Capital Inicial**: R$ 1.000.000,00

---

## Métricas Principais

| Métrica | Valor |
|---------|-------|
| CAGR | 15.2% |
| Volatilidade | 18.5% |
| Sharpe Ratio | 0.82 |
| Max Drawdown | -22.3% |
| Trades | 342 |
| Win Rate | 54% |

---

## Alertas

⚠️ Nenhum alerta crítico.

---

## Melhores Papéis (Top 5)

| Ativo | PnL Net | Trades | Contribuição |
|-------|---------|--------|--------------|
| PETR4 | R$ 125.000 | 45 | 35.2% |
| VALE3 | R$ 82.000 | 32 | 23.1% |
...

## Piores Papéis (Bottom 5)

| Ativo | PnL Net | Trades | Contribuição |
|-------|---------|--------|--------------|
| ITUB4 | -R$ 35.000 | 28 | -9.9% |
...

---

## Conclusão

**Veredito**: ✅ PASS

O backtest passou em todas as validações.
```

---

## 6. Integração

### 6.1 Com CLI

```bash
# Rodar backtest com validação
backtester run --config strategy.toml --validate

# Validar run existente
backtester validate --run-id exp_12345
```

### 6.2 Com SCG

O pipeline SCG integra validação no Stage B:

```rust
// Após backtest de candidato top-K
let artifacts = BacktestArtifacts::from_run(run_id);
let validation = ValidationPipeline::new(config);
let result = validation.run(&artifacts)?;

if result.verdict == Verdict::Fail {
    // Descartar candidato
}
```

---

## 7. Configuração

```toml
[validation]
# Etapas habilitadas
schema_check = true
sanity_check = true
crosscheck = true
attribution = true

[validation.sanity]
# Thresholds
sharpe_warn = 10.0
sharpe_fail = 20.0
vol_min = 0.01
trades_min = 30
cagr_max = 2.0

[validation.crosscheck]
# Tolerância para divergência
tolerance_pct = 0.1

[validation.attribution]
# Concentração que dispara warning
concentration_warn = 0.8
```

---

## 8. Definition of Done

- [ ] Crate `backtester_validation` criado e compila
- [ ] Pipeline executa etapas A-E em sequência
- [ ] Sharpe > 20 gera FAIL com mensagem clara
- [ ] Campos null em métricas obrigatórias gera FAIL
- [ ] Cross-check detecta divergência > 0.1%
- [ ] `asset_attribution.csv` gerado corretamente
- [ ] `backtest_report.md` legível gerado
- [ ] Integrado com `backtester_cli` e `combiner` 
- [ ] Testes cobrindo happy path e edge cases
- [ ] Documentação atualizada

---

## 9. Referências

- `docs/quant_audit_pack/03_MODULO_VALIDACAO_SPEC.md` — Spec original
- `docs/quant_audit_pack/06_GAPS_VALIDACAO_E_REPORTS.md` — Gaps identificados
- `crates/combiner_engine/src/validation.rs` — Validação existente (WFA/PBO)


