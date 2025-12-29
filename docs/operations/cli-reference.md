# Referência CLI

**Versão**: 2.0.0  
**Última Atualização**: 2025-12-28

## Visão Geral

O `backtester_cli` fornece comandos para execução de estratégias, comparação de resultados e geração de documentação.

### Localização no Código

- **Crate**: `backtester_cli`
- **Arquivo**: `src/main.rs`

---

## Comandos Disponíveis

```bash
# Ver ajuda
cargo run -p backtester_cli -- --help

# Ver versão
cargo run -p backtester_cli -- --version
```

---

## `run` - Executar Estratégia

Executa uma única configuração de estratégia.

```bash
cargo run -p backtester_cli -- run --config <path> [options]
```

### Opções

| Flag | Descrição | Default |
|------|-----------|---------|
| `--config <path>` | Caminho para arquivo TOML | (obrigatório) |
| `--output <dir>` | Diretório de output | `output/experiments` |
| `--dry-run` | Apenas validar (sem executar) | `false` |
| `--strict` | Falhar em violações de invariantes | `false` |
| `--execution <mode>` | Modo de execução | `auto` |

### Modos de Execução

| Modo | Descrição | Performance |
|------|-----------|-------------|
| `standard` | Compositor dinâmico | Baseline |
| `compiled` | Estratégia pré-compilada | 5-10% faster |
| `fast` | SoA + zero alocações | 93-124x faster |
| `auto` | Seleciona melhor modo | Varia |

### Exemplos

```bash
# Execução básica
cargo run -p backtester_cli -- run \
  --config configs/strategies/golden_momentum.toml

# Dry run (validação)
cargo run -p backtester_cli -- run \
  --config configs/strategies/my_strategy.toml \
  --dry-run

# Modo fast com strict
cargo run -p backtester_cli -- run \
  --config configs/strategies/my_strategy.toml \
  --execution fast \
  --strict

# Output customizado
cargo run -p backtester_cli -- run \
  --config configs/strategies/my_strategy.toml \
  --output output/my_experiments
```

---

## `run-batch` - Executar Múltiplas Estratégias

Executa todas as estratégias em um diretório.

```bash
cargo run -p backtester_cli -- run-batch --folder <path> [options]
```

### Opções

| Flag | Descrição | Default |
|------|-----------|---------|
| `--folder <path>` | Diretório com TOMLs | (obrigatório) |
| `--output <dir>` | Diretório de output | `output/experiments` |
| `--strict` | Falhar em violações | `false` |

### Exemplo

```bash
cargo run -p backtester_cli -- run-batch \
  --folder configs/strategies \
  --strict
```

---

## `compare` - Comparar Dois Runs

Compara métricas entre dois experimentos.

```bash
cargo run -p backtester_cli -- compare --run-a <path> --run-b <path>
```

### Opções

| Flag | Descrição |
|------|-----------|
| `--run-a <path>` | Caminho do primeiro run |
| `--run-b <path>` | Caminho do segundo run |
| `--sharpe-threshold <pct>` | Threshold de drop Sharpe |
| `--cagr-threshold <pct>` | Threshold de drop CAGR |
| `--dd-threshold <pct>` | Threshold de aumento DD |

### Exemplo

```bash
cargo run -p backtester_cli -- compare \
  --run-a output/experiments/run-001 \
  --run-b output/experiments/run-002 \
  --sharpe-threshold 0.15
```

---

## `compare-to-golden` - Comparar com Baseline

Compara um run contra uma estratégia golden.

```bash
cargo run -p backtester_cli -- compare-to-golden --run <path> --golden <id>
```

### Opções

| Flag | Descrição | Default |
|------|-----------|---------|
| `--run <path>` | Caminho do run | (obrigatório) |
| `--golden <id>` | ID da golden strategy | (obrigatório) |
| `--golden-dir <path>` | Diretório das goldens | `output/experiments` |

### Golden Strategies Disponíveis

| ID | Descrição |
|----|-----------|
| `golden_momentum` | Momentum puro + equal weight |
| `golden_value_quality` | Value + Quality + LowVol + risk parity |
| `golden_trend_vol` | MA crossover + trailing + vol targeting |

### Critérios de Regressão

- Sharpe drop > 20%
- Max DD increase > 25%
- CAGR drop > 30%

### Exemplo

```bash
cargo run -p backtester_cli -- compare-to-golden \
  --run output/experiments/my-run \
  --golden golden_momentum
```

---

## `generate-catalog` - Gerar Catálogo de Blocos

Gera documentação dos blocos disponíveis.

```bash
cargo run -p backtester_cli -- generate-catalog --output <path> [--json]
```

### Opções

| Flag | Descrição | Default |
|------|-----------|---------|
| `--output <path>` | Arquivo de output | (obrigatório) |
| `--json` | Também gerar JSON | `false` |

### Exemplo

```bash
# Gerar Markdown
cargo run -p backtester_cli -- generate-catalog \
  --output docs/strategies/block-catalog.md

# Gerar Markdown + JSON
cargo run -p backtester_cli -- generate-catalog \
  --output docs/strategies/block-catalog.md \
  --json
```

---

## Códigos de Saída

| Código | Significado |
|--------|-------------|
| 0 | Sucesso |
| 1 | Erro de configuração |
| 2 | Erro de execução |
| 3 | Regressão detectada |

---

## Variáveis de Ambiente

| Variável | Descrição |
|----------|-----------|
| `RUST_LOG` | Nível de logging |
| `FRED_API_KEY` | API key para FRED (interest rates) |

### Exemplo

```bash
RUST_LOG=debug cargo run -p backtester_cli -- run \
  --config configs/strategies/my_strategy.toml
```






