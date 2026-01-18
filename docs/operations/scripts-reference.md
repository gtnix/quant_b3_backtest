# Scripts Reference

**Versão**: 1.0.0  
**Última Atualização**: 2026-01-18

## Visão Geral

O diretório `scripts/` contém shell scripts e Python scripts para operações do sistema.

---

## Scripts Principais

### `radar.sh`

Executa campanhas "radar" - descoberta rápida de estratégias.

```bash
./scripts/radar.sh [OPTIONS]
```

**Opções:**

| Flag | Descrição | Default |
|------|-----------|---------|
| `--config <path>` | Config file | `configs/scg_60min_fullcores.toml` |
| `--duration <min>` | Duração em minutos | 60 |
| `--output <dir>` | Diretório de output | `output/scg` |

**Exemplo:**

```bash
# Radar de 2 horas
./scripts/radar.sh --duration 120

# Com config específica
./scripts/radar.sh --config configs/campaigns/momentum.toml
```

---

### `sync_b3.sh`

Sincroniza dados B3 (índices, daily, intraday).

```bash
./scripts/sync_b3.sh
```

**Executa:**

1. `python -m datahub_b3 sync` - Índices
2. `python -m datahub_b3 daily-sync --range 1mo` - Daily
3. `python -m datahub_b3 intraday-sync --interval 30m --range 5d` - Intraday

**Variáveis:**

| Variável | Descrição |
|----------|-----------|
| `NEON_DATABASE_URL` | Connection string PostgreSQL |
| `BRAPI_TOKEN` | Token Brapi API |

---

### `sync_us.sh`

Sincroniza dados US (índices, daily, intraday, dividendos).

```bash
./scripts/sync_us.sh
```

**Executa:**

1. `python -m datahub_us indices-sync` - Índices
2. `python -m datahub_us daily-sync --period 1mo` - Daily
3. `python -m datahub_us intraday-sync --interval 30m --period 5d` - Intraday
4. `python -m datahub_us dividends-sync --skip-existing` - Dividendos

---

### `hof_sync.py`

Sincroniza Hall of Fame com banco Neon.

```bash
python scripts/hof_sync.py [OPTIONS]
```

**Opções:**

| Flag | Descrição | Default |
|------|-----------|---------|
| `--dry-run` | Não fazer alterações | false |
| `--force` | Forçar re-sync | false |

---

### `cleanup_old_runs.sh`

Limpa runs antigos para economizar espaço.

```bash
./scripts/cleanup_old_runs.sh [OPTIONS]
```

**Opções:**

| Flag | Descrição | Default |
|------|-----------|---------|
| `--days <N>` | Manter runs dos últimos N dias | 30 |
| `--dry-run` | Apenas mostrar o que seria deletado | false |

**Exemplo:**

```bash
# Ver o que seria deletado
./scripts/cleanup_old_runs.sh --days 14 --dry-run

# Executar limpeza
./scripts/cleanup_old_runs.sh --days 14
```

---

### `auto_cleanup.sh`

Limpeza automática de cache e arquivos temporários.

```bash
./scripts/auto_cleanup.sh
```

**Remove:**

- `target/debug/` (exceto binários)
- Runs com mais de 30 dias
- Logs com mais de 7 dias
- Cache expirado

---

## Scripts de Auditoria

### `audit_2h.sh` / `audit_4h.sh`

Executa auditoria institucional de runs SCG.

```bash
./scripts/audit_2h.sh <RUN_DIR>
./scripts/audit_4h.sh <RUN_DIR>
```

**Diferença:**

| Script | Timeout | Uso |
|--------|---------|-----|
| `audit_2h.sh` | 2 horas | Runs pequenos |
| `audit_4h.sh` | 4 horas | Runs grandes |

---

## Scripts de Deploy

### `deploy.sh`

Deploy script (DEFERRED - VPS not in scope).

> **NOTA**: VPS deployment is DEFERRED. See `docs/ops/local_only_policy.md`.

```bash
./scripts/deploy.sh   # DEFERRED
```

---

### `start.sh` / `stop.sh`

Iniciar/parar serviços locais.

```bash
./scripts/start.sh   # Inicia dashboard e API
./scripts/stop.sh    # Para todos os serviços
```

---

### `setup-vps.sh` - DEFERRED

> **NOTA**: VPS setup is DEFERRED. See `docs/ops/local_only_policy.md`.

---

## Scripts de Calendário

### `calendar_scraper/`

Scripts para atualização de calendários.

#### `b3_scraper.py`

```bash
python scripts/calendar_scraper/b3_scraper.py [YEAR]
```

Atualiza calendário B3 para o ano especificado.

#### `nyse_scraper.py`

```bash
python scripts/calendar_scraper/nyse_scraper.py [YEAR]
```

Atualiza calendário NYSE.

#### `generate_diff.py`

```bash
python scripts/calendar_scraper/generate_diff.py
```

Gera diff entre calendários antigos e novos.

---

### `calendar_validation/pregoes_validator.py`

Valida consistência dos pregões.

```bash
python scripts/calendar_validation/pregoes_validator.py
```

---

## Scripts de Build

### `clean_build_cache.sh`

Limpa cache de build Rust.

```bash
./scripts/clean_build_cache.sh
```

**Remove:**

- `target/debug/incremental/`
- `target/release/incremental/`
- Artefatos antigos

---

### `export_benchmarks.sh`

Exporta resultados de benchmarks.

```bash
./scripts/export_benchmarks.sh
```

**Output:**

- `benches/results/baseline.json`
- `benches/results/benchmark.json`

---

### `export_market_data.py`

Exporta dados de mercado para CSV.

```bash
python scripts/export_market_data.py [OPTIONS]
```

**Opções:**

| Flag | Descrição |
|------|-----------|
| `--market <br\|us>` | Mercado |
| `--output <path>` | Diretório de output |
| `--symbols <list>` | Símbolos específicos |

---

## Scripts VPS - DEFERRED

> **NOTA**: VPS scripts are DEFERRED. See `docs/ops/local_only_policy.md`.
> These scripts exist in `scripts/vps/` for historical reference only.

---

## Scripts de Teste

### `test_cockpit.sh`

Testa endpoints do Cockpit.

```bash
./scripts/test_cockpit.sh
```

---

### `optimize_rap.sh`

Otimização de parâmetros RAP (Risk-Adjusted Performance).

```bash
./scripts/optimize_rap.sh
```

---

## Cron Jobs Recomendados

```bash
# Sync B3 (21:30 UTC, Mon-Fri)
30 21 * * 1-5 /path/to/scripts/sync_b3.sh >> /var/log/sync_b3.log 2>&1

# Sync US (22:00 UTC, Mon-Fri)
0 22 * * 1-5 /path/to/scripts/sync_us.sh >> /var/log/sync_us.log 2>&1

# Cleanup (01:00 UTC, Sunday)
0 1 * * 0 /path/to/scripts/cleanup_old_runs.sh --days 30 >> /var/log/cleanup.log 2>&1

# Local health check (every 5 min) - configure as needed
# */5 * * * * /path/to/scripts/local-health-check.sh >> /var/log/health.log 2>&1
```

---

## Localização

```
scripts/
├── radar.sh                  # Campanhas radar
├── sync_b3.sh                # Sync dados B3
├── sync_us.sh                # Sync dados US
├── hof_sync.py               # Sync Hall of Fame
├── cleanup_old_runs.sh       # Limpeza de runs
├── auto_cleanup.sh           # Limpeza automática
├── audit_2h.sh               # Auditoria 2h
├── audit_4h.sh               # Auditoria 4h
├── deploy.sh                 # Deploy (DEFERRED)
├── start.sh                  # Iniciar serviços
├── stop.sh                   # Parar serviços
├── setup-vps.sh              # Setup VPS (DEFERRED)
├── clean_build_cache.sh      # Limpar build cache
├── export_benchmarks.sh      # Exportar benchmarks
├── export_market_data.py     # Exportar dados
├── test_cockpit.sh           # Testar cockpit
├── optimize_rap.sh           # Otimizar RAP
├── calendar_scraper/         # Scrapers de calendário
│   ├── b3_scraper.py
│   ├── nyse_scraper.py
│   └── generate_diff.py
├── calendar_validation/      # Validação de calendário
│   └── pregoes_validator.py
└── vps/                      # Scripts VPS (DEFERRED)
    ├── ecosystem.config.cjs
    ├── nginx-dashboard.conf
    ├── health-check.sh
    ├── restart-services.sh
    ├── setup-auth.sh
    ├── sync-from-git.sh
    └── vps-sync.sh
```
