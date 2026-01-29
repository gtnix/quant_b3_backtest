# Setup Guide

Complete setup instructions for running the Quant B3 Backtest system on a new machine.

## Prerequisites

| Tool | Version | Installation |
|------|---------|--------------|
| Rust | 1.75+ | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| Node.js | 18+ | `nvm install 18` or https://nodejs.org |
| Python | 3.11+ | `pyenv install 3.11` or system package |
| PostgreSQL client | any | `sudo apt install postgresql-client` |

## 1. Clone and Configure

```bash
git clone <repo-url>
cd quant_b3_backtest

# Create environment file
cp .env.example .env

# Edit .env with your credentials:
# - DATABASE_URL: Your Neon PostgreSQL connection string
# - BRAPI_API_KEY: Your Brapi API key (get at https://brapi.dev/)
nano .env
```

## 2. Build Rust Binaries

```bash
# Build release binaries (required)
cargo build --release --bin combiner --bin backtest

# Verify binaries exist
./target/release/combiner --version
./target/release/backtest --version
```

## 3. Setup Dashboard

```bash
cd dashboard
npm install
cd ..
```

## 4. Setup Python DataHubs

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all datahub dependencies
pip install -r datahub_b3/requirements.txt
pip install -r datahub_us/requirements.txt
pip install -r datahub_fx/requirements.txt
```

## 5. Run Database Migrations

The database schema is managed via SQL migration files in `/migrations/`.

```bash
# Run all migrations in order
for f in migrations/*.sql; do
  echo "Running $f..."
  psql "$DATABASE_URL" -f "$f"
done
```

## 6. Sync Market Data

```bash
source .venv/bin/activate

# Sync Brazilian market data (B3)
python3 -m datahub_b3 full-sync

# Sync US market data (S&P 500)
python3 -m datahub_us bootstrap
python3 -m datahub_us update

# Sync FX rates
python3 -m datahub_fx sync

# Export to local CSV cache
python3 scripts/sync_all_data.py --export
```

## 7. Verify Installation

```bash
# 1. Rust binaries work
./target/release/backtest --help
./target/release/combiner --help

# 2. Dashboard starts
cd dashboard && npm run dev &
# Open http://localhost:5173

# 3. Python datahubs work
source .venv/bin/activate
python3 -m datahub_b3 --help
python3 -m datahub_us --help

# 4. Database connection works
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM ohlcv_daily_us;"

# 5. Run a quick backtest
./target/release/backtest run --config configs/strategies/golden_momentum.toml
```

## Quick Commands Reference

| Task | Command |
|------|---------|
| Start dashboard | `cd dashboard && npm run dev` |
| Run evolution (mining) | `./target/release/combiner run --config configs/default.toml --ultra` |
| Update BR data | `python3 -m datahub_b3 full-sync` |
| Update US data | `python3 -m datahub_us update` |
| Run backtest | `./target/release/backtest run --config configs/strategies/<name>.toml` |

## Troubleshooting

### "DATABASE_URL not configured"
Ensure your `.env` file exists and contains a valid `DATABASE_URL`.

### Rust build fails
```bash
rustup update stable
cargo clean
cargo build --release
```

### Python import errors
```bash
source .venv/bin/activate
pip install --upgrade pip
pip install -r datahub_b3/requirements.txt -r datahub_us/requirements.txt -r datahub_fx/requirements.txt
```

### Dashboard won't start
```bash
cd dashboard
rm -rf node_modules package-lock.json
npm install
npm run dev
```
