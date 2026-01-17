//! Database layer for Neon Postgres.

use chrono::{NaiveDate, Utc};
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;
use thiserror::Error;
use tokio_postgres::Client;
use tracing::{debug, info};

use crate::brapi::{
    DividendEntry, FundamentalSnapshot, HistoricalBar, QuoteResult, RequestMetrics, StockInfo,
    SummaryProfile, TickerCapabilities,
};
use crate::interest_rates::{InterestRateEntry, InterestRateStats};

fn f64_to_decimal(v: Option<f64>) -> Option<Decimal> {
    v.and_then(Decimal::from_f64)
}

#[derive(Error, Debug)]
pub enum DbError {
    #[error("Connection error: {0}")]
    Connection(String),
    #[error("Query error: {0}")]
    Query(#[from] tokio_postgres::Error),
    #[error("Config error: {0}")]
    Config(String),
}

pub struct Database {
    client: Client,
}

impl Database {
    pub async fn connect() -> Result<Self, DbError> {
        let database_url = std::env::var("DATABASE_URL")
            .map_err(|_| DbError::Config("DATABASE_URL not set".into()))?;

        // Install crypto provider (required for rustls 0.23+)
        let _ = rustls::crypto::ring::default_provider().install_default();

        // Use rustls for SSL
        let root_store =
            rustls::RootCertStore::from_iter(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

        let config = rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();

        let tls = tokio_postgres_rustls::MakeRustlsConnect::new(config);

        let (client, connection) = tokio_postgres::connect(&database_url, tls)
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;

        tokio::spawn(async move {
            if let Err(e) = connection.await {
                eprintln!("Database connection error: {}", e);
            }
        });

        info!("Connected to Neon database");
        Ok(Self { client })
    }

    /// Get reference to underlying client for direct queries.
    pub fn client(&self) -> &Client {
        &self.client
    }

    pub async fn verify_schema(&self) -> Result<(), DbError> {
        let tables = [
            "instruments",
            "universe_membership",
            "ohlcv_daily",
            "ingestion_state",
            "api_request_log",
            "api_budget",
        ];

        for table in tables {
            let row = self
                .client
                .query_one(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = $1",
                    &[&table],
                )
                .await?;
            let count: i64 = row.get(0);
            if count == 0 {
                return Err(DbError::Config(format!("Table '{}' not found", table)));
            }
            debug!("Table '{}' exists", table);
        }
        Ok(())
    }

    /// Log an API request.
    pub async fn log_request(
        &self,
        endpoint: &str,
        tickers: &[&str],
        metrics: &RequestMetrics,
        error: Option<&str>,
    ) -> Result<(), DbError> {
        let tickers_sample = if tickers.len() <= 5 {
            tickers.join(",")
        } else {
            format!("{}...(+{})", tickers[..3].join(","), tickers.len() - 3)
        };

        self.client
            .execute(
                "INSERT INTO api_request_log
                 (endpoint, tickers_count, tickers_sample, http_status, error_code, duration_ms, response_bytes)
                 VALUES ($1, $2, $3, $4, $5, $6, $7)",
                &[
                    &endpoint,
                    &(tickers.len() as i32),
                    &tickers_sample,
                    &(metrics.status as i32),
                    &error,
                    &(metrics.duration_ms as i32),
                    &(metrics.response_bytes as i32),
                ],
            )
            .await?;
        Ok(())
    }

    /// Increment monthly API budget counter.
    pub async fn increment_budget(&self) -> Result<(i32, i32), DbError> {
        let month_key = Utc::now().format("%Y-%m").to_string();

        self.client
            .execute(
                "INSERT INTO api_budget (month_key, requests_used)
                 VALUES ($1, 1)
                 ON CONFLICT (month_key) DO UPDATE SET
                   requests_used = api_budget.requests_used + 1,
                   last_updated_at = NOW()",
                &[&month_key],
            )
            .await?;

        let row = self
            .client
            .query_one(
                "SELECT requests_used, requests_limit FROM api_budget WHERE month_key = $1",
                &[&month_key],
            )
            .await?;

        Ok((row.get(0), row.get(1)))
    }

    /// Upsert instrument.
    pub async fn upsert_instrument(&self, info: &StockInfo) -> Result<(), DbError> {
        self.client
            .execute(
                "INSERT INTO instruments (symbol, name, short_name, sector, market_cap, updated_at)
                 VALUES ($1, $2, $2, $3, $4, NOW())
                 ON CONFLICT (symbol) DO UPDATE SET
                   name = COALESCE(EXCLUDED.name, instruments.name),
                   sector = COALESCE(EXCLUDED.sector, instruments.sector),
                   market_cap = COALESCE(EXCLUDED.market_cap, instruments.market_cap),
                   updated_at = NOW()",
                &[&info.stock, &info.name, &info.sector, &info.market_cap],
            )
            .await?;
        Ok(())
    }

    /// Ensure instrument exists (minimal insert for foreign key).
    pub async fn ensure_instrument_exists(&self, symbol: &str) -> Result<(), DbError> {
        self.client
            .execute(
                "INSERT INTO instruments (symbol, name, updated_at)
                 VALUES ($1, $1, NOW())
                 ON CONFLICT (symbol) DO NOTHING",
                &[&symbol],
            )
            .await?;
        Ok(())
    }

    /// Upsert instrument from quote result.
    pub async fn upsert_instrument_from_quote(&self, quote: &QuoteResult) -> Result<(), DbError> {
        let market_cap = quote.market_cap_i64();
        self.client
            .execute(
                "INSERT INTO instruments (symbol, name, short_name, currency, market_cap, updated_at)
                 VALUES ($1, $2, $3, $4, $5, NOW())
                 ON CONFLICT (symbol) DO UPDATE SET
                   name = COALESCE(EXCLUDED.name, instruments.name),
                   short_name = COALESCE(EXCLUDED.short_name, instruments.short_name),
                   market_cap = COALESCE(EXCLUDED.market_cap, instruments.market_cap),
                   updated_at = NOW()",
                &[
                    &quote.symbol,
                    &quote.long_name,
                    &quote.short_name,
                    &quote.currency,
                    &market_cap,
                ],
            )
            .await?;
        Ok(())
    }

    /// Insert universe membership.
    pub async fn insert_universe_member(
        &self,
        universe: &str,
        symbol: &str,
        rank: i32,
        as_of_date: NaiveDate,
    ) -> Result<(), DbError> {
        self.client
            .execute(
                "INSERT INTO universe_membership (universe_name, symbol, rank, as_of_date)
                 VALUES ($1, $2, $3, $4)
                 ON CONFLICT (universe_name, symbol, as_of_date) DO UPDATE SET rank = EXCLUDED.rank",
                &[&universe, &symbol, &rank, &as_of_date],
            )
            .await?;
        Ok(())
    }

    /// Get symbols in a universe.
    pub async fn get_universe_symbols(&self, universe: &str) -> Result<Vec<String>, DbError> {
        let rows = self
            .client
            .query(
                "SELECT DISTINCT symbol FROM universe_membership
                 WHERE universe_name = $1
                 ORDER BY symbol",
                &[&universe],
            )
            .await?;
        Ok(rows.iter().map(|r| r.get(0)).collect())
    }

    /// Insert OHLCV bars with batch upsert (daily).
    pub async fn upsert_ohlcv_batch(
        &self,
        symbol: &str,
        bars: &[HistoricalBar],
    ) -> Result<usize, DbError> {
        if bars.is_empty() {
            return Ok(0);
        }

        let batch_size = 100;
        let mut total_inserted = 0;

        for chunk in bars.chunks(batch_size) {
            let mut values_parts = Vec::new();
            let mut params: Vec<Box<dyn tokio_postgres::types::ToSql + Sync + Send>> = Vec::new();
            let mut param_idx = 1;

            for bar in chunk {
                let trading_date = match bar.trading_date() {
                    Some(d) => d,
                    None => continue,
                };

                let open = f64_to_decimal(bar.open);
                let high = f64_to_decimal(bar.high);
                let low = f64_to_decimal(bar.low);
                let close = f64_to_decimal(bar.close);
                let adj_close = f64_to_decimal(bar.adjusted_close);

                values_parts.push(format!(
                    "(${}, ${}, ${}, ${}, ${}, ${}, ${}, ${})",
                    param_idx,
                    param_idx + 1,
                    param_idx + 2,
                    param_idx + 3,
                    param_idx + 4,
                    param_idx + 5,
                    param_idx + 6,
                    param_idx + 7
                ));

                params.push(Box::new(symbol.to_string()));
                params.push(Box::new(trading_date));
                params.push(Box::new(open));
                params.push(Box::new(high));
                params.push(Box::new(low));
                params.push(Box::new(close));
                params.push(Box::new(adj_close));
                params.push(Box::new(bar.volume_i64()));

                param_idx += 8;
            }

            if values_parts.is_empty() {
                continue;
            }

            let query = format!(
                "INSERT INTO ohlcv_daily (symbol, trading_date, open, high, low, close, adj_close, volume)
                 VALUES {}
                 ON CONFLICT (symbol, trading_date) DO UPDATE SET
                   adj_close = EXCLUDED.adj_close,
                   ingested_at = NOW()",
                values_parts.join(", ")
            );

            let params_refs: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = params
                .iter()
                .map(|p| p.as_ref() as &(dyn tokio_postgres::types::ToSql + Sync))
                .collect();

            let result = self.client.execute(&query, &params_refs).await?;
            total_inserted += result as usize;
        }

        Ok(total_inserted)
    }

    /// Insert OHLCV bars with batch upsert (intraday).
    pub async fn upsert_ohlcv_intraday_batch(
        &self,
        symbol: &str,
        interval: &str,
        bars: &[HistoricalBar],
    ) -> Result<usize, DbError> {
        if bars.is_empty() {
            return Ok(0);
        }

        // Ensure table exists (create if not)
        self.client
            .execute(
                "CREATE TABLE IF NOT EXISTS ohlcv_intraday (
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    interval TEXT NOT NULL,
                    open DECIMAL(12,4),
                    high DECIMAL(12,4),
                    low DECIMAL(12,4),
                    close DECIMAL(12,4),
                    volume BIGINT,
                    ingested_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (symbol, timestamp, interval)
                )",
                &[],
            )
            .await?;

        let batch_size = 100;
        let mut total_inserted = 0;

        for chunk in bars.chunks(batch_size) {
            let mut values_parts = Vec::new();
            let mut params: Vec<Box<dyn tokio_postgres::types::ToSql + Sync + Send>> = Vec::new();
            let mut param_idx = 1;

            for bar in chunk {
                let timestamp = match bar.timestamp_utc() {
                    Some(ts) => ts,
                    None => continue,
                };

                let open = f64_to_decimal(bar.open);
                let high = f64_to_decimal(bar.high);
                let low = f64_to_decimal(bar.low);
                let close = f64_to_decimal(bar.close);

                values_parts.push(format!(
                    "(${}, ${}, ${}, ${}, ${}, ${}, ${}, ${})",
                    param_idx,
                    param_idx + 1,
                    param_idx + 2,
                    param_idx + 3,
                    param_idx + 4,
                    param_idx + 5,
                    param_idx + 6,
                    param_idx + 7
                ));

                params.push(Box::new(symbol.to_string()));
                params.push(Box::new(timestamp));
                params.push(Box::new(interval.to_string()));
                params.push(Box::new(open));
                params.push(Box::new(high));
                params.push(Box::new(low));
                params.push(Box::new(close));
                params.push(Box::new(bar.volume_i64()));

                param_idx += 8;
            }

            if values_parts.is_empty() {
                continue;
            }

            let query = format!(
                "INSERT INTO ohlcv_intraday (symbol, timestamp, interval, open, high, low, close, volume)
                 VALUES {}
                 ON CONFLICT (symbol, timestamp, interval) DO UPDATE SET
                   close = EXCLUDED.close,
                   ingested_at = NOW()",
                values_parts.join(", ")
            );

            let params_refs: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = params
                .iter()
                .map(|p| p.as_ref() as &(dyn tokio_postgres::types::ToSql + Sync))
                .collect();

            let result = self.client.execute(&query, &params_refs).await?;
            total_inserted += result as usize;
        }

        Ok(total_inserted)
    }

    /// Upsert ticker capabilities.
    pub async fn upsert_ticker_capabilities(
        &self,
        caps: &TickerCapabilities,
    ) -> Result<(), DbError> {
        // Ensure table exists
        self.client
            .execute(
                "CREATE TABLE IF NOT EXISTS ticker_capabilities (
                    symbol TEXT PRIMARY KEY,
                    valid_intervals TEXT[],
                    valid_ranges TEXT[],
                    has_intraday BOOLEAN,
                    max_range TEXT,
                    sector TEXT,
                    industry TEXT,
                    probed_at TIMESTAMPTZ,
                    probe_duration_ms INTEGER
                )",
                &[],
            )
            .await?;

        let intervals: Vec<&str> = caps.valid_intervals.iter().map(|s| s.as_str()).collect();
        let ranges: Vec<&str> = caps.valid_ranges.iter().map(|s| s.as_str()).collect();

        self.client
            .execute(
                "INSERT INTO ticker_capabilities 
                 (symbol, valid_intervals, valid_ranges, has_intraday, max_range, sector, industry, probed_at, probe_duration_ms)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                 ON CONFLICT (symbol) DO UPDATE SET
                   valid_intervals = EXCLUDED.valid_intervals,
                   valid_ranges = EXCLUDED.valid_ranges,
                   has_intraday = EXCLUDED.has_intraday,
                   max_range = EXCLUDED.max_range,
                   sector = EXCLUDED.sector,
                   industry = EXCLUDED.industry,
                   probed_at = EXCLUDED.probed_at,
                   probe_duration_ms = EXCLUDED.probe_duration_ms",
                &[
                    &caps.symbol,
                    &intervals,
                    &ranges,
                    &caps.has_intraday,
                    &caps.max_range,
                    &caps.sector,
                    &caps.industry,
                    &caps.probed_at,
                    &(caps.probe_duration_ms as i32),
                ],
            )
            .await?;

        Ok(())
    }

    /// Get last bar date for a symbol.
    pub async fn get_last_bar_date(&self, symbol: &str) -> Result<Option<NaiveDate>, DbError> {
        let row = self
            .client
            .query_opt(
                "SELECT MAX(trading_date) FROM ohlcv_daily WHERE symbol = $1",
                &[&symbol],
            )
            .await?;

        Ok(row.and_then(|r| r.get(0)))
    }

    /// Update ingestion state.
    pub async fn update_ingestion_state(
        &self,
        symbol: &str,
        first_date: Option<NaiveDate>,
        last_date: Option<NaiveDate>,
        total_bars: i32,
        error: Option<&str>,
    ) -> Result<(), DbError> {
        if error.is_some() {
            self.client
                .execute(
                    "INSERT INTO ingestion_state (symbol, last_attempt_at, consecutive_failures, last_error)
                     VALUES ($1, NOW(), 1, $2)
                     ON CONFLICT (symbol) DO UPDATE SET
                       last_attempt_at = NOW(),
                       consecutive_failures = ingestion_state.consecutive_failures + 1,
                       last_error = EXCLUDED.last_error,
                       updated_at = NOW()",
                    &[&symbol, &error],
                )
                .await?;
        } else {
            self.client
                .execute(
                    "INSERT INTO ingestion_state
                     (symbol, first_bar_date, last_bar_date, total_bars, last_success_at, last_attempt_at, consecutive_failures)
                     VALUES ($1, $2, $3, $4, NOW(), NOW(), 0)
                     ON CONFLICT (symbol) DO UPDATE SET
                       first_bar_date = COALESCE(EXCLUDED.first_bar_date, ingestion_state.first_bar_date),
                       last_bar_date = EXCLUDED.last_bar_date,
                       total_bars = EXCLUDED.total_bars,
                       last_success_at = NOW(),
                       last_attempt_at = NOW(),
                       consecutive_failures = 0,
                       last_error = NULL,
                       updated_at = NOW()",
                    &[&symbol, &first_date, &last_date, &total_bars],
                )
                .await?;
        }
        Ok(())
    }

    /// Get freshness data for report.
    pub async fn get_freshness_data(
        &self,
    ) -> Result<Vec<(String, String, Option<NaiveDate>, i64)>, DbError> {
        // Get daily data freshness
        let rows = self.client
            .query(
                "SELECT symbol, '1d' as interval, MAX(trading_date) as last_date, COUNT(*) as bar_count
                 FROM ohlcv_daily
                 GROUP BY symbol
                 ORDER BY symbol",
                &[],
            )
            .await?;

        let mut results: Vec<(String, String, Option<NaiveDate>, i64)> = rows
            .iter()
            .map(|r| (r.get(0), r.get(1), r.get(2), r.get(3)))
            .collect();

        // Also get intraday if table exists
        let intraday_exists: bool = self.client
            .query_one(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'ohlcv_intraday')",
                &[],
            )
            .await?
            .get(0);

        if intraday_exists {
            let intraday_rows = self.client
                .query(
                    "SELECT symbol, interval, MAX(timestamp::date) as last_date, COUNT(*) as bar_count
                     FROM ohlcv_intraday
                     GROUP BY symbol, interval
                     ORDER BY symbol, interval",
                    &[],
                )
                .await?;

            for row in intraday_rows {
                results.push((row.get(0), row.get(1), row.get(2), row.get(3)));
            }
        }

        Ok(results)
    }

    /// Get ingestion stats.
    pub async fn get_ingestion_stats(&self) -> Result<IngestionStats, DbError> {
        let row = self.client
            .query_one(
                "SELECT
                   (SELECT COUNT(*) FROM instruments WHERE active = true) as total_instruments,
                   (SELECT COUNT(*) FROM ohlcv_daily) as total_bars,
                   (SELECT COUNT(DISTINCT symbol) FROM ohlcv_daily) as symbols_with_data,
                   (SELECT MIN(trading_date) FROM ohlcv_daily) as earliest_date,
                   (SELECT MAX(trading_date) FROM ohlcv_daily) as latest_date,
                   (SELECT COUNT(*) FROM ingestion_state WHERE consecutive_failures > 0) as failed_symbols",
                &[],
            )
            .await?;

        Ok(IngestionStats {
            total_instruments: row.get(0),
            total_bars: row.get(1),
            symbols_with_data: row.get(2),
            earliest_date: row.get(3),
            latest_date: row.get(4),
            failed_symbols: row.get(5),
        })
    }

    /// Verify data integrity.
    pub async fn verify_integrity(&self) -> Result<Vec<IntegrityIssue>, DbError> {
        let mut issues = Vec::new();

        // Check for gaps
        let gap_rows = self.client
            .query(
                "SELECT symbol, COUNT(*) as gap_count
                 FROM (
                   SELECT symbol, trading_date,
                          LAG(trading_date) OVER (PARTITION BY symbol ORDER BY trading_date) as prev_date
                   FROM ohlcv_daily
                 ) sub
                 WHERE prev_date IS NOT NULL
                   AND trading_date - prev_date > 5
                 GROUP BY symbol
                 HAVING COUNT(*) > 0",
                &[],
            )
            .await?;

        for row in gap_rows {
            issues.push(IntegrityIssue {
                symbol: row.get(0),
                issue_type: "gap".into(),
                count: row.get(1),
            });
        }

        // Check for invalid OHLC
        let invalid_rows = self
            .client
            .query(
                "SELECT symbol, COUNT(*) as invalid_count
                 FROM ohlcv_daily
                 WHERE high < low OR high < open OR high < close
                    OR low > open OR low > close
                    OR open <= 0 OR close <= 0
                 GROUP BY symbol",
                &[],
            )
            .await?;

        for row in invalid_rows {
            issues.push(IntegrityIssue {
                symbol: row.get(0),
                issue_type: "invalid_ohlc".into(),
                count: row.get(1),
            });
        }

        Ok(issues)
    }

    /// Get API budget status.
    pub async fn get_budget_status(&self) -> Result<(i32, i32), DbError> {
        let month_key = Utc::now().format("%Y-%m").to_string();
        let row = self
            .client
            .query_opt(
                "SELECT requests_used, requests_limit FROM api_budget WHERE month_key = $1",
                &[&month_key],
            )
            .await?;

        match row {
            Some(r) => Ok((r.get(0), r.get(1))),
            None => Ok((0, 500000)),
        }
    }

    // ========================================================================
    // Fundamentals Methods
    // ========================================================================

    /// Upsert fundamental snapshot.
    pub async fn upsert_fundamental_snapshot(
        &self,
        snap: &FundamentalSnapshot,
    ) -> Result<(), DbError> {
        let market_cap: Option<i64> = snap.market_cap.map(|v| v as i64);
        let enterprise_value: Option<i64> = snap.enterprise_value.map(|v| v as i64);
        let free_cash_flow: Option<i64> = snap.free_cash_flow.map(|v| v as i64);
        let operating_cash_flow: Option<i64> = snap.operating_cash_flow.map(|v| v as i64);

        let pe = f64_to_decimal(snap.price_earnings);
        let pb = f64_to_decimal(snap.price_to_book);
        let eps = f64_to_decimal(snap.earnings_per_share);
        let roe = f64_to_decimal(snap.return_on_equity);
        let roa = f64_to_decimal(snap.return_on_assets);
        let de = f64_to_decimal(snap.debt_to_equity);
        let pm = f64_to_decimal(snap.profit_margins);
        let gm = f64_to_decimal(snap.gross_margins);
        let om = f64_to_decimal(snap.operating_margins);
        let cr = f64_to_decimal(snap.current_ratio);
        let qr = f64_to_decimal(snap.quick_ratio);
        let dy = f64_to_decimal(snap.dividend_yield);
        let ldv = f64_to_decimal(snap.last_dividend_value);
        let eg = f64_to_decimal(snap.earnings_growth);
        let rg = f64_to_decimal(snap.revenue_growth);

        self.client
            .execute(
                "INSERT INTO fundamentals_snapshot (
                    symbol, snapshot_date, price_earnings, price_to_book, earnings_per_share,
                    return_on_equity, return_on_assets, debt_to_equity, profit_margins,
                    gross_margins, operating_margins, current_ratio, quick_ratio,
                    market_cap, enterprise_value, dividend_yield, last_dividend_value,
                    last_dividend_date, earnings_growth, revenue_growth,
                    free_cash_flow, operating_cash_flow, fetched_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                    $14, $15, $16, $17, $18, $19, $20, $21, $22, NOW()
                ) ON CONFLICT (symbol, snapshot_date) DO UPDATE SET
                    price_earnings = EXCLUDED.price_earnings,
                    price_to_book = EXCLUDED.price_to_book,
                    earnings_per_share = EXCLUDED.earnings_per_share,
                    return_on_equity = EXCLUDED.return_on_equity,
                    return_on_assets = EXCLUDED.return_on_assets,
                    debt_to_equity = EXCLUDED.debt_to_equity,
                    profit_margins = EXCLUDED.profit_margins,
                    gross_margins = EXCLUDED.gross_margins,
                    operating_margins = EXCLUDED.operating_margins,
                    current_ratio = EXCLUDED.current_ratio,
                    quick_ratio = EXCLUDED.quick_ratio,
                    market_cap = EXCLUDED.market_cap,
                    enterprise_value = EXCLUDED.enterprise_value,
                    dividend_yield = EXCLUDED.dividend_yield,
                    last_dividend_value = EXCLUDED.last_dividend_value,
                    last_dividend_date = EXCLUDED.last_dividend_date,
                    earnings_growth = EXCLUDED.earnings_growth,
                    revenue_growth = EXCLUDED.revenue_growth,
                    free_cash_flow = EXCLUDED.free_cash_flow,
                    operating_cash_flow = EXCLUDED.operating_cash_flow,
                    fetched_at = NOW()",
                &[
                    &snap.symbol,
                    &snap.snapshot_date,
                    &pe,
                    &pb,
                    &eps,
                    &roe,
                    &roa,
                    &de,
                    &pm,
                    &gm,
                    &om,
                    &cr,
                    &qr,
                    &market_cap,
                    &enterprise_value,
                    &dy,
                    &ldv,
                    &snap.last_dividend_date,
                    &eg,
                    &rg,
                    &free_cash_flow,
                    &operating_cash_flow,
                ],
            )
            .await?;
        Ok(())
    }

    /// Batch upsert fundamental snapshots.
    pub async fn upsert_fundamental_snapshots(
        &self,
        snapshots: &[FundamentalSnapshot],
    ) -> Result<usize, DbError> {
        let mut count = 0;
        for snap in snapshots {
            self.upsert_fundamental_snapshot(snap).await?;
            count += 1;
        }
        Ok(count)
    }

    /// Upsert dividend entry.
    pub async fn upsert_dividend(&self, div: &DividendEntry) -> Result<(), DbError> {
        let rate = f64_to_decimal(Some(div.rate));

        self.client
            .execute(
                "INSERT INTO dividends_history (symbol, payment_date, ex_date, rate, dividend_type, related_to)
                 VALUES ($1, $2, $3, $4, $5, $6)
                 ON CONFLICT (symbol, payment_date, dividend_type, rate) DO NOTHING",
                &[
                    &div.symbol,
                    &div.payment_date,
                    &div.ex_date,
                    &rate,
                    &div.dividend_type,
                    &div.related_to,
                ],
            )
            .await?;
        Ok(())
    }

    /// Batch upsert dividends.
    pub async fn upsert_dividends(&self, dividends: &[DividendEntry]) -> Result<usize, DbError> {
        let mut count = 0;
        for div in dividends {
            self.upsert_dividend(div).await?;
            count += 1;
        }
        Ok(count)
    }

    /// Upsert company profile.
    pub async fn upsert_company_profile(
        &self,
        symbol: &str,
        profile: &SummaryProfile,
    ) -> Result<(), DbError> {
        self.client
            .execute(
                "INSERT INTO company_profile (
                    symbol, long_name, sector, sector_key, industry, industry_key,
                    website, city, state, country, full_time_employees, business_summary, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW())
                ON CONFLICT (symbol) DO UPDATE SET
                    long_name = COALESCE(EXCLUDED.long_name, company_profile.long_name),
                    sector = COALESCE(EXCLUDED.sector, company_profile.sector),
                    sector_key = COALESCE(EXCLUDED.sector_key, company_profile.sector_key),
                    industry = COALESCE(EXCLUDED.industry, company_profile.industry),
                    industry_key = COALESCE(EXCLUDED.industry_key, company_profile.industry_key),
                    website = COALESCE(EXCLUDED.website, company_profile.website),
                    city = COALESCE(EXCLUDED.city, company_profile.city),
                    state = COALESCE(EXCLUDED.state, company_profile.state),
                    country = COALESCE(EXCLUDED.country, company_profile.country),
                    full_time_employees = COALESCE(EXCLUDED.full_time_employees, company_profile.full_time_employees),
                    business_summary = COALESCE(EXCLUDED.business_summary, company_profile.business_summary),
                    updated_at = NOW()",
                &[
                    &symbol,
                    &profile.long_business_summary.as_ref().map(|_| profile.symbol.as_ref().map(|s| s.as_str())).flatten(),
                    &profile.sector,
                    &profile.sector_key,
                    &profile.industry,
                    &profile.industry_key,
                    &profile.website,
                    &profile.city,
                    &profile.state,
                    &profile.country,
                    &profile.full_time_employees,
                    &profile.long_business_summary,
                ],
            )
            .await?;
        Ok(())
    }

    /// Update sync watermark for fundamentals.
    pub async fn update_sync_watermark(
        &self,
        entity: &str,
        symbols_synced: i32,
        errors: i32,
    ) -> Result<(), DbError> {
        self.client
            .execute(
                "INSERT INTO sync_watermarks (entity, last_sync, symbols_synced, errors, next_sync)
                 VALUES ($1, NOW(), $2, $3, NOW() + INTERVAL '24 hours')
                 ON CONFLICT (entity) DO UPDATE SET
                    last_sync = NOW(),
                    symbols_synced = EXCLUDED.symbols_synced,
                    errors = EXCLUDED.errors,
                    next_sync = NOW() + INTERVAL '24 hours'",
                &[&entity, &symbols_synced, &errors],
            )
            .await?;
        Ok(())
    }

    /// Get latest fundamental snapshot for a symbol.
    pub async fn get_latest_fundamental(
        &self,
        symbol: &str,
    ) -> Result<Option<FundamentalsRow>, DbError> {
        let row = self.client
            .query_opt(
                "SELECT symbol, snapshot_date, price_earnings, price_to_book, earnings_per_share,
                        return_on_equity, return_on_assets, debt_to_equity, profit_margins,
                        gross_margins, operating_margins, current_ratio, quick_ratio,
                        market_cap, enterprise_value, dividend_yield, earnings_growth, revenue_growth
                 FROM fundamentals_snapshot
                 WHERE symbol = $1
                 ORDER BY snapshot_date DESC
                 LIMIT 1",
                &[&symbol],
            )
            .await?;

        Ok(row.map(|r| FundamentalsRow {
            symbol: r.get(0),
            snapshot_date: r.get(1),
            price_earnings: r
                .get::<_, Option<Decimal>>(2)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            price_to_book: r
                .get::<_, Option<Decimal>>(3)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            earnings_per_share: r
                .get::<_, Option<Decimal>>(4)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            return_on_equity: r
                .get::<_, Option<Decimal>>(5)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            return_on_assets: r
                .get::<_, Option<Decimal>>(6)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            debt_to_equity: r
                .get::<_, Option<Decimal>>(7)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            profit_margins: r
                .get::<_, Option<Decimal>>(8)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            gross_margins: r
                .get::<_, Option<Decimal>>(9)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            operating_margins: r
                .get::<_, Option<Decimal>>(10)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            current_ratio: r
                .get::<_, Option<Decimal>>(11)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            quick_ratio: r
                .get::<_, Option<Decimal>>(12)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            market_cap: r.get(13),
            enterprise_value: r.get(14),
            dividend_yield: r
                .get::<_, Option<Decimal>>(15)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            earnings_growth: r
                .get::<_, Option<Decimal>>(16)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            revenue_growth: r
                .get::<_, Option<Decimal>>(17)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
        }))
    }

    /// Get all latest fundamentals for active symbols.
    pub async fn get_all_latest_fundamentals(&self) -> Result<Vec<FundamentalsRow>, DbError> {
        let rows = self.client
            .query(
                "SELECT DISTINCT ON (f.symbol)
                        f.symbol, f.snapshot_date, f.price_earnings, f.price_to_book, f.earnings_per_share,
                        f.return_on_equity, f.return_on_assets, f.debt_to_equity, f.profit_margins,
                        f.gross_margins, f.operating_margins, f.current_ratio, f.quick_ratio,
                        f.market_cap, f.enterprise_value, f.dividend_yield, f.earnings_growth, f.revenue_growth
                 FROM fundamentals_snapshot f
                 INNER JOIN instruments i ON f.symbol = i.symbol AND i.active = true
                 ORDER BY f.symbol, f.snapshot_date DESC",
                &[],
            )
            .await?;

        Ok(rows
            .iter()
            .map(|r| FundamentalsRow {
                symbol: r.get(0),
                snapshot_date: r.get(1),
                price_earnings: r
                    .get::<_, Option<Decimal>>(2)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                price_to_book: r
                    .get::<_, Option<Decimal>>(3)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                earnings_per_share: r
                    .get::<_, Option<Decimal>>(4)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                return_on_equity: r
                    .get::<_, Option<Decimal>>(5)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                return_on_assets: r
                    .get::<_, Option<Decimal>>(6)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                debt_to_equity: r
                    .get::<_, Option<Decimal>>(7)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                profit_margins: r
                    .get::<_, Option<Decimal>>(8)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                gross_margins: r
                    .get::<_, Option<Decimal>>(9)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                operating_margins: r
                    .get::<_, Option<Decimal>>(10)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                current_ratio: r
                    .get::<_, Option<Decimal>>(11)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                quick_ratio: r
                    .get::<_, Option<Decimal>>(12)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                market_cap: r.get(13),
                enterprise_value: r.get(14),
                dividend_yield: r
                    .get::<_, Option<Decimal>>(15)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                earnings_growth: r
                    .get::<_, Option<Decimal>>(16)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                revenue_growth: r
                    .get::<_, Option<Decimal>>(17)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
            })
            .collect())
    }

    /// Get fundamental snapshot for a symbol at a specific point-in-time.
    /// Returns the most recent snapshot with snapshot_date <= as_of_date.
    /// This prevents look-ahead bias in backtesting.
    pub async fn get_fundamental_at(
        &self,
        symbol: &str,
        as_of_date: NaiveDate,
    ) -> Result<Option<FundamentalsRow>, DbError> {
        let row = self.client
            .query_opt(
                "SELECT symbol, snapshot_date, price_earnings, price_to_book, earnings_per_share,
                        return_on_equity, return_on_assets, debt_to_equity, profit_margins,
                        gross_margins, operating_margins, current_ratio, quick_ratio,
                        market_cap, enterprise_value, dividend_yield, earnings_growth, revenue_growth
                 FROM fundamentals_snapshot
                 WHERE symbol = $1 AND snapshot_date <= $2
                 ORDER BY snapshot_date DESC
                 LIMIT 1",
                &[&symbol, &as_of_date],
            )
            .await?;

        Ok(row.map(|r| FundamentalsRow {
            symbol: r.get(0),
            snapshot_date: r.get(1),
            price_earnings: r
                .get::<_, Option<Decimal>>(2)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            price_to_book: r
                .get::<_, Option<Decimal>>(3)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            earnings_per_share: r
                .get::<_, Option<Decimal>>(4)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            return_on_equity: r
                .get::<_, Option<Decimal>>(5)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            return_on_assets: r
                .get::<_, Option<Decimal>>(6)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            debt_to_equity: r
                .get::<_, Option<Decimal>>(7)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            profit_margins: r
                .get::<_, Option<Decimal>>(8)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            gross_margins: r
                .get::<_, Option<Decimal>>(9)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            operating_margins: r
                .get::<_, Option<Decimal>>(10)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            current_ratio: r
                .get::<_, Option<Decimal>>(11)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            quick_ratio: r
                .get::<_, Option<Decimal>>(12)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            market_cap: r.get(13),
            enterprise_value: r.get(14),
            dividend_yield: r
                .get::<_, Option<Decimal>>(15)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            earnings_growth: r
                .get::<_, Option<Decimal>>(16)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
            revenue_growth: r
                .get::<_, Option<Decimal>>(17)
                .map(|d| d.to_string().parse().unwrap_or(0.0)),
        }))
    }

    /// Get fundamentals for all active symbols at a specific point-in-time.
    /// Returns the most recent snapshot with snapshot_date <= as_of_date for each symbol.
    pub async fn get_all_fundamentals_at(
        &self,
        as_of_date: NaiveDate,
    ) -> Result<Vec<FundamentalsRow>, DbError> {
        let rows = self.client
            .query(
                "SELECT DISTINCT ON (f.symbol)
                        f.symbol, f.snapshot_date, f.price_earnings, f.price_to_book, f.earnings_per_share,
                        f.return_on_equity, f.return_on_assets, f.debt_to_equity, f.profit_margins,
                        f.gross_margins, f.operating_margins, f.current_ratio, f.quick_ratio,
                        f.market_cap, f.enterprise_value, f.dividend_yield, f.earnings_growth, f.revenue_growth
                 FROM fundamentals_snapshot f
                 INNER JOIN instruments i ON f.symbol = i.symbol AND i.active = true
                 WHERE f.snapshot_date <= $1
                 ORDER BY f.symbol, f.snapshot_date DESC",
                &[&as_of_date],
            )
            .await?;

        Ok(rows
            .iter()
            .map(|r| FundamentalsRow {
                symbol: r.get(0),
                snapshot_date: r.get(1),
                price_earnings: r
                    .get::<_, Option<Decimal>>(2)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                price_to_book: r
                    .get::<_, Option<Decimal>>(3)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                earnings_per_share: r
                    .get::<_, Option<Decimal>>(4)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                return_on_equity: r
                    .get::<_, Option<Decimal>>(5)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                return_on_assets: r
                    .get::<_, Option<Decimal>>(6)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                debt_to_equity: r
                    .get::<_, Option<Decimal>>(7)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                profit_margins: r
                    .get::<_, Option<Decimal>>(8)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                gross_margins: r
                    .get::<_, Option<Decimal>>(9)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                operating_margins: r
                    .get::<_, Option<Decimal>>(10)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                current_ratio: r
                    .get::<_, Option<Decimal>>(11)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                quick_ratio: r
                    .get::<_, Option<Decimal>>(12)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                market_cap: r.get(13),
                enterprise_value: r.get(14),
                dividend_yield: r
                    .get::<_, Option<Decimal>>(15)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                earnings_growth: r
                    .get::<_, Option<Decimal>>(16)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
                revenue_growth: r
                    .get::<_, Option<Decimal>>(17)
                    .map(|d| d.to_string().parse().unwrap_or(0.0)),
            })
            .collect())
    }

    // ========================================================================
    // Interest Rates Methods
    // ========================================================================

    /// Upsert a single interest rate entry (idempotent by date/region/rate_type).
    pub async fn upsert_interest_rate(&self, entry: &InterestRateEntry) -> Result<(), DbError> {
        let rate_decimal = Decimal::from_f64(entry.rate)
            .ok_or_else(|| DbError::Config(format!("Invalid rate value: {}", entry.rate)))?;

        self.client
            .execute(
                "INSERT INTO interest_rates (rate_date, region, rate_type, rate, source)
                 VALUES ($1, $2, $3, $4, $5)
                 ON CONFLICT (rate_date, region, rate_type) DO UPDATE SET
                   rate = EXCLUDED.rate,
                   source = EXCLUDED.source,
                   created_at = NOW()",
                &[
                    &entry.rate_date,
                    &entry.region,
                    &entry.rate_type,
                    &rate_decimal,
                    &entry.source,
                ],
            )
            .await?;
        Ok(())
    }

    /// Batch upsert interest rates. Returns (inserted, updated) counts.
    pub async fn upsert_interest_rates(
        &self,
        entries: &[InterestRateEntry],
    ) -> Result<(usize, usize), DbError> {
        if entries.is_empty() {
            return Ok((0, 0));
        }

        // Count existing before insert
        let existing_count: i64 = self
            .client
            .query_one("SELECT COUNT(*) FROM interest_rates", &[])
            .await?
            .get(0);

        let mut success = 0;
        for entry in entries {
            if self.upsert_interest_rate(entry).await.is_ok() {
                success += 1;
            }
        }

        let new_count: i64 = self
            .client
            .query_one("SELECT COUNT(*) FROM interest_rates", &[])
            .await?
            .get(0);

        let inserted = (new_count - existing_count) as usize;
        let updated = success - inserted;
        Ok((inserted, updated))
    }

    /// Get interest rate at a specific date using point-in-time semantics.
    /// Returns the latest rate where rate_date <= requested date.
    pub async fn get_interest_rate_at(
        &self,
        date: NaiveDate,
        region: &str,
        rate_type: &str,
    ) -> Result<Option<f64>, DbError> {
        let row = self
            .client
            .query_opt(
                "SELECT rate FROM interest_rates 
                 WHERE rate_date <= $1 AND region = $2 AND rate_type = $3
                 ORDER BY rate_date DESC
                 LIMIT 1",
                &[&date, &region, &rate_type],
            )
            .await?;

        Ok(row.map(|r| {
            let rate: Decimal = r.get(0);
            rate.to_string().parse().unwrap_or(0.0)
        }))
    }

    /// Get all interest rates for a region/type within a date range.
    pub async fn get_interest_rates_range(
        &self,
        start: NaiveDate,
        end: NaiveDate,
        region: &str,
        rate_type: &str,
    ) -> Result<Vec<(NaiveDate, f64)>, DbError> {
        let rows = self
            .client
            .query(
                "SELECT rate_date, rate FROM interest_rates 
                 WHERE rate_date BETWEEN $1 AND $2 AND region = $3 AND rate_type = $4
                 ORDER BY rate_date",
                &[&start, &end, &region, &rate_type],
            )
            .await?;

        Ok(rows
            .iter()
            .map(|r| {
                let date: NaiveDate = r.get(0);
                let rate: Decimal = r.get(1);
                (date, rate.to_string().parse().unwrap_or(0.0))
            })
            .collect())
    }

    /// Get interest rate statistics for status reporting.
    pub async fn get_interest_rate_stats(&self) -> Result<InterestRateStats, DbError> {
        let row = self
            .client
            .query_one(
                "SELECT 
                    COUNT(*) FILTER (WHERE region = 'BR') as br_count,
                    MIN(rate_date) FILTER (WHERE region = 'BR') as br_min,
                    MAX(rate_date) FILTER (WHERE region = 'BR') as br_max,
                    COUNT(*) FILTER (WHERE region = 'US') as us_count,
                    MIN(rate_date) FILTER (WHERE region = 'US') as us_min,
                    MAX(rate_date) FILTER (WHERE region = 'US') as us_max
                 FROM interest_rates",
                &[],
            )
            .await?;

        Ok(InterestRateStats {
            br_count: row.get(0),
            br_min_date: row.get(1),
            br_max_date: row.get(2),
            us_count: row.get(3),
            us_min_date: row.get(4),
            us_max_date: row.get(5),
        })
    }

    // ========================================================================
    // Watermark Methods
    // ========================================================================

    /// Get all watermarks for planning.
    pub async fn get_all_watermarks(&self) -> Result<Vec<Watermark>, DbError> {
        let rows = self
            .client
            .query(
                "SELECT symbol, interval, first_ts, last_ts, bar_count, 
                        last_success_at, consecutive_failures, last_error
                 FROM ingestion_watermarks
                 ORDER BY symbol, interval",
                &[],
            )
            .await?;

        Ok(rows
            .iter()
            .map(|r| Watermark {
                symbol: r.get(0),
                interval: r.get(1),
                first_ts: r.get(2),
                last_ts: r.get(3),
                bar_count: r.get(4),
                last_success_at: r.get(5),
                consecutive_failures: r.get(6),
                last_error: r.get(7),
            })
            .collect())
    }

    /// Get watermark for specific (symbol, interval).
    pub async fn get_watermark(
        &self,
        symbol: &str,
        interval: &str,
    ) -> Result<Option<Watermark>, DbError> {
        let row = self
            .client
            .query_opt(
                "SELECT symbol, interval, first_ts, last_ts, bar_count,
                        last_success_at, consecutive_failures, last_error
                 FROM ingestion_watermarks
                 WHERE symbol = $1 AND interval = $2",
                &[&symbol, &interval],
            )
            .await?;

        Ok(row.map(|r| Watermark {
            symbol: r.get(0),
            interval: r.get(1),
            first_ts: r.get(2),
            last_ts: r.get(3),
            bar_count: r.get(4),
            last_success_at: r.get(5),
            consecutive_failures: r.get(6),
            last_error: r.get(7),
        }))
    }

    /// Upsert watermark after successful ingestion.
    pub async fn upsert_watermark(
        &self,
        symbol: &str,
        interval: &str,
        first_ts: Option<chrono::DateTime<Utc>>,
        last_ts: Option<chrono::DateTime<Utc>>,
        bar_count: i32,
    ) -> Result<(), DbError> {
        self.client
            .execute(
                "INSERT INTO ingestion_watermarks 
                 (symbol, interval, first_ts, last_ts, bar_count, last_success_at, last_attempt_at, consecutive_failures)
                 VALUES ($1, $2, $3, $4, $5, NOW(), NOW(), 0)
                 ON CONFLICT (symbol, interval) DO UPDATE SET
                   first_ts = COALESCE(EXCLUDED.first_ts, ingestion_watermarks.first_ts),
                   last_ts = GREATEST(EXCLUDED.last_ts, ingestion_watermarks.last_ts),
                   bar_count = ingestion_watermarks.bar_count + EXCLUDED.bar_count,
                   last_success_at = NOW(),
                   last_attempt_at = NOW(),
                   consecutive_failures = 0,
                   last_error = NULL",
                &[&symbol, &interval, &first_ts, &last_ts, &bar_count],
            )
            .await?;
        Ok(())
    }

    /// Record watermark failure.
    pub async fn record_watermark_failure(
        &self,
        symbol: &str,
        interval: &str,
        error: &str,
    ) -> Result<(), DbError> {
        self.client
            .execute(
                "INSERT INTO ingestion_watermarks 
                 (symbol, interval, last_attempt_at, consecutive_failures, last_error)
                 VALUES ($1, $2, NOW(), 1, $3)
                 ON CONFLICT (symbol, interval) DO UPDATE SET
                   last_attempt_at = NOW(),
                   consecutive_failures = ingestion_watermarks.consecutive_failures + 1,
                   last_error = EXCLUDED.last_error",
                &[&symbol, &interval, &error],
            )
            .await?;
        Ok(())
    }

    /// Get ticker capabilities from database.
    pub async fn get_ticker_capabilities(&self) -> Result<Vec<TickerCapability>, DbError> {
        let rows = self
            .client
            .query(
                "SELECT symbol, valid_intervals, valid_ranges, has_intraday, max_range
                 FROM ticker_capabilities
                 WHERE valid_intervals IS NOT NULL
                 ORDER BY symbol",
                &[],
            )
            .await?;

        Ok(rows
            .iter()
            .map(|r| {
                let intervals: Vec<String> = r.get::<_, Option<Vec<String>>>(1).unwrap_or_default();
                let ranges: Vec<String> = r.get::<_, Option<Vec<String>>>(2).unwrap_or_default();
                TickerCapability {
                    symbol: r.get(0),
                    valid_intervals: intervals,
                    valid_ranges: ranges,
                    has_intraday: r.get::<_, Option<bool>>(3).unwrap_or(false),
                    max_range: r.get(4),
                }
            })
            .collect())
    }

    /// Get coverage stats for reporting.
    pub async fn get_coverage_stats(&self) -> Result<CoverageStats, DbError> {
        let total_tickers: i64 = self
            .client
            .query_one(
                "SELECT COUNT(DISTINCT symbol) FROM ticker_capabilities",
                &[],
            )
            .await?
            .get(0);

        let daily_coverage: i64 = self
            .client
            .query_one(
                "SELECT COUNT(DISTINCT symbol) FROM ingestion_watermarks WHERE interval = '1d'",
                &[],
            )
            .await?
            .get(0);

        let intraday_by_interval = self
            .client
            .query(
                "SELECT interval, COUNT(DISTINCT symbol) as count
                 FROM ingestion_watermarks
                 WHERE interval != '1d'
                 GROUP BY interval
                 ORDER BY interval",
                &[],
            )
            .await?;

        let mut intraday_coverage: std::collections::HashMap<String, i64> =
            std::collections::HashMap::new();
        for row in intraday_by_interval {
            intraday_coverage.insert(row.get(0), row.get(1));
        }

        let failed_count: i64 = self
            .client
            .query_one(
                "SELECT COUNT(*) FROM ingestion_watermarks WHERE consecutive_failures > 0",
                &[],
            )
            .await?
            .get(0);

        Ok(CoverageStats {
            total_tickers,
            daily_coverage,
            intraday_coverage,
            failed_count,
        })
    }

    // ========================================================================
    // Provider Universe Methods
    // ========================================================================

    /// Check if a ticker is ACTIVE in the provider universe.
    pub async fn is_ticker_active(&self, ticker: &str) -> Result<bool, DbError> {
        let row = self
            .client
            .query_opt(
                "SELECT status FROM provider_universe WHERE ticker = $1",
                &[&ticker],
            )
            .await?;

        match row {
            Some(r) => {
                let status: String = r.get(0);
                Ok(status == "ACTIVE")
            }
            None => Ok(false),
        }
    }

    /// Get ticker status from provider universe.
    pub async fn get_ticker_status(&self, ticker: &str) -> Result<Option<TickerStatus>, DbError> {
        let row = self
            .client
            .query_opt(
                "SELECT ticker, status, asset_type, name, last_seen_at, last_error_code
                 FROM provider_universe WHERE ticker = $1",
                &[&ticker],
            )
            .await?;

        Ok(row.map(|r| TickerStatus {
            ticker: r.get(0),
            status: r.get(1),
            asset_type: r.get(2),
            name: r.get(3),
            last_seen_at: r.get(4),
            last_error_code: r.get(5),
        }))
    }

    /// Get all ACTIVE tickers from provider universe.
    pub async fn get_active_tickers(&self) -> Result<Vec<String>, DbError> {
        let rows = self
            .client
            .query(
                "SELECT ticker FROM provider_universe WHERE status = 'ACTIVE' ORDER BY ticker",
                &[],
            )
            .await?;

        Ok(rows.iter().map(|r| r.get(0)).collect())
    }

    /// Get provider universe stats.
    pub async fn get_universe_stats(&self) -> Result<UniverseStats, DbError> {
        let row = self
            .client
            .query_one(
                "SELECT 
                    COUNT(*) FILTER (WHERE status = 'ACTIVE') as active,
                    COUNT(*) FILTER (WHERE status = 'INACTIVE') as inactive,
                    COUNT(*) FILTER (WHERE status = 'SUSPECT') as suspect,
                    COUNT(*) as total
                 FROM provider_universe",
                &[],
            )
            .await?;

        Ok(UniverseStats {
            active: row.get(0),
            inactive: row.get(1),
            suspect: row.get(2),
            total: row.get(3),
        })
    }

    /// Upsert a ticker in provider universe (from /api/quote/list).
    pub async fn upsert_universe_ticker(
        &self,
        ticker: &str,
        asset_type: Option<&str>,
        name: Option<&str>,
        sector: Option<&str>,
        snapshot_id: &str,
    ) -> Result<bool, DbError> {
        let result = self.client
            .execute(
                "INSERT INTO provider_universe (ticker, asset_type, name, sector, status, source_snapshot_id)
                 VALUES ($1, $2, $3, $4, 'ACTIVE', $5)
                 ON CONFLICT (ticker) DO UPDATE SET
                   asset_type = COALESCE(EXCLUDED.asset_type, provider_universe.asset_type),
                   name = COALESCE(EXCLUDED.name, provider_universe.name),
                   sector = COALESCE(EXCLUDED.sector, provider_universe.sector),
                   status = 'ACTIVE',
                   last_seen_at = NOW(),
                   source_snapshot_id = EXCLUDED.source_snapshot_id",
                &[&ticker, &asset_type, &name, &sector, &snapshot_id],
            )
            .await?;

        Ok(result > 0)
    }

    /// Batch upsert universe tickers (fast).
    pub async fn batch_upsert_universe(
        &self,
        stocks: &[crate::brapi::StockInfo],
        snapshot_id: &str,
    ) -> Result<(i32, i32), DbError> {
        // Deduplicate by ticker
        let mut seen = std::collections::HashSet::new();
        let unique_stocks: Vec<_> = stocks
            .iter()
            .filter(|s| seen.insert(s.stock.clone()))
            .collect();

        let existing: i64 = self
            .client
            .query_one("SELECT COUNT(*) FROM provider_universe", &[])
            .await?
            .get(0);

        // Use chunked inserts to avoid parameter limits
        for chunk in unique_stocks.chunks(500) {
            let tickers: Vec<&str> = chunk.iter().map(|s| s.stock.as_str()).collect();
            let types: Vec<Option<&str>> = chunk.iter().map(|s| s.asset_type.as_deref()).collect();
            let names: Vec<Option<&str>> = chunk.iter().map(|s| s.name.as_deref()).collect();
            let sectors: Vec<Option<&str>> = chunk.iter().map(|s| s.sector.as_deref()).collect();

            self.client
                .execute(
                    "INSERT INTO provider_universe (ticker, asset_type, name, sector, status, source_snapshot_id)
                     SELECT t, ty, n, s, 'ACTIVE', $5
                     FROM UNNEST($1::text[], $2::text[], $3::text[], $4::text[]) AS u(t, ty, n, s)
                     ON CONFLICT (ticker) DO UPDATE SET
                       asset_type = COALESCE(EXCLUDED.asset_type, provider_universe.asset_type),
                       name = COALESCE(EXCLUDED.name, provider_universe.name),
                       sector = COALESCE(EXCLUDED.sector, provider_universe.sector),
                       status = 'ACTIVE',
                       last_seen_at = NOW(),
                       source_snapshot_id = EXCLUDED.source_snapshot_id",
                    &[&tickers, &types, &names, &sectors, &snapshot_id],
                )
                .await?;
        }

        let new_count: i64 = self
            .client
            .query_one("SELECT COUNT(*) FROM provider_universe", &[])
            .await?
            .get(0);

        let new_added = (new_count - existing) as i32;
        let updated = (unique_stocks.len() as i32) - new_added;
        Ok((new_added, updated))
    }

    /// Mark tickers as INACTIVE if not in the provided list.
    pub async fn mark_inactive_missing(
        &self,
        active_tickers: &[String],
        snapshot_id: &str,
    ) -> Result<i64, DbError> {
        if active_tickers.is_empty() {
            return Ok(0);
        }

        let result = self
            .client
            .execute(
                "UPDATE provider_universe 
                 SET status = 'INACTIVE', 
                     last_error_code = 'REMOVED_FROM_LIST',
                     last_error_message = $2
                 WHERE status = 'ACTIVE' 
                   AND ticker NOT IN (SELECT unnest($1::text[]))",
                &[
                    &active_tickers,
                    &format!("Removed in snapshot {}", snapshot_id),
                ],
            )
            .await?;

        Ok(result as i64)
    }

    /// Mark a ticker as SUSPECT (got 404 but was ACTIVE).
    pub async fn mark_ticker_suspect(
        &self,
        ticker: &str,
        error_code: &str,
        error_message: &str,
    ) -> Result<(), DbError> {
        self.client
            .execute(
                "UPDATE provider_universe 
                 SET status = 'SUSPECT', 
                     last_error_code = $2,
                     last_error_message = $3,
                     last_validated_at = NOW()
                 WHERE ticker = $1",
                &[&ticker, &error_code, &error_message],
            )
            .await?;
        Ok(())
    }

    /// Mark a ticker as INACTIVE with reason.
    pub async fn mark_ticker_inactive(&self, ticker: &str, reason: &str) -> Result<(), DbError> {
        self.client
            .execute(
                "UPDATE provider_universe 
                 SET status = 'INACTIVE', 
                     last_error_code = 'REMOVED_FROM_PROVIDER_LIST',
                     last_error_message = $2
                 WHERE ticker = $1",
                &[&ticker, &reason],
            )
            .await?;
        Ok(())
    }

    /// Update last_validated_at for a ticker (successful fetch).
    pub async fn mark_ticker_validated(&self, ticker: &str) -> Result<(), DbError> {
        self.client
            .execute(
                "UPDATE provider_universe 
                 SET last_validated_at = NOW(),
                     last_error_code = NULL,
                     last_error_message = NULL,
                     status = 'ACTIVE'
                 WHERE ticker = $1",
                &[&ticker],
            )
            .await?;
        Ok(())
    }

    /// Create a universe snapshot.
    pub async fn create_universe_snapshot(
        &self,
        snapshot_id: &str,
        _query_params: Option<&str>,
        total_count: i32,
        total_pages: i32,
        active_count: i32,
        new_count: i32,
        removed_count: i32,
    ) -> Result<(), DbError> {
        self.client
            .execute(
                "INSERT INTO universe_snapshots 
                 (snapshot_id, total_count, total_pages, active_count, new_count, removed_count)
                 VALUES ($1, $2, $3, $4, $5, $6)",
                &[
                    &snapshot_id,
                    &total_count,
                    &total_pages,
                    &active_count,
                    &new_count,
                    &removed_count,
                ],
            )
            .await?;
        Ok(())
    }

    /// Log a universe divergence event.
    pub async fn log_divergence(
        &self,
        ticker: &str,
        event_type: &str,
        was_listed: bool,
        got_404: bool,
        reconciliation_result: Option<&str>,
        decision: Option<&str>,
    ) -> Result<(), DbError> {
        self.client
            .execute(
                "INSERT INTO universe_divergences 
                 (ticker, event_type, was_listed, got_404, reconciliation_result, decision)
                 VALUES ($1, $2, $3, $4, $5, $6)",
                &[
                    &ticker,
                    &event_type,
                    &was_listed,
                    &got_404,
                    &reconciliation_result,
                    &decision,
                ],
            )
            .await?;
        Ok(())
    }

    /// Get divergences for reporting.
    pub async fn get_divergences(&self, limit: i64) -> Result<Vec<Divergence>, DbError> {
        let rows = self.client
            .query(
                "SELECT ticker, event_type, was_listed, got_404, reconciliation_result, decision, created_at
                 FROM universe_divergences
                 ORDER BY created_at DESC
                 LIMIT $1",
                &[&limit],
            )
            .await?;

        Ok(rows
            .iter()
            .map(|r| Divergence {
                ticker: r.get(0),
                event_type: r.get(1),
                was_listed: r.get(2),
                got_404: r.get(3),
                reconciliation_result: r.get(4),
                decision: r.get(5),
                created_at: r.get(6),
            })
            .collect())
    }

    // ========================================================================
    // V2 Eligibility Methods (listing_date / delisting_date)
    // ========================================================================

    /// Get all tickers with eligibility dates for V2 universe.
    pub async fn get_eligibility_timelines(&self) -> Result<Vec<EligibilityRow>, DbError> {
        let rows = self
            .client
            .query(
                "SELECT ticker, listing_date, delisting_date, eligibility_source, status
                 FROM provider_universe
                 WHERE listing_date IS NOT NULL
                 ORDER BY ticker",
                &[],
            )
            .await?;

        Ok(rows
            .iter()
            .map(|r| EligibilityRow {
                ticker: r.get(0),
                listing_date: r.get(1),
                delisting_date: r.get(2),
                eligibility_source: r.get(3),
                status: r.get(4),
            })
            .collect())
    }

    /// Get eligibility dates for a single ticker.
    pub async fn get_eligibility(&self, ticker: &str) -> Result<Option<EligibilityRow>, DbError> {
        let row = self
            .client
            .query_opt(
                "SELECT ticker, listing_date, delisting_date, eligibility_source, status
                 FROM provider_universe
                 WHERE ticker = $1",
                &[&ticker],
            )
            .await?;

        Ok(row.map(|r| EligibilityRow {
            ticker: r.get(0),
            listing_date: r.get(1),
            delisting_date: r.get(2),
            eligibility_source: r.get(3),
            status: r.get(4),
        }))
    }

    /// Set listing date for a ticker.
    pub async fn set_listing_date(
        &self,
        ticker: &str,
        listing_date: NaiveDate,
        source: &str,
    ) -> Result<(), DbError> {
        self.client
            .execute(
                "UPDATE provider_universe 
                 SET listing_date = $2,
                     eligibility_source = $3
                 WHERE ticker = $1",
                &[&ticker, &listing_date, &source],
            )
            .await?;
        Ok(())
    }

    /// Set delisting date for a ticker.
    pub async fn set_delisting_date(
        &self,
        ticker: &str,
        delisting_date: NaiveDate,
        source: &str,
    ) -> Result<(), DbError> {
        self.client
            .execute(
                "UPDATE provider_universe 
                 SET delisting_date = $2,
                     eligibility_source = $3
                 WHERE ticker = $1",
                &[&ticker, &delisting_date, &source],
            )
            .await?;
        Ok(())
    }

    /// Set both listing and delisting dates for a ticker.
    pub async fn set_eligibility_dates(
        &self,
        ticker: &str,
        listing_date: NaiveDate,
        delisting_date: Option<NaiveDate>,
        source: &str,
    ) -> Result<(), DbError> {
        self.client
            .execute(
                "UPDATE provider_universe 
                 SET listing_date = $2,
                     delisting_date = $3,
                     eligibility_source = $4
                 WHERE ticker = $1",
                &[&ticker, &listing_date, &delisting_date, &source],
            )
            .await?;
        Ok(())
    }

    /// Bulk upsert eligibility dates from CSV data.
    pub async fn bulk_upsert_eligibility(
        &self,
        rows: &[(String, NaiveDate, Option<NaiveDate>)],
        source: &str,
    ) -> Result<usize, DbError> {
        if rows.is_empty() {
            return Ok(0);
        }

        let mut updated = 0;
        for (ticker, listing_date, delisting_date) in rows {
            let result = self
                .client
                .execute(
                    "UPDATE provider_universe 
                     SET listing_date = COALESCE(listing_date, $2),
                         delisting_date = COALESCE(delisting_date, $3),
                         eligibility_source = COALESCE(eligibility_source, $4)
                     WHERE ticker = $1 AND listing_date IS NULL",
                    &[ticker, listing_date, delisting_date, &source],
                )
                .await?;
            updated += result as usize;
        }
        Ok(updated)
    }

    /// Get count of tickers with eligibility data.
    pub async fn get_eligibility_stats(&self) -> Result<EligibilityStats, DbError> {
        let row = self
            .client
            .query_one(
                "SELECT 
                    COUNT(*) FILTER (WHERE listing_date IS NOT NULL) as with_listing,
                    COUNT(*) FILTER (WHERE delisting_date IS NOT NULL) as with_delisting,
                    COUNT(*) FILTER (WHERE eligibility_source = 'DATA_DERIVED') as data_derived,
                    COUNT(*) FILTER (WHERE eligibility_source = 'PROVIDER_API') as provider_api,
                    COUNT(*) FILTER (WHERE eligibility_source = 'MANUAL') as manual,
                    COUNT(*) as total
                 FROM provider_universe",
                &[],
            )
            .await?;

        Ok(EligibilityStats {
            with_listing_date: row.get(0),
            with_delisting_date: row.get(1),
            data_derived: row.get(2),
            provider_api: row.get(3),
            manual: row.get(4),
            total: row.get(5),
        })
    }
}

// ============================================================================
// Provider Universe Types
// ============================================================================

#[derive(Debug, Clone)]
pub struct TickerStatus {
    pub ticker: String,
    pub status: String,
    pub asset_type: Option<String>,
    pub name: Option<String>,
    pub last_seen_at: Option<chrono::DateTime<Utc>>,
    pub last_error_code: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct UniverseStats {
    pub active: i64,
    pub inactive: i64,
    pub suspect: i64,
    pub total: i64,
}

#[derive(Debug, Clone)]
pub struct Divergence {
    pub ticker: String,
    pub event_type: String,
    pub was_listed: Option<bool>,
    pub got_404: Option<bool>,
    pub reconciliation_result: Option<String>,
    pub decision: Option<String>,
    pub created_at: chrono::DateTime<Utc>,
}

/// Eligibility row from provider_universe (V2 universe).
#[derive(Debug, Clone)]
pub struct EligibilityRow {
    pub ticker: String,
    pub listing_date: Option<NaiveDate>,
    pub delisting_date: Option<NaiveDate>,
    pub eligibility_source: Option<String>,
    pub status: String,
}

impl EligibilityRow {
    /// Check if ticker is eligible at a given date.
    pub fn is_eligible_at(&self, date: NaiveDate) -> bool {
        match (self.listing_date, self.delisting_date) {
            (Some(listing), Some(delisting)) => date >= listing && date <= delisting,
            (Some(listing), None) => date >= listing,
            (None, Some(delisting)) => date <= delisting,
            (None, None) => false,
        }
    }
}

/// Statistics about eligibility data coverage.
#[derive(Debug, Clone, Default)]
pub struct EligibilityStats {
    pub with_listing_date: i64,
    pub with_delisting_date: i64,
    pub data_derived: i64,
    pub provider_api: i64,
    pub manual: i64,
    pub total: i64,
}

// ============================================================================
// Watermark Types
// ============================================================================

#[derive(Debug, Clone)]
pub struct Watermark {
    pub symbol: String,
    pub interval: String,
    pub first_ts: Option<chrono::DateTime<Utc>>,
    pub last_ts: Option<chrono::DateTime<Utc>>,
    pub bar_count: Option<i32>,
    pub last_success_at: Option<chrono::DateTime<Utc>>,
    pub consecutive_failures: Option<i32>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TickerCapability {
    pub symbol: String,
    pub valid_intervals: Vec<String>,
    pub valid_ranges: Vec<String>,
    pub has_intraday: bool,
    pub max_range: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CoverageStats {
    pub total_tickers: i64,
    pub daily_coverage: i64,
    pub intraday_coverage: std::collections::HashMap<String, i64>,
    pub failed_count: i64,
}

#[derive(Debug)]
pub struct IngestionStats {
    pub total_instruments: i64,
    pub total_bars: i64,
    pub symbols_with_data: i64,
    pub earliest_date: Option<NaiveDate>,
    pub latest_date: Option<NaiveDate>,
    pub failed_symbols: i64,
}

#[derive(Debug)]
pub struct IntegrityIssue {
    pub symbol: String,
    pub issue_type: String,
    pub count: i64,
}

/// Fundamentals row from database.
#[derive(Debug, Clone)]
pub struct FundamentalsRow {
    pub symbol: String,
    pub snapshot_date: NaiveDate,
    pub price_earnings: Option<f64>,
    pub price_to_book: Option<f64>,
    pub earnings_per_share: Option<f64>,
    pub return_on_equity: Option<f64>,
    pub return_on_assets: Option<f64>,
    pub debt_to_equity: Option<f64>,
    pub profit_margins: Option<f64>,
    pub gross_margins: Option<f64>,
    pub operating_margins: Option<f64>,
    pub current_ratio: Option<f64>,
    pub quick_ratio: Option<f64>,
    pub market_cap: Option<i64>,
    pub enterprise_value: Option<i64>,
    pub dividend_yield: Option<f64>,
    pub earnings_growth: Option<f64>,
    pub revenue_growth: Option<f64>,
}

// ============================================================================
// Inventory Scanner Methods
// ============================================================================

impl Database {
    /// Alias for upsert_ohlcv_batch (consistency with aggregation executor).
    pub async fn upsert_ohlcv_daily_bars(
        &self,
        symbol: &str,
        bars: &[HistoricalBar],
    ) -> Result<usize, DbError> {
        self.upsert_ohlcv_batch(symbol, bars).await
    }

    /// Alias for upsert_ohlcv_intraday_batch (consistency with aggregation executor).
    pub async fn upsert_ohlcv_intraday_bars(
        &self,
        symbol: &str,
        interval: &str,
        bars: &[HistoricalBar],
    ) -> Result<usize, DbError> {
        self.upsert_ohlcv_intraday_batch(symbol, interval, bars)
            .await
    }

    /// Get OHLCV daily bar counts per symbol.
    pub async fn get_ohlcv_counts(
        &self,
        _interval: &str,
    ) -> Result<std::collections::HashMap<String, i64>, DbError> {
        let rows = self
            .client
            .query(
                "SELECT symbol, COUNT(*) as cnt FROM ohlcv_daily GROUP BY symbol",
                &[],
            )
            .await?;

        Ok(rows
            .iter()
            .map(|r| {
                let symbol: String = r.get(0);
                let count: i64 = r.get(1);
                (symbol, count)
            })
            .collect())
    }

    /// Get OHLCV intraday bar counts per (symbol, interval).
    pub async fn get_ohlcv_intraday_counts(
        &self,
    ) -> Result<std::collections::HashMap<(String, String), i64>, DbError> {
        let rows = self.client
            .query(
                "SELECT symbol, interval, COUNT(*) as cnt FROM ohlcv_intraday GROUP BY symbol, interval",
                &[],
            )
            .await?;

        Ok(rows
            .iter()
            .map(|r| {
                let symbol: String = r.get(0);
                let interval: String = r.get(1);
                let count: i64 = r.get(2);
                ((symbol, interval), count)
            })
            .collect())
    }

    // ========================================================================
    // Audit Integrity Queries (Read-Only)
    // ========================================================================

    /// Get detailed OHLCV daily counts with date ranges.
    pub async fn get_ohlcv_daily_audit(&self) -> Result<Vec<OhlcvAuditRow>, DbError> {
        let rows = self
            .client
            .query(
                "SELECT symbol, COUNT(*) as cnt, MIN(trading_date), MAX(trading_date)
                 FROM ohlcv_daily GROUP BY symbol",
                &[],
            )
            .await?;

        Ok(rows
            .iter()
            .map(|r| OhlcvAuditRow {
                symbol: r.get(0),
                interval: "1d".to_string(),
                bar_count: r.get(1),
                min_date: r.get(2),
                max_date: r.get(3),
            })
            .collect())
    }

    /// Get detailed OHLCV intraday counts with timestamp ranges.
    pub async fn get_ohlcv_intraday_audit(&self) -> Result<Vec<OhlcvAuditRow>, DbError> {
        let rows = self.client
            .query(
                "SELECT symbol, interval, COUNT(*) as cnt, MIN(timestamp)::date, MAX(timestamp)::date
                 FROM ohlcv_intraday GROUP BY symbol, interval",
                &[],
            )
            .await?;

        Ok(rows
            .iter()
            .map(|r| OhlcvAuditRow {
                symbol: r.get(0),
                interval: r.get(1),
                bar_count: r.get(2),
                min_date: r.get(3),
                max_date: r.get(4),
            })
            .collect())
    }

    /// Get capabilities map for all tickers.
    pub async fn get_all_capabilities(
        &self,
    ) -> Result<std::collections::HashMap<String, Vec<String>>, DbError> {
        let rows = self.client
            .query(
                "SELECT symbol, valid_intervals FROM ticker_capabilities WHERE valid_intervals IS NOT NULL",
                &[],
            )
            .await?;

        Ok(rows
            .iter()
            .map(|r| {
                let symbol: String = r.get(0);
                let intervals: Vec<String> = r.get::<_, Option<Vec<String>>>(1).unwrap_or_default();
                (symbol, intervals)
            })
            .collect())
    }
}

#[derive(Debug, Clone)]
pub struct OhlcvAuditRow {
    pub symbol: String,
    pub interval: String,
    pub bar_count: i64,
    pub min_date: Option<NaiveDate>,
    pub max_date: Option<NaiveDate>,
}
