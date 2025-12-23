//! Database layer for Neon Postgres.

use chrono::{NaiveDate, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;
use std::sync::Arc;
use thiserror::Error;
use tokio_postgres::Client;
use tracing::{info, debug};

use crate::brapi::{HistoricalBar, QuoteResult, RequestMetrics, StockInfo};

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

        // Use rustls for SSL
        let root_store = rustls::RootCertStore::from_iter(
            webpki_roots::TLS_SERVER_ROOTS.iter().cloned()
        );
        
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

    pub async fn verify_schema(&self) -> Result<(), DbError> {
        let tables = ["instruments", "universe_membership", "ohlcv_daily", 
                      "ingestion_state", "api_request_log", "api_budget"];
        
        for table in tables {
            let row = self.client
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

        let row = self.client
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
        let rows = self.client
            .query(
                "SELECT DISTINCT symbol FROM universe_membership 
                 WHERE universe_name = $1 
                 ORDER BY symbol",
                &[&universe],
            )
            .await?;
        Ok(rows.iter().map(|r| r.get(0)).collect())
    }

    /// Insert OHLCV bars with batch upsert.
    pub async fn upsert_ohlcv_batch(
        &self,
        symbol: &str,
        bars: &[HistoricalBar],
    ) -> Result<usize, DbError> {
        if bars.is_empty() {
            return Ok(0);
        }

        // Build batch insert with multiple VALUES
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
                    param_idx, param_idx + 1, param_idx + 2, param_idx + 3,
                    param_idx + 4, param_idx + 5, param_idx + 6, param_idx + 7
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

            let params_refs: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = 
                params.iter().map(|p| p.as_ref() as &(dyn tokio_postgres::types::ToSql + Sync)).collect();

            let result = self.client.execute(&query, &params_refs).await?;
            total_inserted += result as usize;
        }

        Ok(total_inserted)
    }

    /// Get last bar date for a symbol.
    pub async fn get_last_bar_date(&self, symbol: &str) -> Result<Option<NaiveDate>, DbError> {
        let row = self.client
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
        let invalid_rows = self.client
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
        let row = self.client
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

