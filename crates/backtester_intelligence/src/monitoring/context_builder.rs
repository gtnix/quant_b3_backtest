//! Context Builder for Monitoring - populates MonitoringContext from Neon database.
//!
//! Connects to Neon PostgreSQL and runs queries to gather:
//! - Data health metrics (freshness, coverage, nulls, outliers)
//! - Historical data for drift detection
//! - Performance metrics for regression checks

use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use tokio_postgres::Client;

use crate::filters::Market;
use super::data_health::DataContext;
use super::drift::DriftContext;
use super::regressions::RegressionContext;

/// Result type for context builder operations.
pub type BuilderResult<T> = Result<T, BuilderError>;

/// Errors that can occur during context building.
#[derive(Debug, thiserror::Error)]
pub enum BuilderError {
    #[error("Database error: {0}")]
    Database(String),
    #[error("Connection error: {0}")]
    Connection(String),
    #[error("Query error: {0}")]
    Query(String),
}

impl From<tokio_postgres::Error> for BuilderError {
    fn from(e: tokio_postgres::Error) -> Self {
        BuilderError::Query(e.to_string())
    }
}

/// OHLCV audit data from database.
#[derive(Debug, Clone)]
pub struct OhlcvAudit {
    pub symbol: String,
    pub bar_count: i64,
    pub min_date: Option<NaiveDate>,
    pub max_date: Option<NaiveDate>,
}

/// Watermark data from database.
#[derive(Debug, Clone)]
pub struct WatermarkData {
    pub symbol: String,
    pub interval: String,
    pub last_ts: Option<NaiveDate>,
    pub bar_count: i32,
}

/// Interest rate stats from database.
#[derive(Debug, Clone, Default)]
pub struct InterestRateStats {
    pub br_count: i64,
    pub us_count: i64,
    pub br_last_date: Option<NaiveDate>,
    pub us_last_date: Option<NaiveDate>,
}

/// Context builder that populates MonitoringContext from database.
pub struct ContextBuilder<'a> {
    client: &'a Client,
}

impl<'a> ContextBuilder<'a> {
    /// Create new context builder with database client.
    pub fn new(client: &'a Client) -> Self {
        Self { client }
    }

    /// Build complete DataContext for a specific date and market.
    pub async fn build_data_context(
        &self,
        as_of: NaiveDate,
        markets: &[Market],
    ) -> BuilderResult<DataContext> {
        let mut ctx = DataContext::new(as_of);

        for &market in markets {
            // Get OHLCV audit data
            let audits = self.get_ohlcv_audit(market).await?;
            
            // Calculate freshness (last OHLCV date)
            let max_date = audits.iter()
                .filter_map(|a| a.max_date)
                .max();
            if let Some(date) = max_date {
                ctx.last_ohlcv_date.insert(market, date);
            }

            // Calculate coverage
            let total_symbols = audits.len();
            let min_bars = 30; // Minimum bars for "sufficient data"
            let symbols_with_data = audits.iter()
                .filter(|a| a.bar_count >= min_bars)
                .count();

            ctx.symbol_count.insert(market, total_symbols);
            ctx.symbols_with_data.insert(market, symbols_with_data);

            // Get watermarks for regression check
            let watermarks = self.get_watermarks(market).await?;
            for wm in watermarks {
                if let Some(last) = wm.last_ts {
                    ctx.current_watermarks.insert(market, last);
                }
            }
        }

        // Get null counts
        let null_counts = self.get_null_counts().await?;
        ctx.null_counts = null_counts;

        // Get outlier counts (price <= 0)
        let outlier_counts = self.get_outlier_counts().await?;
        ctx.outlier_counts = outlier_counts;

        // Get total rows for percentage calculation
        ctx.total_rows = self.get_total_ohlcv_rows().await?;

        // Get dividend count (last 30 days)
        ctx.dividends_30d = self.get_dividend_count(as_of, 30).await?;

        // Get dividend types
        ctx.dividend_types = self.get_dividend_types().await?;

        // Get interest rate stats
        let ir_stats = self.get_interest_rate_stats().await?;
        if let Some(date) = ir_stats.br_last_date {
            ctx.last_interest_rate.insert(Market::BR, date);
        }
        if let Some(date) = ir_stats.us_last_date {
            ctx.last_interest_rate.insert(Market::US, date);
        }
        ctx.interest_rate_count.insert(Market::BR, ir_stats.br_count as usize);
        ctx.interest_rate_count.insert(Market::US, ir_stats.us_count as usize);

        // Schema is valid if we got here without errors
        ctx.schema_valid = true;

        Ok(ctx)
    }

    /// Build DriftContext with baseline and current score distributions.
    /// For now, returns empty context since we don't have backtest history.
    pub async fn build_drift_context(
        &self,
        as_of: NaiveDate,
        _baseline_days: u32,
    ) -> BuilderResult<DriftContext> {
        let ctx = DriftContext::new(as_of);
        // Drift context requires historical backtest results which don't exist yet
        // Returns empty context - drift checks will pass due to insufficient samples
        Ok(ctx)
    }

    /// Build RegressionContext with performance metrics.
    /// For now, returns default context since we don't have backtest history.
    pub async fn build_regression_context(
        &self,
        as_of: NaiveDate,
    ) -> BuilderResult<RegressionContext> {
        let ctx = RegressionContext::new(as_of);
        // Regression context requires backtest equity curve and turnover history
        // Returns default context - regression checks will use default thresholds
        Ok(ctx)
    }

    // ========================================================================
    // Database Query Methods
    // ========================================================================

    /// Get OHLCV audit data for a market.
    async fn get_ohlcv_audit(&self, market: Market) -> BuilderResult<Vec<OhlcvAudit>> {
        let pattern = match market {
            Market::BR => "%", // BR symbols like PETR4, VALE3
            Market::US => "%", // US symbols - will filter differently
        };

        // For BR, symbols end with digit. For US, all letters.
        let query = match market {
            Market::BR => {
                "SELECT symbol, COUNT(*) as cnt, MIN(trading_date), MAX(trading_date)
                 FROM ohlcv_daily 
                 WHERE symbol ~ '^[A-Z]{4}[0-9]{1,2}$'
                 GROUP BY symbol"
            }
            Market::US => {
                "SELECT symbol, COUNT(*) as cnt, MIN(trading_date), MAX(trading_date)
                 FROM ohlcv_daily 
                 WHERE symbol ~ '^[A-Z]{1,5}$' OR symbol LIKE '%-%'
                 GROUP BY symbol"
            }
        };

        let rows = self.client.query(query, &[]).await?;

        Ok(rows.iter().map(|r| OhlcvAudit {
            symbol: r.get(0),
            bar_count: r.get(1),
            min_date: r.get(2),
            max_date: r.get(3),
        }).collect())
    }

    /// Get watermarks for a market.
    async fn get_watermarks(&self, market: Market) -> BuilderResult<Vec<WatermarkData>> {
        let query = "SELECT symbol, interval, last_ts, bar_count 
                     FROM ingestion_watermarks 
                     WHERE interval = '1d'";

        let rows = self.client.query(query, &[]).await?;

        // Filter by market pattern
        let wms: Vec<WatermarkData> = rows.iter()
            .map(|r| WatermarkData {
                symbol: r.get(0),
                interval: r.get(1),
                last_ts: r.get::<_, Option<chrono::DateTime<chrono::Utc>>>(2)
                    .map(|dt| dt.date_naive()),
                bar_count: r.get(3),
            })
            .filter(|wm| {
                match market {
                    Market::BR => wm.symbol.chars().last().map(|c| c.is_numeric()).unwrap_or(false),
                    Market::US => wm.symbol.chars().all(|c| c.is_alphabetic() || c == '-'),
                }
            })
            .collect();

        Ok(wms)
    }

    /// Get null counts by field.
    async fn get_null_counts(&self) -> BuilderResult<HashMap<String, usize>> {
        let mut counts = HashMap::new();

        // Check for null close prices
        let row = self.client.query_one(
            "SELECT COUNT(*) FROM ohlcv_daily WHERE close IS NULL",
            &[]
        ).await?;
        let close_nulls: i64 = row.get(0);
        if close_nulls > 0 {
            counts.insert("close".to_string(), close_nulls as usize);
        }

        // Check for null volumes
        let row = self.client.query_one(
            "SELECT COUNT(*) FROM ohlcv_daily WHERE volume IS NULL",
            &[]
        ).await?;
        let vol_nulls: i64 = row.get(0);
        if vol_nulls > 0 {
            counts.insert("volume".to_string(), vol_nulls as usize);
        }

        Ok(counts)
    }

    /// Get outlier counts.
    async fn get_outlier_counts(&self) -> BuilderResult<HashMap<String, usize>> {
        let mut counts = HashMap::new();

        // Check for zero/negative prices
        let row = self.client.query_one(
            "SELECT COUNT(*) FROM ohlcv_daily WHERE close <= 0",
            &[]
        ).await?;
        let price_outliers: i64 = row.get(0);
        if price_outliers > 0 {
            counts.insert("price_zero_negative".to_string(), price_outliers as usize);
        }

        // Check for extremely high prices (potential data errors)
        let row = self.client.query_one(
            "SELECT COUNT(*) FROM ohlcv_daily WHERE close > 100000",
            &[]
        ).await?;
        let high_outliers: i64 = row.get(0);
        if high_outliers > 0 {
            counts.insert("price_extreme_high".to_string(), high_outliers as usize);
        }

        Ok(counts)
    }

    /// Get total OHLCV rows.
    async fn get_total_ohlcv_rows(&self) -> BuilderResult<usize> {
        let row = self.client.query_one(
            "SELECT COUNT(*) FROM ohlcv_daily",
            &[]
        ).await?;
        let count: i64 = row.get(0);
        Ok(count as usize)
    }

    /// Get dividend count in last N days.
    async fn get_dividend_count(&self, as_of: NaiveDate, days: i32) -> BuilderResult<u32> {
        let start_date = as_of - chrono::Duration::days(days as i64);
        
        // Try to query dividends table - may not exist
        let result = self.client.query_opt(
            "SELECT COUNT(*) FROM dividends WHERE ex_date >= $1 AND ex_date <= $2",
            &[&start_date, &as_of]
        ).await;

        match result {
            Ok(Some(row)) => {
                let count: i64 = row.get(0);
                Ok(count as u32)
            }
            Ok(None) => Ok(0),
            Err(_) => Ok(0), // Table may not exist
        }
    }

    /// Get dividend types found in database.
    async fn get_dividend_types(&self) -> BuilderResult<Vec<String>> {
        let result = self.client.query(
            "SELECT DISTINCT type FROM dividends LIMIT 10",
            &[]
        ).await;

        match result {
            Ok(rows) => Ok(rows.iter().map(|r| r.get(0)).collect()),
            Err(_) => Ok(vec![]), // Table may not exist
        }
    }

    /// Get interest rate statistics.
    async fn get_interest_rate_stats(&self) -> BuilderResult<InterestRateStats> {
        let mut stats = InterestRateStats::default();

        // BR rates (SELIC)
        let result = self.client.query_opt(
            "SELECT COUNT(*), MAX(date) FROM interest_rates WHERE region = 'BR'",
            &[]
        ).await;
        if let Ok(Some(row)) = result {
            stats.br_count = row.get(0);
            stats.br_last_date = row.get(1);
        }

        // US rates
        let result = self.client.query_opt(
            "SELECT COUNT(*), MAX(date) FROM interest_rates WHERE region = 'US'",
            &[]
        ).await;
        if let Ok(Some(row)) = result {
            stats.us_count = row.get(0);
            stats.us_last_date = row.get(1);
        }

        Ok(stats)
    }

    /// Get previous watermarks for regression detection.
    pub async fn get_previous_watermarks(
        &self,
        as_of: NaiveDate,
        lookback_days: i32,
    ) -> BuilderResult<HashMap<Market, NaiveDate>> {
        let mut prev = HashMap::new();
        
        // This would require historical watermark tracking which we don't have
        // For now, return empty - watermark regression check will pass
        
        Ok(prev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_error_conversion() {
        let err = BuilderError::Database("test error".to_string());
        assert!(format!("{}", err).contains("test error"));
    }

    #[test]
    fn test_ohlcv_audit_struct() {
        let audit = OhlcvAudit {
            symbol: "PETR4".to_string(),
            bar_count: 1000,
            min_date: Some(NaiveDate::from_ymd_opt(2020, 1, 1).unwrap()),
            max_date: Some(NaiveDate::from_ymd_opt(2024, 1, 1).unwrap()),
        };
        assert_eq!(audit.symbol, "PETR4");
        assert_eq!(audit.bar_count, 1000);
    }
}

