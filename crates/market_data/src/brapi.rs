//! Brapi API Client with rate limiting, backoff, and circuit breaker.
//!
//! Environment variables:
//! - `BRAPI_TOKEN` (required): API authentication token
//! - `BRAPI_BASE_URL` (optional): API base URL, default: https://brapi.dev
//! - `BRAPI_REQUESTS_PER_MINUTE` (optional): Rate limit, default: 60
//! - `BRAPI_MAX_RETRIES` (optional): Max retries per request, default: 3
//! - `BRAPI_TIMEOUT_SECS` (optional): Request timeout in seconds, default: 30
//! - `BRAPI_MAX_TICKERS_PER_REQUEST` (optional): Max tickers per batch, default: 20

use chrono::{DateTime, NaiveDate, Utc};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, info, warn};

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug, Clone)]
pub enum BrapiError {
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("API error: {status} - {message}")]
    Api { status: u16, message: String },
    #[error("Rate limit exceeded after {retries} retries")]
    RateLimit { retries: u32 },
    #[error("Quota exceeded (402 Payment Required)")]
    QuotaExceeded,
    #[error("Not found (404): {message}")]
    NotFound { message: String },
    #[error("Unauthorized (401): Token missing or invalid")]
    Unauthorized,
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Config error: {0}")]
    Config(String),
    #[error("Circuit breaker open: too many consecutive failures")]
    CircuitBreakerOpen,
    #[error("Timeout after {secs}s")]
    Timeout { secs: u64 },
}

impl From<reqwest::Error> for BrapiError {
    fn from(e: reqwest::Error) -> Self {
        if e.is_timeout() {
            BrapiError::Timeout { secs: 30 }
        } else {
            BrapiError::Http(e.to_string())
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Brapi client configuration loaded from environment.
#[derive(Debug, Clone)]
pub struct BrapiConfig {
    pub token: String,
    pub base_url: String,
    pub requests_per_minute: u64,
    pub max_retries: u32,
    pub timeout_secs: u64,
    pub max_tickers_per_request: usize,
}

impl BrapiConfig {
    /// Load configuration from environment variables.
    ///
    /// # Errors
    /// Returns `BrapiError::Config` if `BRAPI_TOKEN` is not set.
    pub fn from_env() -> Result<Self, BrapiError> {
        let token = std::env::var("BRAPI_TOKEN")
            .map_err(|_| BrapiError::Config(
                "BRAPI_TOKEN environment variable is required. Get your token at https://brapi.dev/dashboard".into()
            ))?;

        if token.trim().is_empty() {
            return Err(BrapiError::Config("BRAPI_TOKEN cannot be empty".into()));
        }

        let base_url =
            std::env::var("BRAPI_BASE_URL").unwrap_or_else(|_| "https://brapi.dev".into());

        let requests_per_minute: u64 = std::env::var("BRAPI_REQUESTS_PER_MINUTE")
            .unwrap_or_else(|_| "60".into())
            .parse()
            .unwrap_or(60);

        let max_retries: u32 = std::env::var("BRAPI_MAX_RETRIES")
            .unwrap_or_else(|_| "3".into())
            .parse()
            .unwrap_or(3);

        let timeout_secs: u64 = std::env::var("BRAPI_TIMEOUT_SECS")
            .unwrap_or_else(|_| "30".into())
            .parse()
            .unwrap_or(30);

        let max_tickers_per_request: usize = std::env::var("BRAPI_MAX_TICKERS_PER_REQUEST")
            .unwrap_or_else(|_| "20".into())
            .parse()
            .unwrap_or(20);

        Ok(Self {
            token,
            base_url,
            requests_per_minute,
            max_retries,
            timeout_secs,
            max_tickers_per_request,
        })
    }

    /// Redacted config for logging (hides token).
    pub fn redacted(&self) -> String {
        format!(
            "BrapiConfig {{ base_url: {}, requests_per_minute: {}, max_retries: {}, timeout_secs: {}, max_tickers: {} }}",
            self.base_url, self.requests_per_minute, self.max_retries, self.timeout_secs, self.max_tickers_per_request
        )
    }
}

// ============================================================================
// API Response Types
// ============================================================================

#[derive(Debug, Deserialize, Clone)]
pub struct QuoteResponse {
    #[serde(default)]
    pub results: Vec<QuoteResult>,
    #[serde(rename = "requestedAt")]
    pub requested_at: Option<String>,
    /// API response time (can be number or string)
    #[serde(default)]
    pub took: serde_json::Value,
}

#[derive(Debug, Deserialize, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct QuoteResult {
    #[serde(default)]
    pub symbol: String,
    pub short_name: Option<String>,
    pub long_name: Option<String>,
    pub currency: Option<String>,
    pub market_cap: Option<f64>,
    #[serde(default)]
    pub historical_data_price: Vec<HistoricalBar>,
    /// Valid ranges returned by the API for this ticker
    #[serde(default)]
    pub valid_ranges: Vec<String>,
    /// Valid intervals returned by the API for this ticker
    #[serde(default)]
    pub valid_intervals: Vec<String>,
    /// The range actually used for this request
    pub used_range: Option<String>,
    /// The interval actually used for this request
    pub used_interval: Option<String>,
    /// Sector information
    pub sector: Option<String>,
    /// Industry information
    pub industry: Option<String>,
    /// Logo URL
    pub logourl: Option<String>,
    /// Capture all other fields dynamically
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

impl QuoteResult {
    pub fn market_cap_i64(&self) -> Option<i64> {
        self.market_cap.map(|v| v as i64)
    }

    /// Check if this ticker supports intraday intervals.
    pub fn has_intraday(&self) -> bool {
        self.valid_intervals.iter().any(|i| {
            matches!(
                i.as_str(),
                "1m" | "2m" | "5m" | "15m" | "30m" | "60m" | "90m" | "1h"
            )
        })
    }

    /// Get the maximum range available.
    pub fn max_range(&self) -> Option<&str> {
        // Ordered by duration
        const RANGE_ORDER: &[&str] = &[
            "max", "10y", "5y", "2y", "1y", "ytd", "6mo", "3mo", "1mo", "5d", "1d",
        ];
        for r in RANGE_ORDER {
            if self.valid_ranges.iter().any(|vr| vr == *r) {
                return Some(r);
            }
        }
        self.valid_ranges.first().map(|s| s.as_str())
    }
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoricalBar {
    pub date: i64,
    pub open: Option<f64>,
    pub high: Option<f64>,
    pub low: Option<f64>,
    pub close: Option<f64>,
    pub volume: Option<f64>,
    pub adjusted_close: Option<f64>,
}

impl HistoricalBar {
    pub fn volume_i64(&self) -> Option<i64> {
        self.volume.map(|v| v as i64)
    }

    pub fn trading_date(&self) -> Option<NaiveDate> {
        DateTime::from_timestamp(self.date, 0).map(|dt| dt.date_naive())
    }

    pub fn timestamp_utc(&self) -> Option<DateTime<Utc>> {
        DateTime::from_timestamp(self.date, 0)
    }

    /// Check if bar has valid OHLCV data.
    pub fn is_valid(&self) -> bool {
        self.open.is_some() && self.high.is_some() && self.low.is_some() && self.close.is_some()
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ListResponse {
    pub stocks: Vec<StockInfo>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StockInfo {
    pub stock: String,
    pub name: Option<String>,
    pub close: Option<f64>,
    pub change: Option<f64>,
    pub volume: Option<i64>,
    pub market_cap: Option<i64>,
    pub sector: Option<String>,
    #[serde(rename = "type")]
    pub asset_type: Option<String>,
}

// ============================================================================
// Fundamental Data Types
// ============================================================================

/// Financial data from financialData module (TTM metrics).
#[derive(Debug, Deserialize, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct FinancialData {
    pub symbol: Option<String>,
    pub current_price: Option<f64>,
    pub ebitda: Option<f64>,
    pub quick_ratio: Option<f64>,
    pub current_ratio: Option<f64>,
    pub debt_to_equity: Option<f64>,
    pub revenue_per_share: Option<f64>,
    pub return_on_assets: Option<f64>,
    pub return_on_equity: Option<f64>,
    pub earnings_growth: Option<f64>,
    pub revenue_growth: Option<f64>,
    pub gross_margins: Option<f64>,
    pub ebitda_margins: Option<f64>,
    pub operating_margins: Option<f64>,
    pub profit_margins: Option<f64>,
    pub total_cash: Option<f64>,
    pub total_cash_per_share: Option<f64>,
    pub total_debt: Option<f64>,
    pub total_revenue: Option<f64>,
    pub gross_profits: Option<f64>,
    pub operating_cashflow: Option<f64>,
    pub free_cashflow: Option<f64>,
    pub financial_currency: Option<String>,
    pub updated_at: Option<String>,
    #[serde(rename = "type")]
    pub data_type: Option<String>,
}

/// Key statistics from defaultKeyStatistics module.
#[derive(Debug, Deserialize, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct DefaultKeyStatistics {
    pub symbol: Option<String>,
    pub enterprise_value: Option<f64>,
    pub forward_pe: Option<f64>,
    pub profit_margins: Option<f64>,
    pub float_shares: Option<f64>,
    pub shares_outstanding: Option<f64>,
    pub held_percent_insiders: Option<f64>,
    pub held_percent_institutions: Option<f64>,
    pub beta: Option<f64>,
    pub book_value: Option<f64>,
    pub price_to_book: Option<f64>,
    pub last_fiscal_year_end: Option<String>,
    pub next_fiscal_year_end: Option<String>,
    pub most_recent_quarter: Option<String>,
    pub earnings_quarterly_growth: Option<f64>,
    pub net_income_to_common: Option<f64>,
    pub trailing_eps: Option<f64>,
    pub forward_eps: Option<f64>,
    pub peg_ratio: Option<f64>,
    pub last_split_factor: Option<String>,
    pub last_split_date: Option<i64>,
    pub enterprise_to_revenue: Option<f64>,
    pub enterprise_to_ebitda: Option<f64>,
    #[serde(rename = "52WeekChange")]
    pub week_52_change: Option<f64>,
    pub last_dividend_value: Option<f64>,
    pub last_dividend_date: Option<String>,
    pub dividend_yield: Option<f64>,
    pub ytd_return: Option<f64>,
    pub total_assets: Option<f64>,
    pub updated_at: Option<String>,
    #[serde(rename = "type")]
    pub data_type: Option<String>,
}

/// Company profile from summaryProfile module.
#[derive(Debug, Deserialize, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SummaryProfile {
    pub symbol: Option<String>,
    pub address1: Option<String>,
    pub address2: Option<String>,
    pub city: Option<String>,
    pub state: Option<String>,
    pub zip: Option<String>,
    pub country: Option<String>,
    pub phone: Option<String>,
    pub website: Option<String>,
    pub industry: Option<String>,
    pub industry_key: Option<String>,
    pub industry_disp: Option<String>,
    pub sector: Option<String>,
    pub sector_key: Option<String>,
    pub sector_disp: Option<String>,
    pub long_business_summary: Option<String>,
    pub full_time_employees: Option<i32>,
    pub updated_at: Option<String>,
}

/// Cash dividend entry.
#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CashDividend {
    pub asset_issued: Option<String>,
    pub payment_date: Option<String>,
    pub rate: Option<f64>,
    pub related_to: Option<String>,
    pub approved_on: Option<String>,
    pub isin_code: Option<String>,
    pub label: Option<String>,
    pub last_date_prior: Option<String>,
    pub remarks: Option<String>,
}

/// Dividends data container.
#[derive(Debug, Deserialize, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct DividendsData {
    #[serde(default)]
    pub cash_dividends: Vec<CashDividend>,
    #[serde(default)]
    pub stock_dividends: Vec<serde_json::Value>,
    #[serde(default)]
    pub subscriptions: Vec<serde_json::Value>,
}

/// Extended quote result with fundamental data.
#[derive(Debug, Deserialize, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct FundamentalQuoteResult {
    #[serde(default)]
    pub symbol: String,
    pub short_name: Option<String>,
    pub long_name: Option<String>,
    pub currency: Option<String>,
    pub market_cap: Option<f64>,
    pub regular_market_price: Option<f64>,

    // Basic fundamentals (from fundamental=true)
    pub price_earnings: Option<f64>,
    pub earnings_per_share: Option<f64>,

    // Module data
    pub financial_data: Option<FinancialData>,
    pub default_key_statistics: Option<DefaultKeyStatistics>,
    pub summary_profile: Option<SummaryProfile>,
    pub dividends_data: Option<DividendsData>,

    // Sector/Industry from base
    pub sector: Option<String>,
    pub industry: Option<String>,
    pub logourl: Option<String>,
}

/// Response for fundamental data fetch.
#[derive(Debug, Deserialize, Clone)]
pub struct FundamentalResponse {
    #[serde(default)]
    pub results: Vec<FundamentalQuoteResult>,
    #[serde(rename = "requestedAt")]
    pub requested_at: Option<String>,
    #[serde(default)]
    pub took: serde_json::Value,
}

/// Parsed fundamental snapshot ready for database insert.
#[derive(Debug, Clone)]
pub struct FundamentalSnapshot {
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
    pub market_cap: Option<f64>,
    pub enterprise_value: Option<f64>,
    pub dividend_yield: Option<f64>,
    pub last_dividend_value: Option<f64>,
    pub last_dividend_date: Option<NaiveDate>,
    pub earnings_growth: Option<f64>,
    pub revenue_growth: Option<f64>,
    pub free_cash_flow: Option<f64>,
    pub operating_cash_flow: Option<f64>,
}

impl FundamentalSnapshot {
    /// Build from FundamentalQuoteResult.
    pub fn from_quote(quote: &FundamentalQuoteResult, snapshot_date: NaiveDate) -> Self {
        let fin = quote.financial_data.as_ref();
        let stats = quote.default_key_statistics.as_ref();

        Self {
            symbol: quote.symbol.clone(),
            snapshot_date,
            price_earnings: quote.price_earnings,
            price_to_book: stats.and_then(|s| s.price_to_book),
            earnings_per_share: quote
                .earnings_per_share
                .or(stats.and_then(|s| s.trailing_eps)),
            return_on_equity: fin.and_then(|f| f.return_on_equity),
            return_on_assets: fin.and_then(|f| f.return_on_assets),
            debt_to_equity: fin.and_then(|f| f.debt_to_equity),
            profit_margins: fin.and_then(|f| f.profit_margins),
            gross_margins: fin.and_then(|f| f.gross_margins),
            operating_margins: fin.and_then(|f| f.operating_margins),
            current_ratio: fin.and_then(|f| f.current_ratio),
            quick_ratio: fin.and_then(|f| f.quick_ratio),
            market_cap: quote.market_cap,
            enterprise_value: stats.and_then(|s| s.enterprise_value),
            dividend_yield: stats.and_then(|s| s.dividend_yield),
            last_dividend_value: stats.and_then(|s| s.last_dividend_value),
            last_dividend_date: None, // Parse from string if needed
            earnings_growth: fin.and_then(|f| f.earnings_growth),
            revenue_growth: fin.and_then(|f| f.revenue_growth),
            free_cash_flow: fin.and_then(|f| f.free_cashflow),
            operating_cash_flow: fin.and_then(|f| f.operating_cashflow),
        }
    }
}

/// Parsed dividend entry for database.
#[derive(Debug, Clone)]
pub struct DividendEntry {
    pub symbol: String,
    pub payment_date: Option<NaiveDate>,
    pub ex_date: Option<NaiveDate>,
    pub rate: f64,
    pub dividend_type: String,
    pub related_to: Option<String>,
}

impl DividendEntry {
    /// Parse from CashDividend.
    pub fn from_cash_dividend(symbol: &str, div: &CashDividend) -> Option<Self> {
        let rate = div.rate?;

        let payment_date = div
            .payment_date
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.date_naive());

        let ex_date = div
            .last_date_prior
            .as_ref()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.date_naive());

        Some(Self {
            symbol: symbol.to_string(),
            payment_date,
            ex_date,
            rate,
            dividend_type: div.label.clone().unwrap_or_else(|| "DIVIDENDO".to_string()),
            related_to: div.related_to.clone(),
        })
    }
}

// ============================================================================
// Request Metrics
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct RequestMetrics {
    pub status: u16,
    pub duration_ms: u32,
    pub response_bytes: u32,
    pub retries: u32,
    pub endpoint: String,
}

// ============================================================================
// Circuit Breaker
// ============================================================================

/// Simple circuit breaker to prevent cascading failures.
#[derive(Debug)]
pub struct CircuitBreaker {
    failure_count: AtomicU32,
    is_open: AtomicBool,
    threshold: u32,
    last_failure: std::sync::Mutex<Option<Instant>>,
    reset_timeout: Duration,
}

impl CircuitBreaker {
    pub fn new(threshold: u32, reset_timeout_secs: u64) -> Self {
        Self {
            failure_count: AtomicU32::new(0),
            is_open: AtomicBool::new(false),
            threshold,
            last_failure: std::sync::Mutex::new(None),
            reset_timeout: Duration::from_secs(reset_timeout_secs),
        }
    }

    pub fn is_open(&self) -> bool {
        if !self.is_open.load(Ordering::SeqCst) {
            return false;
        }
        // Check if we should try to reset
        if let Ok(last) = self.last_failure.lock() {
            if let Some(t) = *last {
                if t.elapsed() > self.reset_timeout {
                    self.is_open.store(false, Ordering::SeqCst);
                    self.failure_count.store(0, Ordering::SeqCst);
                    return false;
                }
            }
        }
        true
    }

    pub fn record_success(&self) {
        self.failure_count.store(0, Ordering::SeqCst);
        self.is_open.store(false, Ordering::SeqCst);
    }

    pub fn record_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
        if count >= self.threshold {
            self.is_open.store(true, Ordering::SeqCst);
            if let Ok(mut last) = self.last_failure.lock() {
                *last = Some(Instant::now());
            }
            warn!(
                "Circuit breaker opened after {} consecutive failures",
                count
            );
        }
    }

    pub fn failure_count(&self) -> u32 {
        self.failure_count.load(Ordering::SeqCst)
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(10, 60) // 10 failures, 60s reset
    }
}

// ============================================================================
// Aggregate Statistics
// ============================================================================

/// Aggregated request statistics for reporting.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregateStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_retries: u64,
    pub rate_limit_hits: u64,
    pub total_bytes: u64,
    pub total_duration_ms: u64,
    pub by_status: std::collections::HashMap<u16, u64>,
}

impl AggregateStats {
    pub fn record(&mut self, metrics: &RequestMetrics, success: bool) {
        self.total_requests += 1;
        self.total_duration_ms += metrics.duration_ms as u64;
        self.total_bytes += metrics.response_bytes as u64;
        self.total_retries += metrics.retries as u64;
        *self.by_status.entry(metrics.status).or_insert(0) += 1;
        if success {
            self.successful_requests += 1;
        } else {
            self.failed_requests += 1;
            if metrics.status == 429 {
                self.rate_limit_hits += 1;
            }
        }
    }

    pub fn avg_latency_ms(&self) -> f64 {
        if self.total_requests > 0 {
            self.total_duration_ms as f64 / self.total_requests as f64
        } else {
            0.0
        }
    }
}

// ============================================================================
// Brapi Client
// ============================================================================

pub struct BrapiClient {
    client: reqwest::Client,
    config: BrapiConfig,
    last_request: std::sync::Mutex<Instant>,
    min_request_interval: Duration,
    circuit_breaker: Arc<CircuitBreaker>,
    stats: std::sync::Mutex<AggregateStats>,
}

impl BrapiClient {
    /// Create a new Brapi client from environment configuration.
    pub fn new() -> Result<Self, BrapiError> {
        let config = BrapiConfig::from_env()?;
        Self::with_config(config)
    }

    /// Create a new Brapi client with explicit configuration.
    pub fn with_config(config: BrapiConfig) -> Result<Self, BrapiError> {
        let min_interval = Duration::from_millis(60_000 / config.requests_per_minute.max(1));

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", config.token))
                .map_err(|_| BrapiError::Config("Invalid token format".into()))?,
        );

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()?;

        info!("Brapi client initialized: {}", config.redacted());

        Ok(Self {
            client,
            config,
            last_request: std::sync::Mutex::new(Instant::now() - min_interval),
            min_request_interval: min_interval,
            circuit_breaker: Arc::new(CircuitBreaker::default()),
            stats: std::sync::Mutex::new(AggregateStats::default()),
        })
    }

    /// Get current configuration.
    pub fn config(&self) -> &BrapiConfig {
        &self.config
    }

    /// Get max tickers per request.
    pub fn max_tickers(&self) -> usize {
        self.config.max_tickers_per_request
    }

    /// Get aggregate statistics.
    pub fn stats(&self) -> AggregateStats {
        self.stats.lock().unwrap().clone()
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        *self.stats.lock().unwrap() = AggregateStats::default();
    }

    /// Check if circuit breaker is open.
    pub fn is_circuit_open(&self) -> bool {
        self.circuit_breaker.is_open()
    }

    async fn rate_limit(&self) {
        let mut last = self.last_request.lock().unwrap();
        let elapsed = last.elapsed();
        if elapsed < self.min_request_interval {
            let sleep_time = self.min_request_interval - elapsed;
            drop(last);
            tokio::time::sleep(sleep_time).await;
            let mut last = self.last_request.lock().unwrap();
            *last = Instant::now();
        } else {
            *last = Instant::now();
        }
    }

    /// Make a request with retry logic and circuit breaker.
    async fn request_with_retry<T: for<'de> Deserialize<'de>>(
        &self,
        url: &str,
    ) -> Result<(T, RequestMetrics), BrapiError> {
        // Check circuit breaker
        if self.circuit_breaker.is_open() {
            return Err(BrapiError::CircuitBreakerOpen);
        }

        let mut retries = 0u32;
        let max_retries = self.config.max_retries;
        let endpoint = url.split('?').next().unwrap_or(url).to_string();

        loop {
            self.rate_limit().await;

            let start = Instant::now();
            let response = self.client.get(url).send().await?;
            let duration_ms = start.elapsed().as_millis() as u32;
            let status = response.status().as_u16();

            let mut metrics = RequestMetrics {
                status,
                duration_ms,
                response_bytes: 0,
                retries,
                endpoint: endpoint.clone(),
            };

            match status {
                200 => {
                    let bytes = response.bytes().await?;
                    metrics.response_bytes = bytes.len() as u32;

                    let data: T = serde_json::from_slice(&bytes).map_err(|e| {
                        BrapiError::Parse(format!(
                            "{}: {}",
                            e,
                            String::from_utf8_lossy(&bytes[..bytes.len().min(200)])
                        ))
                    })?;

                    self.circuit_breaker.record_success();
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.record(&metrics, true);
                    }

                    return Ok((data, metrics));
                }
                401 => {
                    self.circuit_breaker.record_failure();
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.record(&metrics, false);
                    }
                    return Err(BrapiError::Unauthorized);
                }
                402 => {
                    self.circuit_breaker.record_failure();
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.record(&metrics, false);
                    }
                    return Err(BrapiError::QuotaExceeded);
                }
                404 => {
                    let body = response.text().await.unwrap_or_default();
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.record(&metrics, false);
                    }
                    return Err(BrapiError::NotFound { message: body });
                }
                429 => {
                    if retries >= max_retries {
                        self.circuit_breaker.record_failure();
                        if let Ok(mut stats) = self.stats.lock() {
                            stats.record(&metrics, false);
                        }
                        return Err(BrapiError::RateLimit { retries });
                    }
                    retries += 1;
                    let backoff = Duration::from_secs(2u64.pow(retries));
                    warn!(
                        "Rate limited (429), backing off for {:?} (retry {}/{})",
                        backoff, retries, max_retries
                    );
                    tokio::time::sleep(backoff).await;
                }
                _ => {
                    let body = response.text().await.unwrap_or_default();
                    if retries >= max_retries {
                        self.circuit_breaker.record_failure();
                        if let Ok(mut stats) = self.stats.lock() {
                            stats.record(&metrics, false);
                        }
                        return Err(BrapiError::Api {
                            status,
                            message: body,
                        });
                    }
                    retries += 1;
                    let backoff = Duration::from_secs(2u64.pow(retries));
                    warn!(
                        "Request failed with {}, retrying in {:?} (retry {}/{})",
                        status, backoff, retries, max_retries
                    );
                    tokio::time::sleep(backoff).await;
                }
            }
        }
    }

    // ========================================================================
    // API Methods
    // ========================================================================

    /// Fetch quotes with historical data for multiple tickers.
    pub async fn fetch_quotes(
        &self,
        tickers: &[&str],
        range: &str,
        interval: &str,
    ) -> Result<(Vec<QuoteResult>, RequestMetrics), BrapiError> {
        let tickers_str = tickers.join(",");
        let url = format!(
            "{}/api/quote/{}?range={}&interval={}",
            self.config.base_url, tickers_str, range, interval
        );

        debug!(
            "Fetching quotes: {} tickers, range={}, interval={}",
            tickers.len(),
            range,
            interval
        );

        let (response, metrics): (QuoteResponse, _) = self.request_with_retry(&url).await?;

        info!(
            "Fetched {} results in {}ms ({} bytes)",
            response.results.len(),
            metrics.duration_ms,
            metrics.response_bytes
        );

        Ok((response.results, metrics))
    }

    /// Fetch a single quote with full metadata (including validRanges/validIntervals).
    pub async fn fetch_quote_with_metadata(
        &self,
        ticker: &str,
    ) -> Result<(QuoteResult, RequestMetrics), BrapiError> {
        let url = format!(
            "{}/api/quote/{}?range=1d&interval=1d",
            self.config.base_url, ticker
        );

        let (response, metrics): (QuoteResponse, _) = self.request_with_retry(&url).await?;

        response
            .results
            .into_iter()
            .next()
            .ok_or_else(|| BrapiError::NotFound {
                message: format!("Ticker {} not found", ticker),
            })
            .map(|r| (r, metrics))
    }

    /// Probe capabilities of a ticker by fetching metadata.
    pub async fn probe_capabilities(&self, ticker: &str) -> Result<TickerCapabilities, BrapiError> {
        let (result, metrics) = self.fetch_quote_with_metadata(ticker).await?;

        let has_intraday = result.has_intraday();
        let max_range = result.max_range().map(|s| s.to_string());

        Ok(TickerCapabilities {
            symbol: result.symbol,
            valid_intervals: result.valid_intervals,
            valid_ranges: result.valid_ranges,
            has_intraday,
            max_range,
            sector: result.sector,
            industry: result.industry,
            probed_at: Utc::now(),
            probe_duration_ms: metrics.duration_ms,
        })
    }

    /// Fetch historical data with specific interval (including intraday).
    pub async fn fetch_historical(
        &self,
        ticker: &str,
        range: &str,
        interval: &str,
    ) -> Result<(Vec<HistoricalBar>, RequestMetrics), BrapiError> {
        let (results, metrics) = self.fetch_quotes(&[ticker], range, interval).await?;

        let bars = results
            .into_iter()
            .next()
            .map(|r| r.historical_data_price)
            .unwrap_or_default();

        Ok((bars, metrics))
    }

    /// Fetch historical bars (convenience wrapper with interval-first signature).
    pub async fn fetch_historical_bars(
        &self,
        ticker: &str,
        interval: &str,
        range: &str,
    ) -> Result<Vec<HistoricalBar>, BrapiError> {
        let (bars, _) = self.fetch_historical(ticker, range, interval).await?;
        Ok(bars)
    }

    /// Fetch fundamental data for multiple tickers.
    /// Uses modules: financialData, defaultKeyStatistics + dividends=true + fundamental=true
    pub async fn fetch_fundamentals(
        &self,
        tickers: &[&str],
    ) -> Result<(Vec<FundamentalQuoteResult>, RequestMetrics), BrapiError> {
        let tickers_str = tickers.join(",");
        let url = format!(
            "{}/api/quote/{}?fundamental=true&dividends=true&modules=financialData,defaultKeyStatistics,summaryProfile",
            self.config.base_url, tickers_str
        );

        debug!("Fetching fundamentals for {} tickers", tickers.len());

        let (response, metrics): (FundamentalResponse, _) = self.request_with_retry(&url).await?;

        info!(
            "Fetched fundamentals for {} tickers in {}ms",
            response.results.len(),
            metrics.duration_ms
        );

        Ok((response.results, metrics))
    }

    /// Fetch fundamental data for a single ticker.
    pub async fn fetch_fundamental(
        &self,
        ticker: &str,
    ) -> Result<(FundamentalQuoteResult, RequestMetrics), BrapiError> {
        let (results, metrics) = self.fetch_fundamentals(&[ticker]).await?;

        results
            .into_iter()
            .next()
            .ok_or_else(|| BrapiError::NotFound {
                message: format!("Ticker {} not found", ticker),
            })
            .map(|r| (r, metrics))
    }

    /// List stocks with pagination.
    pub async fn list_stocks(
        &self,
        limit: usize,
        page: usize,
    ) -> Result<(Vec<StockInfo>, RequestMetrics), BrapiError> {
        let url = format!(
            "{}/api/quote/list?sortBy=volume&sortOrder=desc&limit={}&page={}",
            self.config.base_url, limit, page
        );

        debug!("Listing stocks: limit={}, page={}", limit, page);

        let (response, metrics): (ListResponse, _) = self.request_with_retry(&url).await?;

        info!(
            "Listed {} stocks in {}ms (page {})",
            response.stocks.len(),
            metrics.duration_ms,
            page
        );

        Ok((response.stocks, metrics))
    }

    /// Discover entire universe by paginating through all stocks.
    pub async fn discover_universe(
        &self,
    ) -> Result<(Vec<StockInfo>, Vec<RequestMetrics>), BrapiError> {
        let mut all_stocks = Vec::new();
        let mut all_metrics = Vec::new();
        let page_size = 100;
        let mut page = 1;
        let max_pages = 50; // Safety limit

        info!("Starting universe discovery...");

        loop {
            let (stocks, metrics) = self.list_stocks(page_size, page).await?;
            let count = stocks.len();

            all_metrics.push(metrics);
            all_stocks.extend(stocks);

            if count < page_size || page >= max_pages {
                break;
            }
            page += 1;
        }

        info!(
            "Universe discovery complete: {} stocks across {} pages",
            all_stocks.len(),
            page
        );

        Ok((all_stocks, all_metrics))
    }

    /// Search for a specific ticker in the provider list.
    /// Used for reconciliation when 404 occurs on an ACTIVE ticker.
    pub async fn search_ticker_in_list(&self, ticker: &str) -> Result<bool, BrapiError> {
        let url = format!(
            "{}/api/quote/list?search={}&limit=10",
            self.config.base_url, ticker
        );

        debug!("Searching for ticker in list: {}", ticker);

        let (response, _): (ListResponse, _) = self.request_with_retry(&url).await?;

        // Check if exact match exists
        let found = response.stocks.iter().any(|s| s.stock == ticker);

        info!(
            "Ticker {} {} in provider list",
            ticker,
            if found { "FOUND" } else { "NOT FOUND" }
        );

        Ok(found)
    }

    /// Reconcile a 404 error for a ticker.
    /// Returns true if ticker should remain ACTIVE, false if should be INACTIVE.
    pub async fn reconcile_404(&self, ticker: &str) -> Result<ReconciliationResult, BrapiError> {
        info!("Reconciling 404 for ticker: {}", ticker);

        let still_listed = self.search_ticker_in_list(ticker).await?;

        if still_listed {
            // Ticker is in list but returned 404 - temporary issue
            info!("{}: Listed but 404 - LISTED_BUT_404", ticker);
            Ok(ReconciliationResult::ListedBut404)
        } else {
            // Ticker is not in list anymore - should be INACTIVE
            info!("{}: Not in list - REMOVED_FROM_PROVIDER", ticker);
            Ok(ReconciliationResult::RemovedFromProvider)
        }
    }
}

/// Result of reconciliation after 404.
#[derive(Debug, Clone, PartialEq)]
pub enum ReconciliationResult {
    /// Ticker is still in /api/quote/list but returned 404 on fetch
    ListedBut404,
    /// Ticker was removed from provider list
    RemovedFromProvider,
}

// ============================================================================
// Ticker Capabilities
// ============================================================================

/// Discovered capabilities of a ticker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerCapabilities {
    pub symbol: String,
    pub valid_intervals: Vec<String>,
    pub valid_ranges: Vec<String>,
    pub has_intraday: bool,
    pub max_range: Option<String>,
    pub sector: Option<String>,
    pub industry: Option<String>,
    pub probed_at: DateTime<Utc>,
    pub probe_duration_ms: u32,
}

// ============================================================================
// Test Tickers (no auth required)
// ============================================================================

/// Tickers that can be accessed without authentication (for testing).
pub const TEST_TICKERS: &[&str] = &["PETR4", "MGLU3", "VALE3", "ITUB4"];

/// All supported intervals.
pub const ALL_INTERVALS: &[&str] = &[
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", // Intraday
    "1d", "5d", "1wk", "1mo", "3mo", // Daily+
];

/// All supported ranges.
pub const ALL_RANGES: &[&str] = &[
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
];

/// Intraday intervals only.
pub const INTRADAY_INTERVALS: &[&str] = &["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_requires_token() {
        std::env::remove_var("BRAPI_TOKEN");
        let result = BrapiConfig::from_env();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("BRAPI_TOKEN"));
    }

    #[test]
    fn test_config_with_defaults() {
        std::env::set_var("BRAPI_TOKEN", "test_token");
        let config = BrapiConfig::from_env().unwrap();
        assert_eq!(config.base_url, "https://brapi.dev");
        assert_eq!(config.requests_per_minute, 60);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.timeout_secs, 30);
        std::env::remove_var("BRAPI_TOKEN");
    }

    #[test]
    fn test_circuit_breaker() {
        let cb = CircuitBreaker::new(3, 1);
        assert!(!cb.is_open());

        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open());

        cb.record_failure(); // Third failure
        assert!(cb.is_open());

        cb.record_success();
        assert!(!cb.is_open());
    }

    #[test]
    fn test_quote_result_has_intraday() {
        let result = QuoteResult {
            symbol: "PETR4".into(),
            short_name: None,
            long_name: None,
            currency: None,
            market_cap: None,
            historical_data_price: vec![],
            valid_ranges: vec![],
            valid_intervals: vec!["1m".into(), "5m".into(), "1d".into()],
            used_range: None,
            used_interval: None,
            sector: None,
            industry: None,
            logourl: None,
            extra: std::collections::HashMap::new(),
        };
        assert!(result.has_intraday());

        let result2 = QuoteResult {
            valid_intervals: vec!["1d".into(), "1wk".into()],
            ..result.clone()
        };
        assert!(!result2.has_intraday());
    }
}
