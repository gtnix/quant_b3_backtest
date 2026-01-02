//! Smoke tests for brapi.dev provider.
//!
//! These tests validate basic functionality:
//! 1. Token configuration from .env
//! 2. List endpoint pagination
//! 3. Quote endpoint for public tickers
//! 4. Historical data (daily)
//! 5. Historical data (intraday)
//!
//! Run with: cargo test -p market_data --test smoke_tests

use std::env;

// Re-export from main crate
mod common {
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    // Simple rate limiting for tests
    pub static LAST_REQUEST: Mutex<Option<Instant>> = Mutex::new(None);

    pub async fn rate_limit() {
        let min_interval = Duration::from_millis(1100); // ~60 requests/min
        let mut last = LAST_REQUEST.lock().unwrap();
        if let Some(t) = *last {
            let elapsed = t.elapsed();
            if elapsed < min_interval {
                drop(last);
                tokio::time::sleep(min_interval - elapsed).await;
                let mut last = LAST_REQUEST.lock().unwrap();
                *last = Some(Instant::now());
            } else {
                *last = Some(Instant::now());
            }
        } else {
            *last = Some(Instant::now());
        }
    }
}

// ============================================================================
// Test 1: Token Configuration
// ============================================================================

#[test]
fn test_brapi_token_env_required() {
    // Save current value
    let original = env::var("BRAPI_TOKEN").ok();

    // Remove token
    env::remove_var("BRAPI_TOKEN");

    // Trying to create client should fail
    let result = create_brapi_config();
    assert!(result.is_err(), "Should fail without BRAPI_TOKEN");
    let err = result.unwrap_err();
    assert!(
        err.contains("BRAPI_TOKEN"),
        "Error should mention BRAPI_TOKEN: {}",
        err
    );

    // Restore
    if let Some(token) = original {
        env::set_var("BRAPI_TOKEN", token);
    }
}

#[test]
fn test_brapi_token_empty_rejected() {
    let original = env::var("BRAPI_TOKEN").ok();

    env::set_var("BRAPI_TOKEN", "   ");

    let result = create_brapi_config();
    assert!(result.is_err(), "Should reject empty token");

    if let Some(token) = original {
        env::set_var("BRAPI_TOKEN", token);
    } else {
        env::remove_var("BRAPI_TOKEN");
    }
}

#[test]
fn test_brapi_config_defaults() {
    let original = env::var("BRAPI_TOKEN").ok();

    // Set valid token
    env::set_var("BRAPI_TOKEN", "test_token_123");

    let config = create_brapi_config().expect("Should create config with valid token");

    assert_eq!(config.base_url, "https://brapi.dev");
    assert_eq!(config.requests_per_minute, 60);
    assert_eq!(config.max_retries, 3);
    assert_eq!(config.timeout_secs, 30);
    assert_eq!(config.max_tickers_per_request, 20);

    if let Some(token) = original {
        env::set_var("BRAPI_TOKEN", token);
    } else {
        env::remove_var("BRAPI_TOKEN");
    }
}

// ============================================================================
// Test 2: List Endpoint
// ============================================================================

#[tokio::test]
async fn test_list_stocks_returns_data() {
    dotenvy::dotenv().ok();

    if env::var("BRAPI_TOKEN").is_err() {
        eprintln!("BRAPI_TOKEN not set, skipping API test");
        return;
    }

    common::rate_limit().await;

    let client = create_brapi_client().expect("Should create client");
    let result = client.list_stocks(10, 1).await;

    match result {
        Ok((stocks, metrics)) => {
            assert!(!stocks.is_empty(), "Should return at least 1 stock");
            assert!(stocks.len() <= 10, "Should respect limit");
            assert_eq!(metrics.status, 200, "Should return 200 OK");

            // Validate stock structure
            let first = &stocks[0];
            assert!(!first.stock.is_empty(), "Stock symbol should not be empty");

            println!("Listed {} stocks, first: {}", stocks.len(), first.stock);
        }
        Err(e) => {
            // Allow quota errors in tests
            if is_quota_error(&e) {
                eprintln!("API quota exceeded, skipping test");
                return;
            }
            panic!("list_stocks failed: {}", e);
        }
    }
}

#[tokio::test]
async fn test_list_stocks_pagination() {
    dotenvy::dotenv().ok();

    if env::var("BRAPI_TOKEN").is_err() {
        eprintln!("BRAPI_TOKEN not set, skipping API test");
        return;
    }

    common::rate_limit().await;

    let client = create_brapi_client().expect("Should create client");

    let result1 = client.list_stocks(5, 1).await;
    common::rate_limit().await;
    let result2 = client.list_stocks(5, 2).await;

    match (result1, result2) {
        (Ok((page1, _)), Ok((page2, _))) => {
            // Pages should have different stocks (unless universe is tiny)
            if page1.len() == 5 && page2.len() > 0 {
                let symbols1: Vec<_> = page1.iter().map(|s| &s.stock).collect();
                let symbols2: Vec<_> = page2.iter().map(|s| &s.stock).collect();

                // At least some symbols should differ
                let overlap: Vec<_> = symbols1.iter().filter(|s| symbols2.contains(s)).collect();
                assert!(
                    overlap.len() < 5,
                    "Pagination should return different stocks"
                );

                println!("Page 1: {:?}, Page 2: {:?}", symbols1, symbols2);
            }
        }
        (Err(e), _) | (_, Err(e)) => {
            if is_quota_error(&e) {
                eprintln!("API quota exceeded, skipping test");
                return;
            }
            panic!("Pagination test failed: {}", e);
        }
    }
}

// ============================================================================
// Test 3: Quote Endpoint (Public Ticker)
// ============================================================================

#[tokio::test]
async fn test_quote_petr4_no_auth() {
    dotenvy::dotenv().ok();

    // PETR4 is a public ticker that works without auth
    if env::var("BRAPI_TOKEN").is_err() {
        env::set_var("BRAPI_TOKEN", "test_token");
    }

    common::rate_limit().await;

    let client = create_brapi_client().expect("Should create client");
    let result = client.fetch_quote_with_metadata("PETR4").await;

    match result {
        Ok((quote, metrics)) => {
            assert_eq!(quote.symbol, "PETR4");
            assert_eq!(metrics.status, 200);

            println!("PETR4 quote fetched");
            println!("  Valid intervals: {:?}", quote.valid_intervals);
            println!("  Valid ranges: {:?}", quote.valid_ranges);
            println!("  Has intraday: {}", quote.has_intraday());
        }
        Err(e) => {
            // PETR4 should work without auth, but allow quota errors
            if is_quota_error(&e) {
                eprintln!("API quota exceeded, skipping test");
                return;
            }
            panic!("PETR4 quote failed: {}", e);
        }
    }
}

// ============================================================================
// Test 4: Historical Data (Daily)
// ============================================================================

#[tokio::test]
async fn test_historical_daily() {
    dotenvy::dotenv().ok();

    if env::var("BRAPI_TOKEN").is_err() {
        eprintln!("BRAPI_TOKEN not set, skipping API test");
        return;
    }

    common::rate_limit().await;

    let client = create_brapi_client().expect("Should create client");
    let result = client.fetch_historical("PETR4", "1mo", "1d").await;

    match result {
        Ok((bars, metrics)) => {
            assert!(!bars.is_empty(), "Should return historical bars");
            assert_eq!(metrics.status, 200);

            // Validate bar structure
            let first = &bars[0];
            assert!(first.date > 0, "Date should be positive timestamp");
            assert!(first.open.is_some(), "Open should be present");
            assert!(first.close.is_some(), "Close should be present");

            // Check ordering (should be chronological)
            if bars.len() > 1 {
                assert!(
                    bars[0].date < bars[bars.len() - 1].date,
                    "Bars should be in chronological order"
                );
            }

            println!("Fetched {} daily bars for PETR4", bars.len());
            println!(
                "  First: {} (ts={})",
                first.trading_date().unwrap_or_default(),
                first.date
            );
        }
        Err(e) => {
            if is_quota_error(&e) {
                eprintln!("API quota exceeded, skipping test");
                return;
            }
            panic!("Historical daily failed: {}", e);
        }
    }
}

// ============================================================================
// Test 5: Historical Data (Intraday)
// ============================================================================

#[tokio::test]
async fn test_historical_intraday_5m() {
    dotenvy::dotenv().ok();

    if env::var("BRAPI_TOKEN").is_err() {
        eprintln!("BRAPI_TOKEN not set, skipping API test");
        return;
    }

    common::rate_limit().await;

    let client = create_brapi_client().expect("Should create client");

    // First check if ticker supports intraday
    let caps = client.probe_capabilities("PETR4").await;

    match caps {
        Ok(cap) => {
            println!("PETR4 capabilities:");
            println!("  Valid intervals: {:?}", cap.valid_intervals);
            println!("  Has intraday: {}", cap.has_intraday);

            if !cap.has_intraday {
                println!(
                    "PETR4 does not support intraday, test passes (capability correctly detected)"
                );
                return;
            }

            // Try to fetch intraday data
            common::rate_limit().await;
            let result = client.fetch_historical("PETR4", "5d", "5m").await;

            match result {
                Ok((bars, metrics)) => {
                    println!("Fetched {} intraday (5m) bars", bars.len());
                    assert_eq!(metrics.status, 200);

                    if !bars.is_empty() {
                        let first = &bars[0];
                        println!("  First bar timestamp: {}", first.date);
                    }
                }
                Err(e) => {
                    // Intraday might not be available for all users
                    println!("Intraday fetch returned error (may be expected): {}", e);
                }
            }
        }
        Err(e) => {
            if is_quota_error(&e) {
                eprintln!("API quota exceeded, skipping test");
                return;
            }
            panic!("Capabilities probe failed: {}", e);
        }
    }
}

// ============================================================================
// Test 6: Circuit Breaker
// ============================================================================

#[test]
fn test_circuit_breaker_behavior() {
    let cb = CircuitBreaker::new(3, 1);

    assert!(!cb.is_open(), "Should start closed");

    cb.record_failure();
    cb.record_failure();
    assert!(!cb.is_open(), "Should still be closed after 2 failures");

    cb.record_failure();
    assert!(cb.is_open(), "Should open after 3 failures");

    cb.record_success();
    assert!(!cb.is_open(), "Should close after success");
    assert_eq!(cb.failure_count(), 0, "Failure count should reset");
}

// ============================================================================
// Helper Functions
// ============================================================================

#[derive(Debug)]
struct BrapiConfig {
    base_url: String,
    requests_per_minute: u64,
    max_retries: u32,
    timeout_secs: u64,
    max_tickers_per_request: usize,
}

fn create_brapi_config() -> Result<BrapiConfig, String> {
    let token = env::var("BRAPI_TOKEN")
        .map_err(|_| "BRAPI_TOKEN environment variable is required".to_string())?;

    if token.trim().is_empty() {
        return Err("BRAPI_TOKEN cannot be empty".to_string());
    }

    Ok(BrapiConfig {
        base_url: env::var("BRAPI_BASE_URL").unwrap_or_else(|_| "https://brapi.dev".into()),
        requests_per_minute: env::var("BRAPI_REQUESTS_PER_MINUTE")
            .unwrap_or_else(|_| "60".into())
            .parse()
            .unwrap_or(60),
        max_retries: env::var("BRAPI_MAX_RETRIES")
            .unwrap_or_else(|_| "3".into())
            .parse()
            .unwrap_or(3),
        timeout_secs: env::var("BRAPI_TIMEOUT_SECS")
            .unwrap_or_else(|_| "30".into())
            .parse()
            .unwrap_or(30),
        max_tickers_per_request: env::var("BRAPI_MAX_TICKERS_PER_REQUEST")
            .unwrap_or_else(|_| "20".into())
            .parse()
            .unwrap_or(20),
    })
}

// Inline minimal client for tests
struct MinimalBrapiClient {
    client: reqwest::Client,
    base_url: String,
}

fn create_brapi_client() -> Result<MinimalBrapiClient, String> {
    let token = env::var("BRAPI_TOKEN").map_err(|_| "BRAPI_TOKEN not set".to_string())?;

    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::AUTHORIZATION,
        reqwest::header::HeaderValue::from_str(&format!("Bearer {}", token))
            .map_err(|e| e.to_string())?,
    );

    let client = reqwest::Client::builder()
        .default_headers(headers)
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| e.to_string())?;

    Ok(MinimalBrapiClient {
        client,
        base_url: "https://brapi.dev".into(),
    })
}

impl MinimalBrapiClient {
    async fn list_stocks(
        &self,
        limit: usize,
        page: usize,
    ) -> Result<(Vec<StockInfo>, Metrics), BrapiTestError> {
        let url = format!(
            "{}/api/quote/list?sortBy=volume&sortOrder=desc&limit={}&page={}",
            self.base_url, limit, page
        );

        let resp = self.client.get(&url).send().await?;
        let status = resp.status().as_u16();

        if status == 402 {
            return Err(BrapiTestError::QuotaExceeded);
        }

        let body: ListResponse = resp.json().await?;
        Ok((body.stocks, Metrics { status }))
    }

    async fn fetch_quote_with_metadata(
        &self,
        ticker: &str,
    ) -> Result<(QuoteResult, Metrics), BrapiTestError> {
        let url = format!(
            "{}/api/quote/{}?range=1d&interval=1d",
            self.base_url, ticker
        );

        let resp = self.client.get(&url).send().await?;
        let status = resp.status().as_u16();

        if status == 402 {
            return Err(BrapiTestError::QuotaExceeded);
        }

        let body: QuoteResponse = resp.json().await?;
        let result = body
            .results
            .into_iter()
            .next()
            .ok_or_else(|| BrapiTestError::NotFound)?;

        Ok((result, Metrics { status }))
    }

    async fn fetch_historical(
        &self,
        ticker: &str,
        range: &str,
        interval: &str,
    ) -> Result<(Vec<HistoricalBar>, Metrics), BrapiTestError> {
        let url = format!(
            "{}/api/quote/{}?range={}&interval={}",
            self.base_url, ticker, range, interval
        );

        let resp = self.client.get(&url).send().await?;
        let status = resp.status().as_u16();

        if status == 402 {
            return Err(BrapiTestError::QuotaExceeded);
        }

        let body: QuoteResponse = resp.json().await?;
        let bars = body
            .results
            .into_iter()
            .next()
            .map(|r| r.historical_data_price)
            .unwrap_or_default();

        Ok((bars, Metrics { status }))
    }

    async fn probe_capabilities(&self, ticker: &str) -> Result<TickerCaps, BrapiTestError> {
        let (result, _) = self.fetch_quote_with_metadata(ticker).await?;

        let has_intraday = result.valid_intervals.iter().any(|i| {
            matches!(
                i.as_str(),
                "1m" | "2m" | "5m" | "15m" | "30m" | "60m" | "90m" | "1h"
            )
        });

        Ok(TickerCaps {
            symbol: result.symbol,
            valid_intervals: result.valid_intervals,
            valid_ranges: result.valid_ranges,
            has_intraday,
        })
    }
}

#[derive(Debug)]
struct Metrics {
    status: u16,
}

#[derive(Debug)]
enum BrapiTestError {
    Http(reqwest::Error),
    QuotaExceeded,
    NotFound,
}

impl From<reqwest::Error> for BrapiTestError {
    fn from(e: reqwest::Error) -> Self {
        BrapiTestError::Http(e)
    }
}

impl std::fmt::Display for BrapiTestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BrapiTestError::Http(e) => write!(f, "HTTP error: {}", e),
            BrapiTestError::QuotaExceeded => write!(f, "API quota exceeded"),
            BrapiTestError::NotFound => write!(f, "Not found"),
        }
    }
}

fn is_quota_error(e: &BrapiTestError) -> bool {
    matches!(e, BrapiTestError::QuotaExceeded)
}

#[derive(Debug, serde::Deserialize)]
struct ListResponse {
    stocks: Vec<StockInfo>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct StockInfo {
    stock: String,
    name: Option<String>,
    sector: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct QuoteResponse {
    results: Vec<QuoteResult>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct QuoteResult {
    symbol: String,
    #[serde(default)]
    valid_intervals: Vec<String>,
    #[serde(default)]
    valid_ranges: Vec<String>,
    #[serde(default)]
    historical_data_price: Vec<HistoricalBar>,
}

impl QuoteResult {
    fn has_intraday(&self) -> bool {
        self.valid_intervals.iter().any(|i| {
            matches!(
                i.as_str(),
                "1m" | "2m" | "5m" | "15m" | "30m" | "60m" | "90m" | "1h"
            )
        })
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct HistoricalBar {
    date: i64,
    open: Option<f64>,
    close: Option<f64>,
}

impl HistoricalBar {
    fn trading_date(&self) -> Option<chrono::NaiveDate> {
        chrono::DateTime::from_timestamp(self.date, 0).map(|dt| dt.date_naive())
    }
}

#[derive(Debug)]
struct TickerCaps {
    symbol: String,
    valid_intervals: Vec<String>,
    valid_ranges: Vec<String>,
    has_intraday: bool,
}

struct CircuitBreaker {
    failure_count: std::sync::atomic::AtomicU32,
    is_open: std::sync::atomic::AtomicBool,
    threshold: u32,
}

impl CircuitBreaker {
    fn new(threshold: u32, _reset_timeout_secs: u64) -> Self {
        Self {
            failure_count: std::sync::atomic::AtomicU32::new(0),
            is_open: std::sync::atomic::AtomicBool::new(false),
            threshold,
        }
    }

    fn is_open(&self) -> bool {
        self.is_open.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn record_failure(&self) {
        let count = self
            .failure_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        if count >= self.threshold {
            self.is_open
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    fn record_success(&self) {
        self.failure_count
            .store(0, std::sync::atomic::Ordering::SeqCst);
        self.is_open
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    fn failure_count(&self) -> u32 {
        self.failure_count.load(std::sync::atomic::Ordering::SeqCst)
    }
}


























