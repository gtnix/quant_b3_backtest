//! Compatibility tests for brapi.dev API.
//!
//! Tests the range × interval matrix to verify which combinations work.
//!
//! Run with: cargo test -p market_data --test compatibility_tests

use std::env;
use std::sync::Mutex;
use std::time::{Duration, Instant};

static LAST_REQUEST: Mutex<Option<Instant>> = Mutex::new(None);

async fn rate_limit() {
    let min_interval = Duration::from_millis(1100);
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

const TEST_TICKER: &str = "PETR4";

const INTERVALS: &[&str] = &[
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", // Intraday
    "1d", "5d", "1wk", "1mo", "3mo", // Daily+
];

const RANGES: &[&str] = &[
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max",
];

#[derive(Debug)]
struct TestResult {
    interval: String,
    range: String,
    success: bool,
    bar_count: usize,
    error: Option<String>,
    duration_ms: u64,
}

// ============================================================================
// Matrix Test
// ============================================================================

#[tokio::test]
async fn test_interval_range_matrix() {
    dotenvy::dotenv().ok();

    if env::var("BRAPI_TOKEN").is_err() {
        eprintln!("BRAPI_TOKEN not set, skipping compatibility test");
        return;
    }

    let client = create_client().expect("Should create client");

    println!("\n=== Interval × Range Compatibility Matrix ===\n");
    println!("Testing {} with all combinations...\n", TEST_TICKER);

    let mut results: Vec<TestResult> = Vec::new();

    // Test a subset of combinations to avoid rate limiting
    let test_intervals = &["5m", "1h", "1d"];
    let test_ranges = &["1d", "5d", "1mo", "1y"];

    for interval in test_intervals {
        for range in test_ranges {
            rate_limit().await;

            let start = Instant::now();
            let result = client.fetch_historical(TEST_TICKER, range, interval).await;
            let duration_ms = start.elapsed().as_millis() as u64;

            let test_result = match result {
                Ok((bars, _)) => TestResult {
                    interval: interval.to_string(),
                    range: range.to_string(),
                    success: !bars.is_empty(),
                    bar_count: bars.len(),
                    error: if bars.is_empty() {
                        Some("No data".into())
                    } else {
                        None
                    },
                    duration_ms,
                },
                Err(e) => {
                    if matches!(e, ClientError::QuotaExceeded) {
                        eprintln!("API quota exceeded, stopping test");
                        break;
                    }
                    TestResult {
                        interval: interval.to_string(),
                        range: range.to_string(),
                        success: false,
                        bar_count: 0,
                        error: Some(e.to_string()),
                        duration_ms,
                    }
                }
            };

            let status = if test_result.success { "✓" } else { "✗" };
            println!(
                "  {} interval={:4} range={:4} → {} bars ({}ms)",
                status, interval, range, test_result.bar_count, test_result.duration_ms
            );

            results.push(test_result);
        }
    }

    // Summary
    let successful = results.iter().filter(|r| r.success).count();
    let total = results.len();

    println!("\n=== Summary ===");
    println!("Successful combinations: {}/{}", successful, total);

    // List failures
    let failures: Vec<_> = results.iter().filter(|r| !r.success).collect();
    if !failures.is_empty() {
        println!("\nFailed combinations:");
        for f in &failures {
            println!(
                "  {}/{}: {}",
                f.interval,
                f.range,
                f.error.as_deref().unwrap_or("unknown")
            );
        }
    }

    // At least some combinations should work
    assert!(
        successful > 0,
        "At least one interval/range combination should work"
    );
}

// ============================================================================
// Intraday Detection Test
// ============================================================================

#[tokio::test]
async fn test_intraday_detection() {
    dotenvy::dotenv().ok();

    if env::var("BRAPI_TOKEN").is_err() {
        eprintln!("BRAPI_TOKEN not set, skipping test");
        return;
    }

    rate_limit().await;

    let client = create_client().expect("Should create client");
    let result = client.fetch_quote_metadata(TEST_TICKER).await;

    match result {
        Ok(quote) => {
            println!("\n{} Capabilities:", TEST_TICKER);
            println!("  Valid intervals: {:?}", quote.valid_intervals);
            println!("  Valid ranges: {:?}", quote.valid_ranges);

            let has_intraday = quote.valid_intervals.iter().any(|i| {
                matches!(
                    i.as_str(),
                    "1m" | "2m" | "5m" | "15m" | "30m" | "60m" | "90m" | "1h"
                )
            });

            println!("  Has intraday: {}", has_intraday);

            // Validate structure
            assert!(
                !quote.valid_intervals.is_empty() || !quote.valid_ranges.is_empty(),
                "Should return at least some valid intervals or ranges"
            );
        }
        Err(e) => {
            if matches!(e, ClientError::QuotaExceeded) {
                eprintln!("API quota exceeded, skipping test");
                return;
            }
            panic!("Failed to fetch metadata: {}", e);
        }
    }
}

// ============================================================================
// Data Validation Test
// ============================================================================

#[tokio::test]
async fn test_ohlcv_data_structure() {
    dotenvy::dotenv().ok();

    if env::var("BRAPI_TOKEN").is_err() {
        eprintln!("BRAPI_TOKEN not set, skipping test");
        return;
    }

    rate_limit().await;

    let client = create_client().expect("Should create client");
    let result = client.fetch_historical(TEST_TICKER, "1mo", "1d").await;

    match result {
        Ok((bars, _)) => {
            assert!(!bars.is_empty(), "Should return bars");

            println!("\nValidating OHLCV data structure...");

            for (i, bar) in bars.iter().enumerate() {
                // Timestamp should be positive
                assert!(bar.date > 0, "Bar {} has invalid timestamp", i);

                // OHLCV should be present (at least close)
                assert!(bar.close.is_some(), "Bar {} missing close price", i);

                // If open/high/low present, validate OHLC relationship
                if let (Some(open), Some(high), Some(low), Some(close)) =
                    (bar.open, bar.high, bar.low, bar.close)
                {
                    assert!(high >= low, "Bar {} high < low ({} < {})", i, high, low);
                    assert!(high >= open, "Bar {} high < open", i);
                    assert!(high >= close, "Bar {} high < close", i);
                    assert!(low <= open, "Bar {} low > open", i);
                    assert!(low <= close, "Bar {} low > close", i);
                }
            }

            // Check chronological order
            if bars.len() > 1 {
                for i in 1..bars.len() {
                    assert!(
                        bars[i].date >= bars[i - 1].date,
                        "Bars not in chronological order at index {}",
                        i
                    );
                }
            }

            println!(
                "  Validated {} bars, all OHLCV constraints satisfied",
                bars.len()
            );
        }
        Err(e) => {
            if matches!(e, ClientError::QuotaExceeded) {
                eprintln!("API quota exceeded, skipping test");
                return;
            }
            panic!("Failed to fetch historical: {}", e);
        }
    }
}

// ============================================================================
// Helper Code
// ============================================================================

struct Client {
    inner: reqwest::Client,
    base_url: String,
}

#[derive(Debug)]
enum ClientError {
    Http(reqwest::Error),
    QuotaExceeded,
    NotFound,
}

impl From<reqwest::Error> for ClientError {
    fn from(e: reqwest::Error) -> Self {
        ClientError::Http(e)
    }
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientError::Http(e) => write!(f, "{}", e),
            ClientError::QuotaExceeded => write!(f, "API quota exceeded"),
            ClientError::NotFound => write!(f, "Not found"),
        }
    }
}

fn create_client() -> Result<Client, String> {
    let token = env::var("BRAPI_TOKEN").map_err(|_| "BRAPI_TOKEN not set".to_string())?;

    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::AUTHORIZATION,
        reqwest::header::HeaderValue::from_str(&format!("Bearer {}", token))
            .map_err(|e| e.to_string())?,
    );

    let inner = reqwest::Client::builder()
        .default_headers(headers)
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| e.to_string())?;

    Ok(Client {
        inner,
        base_url: "https://brapi.dev".into(),
    })
}

#[derive(Debug)]
struct Metrics {
    status: u16,
}

impl Client {
    async fn fetch_historical(
        &self,
        ticker: &str,
        range: &str,
        interval: &str,
    ) -> Result<(Vec<Bar>, Metrics), ClientError> {
        let url = format!(
            "{}/api/quote/{}?range={}&interval={}",
            self.base_url, ticker, range, interval
        );

        let resp = self.inner.get(&url).send().await?;
        let status = resp.status().as_u16();

        if status == 402 {
            return Err(ClientError::QuotaExceeded);
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

    async fn fetch_quote_metadata(&self, ticker: &str) -> Result<QuoteResult, ClientError> {
        let url = format!(
            "{}/api/quote/{}?range=1d&interval=1d",
            self.base_url, ticker
        );

        let resp = self.inner.get(&url).send().await?;
        let status = resp.status().as_u16();

        if status == 402 {
            return Err(ClientError::QuotaExceeded);
        }

        let body: QuoteResponse = resp.json().await?;
        body.results.into_iter().next().ok_or(ClientError::NotFound)
    }
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
    historical_data_price: Vec<Bar>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Bar {
    date: i64,
    open: Option<f64>,
    high: Option<f64>,
    low: Option<f64>,
    close: Option<f64>,
    volume: Option<f64>,
}
