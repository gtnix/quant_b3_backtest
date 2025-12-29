//! Scale tests for brapi.dev API.
//!
//! Tests full universe pagination and probe at scale.
//! These tests are marked #[ignore] by default as they consume API quota.
//!
//! Run with: cargo test -p market_data --test scale_tests -- --ignored

use std::collections::HashMap;
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

// ============================================================================
// Full Universe Pagination Test
// ============================================================================

#[tokio::test]
#[ignore = "Consumes API quota - run manually with --ignored"]
async fn test_full_universe_pagination() {
    dotenvy::dotenv().ok();

    if env::var("BRAPI_TOKEN").is_err() {
        eprintln!("BRAPI_TOKEN not set, skipping scale test");
        return;
    }

    println!("\n=== Full Universe Pagination Test ===\n");

    let client = create_client().expect("Should create client");

    let mut all_stocks = Vec::new();
    let page_size = 100;
    let max_pages = 20; // Limit for safety
    let mut page = 1;

    loop {
        rate_limit().await;

        let result = client.list_stocks(page_size, page).await;

        match result {
            Ok(stocks) => {
                let count = stocks.len();
                println!("Page {}: {} stocks", page, count);

                all_stocks.extend(stocks);

                if count < page_size || page >= max_pages {
                    break;
                }
                page += 1;
            }
            Err(e) => {
                if matches!(e, ClientError::QuotaExceeded) {
                    eprintln!("API quota exceeded at page {}", page);
                    break;
                }
                panic!("List stocks failed: {}", e);
            }
        }
    }

    println!("\n=== Results ===");
    println!("Total stocks discovered: {}", all_stocks.len());
    println!("Pages fetched: {}", page);

    // Count by type (if available)
    let mut by_type: HashMap<String, usize> = HashMap::new();
    for stock in &all_stocks {
        let t = stock.asset_type.as_deref().unwrap_or("unknown");
        *by_type.entry(t.to_string()).or_insert(0) += 1;
    }

    println!("\nBy type:");
    for (t, count) in &by_type {
        println!("  {}: {}", t, count);
    }

    assert!(
        all_stocks.len() > 100,
        "Should discover more than 100 stocks"
    );
}

// ============================================================================
// Sample Probe Test (100 tickers)
// ============================================================================

#[tokio::test]
#[ignore = "Consumes API quota - run manually with --ignored"]
async fn test_sample_probe_100() {
    dotenvy::dotenv().ok();

    if env::var("BRAPI_TOKEN").is_err() {
        eprintln!("BRAPI_TOKEN not set, skipping scale test");
        return;
    }

    println!("\n=== Sample Probe (100 tickers) ===\n");

    let client = create_client().expect("Should create client");

    // First, get universe
    rate_limit().await;
    let universe = client.list_stocks(100, 1).await;

    let stocks = match universe {
        Ok(s) => s,
        Err(e) => {
            if matches!(e, ClientError::QuotaExceeded) {
                eprintln!("API quota exceeded");
                return;
            }
            panic!("Failed to get universe: {}", e);
        }
    };

    println!("Probing {} stocks...\n", stocks.len());

    let mut success_count = 0;
    let mut fail_count = 0;
    let mut intraday_count = 0;
    let mut daily_only_count = 0;
    let mut errors: HashMap<String, usize> = HashMap::new();

    for (i, stock) in stocks.iter().enumerate() {
        rate_limit().await;

        let result = client.probe_ticker(&stock.stock).await;

        match result {
            Ok(caps) => {
                success_count += 1;
                if caps.has_intraday {
                    intraday_count += 1;
                } else {
                    daily_only_count += 1;
                }

                if i < 10 || caps.has_intraday {
                    println!("  {} - OK (intraday: {})", stock.stock, caps.has_intraday);
                }
            }
            Err(e) => {
                fail_count += 1;
                let error_type = match &e {
                    ClientError::QuotaExceeded => {
                        eprintln!("API quota exceeded at ticker {}", i + 1);
                        break;
                    }
                    ClientError::NotFound => "not_found",
                    ClientError::Http(_) => "http_error",
                    ClientError::Parse => "parse_error",
                };
                *errors.entry(error_type.to_string()).or_insert(0) += 1;

                if i < 10 {
                    println!("  {} - FAIL: {}", stock.stock, e);
                }
            }
        }

        if (i + 1) % 25 == 0 {
            println!("  Progress: {}/{}", i + 1, stocks.len());
        }
    }

    println!("\n=== Probe Results ===");
    println!("Successful:    {}", success_count);
    println!("Failed:        {}", fail_count);
    println!(
        "With intraday: {} ({:.1}%)",
        intraday_count,
        if success_count > 0 {
            intraday_count as f64 / success_count as f64 * 100.0
        } else {
            0.0
        }
    );
    println!("Daily only:    {}", daily_only_count);

    if !errors.is_empty() {
        println!("\nError breakdown:");
        for (error_type, count) in &errors {
            println!("  {}: {}", error_type, count);
        }
    }

    // At least some should succeed
    assert!(success_count > 0, "At least some probes should succeed");
}

// ============================================================================
// Rate Limit Handling Test
// ============================================================================

#[tokio::test]
#[ignore = "Consumes API quota - run manually with --ignored"]
async fn test_rate_limit_backoff() {
    dotenvy::dotenv().ok();

    if env::var("BRAPI_TOKEN").is_err() {
        eprintln!("BRAPI_TOKEN not set, skipping scale test");
        return;
    }

    println!("\n=== Rate Limit Backoff Test ===\n");

    let client = create_client().expect("Should create client");

    let test_tickers = [
        "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "WEGE3", "RENT3", "JBSS3", "SUZB3", "GGBR4",
    ];

    println!(
        "Rapid-fire {} requests without internal rate limiting...",
        test_tickers.len()
    );

    let start = Instant::now();
    let mut success = 0;
    let mut rate_limited = 0;

    for ticker in &test_tickers {
        // Intentionally NOT calling rate_limit() to test API's rate limiting
        let result = client.probe_ticker(ticker).await;

        match result {
            Ok(_) => success += 1,
            Err(ClientError::QuotaExceeded) => {
                rate_limited += 1;
                // If we hit quota, stop
                break;
            }
            Err(e) => {
                println!("  {} error: {}", ticker, e);
            }
        }
    }

    let duration = start.elapsed();

    println!("\n=== Results ===");
    println!("Requests:     {}", test_tickers.len());
    println!("Successful:   {}", success);
    println!("Rate limited: {}", rate_limited);
    println!("Duration:     {:?}", duration);

    // Either all succeed or we got rate limited
    assert!(
        success > 0 || rate_limited > 0,
        "Should get either success or rate limit"
    );
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
    Parse,
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
            ClientError::Parse => write!(f, "Parse error"),
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

impl Client {
    async fn list_stocks(&self, limit: usize, page: usize) -> Result<Vec<StockInfo>, ClientError> {
        let url = format!(
            "{}/api/quote/list?sortBy=volume&sortOrder=desc&limit={}&page={}",
            self.base_url, limit, page
        );

        let resp = self.inner.get(&url).send().await?;
        let status = resp.status().as_u16();

        if status == 402 {
            return Err(ClientError::QuotaExceeded);
        }

        let body: ListResponse = resp.json().await.map_err(|_| ClientError::Parse)?;
        Ok(body.stocks)
    }

    async fn probe_ticker(&self, ticker: &str) -> Result<TickerCaps, ClientError> {
        let url = format!(
            "{}/api/quote/{}?range=1d&interval=1d",
            self.base_url, ticker
        );

        let resp = self.inner.get(&url).send().await?;
        let status = resp.status().as_u16();

        if status == 402 {
            return Err(ClientError::QuotaExceeded);
        }
        if status == 404 {
            return Err(ClientError::NotFound);
        }

        let body: QuoteResponse = resp.json().await.map_err(|_| ClientError::Parse)?;
        let result = body
            .results
            .into_iter()
            .next()
            .ok_or(ClientError::NotFound)?;

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
    #[serde(rename = "type")]
    asset_type: Option<String>,
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
}

#[derive(Debug)]
struct TickerCaps {
    symbol: String,
    valid_intervals: Vec<String>,
    valid_ranges: Vec<String>,
    has_intraday: bool,
}













