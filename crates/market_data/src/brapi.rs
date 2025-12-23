//! Brapi API Client with rate limiting and backoff.

use chrono::{DateTime, NaiveDate, Utc};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{info, warn, debug};

#[derive(Error, Debug)]
pub enum BrapiError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("API error: {status} - {message}")]
    Api { status: u16, message: String },
    #[error("Rate limit exceeded")]
    RateLimit,
    #[error("Quota exceeded (402)")]
    QuotaExceeded,
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Config error: {0}")]
    Config(String),
}

#[derive(Debug, Deserialize)]
pub struct QuoteResponse {
    pub results: Vec<QuoteResult>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuoteResult {
    pub symbol: String,
    pub short_name: Option<String>,
    pub long_name: Option<String>,
    pub currency: Option<String>,
    pub market_cap: Option<f64>,  // API returns as float
    #[serde(default)]
    pub historical_data_price: Vec<HistoricalBar>,
}

impl QuoteResult {
    pub fn market_cap_i64(&self) -> Option<i64> {
        self.market_cap.map(|v| v as i64)
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct HistoricalBar {
    pub date: i64,
    pub open: Option<f64>,
    pub high: Option<f64>,
    pub low: Option<f64>,
    pub close: Option<f64>,
    pub volume: Option<f64>,  // API returns as float sometimes
    pub adjusted_close: Option<f64>,
}

impl HistoricalBar {
    pub fn volume_i64(&self) -> Option<i64> {
        self.volume.map(|v| v as i64)
    }
}

impl HistoricalBar {
    pub fn trading_date(&self) -> Option<NaiveDate> {
        DateTime::from_timestamp(self.date, 0)
            .map(|dt| dt.date_naive())
    }
}

#[derive(Debug, Deserialize)]
pub struct ListResponse {
    pub stocks: Vec<StockInfo>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StockInfo {
    pub stock: String,
    pub name: Option<String>,
    pub close: Option<f64>,
    pub change: Option<f64>,
    pub volume: Option<i64>,
    pub market_cap: Option<i64>,
    pub sector: Option<String>,
}

pub struct BrapiClient {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    max_tickers_per_request: usize,
    last_request: std::sync::Mutex<Instant>,
    min_request_interval: Duration,
}

impl BrapiClient {
    pub fn new() -> Result<Self, BrapiError> {
        let api_key = std::env::var("BRAPI_API_KEY")
            .map_err(|_| BrapiError::Config("BRAPI_API_KEY not set".into()))?;
        
        let max_tickers = std::env::var("BRAPI_MAX_TICKERS_PER_REQUEST")
            .unwrap_or_else(|_| "20".into())
            .parse()
            .unwrap_or(20);

        let requests_per_min: u64 = std::env::var("BRAPI_REQUESTS_PER_MINUTE")
            .unwrap_or_else(|_| "60".into())
            .parse()
            .unwrap_or(60);

        let min_interval = Duration::from_millis(60_000 / requests_per_min);

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key))
                .map_err(|_| BrapiError::Config("Invalid API key format".into()))?,
        );

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(30))
            .build()?;

        Ok(Self {
            client,
            api_key,
            base_url: "https://brapi.dev/api".into(),
            max_tickers_per_request: max_tickers,
            last_request: std::sync::Mutex::new(Instant::now() - min_interval),
            min_request_interval: min_interval,
        })
    }

    pub fn max_tickers(&self) -> usize {
        self.max_tickers_per_request
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

    async fn request_with_retry<T: for<'de> Deserialize<'de>>(
        &self,
        url: &str,
    ) -> Result<(T, RequestMetrics), BrapiError> {
        let mut retries = 0;
        let max_retries = 3;
        
        loop {
            self.rate_limit().await;
            
            let start = Instant::now();
            let response = self.client.get(url).send().await?;
            let duration_ms = start.elapsed().as_millis() as u32;
            let status = response.status().as_u16();
            
            let metrics = RequestMetrics {
                status,
                duration_ms,
                response_bytes: 0,
            };

            match status {
                200 => {
                    let bytes = response.bytes().await?;
                    let metrics = RequestMetrics {
                        status,
                        duration_ms,
                        response_bytes: bytes.len() as u32,
                    };
                    let data: T = serde_json::from_slice(&bytes)
                        .map_err(|e| BrapiError::Parse(e.to_string()))?;
                    return Ok((data, metrics));
                }
                402 => {
                    return Err(BrapiError::QuotaExceeded);
                }
                429 => {
                    if retries >= max_retries {
                        return Err(BrapiError::RateLimit);
                    }
                    retries += 1;
                    let backoff = Duration::from_secs(2u64.pow(retries));
                    warn!("Rate limited, backing off for {:?}", backoff);
                    tokio::time::sleep(backoff).await;
                }
                _ => {
                    let body = response.text().await.unwrap_or_default();
                    if retries >= max_retries {
                        return Err(BrapiError::Api {
                            status,
                            message: body,
                        });
                    }
                    retries += 1;
                    let backoff = Duration::from_secs(2u64.pow(retries));
                    warn!("Request failed with {}, retrying in {:?}", status, backoff);
                    tokio::time::sleep(backoff).await;
                }
            }
        }
    }

    /// Fetch quotes with historical data for multiple tickers.
    pub async fn fetch_quotes(
        &self,
        tickers: &[&str],
        range: &str,
        interval: &str,
    ) -> Result<(Vec<QuoteResult>, RequestMetrics), BrapiError> {
        let tickers_str = tickers.join(",");
        let url = format!(
            "{}/quote/{}?range={}&interval={}",
            self.base_url, tickers_str, range, interval
        );
        
        debug!("Fetching quotes: {} tickers, range={}", tickers.len(), range);
        
        let (response, metrics): (QuoteResponse, _) = self.request_with_retry(&url).await?;
        
        info!(
            "Fetched {} results in {}ms",
            response.results.len(),
            metrics.duration_ms
        );
        
        Ok((response.results, metrics))
    }

    /// List stocks sorted by volume.
    pub async fn list_stocks(
        &self,
        limit: usize,
        page: usize,
    ) -> Result<(Vec<StockInfo>, RequestMetrics), BrapiError> {
        let url = format!(
            "{}/quote/list?sortBy=volume&sortOrder=desc&type=stock&limit={}&page={}",
            self.base_url, limit, page
        );
        
        debug!("Listing stocks: limit={}, page={}", limit, page);
        
        let (response, metrics): (ListResponse, _) = self.request_with_retry(&url).await?;
        
        info!(
            "Listed {} stocks in {}ms",
            response.stocks.len(),
            metrics.duration_ms
        );
        
        Ok((response.stocks, metrics))
    }
}

#[derive(Debug, Clone)]
pub struct RequestMetrics {
    pub status: u16,
    pub duration_ms: u32,
    pub response_bytes: u32,
}

