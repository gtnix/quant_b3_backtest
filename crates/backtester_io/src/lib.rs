//! # Backtester IO
//!
//! Data ingestion and normalization module.
//!
//! Responsibilities:
//! - Read market data from files (CSV)
//! - Normalize timestamps to UTC
//! - Validate OHLC invariants
//! - Generate ordered `MarketEvent` stream
//! - Load benchmark data (IBOV) for relative return calculations

#![warn(missing_docs)]
#![warn(clippy::pedantic)]

pub mod mmap;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub use backtester_core::{AssetId, Bar, MarketEvent};

// =============================================================================
// Benchmark Data (IBOV) for FuzzyFajuto
// =============================================================================

/// Single benchmark (index) bar data.
#[derive(Debug, Clone)]
pub struct BenchmarkBar {
    /// Timestamp (date) in nanoseconds UTC
    pub timestamp: i64,
    /// Opening price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Volume
    pub volume: f64,
    /// Daily return (close[t] / close[t-1] - 1)
    pub daily_return: f64,
}

/// Benchmark data container indexed by date.
/// Provides O(1) lookup of benchmark returns by timestamp.
#[derive(Debug, Clone, Default)]
pub struct BenchmarkData {
    /// Map from date (truncated to day) to benchmark bar
    bars_by_date: HashMap<i64, BenchmarkBar>,
    /// Ordered list of timestamps (for iteration)
    dates: Vec<i64>,
    /// Ticker symbol (e.g., "^BVSP")
    pub ticker: String,
}

impl BenchmarkData {
    /// Create empty benchmark data.
    #[must_use]
    pub fn new(ticker: &str) -> Self {
        Self {
            bars_by_date: HashMap::new(),
            dates: Vec::new(),
            ticker: ticker.to_string(),
        }
    }

    /// Load benchmark data from a CSV file.
    /// Expected format: timestamp,ticker,open,high,low,close,volume
    pub fn from_csv<P: AsRef<Path>>(path: P, ticker: &str) -> Result<Self, DataError> {
        let loader = CsvLoader::new().skip_invalid(true);
        let raw_bars = loader.load(path)?;

        // Filter for the specific benchmark ticker
        let mut filtered: Vec<RawBar> = raw_bars
            .into_iter()
            .filter(|b| b.ticker == ticker)
            .collect();

        // Sort by timestamp
        filtered.sort_by_key(|b| b.timestamp);

        let mut benchmark = Self::new(ticker);
        let mut prev_close: Option<f64> = None;

        for raw in filtered {
            let daily_return = if let Some(prev) = prev_close {
                if prev > 0.0 {
                    (raw.close / prev) - 1.0
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let bar = BenchmarkBar {
                timestamp: raw.timestamp,
                open: raw.open,
                high: raw.high,
                low: raw.low,
                close: raw.close,
                volume: raw.volume,
                daily_return,
            };

            // Truncate timestamp to day for indexing
            let day_ts = Self::truncate_to_day(raw.timestamp);
            benchmark.bars_by_date.insert(day_ts, bar);
            benchmark.dates.push(raw.timestamp);

            prev_close = Some(raw.close);
        }

        Ok(benchmark)
    }

    /// Truncate nanosecond timestamp to start of day.
    fn truncate_to_day(ts: i64) -> i64 {
        // 86400 seconds per day, converted to nanoseconds
        const NANOS_PER_DAY: i64 = 86_400_000_000_000;
        (ts / NANOS_PER_DAY) * NANOS_PER_DAY
    }

    /// Get benchmark bar for a given timestamp.
    #[must_use]
    pub fn get_bar(&self, timestamp: i64) -> Option<&BenchmarkBar> {
        let day_ts = Self::truncate_to_day(timestamp);
        self.bars_by_date.get(&day_ts)
    }

    /// Get benchmark return for a given date.
    #[must_use]
    pub fn get_return(&self, timestamp: i64) -> Option<f64> {
        self.get_bar(timestamp).map(|b| b.daily_return)
    }

    /// Get benchmark close price for a given date.
    #[must_use]
    pub fn get_close(&self, timestamp: i64) -> Option<f64> {
        self.get_bar(timestamp).map(|b| b.close)
    }

    /// Get number of benchmark bars loaded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bars_by_date.len()
    }

    /// Check if benchmark data is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bars_by_date.is_empty()
    }

    /// Get all dates with benchmark data.
    #[must_use]
    pub fn dates(&self) -> &[i64] {
        &self.dates
    }
}

/// Asset with daily returns computed for FuzzyFajuto.
#[derive(Debug, Clone)]
pub struct AssetDailyData {
    /// Asset identifier
    pub asset_id: AssetId,
    /// Ticker symbol
    pub ticker: String,
    /// Timestamp (date)
    pub timestamp: i64,
    /// Bar data
    pub bar: Bar,
    /// Daily return of the asset
    pub daily_return: f64,
    /// Previous day's close (for reference)
    pub prev_close: Option<f64>,
}

/// Normalized data with benchmark for FuzzyFajuto strategy.
#[derive(Debug, Clone)]
pub struct FuzzyDataset {
    /// Asset daily data indexed by (timestamp_day, asset_id)
    pub asset_data: HashMap<(i64, AssetId), AssetDailyData>,
    /// Benchmark data
    pub benchmark: BenchmarkData,
    /// All unique trading dates (sorted)
    pub trading_dates: Vec<i64>,
    /// Ticker to AssetId mapping
    pub ticker_map: HashMap<String, AssetId>,
    /// AssetId to Ticker mapping (reverse)
    pub id_to_ticker: HashMap<AssetId, String>,
    /// Number of assets
    pub num_assets: usize,
}

impl FuzzyDataset {
    /// Load a FuzzyDataset from asset CSV and benchmark CSV.
    pub fn load<P: AsRef<Path>>(
        asset_path: P,
        benchmark_path: P,
        benchmark_ticker: &str,
    ) -> Result<Self, DataError> {
        // Load benchmark
        let benchmark = BenchmarkData::from_csv(&benchmark_path, benchmark_ticker)?;

        // Load asset data
        let loader = CsvLoader::new().skip_invalid(true);
        let raw_bars = loader.load(&asset_path)?;

        // Build normalizer and compute returns
        let mut normalizer = Normalizer::new();
        let mut prev_closes: HashMap<AssetId, f64> = HashMap::new();
        let mut asset_data: HashMap<(i64, AssetId), AssetDailyData> = HashMap::new();
        let mut trading_dates_set: std::collections::HashSet<i64> =
            std::collections::HashSet::new();

        // Sort by timestamp first
        let mut sorted_bars = raw_bars;
        sorted_bars.sort_by_key(|b| (b.timestamp, b.ticker.clone()));

        for raw in sorted_bars {
            let asset_id = normalizer.register_ticker(raw.ticker.clone());
            let day_ts = BenchmarkData::truncate_to_day(raw.timestamp);

            let prev_close = prev_closes.get(&asset_id).copied();
            let daily_return = if let Some(prev) = prev_close {
                if prev > 0.0 {
                    (raw.close / prev) - 1.0
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let bar = Bar {
                timestamp: raw.timestamp,
                open: raw.open,
                high: raw.high,
                low: raw.low,
                close: raw.close,
                volume: raw.volume,
            };

            let data = AssetDailyData {
                asset_id,
                ticker: raw.ticker.clone(),
                timestamp: raw.timestamp,
                bar,
                daily_return,
                prev_close,
            };

            asset_data.insert((day_ts, asset_id), data);
            trading_dates_set.insert(day_ts);
            prev_closes.insert(asset_id, raw.close);
        }

        // Sort trading dates
        let mut trading_dates: Vec<i64> = trading_dates_set.into_iter().collect();
        trading_dates.sort();

        // Build reverse ticker map
        let mut id_to_ticker = HashMap::new();
        for (ticker, &id) in &normalizer.ticker_map {
            id_to_ticker.insert(id, ticker.clone());
        }

        let num_assets = normalizer.asset_count();

        Ok(Self {
            asset_data,
            benchmark,
            trading_dates,
            ticker_map: normalizer.ticker_map,
            id_to_ticker,
            num_assets,
        })
    }

    /// Get asset data for a specific date and asset.
    #[must_use]
    pub fn get_asset(&self, timestamp: i64, asset_id: AssetId) -> Option<&AssetDailyData> {
        let day_ts = BenchmarkData::truncate_to_day(timestamp);
        self.asset_data.get(&(day_ts, asset_id))
    }

    /// Get all assets for a specific date.
    #[must_use]
    pub fn get_assets_for_date(&self, timestamp: i64) -> Vec<&AssetDailyData> {
        let day_ts = BenchmarkData::truncate_to_day(timestamp);
        self.asset_data
            .iter()
            .filter(|((ts, _), _)| *ts == day_ts)
            .map(|(_, data)| data)
            .collect()
    }

    /// Get benchmark return for a date.
    #[must_use]
    pub fn get_benchmark_return(&self, timestamp: i64) -> Option<f64> {
        self.benchmark.get_return(timestamp)
    }

    /// Get ticker for an AssetId.
    #[must_use]
    pub fn get_ticker(&self, asset_id: AssetId) -> Option<&str> {
        self.id_to_ticker.get(&asset_id).map(|s| s.as_str())
    }
}

/// Error types for data loading.
#[derive(Debug)]
pub enum DataError {
    /// File not found or cannot be opened
    IoError(String),
    /// Invalid CSV format
    ParseError(String),
    /// OHLC invariant violation
    ValidationError(String),
    /// Ticker not found in universe
    UnknownTicker(String),
}

impl std::fmt::Display for DataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(msg) => write!(f, "IO Error: {msg}"),
            Self::ParseError(msg) => write!(f, "Parse Error: {msg}"),
            Self::ValidationError(msg) => write!(f, "Validation Error: {msg}"),
            Self::UnknownTicker(msg) => write!(f, "Unknown Ticker: {msg}"),
        }
    }
}

impl std::error::Error for DataError {}

/// Raw bar data before normalization (includes ticker string).
#[derive(Debug, Clone)]
pub struct RawBar {
    /// Ticker symbol
    pub ticker: String,
    /// Timestamp in nanoseconds (UTC)
    pub timestamp: i64,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Volume
    pub volume: f64,
}

impl RawBar {
    /// Validate OHLC invariants (lenient mode).
    /// - all prices non-negative
    /// - volume non-negative
    /// Note: We allow close outside high/low range to support adjusted prices
    pub fn validate(&self) -> Result<(), DataError> {
        if self.open < 0.0 || self.high < 0.0 || self.low < 0.0 || self.close < 0.0 {
            return Err(DataError::ValidationError(format!(
                "Negative price for {}: O={} H={} L={} C={}",
                self.ticker, self.open, self.high, self.low, self.close
            )));
        }
        if self.volume < 0.0 {
            return Err(DataError::ValidationError(format!(
                "Negative volume for {}: {}",
                self.ticker, self.volume
            )));
        }
        // Lenient validation: only check high >= low
        // Note: close can be outside this range for adjusted prices
        if self.high < self.low {
            return Err(DataError::ValidationError(format!(
                "High {} < Low {} for {}",
                self.high, self.low, self.ticker
            )));
        }
        Ok(())
    }
}

/// CSV data loader with OHLC validation.
pub struct CsvLoader {
    skip_invalid: bool,
}

impl CsvLoader {
    /// Create a new CSV loader.
    #[must_use]
    pub fn new() -> Self {
        Self {
            skip_invalid: false,
        }
    }

    /// Configure whether to skip invalid bars (default: false, return error).
    #[must_use]
    pub fn skip_invalid(mut self, skip: bool) -> Self {
        self.skip_invalid = skip;
        self
    }

    /// Load raw bars from a CSV file.
    /// Expected format: timestamp,ticker,open,high,low,close,volume
    /// Timestamp can be ISO 8601 or epoch nanoseconds.
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<Vec<RawBar>, DataError> {
        let file = File::open(path.as_ref())
            .map_err(|e| DataError::IoError(format!("{}: {e}", path.as_ref().display())))?;
        let reader = BufReader::new(file);
        let mut bars = Vec::new();
        let mut line_num = 0;

        for line_result in reader.lines() {
            line_num += 1;
            let line = line_result.map_err(|e| DataError::IoError(e.to_string()))?;

            // Skip header and empty lines
            if line_num == 1 && line.to_lowercase().contains("timestamp") {
                continue;
            }
            if line.trim().is_empty() {
                continue;
            }

            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() < 7 {
                if self.skip_invalid {
                    continue;
                }
                return Err(DataError::ParseError(format!(
                    "Line {line_num}: expected 7 fields, got {}",
                    fields.len()
                )));
            }

            let timestamp = parse_timestamp(fields[0].trim())
                .map_err(|e| DataError::ParseError(format!("Line {line_num}: {e}")))?;

            let bar = RawBar {
                ticker: fields[1].trim().to_string(),
                timestamp,
                open: parse_f64(fields[2].trim(), line_num, "open")?,
                high: parse_f64(fields[3].trim(), line_num, "high")?,
                low: parse_f64(fields[4].trim(), line_num, "low")?,
                close: parse_f64(fields[5].trim(), line_num, "close")?,
                volume: parse_f64(fields[6].trim(), line_num, "volume")?,
            };

            match bar.validate() {
                Ok(()) => bars.push(bar),
                Err(e) => {
                    if !self.skip_invalid {
                        return Err(e);
                    }
                }
            }
        }

        Ok(bars)
    }
}

impl Default for CsvLoader {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_f64(s: &str, line: usize, field: &str) -> Result<f64, DataError> {
    s.parse()
        .map_err(|_| DataError::ParseError(format!("Line {line}: invalid {field} value '{s}'")))
}

fn parse_timestamp(s: &str) -> Result<i64, String> {
    // Try epoch nanoseconds first
    if let Ok(ts) = s.parse::<i64>() {
        return Ok(ts);
    }
    // Try ISO 8601 format
    use chrono::{DateTime, Utc};
    if let Ok(dt) = s.parse::<DateTime<Utc>>() {
        return Ok(dt.timestamp_nanos_opt().unwrap_or(0));
    }
    // Try date-only format (assume end of day 23:59:59 UTC)
    use chrono::NaiveDate;
    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date.and_hms_opt(23, 59, 59).unwrap();
        let utc = DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc);
        return Ok(utc.timestamp_nanos_opt().unwrap_or(0));
    }
    Err(format!("Cannot parse timestamp: '{s}'"))
}

/// Normalizer: maps tickers to AssetIds and orders events chronologically.
pub struct Normalizer {
    /// Map from ticker string to AssetId
    pub ticker_map: HashMap<String, AssetId>,
    next_id: u16,
}

impl Normalizer {
    /// Create a new normalizer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ticker_map: HashMap::new(),
            next_id: 0,
        }
    }

    /// Create normalizer with predefined ticker mapping.
    #[must_use]
    pub fn with_universe(tickers: &[&str]) -> Self {
        let mut normalizer = Self::new();
        for ticker in tickers {
            normalizer.register_ticker((*ticker).to_string());
        }
        normalizer
    }

    /// Register a ticker and get its AssetId.
    pub fn register_ticker(&mut self, ticker: String) -> AssetId {
        if let Some(&id) = self.ticker_map.get(&ticker) {
            return id;
        }
        let id = AssetId::new(self.next_id);
        self.ticker_map.insert(ticker, id);
        self.next_id += 1;
        id
    }

    /// Get AssetId for a ticker (if registered).
    #[must_use]
    pub fn get_asset_id(&self, ticker: &str) -> Option<AssetId> {
        self.ticker_map.get(ticker).copied()
    }

    /// Get ticker for an AssetId.
    #[must_use]
    pub fn get_ticker(&self, asset_id: AssetId) -> Option<&str> {
        self.ticker_map
            .iter()
            .find(|(_, &id)| id == asset_id)
            .map(|(ticker, _)| ticker.as_str())
    }

    /// Get total number of registered assets.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.ticker_map.len()
    }

    /// Normalize raw bars into ordered MarketEvents.
    /// Events are sorted by (timestamp, asset_id) for deterministic ordering.
    pub fn normalize(&mut self, bars: Vec<RawBar>) -> Result<Vec<MarketEvent>, DataError> {
        let mut events: Vec<MarketEvent> = Vec::with_capacity(bars.len());

        for raw in bars {
            let asset_id = self.register_ticker(raw.ticker.clone());
            events.push(MarketEvent {
                asset_id,
                bar: Bar {
                    timestamp: raw.timestamp,
                    open: raw.open,
                    high: raw.high,
                    low: raw.low,
                    close: raw.close,
                    volume: raw.volume,
                },
            });
        }

        // Sort by (timestamp, asset_id) for stable deterministic order
        events.sort_by(|a, b| {
            a.bar
                .timestamp
                .cmp(&b.bar.timestamp)
                .then_with(|| a.asset_id.cmp(&b.asset_id))
        });

        Ok(events)
    }
}

impl Default for Normalizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over ordered market events.
pub struct MarketEventStream {
    events: Vec<MarketEvent>,
    index: usize,
}

impl MarketEventStream {
    /// Create a new event stream from normalized events.
    #[must_use]
    pub fn new(events: Vec<MarketEvent>) -> Self {
        Self { events, index: 0 }
    }

    /// Load from CSV file and create stream.
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<(Self, Normalizer), DataError> {
        let loader = CsvLoader::new();
        let raw_bars = loader.load(path)?;
        let mut normalizer = Normalizer::new();
        let events = normalizer.normalize(raw_bars)?;
        Ok((Self::new(events), normalizer))
    }

    /// Get total number of events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if stream is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Reset stream to beginning.
    pub fn reset(&mut self) {
        self.index = 0;
    }

    /// Get all events as slice.
    #[must_use]
    pub fn as_slice(&self) -> &[MarketEvent] {
        &self.events
    }
}

impl Iterator for MarketEventStream {
    type Item = MarketEvent;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.events.len() {
            let event = self.events[self.index].clone();
            self.index += 1;
            Some(event)
        } else {
            None
        }
    }
}

/// Brapi.dev API loader for Brazilian stocks.
pub struct BrapiLoader {
    base_url: String,
}

impl BrapiLoader {
    /// Create a new Brapi loader.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base_url: "https://brapi.dev/api".to_string(),
        }
    }

    /// Fetch historical data for a ticker.
    /// Range: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    /// Interval: 1d, 1wk, 1mo
    pub fn fetch(
        &self,
        ticker: &str,
        range: &str,
        interval: &str,
    ) -> Result<Vec<RawBar>, DataError> {
        let url = format!(
            "{}/quote/{}?range={}&interval={}",
            self.base_url, ticker, range, interval
        );

        let response = reqwest::blocking::get(&url)
            .map_err(|e| DataError::IoError(format!("HTTP request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(DataError::IoError(format!(
                "API returned status: {}",
                response.status()
            )));
        }

        let json: serde_json::Value = response
            .json()
            .map_err(|e| DataError::ParseError(format!("JSON parse failed: {e}")))?;

        let results = json["results"]
            .as_array()
            .ok_or_else(|| DataError::ParseError("Missing 'results' array".into()))?;

        if results.is_empty() {
            return Err(DataError::ParseError(format!(
                "No data for ticker: {ticker}"
            )));
        }

        let historical = results[0]["historicalDataPrice"]
            .as_array()
            .ok_or_else(|| DataError::ParseError("Missing 'historicalDataPrice'".into()))?;

        let mut bars = Vec::with_capacity(historical.len());

        for item in historical {
            let timestamp_secs = item["date"].as_i64().unwrap_or(0);
            let timestamp = timestamp_secs * 1_000_000_000; // Convert to nanoseconds

            let bar = RawBar {
                ticker: ticker.to_string(),
                timestamp,
                open: item["open"].as_f64().unwrap_or(0.0),
                high: item["high"].as_f64().unwrap_or(0.0),
                low: item["low"].as_f64().unwrap_or(0.0),
                close: item["adjustedClose"]
                    .as_f64()
                    .unwrap_or(item["close"].as_f64().unwrap_or(0.0)),
                volume: item["volume"].as_f64().unwrap_or(0.0),
            };

            if bar.open > 0.0 && bar.high > 0.0 {
                bars.push(bar);
            }
        }

        // Sort by timestamp ascending
        bars.sort_by_key(|b| b.timestamp);

        Ok(bars)
    }

    /// Fetch multiple tickers and combine into single dataset.
    pub fn fetch_universe(&self, tickers: &[&str], range: &str) -> Result<Vec<RawBar>, DataError> {
        let mut all_bars = Vec::new();

        for ticker in tickers {
            match self.fetch(ticker, range, "1d") {
                Ok(bars) => all_bars.extend(bars),
                Err(e) => eprintln!("Warning: Failed to fetch {}: {}", ticker, e),
            }
        }

        // Sort all bars by timestamp
        all_bars.sort_by_key(|b| b.timestamp);

        Ok(all_bars)
    }

    /// Save fetched data to CSV file.
    pub fn save_to_csv(&self, bars: &[RawBar], path: &Path) -> Result<(), DataError> {
        use std::io::Write;
        let mut file = File::create(path)
            .map_err(|e| DataError::IoError(format!("Cannot create file: {e}")))?;

        writeln!(file, "timestamp,ticker,open,high,low,close,volume")
            .map_err(|e| DataError::IoError(e.to_string()))?;

        for bar in bars {
            // Convert nanoseconds to date string
            let secs = bar.timestamp / 1_000_000_000;
            let date = chrono::DateTime::from_timestamp(secs, 0)
                .map(|dt| dt.format("%Y-%m-%d").to_string())
                .unwrap_or_else(|| secs.to_string());

            writeln!(
                file,
                "{},{},{:.2},{:.2},{:.2},{:.2},{:.0}",
                date, bar.ticker, bar.open, bar.high, bar.low, bar.close, bar.volume
            )
            .map_err(|e| DataError::IoError(e.to_string()))?;
        }

        Ok(())
    }
}

impl Default for BrapiLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// B3 historical file loader (COTAHIST format).
pub struct B3HistoricalLoader;

impl B3HistoricalLoader {
    /// Parse a B3 COTAHIST file (fixed-width format).
    /// Returns only spot market (MERC_VISTA) records.
    pub fn parse_cotahist<P: AsRef<Path>>(path: P) -> Result<Vec<RawBar>, DataError> {
        let file = File::open(path.as_ref())
            .map_err(|e| DataError::IoError(format!("{}: {e}", path.as_ref().display())))?;
        let reader = BufReader::new(file);
        let mut bars = Vec::new();

        for line_result in reader.lines() {
            let line = line_result.map_err(|e| DataError::IoError(e.to_string()))?;

            if line.len() < 245 {
                continue;
            }

            let tipo_reg = &line[0..2];
            if tipo_reg != "01" {
                continue; // Skip header/trailer records
            }

            let tipo_merc = &line[24..27].trim();
            if *tipo_merc != "010" {
                continue; // Only spot market
            }

            let data_pregao = &line[2..10];
            let cod_neg = line[12..24].trim();

            // Parse date YYYYMMDD to timestamp
            let timestamp = parse_b3_date(data_pregao)?;

            // Prices in centavos (divide by 100)
            let preabe = parse_b3_price(&line[56..69])?;
            let premax = parse_b3_price(&line[69..82])?;
            let premin = parse_b3_price(&line[82..95])?;
            let preult = parse_b3_price(&line[108..121])?;
            let volume = parse_b3_volume(&line[170..188])?;

            let bar = RawBar {
                ticker: cod_neg.to_string(),
                timestamp,
                open: preabe,
                high: premax,
                low: premin,
                close: preult,
                volume,
            };

            if bar.validate().is_ok() {
                bars.push(bar);
            }
        }

        bars.sort_by_key(|b| b.timestamp);
        Ok(bars)
    }

    /// Download B3 historical file for a year.
    pub fn download_year(year: u32, output_path: &Path) -> Result<(), DataError> {
        let url = format!(
            "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{}.ZIP",
            year
        );

        let response = reqwest::blocking::get(&url)
            .map_err(|e| DataError::IoError(format!("Download failed: {e}")))?;

        if !response.status().is_success() {
            return Err(DataError::IoError(format!(
                "Download failed: {}",
                response.status()
            )));
        }

        let bytes = response
            .bytes()
            .map_err(|e| DataError::IoError(format!("Read failed: {e}")))?;

        std::fs::write(output_path, bytes)
            .map_err(|e| DataError::IoError(format!("Write failed: {e}")))?;

        Ok(())
    }

    /// Extract and parse a downloaded ZIP file.
    pub fn parse_zip<P: AsRef<Path>>(zip_path: P) -> Result<Vec<RawBar>, DataError> {
        let file = File::open(zip_path.as_ref())
            .map_err(|e| DataError::IoError(format!("{}: {e}", zip_path.as_ref().display())))?;

        let mut archive = zip::ZipArchive::new(file)
            .map_err(|e| DataError::IoError(format!("ZIP error: {e}")))?;

        let mut all_bars = Vec::new();

        for i in 0..archive.len() {
            let mut file = archive
                .by_index(i)
                .map_err(|e| DataError::IoError(format!("ZIP entry error: {e}")))?;

            if file.name().ends_with(".TXT") || file.name().ends_with(".txt") {
                let reader = BufReader::new(&mut file);
                for line_result in reader.lines() {
                    let line = line_result.map_err(|e| DataError::IoError(e.to_string()))?;

                    if line.len() < 245 || &line[0..2] != "01" {
                        continue;
                    }

                    let tipo_merc = &line[24..27].trim();
                    if *tipo_merc != "010" {
                        continue;
                    }

                    let data_pregao = &line[2..10];
                    let cod_neg = line[12..24].trim();

                    if let Ok(timestamp) = parse_b3_date(data_pregao) {
                        let preabe = parse_b3_price(&line[56..69]).unwrap_or(0.0);
                        let premax = parse_b3_price(&line[69..82]).unwrap_or(0.0);
                        let premin = parse_b3_price(&line[82..95]).unwrap_or(0.0);
                        let preult = parse_b3_price(&line[108..121]).unwrap_or(0.0);
                        let volume = parse_b3_volume(&line[170..188]).unwrap_or(0.0);

                        if preabe > 0.0 && premax > 0.0 {
                            all_bars.push(RawBar {
                                ticker: cod_neg.to_string(),
                                timestamp,
                                open: preabe,
                                high: premax,
                                low: premin,
                                close: preult,
                                volume,
                            });
                        }
                    }
                }
            }
        }

        all_bars.sort_by_key(|b| b.timestamp);
        Ok(all_bars)
    }

    /// Filter bars for specific tickers.
    #[must_use]
    pub fn filter_tickers(bars: Vec<RawBar>, tickers: &[&str]) -> Vec<RawBar> {
        let ticker_set: std::collections::HashSet<&str> = tickers.iter().copied().collect();
        bars.into_iter()
            .filter(|b| ticker_set.contains(b.ticker.as_str()))
            .collect()
    }
}

fn parse_b3_date(s: &str) -> Result<i64, DataError> {
    use chrono::NaiveDate;
    let date = NaiveDate::parse_from_str(s, "%Y%m%d")
        .map_err(|_| DataError::ParseError(format!("Invalid B3 date: {s}")))?;
    let dt = date.and_hms_opt(23, 59, 59).unwrap();
    let utc = chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(dt, chrono::Utc);
    Ok(utc.timestamp_nanos_opt().unwrap_or(0))
}

fn parse_b3_price(s: &str) -> Result<f64, DataError> {
    let val: i64 = s
        .trim()
        .parse()
        .map_err(|_| DataError::ParseError(format!("Invalid B3 price: '{s}'")))?;
    Ok(val as f64 / 100.0)
}

fn parse_b3_volume(s: &str) -> Result<f64, DataError> {
    let val: i64 = s
        .trim()
        .parse()
        .map_err(|_| DataError::ParseError(format!("Invalid B3 volume: '{s}'")))?;
    Ok(val as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_csv(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file
    }

    #[test]
    fn csv_loader_parses_valid_data() {
        let csv = "timestamp,ticker,open,high,low,close,volume\n\
                   1700000000000000000,PETR4,35.0,36.0,34.0,35.5,1000000\n\
                   1700000000000000000,VALE3,70.0,72.0,69.0,71.0,500000";
        let file = create_test_csv(csv);
        let loader = CsvLoader::new();
        let bars = loader.load(file.path()).unwrap();
        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].ticker, "PETR4");
        assert_eq!(bars[1].ticker, "VALE3");
    }

    #[test]
    fn csv_loader_validates_ohlc() {
        let csv = "timestamp,ticker,open,high,low,close,volume\n\
                   1700000000000000000,BAD,35.0,30.0,34.0,35.5,1000000";
        let file = create_test_csv(csv);
        let loader = CsvLoader::new();
        let result = loader.load(file.path());
        assert!(result.is_err());
    }

    #[test]
    fn normalizer_assigns_stable_ids() {
        let mut normalizer = Normalizer::new();
        let id1 = normalizer.register_ticker("PETR4".to_string());
        let id2 = normalizer.register_ticker("VALE3".to_string());
        let id1_again = normalizer.register_ticker("PETR4".to_string());
        assert_eq!(id1, AssetId::new(0));
        assert_eq!(id2, AssetId::new(1));
        assert_eq!(id1, id1_again);
    }

    #[test]
    fn normalizer_sorts_chronologically() {
        let bars = vec![
            RawBar {
                ticker: "A".into(),
                timestamp: 2000,
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 1.0,
                volume: 1.0,
            },
            RawBar {
                ticker: "B".into(),
                timestamp: 1000,
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 1.0,
                volume: 1.0,
            },
        ];
        let mut normalizer = Normalizer::new();
        let events = normalizer.normalize(bars).unwrap();
        assert_eq!(events[0].bar.timestamp, 1000);
        assert_eq!(events[1].bar.timestamp, 2000);
    }

    #[test]
    fn event_stream_iterates() {
        let events = vec![
            MarketEvent {
                asset_id: AssetId::new(0),
                bar: Bar {
                    timestamp: 1000,
                    open: 1.0,
                    high: 1.0,
                    low: 1.0,
                    close: 1.0,
                    volume: 1.0,
                },
            },
            MarketEvent {
                asset_id: AssetId::new(1),
                bar: Bar {
                    timestamp: 2000,
                    open: 2.0,
                    high: 2.0,
                    low: 2.0,
                    close: 2.0,
                    volume: 2.0,
                },
            },
        ];
        let stream = MarketEventStream::new(events);
        let collected: Vec<_> = stream.collect();
        assert_eq!(collected.len(), 2);
    }
}
