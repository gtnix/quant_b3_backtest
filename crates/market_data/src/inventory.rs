//! Data Inventory Scanner - analyzes persisted data state.
//!
//! Scans the database to determine coverage status for each (ticker, interval) pair.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::info;

use crate::db::{Database, DbError};

// ============================================================================
// Inventory Status
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InventoryStatus {
    /// No data persisted
    Empty,
    /// Some data but incomplete coverage
    Partial,
    /// Full coverage (within tolerance)
    Complete,
    /// Data exists but is outdated (> threshold days)
    Stale,
    /// Structural inconsistency detected
    Broken,
}

impl std::fmt::Display for InventoryStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InventoryStatus::Empty => write!(f, "EMPTY"),
            InventoryStatus::Partial => write!(f, "PARTIAL"),
            InventoryStatus::Complete => write!(f, "COMPLETE"),
            InventoryStatus::Stale => write!(f, "STALE"),
            InventoryStatus::Broken => write!(f, "BROKEN"),
        }
    }
}

// ============================================================================
// Ticker Inventory
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerInventory {
    pub ticker: String,
    pub interval: String,
    pub first_ts: Option<DateTime<Utc>>,
    pub last_ts: Option<DateTime<Utc>>,
    pub bar_count: i64,
    pub expected_bars: i64,
    pub coverage_pct: f64,
    pub gap_count: i32,
    pub staleness_days: i64,
    pub status: InventoryStatus,
    pub has_capability: bool,
    pub max_range: Option<String>,
}

impl TickerInventory {
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{},{:.1},{},{},{},{},{}",
            self.ticker,
            self.interval,
            self.first_ts.map(|t| t.to_rfc3339()).unwrap_or_default(),
            self.last_ts.map(|t| t.to_rfc3339()).unwrap_or_default(),
            self.bar_count,
            self.expected_bars,
            self.coverage_pct,
            self.gap_count,
            self.staleness_days,
            self.status,
            self.has_capability,
            self.max_range.as_deref().unwrap_or(""),
        )
    }
}

// ============================================================================
// Inventory Summary
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InventorySummary {
    pub total_tickers: usize,
    pub total_pairs: usize,
    pub by_status: HashMap<String, usize>,
    pub by_interval: HashMap<String, IntervalStats>,
    pub total_bars: i64,
    pub scan_duration_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IntervalStats {
    pub total: usize,
    pub empty: usize,
    pub partial: usize,
    pub complete: usize,
    pub stale: usize,
    pub coverage_avg: f64,
}

// ============================================================================
// Data Inventory Scanner
// ============================================================================

pub struct DataInventoryScanner<'a> {
    db: &'a Database,
    stale_threshold_days: i64,
}

impl<'a> DataInventoryScanner<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self {
            db,
            stale_threshold_days: 3, // Data older than 3 days is stale
        }
    }

    pub fn with_stale_threshold(db: &'a Database, days: i64) -> Self {
        Self {
            db,
            stale_threshold_days: days,
        }
    }

    /// Scan the database and build complete inventory.
    pub async fn scan(&self) -> Result<(Vec<TickerInventory>, InventorySummary), DbError> {
        let start = std::time::Instant::now();
        info!("Starting data inventory scan...");

        // Get all ACTIVE tickers
        let active_tickers = self.db.get_active_tickers().await?;
        info!("Found {} ACTIVE tickers", active_tickers.len());

        // Get capabilities for all tickers
        let capabilities = self.db.get_ticker_capabilities().await?;
        let caps_map: HashMap<String, Vec<String>> = capabilities
            .into_iter()
            .map(|c| (c.symbol.clone(), c.valid_intervals))
            .collect();

        // Get all watermarks
        let watermarks = self.db.get_all_watermarks().await?;
        let wm_map: HashMap<(String, String), _> = watermarks
            .into_iter()
            .map(|w| ((w.symbol.clone(), w.interval.clone()), w))
            .collect();

        // Get bar counts from database
        let daily_counts = self.db.get_ohlcv_counts("1d").await?;
        let intraday_counts = self.db.get_ohlcv_intraday_counts().await?;

        let mut inventory = Vec::new();
        let now = Utc::now();

        // Standard intervals to check
        let standard_intervals = vec!["1d", "60m", "15m", "5m", "1m"];

        for ticker in &active_tickers {
            let ticker_caps = caps_map.get(ticker);

            for interval in &standard_intervals {
                let has_capability = ticker_caps
                    .map(|caps| caps.iter().any(|c| c == *interval))
                    .unwrap_or(false);

                // Get watermark for this pair
                let watermark = wm_map.get(&(ticker.clone(), interval.to_string()));

                // Get actual bar count
                let bar_count = if *interval == "1d" {
                    daily_counts.get(ticker).copied().unwrap_or(0)
                } else {
                    intraday_counts
                        .get(&(ticker.clone(), interval.to_string()))
                        .copied()
                        .unwrap_or(0)
                };

                // Calculate expected bars and coverage
                let (expected_bars, coverage_pct) = self.calculate_coverage(
                    *interval,
                    watermark.and_then(|w| w.first_ts),
                    watermark.and_then(|w| w.last_ts),
                    bar_count,
                );

                // Calculate staleness
                let staleness_days = watermark
                    .and_then(|w| w.last_ts)
                    .map(|last| (now - last).num_days())
                    .unwrap_or(999);

                // Determine status
                let status =
                    self.determine_status(bar_count, coverage_pct, staleness_days, has_capability);

                inventory.push(TickerInventory {
                    ticker: ticker.clone(),
                    interval: interval.to_string(),
                    first_ts: watermark.and_then(|w| w.first_ts),
                    last_ts: watermark.and_then(|w| w.last_ts),
                    bar_count,
                    expected_bars,
                    coverage_pct,
                    gap_count: 0, // Would require detailed gap analysis
                    staleness_days,
                    status,
                    has_capability,
                    max_range: self.get_max_range(*interval),
                });
            }
        }

        // Build summary
        let summary = self.build_summary(&inventory, start.elapsed().as_secs_f64());

        info!(
            "Inventory scan complete: {} pairs, {} bars total",
            inventory.len(),
            summary.total_bars
        );

        Ok((inventory, summary))
    }

    fn calculate_coverage(
        &self,
        interval: &str,
        first_ts: Option<DateTime<Utc>>,
        last_ts: Option<DateTime<Utc>>,
        bar_count: i64,
    ) -> (i64, f64) {
        if bar_count == 0 {
            return (0, 0.0);
        }

        let now = Utc::now();

        // Estimate expected bars based on interval
        let expected = match interval {
            "1d" => {
                // Daily: ~252 trading days per year
                if let Some(first) = first_ts {
                    let days = (now - first).num_days();
                    (days as f64 * 0.7) as i64 // ~70% are trading days
                } else {
                    252 // 1 year default
                }
            }
            "60m" | "1h" => {
                // ~7 bars per day, last 6 months
                let days = last_ts.map(|l| (now - l).num_days()).unwrap_or(180);
                (days.min(180) * 7) as i64
            }
            "15m" => {
                // ~26 bars per day, last 3 months
                let days = last_ts.map(|l| (now - l).num_days()).unwrap_or(90);
                (days.min(90) * 26) as i64
            }
            "5m" => {
                // ~78 bars per day, last month
                let days = last_ts.map(|l| (now - l).num_days()).unwrap_or(30);
                (days.min(30) * 78) as i64
            }
            "1m" => {
                // ~390 bars per day, last 5 days
                let days = last_ts.map(|l| (now - l).num_days()).unwrap_or(5);
                (days.min(5) * 390) as i64
            }
            _ => bar_count,
        };

        let expected = expected.max(1);
        let coverage = (bar_count as f64 / expected as f64 * 100.0).min(100.0);

        (expected, coverage)
    }

    fn determine_status(
        &self,
        bar_count: i64,
        coverage_pct: f64,
        staleness_days: i64,
        _has_capability: bool,
    ) -> InventoryStatus {
        if bar_count == 0 {
            return InventoryStatus::Empty;
        }

        if staleness_days > self.stale_threshold_days {
            return InventoryStatus::Stale;
        }

        if coverage_pct >= 90.0 {
            return InventoryStatus::Complete;
        }

        if coverage_pct >= 10.0 {
            return InventoryStatus::Partial;
        }

        InventoryStatus::Partial
    }

    fn get_max_range(&self, interval: &str) -> Option<String> {
        match interval {
            "1d" => Some("max".to_string()),
            "60m" | "1h" => Some("1y".to_string()),
            "15m" => Some("1mo".to_string()),
            "5m" => Some("1mo".to_string()),
            "1m" => Some("5d".to_string()),
            _ => None,
        }
    }

    fn build_summary(&self, inventory: &[TickerInventory], duration: f64) -> InventorySummary {
        let mut by_status: HashMap<String, usize> = HashMap::new();
        let mut by_interval: HashMap<String, IntervalStats> = HashMap::new();
        let mut tickers_seen = std::collections::HashSet::new();
        let mut total_bars = 0i64;

        for item in inventory {
            tickers_seen.insert(item.ticker.clone());
            total_bars += item.bar_count;

            *by_status.entry(item.status.to_string()).or_insert(0) += 1;

            let stats = by_interval.entry(item.interval.clone()).or_default();
            stats.total += 1;
            match item.status {
                InventoryStatus::Empty => stats.empty += 1,
                InventoryStatus::Partial => stats.partial += 1,
                InventoryStatus::Complete => stats.complete += 1,
                InventoryStatus::Stale => stats.stale += 1,
                InventoryStatus::Broken => {}
            }
            stats.coverage_avg += item.coverage_pct;
        }

        // Calculate averages
        for stats in by_interval.values_mut() {
            if stats.total > 0 {
                stats.coverage_avg /= stats.total as f64;
            }
        }

        InventorySummary {
            total_tickers: tickers_seen.len(),
            total_pairs: inventory.len(),
            by_status,
            by_interval,
            total_bars,
            scan_duration_secs: duration,
        }
    }

    /// Write inventory to CSV file.
    pub fn write_csv(inventory: &[TickerInventory], path: &Path) -> std::io::Result<()> {
        let mut content = String::from(
            "ticker,interval,first_ts,last_ts,bar_count,expected_bars,coverage_pct,gap_count,staleness_days,status,has_capability,max_range\n"
        );

        for item in inventory {
            content.push_str(&item.to_csv_row());
            content.push('\n');
        }

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)?;
        info!("Inventory CSV written to {}", path.display());
        Ok(())
    }

    /// Write inventory to JSON file.
    pub fn write_json(
        inventory: &[TickerInventory],
        summary: &InventorySummary,
        path: &Path,
    ) -> std::io::Result<()> {
        let output = serde_json::json!({
            "generated_at": Utc::now().to_rfc3339(),
            "summary": summary,
            "inventory": inventory,
        });

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(&output)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)?;
        info!("Inventory JSON written to {}", path.display());
        Ok(())
    }
}






































