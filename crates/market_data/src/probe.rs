//! Capability Probe for brapi.dev API.
//!
//! Discovers and tests available capabilities (intervals, ranges) for all tickers.

use crate::brapi::{
    AggregateStats, BrapiClient, BrapiError, StockInfo, TickerCapabilities, ALL_INTERVALS,
    ALL_RANGES,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tracing::{debug, error, info, warn};

// ============================================================================
// Probe Results
// ============================================================================

/// Result of probing a single ticker's capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResult {
    pub symbol: String,
    pub asset_type: Option<String>,
    pub sector: Option<String>,
    pub probed_at: DateTime<Utc>,
    pub status: ProbeStatus,
    pub capabilities: Option<TickerCapabilities>,
    pub tested_combos: Vec<IntervalRangeTest>,
    pub error: Option<String>,
    pub duration_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProbeStatus {
    Success,
    PartialSuccess,
    Failed,
    Skipped,
}

/// Result of testing a specific interval/range combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntervalRangeTest {
    pub interval: String,
    pub range: String,
    pub success: bool,
    pub bar_count: usize,
    pub first_timestamp: Option<i64>,
    pub last_timestamp: Option<i64>,
    pub error: Option<String>,
    pub duration_ms: u32,
}

// ============================================================================
// Probe Manifest
// ============================================================================

/// Manifest of a probe run for reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeManifest {
    pub version: String,
    pub git_commit: Option<String>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub config: ProbeConfig,
    pub stats: ProbeStats,
    pub api_stats: AggregateStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeConfig {
    pub base_url: String,
    pub sample_size: Option<usize>,
    pub full_probe: bool,
    pub test_intervals: Vec<String>,
    pub test_ranges: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProbeStats {
    pub total_tickers_discovered: usize,
    pub tickers_probed: usize,
    pub successful_probes: usize,
    pub failed_probes: usize,
    pub tickers_with_intraday: usize,
    pub tickers_daily_only: usize,
    pub tickers_no_data: usize,
    pub total_duration_secs: f64,
    pub errors_by_type: HashMap<String, usize>,
}

// ============================================================================
// Failure Record
// ============================================================================

/// Detailed failure record for a probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeFailure {
    pub symbol: String,
    pub endpoint: String,
    pub params: HashMap<String, String>,
    pub http_status: Option<u16>,
    pub error_type: String,
    pub error_message: String,
    pub attempt: u32,
    pub timestamp: DateTime<Utc>,
}

// ============================================================================
// Capability Probe
// ============================================================================

/// Capability probe for discovering and testing brapi.dev API capabilities.
pub struct CapabilityProbe {
    client: BrapiClient,
    results: Vec<ProbeResult>,
    failures: Vec<ProbeFailure>,
    universe: Vec<StockInfo>,
    started_at: DateTime<Utc>,
}

impl CapabilityProbe {
    pub fn new(client: BrapiClient) -> Self {
        Self {
            client,
            results: Vec::new(),
            failures: Vec::new(),
            universe: Vec::new(),
            started_at: Utc::now(),
        }
    }

    /// Discover the full universe of tickers.
    pub async fn discover_universe(&mut self) -> Result<usize, BrapiError> {
        info!("Discovering universe...");
        let (stocks, _metrics) = self.client.discover_universe().await?;
        let count = stocks.len();
        self.universe = stocks;
        info!("Discovered {} tickers", count);
        Ok(count)
    }

    /// Probe capabilities for a sample of tickers.
    pub async fn probe_sample(&mut self, sample_size: usize) -> Result<(), BrapiError> {
        if self.universe.is_empty() {
            self.discover_universe().await?;
        }

        let sample: Vec<_> = self.universe.iter().take(sample_size).cloned().collect();
        info!("Probing sample of {} tickers", sample.len());

        for stock in sample {
            self.probe_ticker(&stock).await;
        }

        Ok(())
    }

    /// Probe capabilities for all tickers in the universe.
    pub async fn probe_full(&mut self) -> Result<(), BrapiError> {
        if self.universe.is_empty() {
            self.discover_universe().await?;
        }

        let total = self.universe.len();
        info!("Starting full probe of {} tickers", total);

        let universe = self.universe.clone();
        for (i, stock) in universe.iter().enumerate() {
            if i % 50 == 0 {
                info!("Progress: {}/{} tickers probed", i, total);
            }
            self.probe_ticker(stock).await;

            // Check circuit breaker
            if self.client.is_circuit_open() {
                error!("Circuit breaker open, stopping probe");
                break;
            }
        }

        info!(
            "Full probe complete: {} tickers processed",
            self.results.len()
        );
        Ok(())
    }

    /// Probe a single ticker.
    async fn probe_ticker(&mut self, stock: &StockInfo) {
        let start = std::time::Instant::now();
        let symbol = &stock.stock;

        debug!("Probing {}", symbol);

        match self.client.probe_capabilities(symbol).await {
            Ok(caps) => {
                let has_intraday = caps.has_intraday;

                let result = ProbeResult {
                    symbol: symbol.clone(),
                    asset_type: stock.asset_type.clone(),
                    sector: stock.sector.clone(),
                    probed_at: Utc::now(),
                    status: ProbeStatus::Success,
                    capabilities: Some(caps),
                    tested_combos: Vec::new(), // Basic probe doesn't test combos
                    error: None,
                    duration_ms: start.elapsed().as_millis() as u32,
                };

                debug!("{}: success (intraday: {})", symbol, has_intraday);
                self.results.push(result);
            }
            Err(e) => {
                let error_type = match &e {
                    BrapiError::NotFound { .. } => "not_found",
                    BrapiError::RateLimit { .. } => "rate_limit",
                    BrapiError::QuotaExceeded => "quota_exceeded",
                    BrapiError::Unauthorized => "unauthorized",
                    BrapiError::CircuitBreakerOpen => "circuit_breaker",
                    BrapiError::Timeout { .. } => "timeout",
                    BrapiError::Parse(_) => "parse_error",
                    BrapiError::Http(_) => "http_error",
                    BrapiError::Api { .. } => "api_error",
                    BrapiError::Config(_) => "config_error",
                };

                warn!("{}: {} - {}", symbol, error_type, e);

                let failure = ProbeFailure {
                    symbol: symbol.clone(),
                    endpoint: format!("{}/api/quote/{}", self.client.config().base_url, symbol),
                    params: [
                        ("range".into(), "1d".into()),
                        ("interval".into(), "1d".into()),
                    ]
                    .into_iter()
                    .collect(),
                    http_status: match &e {
                        BrapiError::Api { status, .. } => Some(*status),
                        BrapiError::NotFound { .. } => Some(404),
                        BrapiError::Unauthorized => Some(401),
                        BrapiError::QuotaExceeded => Some(402),
                        BrapiError::RateLimit { .. } => Some(429),
                        _ => None,
                    },
                    error_type: error_type.to_string(),
                    error_message: e.to_string(),
                    attempt: 1,
                    timestamp: Utc::now(),
                };
                self.failures.push(failure);

                let result = ProbeResult {
                    symbol: symbol.clone(),
                    asset_type: stock.asset_type.clone(),
                    sector: stock.sector.clone(),
                    probed_at: Utc::now(),
                    status: ProbeStatus::Failed,
                    capabilities: None,
                    tested_combos: Vec::new(),
                    error: Some(e.to_string()),
                    duration_ms: start.elapsed().as_millis() as u32,
                };
                self.results.push(result);
            }
        }
    }

    /// Test specific interval/range combinations for a ticker.
    pub async fn test_combinations(
        &self,
        ticker: &str,
        intervals: &[&str],
        ranges: &[&str],
    ) -> Vec<IntervalRangeTest> {
        let mut tests = Vec::new();

        for interval in intervals {
            for range in ranges {
                let start = std::time::Instant::now();

                let test = match self.client.fetch_historical(ticker, range, interval).await {
                    Ok((bars, _metrics)) => {
                        let first_ts = bars.first().map(|b| b.date);
                        let last_ts = bars.last().map(|b| b.date);

                        IntervalRangeTest {
                            interval: interval.to_string(),
                            range: range.to_string(),
                            success: !bars.is_empty(),
                            bar_count: bars.len(),
                            first_timestamp: first_ts,
                            last_timestamp: last_ts,
                            error: None,
                            duration_ms: start.elapsed().as_millis() as u32,
                        }
                    }
                    Err(e) => IntervalRangeTest {
                        interval: interval.to_string(),
                        range: range.to_string(),
                        success: false,
                        bar_count: 0,
                        first_timestamp: None,
                        last_timestamp: None,
                        error: Some(e.to_string()),
                        duration_ms: start.elapsed().as_millis() as u32,
                    },
                };

                tests.push(test);
            }
        }

        tests
    }

    /// Generate probe statistics.
    pub fn compute_stats(&self) -> ProbeStats {
        let mut stats = ProbeStats {
            total_tickers_discovered: self.universe.len(),
            tickers_probed: self.results.len(),
            ..Default::default()
        };

        for result in &self.results {
            match result.status {
                ProbeStatus::Success | ProbeStatus::PartialSuccess => {
                    stats.successful_probes += 1;
                    if let Some(caps) = &result.capabilities {
                        if caps.has_intraday {
                            stats.tickers_with_intraday += 1;
                        } else if !caps.valid_intervals.is_empty() {
                            stats.tickers_daily_only += 1;
                        } else {
                            stats.tickers_no_data += 1;
                        }
                    }
                }
                ProbeStatus::Failed => stats.failed_probes += 1,
                ProbeStatus::Skipped => {}
            }
        }

        for failure in &self.failures {
            *stats
                .errors_by_type
                .entry(failure.error_type.clone())
                .or_insert(0) += 1;
        }

        stats.total_duration_secs =
            (Utc::now() - self.started_at).num_milliseconds() as f64 / 1000.0;

        stats
    }

    /// Generate the probe manifest.
    pub fn generate_manifest(&self, full_probe: bool, sample_size: Option<usize>) -> ProbeManifest {
        ProbeManifest {
            version: env!("CARGO_PKG_VERSION").to_string(),
            git_commit: get_git_commit(),
            started_at: self.started_at,
            completed_at: Some(Utc::now()),
            config: ProbeConfig {
                base_url: self.client.config().base_url.clone(),
                sample_size,
                full_probe,
                test_intervals: ALL_INTERVALS.iter().map(|s| s.to_string()).collect(),
                test_ranges: ALL_RANGES.iter().map(|s| s.to_string()).collect(),
            },
            stats: self.compute_stats(),
            api_stats: self.client.stats(),
        }
    }

    /// Write all probe artifacts to output directory.
    pub fn write_artifacts(
        &self,
        output_dir: &Path,
        manifest: &ProbeManifest,
    ) -> std::io::Result<()> {
        fs::create_dir_all(output_dir)?;

        // 1. capabilities_manifest.json
        let manifest_path = output_dir.join("capabilities_manifest.json");
        let manifest_json = serde_json::to_string_pretty(manifest)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(&manifest_path, manifest_json)?;
        info!("Wrote {}", manifest_path.display());

        // 2. universe.csv
        let universe_path = output_dir.join("universe.csv");
        self.write_universe_csv(&universe_path)?;
        info!("Wrote {}", universe_path.display());

        // 3. capability_matrix.csv
        let matrix_path = output_dir.join("capability_matrix.csv");
        self.write_capability_matrix(&matrix_path)?;
        info!("Wrote {}", matrix_path.display());

        // 4. failures.json
        let failures_path = output_dir.join("failures.json");
        let failures_json = serde_json::to_string_pretty(&self.failures)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(&failures_path, failures_json)?;
        info!("Wrote {}", failures_path.display());

        // 5. sample_payloads/ (create directory with sample)
        let samples_dir = output_dir.join("sample_payloads");
        fs::create_dir_all(&samples_dir)?;
        self.write_sample_payloads(&samples_dir)?;
        info!("Wrote sample payloads to {}", samples_dir.display());

        Ok(())
    }

    fn write_universe_csv(&self, path: &Path) -> std::io::Result<()> {
        let mut content = String::from("symbol,name,sector,type,market_cap,volume,close\n");
        for stock in &self.universe {
            content.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                stock.stock,
                stock.name.as_deref().unwrap_or("").replace(',', ";"),
                stock.sector.as_deref().unwrap_or(""),
                stock.asset_type.as_deref().unwrap_or(""),
                stock.market_cap.unwrap_or(0),
                stock.volume.unwrap_or(0),
                stock.close.unwrap_or(0.0),
            ));
        }
        fs::write(path, content)
    }

    fn write_capability_matrix(&self, path: &Path) -> std::io::Result<()> {
        let mut content = String::from(
            "symbol,type,sector,has_intraday,max_range,valid_intervals,valid_ranges,status,error\n",
        );

        for result in &self.results {
            let (has_intraday, max_range, intervals, ranges) =
                if let Some(caps) = &result.capabilities {
                    (
                        caps.has_intraday.to_string(),
                        caps.max_range.as_deref().unwrap_or(""),
                        caps.valid_intervals.join(";"),
                        caps.valid_ranges.join(";"),
                    )
                } else {
                    ("".into(), "".into(), "".into(), "".into())
                };

            content.push_str(&format!(
                "{},{},{},{},{},{},{},{:?},{}\n",
                result.symbol,
                result.asset_type.as_deref().unwrap_or(""),
                result.sector.as_deref().unwrap_or(""),
                has_intraday,
                max_range,
                intervals,
                ranges,
                result.status,
                result.error.as_deref().unwrap_or("").replace(',', ";"),
            ));
        }

        fs::write(path, content)
    }

    fn write_sample_payloads(&self, dir: &Path) -> std::io::Result<()> {
        // Write a sample of successful capabilities
        let samples: Vec<_> = self
            .results
            .iter()
            .filter(|r| r.capabilities.is_some())
            .take(5)
            .collect();

        for (i, result) in samples.iter().enumerate() {
            if let Some(caps) = &result.capabilities {
                let sample_path = dir.join(format!("sample_{}.json", i + 1));
                let sample_json = serde_json::to_string_pretty(caps)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                fs::write(&sample_path, sample_json)?;
            }
        }

        Ok(())
    }

    /// Get probe results.
    pub fn results(&self) -> &[ProbeResult] {
        &self.results
    }

    /// Get successful results with capabilities.
    pub fn get_successful_results(&self) -> Vec<&ProbeResult> {
        self.results
            .iter()
            .filter(|r| r.status == ProbeStatus::Success && r.capabilities.is_some())
            .collect()
    }

    /// Get failures.
    pub fn failures(&self) -> &[ProbeFailure] {
        &self.failures
    }

    /// Get discovered universe.
    pub fn universe(&self) -> &[StockInfo] {
        &self.universe
    }
}

fn get_git_commit() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_stats_default() {
        let stats = ProbeStats::default();
        assert_eq!(stats.total_tickers_discovered, 0);
        assert_eq!(stats.successful_probes, 0);
    }
}
