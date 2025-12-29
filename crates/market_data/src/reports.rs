//! Report generation for stress ingestion pipeline.
//!
//! Generates coverage, freshness, and cost estimate reports.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::info;

use crate::db::{Database, DbError};

// ============================================================================
// Coverage Report
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    pub generated_at: DateTime<Utc>,
    pub total_tickers: i64,
    pub daily_coverage: CoverageDetail,
    pub intraday_coverage: HashMap<String, CoverageDetail>,
    pub no_data_tickers: i64,
    pub failed_tickers: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageDetail {
    pub count: i64,
    pub percentage: f64,
}

impl CoverageReport {
    pub async fn generate(db: &Database) -> Result<Self, DbError> {
        let stats = db.get_coverage_stats().await?;

        let daily_percentage = if stats.total_tickers > 0 {
            (stats.daily_coverage as f64 / stats.total_tickers as f64) * 100.0
        } else {
            0.0
        };

        let mut intraday_coverage = HashMap::new();
        for (interval, count) in &stats.intraday_coverage {
            let pct = if stats.total_tickers > 0 {
                (*count as f64 / stats.total_tickers as f64) * 100.0
            } else {
                0.0
            };
            intraday_coverage.insert(
                interval.clone(),
                CoverageDetail {
                    count: *count,
                    percentage: pct,
                },
            );
        }

        let total_with_data = stats.daily_coverage + stats.intraday_coverage.values().sum::<i64>();
        let no_data = stats.total_tickers - total_with_data.min(stats.total_tickers);

        Ok(Self {
            generated_at: Utc::now(),
            total_tickers: stats.total_tickers,
            daily_coverage: CoverageDetail {
                count: stats.daily_coverage,
                percentage: daily_percentage,
            },
            intraday_coverage,
            no_data_tickers: no_data,
            failed_tickers: stats.failed_count,
        })
    }

    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# Coverage Report\n\n");
        md.push_str(&format!(
            "Generated at: {}\n\n",
            self.generated_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));

        md.push_str("## Summary\n\n");
        md.push_str(&format!("| Metric | Value |\n"));
        md.push_str(&format!("|--------|-------|\n"));
        md.push_str(&format!("| Total Tickers | {} |\n", self.total_tickers));
        md.push_str(&format!(
            "| Daily Coverage | {} ({:.1}%) |\n",
            self.daily_coverage.count, self.daily_coverage.percentage
        ));
        md.push_str(&format!("| No Data | {} |\n", self.no_data_tickers));
        md.push_str(&format!("| Failed | {} |\n", self.failed_tickers));

        md.push_str("\n## Intraday Coverage by Interval\n\n");
        md.push_str("| Interval | Count | Percentage |\n");
        md.push_str("|----------|-------|------------|\n");

        let mut intervals: Vec<_> = self.intraday_coverage.iter().collect();
        intervals.sort_by_key(|(k, _)| *k);
        for (interval, detail) in intervals {
            md.push_str(&format!(
                "| {} | {} | {:.1}% |\n",
                interval, detail.count, detail.percentage
            ));
        }

        md
    }

    pub fn write_to_file(&self, path: &Path) -> std::io::Result<()> {
        let md = self.to_markdown();
        std::fs::write(path, md)?;
        info!("Coverage report written to {}", path.display());
        Ok(())
    }
}

// ============================================================================
// Freshness Report
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreshnessEntry {
    pub symbol: String,
    pub interval: String,
    pub last_ts: Option<DateTime<Utc>>,
    pub lag_hours: Option<i64>,
    pub bar_count: i64,
    pub status: FreshnessStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FreshnessStatus {
    Fresh,     // < 24h for daily, < 1h for intraday
    Stale,     // 1-7 days old
    VeryStale, // > 7 days old
    NoData,
}

impl FreshnessEntry {
    pub fn determine_status(interval: &str, lag_hours: Option<i64>) -> FreshnessStatus {
        match lag_hours {
            None => FreshnessStatus::NoData,
            Some(h) if interval == "1d" => {
                if h < 24 {
                    FreshnessStatus::Fresh
                } else if h < 168 {
                    FreshnessStatus::Stale
                } else {
                    FreshnessStatus::VeryStale
                }
            }
            Some(h) => {
                if h < 4 {
                    FreshnessStatus::Fresh
                } else if h < 24 {
                    FreshnessStatus::Stale
                } else {
                    FreshnessStatus::VeryStale
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreshnessReport {
    pub generated_at: DateTime<Utc>,
    pub entries: Vec<FreshnessEntry>,
    pub summary: FreshnessSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FreshnessSummary {
    pub total_entries: usize,
    pub fresh_count: usize,
    pub stale_count: usize,
    pub very_stale_count: usize,
    pub no_data_count: usize,
}

impl FreshnessReport {
    pub async fn generate(db: &Database) -> Result<Self, DbError> {
        let data = db.get_freshness_data().await?;
        let now = Utc::now();

        let mut entries = Vec::new();
        let mut summary = FreshnessSummary::default();

        for (symbol, interval, last_date, bar_count) in data {
            let last_ts = last_date
                .map(|d| {
                    d.and_hms_opt(0, 0, 0)
                        .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
                })
                .flatten();

            let lag_hours = last_ts.map(|ts| (now - ts).num_hours());
            let status = FreshnessEntry::determine_status(&interval, lag_hours);

            match status {
                FreshnessStatus::Fresh => summary.fresh_count += 1,
                FreshnessStatus::Stale => summary.stale_count += 1,
                FreshnessStatus::VeryStale => summary.very_stale_count += 1,
                FreshnessStatus::NoData => summary.no_data_count += 1,
            }

            entries.push(FreshnessEntry {
                symbol,
                interval,
                last_ts,
                lag_hours,
                bar_count,
                status,
            });
        }

        summary.total_entries = entries.len();

        Ok(Self {
            generated_at: now,
            entries,
            summary,
        })
    }

    pub fn to_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str("symbol,interval,last_ts,lag_hours,bar_count,status\n");

        for e in &self.entries {
            let last_ts_str = e.last_ts.map(|t| t.to_rfc3339()).unwrap_or_default();
            let lag_str = e.lag_hours.map(|h| h.to_string()).unwrap_or_default();
            let status_str = format!("{:?}", e.status);

            csv.push_str(&format!(
                "{},{},{},{},{},{}\n",
                e.symbol, e.interval, last_ts_str, lag_str, e.bar_count, status_str
            ));
        }

        csv
    }

    pub fn write_to_file(&self, path: &Path) -> std::io::Result<()> {
        let csv = self.to_csv();
        std::fs::write(path, csv)?;
        info!("Freshness report written to {}", path.display());
        Ok(())
    }
}

// ============================================================================
// Cost Estimate
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    pub generated_at: DateTime<Utc>,
    pub estimated_requests: usize,
    pub cost_per_request: f64,
    pub total_cost: f64,
    pub budget_remaining: Option<i32>,
    pub budget_limit: Option<i32>,
}

impl CostEstimate {
    pub async fn generate(db: &Database, estimated_requests: usize) -> Result<Self, DbError> {
        let (used, limit) = db.get_budget_status().await?;
        let remaining = limit - used;

        // Default cost: $0 for free tier, ~$0.001 for paid
        let cost_per_request = 0.0; // Free tier assumed

        Ok(Self {
            generated_at: Utc::now(),
            estimated_requests,
            cost_per_request,
            total_cost: estimated_requests as f64 * cost_per_request,
            budget_remaining: Some(remaining),
            budget_limit: Some(limit),
        })
    }

    pub fn write_to_file(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)?;
        info!("Cost estimate written to {}", path.display());
        Ok(())
    }
}

// ============================================================================
// Report Generator
// ============================================================================

pub struct ReportGenerator<'a> {
    db: &'a Database,
    output_dir: std::path::PathBuf,
}

impl<'a> ReportGenerator<'a> {
    pub fn new(db: &'a Database, output_dir: impl AsRef<Path>) -> Self {
        Self {
            db,
            output_dir: output_dir.as_ref().to_path_buf(),
        }
    }

    pub async fn generate_all(&self) -> Result<(), DbError> {
        std::fs::create_dir_all(&self.output_dir)
            .map_err(|e| DbError::Connection(e.to_string()))?;

        // Coverage report
        let coverage = CoverageReport::generate(self.db).await?;
        coverage
            .write_to_file(&self.output_dir.join("coverage_report.md"))
            .map_err(|e| DbError::Connection(e.to_string()))?;

        // Freshness report
        let freshness = FreshnessReport::generate(self.db).await?;
        freshness
            .write_to_file(&self.output_dir.join("freshness_report.csv"))
            .map_err(|e| DbError::Connection(e.to_string()))?;

        info!("All reports generated in {}", self.output_dir.display());
        Ok(())
    }
}
















