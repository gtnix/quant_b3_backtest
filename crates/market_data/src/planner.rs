//! Stress download planner for full historical data ingestion.
//!
//! Generates an optimized download plan based on ticker capabilities and watermarks.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::info;

use crate::db::{Database, DbError, Watermark};

// ============================================================================
// Download Task
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadTask {
    pub id: usize,
    pub symbol: String,
    pub interval: String,
    pub range: String,
    pub priority: u32,
    pub is_backfill: bool,
    pub estimated_bars: usize,
    pub watermark_ts: Option<DateTime<Utc>>,
    pub status: TaskStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

impl Default for TaskStatus {
    fn default() -> Self {
        TaskStatus::Pending
    }
}

// ============================================================================
// Stress Plan
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressPlan {
    pub version: String,
    pub created_at: DateTime<Utc>,
    pub config: PlanConfig,
    pub summary: PlanSummary,
    pub tasks: Vec<DownloadTask>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanConfig {
    pub intervals_filter: Option<Vec<String>>,
    pub ticker_limit: Option<usize>,
    pub include_backfill: bool,
    pub include_incremental: bool,
}

impl Default for PlanConfig {
    fn default() -> Self {
        Self {
            intervals_filter: None,
            ticker_limit: None,
            include_backfill: true,
            include_incremental: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlanSummary {
    pub total_tickers: usize,
    pub total_tasks: usize,
    pub backfill_tasks: usize,
    pub incremental_tasks: usize,
    pub tasks_by_interval: HashMap<String, usize>,
    pub estimated_requests: usize,
}

// ============================================================================
// Stress Planner
// ============================================================================

pub struct StressPlanner<'a> {
    db: &'a Database,
    config: PlanConfig,
}

impl<'a> StressPlanner<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self {
            db,
            config: PlanConfig::default(),
        }
    }

    pub fn with_config(db: &'a Database, config: PlanConfig) -> Self {
        Self { db, config }
    }

    /// Generate a stress download plan.
    pub async fn generate_plan(&self) -> Result<StressPlan, DbError> {
        info!("Generating stress download plan...");

        // Load capabilities and watermarks
        let capabilities = self.db.get_ticker_capabilities().await?;
        let watermarks = self.db.get_all_watermarks().await?;

        info!(
            "Loaded {} ticker capabilities, {} watermarks",
            capabilities.len(),
            watermarks.len()
        );

        // Build watermark lookup
        let watermark_map: HashMap<(String, String), Watermark> = watermarks
            .into_iter()
            .map(|w| ((w.symbol.clone(), w.interval.clone()), w))
            .collect();

        // Generate tasks
        let mut tasks = Vec::new();
        let mut task_id = 0;

        let caps_to_process: Vec<_> = if let Some(limit) = self.config.ticker_limit {
            capabilities.into_iter().take(limit).collect()
        } else {
            capabilities
        };

        for cap in &caps_to_process {
            let intervals = self.get_intervals_to_process(&cap.valid_intervals);

            for interval in intervals {
                let priority = get_interval_priority(&interval);
                let max_range = get_max_range_for_interval(&interval, &cap.valid_ranges);
                let watermark = watermark_map.get(&(cap.symbol.clone(), interval.clone()));

                let (is_backfill, range) = match watermark {
                    Some(w) if w.last_ts.is_some() => {
                        // Incremental: determine range based on gap
                        if !self.config.include_incremental {
                            continue;
                        }
                        let range = calculate_incremental_range(w.last_ts);
                        (false, range)
                    }
                    _ => {
                        // Backfill: use max range
                        if !self.config.include_backfill {
                            continue;
                        }
                        (true, max_range)
                    }
                };

                let estimated_bars = estimate_bars(&interval, &range);

                tasks.push(DownloadTask {
                    id: task_id,
                    symbol: cap.symbol.clone(),
                    interval: interval.clone(),
                    range,
                    priority,
                    is_backfill,
                    estimated_bars,
                    watermark_ts: watermark.and_then(|w| w.last_ts),
                    status: TaskStatus::Pending,
                });

                task_id += 1;
            }
        }

        // Sort by priority (daily first, then larger intraday intervals)
        tasks.sort_by_key(|t| (t.priority, t.symbol.clone()));

        // Build summary
        let mut tasks_by_interval: HashMap<String, usize> = HashMap::new();
        let mut backfill_count = 0;
        let mut incremental_count = 0;

        for task in &tasks {
            *tasks_by_interval.entry(task.interval.clone()).or_insert(0) += 1;
            if task.is_backfill {
                backfill_count += 1;
            } else {
                incremental_count += 1;
            }
        }

        let summary = PlanSummary {
            total_tickers: caps_to_process.len(),
            total_tasks: tasks.len(),
            backfill_tasks: backfill_count,
            incremental_tasks: incremental_count,
            tasks_by_interval,
            estimated_requests: tasks.len(), // 1 request per task as baseline
        };

        info!(
            "Generated plan with {} tasks ({} backfill, {} incremental)",
            summary.total_tasks, summary.backfill_tasks, summary.incremental_tasks
        );

        Ok(StressPlan {
            version: env!("CARGO_PKG_VERSION").to_string(),
            created_at: Utc::now(),
            config: self.config.clone(),
            summary,
            tasks,
        })
    }

    fn get_intervals_to_process(&self, valid_intervals: &[String]) -> Vec<String> {
        match &self.config.intervals_filter {
            Some(filter) => valid_intervals
                .iter()
                .filter(|i| filter.contains(i))
                .cloned()
                .collect(),
            None => valid_intervals.to_vec(),
        }
    }

    /// Write plan to JSON file.
    pub fn write_plan(plan: &StressPlan, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(plan)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)?;
        info!("Plan written to {}", path.display());
        Ok(())
    }

    /// Load plan from JSON file.
    pub fn load_plan(path: &Path) -> std::io::Result<StressPlan> {
        let content = std::fs::read_to_string(path)?;
        let plan: StressPlan = serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        info!(
            "Loaded plan with {} tasks from {}",
            plan.tasks.len(),
            path.display()
        );
        Ok(plan)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get priority for interval (lower = higher priority).
fn get_interval_priority(interval: &str) -> u32 {
    match interval {
        "1d" => 1,
        "1wk" => 2,
        "1mo" => 3,
        "60m" | "1h" => 10,
        "30m" => 11,
        "15m" => 12,
        "5m" => 13,
        "2m" => 14,
        "1m" => 15,
        _ => 100,
    }
}

/// Get best range for an interval.
fn get_max_range_for_interval(interval: &str, valid_ranges: &[String]) -> String {
    // For daily, prefer max/10y
    // For intraday, prefer 1mo or 5d depending on interval
    let preferred_order = match interval {
        "1d" | "1wk" | "1mo" => vec!["max", "10y", "5y", "2y", "1y"],
        "60m" | "1h" | "30m" => vec!["1y", "6mo", "3mo", "1mo"],
        _ => vec!["1mo", "5d", "1d"], // Short intervals have limited history
    };

    for pref in preferred_order {
        if valid_ranges.iter().any(|r| r == pref) {
            return pref.to_string();
        }
    }

    // Fallback to first available
    valid_ranges
        .first()
        .cloned()
        .unwrap_or_else(|| "1mo".to_string())
}

/// Calculate range needed for incremental update.
fn calculate_incremental_range(last_ts: Option<DateTime<Utc>>) -> String {
    let now = Utc::now();
    match last_ts {
        Some(ts) => {
            let days = (now - ts).num_days();
            match days {
                0..=1 => "1d".to_string(),
                2..=5 => "5d".to_string(),
                6..=30 => "1mo".to_string(),
                31..=90 => "3mo".to_string(),
                91..=180 => "6mo".to_string(),
                _ => "1y".to_string(),
            }
        }
        None => "max".to_string(),
    }
}

/// Estimate number of bars for an interval/range combination.
fn estimate_bars(interval: &str, range: &str) -> usize {
    let days = match range {
        "1d" => 1,
        "5d" => 5,
        "1mo" => 22,
        "3mo" => 66,
        "6mo" => 132,
        "1y" => 252,
        "2y" => 504,
        "5y" => 1260,
        "10y" => 2520,
        "max" => 5000,
        _ => 100,
    };

    let bars_per_day = match interval {
        "1m" => 390, // ~6.5 hours of trading
        "2m" => 195,
        "5m" => 78,
        "15m" => 26,
        "30m" => 13,
        "60m" | "1h" => 7,
        "1d" => 1,
        "1wk" => 1,
        "1mo" => 1,
        _ => 1,
    };

    days * bars_per_day
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_priority() {
        assert!(get_interval_priority("1d") < get_interval_priority("60m"));
        assert!(get_interval_priority("60m") < get_interval_priority("5m"));
        assert!(get_interval_priority("5m") < get_interval_priority("1m"));
    }

    #[test]
    fn test_estimate_bars() {
        assert!(estimate_bars("1d", "1y") == 252);
        assert!(estimate_bars("1m", "1d") == 390);
    }
}
















