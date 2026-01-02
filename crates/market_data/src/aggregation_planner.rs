//! Aggregation Planner - generates deterministic download plans.
//!
//! Consumes inventory + capabilities and produces minimal task list.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::info;

use crate::coverage_model::{get_backfill_range, parse_range_to_days, CoverageRule};
use crate::db::Database;
use crate::inventory::{InventoryStatus, TickerInventory};

// ============================================================================
// Aggregation Task
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationTask {
    pub task_id: usize,
    pub ticker: String,
    pub interval: String,
    pub start_ts: Option<DateTime<Utc>>,
    pub end_ts: DateTime<Utc>,
    pub range_param: String,
    pub reason: TaskReason,
    pub priority: u32,
    pub estimated_bars: usize,
    pub estimated_requests: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskReason {
    /// Ticker never downloaded for this interval
    InitialFill,
    /// Ticker has gaps that need filling
    GapFill,
    /// Ticker data is stale (needs update)
    StaleUpdate,
    /// Incremental update (delta from last_ts)
    Incremental,
}

impl std::fmt::Display for TaskReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskReason::InitialFill => write!(f, "INITIAL_FILL"),
            TaskReason::GapFill => write!(f, "GAP_FILL"),
            TaskReason::StaleUpdate => write!(f, "STALE_UPDATE"),
            TaskReason::Incremental => write!(f, "INCREMENTAL"),
        }
    }
}

// ============================================================================
// Aggregation Plan
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationPlan {
    pub generated_at: DateTime<Utc>,
    pub plan_version: String,
    pub total_tasks: usize,
    pub total_estimated_requests: usize,
    pub tasks_by_reason: HashMap<String, usize>,
    pub tasks_by_interval: HashMap<String, usize>,
    pub skipped_count: usize,
    pub skipped_reasons: HashMap<String, usize>,
    pub tasks: Vec<AggregationTask>,
}

impl AggregationPlan {
    pub fn write_json(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)?;
        info!("Aggregation plan written to {}", path.display());
        Ok(())
    }

    pub fn write_summary_md(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut content = format!(
            "# Aggregation Plan Summary\n\n\
             Generated: {}\n\n\
             ## Overview\n\n\
             - Total Tasks: {}\n\
             - Estimated Requests: {}\n\
             - Skipped Pairs: {}\n\n",
            self.generated_at.to_rfc3339(),
            self.total_tasks,
            self.total_estimated_requests,
            self.skipped_count,
        );

        content.push_str("## Tasks by Reason\n\n");
        content.push_str("| Reason | Count |\n|--------|-------|\n");
        for (reason, count) in &self.tasks_by_reason {
            content.push_str(&format!("| {} | {} |\n", reason, count));
        }

        content.push_str("\n## Tasks by Interval\n\n");
        content.push_str("| Interval | Count |\n|----------|-------|\n");
        for (interval, count) in &self.tasks_by_interval {
            content.push_str(&format!("| {} | {} |\n", interval, count));
        }

        if !self.skipped_reasons.is_empty() {
            content.push_str("\n## Skipped Reasons\n\n");
            content.push_str("| Reason | Count |\n|--------|-------|\n");
            for (reason, count) in &self.skipped_reasons {
                content.push_str(&format!("| {} | {} |\n", reason, count));
            }
        }

        content.push_str("\n## First 20 Tasks\n\n");
        content.push_str("| ID | Ticker | Interval | Reason | Est. Bars |\n");
        content.push_str("|----|--------|----------|--------|----------|\n");
        for task in self.tasks.iter().take(20) {
            content.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                task.task_id, task.ticker, task.interval, task.reason, task.estimated_bars,
            ));
        }

        std::fs::write(path, content)?;
        info!("Plan summary written to {}", path.display());
        Ok(())
    }

    /// Load plan from JSON file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

// ============================================================================
// Aggregation Planner
// ============================================================================

pub struct AggregationPlanner<'a> {
    db: &'a Database,
    now: DateTime<Utc>,
}

impl<'a> AggregationPlanner<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self {
            db,
            now: Utc::now(),
        }
    }

    /// Generate aggregation plan from inventory.
    pub fn generate_plan(&self, inventory: &[TickerInventory]) -> AggregationPlan {
        let mut tasks = Vec::new();
        let mut task_id = 0usize;
        let mut skipped_count = 0usize;
        let mut skipped_reasons: HashMap<String, usize> = HashMap::new();
        let mut tasks_by_reason: HashMap<String, usize> = HashMap::new();
        let mut tasks_by_interval: HashMap<String, usize> = HashMap::new();

        // Sort inventory by priority (daily first, then by ticker)
        let mut sorted_inventory = inventory.to_vec();
        sorted_inventory.sort_by(|a, b| {
            let pri_a = CoverageRule::priority(&a.interval);
            let pri_b = CoverageRule::priority(&b.interval);
            if pri_a != pri_b {
                pri_a.cmp(&pri_b)
            } else {
                a.ticker.cmp(&b.ticker)
            }
        });

        for item in sorted_inventory {
            // Skip if no capability
            if !item.has_capability && item.interval != "1d" {
                *skipped_reasons
                    .entry("NO_CAPABILITY".to_string())
                    .or_insert(0) += 1;
                skipped_count += 1;
                continue;
            }

            // Skip if already complete
            if item.status == InventoryStatus::Complete {
                *skipped_reasons
                    .entry("ALREADY_COMPLETE".to_string())
                    .or_insert(0) += 1;
                skipped_count += 1;
                continue;
            }

            // Determine task type and parameters
            let (reason, start_ts) = match item.status {
                InventoryStatus::Empty => (TaskReason::InitialFill, None),
                InventoryStatus::Partial => {
                    // For partial, start from last_ts + 1 bar
                    (TaskReason::GapFill, item.last_ts)
                }
                InventoryStatus::Stale => {
                    // For stale, update from last_ts
                    (TaskReason::StaleUpdate, item.last_ts)
                }
                InventoryStatus::Broken => {
                    // For broken, do initial fill
                    (TaskReason::InitialFill, None)
                }
                InventoryStatus::Complete => continue, // Already skipped above
            };

            let range_param = item
                .max_range
                .clone()
                .unwrap_or_else(|| get_backfill_range(&item.interval).to_string());
            let rule = CoverageRule::for_interval(&item.interval);

            // Estimate bars
            let estimated_bars = if start_ts.is_some() {
                // Incremental: estimate from last_ts to now
                let days = (self.now - start_ts.unwrap()).num_days().max(1);
                (days as usize * rule.expected_bars_per_day as usize).max(1)
            } else {
                // Full backfill
                let days = parse_range_to_days(&range_param);
                (days as usize * rule.expected_bars_per_day as usize).max(1)
            };

            // Estimate requests (1 request per range, might need chunking for large ranges)
            let estimated_requests = 1;

            let task = AggregationTask {
                task_id,
                ticker: item.ticker.clone(),
                interval: item.interval.clone(),
                start_ts,
                end_ts: self.now,
                range_param,
                reason,
                priority: CoverageRule::priority(&item.interval),
                estimated_bars,
                estimated_requests,
            };

            *tasks_by_reason.entry(reason.to_string()).or_insert(0) += 1;
            *tasks_by_interval.entry(item.interval.clone()).or_insert(0) += 1;

            tasks.push(task);
            task_id += 1;
        }

        let total_estimated_requests: usize = tasks.iter().map(|t| t.estimated_requests).sum();

        info!(
            "Plan generated: {} tasks, {} estimated requests, {} skipped",
            tasks.len(),
            total_estimated_requests,
            skipped_count
        );

        AggregationPlan {
            generated_at: self.now,
            plan_version: "1.0".to_string(),
            total_tasks: tasks.len(),
            total_estimated_requests,
            tasks_by_reason,
            tasks_by_interval,
            skipped_count,
            skipped_reasons,
            tasks,
        }
    }

    /// Generate incremental sync plan (only stale/incomplete).
    pub fn generate_sync_plan(&self, inventory: &[TickerInventory]) -> AggregationPlan {
        // Filter to only stale items
        let sync_items: Vec<_> = inventory
            .iter()
            .filter(|i| i.status == InventoryStatus::Stale || i.staleness_days > 1)
            .cloned()
            .collect();

        self.generate_plan(&sync_items)
    }
}


























