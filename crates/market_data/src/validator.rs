//! Post-Aggregation Validator - compares BEFORE vs AFTER state.
//!
//! Generates delta reports proving coverage improvements.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::info;

use crate::inventory::{InventoryStatus, InventorySummary, TickerInventory};

// ============================================================================
// Coverage Delta
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageDelta {
    pub ticker: String,
    pub interval: String,
    pub before_status: InventoryStatus,
    pub after_status: InventoryStatus,
    pub before_bars: i64,
    pub after_bars: i64,
    pub bars_added: i64,
    pub before_coverage: f64,
    pub after_coverage: f64,
    pub coverage_delta: f64,
    pub improved: bool,
}

// ============================================================================
// Delta Report
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaReport {
    pub generated_at: DateTime<Utc>,
    pub before_scan_at: DateTime<Utc>,
    pub after_scan_at: DateTime<Utc>,
    pub summary: DeltaSummary,
    pub deltas: Vec<CoverageDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaSummary {
    pub total_pairs_compared: usize,
    pub pairs_improved: usize,
    pub pairs_unchanged: usize,
    pub pairs_degraded: usize,
    pub total_bars_before: i64,
    pub total_bars_after: i64,
    pub total_bars_added: i64,
    pub avg_coverage_before: f64,
    pub avg_coverage_after: f64,
    pub coverage_improvement: f64,
    pub status_transitions: HashMap<String, usize>,
}

impl DeltaReport {
    pub fn write_md(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut content = format!(
            "# Coverage Delta Report\n\n\
             Generated: {}\n\n\
             ## Summary\n\n\
             | Metric | Before | After | Delta |\n\
             |--------|--------|-------|-------|\n\
             | Total Bars | {} | {} | +{} |\n\
             | Avg Coverage | {:.1}% | {:.1}% | +{:.1}% |\n\
             | Pairs Improved | - | - | {} |\n\
             | Pairs Unchanged | - | - | {} |\n\n",
            self.generated_at.to_rfc3339(),
            self.summary.total_bars_before,
            self.summary.total_bars_after,
            self.summary.total_bars_added,
            self.summary.avg_coverage_before,
            self.summary.avg_coverage_after,
            self.summary.coverage_improvement,
            self.summary.pairs_improved,
            self.summary.pairs_unchanged,
        );

        content.push_str("## Status Transitions\n\n");
        content.push_str("| Transition | Count |\n|------------|-------|\n");
        for (transition, count) in &self.summary.status_transitions {
            content.push_str(&format!("| {} | {} |\n", transition, count));
        }

        content.push_str("\n## Top Improvements\n\n");
        content.push_str("| Ticker | Interval | Before | After | Bars Added |\n");
        content.push_str("|--------|----------|--------|-------|------------|\n");

        let mut sorted_deltas = self.deltas.clone();
        sorted_deltas.sort_by(|a, b| b.bars_added.cmp(&a.bars_added));

        for delta in sorted_deltas.iter().take(30) {
            if delta.bars_added > 0 {
                content.push_str(&format!(
                    "| {} | {} | {} | {} | {} |\n",
                    delta.ticker,
                    delta.interval,
                    delta.before_status,
                    delta.after_status,
                    delta.bars_added,
                ));
            }
        }

        std::fs::write(path, content)?;
        info!("Delta report written to {}", path.display());
        Ok(())
    }

    pub fn write_json(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)?;
        info!("Delta JSON written to {}", path.display());
        Ok(())
    }
}

// ============================================================================
// Freshness Report
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreshnessEntry {
    pub ticker: String,
    pub interval: String,
    pub last_ts: Option<DateTime<Utc>>,
    pub staleness_days: i64,
    pub staleness_hours: i64,
    pub is_fresh: bool,
}

pub struct FreshnessReport;

impl FreshnessReport {
    pub fn generate(inventory: &[TickerInventory], threshold_hours: i64) -> Vec<FreshnessEntry> {
        let now = Utc::now();

        inventory
            .iter()
            .map(|item| {
                let staleness_hours = item
                    .last_ts
                    .map(|ts| (now - ts).num_hours())
                    .unwrap_or(999999);

                FreshnessEntry {
                    ticker: item.ticker.clone(),
                    interval: item.interval.clone(),
                    last_ts: item.last_ts,
                    staleness_days: item.staleness_days,
                    staleness_hours,
                    is_fresh: staleness_hours <= threshold_hours,
                }
            })
            .collect()
    }

    pub fn write_csv(entries: &[FreshnessEntry], path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut content =
            "ticker,interval,last_ts,staleness_days,staleness_hours,is_fresh\n".to_string();

        for e in entries {
            content.push_str(&format!(
                "{},{},{},{},{},{}\n",
                e.ticker,
                e.interval,
                e.last_ts.map(|t| t.to_rfc3339()).unwrap_or_default(),
                e.staleness_days,
                e.staleness_hours,
                e.is_fresh,
            ));
        }

        std::fs::write(path, content)?;
        info!("Freshness report written to {}", path.display());
        Ok(())
    }
}

// ============================================================================
// Validator
// ============================================================================

pub struct PostAggregationValidator;

impl PostAggregationValidator {
    /// Compare before and after inventories.
    pub fn compare(
        before: &[TickerInventory],
        _before_summary: &InventorySummary,
        after: &[TickerInventory],
        _after_summary: &InventorySummary,
    ) -> DeltaReport {
        let now = Utc::now();

        // Build lookup for before state
        let before_map: HashMap<(String, String), &TickerInventory> = before
            .iter()
            .map(|i| ((i.ticker.clone(), i.interval.clone()), i))
            .collect();

        let mut deltas = Vec::new();
        let mut status_transitions: HashMap<String, usize> = HashMap::new();
        let mut pairs_improved = 0usize;
        let mut pairs_unchanged = 0usize;
        let mut pairs_degraded = 0usize;
        let mut total_bars_before = 0i64;
        let mut total_bars_after = 0i64;
        let mut coverage_sum_before = 0.0f64;
        let mut coverage_sum_after = 0.0f64;

        for after_item in after {
            let key = (after_item.ticker.clone(), after_item.interval.clone());

            let before_item = before_map.get(&key);
            let before_status = before_item
                .map(|b| b.status)
                .unwrap_or(InventoryStatus::Empty);
            let before_bars = before_item.map(|b| b.bar_count).unwrap_or(0);
            let before_coverage = before_item.map(|b| b.coverage_pct).unwrap_or(0.0);

            let bars_added = after_item.bar_count - before_bars;
            let coverage_delta = after_item.coverage_pct - before_coverage;

            // Track transitions
            if before_status != after_item.status {
                let transition = format!("{} -> {}", before_status, after_item.status);
                *status_transitions.entry(transition).or_insert(0) += 1;
            }

            // Track improvements
            if bars_added > 0 {
                pairs_improved += 1;
            } else if bars_added == 0 {
                pairs_unchanged += 1;
            } else {
                pairs_degraded += 1;
            }

            total_bars_before += before_bars;
            total_bars_after += after_item.bar_count;
            coverage_sum_before += before_coverage;
            coverage_sum_after += after_item.coverage_pct;

            deltas.push(CoverageDelta {
                ticker: after_item.ticker.clone(),
                interval: after_item.interval.clone(),
                before_status,
                after_status: after_item.status,
                before_bars,
                after_bars: after_item.bar_count,
                bars_added,
                before_coverage,
                after_coverage: after_item.coverage_pct,
                coverage_delta,
                improved: bars_added > 0,
            });
        }

        let n = deltas.len().max(1) as f64;
        let avg_coverage_before = coverage_sum_before / n;
        let avg_coverage_after = coverage_sum_after / n;

        DeltaReport {
            generated_at: now,
            before_scan_at: now, // Would need to be passed from actual scans
            after_scan_at: now,
            summary: DeltaSummary {
                total_pairs_compared: deltas.len(),
                pairs_improved,
                pairs_unchanged,
                pairs_degraded,
                total_bars_before,
                total_bars_after,
                total_bars_added: total_bars_after - total_bars_before,
                avg_coverage_before,
                avg_coverage_after,
                coverage_improvement: avg_coverage_after - avg_coverage_before,
                status_transitions,
            },
            deltas,
        }
    }
}



























