//! OHLCV Integrity Audit - read-only audit of persisted data.
//!
//! Builds ticker x interval matrix and calculates integrity metrics.
//! Output to STDOUT only, no file creation, no API calls.

use chrono::NaiveDate;
use std::collections::{HashMap, HashSet};

use crate::db::{Database, DbError, OhlcvAuditRow};

// ============================================================================
// Configuration
// ============================================================================

pub struct AuditConfig {
    pub min_integrity: f64,
    pub sample_outliers: usize,
    pub max_hierarchy_violations_pct: f64,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            min_integrity: 0.95,
            sample_outliers: 20,
            max_hierarchy_violations_pct: 1.0,
        }
    }
}

// ============================================================================
// Audit Data Structures
// ============================================================================

#[derive(Debug, Clone)]
pub struct TickerCell {
    pub bar_count: i64,
    pub min_date: Option<NaiveDate>,
    pub max_date: Option<NaiveDate>,
}

#[derive(Debug)]
pub struct IntegrityMatrix {
    pub tickers: Vec<String>,
    pub intervals: Vec<String>,
    // cells[ticker_idx][interval_idx] = Option<TickerCell>
    pub cells: HashMap<(String, String), TickerCell>,
    pub capabilities: HashMap<String, HashSet<String>>,
}

#[derive(Debug)]
pub struct IntervalStats {
    pub interval: String,
    pub total_tickers: usize,
    pub tickers_with_data: usize,
    pub expected_cap_aware: usize,
    pub filled_cap_aware: usize,
    pub min_bars: i64,
    pub max_bars: i64,
    pub p10: i64,
    pub median: i64,
    pub p90: i64,
    pub lowest_tickers: Vec<(String, i64)>,
    pub highest_tickers: Vec<(String, i64)>,
}

#[derive(Debug)]
pub struct HierarchyViolation {
    pub ticker: String,
    pub fine_interval: String,
    pub fine_count: i64,
    pub coarse_interval: String,
    pub coarse_count: i64,
}

#[derive(Debug)]
pub struct AuditResult {
    pub total_tickers: usize,
    pub intervals: Vec<String>,
    pub cap_aware_expected: usize,
    pub cap_aware_filled: usize,
    pub cap_aware_integrity: f64,
    pub global_total: usize,
    pub global_filled: usize,
    pub global_integrity: f64,
    pub interval_stats: Vec<IntervalStats>,
    pub hierarchy_violations: Vec<HierarchyViolation>,
    pub hierarchy_violation_rate: f64,
    pub passed: bool,
    pub failure_reasons: Vec<String>,
}

// ============================================================================
// Integrity Auditor
// ============================================================================

pub struct IntegrityAuditor<'a> {
    db: &'a Database,
    config: AuditConfig,
}

impl<'a> IntegrityAuditor<'a> {
    pub fn new(db: &'a Database, config: AuditConfig) -> Self {
        Self { db, config }
    }

    /// Run the full audit (read-only).
    pub async fn run(&self) -> Result<AuditResult, DbError> {
        // 1. Load data from database
        let active_tickers = self.db.get_active_tickers().await?;
        let capabilities = self.db.get_all_capabilities().await?;
        let daily_data = self.db.get_ohlcv_daily_audit().await?;
        let intraday_data = self.db.get_ohlcv_intraday_audit().await?;

        // 2. Build matrix
        let matrix = self.build_matrix(&active_tickers, &capabilities, &daily_data, &intraday_data);

        // 3. Calculate metrics
        let result = self.calculate_metrics(&matrix);

        Ok(result)
    }

    fn build_matrix(
        &self,
        active_tickers: &[String],
        capabilities: &HashMap<String, Vec<String>>,
        daily_data: &[OhlcvAuditRow],
        intraday_data: &[OhlcvAuditRow],
    ) -> IntegrityMatrix {
        // Canonical intervals ordered by granularity (finest to coarsest)
        let intervals = vec!["1m", "5m", "15m", "60m", "1d"];

        // Build cells map
        let mut cells: HashMap<(String, String), TickerCell> = HashMap::new();

        // Add daily data
        for row in daily_data {
            if active_tickers.contains(&row.symbol) {
                cells.insert(
                    (row.symbol.clone(), row.interval.clone()),
                    TickerCell {
                        bar_count: row.bar_count,
                        min_date: row.min_date,
                        max_date: row.max_date,
                    },
                );
            }
        }

        // Add intraday data
        for row in intraday_data {
            if active_tickers.contains(&row.symbol) {
                cells.insert(
                    (row.symbol.clone(), row.interval.clone()),
                    TickerCell {
                        bar_count: row.bar_count,
                        min_date: row.min_date,
                        max_date: row.max_date,
                    },
                );
            }
        }

        // Build capabilities map
        let caps: HashMap<String, HashSet<String>> = capabilities
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().cloned().collect()))
            .collect();

        IntegrityMatrix {
            tickers: active_tickers.to_vec(),
            intervals: intervals.iter().map(|s| s.to_string()).collect(),
            cells,
            capabilities: caps,
        }
    }

    fn calculate_metrics(&self, matrix: &IntegrityMatrix) -> AuditResult {
        let total_tickers = matrix.tickers.len();
        let intervals = matrix.intervals.clone();

        // Calculate coverage
        let mut cap_aware_expected = 0usize;
        let mut cap_aware_filled = 0usize;
        let global_total = total_tickers * intervals.len();
        let mut global_filled = 0usize;

        for ticker in &matrix.tickers {
            let ticker_caps = matrix.capabilities.get(ticker);

            for interval in &intervals {
                let has_data = matrix
                    .cells
                    .contains_key(&(ticker.clone(), interval.clone()));

                if has_data {
                    global_filled += 1;
                }

                // Cap-aware: expected if ticker has capability for this interval
                // For daily (1d), always expect it
                let is_expected = if interval == "1d" {
                    true
                } else {
                    ticker_caps
                        .map(|caps| caps.contains(interval))
                        .unwrap_or(false)
                };

                if is_expected {
                    cap_aware_expected += 1;
                    if has_data {
                        cap_aware_filled += 1;
                    }
                }
            }
        }

        let cap_aware_integrity = if cap_aware_expected > 0 {
            cap_aware_filled as f64 / cap_aware_expected as f64
        } else {
            0.0
        };

        let global_integrity = if global_total > 0 {
            global_filled as f64 / global_total as f64
        } else {
            0.0
        };

        // Calculate per-interval stats
        let interval_stats: Vec<IntervalStats> = intervals
            .iter()
            .map(|interval| self.calculate_interval_stats(matrix, interval))
            .collect();

        // Check hierarchy violations
        let hierarchy_violations = self.check_hierarchy_violations(matrix);
        let hierarchy_violation_rate = if total_tickers > 0 {
            let violators: HashSet<_> = hierarchy_violations.iter().map(|v| &v.ticker).collect();
            violators.len() as f64 / total_tickers as f64 * 100.0
        } else {
            0.0
        };

        // Determine pass/fail
        let mut failure_reasons = Vec::new();

        if cap_aware_integrity < self.config.min_integrity {
            failure_reasons.push(format!(
                "Cap-aware integrity {:.1}% < {:.0}% threshold",
                cap_aware_integrity * 100.0,
                self.config.min_integrity * 100.0
            ));
        }

        if hierarchy_violation_rate > self.config.max_hierarchy_violations_pct {
            failure_reasons.push(format!(
                "Hierarchy violations {:.1}% > {:.0}% threshold",
                hierarchy_violation_rate, self.config.max_hierarchy_violations_pct
            ));
        }

        // Check for intervals with very low coverage
        for stats in &interval_stats {
            if stats.expected_cap_aware > 0 {
                let coverage = stats.filled_cap_aware as f64 / stats.expected_cap_aware as f64;
                if coverage < 0.5 {
                    failure_reasons.push(format!(
                        "{} has {:.1}% coverage (cap-aware)",
                        stats.interval,
                        coverage * 100.0
                    ));
                }
            }
        }

        let passed = failure_reasons.is_empty();

        AuditResult {
            total_tickers,
            intervals,
            cap_aware_expected,
            cap_aware_filled,
            cap_aware_integrity,
            global_total,
            global_filled,
            global_integrity,
            interval_stats,
            hierarchy_violations,
            hierarchy_violation_rate,
            passed,
            failure_reasons,
        }
    }

    fn calculate_interval_stats(&self, matrix: &IntegrityMatrix, interval: &str) -> IntervalStats {
        let mut counts: Vec<(String, i64)> = Vec::new();
        let mut expected_cap_aware = 0usize;
        let mut filled_cap_aware = 0usize;

        for ticker in &matrix.tickers {
            let has_data = matrix.cells.get(&(ticker.clone(), interval.to_string()));

            // Check if expected (cap-aware)
            let is_expected = if interval == "1d" {
                true
            } else {
                matrix
                    .capabilities
                    .get(ticker)
                    .map(|caps| caps.contains(interval))
                    .unwrap_or(false)
            };

            if is_expected {
                expected_cap_aware += 1;
            }

            if let Some(cell) = has_data {
                counts.push((ticker.clone(), cell.bar_count));
                if is_expected {
                    filled_cap_aware += 1;
                }
            }
        }

        // Sort by bar count for percentile calculation
        counts.sort_by_key(|(_, c)| *c);

        let tickers_with_data = counts.len();

        let (min_bars, max_bars, p10, median, p90) = if counts.is_empty() {
            (0, 0, 0, 0, 0)
        } else {
            let min_bars = counts.first().map(|(_, c)| *c).unwrap_or(0);
            let max_bars = counts.last().map(|(_, c)| *c).unwrap_or(0);
            let p10_idx = (counts.len() as f64 * 0.10) as usize;
            let p50_idx = counts.len() / 2;
            let p90_idx = (counts.len() as f64 * 0.90) as usize;

            let p10 = counts.get(p10_idx).map(|(_, c)| *c).unwrap_or(0);
            let median = counts.get(p50_idx).map(|(_, c)| *c).unwrap_or(0);
            let p90 = counts
                .get(p90_idx.min(counts.len() - 1))
                .map(|(_, c)| *c)
                .unwrap_or(0);

            (min_bars, max_bars, p10, median, p90)
        };

        // Get lowest and highest tickers
        let sample = self.config.sample_outliers.min(counts.len());
        let lowest_tickers: Vec<(String, i64)> = counts.iter().take(sample).cloned().collect();
        let highest_tickers: Vec<(String, i64)> =
            counts.iter().rev().take(sample).cloned().collect();

        IntervalStats {
            interval: interval.to_string(),
            total_tickers: matrix.tickers.len(),
            tickers_with_data,
            expected_cap_aware,
            filled_cap_aware,
            min_bars,
            max_bars,
            p10,
            median,
            p90,
            lowest_tickers,
            highest_tickers,
        }
    }

    fn check_hierarchy_violations(&self, matrix: &IntegrityMatrix) -> Vec<HierarchyViolation> {
        let mut violations = Vec::new();

        // Hierarchy: n(1m) >= n(5m) >= n(15m) >= n(60m) >= n(1d)
        let hierarchy = [("1m", "5m"), ("5m", "15m"), ("15m", "60m"), ("60m", "1d")];

        for ticker in &matrix.tickers {
            for (fine, coarse) in &hierarchy {
                let fine_count = matrix
                    .cells
                    .get(&(ticker.clone(), fine.to_string()))
                    .map(|c| c.bar_count);
                let coarse_count = matrix
                    .cells
                    .get(&(ticker.clone(), coarse.to_string()))
                    .map(|c| c.bar_count);

                if let (Some(fc), Some(cc)) = (fine_count, coarse_count) {
                    // Violation: finer granularity has FEWER bars than coarser
                    if fc < cc {
                        violations.push(HierarchyViolation {
                            ticker: ticker.clone(),
                            fine_interval: fine.to_string(),
                            fine_count: fc,
                            coarse_interval: coarse.to_string(),
                            coarse_count: cc,
                        });
                    }
                }
            }
        }

        // Sort by severity (biggest difference)
        violations.sort_by_key(|v| std::cmp::Reverse(v.coarse_count - v.fine_count));

        violations
    }

    /// Print audit results to STDOUT.
    pub fn print_results(&self, result: &AuditResult, duration_secs: f64) {
        println!("\n{}", "=".repeat(60));
        println!("              OHLCV INTEGRITY AUDIT");
        println!("{}\n", "=".repeat(60));

        // Config
        println!("Config:");
        println!(
            "  Min Integrity Threshold: {:.0}%",
            self.config.min_integrity * 100.0
        );
        println!(
            "  Max Hierarchy Violations: {:.0}%",
            self.config.max_hierarchy_violations_pct
        );
        println!("  Scan Duration: {:.2}s", duration_secs);

        // Universe
        println!("\nUniverse:");
        println!("  ACTIVE tickers: {}", result.total_tickers);
        println!("  Intervals: {:?}", result.intervals);

        // Rules applied
        println!("\n--- RULES APPLIED ---");
        println!("  1. Cap-Aware Coverage: cell expected if interval in ticker capabilities (1d always expected)");
        println!("  2. Hierarchy Check: n(fine) >= n(coarse) expected");
        println!("  3. Strict Equality: diagnostic only, not gate condition");

        // Coverage
        println!("\n--- COVERAGE ---");
        println!(
            "Cap-Aware Integrity: {:.1}% ({}/{} expected cells)",
            result.cap_aware_integrity * 100.0,
            result.cap_aware_filled,
            result.cap_aware_expected
        );
        println!(
            "Global Integrity:    {:.1}% ({}/{} total cells)",
            result.global_integrity * 100.0,
            result.global_filled,
            result.global_total
        );

        println!("\nBy Interval (cap-aware):");
        for stats in &result.interval_stats {
            let coverage = if stats.expected_cap_aware > 0 {
                stats.filled_cap_aware as f64 / stats.expected_cap_aware as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "  {:>4}: {:>5}/{:<5} ({:>5.1}%) | with_data: {}",
                stats.interval,
                stats.filled_cap_aware,
                stats.expected_cap_aware,
                coverage,
                stats.tickers_with_data
            );
        }

        // Uniformity
        println!("\n--- UNIFORMITY (cross-ticker) ---");
        for stats in &result.interval_stats {
            if stats.tickers_with_data > 0 {
                println!(
                    "[{:>4}] min={} p10={} median={} p90={} max={}",
                    stats.interval,
                    stats.min_bars,
                    stats.p10,
                    stats.median,
                    stats.p90,
                    stats.max_bars
                );

                if !stats.lowest_tickers.is_empty() {
                    let lowest: Vec<String> = stats
                        .lowest_tickers
                        .iter()
                        .take(5)
                        .map(|(t, c)| format!("{}({})", t, c))
                        .collect();
                    println!("  Lowest:  {}", lowest.join(", "));
                }

                if !stats.highest_tickers.is_empty() {
                    let highest: Vec<String> = stats
                        .highest_tickers
                        .iter()
                        .take(5)
                        .map(|(t, c)| format!("{}({})", t, c))
                        .collect();
                    println!("  Highest: {}", highest.join(", "));
                }
            }
        }

        // Hierarchy violations
        println!("\n--- HIERARCHY VIOLATIONS ---");
        let violator_count = {
            let violators: HashSet<_> = result
                .hierarchy_violations
                .iter()
                .map(|v| &v.ticker)
                .collect();
            violators.len()
        };
        println!(
            "Tickers with n(fine) < n(coarse): {}/{} ({:.1}%)",
            violator_count, result.total_tickers, result.hierarchy_violation_rate
        );

        if !result.hierarchy_violations.is_empty() {
            println!("\nTop violators:");
            for v in result.hierarchy_violations.iter().take(20) {
                println!(
                    "  {}: {}={}, {}={} (VIOLATION: {} < {})",
                    v.ticker,
                    v.fine_interval,
                    v.fine_count,
                    v.coarse_interval,
                    v.coarse_count,
                    v.fine_count,
                    v.coarse_count
                );
            }
        }

        // Strict equality (diagnostic)
        println!("\n--- STRICT EQUALITY (diagnostic only) ---");
        for stats in &result.interval_stats {
            if stats.tickers_with_data > 0 {
                // Count how many match median exactly
                let matches = stats
                    .lowest_tickers
                    .iter()
                    .chain(stats.highest_tickers.iter())
                    .filter(|(_, c)| *c == stats.median)
                    .count();
                let pct = matches as f64 / stats.tickers_with_data as f64 * 100.0;
                println!(
                    "  [{}] ~{:.0}% match median={} (strict equality NOT expected in real markets)",
                    stats.interval,
                    pct.min(100.0),
                    stats.median
                );
            }
        }

        // Final verdict
        println!("\n{}", "=".repeat(60));
        if result.passed {
            println!(
                "  INTEGRITY PASS (>={:.0}%): OK TO RUN FULL BACKTEST",
                self.config.min_integrity * 100.0
            );
        } else {
            println!(
                "  INTEGRITY FAIL (<{:.0}%): DO NOT RUN FULL BACKTEST",
                self.config.min_integrity * 100.0
            );
            println!("{}", "=".repeat(60));

            println!("\nFailure reasons:");
            for reason in &result.failure_reasons {
                println!("  - {}", reason);
            }

            println!("\nRecommendations:");
            // Check which intervals are low
            for stats in &result.interval_stats {
                if stats.expected_cap_aware > 0 {
                    let coverage = stats.filled_cap_aware as f64 / stats.expected_cap_aware as f64;
                    if coverage < 0.9 {
                        println!(
                            "  - {} has {:.1}% cap-aware coverage -> run 'aggregate-plan' + 'aggregate-run'",
                            stats.interval,
                            coverage * 100.0
                        );
                    }
                }
            }

            if result.hierarchy_violation_rate > 0.5 {
                println!("  - Hierarchy violations detected -> check timezone/aggregation logic");
            }
        }
        println!("{}", "=".repeat(60));
    }
}




























