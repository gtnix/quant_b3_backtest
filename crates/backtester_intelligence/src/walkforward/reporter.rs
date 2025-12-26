//! Walk-Forward Report generation.
//!
//! Generates deterministic human and AI-readable outputs.

use rust_decimal::Decimal;
use serde::Serialize;

use super::types::{AggregateMetrics, AggregateReport, ParamSet, WindowResult};

/// Reporter for walk-forward validation results.
#[derive(Debug, Default)]
pub struct WalkForwardReporter;

/// JSON output structure for AI consumption.
#[derive(Debug, Clone, Serialize)]
pub struct WalkForwardJson {
    pub config: ConfigJson,
    pub windows: Vec<WindowJson>,
    pub aggregate: AggregateJson,
    pub params_selected: ParamJson,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfigJson {
    pub train_months: u32,
    pub test_months: u32,
    pub step_months: u32,
    pub purge_days: u32,
    pub embargo_days: u32,
    pub market: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct WindowJson {
    pub index: usize,
    pub train_period: String,
    pub test_period: String,
    pub train_sharpe: String,
    pub test_sharpe: String,
    pub test_return_pct: String,
    pub test_max_dd_pct: String,
    pub params: ParamJson,
}

#[derive(Debug, Clone, Serialize)]
pub struct AggregateJson {
    pub mean_sharpe: String,
    pub median_sharpe: String,
    pub std_sharpe: String,
    pub mean_return_pct: String,
    pub worst_drawdown_pct: String,
    pub robustness_score: String,
    pub stability_score: String,
    pub total_windows: usize,
    pub best_window_idx: usize,
    pub worst_window_idx: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParamJson {
    pub top_n: usize,
    pub stop_loss_pct: String,
    pub take_profit_pct: String,
    pub max_weight_pct: String,
    pub turnover_cap_pct: String,
    pub min_score: String,
}

impl WalkForwardReporter {
    pub fn new() -> Self {
        Self
    }

    /// Generate a human-readable summary.
    pub fn to_summary(&self, report: &AggregateReport) -> String {
        let mut lines = Vec::new();

        // Header
        lines.push("=".repeat(60));
        lines.push("WALK-FORWARD VALIDATION REPORT".to_string());
        lines.push("=".repeat(60));
        lines.push(String::new());

        // Configuration
        lines.push("Configuration:".to_string());
        lines.push(format!("  Train period: {} months", report.config.train_months));
        lines.push(format!("  Test period: {} months", report.config.test_months));
        lines.push(format!("  Step: {} months", report.config.step_months));
        lines.push(format!("  Purge: {} days", report.config.purge_days));
        lines.push(format!("  Embargo: {} days", report.config.embargo_days));
        lines.push(format!("  Market: {:?}", report.config.market));
        lines.push(format!("  Grid search: {}", report.config.grid.is_some()));
        lines.push(String::new());

        // Aggregate metrics
        lines.push("-".repeat(60));
        lines.push("AGGREGATE METRICS (Out-of-Sample)".to_string());
        lines.push("-".repeat(60));
        lines.push(format!("  Total windows: {}", report.aggregate.total_windows));
        lines.push(format!("  Total months tested: {}", report.aggregate.total_months_tested));
        lines.push(String::new());
        lines.push("  Sharpe Ratio:".to_string());
        lines.push(format!("    Mean: {:.3}", report.aggregate.mean_sharpe));
        lines.push(format!("    Median: {:.3}", report.aggregate.median_sharpe));
        lines.push(format!("    Std: {:.3}", report.aggregate.std_sharpe));
        lines.push(String::new());
        lines.push("  Returns:".to_string());
        lines.push(format!("    Mean: {:.2}%", report.aggregate.mean_return));
        lines.push(format!("    Median: {:.2}%", report.aggregate.median_return));
        lines.push(format!("    Std: {:.2}%", report.aggregate.std_return));
        lines.push(String::new());
        lines.push("  Risk:".to_string());
        lines.push(format!("    Mean Drawdown: {:.2}%", report.aggregate.mean_drawdown));
        lines.push(format!("    Worst Drawdown: {:.2}%", report.aggregate.worst_drawdown));
        lines.push(format!("    Mean Volatility: {:.2}%", report.aggregate.mean_volatility));
        lines.push(String::new());
        lines.push("  Scores:".to_string());
        lines.push(format!("    Robustness: {:.3}", report.aggregate.robustness_score));
        lines.push(format!("    Stability: {:.3}", report.aggregate.stability_score));
        lines.push(String::new());
        lines.push(format!("  Best window: #{}", report.aggregate.best_window_idx));
        lines.push(format!("  Worst window: #{}", report.aggregate.worst_window_idx));
        lines.push(String::new());

        // Most selected params
        lines.push("-".repeat(60));
        lines.push("MOST SELECTED PARAMETERS".to_string());
        lines.push("-".repeat(60));
        lines.push(format!("  Top-N: {}", report.most_selected_params.top_n));
        lines.push(format!("  Stop-Loss: {:.1}%", report.most_selected_params.stop_loss_pct * Decimal::from(100)));
        lines.push(format!("  Take-Profit: {:.1}%", report.most_selected_params.take_profit_pct * Decimal::from(100)));
        lines.push(format!("  Max Weight: {:.1}%", report.most_selected_params.max_weight * Decimal::from(100)));
        lines.push(format!("  Turnover Cap: {:.1}%", report.most_selected_params.turnover_cap * Decimal::from(100)));
        lines.push(format!("  Min Score: {:.2}", report.most_selected_params.min_score));
        lines.push(String::new());

        // Per-window results (sorted by index)
        lines.push("-".repeat(60));
        lines.push("PER-WINDOW RESULTS".to_string());
        lines.push("-".repeat(60));
        lines.push(format!(
            "{:>4} | {:>10} - {:>10} | {:>8} | {:>8} | {:>8}",
            "#", "Train End", "Test End", "Sharpe", "Return%", "MaxDD%"
        ));
        lines.push("-".repeat(60));

        let mut sorted_windows: Vec<_> = report.windows.iter().collect();
        sorted_windows.sort_by_key(|w| w.split.index);

        for w in sorted_windows {
            lines.push(format!(
                "{:>4} | {:>10} - {:>10} | {:>8.3} | {:>8.2} | {:>8.2}",
                w.split.index,
                w.split.train.end_date.format("%Y-%m-%d"),
                w.split.test.end_date.format("%Y-%m-%d"),
                w.test_metrics.sharpe_ratio,
                w.test_metrics.total_return_pct,
                w.test_metrics.max_drawdown_pct
            ));
        }

        lines.push(String::new());
        lines.push("=".repeat(60));
        lines.push(format!("Generated: {}", report.generated_at));
        lines.push("=".repeat(60));

        lines.join("\n")
    }

    /// Generate a compact one-line summary.
    pub fn to_compact(&self, report: &AggregateReport) -> String {
        format!(
            "WF [{:?}] {} windows | Sharpe: {:.2} (σ={:.2}) | Return: {:.1}% | MaxDD: {:.1}% | Robustness: {:.2}",
            report.config.market,
            report.aggregate.total_windows,
            report.aggregate.mean_sharpe,
            report.aggregate.std_sharpe,
            report.aggregate.mean_return,
            report.aggregate.worst_drawdown,
            report.aggregate.robustness_score
        )
    }

    /// Generate JSON output for AI consumption.
    pub fn to_json(&self, report: &AggregateReport) -> WalkForwardJson {
        let config = ConfigJson {
            train_months: report.config.train_months,
            test_months: report.config.test_months,
            step_months: report.config.step_months,
            purge_days: report.config.purge_days,
            embargo_days: report.config.embargo_days,
            market: format!("{:?}", report.config.market),
        };

        let mut windows: Vec<WindowJson> = report.windows
            .iter()
            .map(|w| self.window_to_json(w))
            .collect();

        // Sort by index for determinism
        windows.sort_by_key(|w| w.index);

        let aggregate = self.aggregate_to_json(&report.aggregate);
        let params_selected = self.params_to_json(&report.most_selected_params);

        WalkForwardJson {
            config,
            windows,
            aggregate,
            params_selected,
        }
    }

    /// Generate JSON string.
    pub fn to_json_string(&self, report: &AggregateReport) -> String {
        let json = self.to_json(report);
        serde_json::to_string_pretty(&json).unwrap_or_else(|_| "{}".to_string())
    }

    fn window_to_json(&self, w: &WindowResult) -> WindowJson {
        WindowJson {
            index: w.split.index,
            train_period: format!(
                "{}/{}",
                w.split.train.start_date.format("%Y-%m-%d"),
                w.split.train.end_date.format("%Y-%m-%d")
            ),
            test_period: format!(
                "{}/{}",
                w.split.test.start_date.format("%Y-%m-%d"),
                w.split.test.end_date.format("%Y-%m-%d")
            ),
            train_sharpe: format!("{:.4}", w.train_metrics.sharpe_ratio),
            test_sharpe: format!("{:.4}", w.test_metrics.sharpe_ratio),
            test_return_pct: format!("{:.2}", w.test_metrics.total_return_pct),
            test_max_dd_pct: format!("{:.2}", w.test_metrics.max_drawdown_pct),
            params: self.params_to_json(&w.selected_params),
        }
    }

    fn aggregate_to_json(&self, agg: &AggregateMetrics) -> AggregateJson {
        AggregateJson {
            mean_sharpe: format!("{:.4}", agg.mean_sharpe),
            median_sharpe: format!("{:.4}", agg.median_sharpe),
            std_sharpe: format!("{:.4}", agg.std_sharpe),
            mean_return_pct: format!("{:.2}", agg.mean_return),
            worst_drawdown_pct: format!("{:.2}", agg.worst_drawdown),
            robustness_score: format!("{:.4}", agg.robustness_score),
            stability_score: format!("{:.4}", agg.stability_score),
            total_windows: agg.total_windows,
            best_window_idx: agg.best_window_idx,
            worst_window_idx: agg.worst_window_idx,
        }
    }

    fn params_to_json(&self, p: &ParamSet) -> ParamJson {
        ParamJson {
            top_n: p.top_n,
            stop_loss_pct: format!("{:.2}", p.stop_loss_pct * Decimal::from(100)),
            take_profit_pct: format!("{:.2}", p.take_profit_pct * Decimal::from(100)),
            max_weight_pct: format!("{:.2}", p.max_weight * Decimal::from(100)),
            turnover_cap_pct: format!("{:.2}", p.turnover_cap * Decimal::from(100)),
            min_score: format!("{:.2}", p.min_score),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::Market;
    use crate::walkforward::types::{WalkForwardConfig, WindowSplit, WindowSpec, WindowType, WindowMetrics};
    use chrono::NaiveDate;
    use rust_decimal_macros::dec;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    fn make_report() -> AggregateReport {
        let config = WalkForwardConfig::default();

        let make_window = |idx: usize, sharpe: f64, ret: f64, dd: f64| -> WindowResult {
            WindowResult {
                split: WindowSplit {
                    train: WindowSpec::new(
                        date(2020, 1, 1),
                        date(2020, 6, 25),
                        WindowType::Train,
                        idx,
                    ),
                    test: WindowSpec::new(
                        date(2020, 7, 5),
                        date(2020, 10, 5),
                        WindowType::Test,
                        idx,
                    ),
                    purge_days: 5,
                    embargo_days: 5,
                    index: idx,
                },
                train_metrics: WindowMetrics {
                    sharpe_ratio: dec!(1.5),
                    ..Default::default()
                },
                test_metrics: WindowMetrics {
                    sharpe_ratio: rust_decimal::Decimal::try_from(sharpe).unwrap(),
                    total_return_pct: rust_decimal::Decimal::try_from(ret).unwrap(),
                    max_drawdown_pct: rust_decimal::Decimal::try_from(dd).unwrap(),
                    ..Default::default()
                },
                selected_params: ParamSet::default(),
                is_oos: true,
            }
        };

        let windows = vec![
            make_window(0, 1.2, 5.5, 8.2),
            make_window(1, 0.9, 3.2, 12.1),
            make_window(2, 1.4, 7.1, 6.5),
        ];

        let aggregate = AggregateMetrics {
            mean_sharpe: dec!(1.167),
            median_sharpe: dec!(1.2),
            std_sharpe: dec!(0.21),
            mean_return: dec!(5.27),
            median_return: dec!(5.5),
            std_return: dec!(1.6),
            mean_drawdown: dec!(8.93),
            worst_drawdown: dec!(12.1),
            mean_volatility: dec!(15.5),
            stability_score: dec!(0.82),
            robustness_score: dec!(0.75),
            best_window_idx: 2,
            worst_window_idx: 1,
            total_windows: 3,
            total_months_tested: 9,
        };

        AggregateReport {
            config,
            windows,
            aggregate,
            most_selected_params: ParamSet::default(),
            generated_at: date(2024, 1, 1),
        }
    }

    #[test]
    fn test_to_summary() {
        let report = make_report();
        let reporter = WalkForwardReporter::new();

        let summary = reporter.to_summary(&report);

        assert!(summary.contains("WALK-FORWARD VALIDATION REPORT"));
        assert!(summary.contains("Train period: 6 months"));
        assert!(summary.contains("Total windows: 3"));
        assert!(summary.contains("Mean: 1.167"));
        assert!(summary.contains("Robustness: 0.75"));
    }

    #[test]
    fn test_to_compact() {
        let report = make_report();
        let reporter = WalkForwardReporter::new();

        let compact = reporter.to_compact(&report);

        assert!(compact.contains("BR"));
        assert!(compact.contains("3 windows"));
        assert!(compact.contains("Sharpe:"));
    }

    #[test]
    fn test_to_json() {
        let report = make_report();
        let reporter = WalkForwardReporter::new();

        let json = reporter.to_json(&report);

        assert_eq!(json.config.train_months, 6);
        assert_eq!(json.windows.len(), 3);
        assert_eq!(json.aggregate.total_windows, 3);
    }

    #[test]
    fn test_to_json_string() {
        let report = make_report();
        let reporter = WalkForwardReporter::new();

        let json_str = reporter.to_json_string(&report);

        assert!(json_str.contains("\"train_months\": 6"));
        assert!(json_str.contains("\"total_windows\": 3"));

        // Verify it parses
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert!(parsed.is_object());
    }

    #[test]
    fn test_determinism() {
        let report = make_report();
        let reporter = WalkForwardReporter::new();

        let summary1 = reporter.to_summary(&report);
        let summary2 = reporter.to_summary(&report);
        assert_eq!(summary1, summary2);

        let json1 = reporter.to_json_string(&report);
        let json2 = reporter.to_json_string(&report);
        assert_eq!(json1, json2);
    }

    #[test]
    fn test_windows_sorted_by_index() {
        let report = make_report();
        let reporter = WalkForwardReporter::new();

        let json = reporter.to_json(&report);

        for (i, w) in json.windows.iter().enumerate() {
            assert_eq!(w.index, i);
        }
    }
}

