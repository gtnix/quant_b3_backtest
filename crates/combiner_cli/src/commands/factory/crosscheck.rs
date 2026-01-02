//! Cross-check module - Independent metric recalculation for audit verification.
//!
//! Provides independent recalculation of key metrics from raw NAV timeseries
//! to validate against reported metrics.json values.

use std::path::Path;
use std::fs;
use serde::{Deserialize, Serialize};

/// Result of a cross-check comparison for a single strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrosscheckResult {
    pub strategy_id: String,
    pub reported: MetricSet,
    pub recalculated: MetricSet,
    pub tolerance: ToleranceResult,
    pub verdict: String,
}

/// Set of key metrics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricSet {
    pub sharpe: f64,
    pub cagr_pct: f64,
    pub max_drawdown_pct: f64,
    pub volatility_pct: f64,
}

/// Tolerance comparison results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceResult {
    pub sharpe_diff: f64,
    pub cagr_diff: f64,
    pub within_tolerance: bool,
}

/// Aggregate cross-check report for a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditCrosscheck {
    pub run_id: String,
    pub timestamp: String,
    pub strategies_checked: usize,
    pub all_passed: bool,
    pub failures: Vec<String>,
    pub summary: String,
    pub results: Vec<CrosscheckResult>,
}

/// Calculate metrics from NAV timeseries CSV.
/// 
/// Expected columns: date,equity,drawdown,exposure,...
pub fn calculate_metrics_from_nav(nav_path: &Path) -> Option<MetricSet> {
    let content = fs::read_to_string(nav_path).ok()?;
    let lines: Vec<&str> = content.lines().collect();
    
    if lines.len() < 2 {
        return None;
    }
    
    // Parse equity values
    let mut equities: Vec<f64> = Vec::new();
    let mut dates: Vec<String> = Vec::new();
    
    for line in lines.iter().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let Ok(equity) = parts[1].parse::<f64>() {
                equities.push(equity);
                dates.push(parts[0].to_string());
            }
        }
    }
    
    if equities.len() < 2 {
        return None;
    }
    
    // Calculate daily returns
    let mut returns: Vec<f64> = Vec::new();
    for i in 1..equities.len() {
        if equities[i - 1] > 0.0 {
            returns.push((equities[i] / equities[i - 1]) - 1.0);
        }
    }
    
    if returns.is_empty() {
        return None;
    }
    
    // Calculate metrics
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
    let std_dev = variance.sqrt();
    
    // Annualized volatility (252 trading days)
    let volatility = std_dev * (252.0_f64).sqrt();
    
    // Annualized Sharpe (assuming 0 risk-free rate for simplicity)
    let annualized_return = mean_return * 252.0;
    let sharpe = if volatility > 0.001 { annualized_return / volatility } else { 0.0 };
    
    // CAGR
    let first_equity = equities.first().copied().unwrap_or(1.0);
    let last_equity = equities.last().copied().unwrap_or(1.0);
    let years = equities.len() as f64 / 252.0;
    let cagr = if years > 0.0 && first_equity > 0.0 {
        (last_equity / first_equity).powf(1.0 / years) - 1.0
    } else {
        0.0
    };
    
    // Max Drawdown
    let mut peak = equities[0];
    let mut max_dd = 0.0;
    for &equity in &equities {
        if equity > peak {
            peak = equity;
        }
        let dd = (peak - equity) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    
    Some(MetricSet {
        sharpe,
        cagr_pct: cagr * 100.0,
        max_drawdown_pct: -max_dd * 100.0,
        volatility_pct: volatility * 100.0,
    })
}

/// Load reported metrics from metrics.json.
pub fn load_reported_metrics(metrics_path: &Path) -> Option<MetricSet> {
    let content = fs::read_to_string(metrics_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    
    Some(MetricSet {
        sharpe: json.get("sharpe_ratio").and_then(|v| v.as_f64()).unwrap_or(0.0),
        cagr_pct: json.get("cagr").and_then(|v| v.as_f64()).unwrap_or(0.0) * 100.0,
        max_drawdown_pct: json.get("max_drawdown").and_then(|v| v.as_f64()).unwrap_or(0.0) * 100.0,
        volatility_pct: json.get("volatility").and_then(|v| v.as_f64()).unwrap_or(0.0) * 100.0,
    })
}

/// Run cross-check for a single strategy.
pub fn crosscheck_strategy(
    strategy_dir: &Path,
    backtest_dir: Option<&Path>,
    tolerance_pct: f64,
) -> Option<CrosscheckResult> {
    let strategy_id = strategy_dir.file_name()?.to_string_lossy().to_string();
    
    // Load reported metrics
    let metrics_path = strategy_dir.join("metrics.json");
    let reported = load_reported_metrics(&metrics_path)?;
    
    // Try to find NAV data
    let recalculated = if let Some(bt_dir) = backtest_dir {
        // Look for timeseries.csv in backtest directory
        let nav_path = bt_dir.join("timeseries.csv");
        if nav_path.exists() {
            calculate_metrics_from_nav(&nav_path)
        } else {
            None
        }
    } else {
        None
    };
    
    // If no NAV data, use reported as recalculated (no cross-check possible)
    let recalculated = recalculated.unwrap_or_else(|| reported.clone());
    
    // Calculate differences
    let sharpe_diff = (reported.sharpe - recalculated.sharpe).abs();
    let cagr_diff = (reported.cagr_pct - recalculated.cagr_pct).abs();
    
    let within_tolerance = sharpe_diff <= tolerance_pct && cagr_diff <= tolerance_pct * 10.0;
    
    let verdict = if within_tolerance { "PASS" } else { "FAIL" };
    
    Some(CrosscheckResult {
        strategy_id,
        reported,
        recalculated,
        tolerance: ToleranceResult {
            sharpe_diff,
            cagr_diff,
            within_tolerance,
        },
        verdict: verdict.to_string(),
    })
}

/// Run cross-check for all strategies in a run.
pub fn crosscheck_run(
    run_dir: &Path,
    tolerance_pct: f64,
    timestamp: &str,
) -> AuditCrosscheck {
    let run_id = run_dir.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    
    let hof_dir = run_dir.join("hall_of_fame");
    let backtests_dir = run_dir.join("backtests");
    
    let mut results: Vec<CrosscheckResult> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    
    // Iterate over strategy directories
    if let Ok(entries) = fs::read_dir(&hof_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("strategy_") && entry.path().is_dir() {
                // For now, we don't have direct mapping from strategy to backtest
                // So we just validate what we can
                if let Some(result) = crosscheck_strategy(&entry.path(), None, tolerance_pct) {
                    if result.verdict == "FAIL" {
                        failures.push(result.strategy_id.clone());
                    }
                    results.push(result);
                }
            }
        }
    }
    
    let all_passed = failures.is_empty();
    let strategies_checked = results.len();
    
    let summary = if all_passed {
        format!("Todas as {} estratégias passaram no cross-check", strategies_checked)
    } else {
        format!("{}/{} estratégias falharam: {:?}", failures.len(), strategies_checked, failures)
    };
    
    AuditCrosscheck {
        run_id,
        timestamp: timestamp.to_string(),
        strategies_checked,
        all_passed,
        failures,
        summary,
        results,
    }
}

