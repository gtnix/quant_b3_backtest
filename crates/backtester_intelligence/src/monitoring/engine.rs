//! Monitoring Engine - Orchestrator for all checks.
//!
//! Runs data health, drift, and regression checks in sequence,
//! evaluates circuit breaker, and generates report.

use chrono::{NaiveDate, Utc};
use std::time::Instant;

use crate::filters::Market;
use super::circuit_breaker::CircuitBreaker;
use super::config::MonitoringConfig;
use super::data_health::{DataContext, DataHealthEngine};
use super::drift::{DriftContext, DriftEngine};
use super::regressions::{RegressionContext, RegressionEngine};
use super::types::{CheckResult, CircuitAction, MonitoringReport, MonitoringSummary};

/// Combined context for all monitoring checks.
#[derive(Debug, Clone, Default)]
pub struct MonitoringContext {
    /// Data health context
    pub data: DataContext,
    /// Drift detection context
    pub drift: DriftContext,
    /// Regression detection context
    pub regression: RegressionContext,
    /// Markets to monitor
    pub markets: Vec<Market>,
    /// Techniques to monitor for drift
    pub techniques: Vec<String>,
    /// Reference date
    pub as_of: NaiveDate,
}

impl MonitoringContext {
    pub fn new(as_of: NaiveDate) -> Self {
        Self {
            data: DataContext::new(as_of),
            drift: DriftContext::new(as_of),
            regression: RegressionContext::new(as_of),
            markets: vec![Market::BR, Market::US],
            techniques: vec![
                "Momentum".to_string(),
                "Value".to_string(),
                "Quality".to_string(),
                "Size".to_string(),
                "LowVol".to_string(),
                "Dividend".to_string(),
                "Carry".to_string(),
            ],
            as_of,
        }
    }

    pub fn with_markets(mut self, markets: Vec<Market>) -> Self {
        self.markets = markets;
        self
    }

    pub fn with_techniques(mut self, techniques: Vec<String>) -> Self {
        self.techniques = techniques;
        self
    }
}

/// Main monitoring engine.
pub struct MonitoringEngine {
    config: MonitoringConfig,
    data_health: DataHealthEngine,
    drift: DriftEngine,
    regressions: RegressionEngine,
    circuit_breaker: CircuitBreaker,
}

impl MonitoringEngine {
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            data_health: DataHealthEngine::new(&[Market::BR, Market::US]),
            drift: DriftEngine::default(),
            regressions: RegressionEngine::new(),
            circuit_breaker: CircuitBreaker::new(config.circuit_breaker.clone()),
            config,
        }
    }

    /// Create engine with custom markets and techniques.
    pub fn with_context(config: MonitoringConfig, markets: &[Market], techniques: &[String]) -> Self {
        Self {
            data_health: DataHealthEngine::new(markets),
            drift: DriftEngine::new(techniques),
            regressions: RegressionEngine::new(),
            circuit_breaker: CircuitBreaker::new(config.circuit_breaker.clone()),
            config,
        }
    }

    /// Run all monitoring checks.
    pub fn run_all(&mut self, ctx: &MonitoringContext) -> MonitoringReport {
        let start = Instant::now();
        let mut results = Vec::new();

        // Phase 1: Data Health Checks (required for other checks)
        let data_results = self.data_health.run_all(&ctx.data, &self.config.data_health);
        results.extend(data_results);

        // Phase 2: Drift Detection
        let drift_results = self.drift.run_all(&ctx.drift, &self.config.drift);
        results.extend(drift_results);

        // Phase 3: Performance Regressions
        let regression_results = self.regressions.run_all(&ctx.regression, &self.config.regression);
        results.extend(regression_results);

        // Phase 4: Circuit Breaker Evaluation
        let action = self.circuit_breaker.evaluate(&results);
        let cb_state = self.circuit_breaker.to_state();

        // Build report
        let summary = MonitoringSummary::from_results(&results);
        let no_trade = matches!(action, CircuitAction::FlagNoTrade | CircuitAction::HaltWithError);

        let _elapsed = start.elapsed();
        
        // Add latency to regression context if needed
        let final_results = results;
        
        MonitoringReport {
            timestamp: Utc::now(),
            results: final_results,
            summary,
            circuit_breaker: cb_state,
            action,
            no_trade,
            version: "1.0.0".to_string(),
        }
    }

    /// Run only data health checks.
    pub fn run_data_health(&self, ctx: &DataContext) -> Vec<CheckResult> {
        self.data_health.run_all(ctx, &self.config.data_health)
    }

    /// Run only drift checks.
    pub fn run_drift(&self, ctx: &DriftContext) -> Vec<CheckResult> {
        self.drift.run_all(ctx, &self.config.drift)
    }

    /// Run only regression checks.
    pub fn run_regressions(&self, ctx: &RegressionContext) -> Vec<CheckResult> {
        self.regressions.run_all(ctx, &self.config.regression)
    }

    /// Get current circuit breaker state.
    pub fn circuit_breaker_state(&self) -> super::types::CircuitBreakerState {
        self.circuit_breaker.to_state()
    }

    /// Reset circuit breaker (manual intervention).
    pub fn reset_circuit_breaker(&mut self) {
        self.circuit_breaker.reset();
    }

    /// Get configuration.
    pub fn config(&self) -> &MonitoringConfig {
        &self.config
    }
}

impl Default for MonitoringEngine {
    fn default() -> Self {
        Self::new(MonitoringConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    use std::collections::HashMap;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn test_engine_default() {
        let engine = MonitoringEngine::default();
        assert!(engine.config().circuit_breaker.enabled);
    }

    #[test]
    fn test_run_all_empty_context() {
        let mut engine = MonitoringEngine::default();
        let ctx = MonitoringContext::new(date(2024, 1, 10));
        
        let report = engine.run_all(&ctx);
        
        assert!(!report.results.is_empty());
        assert_eq!(report.version, "1.0.0");
    }

    #[test]
    fn test_run_all_healthy_data() {
        let mut engine = MonitoringEngine::default();
        let mut ctx = MonitoringContext::new(date(2024, 1, 10));
        
        // Set up healthy data context
        ctx.data.last_ohlcv_date.insert(Market::BR, date(2024, 1, 9));
        ctx.data.last_ohlcv_date.insert(Market::US, date(2024, 1, 9));
        ctx.data.symbol_count.insert(Market::BR, 100);
        ctx.data.symbols_with_data.insert(Market::BR, 95);
        ctx.data.symbol_count.insert(Market::US, 500);
        ctx.data.symbols_with_data.insert(Market::US, 480);
        ctx.data.schema_valid = true;
        ctx.data.dividends_30d = 50;
        ctx.data.last_interest_rate.insert(Market::BR, date(2024, 1, 9));
        ctx.data.last_interest_rate.insert(Market::US, date(2024, 1, 9));
        
        let report = engine.run_all(&ctx);
        
        // Most checks should pass
        let passed = report.results.iter().filter(|r| r.passed).count();
        assert!(passed > report.results.len() / 2);
    }

    #[test]
    fn test_run_all_stale_data() {
        let mut engine = MonitoringEngine::default();
        let mut ctx = MonitoringContext::new(date(2024, 1, 20));
        
        // Set up stale data
        ctx.data.last_ohlcv_date.insert(Market::BR, date(2024, 1, 1)); // 19 days ago
        ctx.data.symbol_count.insert(Market::BR, 100);
        ctx.data.symbols_with_data.insert(Market::BR, 95);
        
        let report = engine.run_all(&ctx);
        
        // Should have critical issues
        assert!(report.summary.criticals > 0);
        assert!(report.no_trade);
    }

    #[test]
    fn test_circuit_breaker_integration() {
        let mut engine = MonitoringEngine::default();
        let mut ctx = MonitoringContext::new(date(2024, 1, 10));
        
        // Trigger drawdown critical
        ctx.regression.current_drawdown = dec!(25);
        
        let report = engine.run_all(&ctx);
        
        // Should flag no trade due to drawdown
        assert!(report.summary.criticals > 0 || report.summary.halts > 0);
    }

    #[test]
    fn test_run_data_health_only() {
        let engine = MonitoringEngine::default();
        let mut ctx = DataContext::new(date(2024, 1, 10));
        ctx.last_ohlcv_date.insert(Market::BR, date(2024, 1, 9));
        ctx.symbol_count.insert(Market::BR, 100);
        ctx.symbols_with_data.insert(Market::BR, 95);
        
        let results = engine.run_data_health(&ctx);
        
        assert!(!results.is_empty());
    }

    #[test]
    fn test_context_builder() {
        let ctx = MonitoringContext::new(date(2024, 1, 10))
            .with_markets(vec![Market::BR])
            .with_techniques(vec!["Momentum".to_string()]);
        
        assert_eq!(ctx.markets.len(), 1);
        assert_eq!(ctx.techniques.len(), 1);
    }
}

