//! Baseline Aggregator for Monitoring - collects and analyzes historical check results.
//!
//! Aggregates results from multiple days to calculate:
//! - Alert rates (% runs with WARN/CRIT)
//! - Top checks by frequency
//! - Category breakdown
//! - Circuit breaker statistics
//! - Threshold recommendations

use chrono::NaiveDate;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write;

use crate::filters::Market;
use super::circuit_breaker::CircuitState;
use super::statistics::calculate_percentile;
use super::types::{CheckCategory, CheckResult, CircuitAction, MonitoringReport, Severity};

/// Result of monitoring for a single day.
#[derive(Debug, Clone)]
pub struct DailyResult {
    pub date: NaiveDate,
    pub market: Option<Market>,
    pub info_count: usize,
    pub warn_count: usize,
    pub crit_count: usize,
    pub halt_count: usize,
    pub circuit_state: CircuitState,
    pub action: CircuitAction,
    pub check_details: Vec<CheckResult>,
}

impl DailyResult {
    /// Create from a MonitoringReport.
    pub fn from_report(date: NaiveDate, report: &MonitoringReport) -> Self {
        Self {
            date,
            market: None,
            info_count: report.summary.passed,
            warn_count: report.summary.warnings,
            crit_count: report.summary.criticals,
            halt_count: report.summary.halts,
            circuit_state: if report.circuit_breaker.state == "Open" {
                CircuitState::Open
            } else if report.circuit_breaker.state == "HalfOpen" {
                CircuitState::HalfOpen
            } else {
                CircuitState::Closed
            },
            action: report.action.clone(),
            check_details: report.results.clone(),
        }
    }

    /// Check if this day had any critical issues.
    pub fn has_crit(&self) -> bool {
        self.crit_count > 0 || self.halt_count > 0
    }

    /// Check if this day had any warnings.
    pub fn has_warn(&self) -> bool {
        self.warn_count > 0
    }
}

/// Frequency data for a specific check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckFrequency {
    pub check_name: String,
    pub category: CheckCategory,
    pub trigger_count: usize,
    pub trigger_pct: Decimal,
    pub severity: Severity,
    pub example_message: String,
}

/// Category breakdown statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CategoryBreakdown {
    pub data_health_crit_pct: Decimal,
    pub data_health_warn_pct: Decimal,
    pub drift_crit_pct: Decimal,
    pub drift_warn_pct: Decimal,
    pub regression_crit_pct: Decimal,
    pub regression_warn_pct: Decimal,
    pub data_health_top_causes: Vec<String>,
    pub drift_top_causes: Vec<String>,
    pub regression_top_causes: Vec<String>,
}

/// Circuit breaker statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CircuitBreakerStats {
    pub half_open_count: usize,
    pub open_count: usize,
    pub no_trade_count: usize,
    pub halt_count: usize,
    pub top_halt_causes: Vec<String>,
}

/// Threshold recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdRecommendation {
    pub severity_level: u8, // 1=critical, 2=high, 3=medium
    pub check_name: String,
    pub config_key: String,
    pub direction: String, // "increase" or "decrease"
    pub rationale: String,
    pub current_trigger_pct: Decimal,
    pub target_trigger_pct: Decimal,
}

/// Complete baseline report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineReport {
    pub total_days: usize,
    pub start_date: Option<NaiveDate>,
    pub end_date: Option<NaiveDate>,
    pub markets_analyzed: Vec<String>,
    
    // Alert rates
    pub pct_runs_with_warn: Decimal,
    pub pct_runs_with_crit: Decimal,
    pub avg_warns_per_run: Decimal,
    pub avg_crits_per_run: Decimal,
    pub p95_warns_per_run: Decimal,
    pub p95_crits_per_run: Decimal,
    
    // Top checks
    pub top_warn_checks: Vec<CheckFrequency>,
    pub top_crit_checks: Vec<CheckFrequency>,
    
    // Category breakdown
    pub category_breakdown: CategoryBreakdown,
    
    // Circuit breaker
    pub circuit_breaker_stats: CircuitBreakerStats,
    
    // Recommendations
    pub threshold_recommendations: Vec<ThresholdRecommendation>,
    
    // Limitations
    pub limitations: Vec<String>,
}

impl Default for BaselineReport {
    fn default() -> Self {
        Self {
            total_days: 0,
            start_date: None,
            end_date: None,
            markets_analyzed: vec![],
            pct_runs_with_warn: dec!(0),
            pct_runs_with_crit: dec!(0),
            avg_warns_per_run: dec!(0),
            avg_crits_per_run: dec!(0),
            p95_warns_per_run: dec!(0),
            p95_crits_per_run: dec!(0),
            top_warn_checks: vec![],
            top_crit_checks: vec![],
            category_breakdown: CategoryBreakdown::default(),
            circuit_breaker_stats: CircuitBreakerStats::default(),
            threshold_recommendations: vec![],
            limitations: vec![],
        }
    }
}

impl BaselineReport {
    /// Generate markdown summary.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        // A) RESUMO EXECUTIVO
        writeln!(md, "## A) RESUMO EXECUTIVO\n").unwrap();
        writeln!(md, "- **Periodo**: {} dias analisados ({} a {})",
            self.total_days,
            self.start_date.map(|d| d.to_string()).unwrap_or("N/A".to_string()),
            self.end_date.map(|d| d.to_string()).unwrap_or("N/A".to_string())
        ).unwrap();
        writeln!(md, "- **Mercados**: {}", self.markets_analyzed.join(", ")).unwrap();
        writeln!(md, "- **Runs com WARN**: {:.1}%", self.pct_runs_with_warn).unwrap();
        writeln!(md, "- **Runs com CRIT**: {:.1}%", self.pct_runs_with_crit).unwrap();
        writeln!(md, "- **Circuit Breaker HALT**: {} vezes", self.circuit_breaker_stats.halt_count).unwrap();
        writeln!(md).unwrap();

        // B) TABELA DE TAXAS
        writeln!(md, "## B) TABELA DE TAXAS (GERAL)\n").unwrap();
        writeln!(md, "| Metrica | Valor |").unwrap();
        writeln!(md, "|---------|-------|").unwrap();
        writeln!(md, "| % runs com WARN | {:.1}% |", self.pct_runs_with_warn).unwrap();
        writeln!(md, "| % runs com CRIT | {:.1}% |", self.pct_runs_with_crit).unwrap();
        writeln!(md, "| Media WARNs/run | {:.2} |", self.avg_warns_per_run).unwrap();
        writeln!(md, "| p95 WARNs/run | {:.2} |", self.p95_warns_per_run).unwrap();
        writeln!(md, "| Media CRITs/run | {:.2} |", self.avg_crits_per_run).unwrap();
        writeln!(md, "| p95 CRITs/run | {:.2} |", self.p95_crits_per_run).unwrap();
        writeln!(md).unwrap();

        // C) TOP CHECKS
        writeln!(md, "## C) TOP CHECKS\n").unwrap();
        
        writeln!(md, "### Top 10 WARN\n").unwrap();
        writeln!(md, "| Check | % Runs | Categoria | Exemplo |").unwrap();
        writeln!(md, "|-------|--------|-----------|---------|").unwrap();
        for check in self.top_warn_checks.iter().take(10) {
            writeln!(md, "| {} | {:.1}% | {} | {} |",
                check.check_name, check.trigger_pct, check.category,
                truncate(&check.example_message, 40)
            ).unwrap();
        }
        writeln!(md).unwrap();

        writeln!(md, "### Top 10 CRIT\n").unwrap();
        writeln!(md, "| Check | % Runs | Categoria | Exemplo |").unwrap();
        writeln!(md, "|-------|--------|-----------|---------|").unwrap();
        for check in self.top_crit_checks.iter().take(10) {
            writeln!(md, "| {} | {:.1}% | {} | {} |",
                check.check_name, check.trigger_pct, check.category,
                truncate(&check.example_message, 40)
            ).unwrap();
        }
        writeln!(md).unwrap();

        // D) CRIT POR CATEGORIA
        writeln!(md, "## D) CRIT POR CATEGORIA\n").unwrap();
        writeln!(md, "| Categoria | % runs CRIT | % runs WARN | Principais Causadores |").unwrap();
        writeln!(md, "|-----------|-------------|-------------|----------------------|").unwrap();
        writeln!(md, "| DataHealth | {:.1}% | {:.1}% | {} |",
            self.category_breakdown.data_health_crit_pct,
            self.category_breakdown.data_health_warn_pct,
            self.category_breakdown.data_health_top_causes.join(", ")
        ).unwrap();
        writeln!(md, "| Drift | {:.1}% | {:.1}% | {} |",
            self.category_breakdown.drift_crit_pct,
            self.category_breakdown.drift_warn_pct,
            self.category_breakdown.drift_top_causes.join(", ")
        ).unwrap();
        writeln!(md, "| Regression | {:.1}% | {:.1}% | {} |",
            self.category_breakdown.regression_crit_pct,
            self.category_breakdown.regression_warn_pct,
            self.category_breakdown.regression_top_causes.join(", ")
        ).unwrap();
        writeln!(md).unwrap();

        // E) CIRCUIT BREAKER
        writeln!(md, "## E) CIRCUIT BREAKER\n").unwrap();
        writeln!(md, "| Estado | Contagem |").unwrap();
        writeln!(md, "|--------|----------|").unwrap();
        writeln!(md, "| HALF_OPEN | {} |", self.circuit_breaker_stats.half_open_count).unwrap();
        writeln!(md, "| OPEN | {} |", self.circuit_breaker_stats.open_count).unwrap();
        writeln!(md, "| NO_TRADE | {} |", self.circuit_breaker_stats.no_trade_count).unwrap();
        writeln!(md, "| HALT | {} |", self.circuit_breaker_stats.halt_count).unwrap();
        writeln!(md).unwrap();
        if !self.circuit_breaker_stats.top_halt_causes.is_empty() {
            writeln!(md, "**Top causas de HALT**: {}", 
                self.circuit_breaker_stats.top_halt_causes.join(", ")).unwrap();
        }
        writeln!(md).unwrap();

        // F) RECOMENDACOES
        writeln!(md, "## F) RECOMENDACOES DE AJUSTE\n").unwrap();
        writeln!(md, "| SEV | Check | Config | Direcao | Racional |").unwrap();
        writeln!(md, "|-----|-------|--------|---------|----------|").unwrap();
        for rec in &self.threshold_recommendations {
            writeln!(md, "| {} | {} | `{}` | {} | {} |",
                rec.severity_level, rec.check_name, rec.config_key,
                rec.direction, truncate(&rec.rationale, 50)
            ).unwrap();
        }
        writeln!(md).unwrap();

        // G) LIMITACOES
        writeln!(md, "## G) LIMITACOES\n").unwrap();
        for limitation in &self.limitations {
            writeln!(md, "- {}", limitation).unwrap();
        }

        md
    }
}

/// Aggregator that collects daily results and generates baseline report.
pub struct BaselineAggregator {
    daily_results: Vec<DailyResult>,
    markets: Vec<Market>,
}

impl BaselineAggregator {
    pub fn new() -> Self {
        Self {
            daily_results: vec![],
            markets: vec![],
        }
    }

    /// Add a daily result.
    pub fn add(&mut self, result: DailyResult) {
        self.daily_results.push(result);
    }

    /// Set markets analyzed.
    pub fn set_markets(&mut self, markets: Vec<Market>) {
        self.markets = markets;
    }

    /// Get number of days collected.
    pub fn days_collected(&self) -> usize {
        self.daily_results.len()
    }

    /// Generate the baseline report.
    pub fn generate_report(&self) -> BaselineReport {
        let n = self.daily_results.len();
        if n == 0 {
            return BaselineReport {
                limitations: vec!["Nenhum dia analisado".to_string()],
                ..Default::default()
            };
        }

        let n_dec = Decimal::from(n);

        // Calculate alert rates
        let runs_with_warn = self.daily_results.iter().filter(|r| r.has_warn()).count();
        let runs_with_crit = self.daily_results.iter().filter(|r| r.has_crit()).count();

        let warn_counts: Vec<Decimal> = self.daily_results.iter()
            .map(|r| Decimal::from(r.warn_count))
            .collect();
        let crit_counts: Vec<Decimal> = self.daily_results.iter()
            .map(|r| Decimal::from(r.crit_count))
            .collect();

        let avg_warns = warn_counts.iter().sum::<Decimal>() / n_dec;
        let avg_crits = crit_counts.iter().sum::<Decimal>() / n_dec;
        let p95_warns = calculate_percentile(&warn_counts, dec!(95)).unwrap_or(dec!(0));
        let p95_crits = calculate_percentile(&crit_counts, dec!(95)).unwrap_or(dec!(0));

        // Collect check frequencies
        let mut warn_freq: HashMap<String, (usize, CheckCategory, String)> = HashMap::new();
        let mut crit_freq: HashMap<String, (usize, CheckCategory, String)> = HashMap::new();

        for result in &self.daily_results {
            for check in &result.check_details {
                match check.severity {
                    Severity::Warn => {
                        let entry = warn_freq.entry(check.check_name.clone())
                            .or_insert((0, check.category.clone(), check.message.clone()));
                        entry.0 += 1;
                    }
                    Severity::Crit | Severity::Halt => {
                        let entry = crit_freq.entry(check.check_name.clone())
                            .or_insert((0, check.category.clone(), check.message.clone()));
                        entry.0 += 1;
                    }
                    _ => {}
                }
            }
        }

        // Convert to sorted vectors
        let mut top_warn: Vec<CheckFrequency> = warn_freq.into_iter()
            .map(|(name, (count, cat, msg))| CheckFrequency {
                check_name: name,
                category: cat,
                trigger_count: count,
                trigger_pct: Decimal::from(count * 100) / n_dec,
                severity: Severity::Warn,
                example_message: msg,
            })
            .collect();
        top_warn.sort_by(|a, b| b.trigger_count.cmp(&a.trigger_count));

        let mut top_crit: Vec<CheckFrequency> = crit_freq.into_iter()
            .map(|(name, (count, cat, msg))| CheckFrequency {
                check_name: name,
                category: cat,
                trigger_count: count,
                trigger_pct: Decimal::from(count * 100) / n_dec,
                severity: Severity::Crit,
                example_message: msg,
            })
            .collect();
        top_crit.sort_by(|a, b| b.trigger_count.cmp(&a.trigger_count));

        // Category breakdown
        let category_breakdown = self.calculate_category_breakdown(&top_warn, &top_crit, n);

        // Circuit breaker stats
        let circuit_stats = self.calculate_circuit_stats();

        // Recommendations
        let recommendations = self.generate_recommendations(&top_warn, &top_crit, n);

        // Limitations
        let mut limitations = vec![];
        if n < 30 {
            limitations.push(format!("Apenas {} dias analisados (meta: 30)", n));
        }
        if self.markets.contains(&Market::US) {
            limitations.push("US Market: apenas 10 simbolos disponiveis".to_string());
        }
        limitations.push("Drift/Regression: dados sinteticos (sem historico de backtest)".to_string());

        BaselineReport {
            total_days: n,
            start_date: self.daily_results.first().map(|r| r.date),
            end_date: self.daily_results.last().map(|r| r.date),
            markets_analyzed: self.markets.iter().map(|m| format!("{:?}", m)).collect(),
            pct_runs_with_warn: Decimal::from(runs_with_warn * 100) / n_dec,
            pct_runs_with_crit: Decimal::from(runs_with_crit * 100) / n_dec,
            avg_warns_per_run: avg_warns,
            avg_crits_per_run: avg_crits,
            p95_warns_per_run: p95_warns,
            p95_crits_per_run: p95_crits,
            top_warn_checks: top_warn,
            top_crit_checks: top_crit,
            category_breakdown,
            circuit_breaker_stats: circuit_stats,
            threshold_recommendations: recommendations,
            limitations,
        }
    }

    fn calculate_category_breakdown(
        &self,
        _top_warn: &[CheckFrequency],
        top_crit: &[CheckFrequency],
        n: usize,
    ) -> CategoryBreakdown {
        let n_dec = Decimal::from(n);
        let mut breakdown = CategoryBreakdown::default();

        // Count days with CRIT/WARN per category
        let mut dh_crit_days = 0;
        let mut dh_warn_days = 0;
        let mut drift_crit_days = 0;
        let mut drift_warn_days = 0;
        let mut reg_crit_days = 0;
        let mut reg_warn_days = 0;

        for result in &self.daily_results {
            let mut has_dh_crit = false;
            let mut has_dh_warn = false;
            let mut has_drift_crit = false;
            let mut has_drift_warn = false;
            let mut has_reg_crit = false;
            let mut has_reg_warn = false;

            for check in &result.check_details {
                match (&check.category, &check.severity) {
                    (CheckCategory::DataHealth, Severity::Crit | Severity::Halt) => has_dh_crit = true,
                    (CheckCategory::DataHealth, Severity::Warn) => has_dh_warn = true,
                    (CheckCategory::Drift, Severity::Crit | Severity::Halt) => has_drift_crit = true,
                    (CheckCategory::Drift, Severity::Warn) => has_drift_warn = true,
                    (CheckCategory::Regression, Severity::Crit | Severity::Halt) => has_reg_crit = true,
                    (CheckCategory::Regression, Severity::Warn) => has_reg_warn = true,
                    _ => {}
                }
            }

            if has_dh_crit { dh_crit_days += 1; }
            if has_dh_warn { dh_warn_days += 1; }
            if has_drift_crit { drift_crit_days += 1; }
            if has_drift_warn { drift_warn_days += 1; }
            if has_reg_crit { reg_crit_days += 1; }
            if has_reg_warn { reg_warn_days += 1; }
        }

        breakdown.data_health_crit_pct = Decimal::from(dh_crit_days * 100) / n_dec;
        breakdown.data_health_warn_pct = Decimal::from(dh_warn_days * 100) / n_dec;
        breakdown.drift_crit_pct = Decimal::from(drift_crit_days * 100) / n_dec;
        breakdown.drift_warn_pct = Decimal::from(drift_warn_days * 100) / n_dec;
        breakdown.regression_crit_pct = Decimal::from(reg_crit_days * 100) / n_dec;
        breakdown.regression_warn_pct = Decimal::from(reg_warn_days * 100) / n_dec;

        // Top causes per category
        breakdown.data_health_top_causes = top_crit.iter()
            .filter(|c| c.category == CheckCategory::DataHealth)
            .take(3)
            .map(|c| c.check_name.clone())
            .collect();
        breakdown.drift_top_causes = top_crit.iter()
            .filter(|c| c.category == CheckCategory::Drift)
            .take(3)
            .map(|c| c.check_name.clone())
            .collect();
        breakdown.regression_top_causes = top_crit.iter()
            .filter(|c| c.category == CheckCategory::Regression)
            .take(3)
            .map(|c| c.check_name.clone())
            .collect();

        breakdown
    }

    fn calculate_circuit_stats(&self) -> CircuitBreakerStats {
        let mut stats = CircuitBreakerStats::default();
        let mut halt_causes: HashMap<String, usize> = HashMap::new();

        for result in &self.daily_results {
            match result.circuit_state {
                CircuitState::HalfOpen => stats.half_open_count += 1,
                CircuitState::Open => stats.open_count += 1,
                _ => {}
            }

            match result.action {
                CircuitAction::FlagNoTrade => stats.no_trade_count += 1,
                CircuitAction::HaltWithError => {
                    stats.halt_count += 1;
                    // Track which checks caused the halt
                    for check in &result.check_details {
                        if check.severity == Severity::Crit || check.severity == Severity::Halt {
                            *halt_causes.entry(check.check_name.clone()).or_insert(0) += 1;
                        }
                    }
                }
                _ => {}
            }
        }

        // Get top halt causes
        let mut causes: Vec<_> = halt_causes.into_iter().collect();
        causes.sort_by(|a, b| b.1.cmp(&a.1));
        stats.top_halt_causes = causes.into_iter().take(5).map(|(name, _)| name).collect();

        stats
    }

    fn generate_recommendations(
        &self,
        top_warn: &[CheckFrequency],
        top_crit: &[CheckFrequency],
        _n: usize,
    ) -> Vec<ThresholdRecommendation> {
        let mut recs = vec![];

        // Checks with >50% WARN are too sensitive
        for check in top_warn.iter().take(20) {
            if check.trigger_pct > dec!(50) {
                recs.push(ThresholdRecommendation {
                    severity_level: 2,
                    check_name: check.check_name.clone(),
                    config_key: infer_config_key(&check.check_name),
                    direction: "aumentar".to_string(),
                    rationale: format!("Dispara em {:.0}% dos runs - muito sensivel", check.trigger_pct),
                    current_trigger_pct: check.trigger_pct,
                    target_trigger_pct: dec!(20),
                });
            }
        }

        // Checks with >20% CRIT may need review
        for check in top_crit.iter().take(10) {
            if check.trigger_pct > dec!(20) {
                recs.push(ThresholdRecommendation {
                    severity_level: 1,
                    check_name: check.check_name.clone(),
                    config_key: infer_config_key(&check.check_name),
                    direction: "aumentar".to_string(),
                    rationale: format!("CRIT em {:.0}% - revisar se threshold faz sentido", check.trigger_pct),
                    current_trigger_pct: check.trigger_pct,
                    target_trigger_pct: dec!(5),
                });
            }
        }

        // Sort by severity
        recs.sort_by(|a, b| a.severity_level.cmp(&b.severity_level));
        recs
    }
}

impl Default for BaselineAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Infer config key from check name.
fn infer_config_key(check_name: &str) -> String {
    if check_name.contains("Freshness") {
        "data_health.freshness_max_days".to_string()
    } else if check_name.contains("Coverage") {
        "data_health.coverage_min_pct".to_string()
    } else if check_name.contains("Drawdown") {
        "regression.drawdown.warn_pct".to_string()
    } else if check_name.contains("Turnover") {
        "regression.turnover.hard_cap".to_string()
    } else if check_name.contains("Selection") {
        "drift.selection_overlap_warn".to_string()
    } else if check_name.contains("Dividend") {
        "data_health.dividends_min_30d".to_string()
    } else if check_name.contains("InterestRate") {
        "data_health.interest_rates_max_days".to_string()
    } else {
        format!("config.{}", check_name.to_lowercase().replace("_", "."))
    }
}

/// Truncate string for table display.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn test_daily_result() {
        let result = DailyResult {
            date: date(2024, 1, 10),
            market: Some(Market::BR),
            info_count: 10,
            warn_count: 2,
            crit_count: 1,
            halt_count: 0,
            circuit_state: CircuitState::Closed,
            action: CircuitAction::Continue,
            check_details: vec![],
        };

        assert!(result.has_warn());
        assert!(result.has_crit());
    }

    #[test]
    fn test_aggregator_empty() {
        let agg = BaselineAggregator::new();
        let report = agg.generate_report();

        assert_eq!(report.total_days, 0);
        assert!(!report.limitations.is_empty());
    }

    #[test]
    fn test_aggregator_single_day() {
        let mut agg = BaselineAggregator::new();
        agg.set_markets(vec![Market::BR]);
        agg.add(DailyResult {
            date: date(2024, 1, 10),
            market: Some(Market::BR),
            info_count: 5,
            warn_count: 2,
            crit_count: 1,
            halt_count: 0,
            circuit_state: CircuitState::Closed,
            action: CircuitAction::Continue,
            check_details: vec![
                CheckResult::warn("TestCheck", CheckCategory::DataHealth, "test warn"),
            ],
        });

        let report = agg.generate_report();

        assert_eq!(report.total_days, 1);
        assert_eq!(report.pct_runs_with_warn, dec!(100));
        assert_eq!(report.pct_runs_with_crit, dec!(100));
    }

    #[test]
    fn test_report_markdown() {
        let report = BaselineReport::default();
        let md = report.to_markdown();

        assert!(md.contains("## A) RESUMO EXECUTIVO"));
        assert!(md.contains("## B) TABELA DE TAXAS"));
        assert!(md.contains("## C) TOP CHECKS"));
    }

    #[test]
    fn test_infer_config_key() {
        assert_eq!(infer_config_key("Freshness_BR"), "data_health.freshness_max_days");
        assert_eq!(infer_config_key("Coverage_US"), "data_health.coverage_min_pct");
        assert_eq!(infer_config_key("DrawdownGuardrail"), "regression.drawdown.warn_pct");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("this is a long message", 10), "this is...");
    }
}

