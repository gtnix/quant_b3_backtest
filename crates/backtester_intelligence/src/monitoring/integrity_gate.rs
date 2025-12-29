//! Data Integrity Gate - Orchestrates integrity checks for Strategy Factory.
//!
//! Provides a unified gate that runs all data integrity checks and produces
//! a PASS/FAIL verdict used by factory run/resume/promote commands.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use crate::filters::Market;
use super::config::DataHealthConfig;
use super::data_health::{
    DataContext, DataHealthCheck,
    TemporalIntegrityCheck, LookaheadPolicyCheck, CorpActionCheck, SurvivorshipCheck,
};
use super::types::{CheckCategory, CheckResult, Severity};

// =============================================================================
// DATA INTEGRITY REPORT
// =============================================================================

/// Overall verdict for data integrity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    /// All critical checks passed
    Pass,
    /// One or more critical checks failed
    Fail,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Pass => write!(f, "PASS"),
            Verdict::Fail => write!(f, "FAIL"),
        }
    }
}

/// Audit mode - controls performance vs thoroughness tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AuditMode {
    /// Fast sampling-based audit (default)
    #[default]
    Fast,
    /// Strict full-scan audit
    Strict,
}

impl std::fmt::Display for AuditMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditMode::Fast => write!(f, "fast"),
            AuditMode::Strict => write!(f, "strict"),
        }
    }
}

impl AuditMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "strict" => AuditMode::Strict,
            _ => AuditMode::Fast,
        }
    }
}

/// Complete data integrity report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataIntegrityReport {
    /// Overall verdict (PASS/FAIL)
    pub verdict: Verdict,
    /// Confidence score (0.0 to 1.0)
    pub score: f64,
    /// Dataset hash for reproducibility
    pub dataset_hash: String,
    /// Market being audited
    pub market: String,
    /// Timezone used
    pub timezone: String,
    /// All check results
    pub checks: Vec<CheckResult>,
    /// Hard failure reasons (critical issues)
    pub hard_fails: Vec<String>,
    /// Warnings (non-blocking issues)
    pub warnings: Vec<String>,
    /// Audit statistics
    pub stats: AuditStats,
    /// Timestamp of audit
    pub created_at: DateTime<Utc>,
    /// Audit mode used
    pub audit_mode: String,
    /// Version for schema compatibility
    pub version: String,
}

/// Audit statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditStats {
    pub total_checks: usize,
    pub passed: usize,
    pub warnings: usize,
    pub critical: usize,
    pub duration_ms: u64,
}

impl Default for DataIntegrityReport {
    fn default() -> Self {
        Self {
            verdict: Verdict::Pass,
            score: 1.0,
            dataset_hash: String::new(),
            market: String::new(),
            timezone: String::new(),
            checks: Vec::new(),
            hard_fails: Vec::new(),
            warnings: Vec::new(),
            stats: AuditStats::default(),
            created_at: Utc::now(),
            audit_mode: "fast".to_string(),
            version: "1.0.0".to_string(),
        }
    }
}

impl DataIntegrityReport {
    /// Create a new empty report.
    pub fn new(dataset_hash: &str, market: &str) -> Self {
        Self {
            dataset_hash: dataset_hash.to_string(),
            market: market.to_string(),
            ..Default::default()
        }
    }

    /// Check if the report passed.
    pub fn passed(&self) -> bool {
        self.verdict == Verdict::Pass
    }

    /// Get a summary string.
    pub fn summary(&self) -> String {
        format!(
            "{}: {} checks, {} passed, {} warnings, {} critical (score: {:.2})",
            self.verdict,
            self.stats.total_checks,
            self.stats.passed,
            self.stats.warnings,
            self.stats.critical,
            self.score
        )
    }

    /// Save report to JSON file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, json)
    }

    /// Load report from JSON file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let json = fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

// =============================================================================
// DATA INTEGRITY GATE
// =============================================================================

/// Data Integrity Gate - orchestrates all integrity checks.
pub struct DataIntegrityGate {
    checks: Vec<Box<dyn DataHealthCheck>>,
    config: DataHealthConfig,
    mode: AuditMode,
    market: Market,
}

impl DataIntegrityGate {
    /// Create a new data integrity gate.
    pub fn new(market: Market, delay_bars: u8, max_gap_days: u32, mode: AuditMode) -> Self {
        let checks: Vec<Box<dyn DataHealthCheck>> = vec![
            Box::new(TemporalIntegrityCheck::new(market, max_gap_days)),
            Box::new(LookaheadPolicyCheck::new(delay_bars)),
            Box::new(CorpActionCheck::new(market, 30.0)),
            Box::new(SurvivorshipCheck::new(market)),
        ];

        Self {
            checks,
            config: DataHealthConfig::default(),
            mode,
            market,
        }
    }

    /// Create gate for B3 market with defaults.
    pub fn b3_default() -> Self {
        Self::new(Market::BR, 1, 5, AuditMode::Fast)
    }

    /// Create gate for US market with defaults.
    pub fn us_default() -> Self {
        Self::new(Market::US, 1, 5, AuditMode::Fast)
    }

    /// Run all integrity checks and produce report.
    pub fn audit(&self, ctx: &DataContext, dataset_hash: &str) -> DataIntegrityReport {
        let start = std::time::Instant::now();

        // Run all checks
        let results: Vec<CheckResult> = self.checks.iter()
            .map(|check| check.run(ctx, &self.config))
            .collect();

        // Categorize results
        let mut hard_fails = Vec::new();
        let mut warnings = Vec::new();
        let mut passed = 0;
        let mut critical = 0;
        let mut warn_count = 0;

        for result in &results {
            if result.category == CheckCategory::DataIntegrity {
                match result.severity {
                    Severity::Crit | Severity::Halt => {
                        hard_fails.push(result.message.clone());
                        critical += 1;
                    }
                    Severity::Warn => {
                        warnings.push(result.message.clone());
                        warn_count += 1;
                    }
                    Severity::Info => {
                        if result.passed {
                            passed += 1;
                        }
                    }
                }
            } else if result.passed {
                passed += 1;
            }
        }

        // Determine verdict
        let verdict = if hard_fails.is_empty() {
            Verdict::Pass
        } else {
            Verdict::Fail
        };

        // Calculate score (1.0 = perfect, 0.0 = all failed)
        let total = results.len();
        let score = if total > 0 {
            (passed as f64 / total as f64).max(0.0).min(1.0)
        } else {
            1.0
        };

        let duration = start.elapsed();

        DataIntegrityReport {
            verdict,
            score,
            dataset_hash: dataset_hash.to_string(),
            market: format!("{:?}", self.market),
            timezone: self.market_timezone(),
            checks: results,
            hard_fails,
            warnings,
            stats: AuditStats {
                total_checks: total,
                passed,
                warnings: warn_count,
                critical,
                duration_ms: duration.as_millis() as u64,
            },
            created_at: Utc::now(),
            audit_mode: self.mode.to_string(),
            version: "1.0.0".to_string(),
        }
    }

    /// Get timezone string for market.
    fn market_timezone(&self) -> String {
        match self.market {
            Market::BR => "America/Sao_Paulo".to_string(),
            Market::US => "America/New_York".to_string(),
        }
    }

    /// Get the audit mode.
    pub fn mode(&self) -> AuditMode {
        self.mode
    }

    /// Get the market.
    pub fn market(&self) -> Market {
        self.market
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use std::collections::HashMap;

    fn date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap()
    }

    #[test]
    fn test_verdict_display() {
        assert_eq!(format!("{}", Verdict::Pass), "PASS");
        assert_eq!(format!("{}", Verdict::Fail), "FAIL");
    }

    #[test]
    fn test_audit_mode_from_str() {
        assert_eq!(AuditMode::from_str("fast"), AuditMode::Fast);
        assert_eq!(AuditMode::from_str("strict"), AuditMode::Strict);
        assert_eq!(AuditMode::from_str("STRICT"), AuditMode::Strict);
        assert_eq!(AuditMode::from_str("unknown"), AuditMode::Fast);
    }

    #[test]
    fn test_gate_creation() {
        let gate = DataIntegrityGate::b3_default();
        assert_eq!(gate.market(), Market::BR);
        assert_eq!(gate.mode(), AuditMode::Fast);
    }

    #[test]
    fn test_audit_empty_context() {
        let gate = DataIntegrityGate::b3_default();
        let ctx = DataContext::new(date(2024, 1, 10));
        
        let report = gate.audit(&ctx, "test_hash");
        
        // Should pass with no issues in empty context
        assert_eq!(report.verdict, Verdict::Pass);
        assert!(report.hard_fails.is_empty());
    }

    #[test]
    fn test_audit_with_duplicates() {
        let gate = DataIntegrityGate::b3_default();
        let mut ctx = DataContext::new(date(2024, 1, 10));
        ctx.duplicate_count.insert(Market::BR, 5);
        
        let report = gate.audit(&ctx, "test_hash");
        
        // Should fail due to duplicates
        assert_eq!(report.verdict, Verdict::Fail);
        assert!(!report.hard_fails.is_empty());
    }

    #[test]
    fn test_audit_with_lookahead_violation() {
        let gate = DataIntegrityGate::new(Market::BR, 1, 5, AuditMode::Fast);
        let mut ctx = DataContext::new(date(2024, 1, 10));
        ctx.delay_bars_policy = 0; // Violation: should be >= 1
        
        let report = gate.audit(&ctx, "test_hash");
        
        // Should fail due to lookahead violation
        assert_eq!(report.verdict, Verdict::Fail);
        assert!(report.hard_fails.iter().any(|f| f.contains("Lookahead")));
    }

    #[test]
    fn test_report_save_load() {
        let report = DataIntegrityReport::new("hash123", "BR");
        
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test_integrity_report.json");
        
        report.save(&path).unwrap();
        
        let loaded = DataIntegrityReport::load(&path).unwrap();
        assert_eq!(loaded.dataset_hash, "hash123");
        assert_eq!(loaded.market, "BR");
    }
}
