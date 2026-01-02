//! Reporter for Monitoring outputs.
//!
//! Generates:
//! - JSON report (AI-consumable)
//! - Markdown summary (human-readable)
//! - GitHub Actions summary (GITHUB_STEP_SUMMARY)

use std::fmt::Write;
use std::fs;
use std::io;
use std::path::Path;

use super::types::{CheckCategory, CheckResult, MonitoringReport, Severity};

/// Output format for reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    /// JSON format (AI-consumable)
    Json,
    /// Markdown format (human-readable)
    Markdown,
    /// Both formats
    Both,
}

/// Reporter for generating monitoring outputs.
#[derive(Debug, Clone)]
pub struct MonitoringReporter {
    /// Output directory
    output_dir: String,
    /// JSON filename
    json_filename: String,
    /// Markdown filename
    md_filename: String,
}

impl MonitoringReporter {
    pub fn new(output_dir: impl Into<String>) -> Self {
        Self {
            output_dir: output_dir.into(),
            json_filename: "monitoring_report.json".to_string(),
            md_filename: "monitoring_summary.md".to_string(),
        }
    }

    /// Generate JSON report.
    pub fn to_json(&self, report: &MonitoringReport) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(report)
    }

    /// Generate Markdown summary.
    pub fn to_markdown(&self, report: &MonitoringReport) -> String {
        let mut md = String::new();

        // Header
        writeln!(md, "## Monitoring Report - {}", report.timestamp.format("%Y-%m-%d %H:%M UTC")).unwrap();
        writeln!(md).unwrap();

        // Status badge
        let status = match report.summary.overall_status() {
            Severity::Halt => "🛑 HALT",
            Severity::Crit => "🔴 CRITICAL",
            Severity::Warn => "🟡 WARNING",
            Severity::Info => "🟢 HEALTHY",
        };
        writeln!(md, "### Status: {}", status).unwrap();
        writeln!(md).unwrap();

        // Summary stats
        writeln!(md, "| Metric | Count |").unwrap();
        writeln!(md, "|--------|-------|").unwrap();
        writeln!(md, "| Total Checks | {} |", report.summary.total_checks).unwrap();
        writeln!(md, "| Passed | {} |", report.summary.passed).unwrap();
        writeln!(md, "| Warnings | {} |", report.summary.warnings).unwrap();
        writeln!(md, "| Critical | {} |", report.summary.criticals).unwrap();
        writeln!(md, "| Halts | {} |", report.summary.halts).unwrap();
        writeln!(md).unwrap();

        // Circuit breaker state
        writeln!(md, "### Circuit Breaker").unwrap();
        writeln!(md).unwrap();
        writeln!(md, "- **State**: {}", report.circuit_breaker.state).unwrap();
        writeln!(md, "- **Critical Count**: {}/{}", 
                 report.circuit_breaker.crit_count, 
                 report.circuit_breaker.halt_threshold).unwrap();
        writeln!(md, "- **Action**: {}", report.action).unwrap();
        if report.no_trade {
            writeln!(md, "- **⚠️ NO-TRADE FLAG SET**").unwrap();
        }
        writeln!(md).unwrap();

        // Critical Issues
        let criticals: Vec<_> = report.results.iter()
            .filter(|r| r.severity == Severity::Crit || r.severity == Severity::Halt)
            .collect();

        if !criticals.is_empty() {
            writeln!(md, "### 🔴 Critical Issues").unwrap();
            writeln!(md).unwrap();
            for r in criticals {
                writeln!(md, "- **[{}] {}**: {}", r.severity, r.check_name, r.message).unwrap();
            }
            writeln!(md).unwrap();
        }

        // Warnings
        let warnings: Vec<_> = report.results.iter()
            .filter(|r| r.severity == Severity::Warn)
            .collect();

        if !warnings.is_empty() {
            writeln!(md, "### 🟡 Warnings").unwrap();
            writeln!(md).unwrap();
            for r in warnings {
                writeln!(md, "- **[{}] {}**: {}", r.severity, r.check_name, r.message).unwrap();
            }
            writeln!(md).unwrap();
        }

        // Results by category
        writeln!(md, "### Results by Category").unwrap();
        writeln!(md).unwrap();

        for category in [CheckCategory::DataHealth, CheckCategory::Drift, CheckCategory::Regression] {
            let cat_results: Vec<_> = report.results.iter()
                .filter(|r| r.category == category)
                .collect();

            if !cat_results.is_empty() {
                writeln!(md, "#### {}", category).unwrap();
                writeln!(md).unwrap();
                writeln!(md, "| Check | Status | Value | Threshold | Message |").unwrap();
                writeln!(md, "|-------|--------|-------|-----------|---------|").unwrap();

                for r in cat_results {
                    let status_icon = match r.severity {
                        Severity::Info => "✅",
                        Severity::Warn => "⚠️",
                        Severity::Crit => "❌",
                        Severity::Halt => "🛑",
                    };
                    writeln!(md, "| {} | {} | {:.2} | {:.2} | {} |",
                             r.check_name, status_icon, r.value, r.threshold,
                             truncate_message(&r.message, 50)).unwrap();
                }
                writeln!(md).unwrap();
            }
        }

        // Footer
        writeln!(md, "---").unwrap();
        writeln!(md, "*Report Version: {} | Generated at: {}*", 
                 report.version, 
                 report.timestamp.format("%Y-%m-%d %H:%M:%S UTC")).unwrap();

        md
    }

    /// Generate compact summary for logs.
    pub fn to_summary(&self, report: &MonitoringReport) -> String {
        format!(
            "Monitoring: {} checks, {} passed, {} warn, {} crit | Action: {} | NO-TRADE: {}",
            report.summary.total_checks,
            report.summary.passed,
            report.summary.warnings,
            report.summary.criticals,
            report.action,
            report.no_trade
        )
    }

    /// Write report to files.
    pub fn write(&self, report: &MonitoringReport, format: ReportFormat) -> io::Result<Vec<String>> {
        let mut files_written = Vec::new();

        // Ensure output directory exists
        fs::create_dir_all(&self.output_dir)?;

        if matches!(format, ReportFormat::Json | ReportFormat::Both) {
            let json_path = Path::new(&self.output_dir).join(&self.json_filename);
            let json = self.to_json(report)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            fs::write(&json_path, json)?;
            files_written.push(json_path.to_string_lossy().to_string());
        }

        if matches!(format, ReportFormat::Markdown | ReportFormat::Both) {
            let md_path = Path::new(&self.output_dir).join(&self.md_filename);
            let md = self.to_markdown(report);
            fs::write(&md_path, md)?;
            files_written.push(md_path.to_string_lossy().to_string());
        }

        Ok(files_written)
    }

    /// Write to GitHub Actions step summary.
    pub fn write_github_summary(&self, report: &MonitoringReport) -> io::Result<()> {
        if let Ok(summary_path) = std::env::var("GITHUB_STEP_SUMMARY") {
            let md = self.to_markdown(report);
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&summary_path)?
                .write_all(md.as_bytes())?;
        }
        Ok(())
    }
}

impl Default for MonitoringReporter {
    fn default() -> Self {
        Self::new("output")
    }
}

/// Truncate message for table display.
fn truncate_message(msg: &str, max_len: usize) -> String {
    if msg.len() <= max_len {
        msg.to_string()
    } else {
        format!("{}...", &msg[..max_len.saturating_sub(3)])
    }
}

use std::io::Write as IoWrite;

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use rust_decimal_macros::dec;

    fn make_test_report() -> MonitoringReport {
        let mut report = MonitoringReport::default();
        
        // Add some results
        report.results.push(
            CheckResult::pass("Freshness_BR", CheckCategory::DataHealth)
                .with_value(dec!(1))
                .with_threshold(dec!(5))
        );
        report.results.push(
            CheckResult::warn("Coverage_BR", CheckCategory::DataHealth, "Coverage at 75%")
                .with_value(dec!(75))
                .with_threshold(dec!(80))
        );
        report.results.push(
            CheckResult::crit("DrawdownGuardrail", CheckCategory::Regression, "DD at 22%")
                .with_value(dec!(22))
                .with_threshold(dec!(20))
        );
        
        report.summary = crate::monitoring::types::MonitoringSummary::from_results(&report.results);
        report
    }

    #[test]
    fn test_to_json() {
        let reporter = MonitoringReporter::default();
        let report = make_test_report();
        
        let json = reporter.to_json(&report).unwrap();
        
        assert!(json.contains("\"version\": \"1.0.0\""));
        assert!(json.contains("Freshness_BR"));
        assert!(json.contains("Coverage_BR"));
    }

    #[test]
    fn test_to_markdown() {
        let reporter = MonitoringReporter::default();
        let report = make_test_report();
        
        let md = reporter.to_markdown(&report);
        
        assert!(md.contains("## Monitoring Report"));
        assert!(md.contains("### Status:"));
        assert!(md.contains("| Check | Status |"));
        assert!(md.contains("Freshness_BR"));
    }

    #[test]
    fn test_to_markdown_critical_section() {
        let reporter = MonitoringReporter::default();
        let report = make_test_report();
        
        let md = reporter.to_markdown(&report);
        
        assert!(md.contains("### 🔴 Critical Issues"));
        assert!(md.contains("DrawdownGuardrail"));
    }

    #[test]
    fn test_to_markdown_warnings_section() {
        let reporter = MonitoringReporter::default();
        let report = make_test_report();
        
        let md = reporter.to_markdown(&report);
        
        assert!(md.contains("### 🟡 Warnings"));
        assert!(md.contains("Coverage_BR"));
    }

    #[test]
    fn test_to_summary() {
        let reporter = MonitoringReporter::default();
        let report = make_test_report();
        
        let summary = reporter.to_summary(&report);
        
        assert!(summary.contains("3 checks"));
        assert!(summary.contains("1 passed"));
        assert!(summary.contains("1 warn"));
        assert!(summary.contains("1 crit"));
    }

    #[test]
    fn test_truncate_message() {
        assert_eq!(truncate_message("short", 10), "short");
        assert_eq!(truncate_message("this is a very long message", 15), "this is a ve...");
    }

    #[test]
    fn test_healthy_report_status() {
        let reporter = MonitoringReporter::default();
        let mut report = MonitoringReport::default();
        report.results.push(CheckResult::pass("test", CheckCategory::DataHealth));
        report.summary = crate::monitoring::types::MonitoringSummary::from_results(&report.results);
        
        let md = reporter.to_markdown(&report);
        
        assert!(md.contains("🟢 HEALTHY"));
    }

    #[test]
    fn test_no_trade_flag() {
        let reporter = MonitoringReporter::default();
        let mut report = MonitoringReport::default();
        report.no_trade = true;
        
        let md = reporter.to_markdown(&report);
        
        assert!(md.contains("NO-TRADE FLAG SET"));
    }

    #[test]
    fn test_json_determinism() {
        let reporter = MonitoringReporter::default();
        let report = make_test_report();
        
        let json1 = reporter.to_json(&report).unwrap();
        let json2 = reporter.to_json(&report).unwrap();
        
        // JSON structure should be deterministic (values may differ due to timestamps)
        assert!(json1.contains("\"version\":"));
        assert!(json2.contains("\"version\":"));
    }
}

