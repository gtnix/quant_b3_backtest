//! Markdown report generation for human-readable backtest reports.

use std::io::Write;
use std::path::Path;

use crate::{
    AttributionResult, SanityCheckResult, CrosscheckResult,
    ValidationError, Verdict,
    attribution::AssetAttribution,
};

/// Report generator for backtest results.
pub struct ReportGenerator {
    /// Run ID.
    run_id: String,
}

impl ReportGenerator {
    /// Create a new report generator.
    pub fn new(run_id: impl Into<String>) -> Self {
        Self { run_id: run_id.into() }
    }

    /// Generate a complete backtest report.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        sanity: &SanityCheckResult,
        crosscheck: Option<&CrosscheckResult>,
        attribution: Option<&AttributionResult>,
        verdict: Verdict,
        start_date: Option<&str>,
        end_date: Option<&str>,
        initial_capital: Option<f64>,
    ) -> String {
        let mut report = String::new();

        // Header
        report.push_str(&self.generate_header(verdict, start_date, end_date, initial_capital));

        // Metrics section
        report.push_str(&self.generate_metrics_section(sanity));

        // Alerts section
        report.push_str(&self.generate_alerts_section(sanity, crosscheck));

        // Attribution section
        if let Some(attr) = attribution {
            report.push_str(&self.generate_attribution_section(attr));
        }

        // Trade health section
        report.push_str(&self.generate_trade_health_section(sanity, attribution));

        // Conclusion
        report.push_str(&self.generate_conclusion(verdict, sanity));

        report
    }

    fn generate_header(
        &self,
        verdict: Verdict,
        start_date: Option<&str>,
        end_date: Option<&str>,
        initial_capital: Option<f64>,
    ) -> String {
        let verdict_icon = match verdict {
            Verdict::Pass => "✅",
            Verdict::Warn => "⚠️",
            Verdict::Fail => "❌",
        };

        let period = match (start_date, end_date) {
            (Some(s), Some(e)) => format!("{} a {}", s, e),
            _ => "Não especificado".to_string(),
        };

        let capital = initial_capital
            .map(|c| format!("R$ {:.2}", c))
            .unwrap_or_else(|| "Não especificado".to_string());

        format!(
            r#"# Backtest Report

**Run ID**: `{}`  
**Veredito**: {} {:?}  
**Período**: {}  
**Capital Inicial**: {}  
**Gerado em**: {}

---

"#,
            self.run_id,
            verdict_icon,
            verdict,
            period,
            capital,
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
        )
    }

    fn generate_metrics_section(&self, sanity: &SanityCheckResult) -> String {
        let snapshot = &sanity.metrics_snapshot;

        let sharpe = snapshot.sharpe_ratio.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "N/A".to_string());
        let vol = snapshot.annual_volatility.map(|v| format!("{:.1}%", v * 100.0)).unwrap_or_else(|| "N/A".to_string());
        let cagr = snapshot.cagr.map(|v| format!("{:.1}%", v * 100.0)).unwrap_or_else(|| "N/A".to_string());
        let maxdd = snapshot.max_drawdown.map(|v| format!("{:.1}%", v * 100.0)).unwrap_or_else(|| "N/A".to_string());
        let trades = snapshot.num_trades.map(|v| format!("{}", v)).unwrap_or_else(|| "N/A".to_string());
        let calmar = snapshot.calmar_ratio.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "N/A".to_string());

        format!(
            r#"## Métricas Principais

| Métrica | Valor |
|---------|-------|
| CAGR | {} |
| Volatilidade | {} |
| Sharpe Ratio | {} |
| Max Drawdown | {} |
| Calmar Ratio | {} |
| Total Trades | {} |

---

"#,
            cagr, vol, sharpe, maxdd, calmar, trades
        )
    }

    fn generate_alerts_section(
        &self,
        sanity: &SanityCheckResult,
        crosscheck: Option<&CrosscheckResult>,
    ) -> String {
        let mut alerts = String::new();
        alerts.push_str("## Alertas\n\n");

        let mut has_alerts = false;

        // Sanity warnings
        for warning in &sanity.warnings {
            has_alerts = true;
            alerts.push_str(&format!("⚠️ **{}**: {}\n\n", warning.code, warning.message));
        }

        // Crosscheck warnings
        if let Some(cc) = crosscheck {
            for warning in &cc.warnings {
                has_alerts = true;
                alerts.push_str(&format!("⚠️ **{}**: {}\n\n", warning.code, warning.message));
            }
        }

        if !has_alerts {
            alerts.push_str("✅ Nenhum alerta.\n\n");
        }

        alerts.push_str("---\n\n");
        alerts
    }

    fn generate_attribution_section(&self, attribution: &AttributionResult) -> String {
        let mut section = String::new();
        section.push_str("## Atribuição por Ativo\n\n");

        // Top winners
        let winners: Vec<_> = attribution.attributions.iter()
            .filter(|a| a.net_pnl > 0.0)
            .take(5)
            .collect();

        if !winners.is_empty() {
            section.push_str("### Melhores Papéis (Top 5)\n\n");
            section.push_str("| Ativo | PnL Net | Trades | Win Rate | Contribuição |\n");
            section.push_str("|-------|---------|--------|----------|-------------|\n");
            for attr in winners {
                section.push_str(&self.format_attribution_row(attr));
            }
            section.push('\n');
        }

        // Top losers
        let losers: Vec<_> = attribution.attributions.iter()
            .filter(|a| a.net_pnl < 0.0)
            .rev()
            .take(5)
            .collect();

        if !losers.is_empty() {
            section.push_str("### Piores Papéis (Bottom 5)\n\n");
            section.push_str("| Ativo | PnL Net | Trades | Win Rate | Contribuição |\n");
            section.push_str("|-------|---------|--------|----------|-------------|\n");
            for attr in losers {
                section.push_str(&self.format_attribution_row(attr));
            }
            section.push('\n');
        }

        // Concentration
        section.push_str("### Concentração\n\n");
        section.push_str(&format!(
            "- Top 1: {:.1}%\n",
            attribution.concentration.top_1_pct * 100.0
        ));
        section.push_str(&format!(
            "- Top 5: {:.1}%\n",
            attribution.concentration.top_5_pct * 100.0
        ));
        section.push_str(&format!(
            "- Top 10: {:.1}%\n",
            attribution.concentration.top_10_pct * 100.0
        ));
        section.push_str(&format!(
            "- HHI: {:.4}\n\n",
            attribution.concentration.hhi
        ));

        section.push_str("---\n\n");
        section
    }

    fn format_attribution_row(&self, attr: &AssetAttribution) -> String {
        format!(
            "| {} | R$ {:.2} | {} | {:.1}% | {:.1}% |\n",
            attr.symbol,
            attr.net_pnl,
            attr.num_trades,
            attr.win_rate * 100.0,
            attr.contribution_pct * 100.0
        )
    }

    fn generate_trade_health_section(
        &self,
        sanity: &SanityCheckResult,
        attribution: Option<&AttributionResult>,
    ) -> String {
        let mut section = String::new();
        section.push_str("## Trade Health\n\n");

        let trades = sanity.metrics_snapshot.num_trades.unwrap_or(0);
        section.push_str(&format!("- **Total Trades**: {}\n", trades));

        if let Some(attr) = attribution {
            let total_winners: u32 = attr.attributions.iter().map(|a| a.winning_trades).sum();
            let total_losers: u32 = attr.attributions.iter().map(|a| a.losing_trades).sum();
            
            if attr.total_trades > 0 {
                let overall_win_rate = total_winners as f64 / attr.total_trades as f64;
                section.push_str(&format!("- **Win Rate**: {:.1}%\n", overall_win_rate * 100.0));
            }

            section.push_str(&format!("- **Winning Trades**: {}\n", total_winners));
            section.push_str(&format!("- **Losing Trades**: {}\n", total_losers));

            let avg_pnl = if attr.total_trades > 0 {
                attr.total_net_pnl / attr.total_trades as f64
            } else {
                0.0
            };
            section.push_str(&format!("- **Avg Trade PnL**: R$ {:.2}\n", avg_pnl));
        }

        section.push_str("\n---\n\n");
        section
    }

    fn generate_conclusion(&self, verdict: Verdict, sanity: &SanityCheckResult) -> String {
        let mut section = String::new();
        section.push_str("## Conclusão\n\n");

        let (icon, text) = match verdict {
            Verdict::Pass => ("✅", "O backtest passou em todas as validações."),
            Verdict::Warn => ("⚠️", "O backtest passou com alertas. Recomenda-se investigar."),
            Verdict::Fail => ("❌", "O backtest FALHOU na validação. Resultados não confiáveis."),
        };

        section.push_str(&format!("**Veredito**: {} {}\n\n", icon, text));

        if !sanity.warnings.is_empty() {
            section.push_str("**Ações recomendadas**:\n\n");
            for (i, warning) in sanity.warnings.iter().enumerate() {
                section.push_str(&format!("{}. {}\n", i + 1, warning.message));
            }
        }

        section.push_str("\n---\n\n");
        section.push_str("*Relatório gerado automaticamente pelo Módulo de Validação.*\n");

        section
    }

    /// Write report to file.
    pub fn write_to_file(&self, report: &str, path: &Path) -> Result<(), ValidationError> {
        let mut file = std::fs::File::create(path)?;
        file.write_all(report.as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ValidationWarning;
    use crate::sanity::{SanityFlags, MetricsSnapshot};

    #[test]
    fn test_generate_report() {
        let gen = ReportGenerator::new("test_run_123");
        
        let sanity = SanityCheckResult {
            flags: SanityFlags::default(),
            verdict: Verdict::Pass,
            message: "All checks passed".to_string(),
            warnings: vec![],
            metrics_snapshot: MetricsSnapshot {
                sharpe_ratio: Some(1.5),
                annual_volatility: Some(0.18),
                cagr: Some(0.15),
                num_trades: Some(100),
                max_drawdown: Some(-0.12),
                calmar_ratio: Some(1.25),
            },
        };

        let report = gen.generate(
            &sanity,
            None,
            None,
            Verdict::Pass,
            Some("2020-01-01"),
            Some("2024-12-31"),
            Some(1_000_000.0),
        );

        assert!(report.contains("# Backtest Report"));
        assert!(report.contains("test_run_123"));
        assert!(report.contains("1.5")); // Sharpe
        assert!(report.contains("15.0%")); // CAGR
    }

    #[test]
    fn test_report_with_warnings() {
        let gen = ReportGenerator::new("test_warn");
        
        let sanity = SanityCheckResult {
            flags: SanityFlags { sharpe_suspicious: true, ..Default::default() },
            verdict: Verdict::Warn,
            message: "Warnings detected".to_string(),
            warnings: vec![
                ValidationWarning::new("SHARPE_HIGH", "Sharpe ratio 12.0 > 10"),
            ],
            metrics_snapshot: MetricsSnapshot::default(),
        };

        let report = gen.generate(&sanity, None, None, Verdict::Warn, None, None, None);

        assert!(report.contains("⚠️"));
        assert!(report.contains("SHARPE_HIGH"));
    }
}

