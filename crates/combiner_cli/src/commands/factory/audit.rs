//! Factory audit-data command - Standalone data integrity audit.

use anyhow::Result;
use std::path::Path;

use backtester_intelligence::filters::Market;
use backtester_intelligence::monitoring::{
    AuditMode, DataContext, DataIntegrityGate, UniverseType,
};

use super::config::CampaignConfig;

/// Execute factory audit-data command.
pub fn execute_audit(campaign_path: &str, mode: &str) -> Result<()> {
    // Load campaign config
    let config = CampaignConfig::load(campaign_path)?;
    
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              DATA INTEGRITY AUDIT                            ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Campaign:    {}                            ", config.campaign.name);
    println!("║ Market:      {}                            ", config.dataset.market);
    println!("║ Mode:        {}                            ", mode);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Parse market
    let market = match config.dataset.market.to_uppercase().as_str() {
        "BR" | "B3" => Market::BR,
        "US" | "NYSE" | "NASDAQ" => Market::US,
        _ => Market::BR,
    };

    // Parse mode
    let audit_mode = match mode.to_lowercase().as_str() {
        "strict" => AuditMode::Strict,
        _ => AuditMode::Fast,
    };

    // Create gate
    let delay_bars = config.execution.delay_bars.unwrap_or(1);
    let max_gap_days = config.data_integrity.max_gap_days;
    
    let gate = DataIntegrityGate::new(market, delay_bars, max_gap_days, audit_mode);

    // Build context
    let mut ctx = DataContext::new(chrono::Utc::now().date_naive());
    ctx.delay_bars_policy = delay_bars;
    ctx.universe_type = match config.data_integrity.universe_type.to_lowercase().as_str() {
        "point_in_time" | "pit" => UniverseType::PointInTime,
        "static" => UniverseType::Static,
        _ => UniverseType::Unknown,
    };

    // Run audit
    let dataset_hash = config.dataset_hash().unwrap_or_default();
    let report = gate.audit(&ctx, &dataset_hash);

    // Display results
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│                    AUDIT RESULTS                             │");
    println!("├──────────────────────────────────────────────────────────────┤");
    println!("│ Verdict:     {}                                         ", report.verdict);
    println!("│ Score:       {:.2}                                       ", report.score);
    println!("│ Duration:    {} ms                                        ", report.stats.duration_ms);
    println!("│ Checks:      {} total, {} passed, {} warnings, {} critical",
             report.stats.total_checks,
             report.stats.passed,
             report.stats.warnings,
             report.stats.critical);
    println!("└──────────────────────────────────────────────────────────────┘");

    if !report.hard_fails.is_empty() {
        println!("\n❌ HARD FAILURES:");
        for (i, fail) in report.hard_fails.iter().enumerate() {
            println!("   {}. {}", i + 1, fail);
        }
    }

    if !report.warnings.is_empty() {
        println!("\n⚠️  WARNINGS:");
        for (i, warn) in report.warnings.iter().enumerate() {
            println!("   {}. {}", i + 1, warn);
        }
    }

    // Save report
    let report_dir = "artifacts/data_integrity";
    std::fs::create_dir_all(report_dir)?;
    let report_path = format!("{}/audit_report.json", report_dir);
    report.save(Path::new(&report_path))?;
    println!("\nReport saved to: {}", report_path);

    // Exit with appropriate code
    if report.passed() {
        println!("\n✅ Data integrity audit PASSED");
        Ok(())
    } else {
        println!("\n❌ Data integrity audit FAILED");
        Err(anyhow::anyhow!("Data integrity check failed"))
    }
}
