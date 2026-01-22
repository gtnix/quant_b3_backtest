//! Diagnose command - Generate Stage A vs Stage B diagnostic reports.
//!
//! Analyzes evolution results to identify why strategies are failing Stage B validation.

use std::path::PathBuf;
use std::fs;
use anyhow::{Result, Context};
use clap::Args;
use tracing::info;

use combiner_engine::{MarketDiagnosticReport, FailedCandidate};

/// Arguments for the diagnose command.
#[derive(Debug, Args)]
pub struct DiagnoseArgs {
    /// Market to analyze (BR or US)
    #[arg(long, default_value = "BR")]
    pub market: String,
    
    /// Input directory containing evolution results
    #[arg(long, default_value = "output/scg")]
    pub input: PathBuf,
    
    /// Output directory for diagnostic reports
    #[arg(long, default_value = "artifacts/diagnostics")]
    pub output: PathBuf,
    
    /// Include near-miss analysis
    #[arg(long, default_value = "true")]
    pub near_miss: bool,
}

/// Run the diagnose command.
pub fn run(args: DiagnoseArgs) -> Result<()> {
    info!("🔬 Generating diagnostic report for {} market", args.market);
    
    // Create output directory
    fs::create_dir_all(&args.output)?;
    
    // Look for failed candidates file
    let failed_path = args.input.join("failed_candidates.json");
    
    let failed_candidates: Vec<FailedCandidate> = if failed_path.exists() {
        info!("Loading failed candidates from {:?}", failed_path);
        let content = fs::read_to_string(&failed_path)?;
        serde_json::from_str(&content)
            .context("Failed to parse failed_candidates.json")?
    } else {
        // Try to load from recent runs
        info!("No failed_candidates.json found, scanning for recent results...");
        Vec::new()
    };
    
    if failed_candidates.is_empty() {
        info!("⚠️ No failed candidates found to analyze.");
        info!("Run 'combiner run' with the --in-process flag to generate failed candidate data.");
        
        // Generate empty report
        let report = MarketDiagnosticReport::from_failed_candidates(&args.market, &[]);
        let report_path = args.output.join(format!("{}_diagnostic_report.json", args.market.to_lowercase()));
        fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
        info!("Empty report saved to {:?}", report_path);
        
        return Ok(());
    }
    
    info!("Analyzing {} failed candidates", failed_candidates.len());
    
    // Generate the report
    let report = MarketDiagnosticReport::from_failed_candidates(&args.market, &failed_candidates);
    
    // Save main report
    let report_path = args.output.join(format!("{}_diagnostic_report.json", args.market.to_lowercase()));
    fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    info!("✅ Report saved to {:?}", report_path);
    
    // Save failure breakdown
    let breakdown_path = args.output.join(format!("{}_failure_breakdown.json", args.market.to_lowercase()));
    fs::write(&breakdown_path, serde_json::to_string_pretty(&report.top_failure_reasons)?)?;
    info!("✅ Breakdown saved to {:?}", breakdown_path);
    
    // Save near-miss strategies
    if args.near_miss && !report.near_miss_strategies.is_empty() {
        let near_miss_path = args.output.join(format!("{}_near_miss.json", args.market.to_lowercase()));
        fs::write(&near_miss_path, serde_json::to_string_pretty(&report.near_miss_strategies)?)?;
        info!("✅ Near-miss strategies saved to {:?}", near_miss_path);
    }
    
    // Save human-readable summary
    let summary_path = args.output.join(format!("{}_gap_analysis.md", args.market.to_lowercase()));
    let summary = generate_markdown_summary(&report);
    fs::write(&summary_path, summary)?;
    info!("✅ Gap analysis saved to {:?}", summary_path);
    
    // Print summary to console
    println!("\n{} Market Diagnostic Summary:", args.market);
    println!("================================");
    println!("Total strategies analyzed: {}", report.total_strategies);
    println!("Passed: {} ({:.1}%)", report.passed_count, report.pass_rate);
    println!("Failed: {}", report.failed_count);
    println!("\nTop Failure Reasons:");
    for (reason, count, pct) in &report.top_failure_reasons {
        println!("  - {}: {} ({:.1}%)", reason, count, pct);
    }
    println!("\nDiagnosis: {}", report.gap_diagnosis);
    for note in &report.diagnosis_notes {
        println!("  → {}", note);
    }
    println!("\nStage A Sharpe: mean={:.3}, median={:.3}", 
        report.stage_a_sharpe_dist.mean,
        report.stage_a_sharpe_dist.median);
    println!("Stage B Sharpe: mean={:.3}, median={:.3}", 
        report.stage_b_sharpe_dist.mean,
        report.stage_b_sharpe_dist.median);
    
    Ok(())
}

/// Generate a markdown summary of the gap analysis.
fn generate_markdown_summary(report: &MarketDiagnosticReport) -> String {
    let mut md = String::new();
    
    md.push_str(&format!("# {} Market Diagnostic Report\n\n", report.market));
    md.push_str(&format!("Generated: {}\n\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
    
    md.push_str("## Summary\n\n");
    md.push_str(&format!("- Total strategies analyzed: **{}**\n", report.total_strategies));
    md.push_str(&format!("- Passed Stage B: **{}** ({:.1}%)\n", report.passed_count, report.pass_rate));
    md.push_str(&format!("- Failed Stage B: **{}**\n\n", report.failed_count));
    
    md.push_str("## Top Failure Reasons\n\n");
    md.push_str("| Reason | Count | % |\n");
    md.push_str("|--------|-------|---|\n");
    for (reason, count, pct) in &report.top_failure_reasons {
        md.push_str(&format!("| {} | {} | {:.1}% |\n", reason, count, pct));
    }
    md.push('\n');
    
    md.push_str("## Gap Diagnosis\n\n");
    md.push_str(&format!("**Primary Diagnosis:** {}\n\n", report.gap_diagnosis));
    if !report.diagnosis_notes.is_empty() {
        md.push_str("**Notes:**\n\n");
        for note in &report.diagnosis_notes {
            md.push_str(&format!("- {}\n", note));
        }
    }
    md.push('\n');
    
    md.push_str("## Stage A vs Stage B Distribution\n\n");
    md.push_str("### Stage A Sharpe\n");
    md.push_str(&format!("- Min: {:.3}\n", report.stage_a_sharpe_dist.min));
    md.push_str(&format!("- Max: {:.3}\n", report.stage_a_sharpe_dist.max));
    md.push_str(&format!("- Mean: {:.3}\n", report.stage_a_sharpe_dist.mean));
    md.push_str(&format!("- Median: {:.3}\n\n", report.stage_a_sharpe_dist.median));
    
    md.push_str("### Stage B Sharpe (OOS)\n");
    md.push_str(&format!("- Min: {:.3}\n", report.stage_b_sharpe_dist.min));
    md.push_str(&format!("- Max: {:.3}\n", report.stage_b_sharpe_dist.max));
    md.push_str(&format!("- Mean: {:.3}\n", report.stage_b_sharpe_dist.mean));
    md.push_str(&format!("- Median: {:.3}\n\n", report.stage_b_sharpe_dist.median));
    
    md.push_str("### Degradation (Stage B - Stage A)\n");
    md.push_str(&format!("- Mean: {:.1}%\n", report.degradation_dist.mean));
    md.push_str(&format!("- Median: {:.1}%\n\n", report.degradation_dist.median));
    
    if !report.near_miss_strategies.is_empty() {
        md.push_str("## Near-Miss Strategies (Almost Passed)\n\n");
        md.push_str("Top 10 strategies closest to passing Stage B:\n\n");
        md.push_str("| Rank | Strategy ID | Stage B Sharpe | Failure Reasons |\n");
        md.push_str("|------|-------------|----------------|----------------|\n");
        for (i, s) in report.near_miss_strategies.iter().enumerate().take(10) {
            md.push_str(&format!("| {} | {} | {:.3} | {} |\n", 
                i + 1,
                &s.strategy_id[..8.min(s.strategy_id.len())],
                s.stage_b_sharpe,
                s.failure_reasons.join(", ")
            ));
        }
    }
    
    md
}
