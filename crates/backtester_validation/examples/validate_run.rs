//! Example: Validate a backtest run
//!
//! This example shows how to use the validation pipeline to validate
//! a backtest run's output artifacts.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example validate_run -- /path/to/run/directory
//! ```

use backtester_validation::{
    BacktestArtifacts, ValidationConfig, ValidationPipeline, Verdict,
};
use std::env;
use std::path::Path;

fn main() {
    // Get run directory from command line
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <run_directory>", args[0]);
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} output/experiments/exp_001", args[0]);
        std::process::exit(1);
    }
    
    let run_dir = Path::new(&args[1]);
    
    if !run_dir.exists() {
        eprintln!("Error: Directory not found: {}", run_dir.display());
        std::process::exit(1);
    }
    
    // Extract run ID from directory name
    let run_id = run_dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║            BACKTESTER VALIDATION EXAMPLE                     ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Run directory: {:48} ║", run_dir.display());
    println!("║ Run ID:        {:48} ║", run_id);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    
    // Load artifacts
    let artifacts = BacktestArtifacts::from_dir(run_dir, &run_id);
    
    // Check files exist
    if !artifacts.files_exist() {
        eprintln!("Error: Required files not found");
        eprintln!("  Expected: metrics.json, nav_history.csv");
        std::process::exit(1);
    }
    
    // Configure validation
    let config = ValidationConfig {
        strict_mode: false,
        crosscheck_enabled: true,
        attribution_enabled: true,
        report_enabled: true,
        ..Default::default()
    };
    
    // Run pipeline
    let pipeline = ValidationPipeline::new(config);
    
    match pipeline.validate(&artifacts) {
        Ok(result) => {
            // Print results
            println!("=== Schema Check ===");
            if result.schema_check.has_failures() {
                println!("  ✗ FAILED");
                println!("    Missing fields: {:?}", result.schema_check.missing_fields);
                println!("    Null fields: {:?}", result.schema_check.null_fields);
            } else {
                println!("  ✓ PASSED");
            }
            println!();
            
            println!("=== Sanity Check ===");
            match result.sanity_check.verdict {
                Verdict::Pass => println!("  ✓ PASSED"),
                Verdict::Warn => {
                    println!("  ⚠ WARNINGS:");
                    for w in &result.sanity_check.warnings {
                        println!("    - {}", w.message);
                    }
                }
                Verdict::Fail => {
                    println!("  ✗ FAILED: {}", result.sanity_check.message);
                }
            }
            println!();
            
            if let Some(ref cc) = result.crosscheck {
                println!("=== Cross-Check ===");
                if cc.passed {
                    println!("  ✓ PASSED");
                } else {
                    println!("  ✗ Mismatches detected:");
                    for comp in &cc.comparisons {
                        if !comp.passed {
                            println!("    {} - diff: {:.2}%", comp.name, comp.relative_diff * 100.0);
                        }
                    }
                }
                println!();
                
                println!("  Recomputed metrics:");
                println!("    CAGR: {:.2}%", cc.recomputed.cagr * 100.0);
                println!("    Volatility: {:.2}%", cc.recomputed.volatility * 100.0);
                println!("    Sharpe: {:.2}", cc.recomputed.sharpe_ratio);
                println!("    Max DD: {:.2}%", cc.recomputed.max_drawdown * 100.0);
                println!();
            }
            
            if let Some(ref attr) = result.attribution {
                println!("=== Asset Attribution ===");
                println!("  Total assets: {}", attr.attributions.len());
                println!("  Total net PnL: {:.2}", attr.total_net_pnl);
                println!("  Total trades: {}", attr.total_trades);
                
                // Top 5 winners
                let mut sorted = attr.attributions.clone();
                sorted.sort_by(|a, b| b.net_pnl.partial_cmp(&a.net_pnl).unwrap_or(std::cmp::Ordering::Equal));
                
                println!();
                println!("  Top 5 winners:");
                for (i, a) in sorted.iter().take(5).enumerate() {
                    println!("    {}. {} - PnL: {:.2} ({:.1}%)", 
                             i + 1, a.symbol, a.net_pnl, a.contribution_pct * 100.0);
                }
                
                // Bottom 5 losers
                println!();
                println!("  Bottom 5 losers:");
                for (i, a) in sorted.iter().rev().take(5).enumerate() {
                    println!("    {}. {} - PnL: {:.2} ({:.1}%)", 
                             i + 1, a.symbol, a.net_pnl, a.contribution_pct * 100.0);
                }
                println!();
            }
            
            // Final verdict
            println!("╔══════════════════════════════════════════════════════════════╗");
            match result.verdict {
                Verdict::Pass => {
                    println!("║                    ✓ VALIDATION PASSED                       ║");
                }
                Verdict::Warn => {
                    println!("║                    ⚠ VALIDATION WARNINGS                     ║");
                }
                Verdict::Fail => {
                    println!("║                    ✗ VALIDATION FAILED                       ║");
                }
            }
            println!("╚══════════════════════════════════════════════════════════════╝");
            
            // Generate artifacts
            let output_dir = run_dir.join("validation");
            if let Err(e) = pipeline.generate_artifacts(&result, &output_dir) {
                eprintln!("Warning: Could not generate artifacts: {}", e);
            } else {
                println!();
                println!("Generated artifacts in: {}", output_dir.display());
                println!("  - validation_summary.json");
                println!("  - sanity.json");
                println!("  - backtest_report.md");
                if result.attribution.is_some() {
                    println!("  - asset_attribution.csv");
                }
            }
        }
        Err(e) => {
            eprintln!("Validation error: {}", e);
            std::process::exit(1);
        }
    }
}


