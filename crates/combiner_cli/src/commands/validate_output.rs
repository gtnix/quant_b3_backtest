//! Validate output command - Run comprehensive validation on backtest/SCG outputs.
//!
//! Uses the `backtester_validation` module to perform:
//! - Schema validation (no null required fields)
//! - Sanity checks (plausible Sharpe, volatility, trades)
//! - Cross-check (recompute metrics from nav_history.csv)
//! - Attribution (best/worst assets by PnL)
//!
//! Generates:
//! - `validation_summary.json`
//! - `backtest_report.md`
//! - `asset_attribution.csv`

use anyhow::Result;
use backtester_validation::{BacktestArtifacts, ValidationConfig, ValidationPipeline, Verdict};
use std::path::Path;
use tracing::info;

/// Execute the validate-output command.
///
/// # Arguments
/// * `run_dir` - Path to the run directory (containing metrics.json, nav_history.csv, etc.)
/// * `output_dir` - Optional output directory for validation artifacts
/// * `strict` - If true, warnings become failures
/// * `no_crosscheck` - If true, skip metric cross-checking (faster but less thorough)
pub fn execute(
    run_dir: &str,
    output_dir: Option<&str>,
    strict: bool,
    no_crosscheck: bool,
) -> Result<()> {
    let run_path = Path::new(run_dir);

    if !run_path.exists() {
        anyhow::bail!("Run directory not found: {}", run_dir);
    }

    // Extract run_id from directory name
    let run_id = run_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    info!("Validating output from: {} (run_id: {})", run_dir, run_id);

    // Load artifacts
    let artifacts = BacktestArtifacts::from_dir(run_path, &run_id);

    if !artifacts.files_exist() {
        anyhow::bail!(
            "Required files not found. Expected metrics.json and nav_history.csv in {}",
            run_dir
        );
    }

    // Configure validation
    let config = ValidationConfig {
        strict_mode: strict,
        crosscheck_enabled: !no_crosscheck,
        ..Default::default()
    };

    // Run validation pipeline
    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts)?;

    // Determine output directory
    let out_dir = match output_dir {
        Some(dir) => Path::new(dir).to_path_buf(),
        None => run_path.join("validation"),
    };

    // Generate all artifacts
    pipeline.generate_artifacts(&result, &out_dir)?;

    // Print results
    println!("\n=== Validation Results ===\n");
    println!("Run directory: {}", run_dir);
    println!("Run ID: {}", run_id);
    println!("Output directory: {}", out_dir.display());
    println!();

    // Print verdict
    let verdict_str = match result.verdict {
        Verdict::Pass => "\x1b[32m✓ PASS\x1b[0m",
        Verdict::Warn => "\x1b[33m⚠ WARN\x1b[0m",
        Verdict::Fail => "\x1b[31m✗ FAIL\x1b[0m",
    };
    println!("Verdict: {}", verdict_str);
    println!();

    // Print schema check
    println!("Schema Validation:");
    if result.schema_check.missing_fields.is_empty() && result.schema_check.null_fields.is_empty() {
        println!("  ✓ All required fields present and non-null");
    } else {
        if !result.schema_check.missing_fields.is_empty() {
            println!("  ✗ Missing fields: {:?}", result.schema_check.missing_fields);
        }
        if !result.schema_check.null_fields.is_empty() {
            println!("  ✗ Null fields: {:?}", result.schema_check.null_fields);
        }
    }
    println!();

    // Print sanity check
    println!("Sanity Checks:");
    if result.sanity_check.warnings.is_empty() && result.sanity_check.verdict != Verdict::Fail {
        println!("  ✓ All sanity checks passed");
    } else {
        for warning in &result.sanity_check.warnings {
            println!("  ⚠ {}", warning.message);
        }
        if result.sanity_check.verdict == Verdict::Fail {
            println!("  ✗ {}", result.sanity_check.message);
        }
    }
    println!();

    // Print cross-check results
    if let Some(ref crosscheck) = result.crosscheck {
        println!("Cross-Check (recomputed vs reported):");
        if crosscheck.passed {
            println!("  ✓ Metrics match within tolerance");
        } else {
            let icon = if crosscheck.verdict == Verdict::Warn { "⚠" } else { "✗" };
            println!("  {} Metrics mismatch detected:", icon);
            for comp in &crosscheck.comparisons {
                if !comp.passed {
                    println!(
                        "    {} - reported: {:.4}, recomputed: {:.4}, diff: {:.2}%",
                        comp.name, comp.reported, comp.recomputed, comp.relative_diff * 100.0
                    );
                }
            }
        }
        println!();
    }

    // Print attribution summary
    if let Some(ref attr) = result.attribution {
        println!("Asset Attribution:");
        println!("  Total symbols: {}", attr.attributions.len());
        println!("  Total trades: {}", attr.total_trades);
        println!("  Total net PnL: {:.2}", attr.total_net_pnl);
        
        // Top 3 best
        let mut sorted = attr.attributions.clone();
        sorted.sort_by(|a, b| b.net_pnl.partial_cmp(&a.net_pnl).unwrap_or(std::cmp::Ordering::Equal));
        
        if !sorted.is_empty() {
            println!("  Top 3 winners:");
            for data in sorted.iter().take(3) {
                println!("    {} - PnL: {:.2}", data.symbol, data.net_pnl);
            }
        }
        
        // Concentration
        println!("  Concentration: top1={:.1}%, top5={:.1}%, top10={:.1}%",
            attr.concentration.top_1_pct * 100.0,
            attr.concentration.top_5_pct * 100.0,
            attr.concentration.top_10_pct * 100.0
        );
        println!();
    }

    // Print file locations
    println!("Generated artifacts:");
    println!("  - {}", out_dir.join("validation_summary.json").display());
    println!("  - {}", out_dir.join("sanity.json").display());
    println!("  - {}", out_dir.join("backtest_report.md").display());
    if result.attribution.is_some() {
        println!("  - {}", out_dir.join("asset_attribution.csv").display());
    }

    // Exit with appropriate code
    match result.verdict {
        Verdict::Pass => {
            println!("\n\x1b[32mValidation passed!\x1b[0m");
            Ok(())
        }
        Verdict::Warn => {
            println!("\n\x1b[33mValidation passed with warnings. Review the report.\x1b[0m");
            Ok(())
        }
        Verdict::Fail => {
            if strict {
                anyhow::bail!("Validation failed in strict mode. See report for details.");
            } else {
                println!("\n\x1b[31mValidation failed. See report for details.\x1b[0m");
                Ok(())
            }
        }
    }
}
