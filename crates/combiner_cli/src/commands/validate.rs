//! Validate command - Run anti-overfitting validation on top strategies.

use anyhow::Result;
use backtester_execution::ExecutionModelConfig;
use combiner_core::StrategyGenome;
use combiner_engine::{
    GenomeValidatorAntiOverfit, ValidationConfig, ValidationReport,
};
use combiner_runner::CliExecutor;
use std::fs;
use std::path::Path;
use tracing::info;

/// Execute the validate command.
pub fn execute(experiment_id: &str, top_k: usize, full: bool, stress_enabled: bool) -> Result<()> {
    let hof_dir = Path::new("output/scg").join(experiment_id).join("hall_of_fame");

    if !hof_dir.exists() {
        println!("Experiment not found or has no Hall of Fame: {}", experiment_id);
        return Ok(());
    }

    info!("Loading top {} strategies from {}", top_k, experiment_id);

    // Load genomes from Hall of Fame
    let genomes = load_genomes_from_hof(&hof_dir, top_k)?;

    if genomes.is_empty() {
        println!("No genomes found in Hall of Fame");
        return Ok(());
    }

    println!("\n=== Validation Results ===\n");
    println!("Validating {} strategies from experiment {}", genomes.len(), experiment_id);
    println!("Mode: {}", if full { "Full (WFA + CPCV + PBO/DSR)" } else { "WFA only" });
    println!("Stress testing: {}", if stress_enabled { "ENABLED" } else { "disabled" });
    println!();

    // Create validator with execution config (fail-fast if backtester not found)
    let executor = CliExecutor::try_new()
        .map_err(|e| anyhow::anyhow!("Backtester not found: {}. \
            Build with `cargo build --release --bin backtest` or set BACKTEST_CLI_PATH.", e))?;
    let mut config = ValidationConfig::default();
    config.execution = ExecutionModelConfig::mvp();
    config.stress_testing_enabled = stress_enabled;
    config.min_stress_scenarios_passed = 4;
    
    let validator = GenomeValidatorAntiOverfit::new(executor, config);

    // Run validation
    let total_trials = 100; // Placeholder - would be actual trial count
    let reports = validator.validate_top_k(&genomes, top_k, total_trials);

    // Print results
    print_validation_results(&reports, stress_enabled);

    // Save validation report
    let report_path = Path::new("output/scg")
        .join(experiment_id)
        .join("validation_report.json");
    let report_json = serde_json::to_string_pretty(&reports)?;
    fs::write(&report_path, report_json)?;
    println!("\nValidation report saved to {:?}", report_path);

    // Summary
    let passed = reports.iter().filter(|r| r.overall_passed).count();
    let failed = reports.len() - passed;

    println!("\n=== Summary ===");
    println!("Passed: {} / {}", passed, reports.len());
    println!("Failed: {} (see discard reasons)", failed);

    if stress_enabled {
        // Print stress test summary
        let stress_passed: usize = reports
            .iter()
            .filter_map(|r| r.stress_result.as_ref())
            .map(|s| s.passed_count)
            .sum();
        let stress_total: usize = reports
            .iter()
            .filter_map(|r| r.stress_result.as_ref())
            .map(|s| s.total_count)
            .sum();
        println!("Stress tests passed: {} / {}", stress_passed, stress_total);
    }

    if full {
        println!("\nNote: Full CPCV validation not yet implemented - WFA results shown");
    }

    Ok(())
}

/// Load genomes from Hall of Fame directory.
fn load_genomes_from_hof(hof_dir: &Path, limit: usize) -> Result<Vec<StrategyGenome>> {
    let mut genomes = Vec::new();

    // Read strategy directories
    let mut entries: Vec<_> = fs::read_dir(hof_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir() && e.file_name().to_string_lossy().starts_with("strategy_"))
        .collect();

    // Sort by name
    entries.sort_by(|a, b| a.path().cmp(&b.path()));

    for entry in entries.into_iter().take(limit) {
        let genome_path = entry.path().join("genome.json");
        if genome_path.exists() {
            let content = fs::read_to_string(&genome_path)?;
            match serde_json::from_str::<StrategyGenome>(&content) {
                Ok(genome) => genomes.push(genome),
                Err(e) => {
                    eprintln!("Warning: Failed to parse {:?}: {}", genome_path, e);
                }
            }
        }
    }

    Ok(genomes)
}

/// Print validation results as a table.
fn print_validation_results(reports: &[ValidationReport], stress_enabled: bool) {
    if stress_enabled {
        println!("{:<6} {:<10} {:<10} {:<10} {:<8} {:<10} {}", 
                 "Rank", "IS Sharpe", "OOS Sharpe", "Degrad%", "Status", "Stress", "Reason");
        println!("{}", "-".repeat(80));
    } else {
        println!("{:<6} {:<10} {:<10} {:<10} {:<8} {}", 
                 "Rank", "IS Sharpe", "OOS Sharpe", "Degrad%", "Status", "Reason");
        println!("{}", "-".repeat(70));
    }

    for (i, report) in reports.iter().enumerate() {
        let (is_sharpe, oos_sharpe, degradation) = if let Some(ref wfa) = report.wfa_result {
            (
                format!("{:.2}", wfa.is_sharpe_net),
                format!("{:.2}", wfa.oos_sharpe_net),
                format!("{:.1}%", wfa.degradation_pct),
            )
        } else {
            ("N/A".into(), "N/A".into(), "N/A".into())
        };

        let status = if report.overall_passed { "✓ PASS" } else { "✗ FAIL" };
        let reason = report.discard_reason.as_deref().unwrap_or("-");

        if stress_enabled {
            let stress_info = if let Some(ref stress) = report.stress_result {
                format!("{}/{}", stress.passed_count, stress.total_count)
            } else {
                "N/A".to_string()
            };
            
            println!(
                "{:<6} {:<10} {:<10} {:<10} {:<8} {:<10} {}",
                i + 1,
                is_sharpe,
                oos_sharpe,
                degradation,
                status,
                stress_info,
                reason
            );
        } else {
            println!(
                "{:<6} {:<10} {:<10} {:<10} {:<8} {}",
                i + 1,
                is_sharpe,
                oos_sharpe,
                degradation,
                status,
                reason
            );
        }
    }
}
