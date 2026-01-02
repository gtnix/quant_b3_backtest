//! Audit Command - Institutional-grade audit of SCG runs.
//!
//! Executes a comprehensive 6-marco audit pipeline that validates:
//! - Marco 0: Campaign initialization (seeds, hashes, dates)
//! - Marco 1: Data integrity (anti-lookahead, universe, adjustments)
//! - Marco 2: Evolution quality (diversity, convergence, penalties)
//! - Marco 3: Validation robustness (WFA, PBO, DSR, stress tests)
//! - Marco 4: Promotion gates (thresholds, hard-fails)
//! - Marco 5: Artifact completeness (replay, provenance)
//!
//! # Exit Codes
//! - 0: All marcos PASS
//! - 1: One or more marcos FAIL
//! - 2: Error (missing files, invalid input)
//!
//! # References
//! - Bailey & Lopez de Prado (2014): PBO methodology
//! - Lopez de Prado (2018): Deflated Sharpe Ratio

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use tracing::{info, warn, error, debug, instrument};

use combiner_engine::audit_framework::{
    AuditRunner, AuditMarco, CheckVerdict,
};
use combiner_engine::audit_checks;

/// Execute the audit command.
///
/// # Arguments
/// - `run_dir`: Directory containing the SCG run artifacts
/// - `output`: Directory where audit results will be saved
/// - `strict`: If true, warnings are treated as failures
/// - `stop_on_fail`: If true, stop at first failing marco
/// - `verbose`: Enable detailed output
///
/// # Returns
/// - `Ok(())` if audit passed (exit code 0)
/// - `Err` with exit code 1 if audit failed, 2 if error
#[instrument(skip_all, fields(run_dir = %run_dir.display()))]
pub fn execute(
    run_dir: PathBuf,
    output: PathBuf,
    strict: bool,
    stop_on_fail: bool,
    verbose: bool,
) -> Result<()> {
    let start = Instant::now();
    
    // =========================================================================
    // PHASE 1: Validate input directory exists
    // =========================================================================
    
    if !run_dir.exists() {
        error!("Run directory does not exist: {}", run_dir.display());
        return Err(anyhow::anyhow!("Run directory not found: {}", run_dir.display()));
    }
    
    info!("Starting institutional audit of: {}", run_dir.display());
    
    // =========================================================================
    // PHASE 2: Load artifacts
    // =========================================================================
    
    let artifacts = audit_checks::load_run_artifacts(&run_dir)
        .context("Failed to load run artifacts")?;
    
    debug!("Loaded artifacts: manifest={}, report={}, ranking={}",
        artifacts.manifest.is_some(),
        artifacts.report.is_some(),
        artifacts.ranking.is_some()
    );
    
    // =========================================================================
    // PHASE 3: Initialize audit runner
    // =========================================================================
    
    let config_path = run_dir.join("manifest.json").display().to_string();
    let config_hash = artifacts.manifest
        .as_ref()
        .and_then(|m| m.get("config_hash"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    
    let mut runner = AuditRunner::new(&config_path, &config_hash, &output)
        .with_verbose(verbose);
    
    if let Some(exp_id) = artifacts.manifest.as_ref()
        .and_then(|m| m.get("experiment_id"))
        .and_then(|v| v.as_str())
    {
        runner = runner.with_campaign_id(exp_id);
    }
    
    // =========================================================================
    // PHASE 4: Execute Marcos
    // =========================================================================
    
    let mut any_failed = false;
    
    // --- Marco 0: Initialization ---
    let marco0_result = run_marco_0(&mut runner, &artifacts, &run_dir, strict);
    if marco0_result.is_err() {
        any_failed = true;
        if stop_on_fail {
            return finalize_audit(runner, any_failed, start, strict);
        }
    }
    
    // --- Marco 1: Data Integrity ---
    let marco1_result = run_marco_1(&mut runner, &artifacts, &run_dir, strict);
    if marco1_result.is_err() {
        any_failed = true;
        if stop_on_fail {
            return finalize_audit(runner, any_failed, start, strict);
        }
    }
    
    // --- Marco 2: Evolution ---
    let marco2_result = run_marco_2(&mut runner, &artifacts, strict);
    if marco2_result.is_err() {
        any_failed = true;
        if stop_on_fail {
            return finalize_audit(runner, any_failed, start, strict);
        }
    }
    
    // --- Marco 3: Validation ---
    let marco3_result = run_marco_3(&mut runner, &artifacts, &run_dir, strict);
    if marco3_result.is_err() {
        any_failed = true;
        if stop_on_fail {
            return finalize_audit(runner, any_failed, start, strict);
        }
    }
    
    // --- Marco 4: Promotion Gates ---
    let marco4_result = run_marco_4(&mut runner, &artifacts, &run_dir, strict);
    if marco4_result.is_err() {
        any_failed = true;
        if stop_on_fail {
            return finalize_audit(runner, any_failed, start, strict);
        }
    }
    
    // --- Marco 5: Artifacts ---
    let marco5_result = run_marco_5(&mut runner, &artifacts, &run_dir, strict);
    if marco5_result.is_err() {
        any_failed = true;
    }
    
    // =========================================================================
    // PHASE 5: Finalize
    // =========================================================================
    
    finalize_audit(runner, any_failed, start, strict)
}

/// Run Marco 0: Initialization checks
fn run_marco_0(
    runner: &mut AuditRunner,
    artifacts: &audit_checks::RunArtifacts,
    run_dir: &Path,
    _strict: bool,
) -> Result<(), ()> {
    runner.run_marco(AuditMarco::Initialization, |result| {
        // Check 1: Seed present
        result.add_check(audit_checks::check_seed_present(&artifacts.manifest));
        
        // Check 2: Config hash present
        result.add_check(audit_checks::check_config_hash(&artifacts.manifest));
        
        // Check 3: Dates valid
        result.add_check(audit_checks::check_dates_valid(&artifacts.manifest));
        
        // Check 4: Output structure
        result.add_check(audit_checks::check_output_structure(run_dir));
        
        // Check 5: Experiment ID present
        result.add_check(audit_checks::check_experiment_id(&artifacts.manifest));
    }).map_err(|e| {
        error!("Marco 0 failed: {}", e);
    })?;
    
    Ok(())
}

/// Run Marco 1: Data Integrity checks
fn run_marco_1(
    runner: &mut AuditRunner,
    artifacts: &audit_checks::RunArtifacts,
    _run_dir: &Path,
    _strict: bool,
) -> Result<(), ()> {
    runner.run_marco(AuditMarco::DataIntegrity, |result| {
        // Check 1: Delay bars >= 1 (anti-lookahead) - check manifest.execution_config
        result.add_check(audit_checks::check_delay_bars(&artifacts.manifest));
        
        // Check 2: Universe configuration - check manifest.dataset_config
        result.add_check(audit_checks::check_universe_config(&artifacts.manifest));
        
        // Check 3: No gaps in generations
        result.add_check(audit_checks::check_no_generation_gaps(&artifacts.report));
        
        // Check 4: Consistent timestamps
        result.add_check(audit_checks::check_timestamps_consistent(&artifacts.manifest));
    }).map_err(|e| {
        error!("Marco 1 failed: {}", e);
    })?;
    
    Ok(())
}

/// Run Marco 2: Evolution checks (CRITICAL - detects diversity issues)
fn run_marco_2(
    runner: &mut AuditRunner,
    artifacts: &audit_checks::RunArtifacts,
    _strict: bool,
) -> Result<(), ()> {
    runner.run_marco(AuditMarco::Evolution, |result| {
        // Check 1: Population diversity > 10%
        result.add_check(audit_checks::check_population_diversity(&artifacts.report, 0.10));
        
        // Check 2: Fitness variance > 0
        result.add_check(audit_checks::check_fitness_variance(&artifacts.report));
        
        // Check 3: Convergence is real (best improves over generations)
        result.add_check(audit_checks::check_convergence_real(&artifacts.report));
        
        // Check 4: No degenerate population (all same metrics)
        result.add_check(audit_checks::check_no_degenerate_population(&artifacts.ranking));
        
        // Check 5: Penalties applied for low trades
        result.add_check(audit_checks::check_penalties_applied(&artifacts.report));
    }).map_err(|e| {
        error!("Marco 2 failed: {}", e);
    })?;
    
    Ok(())
}

/// Run Marco 3: Validation checks
fn run_marco_3(
    runner: &mut AuditRunner,
    artifacts: &audit_checks::RunArtifacts,
    run_dir: &Path,
    _strict: bool,
) -> Result<(), ()> {
    runner.run_marco(AuditMarco::Validation, |result| {
        let hof_dir = run_dir.join("hall_of_fame");
        
        // Check 1: WFA reports present
        result.add_check(audit_checks::check_wfa_present(&hof_dir));
        
        // Check 2: OOS Sharpe threshold (>= 0.2 for research mode)
        result.add_check(audit_checks::check_oos_sharpe_threshold(&artifacts.ranking, 0.2));
        
        // Check 3: PBO threshold (<= 0.50 for research mode)
        result.add_check(audit_checks::check_pbo_threshold(&hof_dir, 0.50));
        
        // Check 4: DSR threshold (>= 0.0 for research mode - disabled)
        result.add_check(audit_checks::check_dsr_threshold(&hof_dir, 0.0));
        
        // Check 5: Stress test pass rate (>= 1/5 for research mode)
        result.add_check(audit_checks::check_stress_pass_rate(&hof_dir, 1, 5));
        
        // Check 6: Sharpe sanity (< 10)
        result.add_check(audit_checks::check_sharpe_sanity(&artifacts.ranking, 10.0));
        
        // Check 7: Trades threshold (>= 30)
        result.add_check(audit_checks::check_trades_threshold(&artifacts.ranking, 30));
    }).map_err(|e| {
        error!("Marco 3 failed: {}", e);
    })?;
    
    Ok(())
}

/// Run Marco 4: Promotion Gates checks
fn run_marco_4(
    runner: &mut AuditRunner,
    artifacts: &audit_checks::RunArtifacts,
    run_dir: &Path,
    _strict: bool,
) -> Result<(), ()> {
    runner.run_marco(AuditMarco::PromotionGates, |result| {
        let hof_dir = run_dir.join("hall_of_fame");
        
        // Check 1: Bundle complete for each strategy
        result.add_check(audit_checks::check_bundle_complete(&hof_dir));
        
        // Check 2: Validation summary present
        result.add_check(audit_checks::check_validation_summary_present(&hof_dir));
        
        // Check 3: No strategy with FAIL verdict promoted
        result.add_check(audit_checks::check_no_failed_promoted(&hof_dir));
        
        // Check 4: Thresholds enforced
        result.add_check(audit_checks::check_thresholds_enforced(&artifacts.ranking));
    }).map_err(|e| {
        error!("Marco 4 failed: {}", e);
    })?;
    
    Ok(())
}

/// Run Marco 5: Artifacts checks
fn run_marco_5(
    runner: &mut AuditRunner,
    artifacts: &audit_checks::RunArtifacts,
    run_dir: &Path,
    _strict: bool,
) -> Result<(), ()> {
    runner.run_marco(AuditMarco::Artifacts, |result| {
        let hof_dir = run_dir.join("hall_of_fame");
        
        // Check 1: Provenance complete
        result.add_check(audit_checks::check_provenance_complete(&artifacts.manifest));
        
        // Check 2: All required files present
        result.add_check(audit_checks::check_all_files_present(&hof_dir));
        
        // Check 3: Ranking consistent with strategies
        result.add_check(audit_checks::check_ranking_consistent(&hof_dir, &artifacts.ranking));
        
        // Check 4: Report JSON valid
        result.add_check(audit_checks::check_report_valid(&artifacts.report));
    }).map_err(|e| {
        error!("Marco 5 failed: {}", e);
    })?;
    
    Ok(())
}

/// Finalize audit and return appropriate result
fn finalize_audit(
    runner: AuditRunner,
    any_failed: bool,
    start: Instant,
    strict: bool,
) -> Result<()> {
    let duration = start.elapsed();
    
    match runner.finalize() {
        Ok(manifest) => {
            info!(
                "Audit completed in {:.2}s - Verdict: {:?}",
                duration.as_secs_f64(),
                manifest.final_verdict
            );
            
            println!("\n{}", "=".repeat(70));
            println!("  AUDIT COMPLETE");
            println!("{}", "=".repeat(70));
            println!("  Audit ID:     {}", manifest.audit_id);
            println!("  Output:       {}", manifest.output_dir.display());
            println!("  Duration:     {:.2}s", duration.as_secs_f64());
            println!("  Final Verdict: {:?}", manifest.final_verdict);
            println!("{}", "=".repeat(70));
            
            // Print summary per marco
            println!("\n  Marcos Summary:");
            for marco in AuditMarco::all() {
                if let Some(result) = manifest.marcos.get(marco.evidence_filename()) {
                    let emoji = match result.verdict {
                        CheckVerdict::Pass => "✓",
                        CheckVerdict::Warn => "⚠",
                        CheckVerdict::Fail => "✗",
                        CheckVerdict::Skip => "○",
                    };
                    println!("    {} Marco {}: {} - {:?}",
                        emoji,
                        marco.index(),
                        marco.name(),
                        result.verdict
                    );
                }
            }
            println!();
            
            // Print recommendation
            let recommendation = match manifest.final_verdict {
                CheckVerdict::Pass => "APROVAR - Estratégia passou em todos os marcos",
                CheckVerdict::Warn => "REVISAR - Há warnings que precisam de atenção",
                CheckVerdict::Fail => "REJEITAR - Falhas críticas detectadas",
                CheckVerdict::Skip => "INCOMPLETO - Alguns marcos não foram executados",
            };
            println!("  Recomendação: {}", recommendation);
            println!();
            
            if manifest.final_verdict == CheckVerdict::Fail || any_failed {
                Err(anyhow::anyhow!("Audit failed - see report for details"))
            } else if strict && manifest.final_verdict == CheckVerdict::Warn {
                Err(anyhow::anyhow!("Audit has warnings and strict mode is enabled"))
            } else {
                Ok(())
            }
        }
        Err(e) => {
            error!("Failed to finalize audit: {}", e);
            Err(anyhow::anyhow!("Audit finalization failed: {}", e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    
    #[test]
    fn test_execute_missing_dir() {
        let result = execute(
            PathBuf::from("/nonexistent/path"),
            PathBuf::from("/tmp/audit_output"),
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }
    
    #[test]
    fn test_execute_empty_dir() {
        let temp = TempDir::new().unwrap();
        let result = execute(
            temp.path().to_path_buf(),
            PathBuf::from("/tmp/audit_output"),
            false,
            false,
            false,
        );
        // Should fail gracefully with clear error
        assert!(result.is_err());
    }
}

