//! Factory run and resume commands - Execute multi-seed campaigns.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use sha2::Digest;
use tokio::runtime::Runtime;
use tracing::{error, info};

use chrono::NaiveDate;
use std::path::Path;

use combiner_engine::{EvolutionConfig, EvolutionEngine};
use backtester_intelligence::monitoring::{
    DataIntegrityGate, DataIntegrityReport, AuditMode, DataContext, UniverseType,
};
use backtester_intelligence::filters::Market;
use combiner_runner::{CliExecutor, ValidationCache};

use super::config::CampaignConfig;
use super::registry::{
    generate_campaign_id, generate_candidate_id, generate_run_id,
    CampaignStatus, Registry, RunStatus,
};

/// Execute factory run command.
pub fn execute_run(campaign_path: &str) -> Result<()> {
    run_campaign(campaign_path, false)
}

/// Execute factory resume command.
pub fn execute_resume(campaign_path: &str) -> Result<()> {
    run_campaign(campaign_path, true)
}

/// Core campaign execution logic.
fn run_campaign(campaign_path: &str, is_resume: bool) -> Result<()> {
    // Load campaign config
    let config = CampaignConfig::load(campaign_path)?;
    info!(
        name = config.campaign.name,
        seeds = config.seeds.count,
        "Loaded campaign config"
    );

    // Compute hashes for reproducibility
    let config_hash = config.config_hash();
    let dataset_hash = config.dataset_hash();
    let git_branch = CampaignConfig::git_branch();
    let git_sha = CampaignConfig::git_sha();
    let seeds = config.seeds.generate_seeds();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              STRATEGY FACTORY - CAMPAIGN RUN                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Campaign:    {}                            ", config.campaign.name);
    println!("║ Config Hash: {}                            ", config_hash);
    println!("║ Seeds:       {:?}                          ", seeds);
    println!("║ Mode:        {}                            ", if is_resume { "RESUME" } else { "NEW" });
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Run with tokio runtime
    let rt = Runtime::new()?;
    rt.block_on(async {
        let registry = Registry::connect().await?;

        // Generate or retrieve campaign ID
        let campaign_id = if is_resume {
            // Look for existing campaign with same config_hash
            let campaigns = registry.list_campaigns(None).await?;
            campaigns
                .iter()
                .find(|c| c.config_hash == config_hash)
                .map(|c| c.campaign_id.clone())
                .unwrap_or_else(generate_campaign_id)
        } else {
            generate_campaign_id()
        };

        // Register campaign
        registry
            .register_campaign(
                &campaign_id,
                &config.campaign.name,
                config.campaign.tag.as_deref(),
                config.campaign.owner.as_deref(),
                git_branch.as_deref(),
                git_sha.as_deref(),
                &config_hash,
                dataset_hash.as_deref(),
                &seeds,
                config.campaign.notes.as_deref(),
            )
            .await?;

        // Get seeds to run
        let seeds_to_run = if is_resume {
            let incomplete = registry.get_incomplete_seeds(&campaign_id).await?;
            if incomplete.is_empty() {
                println!("\nAll seeds already completed for this campaign.");
                return Ok(());
            }
            println!("\nResuming {} incomplete seeds: {:?}", incomplete.len(), incomplete);
            incomplete
        } else {
            seeds.iter().map(|&s| s as i64).collect()
        };

        // Update campaign status to running
        registry
            .update_campaign_status(&campaign_id, CampaignStatus::Running)
            .await?;

        // === DATA INTEGRITY GATE ===
        if config.data_integrity.enabled {
            println!("\n[Data Integrity] Running pre-flight audit...");
            
            let market = parse_market(&config.dataset.market);
            let delay_bars = config.execution.delay_bars.unwrap_or(1);
            let mode = AuditMode::from_str(&config.data_integrity.mode);
            
            let gate = DataIntegrityGate::new(
                market,
                delay_bars,
                config.data_integrity.max_gap_days,
                mode,
            );
            
            // Build minimal DataContext for audit
            let mut ctx = DataContext::new(chrono::Utc::now().date_naive());
            ctx.delay_bars_policy = delay_bars;
            ctx.universe_type = parse_universe_type(&config.data_integrity.universe_type);
            
            // Run audit
            let dataset_hash_str = dataset_hash.clone().unwrap_or_default();
            let report = gate.audit(&ctx, &dataset_hash_str);
            
            // Save report
            let report_dir = format!("artifacts/data_integrity/{}", campaign_id);
            std::fs::create_dir_all(&report_dir).ok();
            let report_path = format!("{}/report.json", report_dir);
            if let Err(e) = report.save(Path::new(&report_path)) {
                error!("Failed to save data integrity report: {}", e);
            }
            
            // Check verdict
            println!("[Data Integrity] {}", report.summary());
            
            if !report.passed() {
                println!("\n[Data Integrity] FAILED - Blocking campaign execution");
                for reason in &report.hard_fails {
                    println!("  ❌ {}", reason);
                }
                registry
                    .update_campaign_status(&campaign_id, CampaignStatus::Failed)
                    .await?;
                return Err(anyhow::anyhow!("Data integrity check failed"));
            }
            
            println!("[Data Integrity] PASSED - Proceeding with campaign\n");
        }

        // Progress bar for seeds
        let pb = ProgressBar::new(seeds_to_run.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} seeds ({msg})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut completed = 0;
        let mut failed = 0;

        // Execute each seed
        for seed in seeds_to_run {
            let run_id = generate_run_id();
            pb.set_message(format!("seed {}", seed));

            // Register run start
            registry
                .register_run_start(&run_id, &campaign_id, seed)
                .await?;

            // Execute the SCG run
            let start_time = Instant::now();
            let result = execute_single_run(&config, seed as u64, &run_id).await;

            let duration = start_time.elapsed().as_secs() as i32;

            match result {
                Ok(run_result) => {
                    // Register candidates
                    for (rank, cand) in run_result.candidates.iter().enumerate() {
                        let candidate_id = generate_candidate_id();
                        registry
                            .register_candidate(
                                &candidate_id,
                                &run_id,
                                &cand.genome_hash,
                                rank as i32,
                                Some(cand.oos_sharpe_net),
                                Some(cand.oos_sharpe_gross),
                                Some(cand.pbo),
                                cand.dsr,
                                Some(cand.stress_passed),
                                Some(cand.stress_total),
                                Some(cand.gates_passed),
                                Some(cand.turnover_annual),
                                cand.capacity_usd,
                                cand.oos_cagr_net,
                                cand.max_drawdown_net,
                            )
                            .await?;
                    }

                    // Register run end (success)
                    registry
                        .register_run_end(
                            &run_id,
                            RunStatus::Completed,
                            Some(duration),
                            Some(run_result.generations as i32),
                            Some(run_result.evaluations as i64),
                            Some(&run_result.artifact_path),
                            None,
                            run_result.candidates.first().map(|c| c.oos_sharpe_net),
                            run_result.candidates.first().map(|c| c.pbo),
                            Some(run_result.candidates.len() as i32),
                        )
                        .await?;

                    completed += 1;
                    info!(run_id, seed, "Run completed successfully");
                }
                Err(e) => {
                    // Register run end (failed)
                    registry
                        .register_run_end(
                            &run_id,
                            RunStatus::Failed,
                            Some(duration),
                            None,
                            None,
                            None,
                            Some(&e.to_string()),
                            None,
                            None,
                            None,
                        )
                        .await?;

                    failed += 1;
                    error!(run_id, seed, error = %e, "Run failed");
                }
            }

            pb.inc(1);
        }

        pb.finish_with_message("done");

        // Update campaign status
        let final_status = if failed == 0 {
            CampaignStatus::Completed
        } else if completed == 0 {
            CampaignStatus::Failed
        } else {
            CampaignStatus::Completed // Partial success
        };
        registry
            .update_campaign_status(&campaign_id, final_status)
            .await?;

        // Print summary
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                    CAMPAIGN COMPLETE                         ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Campaign ID: {}                          ", campaign_id);
        println!("║ Completed:   {} runs                     ", completed);
        println!("║ Failed:      {} runs                     ", failed);
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Next steps:                                                  ║");
        println!("║   combiner factory show {}               ", campaign_id);
        println!("║   combiner factory promote --campaign {} ", &campaign_id[..14.min(campaign_id.len())]);
        println!("╚══════════════════════════════════════════════════════════════╝");

        Ok(())
    })
}

/// Result from a single run.
struct RunResult {
    generations: u32,
    evaluations: u64,
    artifact_path: String,
    candidates: Vec<CandidateResult>,
}

/// Candidate result data.
struct CandidateResult {
    genome_hash: String,
    oos_sharpe_net: f32,
    oos_sharpe_gross: f32,
    pbo: f32,
    dsr: Option<f32>,
    stress_passed: i32,
    stress_total: i32,
    gates_passed: bool,
    turnover_annual: f32,
    capacity_usd: Option<f32>,
    oos_cagr_net: Option<f32>,
    max_drawdown_net: Option<f32>,
}

/// Execute a single SCG run.
async fn execute_single_run(
    config: &CampaignConfig,
    seed: u64,
    run_id: &str,
) -> Result<RunResult> {
    // Build evolution config
    let mut evo_config = EvolutionConfig::default();
    evo_config.seed = Some(seed);

    if let Some(pop) = config.evolution.population_size {
        evo_config.population_size = pop;
    }
    if let Some(gen) = config.evolution.max_generations {
        evo_config.max_generations = gen;
    }
    if let Some(conv) = config.evolution.convergence_generations {
        evo_config.convergence_generations = conv;
    }

    // Set execution config
    evo_config.stress_testing_enabled = config.budget.stress_enabled;

    // Create output directory
    let output_dir = format!("output/scg/{}", run_id);
    std::fs::create_dir_all(&output_dir)?;

    // Create executor
    let executor = CliExecutor::new()
        .with_output_dir(std::path::PathBuf::from(&output_dir).join("backtests"));

    // Create validation cache
    let validation_cache = Arc::new(ValidationCache::new());

    // Create and run evolution engine
    let mut engine = EvolutionEngine::new(evo_config.clone(), executor);

    info!(run_id, seed, "Starting SCG evolution");

    // Run ultra mode
    let result = engine.evolve_ultra(validation_cache, config.budget.top_k)?;

    // Collect candidates from validated HoF
    let mut candidates = Vec::new();
    for (rank, entry) in result.validated_hall_of_fame.entries().iter().enumerate() {
        // Compute genome hash
        let genome_json = serde_json::to_string(&entry.genome)?;
        let genome_hash = format!(
            "sha256:{}",
            hex::encode(&sha2::Sha256::digest(genome_json.as_bytes())[..8])
        );

        // Use available fields from ValidationResultSummary
        candidates.push(CandidateResult {
            genome_hash,
            oos_sharpe_net: entry.validation.oos_sharpe_median as f32,
            oos_sharpe_gross: entry.validation.oos_sharpe_mean as f32,
            pbo: entry.validation.pbo as f32,
            dsr: Some(entry.validation.dsr as f32),
            stress_passed: entry.validation.splits_passed as i32,
            stress_total: entry.validation.splits_evaluated as i32,
            // Heuristic: gate passed if at least half of splits passed
            gates_passed: entry.validation.splits_passed >= entry.validation.splits_evaluated / 2,
            turnover_annual: 0.0, // Not available in summary
            capacity_usd: None,   // Not available in summary
            oos_cagr_net: Some(entry.validation.oos_cagr_median as f32),
            max_drawdown_net: None, // Not available in ValidationResultSummary
        });

        // Save strategy config
        if let Ok(toml_str) = entry.genome.to_toml() {
            let strategy_path = format!("{}/strategy_{:03}.toml", output_dir, rank);
            std::fs::write(&strategy_path, toml_str)?;
        }

        if rank >= config.budget.top_k {
            break;
        }
    }

    Ok(RunResult {
        generations: result.total_generations,
        evaluations: engine.stats().iter().map(|s| s.evaluated as u64).sum(),
        artifact_path: output_dir,
        candidates,
    })
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Parse market string to Market enum.
fn parse_market(market_str: &str) -> Market {
    match market_str.to_uppercase().as_str() {
        "BR" | "B3" => Market::BR,
        "US" | "NYSE" | "NASDAQ" => Market::US,
        _ => Market::BR, // Default to BR
    }
}

/// Parse universe type string to UniverseType enum.
fn parse_universe_type(universe_str: &str) -> UniverseType {
    match universe_str.to_lowercase().as_str() {
        "point_in_time" | "pit" => UniverseType::PointInTime,
        "static" => UniverseType::Static,
        _ => UniverseType::Unknown,
    }
}
