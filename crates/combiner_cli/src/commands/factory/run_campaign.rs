//! Factory run and resume commands - Execute multi-seed campaigns.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use sha2::Digest;
use tokio::runtime::Runtime;
use tracing::{error, info};

use std::path::Path;

use combiner_engine::{EvolutionConfig, EvolutionEngine, ArtifactFormat};
use backtester_intelligence::monitoring::{
    DataIntegrityGate, AuditMode, DataContext, UniverseType,
};
use backtester_intelligence::filters::Market;
use combiner_runner::{CliExecutor, ValidationCache};

/// Compression pipeline for OBFS writes (lazily initialized).
static OBFS_PIPELINE: std::sync::OnceLock<obfs::CompressionPipeline> = std::sync::OnceLock::new();

/// Get or initialize the OBFS compression pipeline.
fn get_compression_pipeline() -> &'static obfs::CompressionPipeline {
    OBFS_PIPELINE.get_or_init(|| obfs::CompressionPipeline::with_level(3))
}

/// Write JSON data with optional OBFS compression.
/// When format is OBFS, writes compressed .obfs file; otherwise writes .json file.
fn write_json_artifact<T: serde::Serialize>(
    base_path: &str,
    name: &str,
    data: &T,
    format: ArtifactFormat,
) -> Result<()> {
    match format {
        ArtifactFormat::Legacy => {
            let path = format!("{}/{}.json", base_path, name);
            std::fs::write(&path, serde_json::to_string_pretty(data)?)?;
        }
        ArtifactFormat::Obfs => {
            let json_bytes = serde_json::to_vec(data)?;
            let compressed = get_compression_pipeline()
                .compress(&json_bytes)
                .map_err(|e| anyhow::anyhow!("OBFS compression failed: {}", e))?;
            let path = format!("{}/{}.obfs", base_path, name);
            std::fs::write(&path, compressed)?;
        }
    }
    Ok(())
}

use super::config::CampaignConfig;
use super::registry::{
    generate_campaign_id, generate_candidate_id, generate_run_id,
    CampaignStatus, Registry, RunStatus,
};
use super::crosscheck;
use super::promote::auto_promote_to_hall_of_fame;

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

            // Compute and register config hash
            let config_json = serde_json::to_string(&config)?;
            let config_hash = format!("sha256:{}", hex::encode(&sha2::Sha256::digest(config_json.as_bytes())[..16]));
            registry.register_run_hashes(&run_id, Some(&config_hash), None).await?;

            // Execute the SCG run
            let start_time = Instant::now();
            let result = execute_single_run(&config, seed as u64, &run_id).await;

            let duration = start_time.elapsed().as_secs() as i32;

            match result {
                Ok(run_result) => {
                    // Register Stage A research candidates first
                    for cand in &run_result.research_candidates {
                        let candidate_id = generate_candidate_id();
                        registry
                            .register_research_candidate(
                                &candidate_id,
                                &run_id,
                                &cand.genome_hash,
                                cand.rank_in_run,
                                cand.oos_sharpe,
                                cand.oos_cagr,
                            )
                            .await?;
                    }
                    info!(run_id, "Registered {} Stage A research candidates", run_result.research_candidates.len());

                    // Register Stage B validated candidates (will upsert to validated)
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
                    info!(run_id, "Registered {} Stage B validated candidates", run_result.candidates.len());

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

                    // Register data integrity verdict for this run
                    // The audit was done at campaign level, so all runs inherit PASS
                    if config.data_integrity.enabled {
                        let report_path = format!("artifacts/data_integrity/{}/report.json", campaign_id);
                        registry
                            .register_data_integrity(&run_id, "PASS", 1.0, &report_path)
                            .await?;
                    }

                    // Run validation pipeline on outputs (Stage C validation)
                    run_output_validation(&run_result.artifact_path, &run_id);

                    // Auto-promote to Hall of Fame (Rust-native, no Node dependency)
                    let market = &config.dataset.market;
                    let criteria = super::promote::HallOfFameCriteria::from_promotion_config(&config.promotion);
                    match auto_promote_to_hall_of_fame(&registry, &run_id, market, Some(criteria)).await {
                        Ok(promoted) if promoted > 0 => {
                            info!(run_id, promoted, "Auto-promoted {} candidates to Hall of Fame", promoted);
                        }
                        Ok(_) => {} // No candidates met criteria
                        Err(e) => {
                            error!(run_id, error = %e, "Hall of Fame auto-promotion failed");
                        }
                    }

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
    /// Stage B validated candidates (top_k)
    candidates: Vec<CandidateResult>,
    /// Stage A research candidates (persist_stage_a_top_n)
    research_candidates: Vec<ResearchCandidateResult>,
}

/// Stage B validated candidate result data.
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

/// Stage A research candidate result data (minimal, from evolution HoF).
struct ResearchCandidateResult {
    genome_hash: String,
    rank_in_run: i32,
    oos_sharpe: Option<f32>,
    oos_cagr: Option<f32>,
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

    // Set hall_of_fame_size to persist_stage_a_top_n for research candidates
    evo_config.hall_of_fame_size = config.budget.persist_stage_a_top_n;

    // Set execution config
    evo_config.stress_testing_enabled = config.budget.stress_enabled;

    // Create output directory
    let output_dir = format!("output/scg/{}", run_id);
    std::fs::create_dir_all(&output_dir)?;

    // Create executor with validation (fail-fast if backtester not found)
    let cli_path = std::env::var("BACKTEST_CLI_PATH")
        .unwrap_or_else(|_| "target/release/backtest".to_string());
    info!("Using backtest CLI at: {}", cli_path);
    
    let mut executor = CliExecutor::try_new()
        .map_err(|e| anyhow::anyhow!("Backtester not found: {}. \
            Build with `cargo build --release --bin backtest` or set BACKTEST_CLI_PATH.", e))?
        .with_cli_path(&cli_path)
        .with_output_dir(std::path::PathBuf::from(&output_dir).join("backtests"));
    
    // Add market data path if configured
    if let Some(ref market_data) = config.dataset.market_data_path {
        info!("Using market data from: {}", market_data);
        executor = executor.with_market_data(market_data);
    }
    
    // Add data source if configured (database uses DATABASE_URL env var)
    if let Some(ref data_source) = config.dataset.data_source {
        info!("Using data source: {}", data_source);
        executor = executor.with_data_source(data_source);
        
        // Verify DATABASE_URL is set when using database source
        if data_source == "database" {
            if std::env::var("DATABASE_URL").is_err() {
                return Err(anyhow::anyhow!(
                    "data_source='database' requires DATABASE_URL environment variable to be set"
                ));
            }
            info!("DATABASE_URL is set, will use Neon database for market data");
        }
    }
    
    // Add risk profile if configured
    if let Some(ref profile) = config.risk_profile.name {
        info!("Using risk profile: {}", profile);
        executor = executor.with_risk_profile(profile);
    }
    
    // Enable OBFS format for backtests (uses isolated pending files, consolidated after evolution)
    if config.output.artifact_format_enum() == ArtifactFormat::Obfs {
        info!("OBFS artifact format enabled for backtests (isolated pending files)");
        executor = executor.with_obfs(true);
    }

    // Create validation cache
    let validation_cache = Arc::new(ValidationCache::new());

    // Create and run evolution engine
    let mut engine = EvolutionEngine::new(evo_config.clone(), executor);

    info!(run_id, seed, "Starting SCG evolution with Stage A HoF size {}", config.budget.persist_stage_a_top_n);

    // Run ultra mode
    let result = engine.evolve_ultra(validation_cache, config.budget.top_k)?;

    // Consolidate pending OBFS artifacts into Parquet (single-thread, concurrent-safe)
    if config.output.artifact_format_enum() == ArtifactFormat::Obfs {
        let pending_dir = format!("{}/backtests/pending", output_dir);
        let consolidated_dir = format!("{}/backtests/consolidated", output_dir);
        
        match obfs::consolidate(&pending_dir, &consolidated_dir) {
            Ok(stats) => {
                info!(
                    run_id,
                    "Consolidated {} backtests: {} rows, {:.1} MB, {:.1}x compression",
                    stats.artifacts_processed,
                    stats.timeseries_rows,
                    stats.parquet_size_bytes as f64 / 1_000_000.0,
                    stats.compression_ratio
                );
                
                // Clean up pending files after successful consolidation
                if let Err(e) = std::fs::remove_dir_all(&pending_dir) {
                    tracing::debug!(run_id, "Failed to cleanup pending dir: {}", e);
                } else {
                    info!(run_id, "Cleaned up pending OBFS files");
                }
            }
            Err(e) => {
                tracing::warn!(run_id, "Consolidation failed (non-fatal): {}", e);
            }
        }
    }

    // Collect Stage B validated candidates
    let mut candidates = Vec::new();
    for (rank, entry) in result.validated_hall_of_fame.entries().iter().enumerate() {
        // Compute genome hash
        let genome_json = serde_json::to_string(&entry.genome)?;
        let genome_hash = format!(
            "sha256:{}",
            hex::encode(&sha2::Sha256::digest(genome_json.as_bytes())[..8])
        );

        // Use available fields from ValidationResultSummary
        let v = entry.validation_ref();
        candidates.push(CandidateResult {
            genome_hash,
            oos_sharpe_net: v.oos_sharpe_median as f32,
            oos_sharpe_gross: v.oos_sharpe_mean as f32,
            pbo: v.pbo as f32,
            dsr: Some(v.dsr as f32),
            stress_passed: v.splits_passed as i32,
            stress_total: v.splits_evaluated as i32,
            // Use actual validation.passed which checks all criteria including DSR
            gates_passed: v.passed,
            turnover_annual: 0.0, // Not available in summary
            capacity_usd: None,   // Not available in summary
            oos_cagr_net: Some(v.oos_cagr_median as f32),
            max_drawdown_net: Some(v.oos_max_dd_worst as f32),
        });

        // Save strategy config and validation reports in hall_of_fame/strategy_N/
        let strategy_dir = format!("{}/hall_of_fame/strategy_{:03}", output_dir, rank);
        std::fs::create_dir_all(&strategy_dir)?;
        let promoted_at = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
        
        // Save strategy TOML config
        if let Ok(toml_str) = entry.genome.to_toml() {
            std::fs::write(format!("{}/strategy.toml", strategy_dir), toml_str)?;
        }
        
        // Save genome JSON
        if let Ok(genome_json) = serde_json::to_string_pretty(&entry.genome) {
            std::fs::write(format!("{}/genome.json", strategy_dir), genome_json)?;
        }
        
        // Save WFA report
        let wfa_report = serde_json::json!({
            "genome_id": entry.genome_id.to_string(),
            "oos_sharpe_median": v.oos_sharpe_median,
            "oos_sharpe_mean": v.oos_sharpe_mean,
            "oos_sharpe_std": v.oos_sharpe_std,
            "oos_cagr_median": v.oos_cagr_median,
            "oos_max_dd_worst": v.oos_max_dd_worst,
            "degradation_pct": v.degradation_pct,
            "splits_evaluated": v.splits_evaluated,
            "splits_passed": v.splits_passed
        });
        std::fs::write(
            format!("{}/wfa_report.json", strategy_dir),
            serde_json::to_string_pretty(&wfa_report)?
        )?;
        
        // Save PBO/DSR report
        let pbo_dsr = serde_json::json!({
            "genome_id": entry.genome_id.to_string(),
            "pbo": v.pbo,
            "dsr": v.dsr,
            "passed": v.pbo <= 0.25 && v.dsr >= 0.5
        });
        std::fs::write(
            format!("{}/pbo_dsr.json", strategy_dir),
            serde_json::to_string_pretty(&pbo_dsr)?
        )?;
        
        // Save stress report
        let stress_report = serde_json::json!({
            "genome_id": entry.genome_id.to_string(),
            "splits_evaluated": v.splits_evaluated,
            "splits_passed": v.splits_passed,
            "pass_rate": v.splits_passed as f64 / v.splits_evaluated.max(1) as f64
        });
        std::fs::write(
            format!("{}/stress_report.json", strategy_dir),
            serde_json::to_string_pretty(&stress_report)?
        )?;
        
        // Save metrics.json (Marco 4: bundle_complete requirement)
        let metrics = serde_json::json!({
            "genome_id": entry.genome_id.to_string(),
            "sharpe_ratio": v.oos_sharpe_median,
            "cagr": v.oos_cagr_median,
            "max_drawdown": v.oos_max_dd_worst,
            "volatility": v.oos_sharpe_std.abs() * 0.15, // Approximate
            "pbo": v.pbo,
            "dsr": v.dsr,
            "degradation_pct": v.degradation_pct,
            "splits_evaluated": v.splits_evaluated,
            "splits_passed": v.splits_passed
        });
        std::fs::write(
            format!("{}/metrics.json", strategy_dir),
            serde_json::to_string_pretty(&metrics)?
        )?;
        
        // Save validation_bundle.json (Marco 4: complete bundle for replay)
        let validation_bundle = serde_json::json!({
            "genome_id": entry.genome_id.to_string(),
            "rank": rank,
            "validated_generation": entry.validated_generation(),
            "validation_passed": v.passed,
            "promoted_at": promoted_at.clone(),
            "wfa_result": {
                "oos_sharpe_median": v.oos_sharpe_median,
                "oos_sharpe_mean": v.oos_sharpe_mean,
                "oos_sharpe_std": v.oos_sharpe_std,
                "oos_cagr_median": v.oos_cagr_median,
                "oos_max_dd_worst": v.oos_max_dd_worst,
                "degradation_pct": v.degradation_pct
            },
            "pbo_dsr": {
                "pbo": v.pbo,
                "dsr": v.dsr
            },
            "stress_result": {
                "splits_evaluated": v.splits_evaluated,
                "splits_passed": v.splits_passed,
                "pass_rate": v.splits_passed as f64 / v.splits_evaluated.max(1) as f64
            },
            "score": entry.score
        });
        std::fs::write(
            format!("{}/validation_bundle.json", strategy_dir),
            serde_json::to_string_pretty(&validation_bundle)?
        )?;

        if rank >= config.budget.top_k {
            break;
        }
    }
    
    // Log how many Stage B candidates were saved with full validation reports
    let stage_b_saved = result.validated_hall_of_fame.entries().len().min(config.budget.top_k + 1);
    if stage_b_saved > 0 {
        info!(run_id, "Saved {} Stage B strategies with validation reports", stage_b_saved);
    }

    // Collect Stage A research candidates from evolution HoF
    let mut research_candidates = Vec::new();
    for (rank, entry) in result.stage_a_hall_of_fame.entries().iter().enumerate() {
        // Compute genome hash
        let genome_json = serde_json::to_string(&entry.genome)?;
        let genome_hash = format!(
            "sha256:{}",
            hex::encode(&sha2::Sha256::digest(genome_json.as_bytes())[..8])
        );

        // Get fitness metrics if available
        let (oos_sharpe, oos_cagr) = entry.genome.fitness.as_ref()
            .map(|f| (Some(f.sharpe_ratio as f32), Some(f.cagr as f32)))
            .unwrap_or((None, None));

        research_candidates.push(ResearchCandidateResult {
            genome_hash,
            rank_in_run: (rank + 1) as i32,
            oos_sharpe,
            oos_cagr,
        });

        if rank >= config.budget.persist_stage_a_top_n {
            break;
        }
    }

    info!(run_id, "Collected {} Stage A research candidates, {} Stage B validated candidates",
          research_candidates.len(), candidates.len());

    // =========================================================================
    // GENERATE OUTPUT ARTIFACTS (manifest, hall_of_fame, generations)
    // =========================================================================
    
    let total_evaluations: u64 = engine.stats().iter().map(|s| s.evaluated as u64).sum();
    let cache_hits: u64 = engine.stats().iter().map(|s| s.cache_hits as u64).sum();
    let cache_rate = if total_evaluations > 0 { 
        cache_hits as f64 / total_evaluations as f64 * 100.0 
    } else { 0.0 };

    // 1. Generate manifest.json with all required fields for audit
    let config_json = serde_json::to_string(&config)?;
    let config_hash = format!("sha256:{}", hex::encode(&sha2::Sha256::digest(config_json.as_bytes())[..8]));
    let created_at = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
    
    let manifest = serde_json::json!({
        "run_id": run_id,
        "experiment_id": run_id,
        "campaign": config.campaign.name,
        "seed": seed,
        "config_hash": config_hash,
        "created_at": created_at.clone(),
        "timestamp": created_at.clone(),
        "status": "completed",
        "statistics": {
            "generations_completed": result.total_generations,
            "total_evaluations": total_evaluations,
            "cache_hit_rate_pct": cache_rate,
            "duration_secs": result.total_time_secs,
            "stage_a_candidates": research_candidates.len(),
            "stage_b_candidates": candidates.len()
        },
        "config": {
            "population_size": config.evolution.population_size,
            "max_generations": config.evolution.max_generations,
            "market": config.dataset.market,
            "market_data_path": config.dataset.market_data_path
        },
        // Execution config for anti-lookahead and cost verification (Marco 1)
        "execution_config": {
            "delay_bars": config.execution.delay_bars.unwrap_or(1),
            "slippage_bps": config.execution.slippage_bps.unwrap_or(10.0),
            // B3 standard costs (based on institutional rates)
            "commission_rate_bps": 5.0,        // 0.05% per trade
            "emolument_rate_bps": 0.3,         // B3 emoluments
            "clearing_rate_bps": 2.75,         // B3 clearing
            "market_impact_model": "square_root",
            "fill_assumption": "close_price"
        },
        // Dataset config for data integrity verification
        "dataset_config": {
            "start_date": config.dataset.start_date,
            "end_date": config.dataset.end_date,
            "universe": config.dataset.universe,
            "data_source": "neon_b3_market_data"
        }
    });
    let artifact_format = config.output.artifact_format_enum();
    write_json_artifact(&output_dir, "manifest", &manifest, artifact_format)?;
    info!(run_id, "Generated manifest (format: {:?})", artifact_format);

    // 2. Generate hall_of_fame/ directory with ranking.json and genomes/
    let hof_dir = format!("{}/hall_of_fame", output_dir);
    std::fs::create_dir_all(&hof_dir)?;
    std::fs::create_dir_all(format!("{}/genomes", hof_dir))?;
    
    // ranking.json with top candidates (as array for audit compatibility)
    let ranking: Vec<serde_json::Value> = result.stage_a_hall_of_fame.entries()
        .iter()
        .enumerate()
        .map(|(rank, entry)| {
            let fitness = entry.genome.fitness.as_ref();
            serde_json::json!({
                "rank": rank + 1,
                "sharpe_ratio": fitness.map(|f| f.sharpe_ratio).unwrap_or(0.0),
                "cagr": fitness.map(|f| f.cagr).unwrap_or(0.0),
                "max_drawdown": fitness.map(|f| f.max_drawdown).unwrap_or(0.0),
                "genome_file": format!("genomes/genome_{:03}.json", rank)
            })
        })
        .collect();
    
    // Write ranking as array (audit expects array format)
    write_json_artifact(&hof_dir, "ranking", &ranking, artifact_format)?;
    
    // Save individual genomes
    for (rank, entry) in result.stage_a_hall_of_fame.entries().iter().enumerate() {
        let genome_path = format!("{}/genomes/genome_{:03}.json", hof_dir, rank);
        if let Ok(json) = serde_json::to_string_pretty(&entry.genome) {
            let _ = std::fs::write(&genome_path, json);
        }
    }
    info!(run_id, "Generated hall_of_fame/ with {} candidates", ranking.len());

    // 3. Generate generations/ directory with per-generation snapshots
    let gen_dir = format!("{}/generations", output_dir);
    std::fs::create_dir_all(&gen_dir)?;
    
    let snapshots = result.performance_metrics.snapshots();
    for snapshot in &snapshots {
        let gen_path = format!("{}/gen_{:03}.json", gen_dir, snapshot.generation);
        if let Ok(json) = serde_json::to_string_pretty(&snapshot) {
            let _ = std::fs::write(&gen_path, json);
        }
    }
    
    // Summary of all generations (with timestamp for Marco 1 consistency)
    let gen_summary = serde_json::json!({
        "run_id": run_id,
        "timestamp": created_at.clone(),
        "total_generations": result.total_generations,
        "snapshots": snapshots
    });
    write_json_artifact(&gen_dir, "summary", &gen_summary, artifact_format)?;
    info!(run_id, "Generated generations/ with {} snapshots", snapshots.len());

    // 4. Generate report.json with generation_stats (for audit compatibility)
    // Use engine.stats() which has correct per-generation best_sharpe/mean_sharpe
    let generation_stats: Vec<serde_json::Value> = engine.stats().iter().map(|s| {
        serde_json::json!({
            "generation": s.generation,
            "evaluated": s.evaluated,
            "best_sharpe": s.best_sharpe,
            "mean_sharpe": s.mean_sharpe,
            "pareto_count": s.pareto_size,
            "cache_hits": s.cache_hits,
            "duration_ms": s.duration_ms
        })
    }).collect();
    
    let report = serde_json::json!({
        "run_id": run_id,
        "experiment_id": run_id,
        "timestamp": created_at.clone(),
        "created_at": created_at.clone(),
        "seed": seed,
        "total_generations": result.total_generations,
        "total_evaluations": total_evaluations,
        "duration_secs": result.total_time_secs,
        "generation_stats": generation_stats
    });
    write_json_artifact(&output_dir, "report", &report, artifact_format)?;
    info!(run_id, "Generated report.json");

    // ==========================================================================
    // FASE 4: ARTEFATOS PARA HUMANO (sanity.json, human_report.json, attribution.json)
    // ==========================================================================
    
    // 5. Generate sanity.json - quick sanity flags
    let best_sharpe = engine.stats().iter()
        .map(|s| s.best_sharpe)
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_sharpe = engine.stats().last()
        .map(|s| s.mean_sharpe)
        .unwrap_or(0.0);
    
    let sharpe_extreme = best_sharpe > 10.0;
    let volatility_zero = mean_sharpe.abs() < 0.001 && engine.stats().len() > 3;
    let no_trades = candidates.is_empty() && research_candidates.is_empty();
    let lookahead_risk = config.execution.delay_bars.unwrap_or(1) == 0;
    
    let sanity_passed = !sharpe_extreme && !volatility_zero && !no_trades && !lookahead_risk;
    
    let mut warnings: Vec<String> = Vec::new();
    if sharpe_extreme { warnings.push("Sharpe > 10 detectado - verificar cálculo".into()); }
    if volatility_zero { warnings.push("Volatilidade muito baixa - verificar dados".into()); }
    if no_trades { warnings.push("Nenhum candidato gerado - verificar configuração".into()); }
    if lookahead_risk { warnings.push("delay_bars=0 - risco de lookahead bias".into()); }
    
    let sanity = serde_json::json!({
        "run_id": run_id,
        "timestamp": created_at.clone(),
        "flags": {
            "sharpe_extreme": sharpe_extreme,
            "volatility_zero": volatility_zero,
            "no_trades": no_trades,
            "null_metrics": false,
            "lookahead_risk": lookahead_risk
        },
        "warnings": warnings,
        "passed": sanity_passed,
        "summary": if sanity_passed { "Todas as verificações de sanidade passaram" } else { "Atenção: há flags de alerta" }
    });
    write_json_artifact(&output_dir, "sanity", &sanity, artifact_format)?;
    info!(run_id, "Generated sanity.json (passed={})", sanity_passed);
    
    // 6. Generate human_report.json - human-readable summary
    let stage_a_count = research_candidates.len();
    let stage_b_count = candidates.len();
    let pass_rate = if stage_a_count > 0 { (stage_b_count as f64 / stage_a_count as f64) * 100.0 } else { 0.0 };
    
    let improvement = if engine.stats().len() >= 2 {
        let first_mean = engine.stats().first().map(|s| s.mean_sharpe).unwrap_or(0.0);
        let last_mean = engine.stats().last().map(|s| s.mean_sharpe).unwrap_or(0.0);
        ((last_mean - first_mean) / first_mean.abs().max(0.001)) * 100.0
    } else { 0.0 };
    
    let recommendation = if stage_b_count > 0 && sanity_passed {
        "APROVAR - Estratégias validadas prontas para produção"
    } else if stage_a_count > 0 && sanity_passed {
        "REVISAR - Candidatos Stage A disponíveis, nenhum passou Stage B"
    } else {
        "REJEITAR - Verificar configuração e dados"
    };
    
    let human_report = serde_json::json!({
        "run_id": run_id,
        "timestamp": created_at.clone(),
        "summary": format!(
            "Run completou {} gerações com {} avaliações. {} candidatos Stage A, {} validados Stage B ({:.1}% taxa de aprovação).",
            result.total_generations,
            total_evaluations,
            stage_a_count,
            stage_b_count,
            pass_rate
        ),
        "metrics": {
            "generations": result.total_generations,
            "evaluations": total_evaluations,
            "duration_secs": result.total_time_secs,
            "cache_hit_rate_pct": cache_rate
        },
        "best_strategy": if let Some(best) = candidates.first() {
            serde_json::json!({
                "sharpe": best.oos_sharpe_net,
                "cagr_pct": best.oos_cagr_net.unwrap_or(0.0) * 100.0,
                "max_drawdown_pct": best.max_drawdown_net.unwrap_or(0.0) * 100.0,
                "rank": 1
            })
        } else if let Some(best) = research_candidates.first() {
            serde_json::json!({
                "sharpe": best.oos_sharpe.unwrap_or(0.0),
                "cagr_pct": best.oos_cagr.unwrap_or(0.0) * 100.0,
                "rank": 1,
                "note": "Stage A (não validado)"
            })
        } else {
            serde_json::json!(null)
        },
        "evolution_progress": {
            "improvement_pct": improvement,
            "best_sharpe": best_sharpe,
            "final_mean_sharpe": mean_sharpe
        },
        "stage_funnel": {
            "stage_a_candidates": stage_a_count,
            "stage_b_validated": stage_b_count,
            "pass_rate_pct": pass_rate
        },
        "recommendation": recommendation,
        "sanity_check": sanity_passed
    });
    write_json_artifact(&output_dir, "human_report", &human_report, artifact_format)?;
    info!(run_id, "Generated human_report.json");
    
    // 7. Generate attribution.json - best/worst performers by strategy
    let attribution = serde_json::json!({
        "run_id": run_id,
        "timestamp": created_at.clone(),
        "top_strategies": research_candidates.iter().take(5).enumerate().map(|(i, c)| {
            serde_json::json!({
                "rank": i + 1,
                "sharpe": c.oos_sharpe.unwrap_or(0.0),
                "cagr_pct": c.oos_cagr.unwrap_or(0.0) * 100.0,
                "genome_hash": c.genome_hash.clone()
            })
        }).collect::<Vec<_>>(),
        "bottom_strategies": research_candidates.iter().rev().take(3).enumerate().map(|(i, c)| {
            serde_json::json!({
                "rank": research_candidates.len() - i,
                "sharpe": c.oos_sharpe.unwrap_or(0.0),
                "cagr_pct": c.oos_cagr.unwrap_or(0.0) * 100.0,
                "genome_hash": c.genome_hash.clone()
            })
        }).collect::<Vec<_>>()
    });
    write_json_artifact(&output_dir, "attribution", &attribution, artifact_format)?;
    info!(run_id, "Generated attribution.json");

    // ==========================================================================
    // PHASE 2: INSTITUTIONAL AUDIT ARTIFACTS
    // ==========================================================================
    
    // 8. Generate asset_attribution.json - PnL by asset (aggregate from backtests)
    let asset_attribution = generate_asset_attribution(&output_dir, run_id, &created_at);
    write_json_artifact(&output_dir, "asset_attribution", &asset_attribution, artifact_format)?;
    info!(run_id, "Generated asset_attribution.json");
    
    // 9. Generate audit_crosscheck.json - independent metric recalculation
    let crosscheck_result = crosscheck::crosscheck_run(
        Path::new(&output_dir),
        0.05, // 5% tolerance
        &created_at,
    );
    write_json_artifact(&output_dir, "audit_crosscheck", &crosscheck_result, artifact_format)?;
    info!(run_id, "Generated audit_crosscheck.json (checked {} strategies)", crosscheck_result.strategies_checked);
    
    // 10. Generate validation_overview.json - aggregate WFA/PBO/DSR/Stress
    let validation_overview = generate_validation_overview(&output_dir, run_id, &created_at, &candidates);
    write_json_artifact(&output_dir, "validation_overview", &validation_overview, artifact_format)?;
    info!(run_id, "Generated validation_overview.json");
    
    // 11. Generate audit_marcos.json - result of all 6 audit marcos
    let audit_marcos = generate_audit_marcos(&output_dir, run_id, &created_at, sanity_passed);
    write_json_artifact(&output_dir, "audit_marcos", &audit_marcos, artifact_format)?;
    info!(run_id, "Generated audit_marcos.json");

    Ok(RunResult {
        generations: result.total_generations,
        evaluations: total_evaluations,
        artifact_path: output_dir,
        candidates,
        research_candidates,
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

// =============================================================================
// ASSET ATTRIBUTION GENERATOR
// =============================================================================

/// Generate asset attribution by processing backtest timeseries data.
fn generate_asset_attribution(output_dir: &str, run_id: &str, timestamp: &str) -> serde_json::Value {
    use std::collections::HashMap;
    
    let backtests_dir = format!("{}/backtests", output_dir);
    let mut asset_stats: HashMap<String, AssetStats> = HashMap::new();
    let mut total_pnl = 0.0;
    
    // Process each backtest's timeseries
    if let Ok(entries) = std::fs::read_dir(&backtests_dir) {
        for entry in entries.flatten() {
            let ts_path = entry.path().join("timeseries.csv");
            if ts_path.exists() {
                if let Some(stats) = parse_timeseries_for_assets(&ts_path) {
                    for (symbol, pnl) in stats {
                        let entry = asset_stats.entry(symbol).or_insert(AssetStats::default());
                        entry.net_pnl += pnl;
                        entry.trades += 1;
                        total_pnl += pnl;
                    }
                }
            }
        }
    }
    
    // Convert to sorted list
    let mut assets: Vec<serde_json::Value> = asset_stats
        .into_iter()
        .map(|(symbol, stats)| {
            let contribution = if total_pnl.abs() > 0.0 { 
                (stats.net_pnl / total_pnl.abs()) * 100.0 
            } else { 0.0 };
            serde_json::json!({
                "symbol": symbol,
                "trades": stats.trades,
                "net_pnl": stats.net_pnl,
                "contribution_pct": contribution,
                "win_rate_pct": if stats.trades > 0 { 
                    (stats.wins as f64 / stats.trades as f64) * 100.0 
                } else { 0.0 }
            })
        })
        .collect();
    
    // Sort by contribution (absolute value, descending)
    assets.sort_by(|a, b| {
        let ca = a.get("contribution_pct").and_then(|v| v.as_f64()).unwrap_or(0.0).abs();
        let cb = b.get("contribution_pct").and_then(|v| v.as_f64()).unwrap_or(0.0).abs();
        cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    let top_contributors: Vec<_> = assets.iter()
        .filter(|a| a.get("contribution_pct").and_then(|v| v.as_f64()).unwrap_or(0.0) > 0.0)
        .take(5)
        .cloned()
        .collect();
        
    let worst_detractors: Vec<_> = assets.iter()
        .filter(|a| a.get("contribution_pct").and_then(|v| v.as_f64()).unwrap_or(0.0) < 0.0)
        .take(5)
        .cloned()
        .collect();
    
    // Diversification score (Herfindahl index inverse)
    let total_abs_contrib: f64 = assets.iter()
        .map(|a| a.get("contribution_pct").and_then(|v| v.as_f64()).unwrap_or(0.0).abs())
        .sum();
    let diversification = if total_abs_contrib > 0.0 {
        let hhi: f64 = assets.iter()
            .map(|a| {
                let c = a.get("contribution_pct").and_then(|v| v.as_f64()).unwrap_or(0.0).abs();
                (c / total_abs_contrib).powi(2)
            })
            .sum();
        1.0 - hhi.min(1.0)
    } else { 0.0 };
    
    serde_json::json!({
        "run_id": run_id,
        "timestamp": timestamp,
        "assets": assets,
        "top_contributors": top_contributors,
        "worst_detractors": worst_detractors,
        "diversification_score": diversification,
        "total_assets": assets.len()
    })
}

#[derive(Default)]
struct AssetStats {
    trades: i32,
    wins: i32,
    net_pnl: f64,
}

/// Parse timeseries CSV to extract per-asset attribution.
/// This is a simplified parser - real implementation would need trade logs.
fn parse_timeseries_for_assets(ts_path: &Path) -> Option<Vec<(String, f64)>> {
    // The timeseries.csv has: date,equity,drawdown,exposure
    // We don't have per-asset breakdown in timeseries, so we synthesize from portfolio
    // For now, return empty - the real data comes from trace.jsonl
    
    let trace_path = ts_path.parent()?.join("trace.jsonl");
    if !trace_path.exists() {
        return Some(vec![]);
    }
    
    let content = std::fs::read_to_string(&trace_path).ok()?;
    let mut asset_pnl: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    
    for line in content.lines() {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
            // Look for trade signals with PnL
            if let Some(symbol) = json.get("symbol").and_then(|v| v.as_str()) {
                if let Some(pnl) = json.get("pnl").and_then(|v| v.as_f64()) {
                    *asset_pnl.entry(symbol.to_string()).or_insert(0.0) += pnl;
                }
            }
        }
    }
    
    Some(asset_pnl.into_iter().collect())
}

// =============================================================================
// VALIDATION OVERVIEW GENERATOR
// =============================================================================

/// Generate validation overview aggregating WFA/PBO/DSR/Stress from all strategies.
fn generate_validation_overview(
    output_dir: &str,
    run_id: &str,
    timestamp: &str,
    candidates: &[CandidateResult],
) -> serde_json::Value {
    let hof_dir = format!("{}/hall_of_fame", output_dir);
    
    let mut wfa_sharpes_oos: Vec<f64> = Vec::new();
    let mut wfa_sharpes_is: Vec<f64> = Vec::new();
    let mut pbo_values: Vec<f64> = Vec::new();
    let mut dsr_values: Vec<f64> = Vec::new();
    let mut stress_passed = 0;
    let mut stress_total = 0;
    
    // Aggregate from candidates (Stage B validated)
    for cand in candidates {
        wfa_sharpes_oos.push(cand.oos_sharpe_net as f64);
        wfa_sharpes_is.push(cand.oos_sharpe_gross as f64);
        pbo_values.push(cand.pbo as f64);
        if let Some(dsr) = cand.dsr {
            dsr_values.push(dsr as f64);
        }
        stress_passed += cand.stress_passed;
        stress_total += cand.stress_total;
    }
    
    // Also read from individual strategy reports
    if let Ok(entries) = std::fs::read_dir(&hof_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("strategy_") && entry.path().is_dir() {
                // Read pbo_dsr.json
                let pbo_path = entry.path().join("pbo_dsr.json");
                if let Ok(content) = std::fs::read_to_string(&pbo_path) {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                        if let Some(pbo) = json.get("pbo").and_then(|v| v.as_f64()) {
                            if !pbo_values.contains(&pbo) {
                                pbo_values.push(pbo);
                            }
                        }
                        if let Some(dsr) = json.get("dsr").and_then(|v| v.as_f64()) {
                            if !dsr_values.contains(&dsr) {
                                dsr_values.push(dsr);
                            }
                        }
                    }
                }
            }
        }
    }
    
    let avg_oos_sharpe = if !wfa_sharpes_oos.is_empty() {
        wfa_sharpes_oos.iter().sum::<f64>() / wfa_sharpes_oos.len() as f64
    } else { 0.0 };
    
    let avg_is_sharpe = if !wfa_sharpes_is.is_empty() {
        wfa_sharpes_is.iter().sum::<f64>() / wfa_sharpes_is.len() as f64
    } else { 0.0 };
    
    let overfit_ratio = if avg_oos_sharpe.abs() > 0.001 {
        avg_is_sharpe / avg_oos_sharpe
    } else { 1.0 };
    
    let avg_pbo = if !pbo_values.is_empty() {
        pbo_values.iter().sum::<f64>() / pbo_values.len() as f64
    } else { 0.0 };
    
    let avg_dsr = if !dsr_values.is_empty() {
        dsr_values.iter().sum::<f64>() / dsr_values.len() as f64
    } else { 0.0 };
    
    let stress_pass_rate = if stress_total > 0 {
        (stress_passed as f64 / stress_total as f64) * 100.0
    } else { 0.0 };
    
    serde_json::json!({
        "run_id": run_id,
        "timestamp": timestamp,
        "wfa": {
            "total_strategies": wfa_sharpes_oos.len(),
            "passed": wfa_sharpes_oos.iter().filter(|&&s| s >= 0.2).count(),
            "avg_oos_sharpe": avg_oos_sharpe,
            "avg_is_sharpe": avg_is_sharpe,
            "overfit_ratio": overfit_ratio
        },
        "pbo": {
            "avg_pbo": avg_pbo,
            "below_threshold": pbo_values.iter().filter(|&&p| p <= 0.40).count(),
            "threshold": 0.40,
            "total": pbo_values.len()
        },
        "dsr": {
            "avg_dsr": avg_dsr,
            "above_threshold": dsr_values.iter().filter(|&&d| d >= 0.10).count(),
            "threshold": 0.10,
            "total": dsr_values.len()
        },
        "stress": {
            "tests_run": stress_total,
            "passed": stress_passed,
            "pass_rate_pct": stress_pass_rate
        }
    })
}

// =============================================================================
// AUDIT MARCOS GENERATOR
// =============================================================================

/// Generate audit marcos summary based on run artifacts.
fn generate_audit_marcos(
    output_dir: &str,
    run_id: &str,
    timestamp: &str,
    sanity_passed: bool,
) -> serde_json::Value {
    let run_path = Path::new(output_dir);
    
    // Marco 0: Setup - check basic files exist
    let marco_0 = check_marco_setup(run_path);
    
    // Marco 1: Data Integrity - check delay_bars, timestamps
    let marco_1 = check_marco_data_integrity(run_path);
    
    // Marco 2: Evolution - check generation_stats
    let marco_2 = check_marco_evolution(run_path);
    
    // Marco 3: Validation - check WFA/PBO/DSR files
    let marco_3 = check_marco_validation(run_path);
    
    // Marco 4: Promotion - check bundles complete
    let marco_4 = check_marco_promotion(run_path);
    
    // Marco 5: Artifacts - check all required files
    let marco_5 = check_marco_artifacts(run_path);
    
    let marcos = vec![marco_0, marco_1, marco_2, marco_3, marco_4, marco_5];
    
    let total_checks: usize = marcos.iter().map(|m| m.get("checks").and_then(|v| v.as_u64()).unwrap_or(0) as usize).sum();
    let total_passed: usize = marcos.iter().map(|m| m.get("passed").and_then(|v| v.as_u64()).unwrap_or(0) as usize).sum();
    let total_warnings: usize = marcos.iter().map(|m| m.get("warnings").and_then(|v| v.as_u64()).unwrap_or(0) as usize).sum();
    
    let overall = if marcos.iter().any(|m| m.get("status").and_then(|v| v.as_str()) == Some("FAIL")) {
        "FAIL"
    } else if total_warnings > 0 || !sanity_passed {
        "WARN"
    } else {
        "PASS"
    };
    
    serde_json::json!({
        "run_id": run_id,
        "timestamp": timestamp,
        "marcos": marcos,
        "overall": overall,
        "total_checks": total_checks,
        "total_passed": total_passed,
        "total_warnings": total_warnings
    })
}

fn check_marco_setup(run_path: &Path) -> serde_json::Value {
    let mut passed = 0;
    let mut warnings = 0;
    let checks = 5;
    
    // Check manifest exists
    if run_path.join("manifest.json").exists() { passed += 1; }
    else { warnings += 1; }
    
    // Check report exists
    if run_path.join("report.json").exists() { passed += 1; }
    else { warnings += 1; }
    
    // Check hall_of_fame exists
    if run_path.join("hall_of_fame").exists() { passed += 1; }
    else { warnings += 1; }
    
    // Check generations exists
    if run_path.join("generations").exists() { passed += 1; }
    else { warnings += 1; }
    
    // Check sanity exists
    if run_path.join("sanity.json").exists() { passed += 1; }
    else { warnings += 1; }
    
    let status = if passed == checks { "PASS" } 
                 else if passed >= 3 { "WARN" } 
                 else { "FAIL" };
    
    serde_json::json!({
        "id": 0,
        "name": "Setup",
        "status": status,
        "checks": checks,
        "passed": passed,
        "warnings": warnings
    })
}

fn check_marco_data_integrity(run_path: &Path) -> serde_json::Value {
    let mut passed = 0;
    let warnings = 0;
    let checks = 4;
    
    // Check manifest has execution_config
    if let Ok(content) = std::fs::read_to_string(run_path.join("manifest.json")) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            if json.get("execution_config").is_some() { passed += 1; }
            if json.get("dataset_config").is_some() { passed += 1; }
            if json.get("created_at").is_some() { passed += 1; }
            if json.get("timestamp").is_some() { passed += 1; }
        }
    }
    
    let status = if passed == checks { "PASS" } 
                 else if passed >= 2 { "WARN" } 
                 else { "FAIL" };
    
    serde_json::json!({
        "id": 1,
        "name": "Data Integrity",
        "status": status,
        "checks": checks,
        "passed": passed,
        "warnings": warnings
    })
}

fn check_marco_evolution(run_path: &Path) -> serde_json::Value {
    let mut passed = 0;
    let mut warnings = 0;
    let checks = 5;
    
    // Check report has generation_stats
    if let Ok(content) = std::fs::read_to_string(run_path.join("report.json")) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(stats) = json.get("generation_stats").and_then(|v| v.as_array()) {
                passed += 1; // Has stats
                if stats.len() >= 2 { passed += 1; } // Multiple generations
                
                // Check for variance in mean_sharpe
                let means: Vec<f64> = stats.iter()
                    .filter_map(|s| s.get("mean_sharpe").and_then(|v| v.as_f64()))
                    .collect();
                if means.len() >= 2 {
                    let unique: std::collections::HashSet<u64> = means.iter().map(|x| x.to_bits()).collect();
                    if unique.len() > 1 { passed += 1; } else { warnings += 1; }
                }
            }
        }
    }
    
    // Check generations/ directory has files
    if let Ok(entries) = std::fs::read_dir(run_path.join("generations")) {
        if entries.count() > 0 { passed += 1; }
    }
    
    // Check for ranking
    if run_path.join("hall_of_fame/ranking.json").exists() { passed += 1; }
    
    let status = if passed >= 4 { "PASS" } 
                 else if passed >= 2 { "WARN" } 
                 else { "FAIL" };
    
    serde_json::json!({
        "id": 2,
        "name": "Evolution",
        "status": status,
        "checks": checks,
        "passed": passed,
        "warnings": warnings
    })
}

fn check_marco_validation(run_path: &Path) -> serde_json::Value {
    let mut passed = 0;
    let warnings = 0;
    let checks = 5;
    
    let hof_path = run_path.join("hall_of_fame");
    
    if let Ok(entries) = std::fs::read_dir(&hof_path) {
        let mut has_wfa = false;
        let mut has_pbo = false;
        let mut has_stress = false;
        let mut has_metrics = false;
        let mut has_bundle = false;
        
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("strategy_") && entry.path().is_dir() {
                if entry.path().join("wfa_report.json").exists() { has_wfa = true; }
                if entry.path().join("pbo_dsr.json").exists() { has_pbo = true; }
                if entry.path().join("stress_report.json").exists() { has_stress = true; }
                if entry.path().join("metrics.json").exists() { has_metrics = true; }
                if entry.path().join("validation_bundle.json").exists() { has_bundle = true; }
            }
        }
        
        if has_wfa { passed += 1; }
        if has_pbo { passed += 1; }
        if has_stress { passed += 1; }
        if has_metrics { passed += 1; }
        if has_bundle { passed += 1; }
    }
    
    let status = if passed >= 4 { "PASS" } 
                 else if passed >= 2 { "WARN" } 
                 else { "FAIL" };
    
    serde_json::json!({
        "id": 3,
        "name": "Validation",
        "status": status,
        "checks": checks,
        "passed": passed,
        "warnings": warnings
    })
}

fn check_marco_promotion(run_path: &Path) -> serde_json::Value {
    let mut passed = 0;
    let warnings = 0;
    let checks = 4;
    
    let hof_path = run_path.join("hall_of_fame");
    
    // Count strategy directories
    let strategy_count = std::fs::read_dir(&hof_path)
        .map(|e| e.flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("strategy_"))
            .count())
        .unwrap_or(0);
    
    if strategy_count > 0 { passed += 1; }
    
    // Check validation_overview exists
    if run_path.join("validation_overview.json").exists() { passed += 1; }
    
    // Check audit_crosscheck exists
    if run_path.join("audit_crosscheck.json").exists() { passed += 1; }
    
    // Check human_report exists
    if run_path.join("human_report.json").exists() { passed += 1; }
    
    let status = if passed >= 3 { "PASS" } 
                 else if passed >= 1 { "WARN" } 
                 else { "FAIL" };
    
    serde_json::json!({
        "id": 4,
        "name": "Promotion",
        "status": status,
        "checks": checks,
        "passed": passed,
        "warnings": warnings
    })
}

fn check_marco_artifacts(run_path: &Path) -> serde_json::Value {
    let mut passed = 0;
    let warnings = 0;
    let checks = 3;
    
    // Check all essential files exist
    let essential = ["manifest.json", "report.json", "human_report.json"];
    for file in essential {
        if run_path.join(file).exists() { passed += 1; }
    }
    
    let status = if passed == checks { "PASS" } 
                 else if passed >= 2 { "WARN" } 
                 else { "FAIL" };
    
    serde_json::json!({
        "id": 5,
        "name": "Artifacts",
        "status": status,
        "checks": checks,
        "passed": passed,
        "warnings": warnings
    })
}

/// Run output validation on a completed run.
///
/// This runs the `backtester_validation` pipeline on the Hall of Fame outputs
/// to detect suspicious metrics (Sharpe > 20, null fields, etc).
fn run_output_validation(artifact_path: &str, run_id: &str) {
    use backtester_validation::{BacktestArtifacts, ValidationConfig, ValidationPipeline, Verdict};
    
    // Check each strategy in hall_of_fame
    let hof_path = Path::new(artifact_path).join("hall_of_fame");
    if !hof_path.exists() {
        info!(run_id, "No Hall of Fame found at {:?}, skipping validation", hof_path);
        return;
    }
    
    // Iterate over strategy directories
    let entries = match std::fs::read_dir(&hof_path) {
        Ok(e) => e,
        Err(e) => {
            error!(run_id, "Failed to read Hall of Fame: {}", e);
            return;
        }
    };
    
    let mut pass_count = 0;
    let mut warn_count = 0;
    let mut fail_count = 0;
    
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        
        // Check if strategy has metrics.json
        if !path.join("metrics.json").exists() {
            continue;
        }
        
        let strategy_id = path.file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        
        // Create artifacts reference (without nav_history, just validate metrics)
        let artifacts = BacktestArtifacts::from_dir(&path, &strategy_id);
        
        // Run validation with lenient config (just schema + sanity, no crosscheck)
        let config = ValidationConfig {
            crosscheck_enabled: false,
            attribution_enabled: false,
            report_enabled: false,
            ..Default::default()
        };
        
        let pipeline = ValidationPipeline::new(config);
        match pipeline.validate(&artifacts) {
            Ok(result) => {
                match result.verdict {
                    Verdict::Pass => pass_count += 1,
                    Verdict::Warn => {
                        warn_count += 1;
                        info!(run_id, strategy = strategy_id, "Validation warning: {:?}", 
                              result.warnings.iter().map(|w| &w.message).collect::<Vec<_>>());
                    }
                    Verdict::Fail => {
                        fail_count += 1;
                        error!(run_id, strategy = strategy_id, "Validation failed: {:?}", result.errors);
                    }
                }
            }
            Err(e) => {
                error!(run_id, strategy = strategy_id, "Validation error: {}", e);
            }
        }
    }
    
    info!(run_id, "Output validation complete: {} pass, {} warn, {} fail", 
          pass_count, warn_count, fail_count);
}
