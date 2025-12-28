//! Run command - Execute evolution experiment.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use combiner_core::{GenomeValidator, ParamRanges};
use combiner_engine::{
    EvolutionConfig, EvolutionEngine, ExperimentManifest, ExperimentPersistence,
    ExperimentStatus, Population, generate_experiment_id, UltraEvolutionResult,
    FinalReportGenerator,
};
use combiner_runner::{BacktestExecutor, CliExecutor, ValidationCache};

/// SCG configuration file format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScgConfig {
    /// Evolution configuration.
    #[serde(default)]
    pub evolution: EvolutionConfig,

    /// Output settings.
    #[serde(default)]
    pub output: OutputConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutputConfig {
    /// Output directory.
    #[serde(default = "default_output_dir")]
    pub directory: String,

    /// Save all generations.
    #[serde(default)]
    pub save_all_generations: bool,

    /// Save population each N generations.
    #[serde(default = "default_save_interval")]
    pub save_interval: u32,
}

fn default_output_dir() -> String {
    "output/scg".into()
}

fn default_save_interval() -> u32 {
    5
}

impl Default for ScgConfig {
    fn default() -> Self {
        Self {
            evolution: EvolutionConfig::default(),
            output: OutputConfig::default(),
        }
    }
}

/// Execute the run command.
pub fn execute(
    config_path: &str, 
    output_dir: &str, 
    seed: Option<u64>, 
    dry_run: bool,
    ultra: bool,
    top_k: usize,
) -> Result<()> {
    // Load configuration
    let config = load_config(config_path)?;
    info!("Loaded configuration from {}", config_path);

    // Override seed if provided
    let mut evo_config = config.evolution.clone();
    if let Some(s) = seed {
        evo_config.seed = Some(s);
    }

    if dry_run {
        return execute_dry_run(&evo_config);
    }

    // Create output directory
    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path)?;

    // Create executor
    let executor = CliExecutor::new()
        .with_output_dir(output_path.join("backtests"));

    // Track start time
    let start_time = std::time::Instant::now();

    // Create evolution engine
    let mut engine = EvolutionEngine::new(evo_config.clone(), executor);

    // Progress bar
    let pb = ProgressBar::new(evo_config.max_generations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    if ultra {
        info!(
            "Starting ULTRA evolution: {} generations, {} population, top-k={}",
            evo_config.max_generations, evo_config.population_size, top_k
        );

        // Create validation cache
        let validation_cache = Arc::new(ValidationCache::new());

        // Run ultra mode
        let result = engine.evolve_ultra(validation_cache, top_k)?;

        pb.finish_with_message("ULTRA evolution complete");

        // Save ultra results
        save_ultra_results(&engine, &result, output_path, &evo_config, start_time)?;

        // Print ultra summary
        print_ultra_summary(&result);
    } else {
        info!(
            "Starting evolution: {} generations, {} population",
            evo_config.max_generations, evo_config.population_size
        );

        // Run standard evolution
        engine.evolve()?;

        pb.finish_with_message("Evolution complete");

        // Save results
        save_results(&engine, output_path, &evo_config, start_time)?;

        // Print summary
        print_summary(&engine);
    }

    Ok(())
}

/// Execute dry run (validation only).
fn execute_dry_run(config: &EvolutionConfig) -> Result<()> {
    info!("Dry run: validating configuration and generating sample population");

    let param_ranges = ParamRanges::new();
    let validator = GenomeValidator::new();

    // Generate sample population
    let seed = config.seed.unwrap_or(42);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let population = Population::random(config.population_size.min(10), &mut rng, &param_ranges);

    info!("Generated {} sample genomes", population.len());

    // Validate each genome
    let mut valid = 0;
    let mut invalid = 0;

    for genome in &population.genomes {
        match validator.validate(genome) {
            Ok(()) => {
                valid += 1;
                // Try to convert to TOML
                match genome.to_toml() {
                    Ok(toml) => {
                        info!("Genome {} valid, {} blocks", &genome.id.to_string()[..8], genome.genes.len());
                        // Print first TOML as example
                        if valid == 1 {
                            println!("\n--- Sample TOML ---\n{}", toml);
                        }
                    }
                    Err(e) => {
                        warn!("Genome {} conversion failed: {}", &genome.id.to_string()[..8], e);
                        invalid += 1;
                    }
                }
            }
            Err(e) => {
                warn!("Genome {} invalid: {}", &genome.id.to_string()[..8], e);
                invalid += 1;
            }
        }
    }

    println!("\n--- Dry Run Summary ---");
    println!("Valid genomes: {}", valid);
    println!("Invalid genomes: {}", invalid);
    println!("Configuration OK");

    Ok(())
}

/// Load SCG configuration from TOML file.
fn load_config(path: &str) -> Result<ScgConfig> {
    if Path::new(path).exists() {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path))?;
        toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", path))
    } else {
        // Return default config if file doesn't exist
        warn!("Config file not found, using defaults");
        Ok(ScgConfig::default())
    }
}

/// Save evolution results using persistence.
fn save_results<E: BacktestExecutor>(
    engine: &EvolutionEngine<E>,
    output_path: &Path,
    config: &EvolutionConfig,
    start_time: std::time::Instant,
) -> Result<()> {
    let experiment_id = generate_experiment_id();
    let persistence = ExperimentPersistence::new(output_path, &experiment_id);

    // Initialize directory structure
    persistence.init()?;

    // Create manifest
    let manifest = ExperimentManifest {
        experiment_id: experiment_id.clone(),
        created_at: chrono::Utc::now(),
        seed: config.seed.unwrap_or(42),
        status: ExperimentStatus::Completed,
        generations_completed: engine.stats().len() as u32,
        total_evaluations: engine.stats().iter().map(|s| s.evaluated as u64).sum(),
        cache_hits: engine.stats().iter().map(|s| s.cache_hits as u64).sum(),
        duration_seconds: start_time.elapsed().as_secs(),
        final_pareto_size: engine.stats().last().map(|s| s.pareto_size).unwrap_or(0),
        config_hash: format!("{:x}", config.seed.unwrap_or(0)),
    };

    // Write manifest
    persistence.write_manifest(&manifest)?;

    // Write Hall of Fame
    persistence.write_hall_of_fame(engine.hall_of_fame())?;

    // Write report
    persistence.write_report(&manifest, engine.stats(), engine.hall_of_fame())?;

    info!(
        "Saved experiment {} with {} strategies to {:?}",
        experiment_id,
        engine.hall_of_fame().len(),
        output_path.join(&experiment_id)
    );

    Ok(())
}

/// Print summary of evolution results.
fn print_summary<E: BacktestExecutor>(engine: &EvolutionEngine<E>) {
    println!("\n=== Evolution Summary ===\n");

    println!("Generations completed: {}", engine.stats().len());
    println!("Hall of Fame size: {}", engine.hall_of_fame().len());

    if let Some(best) = engine.hall_of_fame().best() {
        println!("\nBest Strategy:");
        println!("  ID: {}", &best.genome.id.to_string()[..8]);
        println!("  Blocks: {}", best.genome.genes.len());

        if let Some(ref fitness) = best.genome.fitness {
            println!("  CAGR: {:.1}%", fitness.cagr * 100.0);
            println!("  Sharpe: {:.2}", fitness.sharpe_ratio);
            println!("  Max DD: {:.1}%", fitness.max_drawdown * 100.0);
        }
    }

    if let Some(last_stats) = engine.stats().last() {
        println!("\nFinal Generation Stats:");
        println!("  Pareto size: {}", last_stats.pareto_size);
        println!("  Best Sharpe: {:.2}", last_stats.best_sharpe);
        println!("  Mean Sharpe: {:.2}", last_stats.mean_sharpe);
    }
}

/// Save evolution results with ultra mode.
fn save_ultra_results<E: BacktestExecutor>(
    engine: &EvolutionEngine<E>,
    ultra_result: &UltraEvolutionResult,
    output_path: &Path,
    config: &EvolutionConfig,
    start_time: std::time::Instant,
) -> Result<()> {
    let experiment_id = generate_experiment_id();
    let persistence = ExperimentPersistence::new(output_path, &experiment_id);

    // Initialize directory structure
    persistence.init()?;

    // Create manifest
    let manifest = ExperimentManifest {
        experiment_id: experiment_id.clone(),
        created_at: chrono::Utc::now(),
        seed: config.seed.unwrap_or(42),
        status: ExperimentStatus::Completed,
        generations_completed: ultra_result.total_generations,
        total_evaluations: engine.stats().iter().map(|s| s.evaluated as u64).sum(),
        cache_hits: engine.stats().iter().map(|s| s.cache_hits as u64).sum(),
        duration_seconds: start_time.elapsed().as_secs(),
        final_pareto_size: engine.stats().last().map(|s| s.pareto_size).unwrap_or(0),
        config_hash: format!("{:x}", config.seed.unwrap_or(0)),
    };

    // Write manifest
    persistence.write_manifest(&manifest)?;

    // Write Hall of Fame
    persistence.write_hall_of_fame(engine.hall_of_fame())?;

    // Generate and save final report
    let report_generator = FinalReportGenerator::new(
        output_path.join(&experiment_id),
        &experiment_id,
        config.clone(),
    );

    let snapshots = ultra_result.performance_metrics.snapshots();
    let report_path = report_generator.generate_and_save(
        &ultra_result.validated_hall_of_fame,
        &ultra_result.performance_metrics,
        &snapshots,
    )?;

    info!(
        "ULTRA: Saved experiment {} with {} validated strategies",
        experiment_id,
        ultra_result.validated_hall_of_fame.len()
    );
    info!("Final report: {:?}", report_path);

    Ok(())
}

/// Print summary of ultra evolution results.
fn print_ultra_summary(result: &UltraEvolutionResult) {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    ULTRA EVOLUTION SUMMARY                   ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    
    let perf = result.performance_metrics.summary();
    
    println!("║ Generations: {:>5}   │   Time: {:>7.1}s                    ║", 
             result.total_generations, result.total_time_secs);
    println!("║ Genomes Evaluated: {:>8}                                  ║", 
             perf.total_genomes_evaluated);
    println!("║ Throughput: {:>7.1} genomes/sec                           ║", 
             perf.throughput_genomes_per_sec);
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                      CACHE PERFORMANCE                       ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Stage A Cache Hit Rate: {:>6.1}%                             ║", 
             perf.stage_a_cache_hit_rate);
    println!("║ Split Cache Hit Rate:   {:>6.1}%                             ║", 
             perf.split_cache_hit_rate);
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                    VALIDATED HALL OF FAME                    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Validated Strategies: {:>4}                                  ║", 
             result.validated_hall_of_fame.len());
    
    let hof_summary = result.validated_hall_of_fame.summary();
    if result.validated_hall_of_fame.len() > 0 {
        println!("║ Avg OOS Sharpe: {:>7.2}                                     ║", 
                 hof_summary.avg_oos_sharpe);
        println!("║ Avg PBO:        {:>7.2}                                     ║", 
                 hof_summary.avg_pbo);
        println!("║ Best OOS Sharpe: {:>6.2}                                     ║", 
                 hof_summary.best_oos_sharpe);
    }
    
    if let Some(best) = result.validated_hall_of_fame.best() {
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║                      TOP CANDIDATE                           ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ ID: {}...                                      ║", 
                 &best.genome_id.to_string()[..8]);
        println!("║ OOS Sharpe: {:>6.2}  │  PBO: {:>5.2}  │  Score: {:>6.2}      ║", 
                 best.validation.oos_sharpe_median, 
                 best.validation.pbo,
                 best.score);
        println!("║ Splits: {}/{}  │  Degradation: {:>5.1}%                     ║", 
                 best.validation.splits_passed, 
                 best.validation.splits_evaluated,
                 best.validation.degradation_pct);
    }
    
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                      TIME BREAKDOWN                          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Stage A: {:>5.1}%  │  Stage B: {:>5.1}%  │  Pareto: {:>5.1}%   ║", 
             perf.stage_a_time_pct, perf.stage_b_time_pct, perf.pareto_time_pct);
    println!("╚══════════════════════════════════════════════════════════════╝");
}

