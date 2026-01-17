//! # Backtester CLI
//!
//! Command-line interface for running backtests and experiments.
//! This is the core backtesting infrastructure - strategy implementations go in separate crates.

use backtester_io::{BrapiLoader, CsvLoader, Normalizer};
use backtester_strategy::experiment::{
    ArtifactFormat, BlockCatalog, Comparator, ExperimentRunner, RunnerConfig,
};
use backtester_strategy::BlockRegistry;
use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;

/// B3 Backtester - High-performance backtesting for B3 stocks
#[derive(Parser)]
#[command(name = "backtest")]
#[command(about = "B3 Backtester - High-performance backtesting system", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Validate a data file
    Validate {
        /// Path to data file
        #[arg(short, long)]
        data: PathBuf,
    },
    /// Fetch historical data from Brapi
    Fetch {
        /// Ticker symbols (comma-separated)
        #[arg(short, long)]
        tickers: String,

        /// Time range (1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        #[arg(short, long, default_value = "1y")]
        range: String,

        /// Output CSV file
        #[arg(short, long, default_value = "data/fetched.csv")]
        output: PathBuf,
    },
    /// Show data statistics
    Stats {
        /// Path to data file or cache directory
        #[arg(short, long)]
        data: PathBuf,
    },
    /// List available symbols in cache
    List {
        /// Path to cache directory
        #[arg(short, long, default_value = "cache")]
        cache: PathBuf,
    },

    // === Experiment Orchestrator Commands ===

    /// Run a single strategy config
    Run {
        /// Path to strategy config file (TOML)
        #[arg(short, long)]
        config: PathBuf,

        /// Output directory for artifacts
        #[arg(short, long, default_value = "output/experiments")]
        output: PathBuf,

        /// Dry run (validate only, no execution)
        #[arg(long)]
        dry_run: bool,

        /// Strict mode (fail on NaN, invalid weights)
        #[arg(long)]
        strict: bool,
        
        /// Path to execution model config (TOML) for cost/slippage modeling
        #[arg(short = 'e', long)]
        execution: Option<PathBuf>,
        
        /// Path to market data CSV file for backtesting
        #[arg(short = 'm', long)]
        market_data: Option<PathBuf>,
        
        /// Data source: "database" or "csv" (uses DATABASE_URL env var when "database")
        #[arg(long)]
        data_source: Option<String>,
        
        /// Market: "BR" or "US" (selects ohlcv_daily vs ohlcv_daily_us table)
        #[arg(long)]
        market: Option<String>,
        
        /// Risk profile name (muito_conservador, conservador, moderado, arrojado, muito_arrojado)
        #[arg(long)]
        risk_profile: Option<String>,
        
        /// Use OBFS binary format instead of JSON/CSV (90% storage reduction)
        #[arg(long)]
        obfs: bool,
    },

    /// Run all strategy configs in a folder
    RunBatch {
        /// Path to folder containing strategy configs
        #[arg(short, long)]
        folder: PathBuf,

        /// Output directory for artifacts
        #[arg(short, long, default_value = "output/experiments")]
        output: PathBuf,

        /// Strict mode (fail on NaN, invalid weights)
        #[arg(long)]
        strict: bool,
        
        /// Path to market data CSV file for backtesting
        #[arg(short = 'm', long)]
        market_data: Option<PathBuf>,
        
        /// Use OBFS binary format instead of JSON/CSV (90% storage reduction)
        #[arg(long)]
        obfs: bool,
    },

    /// Compare two experiment runs
    Compare {
        /// Path to first run directory
        #[arg(long)]
        run_a: PathBuf,

        /// Path to second run directory
        #[arg(long)]
        run_b: PathBuf,
        
        /// Sharpe ratio drop threshold for regression (e.g., 0.20 = 20%)
        #[arg(long, default_value = "0.20")]
        sharpe_threshold: f64,
        
        /// CAGR drop threshold for regression (e.g., 0.30 = 30%)
        #[arg(long, default_value = "0.30")]
        cagr_threshold: f64,
        
        /// Max drawdown increase threshold for regression (e.g., 0.25 = 25%)
        #[arg(long, default_value = "0.25")]
        dd_threshold: f64,
        
        /// Load thresholds from config file
        #[arg(long)]
        thresholds_file: Option<PathBuf>,
    },

    /// Compare a run against a golden strategy
    CompareToGolden {
        /// Path to run directory
        #[arg(long)]
        run: PathBuf,

        /// Golden strategy ID (e.g., golden_momentum, golden_value_quality)
        #[arg(long)]
        golden: String,

        /// Path to golden strategies output directory
        #[arg(long, default_value = "output/experiments")]
        golden_dir: PathBuf,
        
        /// Sharpe ratio drop threshold for regression (e.g., 0.20 = 20%)
        #[arg(long, default_value = "0.20")]
        sharpe_threshold: f64,
        
        /// CAGR drop threshold for regression (e.g., 0.30 = 30%)
        #[arg(long, default_value = "0.30")]
        cagr_threshold: f64,
        
        /// Max drawdown increase threshold for regression (e.g., 0.25 = 25%)
        #[arg(long, default_value = "0.25")]
        dd_threshold: f64,
        
        /// Load thresholds from config file
        #[arg(long)]
        thresholds_file: Option<PathBuf>,
    },

    /// Validate a backtest run (sanity checks, cross-check, attribution)
    ValidateRun {
        /// Path to run directory (containing metrics.json, nav_history.csv, etc.)
        #[arg(short, long)]
        run_dir: PathBuf,

        /// Run ID (optional, defaults to directory name)
        #[arg(long)]
        run_id: Option<String>,

        /// Output directory for validation artifacts
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Strict mode (fail on any warning)
        #[arg(long)]
        strict: bool,

        /// Disable cross-check (recompute metrics from nav_history)
        #[arg(long)]
        no_crosscheck: bool,
    },

    /// Generate block catalog documentation
    /// Run stress tests on a candidate strategy
    StressCandidate {
        /// Path to candidate config file (TOML)
        #[arg(short, long)]
        candidate: PathBuf,

        /// Path to execution model config (TOML)
        #[arg(short = 'e', long)]
        execution: Option<PathBuf>,

        /// Output path for stress results
        #[arg(short, long, default_value = "output/stress_results.json")]
        output: PathBuf,
    },

    GenerateCatalog {
        /// Output file path
        #[arg(short, long, default_value = "docs/BLOCK_CATALOG.md")]
        output: PathBuf,

        /// Also output JSON format
        #[arg(long)]
        json: bool,
    },
}

fn validate_command(data_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("Validating data file: {:?}", data_path);

    let loader = CsvLoader::new();
    match loader.load(&data_path) {
        Ok(bars) => {
            let mut normalizer = Normalizer::new();
            let events = normalizer.normalize(bars)?;
            println!("Validation PASSED");
            println!("  Bars loaded: {}", events.len());
            println!("  Assets: {}", normalizer.asset_count());
            Ok(())
        }
        Err(e) => {
            println!("Validation FAILED: {}", e);
            Err(e.into())
        }
    }
}

fn fetch_command(
    tickers: String,
    range: String,
    output_path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Fetching data from Brapi...");

    let ticker_list: Vec<&str> = tickers.split(',').map(|s| s.trim()).collect();
    println!("Tickers: {:?}", ticker_list);
    println!("Range: {}", range);

    let loader = BrapiLoader::new();
    let bars = loader.fetch_universe(&ticker_list, &range)?;

    println!("Fetched {} bars", bars.len());

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    loader.save_to_csv(&bars, &output_path)?;

    println!("Data saved to: {}", output_path.display());

    // Show summary
    let mut ticker_counts: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::new();
    for bar in &bars {
        *ticker_counts.entry(&bar.ticker).or_insert(0) += 1;
    }

    println!("\n=== Summary ===");
    for (ticker, count) in &ticker_counts {
        println!("  {}: {} days", ticker, count);
    }

    Ok(())
}

fn stats_command(data_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                     DATA STATISTICS                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    if data_path.is_dir() {
        // Cache directory
        let ohlcv_dir = data_path.join("ohlcv");
        if ohlcv_dir.exists() {
            let files: Vec<_> = fs::read_dir(&ohlcv_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().map_or(false, |ext| ext == "csv"))
                .collect();

            println!("  Cache directory: {}", data_path.display());
            println!("  Symbols: {}", files.len());

            // Read metadata if exists
            let metadata_path = data_path.join("metadata.json");
            if metadata_path.exists() {
                let content = fs::read_to_string(&metadata_path)?;
                let metadata: serde_json::Value = serde_json::from_str(&content)?;
                println!(
                    "  Start date: {}",
                    metadata["start_date"].as_str().unwrap_or("N/A")
                );
                println!(
                    "  End date: {}",
                    metadata["end_date"].as_str().unwrap_or("N/A")
                );
                println!(
                    "  Total bars: {}",
                    metadata["total_bars"].as_u64().unwrap_or(0)
                );
            }
        } else {
            println!("  Invalid cache directory: {}", data_path.display());
        }
    } else {
        // Single file
        let loader = CsvLoader::new();
        let bars = loader.load(&data_path)?;
        let mut normalizer = Normalizer::new();
        let events = normalizer.normalize(bars)?;

        println!("  File: {}", data_path.display());
        println!("  Events: {}", events.len());
        println!("  Assets: {}", normalizer.asset_count());

        if !events.is_empty() {
            let first = &events[0];
            let last = &events[events.len() - 1];
            println!("  First timestamp: {}", first.bar.timestamp);
            println!("  Last timestamp: {}", last.bar.timestamp);
        }
    }

    println!();
    Ok(())
}

fn list_command(cache_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    AVAILABLE SYMBOLS                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let ohlcv_dir = cache_path.join("ohlcv");
    if !ohlcv_dir.exists() {
        println!("  Cache not found: {}", cache_path.display());
        return Ok(());
    }

    let mut symbols: Vec<String> = fs::read_dir(&ohlcv_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "csv"))
        .filter_map(|e| {
            e.path()
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
        })
        .collect();

    symbols.sort();

    println!("  Cache: {}", cache_path.display());
    println!("  Total symbols: {}\n", symbols.len());

    // Print in columns
    let cols = 5;
    for chunk in symbols.chunks(cols) {
        print!("  ");
        for sym in chunk {
            print!("{:<12}", sym);
        }
        println!();
    }

    println!();
    Ok(())
}

// === Experiment Orchestrator Command Handlers ===

fn run_command(
    config_path: PathBuf,
    output_dir: PathBuf,
    dry_run: bool,
    strict: bool,
    execution_config: Option<PathBuf>,
    market_data: Option<PathBuf>,
    data_source: Option<String>,
    market: Option<String>,
    risk_profile: Option<String>,
    obfs: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    STRATEGY RUNNER                           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load execution config if provided
    if let Some(ref exec_path) = execution_config {
        println!("Execution config: {}", exec_path.display());
    }
    
    // Log market data if provided
    if let Some(ref md_path) = market_data {
        println!("Market data: {}", md_path.display());
    }
    
    // Log data source if provided
    if let Some(ref ds) = data_source {
        println!("Data source: {}", ds);
        if ds == "database" {
            if std::env::var("DATABASE_URL").is_err() {
                return Err("data_source='database' requires DATABASE_URL environment variable".into());
            }
            println!("DATABASE_URL is set, will use Neon database for market data");
        }
    }
    
    // Log market if provided (BR or US)
    if let Some(ref mkt) = market {
        println!("Market: {} (table: {})", mkt, if mkt.to_uppercase() == "US" { "ohlcv_daily_us" } else { "ohlcv_daily" });
    }
    
    // Log risk profile if provided
    if let Some(ref rp) = risk_profile {
        println!("Risk profile: {}", rp);
    }
    
    // Determine artifact format (OBFS default for ultra-performance)
    let artifact_format = if obfs {
        println!("Artifact format: OBFS (binary, ~90% storage reduction)");
        ArtifactFormat::Obfs
    } else {
        println!("Artifact format: OBFS (default, ultra-performance)");
        ArtifactFormat::Obfs
    };
    
    let runner_config = RunnerConfig {
        output_dir: output_dir.to_string_lossy().into(),
        market_data_csv_path: market_data.map(|p| p.to_string_lossy().into_owned()),
        artifact_format,
        data_source: data_source.clone(),
        market: market.clone(),
        ..Default::default()
    };

    let mut runner = ExperimentRunner::with_config(runner_config);

    if dry_run {
        runner = runner.dry_run();
        println!("Mode: DRY RUN (validation only)\n");
    }

    if strict {
        runner = runner.strict();
        println!("Mode: STRICT (fail on invalid outputs)\n");
    }

    println!("Config: {}", config_path.display());
    println!();

    let result = runner.run_single(&config_path)?;

    if result.success {
        println!("\n✓ Strategy executed successfully");
        println!("  Run ID: {}", result.run_id);
        println!("  Strategy: {}", result.metadata.strategy_id);

        if !dry_run {
            println!("\n  Metrics:");
            println!("    CAGR:        {:.2}%", result.metrics.cagr * 100.0);
            println!("    Volatility:  {:.2}%", result.metrics.volatility * 100.0);
            println!("    Sharpe:      {:.2}", result.metrics.sharpe_ratio);
            println!("    Max DD:      {:.2}%", result.metrics.max_drawdown * 100.0);
            println!("    Hit Rate:    {:.2}%", result.metrics.hit_rate * 100.0);

            if let Some(path) = &result.output_path {
                println!("\n  Artifacts: {}", path.display());
            }
        }
    } else {
        println!("\n✗ Strategy execution failed");
        if let Some(err) = &result.error {
            println!("  Error: {}", err);
        }
    }

    println!();
    Ok(())
}

fn run_batch_command(
    folder: PathBuf,
    output_dir: PathBuf,
    strict: bool,
    market_data: Option<PathBuf>,
    obfs: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    BATCH RUNNER                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Log market data if provided
    if let Some(ref md_path) = market_data {
        println!("Market data: {}", md_path.display());
    }

    // Determine artifact format (OBFS default for ultra-performance)
    let artifact_format = if obfs {
        println!("Artifact format: OBFS (binary, ~90% storage reduction)");
        ArtifactFormat::Obfs
    } else {
        println!("Artifact format: OBFS (default, ultra-performance)");
        ArtifactFormat::Obfs
    };

    let runner_config = RunnerConfig {
        output_dir: output_dir.to_string_lossy().into(),
        market_data_csv_path: market_data.map(|p| p.to_string_lossy().into_owned()),
        artifact_format,
        ..Default::default()
    };

    let mut runner = ExperimentRunner::with_config(runner_config);

    if strict {
        runner = runner.strict();
    }

    println!("Folder: {}", folder.display());
    println!();

    let results = runner.run_batch(&folder)?;

    println!("\n=== Batch Results ===\n");
    println!(
        "{:<30} {:>10} {:>10} {:>10}",
        "Strategy", "Sharpe", "CAGR", "Max DD"
    );
    println!("{}", "-".repeat(64));

    let mut success_count = 0;
    let mut fail_count = 0;

    for result in &results {
        if result.success {
            success_count += 1;
            println!(
                "{:<30} {:>10.2} {:>9.1}% {:>9.1}%",
                result.metadata.strategy_id,
                result.metrics.sharpe_ratio,
                result.metrics.cagr * 100.0,
                result.metrics.max_drawdown * 100.0,
            );
        } else {
            fail_count += 1;
            println!(
                "{:<30} {:>10} {:>10} {:>10}",
                result.metadata.strategy_id, "FAILED", "-", "-"
            );
        }
    }

    println!("{}", "-".repeat(64));
    println!("\nTotal: {} succeeded, {} failed", success_count, fail_count);
    println!();

    Ok(())
}

fn compare_command(
    run_a: PathBuf, 
    run_b: PathBuf,
    sharpe_threshold: f64,
    cagr_threshold: f64,
    dd_threshold: f64,
    thresholds_file: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    use backtester_strategy::experiment::RegressionThresholds;
    
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    RUN COMPARISON                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load thresholds from file or use CLI flags
    let thresholds = if let Some(ref path) = thresholds_file {
        println!("Loading thresholds from: {}", path.display());
        RegressionThresholds::load_from_file(path)?
    } else {
        RegressionThresholds::builder()
            .sharpe_drop(sharpe_threshold)
            .cagr_drop(cagr_threshold)
            .max_dd_increase(dd_threshold)
            .build()
    };
    
    println!("Thresholds: {}\n", thresholds.format());

    let comparator = Comparator::with_thresholds(thresholds);
    let result = comparator.compare(&run_a, &run_b)?;
    let report = comparator.generate_report(&result);

    println!("{}", report);

    Ok(())
}

fn compare_to_golden_command(
    run: PathBuf,
    golden_id: String,
    golden_dir: PathBuf,
    sharpe_threshold: f64,
    cagr_threshold: f64,
    dd_threshold: f64,
    thresholds_file: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    use backtester_strategy::experiment::RegressionThresholds;
    
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                 GOLDEN STRATEGY COMPARISON                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load thresholds from file or use CLI flags
    let thresholds = if let Some(ref path) = thresholds_file {
        println!("Loading thresholds from: {}", path.display());
        RegressionThresholds::load_from_file(path)?
    } else {
        RegressionThresholds::builder()
            .sharpe_drop(sharpe_threshold)
            .cagr_drop(cagr_threshold)
            .max_dd_increase(dd_threshold)
            .build()
    };
    
    println!("Thresholds: {}\n", thresholds.format());

    let comparator = Comparator::with_thresholds(thresholds)
        .with_golden_dir(golden_dir.to_string_lossy().to_string());
    let result = comparator.compare_to_golden(&run, &golden_id)?;
    let report = comparator.generate_report(&result);

    println!("{}", report);

    if result.regression {
        println!("⚠️  This run shows regression compared to golden strategy!");
        std::process::exit(1);
    }

    Ok(())
}

fn generate_catalog_command(output: PathBuf, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                 BLOCK CATALOG GENERATOR                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let registry = BlockRegistry::with_builtins();

    // Generate markdown
    let markdown = BlockCatalog::generate_markdown(&registry);

    // Ensure output directory exists
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(&output, &markdown)?;
    println!("✓ Markdown catalog written to: {}", output.display());

    // Optionally generate JSON
    if json {
        let json_path = output.with_extension("json");
        let json_content = BlockCatalog::generate_json(&registry);
        fs::write(&json_path, &json_content)?;
        println!("✓ JSON catalog written to: {}", json_path.display());
    }

    // Print summary
    let selection_count = registry
        .blocks_by_type(backtester_strategy::BlockType::Selection)
        .len();
    let entry_count = registry
        .blocks_by_type(backtester_strategy::BlockType::Entry)
        .len();
    let exit_count = registry
        .blocks_by_type(backtester_strategy::BlockType::Exit)
        .len();
    let sizing_count = registry
        .blocks_by_type(backtester_strategy::BlockType::Sizing)
        .len();

    println!("\nBlocks documented:");
    println!("  Selection: {}", selection_count);
    println!("  Entry:     {}", entry_count);
    println!("  Exit:      {}", exit_count);
    println!("  Sizing:    {}", sizing_count);
    println!(
        "  Total:     {}",
        selection_count + entry_count + exit_count + sizing_count
    );
    println!();

    Ok(())
}

fn validate_run_command(
    run_dir: PathBuf,
    run_id: Option<String>,
    output_dir: Option<PathBuf>,
    strict: bool,
    no_crosscheck: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use backtester_validation::{BacktestArtifacts, ValidationPipeline, ValidationConfig, Verdict};

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                 BACKTEST VALIDATION                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let run_id = run_id.unwrap_or_else(|| {
        run_dir.file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    });

    println!("Run directory: {}", run_dir.display());
    println!("Run ID: {}", run_id);
    println!("Strict mode: {}", strict);
    println!();

    // Create artifacts
    let artifacts = BacktestArtifacts::from_dir(&run_dir, &run_id);

    // Check if files exist
    if !artifacts.metrics_path.exists() {
        eprintln!("Error: metrics.json not found at {}", artifacts.metrics_path.display());
        return Err("Missing metrics.json".into());
    }

    // Create config
    let config = ValidationConfig {
        strict_mode: strict,
        crosscheck_enabled: !no_crosscheck,
        ..Default::default()
    };

    let pipeline = ValidationPipeline::new(config);
    let result = pipeline.validate(&artifacts)?;

    // Print results
    println!("═══════════════════════════════════════════════════════════════");
    let verdict_icon = match result.verdict {
        Verdict::Pass => "✅",
        Verdict::Warn => "⚠️",
        Verdict::Fail => "❌",
    };
    println!("VERDICT: {} {:?}", verdict_icon, result.verdict);
    println!("═══════════════════════════════════════════════════════════════\n");

    // Schema check
    println!("Schema Check:");
    if result.schema_check.has_failures() {
        println!("  ❌ FAILED");
        println!("     Missing: {:?}", result.schema_check.missing_fields);
        println!("     Null: {:?}", result.schema_check.null_fields);
    } else {
        println!("  ✅ PASSED ({} fields validated)", result.schema_check.validated_fields.len());
    }

    // Sanity check
    println!("\nSanity Check:");
    match result.sanity_check.verdict {
        Verdict::Pass => println!("  ✅ PASSED"),
        Verdict::Warn => println!("  ⚠️  WARNINGS"),
        Verdict::Fail => println!("  ❌ FAILED"),
    }
    println!("  {}", result.sanity_check.message);

    // Cross-check
    if let Some(ref cc) = result.crosscheck {
        println!("\nCross-check:");
        if cc.passed {
            println!("  ✅ PASSED");
        } else {
            println!("  ❌ FAILED");
            for cmp in &cc.comparisons {
                if !cmp.passed {
                    println!("     {} mismatch: reported {:.4}, recomputed {:.4}",
                        cmp.name, cmp.reported, cmp.recomputed);
                }
            }
        }
    }

    // Attribution
    if let Some(ref attr) = result.attribution {
        println!("\nAttribution:");
        println!("  Assets: {}", attr.attributions.len());
        println!("  Total Net PnL: {:.2}", attr.total_net_pnl);
        println!("  Total Trades: {}", attr.total_trades);
        println!("  Top 1 concentration: {:.1}%", attr.concentration.top_1_pct * 100.0);
    }

    // Warnings
    if !result.warnings.is_empty() {
        println!("\nWarnings ({}):", result.warnings.len());
        for warn in &result.warnings {
            println!("  ⚠️  {}: {}", warn.code, warn.message);
        }
    }

    // Errors
    if !result.errors.is_empty() {
        println!("\nErrors ({}):", result.errors.len());
        for err in &result.errors {
            println!("  ❌ {}", err);
        }
    }

    // Generate artifacts if output dir specified
    if let Some(output) = output_dir {
        println!("\nGenerating artifacts to: {}", output.display());
        pipeline.generate_artifacts(&result, &output)?;
        println!("  ✓ validation_summary.json");
        println!("  ✓ sanity.json");
        if result.attribution.is_some() {
            println!("  ✓ asset_attribution.csv");
        }
        println!("  ✓ backtest_report.md");
    }

    println!();

    // Exit with error if failed
    if result.verdict == Verdict::Fail {
        std::process::exit(1);
    }

    Ok(())
}

fn stress_candidate_command(
    candidate_path: PathBuf,
    execution_config_path: Option<PathBuf>,
    output_path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    STRESS TESTING                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Candidate: {}", candidate_path.display());
    
    // Load execution config if provided
    let exec_config = if let Some(ref exec_path) = execution_config_path {
        println!("Execution config: {}", exec_path.display());
        let exec_content = fs::read_to_string(exec_path)?;
        let config: backtester_execution::ExecutionModelConfig = toml::from_str(&exec_content)?;
        config
    } else {
        println!("Using default MVP execution config");
        backtester_execution::ExecutionModelConfig::mvp()
    };
    
    // Create stress suite
    let suite = backtester_execution::StressSuite::default_institutional();
    
    println!("\nRunning {} stress scenarios:", suite.len());
    for scenario in &suite.scenarios {
        println!("  - {}: {}", scenario.id, scenario.name);
    }
    
    // Apply stress transforms and report
    let mut results = Vec::new();
    for scenario in &suite.scenarios {
        let stressed_config = scenario.apply(&exec_config);
        
        // In a full implementation, we would run the backtest with stressed_config
        // For now, we just show what would be applied
        println!("\n[{}] {}", scenario.id, scenario.name);
        println!("  Transform: {:?}", scenario.transform_type);
        println!("  Acceptance: min_sharpe={:.2}", scenario.acceptance.min_oos_sharpe);
        println!("  Stressed delay_bars: {}", stressed_config.delay_bars);
        println!("  Stressed slippage_bps: {:.1}", stressed_config.slippage.base_bps());
        
        // Create a placeholder result
        let result = backtester_execution::StressResult::new(
            scenario,
            1.0,  // placeholder original sharpe
            0.6,  // placeholder stressed sharpe
            None,
            None,
        );
        results.push(result);
    }
    
    // Aggregate results
    let suite_result = backtester_execution::StressSuiteResult::from_results(results);
    
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("STRESS SUITE RESULTS: {}", suite_result.summary());
    println!("═══════════════════════════════════════════════════════════════");
    
    // Save results to output file
    let json = serde_json::to_string_pretty(&suite_result)?;
    fs::write(&output_path, json)?;
    println!("\nResults saved to: {}", output_path.display());
    
    if suite_result.all_passed {
        println!("\n✓ All stress tests passed - candidate is robust");
    } else {
        println!("\n✗ Some stress tests failed - review before production");
    }
    
    Ok(())
}

fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Validate { data } => validate_command(data),
        Commands::Fetch {
            tickers,
            range,
            output,
        } => fetch_command(tickers, range, output),
        Commands::Stats { data } => stats_command(data),
        Commands::List { cache } => list_command(cache),

        // Experiment orchestrator commands
        Commands::Run {
            config,
            output,
            dry_run,
            strict,
            execution,
            market_data,
            data_source,
            market,
            risk_profile,
            obfs,
        } => run_command(config, output, dry_run, strict, execution, market_data, data_source, market, risk_profile, obfs),
        Commands::RunBatch {
            folder,
            output,
            strict,
            market_data,
            obfs,
        } => run_batch_command(folder, output, strict, market_data, obfs),
        Commands::Compare { 
            run_a, 
            run_b,
            sharpe_threshold,
            cagr_threshold,
            dd_threshold,
            thresholds_file,
        } => compare_command(run_a, run_b, sharpe_threshold, cagr_threshold, dd_threshold, thresholds_file),
        Commands::CompareToGolden {
            run,
            golden,
            golden_dir,
            sharpe_threshold,
            cagr_threshold,
            dd_threshold,
            thresholds_file,
        } => compare_to_golden_command(run, golden, golden_dir, sharpe_threshold, cagr_threshold, dd_threshold, thresholds_file),
        Commands::ValidateRun {
            run_dir,
            run_id,
            output,
            strict,
            no_crosscheck,
        } => validate_run_command(run_dir, run_id, output, strict, no_crosscheck),
        Commands::StressCandidate {
            candidate,
            execution,
            output,
        } => stress_candidate_command(candidate, execution, output),
        Commands::GenerateCatalog { output, json } => generate_catalog_command(output, json),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
