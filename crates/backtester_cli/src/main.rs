//! # Backtester CLI
//!
//! Command-line interface for running backtests.

use backtester_engine::SimulationEngine;
use backtester_execution::{AdvancedExecutionModel, CostModel, ExecutionConfig, LiquidityModel, SlippageModel};
use backtester_io::{BrapiLoader, CsvLoader, Normalizer};
use backtester_reports::RunManifest;
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use strategy_lib::{BuyAndHoldStrategy, DailyTrendStrategy, MeanReversionStrategy, NoOpStrategy, PairsSpreadStrategy, Strategy};

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
    /// Run a backtest with configuration file
    Run {
        /// Path to configuration TOML file
        #[arg(short, long)]
        config: PathBuf,

        /// Output directory for results
        #[arg(short, long, default_value = "output")]
        output: PathBuf,
    },
    /// Validate a data file
    Validate {
        /// Path to data file
        #[arg(short, long)]
        data: PathBuf,
    },
    /// Run determinism test (execute twice and compare hashes)
    Determinism {
        /// Path to configuration TOML file
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Validate anti-look-ahead with trap dataset
    Lookahead {
        /// Path to configuration TOML file
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Generate benchmark comparison report
    Compare {
        /// Path to baseline JSON
        #[arg(short, long, default_value = "benches/results/baseline.json")]
        baseline: PathBuf,

        /// Output path for comparison.md
        #[arg(short, long, default_value = "output/comparison.md")]
        output: PathBuf,
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
}

/// Backtest configuration (loaded from TOML).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// General settings
    pub general: GeneralConfig,
    /// Data settings
    pub data: DataConfig,
    /// Strategy settings
    pub strategy: StrategyConfig,
    /// Execution settings
    #[serde(default)]
    pub execution: ExecutionModelConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Backtest name
    pub name: String,
    /// Initial capital
    pub initial_capital: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Path to data file
    pub path: PathBuf,
    /// Skip invalid bars
    #[serde(default)]
    pub skip_invalid: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Strategy type: "buy_and_hold", "trend", "mean_reversion", "pairs", "noop"
    #[serde(rename = "type")]
    pub strategy_type: String,
    /// Short MA period (for trend strategy)
    #[serde(default = "default_short_period")]
    pub short_period: usize,
    /// Long MA period (for trend strategy)
    #[serde(default = "default_long_period")]
    pub long_period: usize,
    /// Threshold (for mean reversion / pairs entry in std devs)
    #[serde(default = "default_threshold")]
    pub threshold: f64,
    /// Max trades per day (for mean reversion)
    #[serde(default = "default_max_trades")]
    pub max_trades_per_day: u32,
    /// Asset A for pairs strategy
    #[serde(default)]
    pub asset_a: u32,
    /// Asset B for pairs strategy
    #[serde(default = "default_asset_b")]
    pub asset_b: u32,
    /// Lookback for pairs spread statistics
    #[serde(default = "default_lookback")]
    pub lookback: usize,
}

fn default_short_period() -> usize { 20 }
fn default_long_period() -> usize { 50 }
fn default_threshold() -> f64 { 0.01 }
fn default_max_trades() -> u32 { 10 }
fn default_asset_b() -> u32 { 1 }
fn default_lookback() -> usize { 20 }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecutionModelConfig {
    /// Slippage type: "none", "constant", "volume", "volatility"
    #[serde(default = "default_slippage_type")]
    pub slippage_type: String,
    /// Slippage parameter (bps for constant, coefficient for others)
    #[serde(default = "default_slippage_param")]
    pub slippage_param: f64,
    /// Fixed cost per trade
    #[serde(default = "default_fixed_cost")]
    pub fixed_cost: f64,
    /// Proportional cost in bps
    #[serde(default = "default_prop_cost")]
    pub proportional_bps: f64,
    /// Max volume participation rate
    #[serde(default = "default_max_participation")]
    pub max_participation: f64,
    /// Allow partial fills
    #[serde(default = "default_allow_partial")]
    pub allow_partial: bool,
}

fn default_slippage_type() -> String { "constant".to_string() }
fn default_slippage_param() -> f64 { 5.0 }
fn default_fixed_cost() -> f64 { 10.0 }
fn default_prop_cost() -> f64 { 10.0 }
fn default_max_participation() -> f64 { 0.1 }
fn default_allow_partial() -> bool { true }

impl ExecutionModelConfig {
    fn to_execution_config(&self) -> ExecutionConfig {
        let slippage = match self.slippage_type.as_str() {
            "none" => SlippageModel::None,
            "volume" => SlippageModel::VolumeLinear { coefficient: self.slippage_param },
            "volatility" => SlippageModel::Volatility { coefficient: self.slippage_param },
            _ => SlippageModel::Constant { bps: self.slippage_param },
        };

        ExecutionConfig {
            slippage,
            costs: CostModel::new(self.fixed_cost, self.proportional_bps, 0.0),
            liquidity: LiquidityModel::new(self.max_participation, self.allow_partial),
        }
    }
}

/// Run backtest with a specific strategy type.
fn run_backtest_with_strategy<S: Strategy>(
    mut strategy: S,
    config: &BacktestConfig,
    events: Vec<backtester_core::MarketEvent>,
    num_assets: usize,
) -> backtester_engine::BacktestResult {
    strategy.on_init(&backtester_core::StrategyConfig::default(), num_assets);
    
    let exec_config = config.execution.to_execution_config();
    let execution_model = AdvancedExecutionModel::new(exec_config);
    
    let mut engine = SimulationEngine::new(
        strategy,
        execution_model,
        config.general.initial_capital,
        num_assets,
    );
    
    engine.run(events)
}

fn run_command(config_path: PathBuf, output_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config_str = fs::read_to_string(&config_path)?;
    let config: BacktestConfig = toml::from_str(&config_str)?;
    
    println!("Running backtest: {}", config.general.name);
    println!("Loading data from: {:?}", config.data.path);
    
    // Load and normalize data
    let loader = CsvLoader::new().skip_invalid(config.data.skip_invalid);
    let raw_bars = loader.load(&config.data.path)?;
    let mut normalizer = Normalizer::new();
    let events = normalizer.normalize(raw_bars)?;
    let num_assets = normalizer.asset_count();
    
    println!("Loaded {} events for {} assets", events.len(), num_assets);
    
    // Run backtest based on strategy type
    let result = match config.strategy.strategy_type.as_str() {
        "buy_and_hold" => {
            run_backtest_with_strategy(BuyAndHoldStrategy::new(), &config, events, num_assets)
        }
        "trend" => {
            let strategy = DailyTrendStrategy::new(
                config.strategy.short_period,
                config.strategy.long_period,
            );
            run_backtest_with_strategy(strategy, &config, events, num_assets)
        }
        "mean_reversion" => {
            let strategy = MeanReversionStrategy::new(
                config.strategy.threshold,
                config.strategy.max_trades_per_day,
            );
            run_backtest_with_strategy(strategy, &config, events, num_assets)
        }
        "pairs" => {
            let strategy = PairsSpreadStrategy::new(
                config.strategy.asset_a,
                config.strategy.asset_b,
                config.strategy.threshold.max(0.5), // Use threshold as std dev entry
                config.strategy.lookback,
            );
            run_backtest_with_strategy(strategy, &config, events, num_assets)
        }
        "noop" | _ => {
            run_backtest_with_strategy(NoOpStrategy, &config, events, num_assets)
        }
    };
    
    // Create output directory
    fs::create_dir_all(&output_path)?;
    
    // Generate run manifest
    let manifest = RunManifest::new(
        &config.general.name,
        &config_path,
        &config.data.path,
    );
    
    // Save results
    let results_path = output_path.join("results.json");
    let results_json = serde_json::json!({
        "name": config.general.name,
        "initial_capital": config.general.initial_capital,
        "final_nav": result.final_nav,
        "total_return": (result.final_nav - config.general.initial_capital) / config.general.initial_capital,
        "max_drawdown": result.max_drawdown,
        "events_processed": result.events_processed,
        "trades_executed": result.trades_executed,
        "total_realized_pnl": result.total_realized_pnl,
        "total_costs": result.total_costs,
    });
    fs::write(&results_path, serde_json::to_string_pretty(&results_json)?)?;
    
    // Save manifest
    let manifest_path = output_path.join("run_manifest.json");
    manifest.save(&manifest_path)?;
    
    // Calculate and save result hash
    let result_hash = manifest.calculate_result_hash(&results_json.to_string());
    let hash_path = output_path.join("results.hash");
    fs::write(&hash_path, &result_hash)?;
    
    println!("\n=== Backtest Results ===");
    println!("Initial Capital: R$ {:.2}", config.general.initial_capital);
    println!("Final NAV:       R$ {:.2}", result.final_nav);
    println!("Total Return:    {:.2}%", (result.final_nav - config.general.initial_capital) / config.general.initial_capital * 100.0);
    println!("Max Drawdown:    {:.2}%", result.max_drawdown * 100.0);
    println!("Events:          {}", result.events_processed);
    println!("Trades:          {}", result.trades_executed);
    println!("Realized PnL:    R$ {:.2}", result.total_realized_pnl);
    println!("Total Costs:     R$ {:.2}", result.total_costs);
    println!("\nResults saved to: {:?}", output_path);
    println!("Result Hash: {}", result_hash);
    
    Ok(())
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

fn determinism_command(config_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running determinism test...");
    
    // Run twice with temporary outputs
    let tmp1 = PathBuf::from("/tmp/backtest_det_1");
    let tmp2 = PathBuf::from("/tmp/backtest_det_2");
    
    run_command(config_path.clone(), tmp1.clone())?;
    run_command(config_path, tmp2.clone())?;
    
    // Compare hashes
    let hash1 = fs::read_to_string(tmp1.join("results.hash"))?;
    let hash2 = fs::read_to_string(tmp2.join("results.hash"))?;
    
    if hash1 == hash2 {
        println!("\nDeterminism test PASSED");
        println!("Hash: {}", hash1);
        Ok(())
    } else {
        println!("\nDeterminism test FAILED");
        println!("Hash 1: {}", hash1);
        println!("Hash 2: {}", hash2);
        Err("Determinism violation: hashes do not match".into())
    }
}

fn lookahead_command(config_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running anti-look-ahead validation...");
    
    let tmp = PathBuf::from("/tmp/backtest_lookahead");
    run_command(config_path, tmp.clone())?;
    
    // Read results
    let results_str = fs::read_to_string(tmp.join("results.json"))?;
    let results: serde_json::Value = serde_json::from_str(&results_str)?;
    
    let total_return = results["total_return"].as_f64().unwrap_or(0.0);
    let trades = results["trades_executed"].as_u64().unwrap_or(0);
    
    println!("\n=== Anti-Look-Ahead Validation ===");
    println!("Total Return: {:.2}%", total_return * 100.0);
    println!("Trades: {}", trades);
    
    // Gate: positive return on trap data indicates look-ahead
    if total_return > 0.01 {
        println!("\nLOOK-AHEAD DETECTED!");
        println!("Strategy profited on trap data - possible future data access.");
        Err("Look-ahead bias detected".into())
    } else {
        println!("\nAnti-look-ahead validation PASSED");
        println!("Strategy did not profit from future knowledge.");
        Ok(())
    }
}

fn compare_command(baseline_path: PathBuf, output_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating benchmark comparison...");
    
    // Read baseline
    let baseline_str = fs::read_to_string(&baseline_path)?;
    let baseline: serde_json::Value = serde_json::from_str(&baseline_str)?;
    
    // Read current Criterion results
    let criterion_path = PathBuf::from("target/criterion");
    
    let mut report = String::new();
    report.push_str("# Performance Comparison Report\n\n");
    report.push_str(&format!("Baseline: {}\n", baseline_path.display()));
    report.push_str(&format!("Generated: {}\n\n", chrono::Utc::now().to_rfc3339()));
    report.push_str("## Scenario Results\n\n");
    report.push_str("| Scenario | Baseline (events/s) | Current | Delta |\n");
    report.push_str("|----------|---------------------|---------|-------|\n");
    
    let scenarios = [
        ("intraday_net_zero/mean_reversion_10k_events", "intraday_net_zero.mean_reversion_10k_events"),
        ("daily_swing/trend_200_assets", "daily_swing.trend_200_assets"),
        ("stress_universe/noop_1000_assets", "stress_universe.noop_1000_assets"),
    ];
    
    let mut all_pass = true;
    let max_regression = baseline["gates"]["max_regression_percent"].as_f64().unwrap_or(5.0);
    
    for (criterion_name, baseline_key) in scenarios {
        let est_path = criterion_path.join(criterion_name).join("new/estimates.json");
        
        let baseline_throughput = {
            let parts: Vec<&str> = baseline_key.split('.').collect();
            baseline["scenarios"][parts[0]][parts[1]]["throughput_events_per_sec"]
                .as_f64()
                .unwrap_or(0.0)
        };
        
        if let Ok(est_str) = fs::read_to_string(&est_path) {
            let est: serde_json::Value = serde_json::from_str(&est_str)?;
            let mean_ns = est["mean"]["point_estimate"].as_f64().unwrap_or(1.0);
            
            // Calculate current throughput (10000 events for intraday, etc)
            let events = if criterion_name.contains("10k") { 10000.0 }
                        else if criterion_name.contains("200") { 50400.0 }
                        else { 252000.0 };
            let current_throughput = events / (mean_ns / 1_000_000_000.0);
            
            let delta_pct = ((current_throughput - baseline_throughput) / baseline_throughput) * 100.0;
            let status = if delta_pct < -max_regression { "⚠️ REGRESSION" } else { "✅" };
            
            if delta_pct < -max_regression {
                all_pass = false;
            }
            
            report.push_str(&format!(
                "| {} | {:.0} | {:.0} | {:+.1}% {} |\n",
                criterion_name.split('/').next().unwrap_or(criterion_name),
                baseline_throughput,
                current_throughput,
                delta_pct,
                status
            ));
        } else {
            report.push_str(&format!("| {} | {:.0} | N/A | - |\n", 
                criterion_name.split('/').next().unwrap_or(criterion_name), 
                baseline_throughput));
        }
    }
    
    report.push_str("\n## Gates\n\n");
    report.push_str(&format!("- Max Regression Allowed: {}%\n", max_regression));
    report.push_str(&format!("- Status: {}\n", if all_pass { "✅ PASS" } else { "❌ FAIL" }));
    
    // Write report
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, &report)?;
    
    println!("Comparison report written to: {}", output_path.display());
    print!("{}", report);
    
    if all_pass {
        Ok(())
    } else {
        Err("Performance regression detected".into())
    }
}

fn fetch_command(tickers: String, range: String, output_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
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
    let mut ticker_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for bar in &bars {
        *ticker_counts.entry(&bar.ticker).or_insert(0) += 1;
    }
    
    println!("\n=== Summary ===");
    for (ticker, count) in &ticker_counts {
        println!("  {}: {} days", ticker, count);
    }
    
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    
    let result = match cli.command {
        Commands::Run { config, output } => run_command(config, output),
        Commands::Validate { data } => validate_command(data),
        Commands::Determinism { config } => determinism_command(config),
        Commands::Lookahead { config } => lookahead_command(config),
        Commands::Compare { baseline, output } => compare_command(baseline, output),
        Commands::Fetch { tickers, range, output } => fetch_command(tickers, range, output),
    };
    
    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

