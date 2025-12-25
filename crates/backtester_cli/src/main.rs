//! # Backtester CLI
//!
//! Command-line interface for running backtests.
//! This is the core backtesting infrastructure - strategy implementations go in separate crates.

use backtester_io::{BrapiLoader, CsvLoader, Normalizer};
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

fn main() {
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
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
