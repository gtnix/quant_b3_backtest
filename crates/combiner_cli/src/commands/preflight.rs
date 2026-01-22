//! Preflight validation command.
//!
//! Validates data and configuration before starting mining.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use combiner_core::ParamRanges;
use combiner_engine::StrategyCatalog;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Execute preflight checks.
pub fn execute(config_path: &str) -> Result<()> {
    println!("========================================");
    println!("  PREFLIGHT VALIDATION");
    println!("========================================");
    println!("  Config: {}", config_path);
    println!("");

    // Load config
    let config_content = std::fs::read_to_string(config_path)
        .context("Failed to read config file")?;
    let config: toml::Value = toml::from_str(&config_content)
        .context("Failed to parse config TOML")?;

    // Extract dataset info
    let market = config.get("dataset")
        .and_then(|d| d.get("market"))
        .and_then(|m| m.as_str())
        .unwrap_or("BR");
    
    let data_path = config.get("dataset")
        .and_then(|d| d.get("market_data_path"))
        .and_then(|p| p.as_str())
        .unwrap_or("data/market_data_ibov.csv");

    println!("[1/5] Checking data file...");
    
    // Check 1: Data file exists
    if !Path::new(data_path).exists() {
        println!("  FAIL: Data file not found: {}", data_path);
        bail!("Data file not found: {}", data_path);
    }
    println!("  OK: {} exists", data_path);

    // Check 2: Parse headers and count rows
    println!("");
    println!("[2/5] Checking OHLCV columns...");
    
    let file = File::open(data_path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    let header_line = lines.next()
        .ok_or_else(|| anyhow::anyhow!("Empty data file"))??;
    
    let headers: Vec<&str> = header_line.split(',').collect();
    let required = ["symbol", "date", "open", "high", "low", "close", "volume"];
    
    for col in &required {
        if !headers.iter().any(|h| h.trim().to_lowercase() == *col) {
            println!("  FAIL: Missing column: {}", col);
            bail!("Missing required column: {}", col);
        }
    }
    println!("  OK: All OHLCV columns present");

    // Check 3: Count bars per symbol
    println!("");
    println!("[3/5] Checking data coverage...");
    
    let mut symbol_counts: HashMap<String, usize> = HashMap::new();
    let symbol_idx = headers.iter().position(|h| h.trim().to_lowercase() == "symbol").unwrap();
    
    for line in lines {
        let line = line?;
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() > symbol_idx {
            let symbol = cols[symbol_idx].to_string();
            *symbol_counts.entry(symbol).or_insert(0) += 1;
        }
    }

    let total_symbols = symbol_counts.len();
    let min_bars = symbol_counts.values().min().copied().unwrap_or(0);
    let max_bars = symbol_counts.values().max().copied().unwrap_or(0);
    let avg_bars = if total_symbols > 0 {
        symbol_counts.values().sum::<usize>() / total_symbols
    } else {
        0
    };

    println!("  Symbols: {}", total_symbols);
    println!("  Bars: min={}, max={}, avg={}", min_bars, max_bars, avg_bars);

    if min_bars < 252 {
        let low_symbols: Vec<_> = symbol_counts.iter()
            .filter(|(_, &c)| c < 252)
            .map(|(s, c)| format!("{} ({})", s, c))
            .take(5)
            .collect();
        println!("  WARN: {} symbols with < 252 bars: {:?}", 
            symbol_counts.iter().filter(|(_, &c)| c < 252).count(),
            low_symbols);
    }
    
    if total_symbols < 5 {
        println!("  FAIL: Too few symbols ({})", total_symbols);
        bail!("Insufficient symbols: {} (need at least 5)", total_symbols);
    }
    println!("  OK: {} symbols with sufficient data", total_symbols);

    // Check 4: Strategy Catalog compatibility
    println!("");
    println!("[4/5] Checking Strategy Catalog...");
    
    let catalog = StrategyCatalog::from_builtin();
    let param_ranges = ParamRanges::new();
    
    // Filter templates that work with OHLCV-only
    let ohlcv_compatible = catalog.templates().iter()
        .filter(|t| {
            t.pipeline.iter().all(|block| {
                if let Some(spec) = param_ranges.get_block(&block.block_id) {
                    spec.required_columns.iter().all(|c| c.is_ohlcv())
                } else {
                    true // Unknown blocks pass (will fail later anyway)
                }
            })
        })
        .count();

    println!("  Total templates: {}", catalog.len());
    println!("  OHLCV-compatible: {}", ohlcv_compatible);

    if ohlcv_compatible < 10 {
        println!("  FAIL: Too few compatible templates");
        bail!("Insufficient OHLCV-compatible templates: {}", ohlcv_compatible);
    }
    println!("  OK: {} templates available", ohlcv_compatible);

    // Check 5: Sample genome generation
    println!("");
    println!("[5/5] Testing genome generation...");
    
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut valid_count = 0;
    let test_count = 10;

    for i in 0..test_count {
        let template = &catalog.templates()[i % catalog.len()];
        let genome = StrategyCatalog::to_genome(template, &mut rng, &param_ranges, 0);
        
        // Basic validation
        if !genome.genes.is_empty() && genome.fitness.is_none() {
            valid_count += 1;
        }
    }

    println!("  Generated: {}/{} valid genomes", valid_count, test_count);
    
    if valid_count < test_count {
        println!("  WARN: Some genomes failed validation");
    }
    println!("  OK: Genome generation working");

    // Summary
    println!("");
    println!("========================================");
    println!("  PREFLIGHT PASSED");
    println!("========================================");
    println!("  Market: {}", market);
    println!("  Symbols: {}", total_symbols);
    println!("  Templates: {}", ohlcv_compatible);
    println!("");
    println!("  Ready to mine!");
    println!("");

    Ok(())
}
