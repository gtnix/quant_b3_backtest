//! Validate campaign config without executing - for debugging Cockpit-generated TOMLs.

use std::path::Path;
use anyhow::{Context, Result};
use tracing::info;

use super::config::CampaignConfig;

/// Execute config validation command.
pub fn execute_validate(campaign_path: &str, verbose: bool) -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          STRATEGY FACTORY - CONFIG VALIDATION                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    
    // 1. Check file exists
    let path = Path::new(campaign_path);
    if !path.exists() {
        anyhow::bail!("Campaign config not found: {}", campaign_path);
    }
    println!("✓ Config file exists: {}", campaign_path);
    
    // 2. Parse TOML
    let config = CampaignConfig::load(campaign_path)
        .context("Failed to parse campaign config")?;
    println!("✓ TOML syntax valid");
    
    // 3. Validate required fields
    validate_campaign_meta(&config)?;
    println!("✓ Campaign metadata valid");
    
    // 4. Validate paths
    validate_paths(&config)?;
    println!("✓ Referenced paths valid");
    
    // 5. Validate numeric ranges
    validate_ranges(&config)?;
    println!("✓ Numeric values in range");
    
    // 6. Compute hashes
    let config_hash = config.config_hash();
    let dataset_hash = config.dataset_hash();
    println!();
    println!("Config Hash:   {}", config_hash);
    println!("Dataset Hash:  {}", dataset_hash.as_deref().unwrap_or("N/A"));
    
    // 7. Show what would be executed
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("EXECUTION PLAN (--dry-run):");
    println!("═══════════════════════════════════════════════════════════════");
    
    let seeds = config.seeds.generate_seeds();
    println!("  Campaign:     {}", config.campaign.name);
    println!("  Tag:          {}", config.campaign.tag.as_deref().unwrap_or("none"));
    println!("  Market:       {}", config.dataset.market);
    println!("  Date Range:   {} to {}", 
        config.dataset.start_date.as_deref().unwrap_or("default"),
        config.dataset.end_date.as_deref().unwrap_or("default"));
    println!("  Seeds:        {:?}", seeds);
    println!("  Population:   {}", config.evolution.population_size.unwrap_or(100));
    println!("  Max Gens:     {}", config.evolution.max_generations.unwrap_or(50));
    println!("  Timeout:      {} seconds", config.budget.timeout_per_run_secs);
    println!("  Stress:       {}", if config.budget.stress_enabled { "enabled" } else { "disabled" });
    println!();
    println!("Gates:");
    println!("  Min Sharpe:   {:.2}", config.promotion.min_oos_sharpe_net);
    println!("  Max PBO:      {:.2}", config.promotion.max_pbo);
    println!("  Min Stress:   {} passed", config.promotion.min_stress_passed);
    println!();
    
    if verbose {
        println!("═══════════════════════════════════════════════════════════════");
        println!("FULL CONFIG:");
        println!("═══════════════════════════════════════════════════════════════");
        let toml_str = toml::to_string_pretty(&config)?;
        println!("{}", toml_str);
    }
    
    println!("✓ Validation PASSED - config is ready to execute");
    
    Ok(())
}

fn validate_campaign_meta(config: &CampaignConfig) -> Result<()> {
    if config.campaign.name.is_empty() {
        anyhow::bail!("Campaign name cannot be empty");
    }
    if config.campaign.name.contains(' ') {
        info!("Warning: Campaign name contains spaces, this may cause issues");
    }
    Ok(())
}

fn validate_paths(config: &CampaignConfig) -> Result<()> {
    // Check execution config path if specified
    if let Some(ref exec_path) = config.execution.config_path {
        let p = Path::new(exec_path);
        if !p.exists() {
            anyhow::bail!("Execution config not found: {} (check execution.config_path)", exec_path);
        }
    }
    
    // Check evolution base config if specified
    if let Some(ref base_config) = config.evolution.base_config {
        let p = Path::new(base_config);
        if !p.exists() {
            anyhow::bail!("Evolution base config not found: {} (check evolution.base_config)", base_config);
        }
    }
    
    // Check data path if specified
    if let Some(ref data_path) = config.dataset.data_path {
        let p = Path::new(data_path);
        if !p.exists() {
            info!("Warning: Data path does not exist: {} (dataset hash will be N/A)", data_path);
        }
    }
    
    Ok(())
}

fn validate_ranges(config: &CampaignConfig) -> Result<()> {
    // Population size
    if let Some(pop) = config.evolution.population_size {
        if pop < 10 {
            anyhow::bail!("Population size too small: {} (min: 10)", pop);
        }
        if pop > 10000 {
            info!("Warning: Very large population size: {} (may be slow)", pop);
        }
    }
    
    // Max generations
    if let Some(gens) = config.evolution.max_generations {
        if gens > 1_000_000_000 {
            info!("Info: Unlimited generations detected (timeout will control run)");
        }
    }
    
    // Timeout
    if config.budget.timeout_per_run_secs < 30 {
        anyhow::bail!("Timeout too short: {} seconds (min: 30)", config.budget.timeout_per_run_secs);
    }
    
    // Seed count
    if config.seeds.count == 0 {
        anyhow::bail!("Seed count cannot be 0");
    }
    
    // Promotion thresholds
    if config.promotion.max_pbo < 0.0 || config.promotion.max_pbo > 1.0 {
        anyhow::bail!("Invalid max_pbo: {} (must be 0.0-1.0)", config.promotion.max_pbo);
    }
    
    Ok(())
}




