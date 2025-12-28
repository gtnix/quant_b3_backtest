//! Factory init command - Create campaign config template and directory structure.

use anyhow::Result;
use std::fs;
use std::path::Path;
use tracing::info;

use super::config::generate_example_config;

/// Execute the factory init command.
pub fn execute_init(name: &str) -> Result<()> {
    // Create directories
    let campaigns_dir = Path::new("configs/campaigns");
    let artifacts_dir = Path::new("artifacts/candidates");

    fs::create_dir_all(campaigns_dir)?;
    fs::create_dir_all(artifacts_dir)?;

    // Generate config file
    let config_path = campaigns_dir.join(format!("{}.toml", name));

    if config_path.exists() {
        println!("Campaign config already exists: {}", config_path.display());
        println!("Use a different name or edit the existing file.");
        return Ok(());
    }

    let config_content = generate_example_config(name);
    fs::write(&config_path, config_content)?;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              STRATEGY FACTORY INITIALIZED                    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Created:                                                     ║");
    println!("║   - configs/campaigns/{}.toml                       ║", name);
    println!("║   - artifacts/candidates/                                    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Next steps:                                                  ║");
    println!("║   1. Edit the campaign config file                           ║");
    println!("║   2. Run: combiner factory run --campaign {}  ║", config_path.display());
    println!("╚══════════════════════════════════════════════════════════════╝");

    info!(name, path = %config_path.display(), "Created campaign config");

    Ok(())
}

