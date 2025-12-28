//! Export command - Export top strategies.

use anyhow::Result;
use std::fs;
use std::path::Path;

/// Execute the export command.
pub fn execute(experiment_id: &str, n: usize, output: Option<&str>, include_execution_config: bool) -> Result<()> {
    let input_dir = Path::new("output/scg").join(experiment_id).join("hall_of_fame");

    if !input_dir.exists() {
        println!("Experiment not found or has no Hall of Fame: {}", experiment_id);
        return Ok(());
    }

    // Determine output directory
    let output_dir = output
        .map(|s| Path::new(s).to_path_buf())
        .unwrap_or_else(|| Path::new("output/scg/exports").join(experiment_id));

    fs::create_dir_all(&output_dir)?;

    // Read and sort entries
    let mut entries: Vec<_> = fs::read_dir(&input_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();

    // Sort by directory name (which should be strategy_001, strategy_002, etc.)
    entries.sort_by(|a, b| a.path().cmp(&b.path()));

    let to_export = entries.into_iter().take(n);

    println!("Exporting top {} strategies to {:?}", n, output_dir);
    if include_execution_config {
        println!("  (including execution config parameters)");
    }
    println!();

    let mut exported = 0;
    for entry in to_export {
        let source_dir = entry.path();
        let toml_path = source_dir.join("config.toml");

        if toml_path.exists() {
            let dest_name = format!(
                "{}.toml",
                source_dir.file_name().unwrap().to_string_lossy()
            );
            let dest_path = output_dir.join(&dest_name);

            if include_execution_config {
                // Read the config and append execution parameters
                let mut content = fs::read_to_string(&toml_path)?;
                content.push_str("\n\n# === Execution Configuration (for production) ===\n");
                content.push_str("[execution]\n");
                content.push_str("delay_bars = 1\n");
                content.push_str("\n[execution.slippage]\n");
                content.push_str("type = \"Constant\"\n");
                content.push_str("bps = 10.0\n");
                content.push_str("\n[execution.fees]\n");
                content.push_str("tier = \"B3Retail\"\n");
                content.push_str("commission_rate = 0.0015\n");
                content.push_str("emolument_rate = 0.00035\n");
                content.push_str("\n[execution.fill_policy]\n");
                content.push_str("allow_partial = false\n");
                content.push_str("max_participation = 0.05\n");
                fs::write(&dest_path, content)?;
            } else {
                fs::copy(&toml_path, &dest_path)?;
            }
            
            println!("  Exported: {}", dest_name);
            exported += 1;
        }
    }

    println!("\nExported {} strategies", exported);

    Ok(())
}
