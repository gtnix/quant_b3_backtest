//! Export command - Export top strategies.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

/// Execute the export command.
pub fn execute(experiment_id: &str, n: usize, output: Option<&str>) -> Result<()> {
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

    println!("Exporting top {} strategies to {:?}\n", n, output_dir);

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

            fs::copy(&toml_path, &dest_path)?;
            println!("  Exported: {}", dest_name);
            exported += 1;
        }
    }

    println!("\nExported {} strategies", exported);

    Ok(())
}

