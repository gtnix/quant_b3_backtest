//! Status command - Check experiment status.

use anyhow::Result;
use std::fs;
use std::path::Path;

/// Execute the status command.
pub fn execute(experiment_id: &str) -> Result<()> {
    let output_dir = Path::new("output/scg").join(experiment_id);

    if !output_dir.exists() {
        println!("Experiment not found: {}", experiment_id);
        return Ok(());
    }

    println!("=== Experiment Status ===\n");
    println!("ID: {}", experiment_id);
    println!("Path: {:?}", output_dir);

    // Check for manifest
    let manifest_path = output_dir.join("manifest.json");
    if manifest_path.exists() {
        let content = fs::read_to_string(&manifest_path)?;
        let manifest: serde_json::Value = serde_json::from_str(&content)?;

        if let Some(status) = manifest.get("status").and_then(|v| v.as_str()) {
            println!("Status: {}", status);
        }
        if let Some(gens) = manifest.get("generations_completed").and_then(|v| v.as_u64()) {
            println!("Generations: {}", gens);
        }
    }

    // Check Hall of Fame
    let hof_dir = output_dir.join("hall_of_fame");
    if hof_dir.exists() {
        let entries: Vec<_> = fs::read_dir(&hof_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();
        println!("Hall of Fame entries: {}", entries.len());
    }

    // Check generation stats
    let stats_path = output_dir.join("generation_stats.json");
    if stats_path.exists() {
        let content = fs::read_to_string(&stats_path)?;
        let stats: Vec<serde_json::Value> = serde_json::from_str(&content)?;

        if let Some(last) = stats.last() {
            println!("\nLatest Generation:");
            if let Some(gen) = last.get("generation").and_then(|v| v.as_u64()) {
                println!("  Generation: {}", gen);
            }
            if let Some(pareto) = last.get("pareto_size").and_then(|v| v.as_u64()) {
                println!("  Pareto size: {}", pareto);
            }
            if let Some(sharpe) = last.get("best_sharpe").and_then(|v| v.as_f64()) {
                println!("  Best Sharpe: {:.2}", sharpe);
            }
        }
    }

    Ok(())
}

