//! Extract command - Read .obfs artifacts and export to JSON.
//!
//! This command allows extracting pending backtest artifacts from
//! interrupted campaigns or for analysis purposes.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use uuid::Uuid;

/// Execute the extract command.
///
/// # Arguments
/// * `pending_dir` - Path to the pending directory containing .obfs files
/// * `run_ids` - Optional list of specific run IDs to extract (extracts all if empty)
/// * `output_dir` - Output directory for JSON files
/// * `top_n` - If set, extract only top N by sharpe ratio
pub fn execute(
    pending_dir: &str,
    run_ids: &[String],
    output_dir: &str,
    top_n: Option<usize>,
) -> Result<()> {
    let pending_path = Path::new(pending_dir);
    
    if !pending_path.exists() {
        anyhow::bail!("Pending directory not found: {}", pending_dir);
    }
    
    // Create output directory
    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path)?;
    
    // Open pending store
    let store = obfs::PendingStore::new(pending_path)
        .context("Failed to open pending store")?;
    
    // Get list of UUIDs to extract
    let uuids_to_extract: Vec<Uuid> = if run_ids.is_empty() {
        // Extract all
        store.list_pending()?
    } else {
        // Parse specific UUIDs
        run_ids
            .iter()
            .map(|s| Uuid::parse_str(s))
            .collect::<Result<Vec<_>, _>>()
            .context("Invalid UUID format in run_ids")?
    };
    
    if uuids_to_extract.is_empty() {
        println!("No artifacts found to extract.");
        return Ok(());
    }
    
    println!("Found {} artifacts to process", uuids_to_extract.len());
    
    // Read all artifacts and collect with metrics
    let mut artifacts: Vec<(Uuid, obfs::pending_store::PendingArtifact)> = Vec::new();
    let mut errors = 0;
    
    for uuid in &uuids_to_extract {
        match store.read_pending(*uuid) {
            Ok(artifact) => {
                artifacts.push((*uuid, artifact));
            }
            Err(e) => {
                eprintln!("Warning: Failed to read {}: {}", uuid, e);
                errors += 1;
            }
        }
    }
    
    println!("Successfully read {} artifacts ({} errors)", artifacts.len(), errors);
    
    // Sort by sharpe ratio (descending)
    artifacts.sort_by(|a, b| {
        b.1.metrics.sharpe_ratio
            .partial_cmp(&a.1.metrics.sharpe_ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    
    // Apply top_n filter if specified
    let artifacts_to_export: Vec<_> = if let Some(n) = top_n {
        artifacts.into_iter().take(n).collect()
    } else {
        artifacts
    };
    
    println!("Exporting {} artifacts to {}", artifacts_to_export.len(), output_dir);
    
    // Export each artifact as JSON
    for (i, (uuid, artifact)) in artifacts_to_export.iter().enumerate() {
        let rank = i + 1;
        let filename = format!(
            "rank{:04}_{}_sharpe{:.3}.json",
            rank,
            uuid,
            artifact.metrics.sharpe_ratio
        );
        let output_file = output_path.join(&filename);
        
        // Create summary structure
        let summary = serde_json::json!({
            "rank": rank,
            "run_id": uuid.to_string(),
            "metadata": {
                "strategy_id": artifact.metadata.strategy_id,
                "strategy_version": artifact.metadata.strategy_version,
                "universe": artifact.metadata.universe,
                "start_date": artifact.metadata.start_date,
                "end_date": artifact.metadata.end_date,
                "initial_capital": artifact.metadata.initial_capital,
                "mode": artifact.metadata.mode,
            },
            "metrics": {
                "sharpe_ratio": artifact.metrics.sharpe_ratio,
                "cagr": artifact.metrics.cagr,
                "volatility": artifact.metrics.volatility,
                "sortino_ratio": artifact.metrics.sortino_ratio,
                "max_drawdown": artifact.metrics.max_drawdown,
                "max_drawdown_duration_days": artifact.metrics.max_drawdown_duration_days,
                "hit_rate": artifact.metrics.hit_rate,
                "profit_factor": artifact.metrics.profit_factor,
                "turnover_annual": artifact.metrics.turnover_annual,
                "total_trades": artifact.metrics.total_trades,
            },
            "timeseries_points": artifact.timeseries.len(),
            "trace_events": artifact.trace.len(),
        });
        
        let json = serde_json::to_string_pretty(&summary)?;
        fs::write(&output_file, json)?;
        
        println!(
            "  [{}] {} - Sharpe: {:.3}, CAGR: {:.2}%, Trades: {}",
            rank,
            uuid,
            artifact.metrics.sharpe_ratio,
            artifact.metrics.cagr * 100.0,
            artifact.metrics.total_trades
        );
    }
    
    // Write summary file
    let summary_path = output_path.join("extraction_summary.json");
    let summary = serde_json::json!({
        "source_dir": pending_dir,
        "total_found": uuids_to_extract.len(),
        "successfully_read": artifacts_to_export.len() + errors,
        "exported": artifacts_to_export.len(),
        "errors": errors,
        "top_n_filter": top_n,
        "extraction_time": chrono::Utc::now().to_rfc3339(),
    });
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    
    println!("\nExtraction complete. Summary written to {:?}", summary_path);
    
    Ok(())
}

/// Extract a single artifact by UUID and return full details including timeseries.
pub fn extract_full(pending_dir: &str, run_id: &str, output_file: &str) -> Result<()> {
    let pending_path = Path::new(pending_dir);
    let store = obfs::PendingStore::new(pending_path)?;
    
    let uuid = Uuid::parse_str(run_id).context("Invalid UUID format")?;
    let artifact = store.read_pending(uuid)?;
    
    // Export full artifact including timeseries
    let full_export = serde_json::json!({
        "run_id": uuid.to_string(),
        "version": artifact.version,
        "metadata": artifact.metadata,
        "metrics": artifact.metrics,
        "timeseries": artifact.timeseries.iter().map(|p| {
            serde_json::json!({
                "date_offset": p.date_offset,
                "equity": p.equity,
                "drawdown": p.drawdown,
                "exposure": p.exposure,
            })
        }).collect::<Vec<_>>(),
        "trace": artifact.trace,
    });
    
    let json = serde_json::to_string_pretty(&full_export)?;
    fs::write(output_file, json)?;
    
    println!("Full artifact exported to {}", output_file);
    println!("  Sharpe: {:.3}", artifact.metrics.sharpe_ratio);
    println!("  CAGR: {:.2}%", artifact.metrics.cagr * 100.0);
    println!("  Timeseries points: {}", artifact.timeseries.len());
    println!("  Trace events: {}", artifact.trace.len());
    
    Ok(())
}
