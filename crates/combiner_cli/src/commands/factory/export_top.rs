//! Factory export-top command - Deterministic Top N candidate export.
//!
//! Ranking rules (deterministic):
//! 1. oos_sharpe_net DESC
//! 2. pbo ASC
//! 3. oos_cagr_net DESC (fallback: 0.0 if NULL)
//! 4. max_drawdown_net ASC (fallback: 0.0 if NULL, less negative is better)
//! 5. genome_hash ASC (tie-breaker for stability)
//!
//! Filters:
//! - Run must have data_integrity_verdict = 'PASS'
//! - Candidate must have gates_passed = true (if field populated)

use std::cmp::Ordering;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;

use super::registry::{Candidate, Registry};

/// Schema version for export format.
const EXPORT_SCHEMA_VERSION: &str = "1.0.0";

/// Exported candidate with all ranking-relevant fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedCandidate {
    pub rank: i32,
    pub candidate_id: String,
    pub genome_hash: String,
    pub oos_sharpe_net: f32,
    pub pbo: f32,
    pub oos_cagr_net: f32,
    pub max_drawdown_net: f32,
    pub dsr: Option<f32>,
    pub stress_passed: Option<i32>,
    pub stress_total: Option<i32>,
    pub gates_passed: Option<bool>,
    pub turnover_annual: Option<f32>,
    pub capacity_usd: Option<f32>,
    pub created_at: DateTime<Utc>,
}

/// Export metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMetadata {
    pub schema_version: String,
    pub run_id: String,
    pub campaign_id: String,
    pub data_integrity_verdict: String,
    pub top_n: usize,
    pub actual_count: usize,
    pub exported_at: DateTime<Utc>,
    pub ranking_rules: Vec<String>,
    pub filters_applied: Vec<String>,
}

/// Full export structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopCandidatesExport {
    pub metadata: ExportMetadata,
    pub candidates: Vec<ExportedCandidate>,
}

/// Execute factory export-top command.
pub fn execute_export_top(
    run_id: &str,
    top_n: usize,
    formats: &str,
) -> Result<()> {
    let rt = Runtime::new()?;
    rt.block_on(async {
        let registry = Registry::connect().await?;

        // Get run details
        let run = registry.get_run(run_id).await?
            .ok_or_else(|| anyhow::anyhow!("Run not found: {}", run_id))?;

        // Check data integrity verdict
        let integrity_verdict = run.data_integrity_verdict.as_deref().unwrap_or("UNKNOWN");
        if integrity_verdict != "PASS" {
            return Err(anyhow::anyhow!(
                "Run {} has data_integrity_verdict = '{}'. Only PASS runs can be exported.",
                run_id, integrity_verdict
            ));
        }

        // Get ALL candidates for this run (we'll sort and filter in memory)
        let all_candidates = registry.get_all_candidates(run_id).await?;

        if all_candidates.is_empty() {
            println!("No candidates found for run {}", run_id);
            return Ok(());
        }

        // Filter: gates_passed = true (if populated)
        let filtered: Vec<Candidate> = all_candidates
            .into_iter()
            .filter(|c| c.gates_passed.unwrap_or(true)) // If NULL, assume passed
            .collect();

        // Sort with deterministic ranking
        let mut sorted = filtered;
        sorted.sort_by(|a, b| compare_candidates(a, b));

        // Take top N
        let top: Vec<Candidate> = sorted.into_iter().take(top_n).collect();

        // Convert to export format
        let exported: Vec<ExportedCandidate> = top
            .iter()
            .enumerate()
            .map(|(i, c)| ExportedCandidate {
                rank: (i + 1) as i32,
                candidate_id: c.candidate_id.clone(),
                genome_hash: c.genome_hash.clone(),
                oos_sharpe_net: c.oos_sharpe_net.unwrap_or(f32::NEG_INFINITY),
                pbo: c.pbo.unwrap_or(1.0),
                oos_cagr_net: c.oos_cagr_net.unwrap_or(0.0),
                max_drawdown_net: c.max_drawdown_net.unwrap_or(0.0),
                dsr: c.dsr,
                stress_passed: c.stress_passed,
                stress_total: c.stress_total,
                gates_passed: c.gates_passed,
                turnover_annual: c.turnover_annual,
                capacity_usd: c.capacity_usd,
                created_at: c.created_at,
            })
            .collect();

        // Build metadata
        let metadata = ExportMetadata {
            schema_version: EXPORT_SCHEMA_VERSION.to_string(),
            run_id: run_id.to_string(),
            campaign_id: run.campaign_id.clone(),
            data_integrity_verdict: integrity_verdict.to_string(),
            top_n,
            actual_count: exported.len(),
            exported_at: Utc::now(),
            ranking_rules: vec![
                "1. oos_sharpe_net DESC".to_string(),
                "2. pbo ASC".to_string(),
                "3. oos_cagr_net DESC (fallback: 0.0)".to_string(),
                "4. max_drawdown_net ASC (fallback: 0.0)".to_string(),
                "5. genome_hash ASC (tie-breaker)".to_string(),
            ],
            filters_applied: vec![
                "data_integrity_verdict = 'PASS'".to_string(),
                "gates_passed = true (or NULL)".to_string(),
            ],
        };

        let export = TopCandidatesExport {
            metadata,
            candidates: exported,
        };

        // Create output directory
        let output_dir = format!("artifacts/top_candidates/{}", run_id);
        fs::create_dir_all(&output_dir)
            .context("Failed to create output directory")?;

        // Export in requested formats
        let format_list: Vec<&str> = formats.split(',').map(|s| s.trim()).collect();

        for fmt in &format_list {
            match *fmt {
                "json" => {
                    let json_path = format!("{}/top{}.json", output_dir, top_n);
                    let json = serde_json::to_string_pretty(&export)
                        .context("Failed to serialize JSON")?;
                    fs::write(&json_path, json)
                        .context("Failed to write JSON file")?;
                    println!("✓ Exported: {}", json_path);
                }
                "csv" => {
                    let csv_path = format!("{}/top{}.csv", output_dir, top_n);
                    write_csv(&csv_path, &export)?;
                    println!("✓ Exported: {}", csv_path);
                }
                _ => {
                    println!("⚠️  Unknown format: {}", fmt);
                }
            }
        }

        // Print summary
        println!();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║                  EXPORT TOP SUMMARY                          ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Run ID:           {}                         ", run_id);
        println!("║ Data Integrity:   {}                                      ", integrity_verdict);
        println!("║ Requested Top:    {}                                      ", top_n);
        println!("║ Actual Exported:  {}                                      ", export.candidates.len());
        println!("║ Output Dir:       {}          ", output_dir);
        println!("╚══════════════════════════════════════════════════════════════╝");

        if !export.candidates.is_empty() {
            println!("\nTop 5 candidates:");
            println!("{:>5} {:>12} {:>8} {:>10} {:>10}", "Rank", "Sharpe", "PBO", "CAGR", "MaxDD");
            for c in export.candidates.iter().take(5) {
                println!(
                    "{:>5} {:>12.4} {:>8.4} {:>10.4} {:>10.4}",
                    c.rank, c.oos_sharpe_net, c.pbo, c.oos_cagr_net, c.max_drawdown_net
                );
            }
        }

        Ok(())
    })
}

/// Deterministic comparison for ranking.
fn compare_candidates(a: &Candidate, b: &Candidate) -> Ordering {
    // 1. oos_sharpe_net DESC (higher is better)
    let sharpe_a = a.oos_sharpe_net.unwrap_or(f32::NEG_INFINITY);
    let sharpe_b = b.oos_sharpe_net.unwrap_or(f32::NEG_INFINITY);
    match sharpe_b.partial_cmp(&sharpe_a).unwrap_or(Ordering::Equal) {
        Ordering::Equal => {}
        other => return other,
    }

    // 2. pbo ASC (lower is better)
    let pbo_a = a.pbo.unwrap_or(1.0);
    let pbo_b = b.pbo.unwrap_or(1.0);
    match pbo_a.partial_cmp(&pbo_b).unwrap_or(Ordering::Equal) {
        Ordering::Equal => {}
        other => return other,
    }

    // 3. oos_cagr_net DESC (higher is better)
    let cagr_a = a.oos_cagr_net.unwrap_or(0.0);
    let cagr_b = b.oos_cagr_net.unwrap_or(0.0);
    match cagr_b.partial_cmp(&cagr_a).unwrap_or(Ordering::Equal) {
        Ordering::Equal => {}
        other => return other,
    }

    // 4. max_drawdown_net ASC (less negative is better, closer to 0)
    let dd_a = a.max_drawdown_net.unwrap_or(0.0);
    let dd_b = b.max_drawdown_net.unwrap_or(0.0);
    match dd_a.partial_cmp(&dd_b).unwrap_or(Ordering::Equal) {
        Ordering::Equal => {}
        other => return other,
    }

    // 5. genome_hash ASC (alphabetical tie-breaker for stability)
    a.genome_hash.cmp(&b.genome_hash)
}

/// Write CSV export.
fn write_csv(path: &str, export: &TopCandidatesExport) -> Result<()> {
    let mut csv = String::new();

    // Header
    csv.push_str("rank,candidate_id,genome_hash,oos_sharpe_net,pbo,oos_cagr_net,max_drawdown_net,dsr,stress_passed,stress_total,gates_passed,turnover_annual,capacity_usd,created_at\n");

    // Rows
    for c in &export.candidates {
        csv.push_str(&format!(
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{},{}\n",
            c.rank,
            c.candidate_id,
            c.genome_hash,
            c.oos_sharpe_net,
            c.pbo,
            c.oos_cagr_net,
            c.max_drawdown_net,
            c.dsr.map(|v| format!("{:.6}", v)).unwrap_or_default(),
            c.stress_passed.map(|v| v.to_string()).unwrap_or_default(),
            c.stress_total.map(|v| v.to_string()).unwrap_or_default(),
            c.gates_passed.map(|v| v.to_string()).unwrap_or_default(),
            c.turnover_annual.map(|v| format!("{:.6}", v)).unwrap_or_default(),
            c.capacity_usd.map(|v| format!("{:.2}", v)).unwrap_or_default(),
            c.created_at.format("%Y-%m-%dT%H:%M:%SZ"),
        ));
    }

    fs::write(path, csv).context("Failed to write CSV file")?;
    Ok(())
}
