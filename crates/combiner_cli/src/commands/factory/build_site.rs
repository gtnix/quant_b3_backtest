//! Factory build-site command - Generate site-ready JSON bundle.
//!
//! Produces a structured set of JSON files for web consumption:
//! - artifacts/site/index.json - global index
//! - artifacts/site/campaign_<id>.json - campaign details with runs
//! - artifacts/site/run_<id>.json - run details with config and metrics
//! - artifacts/site/candidate_<id>.json - candidate details (optional, for promoted)

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;

use super::registry::{Campaign, Candidate, Registry, Run};

/// Site index - entry point for web consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteIndex {
    pub schema_version: String,
    pub generated_at: DateTime<Utc>,
    pub campaigns: Vec<CampaignSummary>,
}

/// Campaign summary for index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignSummary {
    pub campaign_id: String,
    pub name: String,
    pub tag: Option<String>,
    pub status: String,
    pub runs_count: i32,
    pub created_at: DateTime<Utc>,
    pub detail_path: String,
}

/// Campaign detail with runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignDetail {
    pub schema_version: String,
    pub campaign: CampaignInfo,
    pub runs: Vec<RunSummary>,
}

/// Campaign info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignInfo {
    pub campaign_id: String,
    pub name: String,
    pub tag: Option<String>,
    pub owner: Option<String>,
    pub status: String,
    pub config_hash: String,
    pub git_sha: Option<String>,
    pub created_at: DateTime<Utc>,
    pub notes: Option<String>,
}

/// Run summary for campaign detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub run_id: String,
    pub seed: i64,
    pub status: String,
    pub data_integrity_verdict: Option<String>,
    pub data_integrity_score: Option<f32>,
    pub candidates_count: Option<i32>,
    pub research_candidates_count: i32,
    pub validated_candidates_count: i32,
    pub best_oos_sharpe_net: Option<f32>,
    pub duration_secs: Option<i32>,
    pub detail_path: String,
    pub export_path: Option<String>,
}

/// Run detail with config and metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunDetail {
    pub schema_version: String,
    pub run: RunInfo,
    pub config_snapshot: Option<serde_json::Value>,
    pub metrics: RunMetrics,
    pub top_candidates: Vec<CandidateSummary>,
    pub exports: RunExports,
}

/// Run info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunInfo {
    pub run_id: String,
    pub campaign_id: String,
    pub seed: i64,
    pub status: String,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub duration_secs: Option<i32>,
    pub artifact_path: Option<String>,
}

/// Run metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetrics {
    pub generations_completed: Option<i32>,
    pub total_evaluations: Option<i64>,
    pub data_integrity_verdict: Option<String>,
    pub data_integrity_score: Option<f32>,
    pub best_oos_sharpe_net: Option<f32>,
    pub best_pbo: Option<f32>,
    pub research_candidates_count: i32,
    pub validated_candidates_count: i32,
}

/// Candidate summary for run detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateSummary {
    pub candidate_id: String,
    pub genome_hash: String,
    pub rank: i32,
    pub candidate_class: String,
    pub oos_sharpe_net: Option<f32>,
    pub pbo: Option<f32>,
    pub oos_cagr_net: Option<f32>,
    pub gates_passed: Option<bool>,
}

/// Run exports links.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunExports {
    pub top1000_json: Option<String>,
    pub top1000_csv: Option<String>,
    pub data_integrity_report: Option<String>,
}

const SITE_SCHEMA_VERSION: &str = "1.0.0";

/// Execute factory build-site command.
pub fn execute_build_site(
    campaign_id: Option<&str>,
    run_id: Option<&str>,
) -> Result<()> {
    let rt = Runtime::new()?;
    rt.block_on(async {
        let registry = Registry::connect().await?;

        // Create output directory
        let site_dir = Path::new("artifacts/site");
        fs::create_dir_all(site_dir)
            .context("Failed to create site directory")?;

        // If specific run requested, just build that run
        if let Some(rid) = run_id {
            build_run_detail(&registry, rid, site_dir).await?;
            println!("✓ Built run detail: artifacts/site/run_{}.json", rid);
            return Ok(());
        }

        // Build campaign(s)
        let campaigns = if let Some(cid) = campaign_id {
            vec![registry.get_campaign(cid).await?
                .ok_or_else(|| anyhow::anyhow!("Campaign not found: {}", cid))?]
        } else {
            registry.list_campaigns(None).await?
        };

        // Build index
        let mut campaign_summaries = Vec::new();

        for campaign in &campaigns {
            // Get runs for this campaign
            let runs = registry.list_runs(&campaign.campaign_id).await?;

            // Build campaign detail
            build_campaign_detail(&registry, &campaign.campaign_id, site_dir).await?;
            println!("✓ Built campaign detail: artifacts/site/campaign_{}.json", campaign.campaign_id);

            // Build run details
            for run in &runs {
                build_run_detail(&registry, &run.run_id, site_dir).await?;
                println!("  ✓ Built run detail: artifacts/site/run_{}.json", run.run_id);
            }

            campaign_summaries.push(CampaignSummary {
                campaign_id: campaign.campaign_id.clone(),
                name: campaign.name.clone(),
                tag: campaign.tag.clone(),
                status: campaign.status.clone(),
                runs_count: runs.len() as i32,
                created_at: campaign.created_at,
                detail_path: format!("campaign_{}.json", campaign.campaign_id),
            });
        }

        // Build index
        let index = SiteIndex {
            schema_version: SITE_SCHEMA_VERSION.to_string(),
            generated_at: Utc::now(),
            campaigns: campaign_summaries,
        };

        let index_path = site_dir.join("index.json");
        let index_json = serde_json::to_string_pretty(&index)
            .context("Failed to serialize index")?;
        fs::write(&index_path, index_json)
            .context("Failed to write index")?;
        println!("✓ Built index: artifacts/site/index.json");

        // Print summary
        println!();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║                    SITE BUNDLE COMPLETE                      ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Schema Version:   {}                                     ", SITE_SCHEMA_VERSION);
        println!("║ Campaigns:        {}                                      ", campaigns.len());
        println!("║ Output Dir:       artifacts/site/                            ║");
        println!("╚══════════════════════════════════════════════════════════════╝");

        Ok(())
    })
}

/// Build campaign detail JSON.
async fn build_campaign_detail(
    registry: &Registry,
    campaign_id: &str,
    site_dir: &Path,
) -> Result<()> {
    let campaign = registry.get_campaign(campaign_id).await?
        .ok_or_else(|| anyhow::anyhow!("Campaign not found: {}", campaign_id))?;

    let runs = registry.list_runs(campaign_id).await?;

    let mut run_summaries = Vec::new();
    for run in &runs {
        // Count candidates by class
        let research_count = registry.get_candidates_by_class(&run.run_id, "research").await?
            .len() as i32;
        let validated_count = registry.get_candidates_by_class(&run.run_id, "validated").await?
            .len() as i32;

        // Check if exports exist
        let export_path = format!("artifacts/top_candidates/{}/top1000.json", run.run_id);
        let export_exists = Path::new(&export_path).exists();

        run_summaries.push(RunSummary {
            run_id: run.run_id.clone(),
            seed: run.seed,
            status: format!("{:?}", run.status),
            data_integrity_verdict: run.data_integrity_verdict.clone(),
            data_integrity_score: run.data_integrity_score,
            candidates_count: run.candidates_count,
            research_candidates_count: research_count,
            validated_candidates_count: validated_count,
            best_oos_sharpe_net: run.best_oos_sharpe_net,
            duration_secs: run.duration_secs,
            detail_path: format!("run_{}.json", run.run_id),
            export_path: if export_exists { Some(export_path) } else { None },
        });
    }

    let detail = CampaignDetail {
        schema_version: SITE_SCHEMA_VERSION.to_string(),
        campaign: CampaignInfo {
            campaign_id: campaign.campaign_id.clone(),
            name: campaign.name.clone(),
            tag: campaign.tag.clone(),
            owner: campaign.owner.clone(),
            status: campaign.status.clone(),
            config_hash: campaign.config_hash.clone(),
            git_sha: campaign.git_sha.clone(),
            created_at: campaign.created_at,
            notes: campaign.notes.clone(),
        },
        runs: run_summaries,
    };

    let path = site_dir.join(format!("campaign_{}.json", campaign_id));
    let json = serde_json::to_string_pretty(&detail)
        .context("Failed to serialize campaign detail")?;
    fs::write(&path, json)
        .context("Failed to write campaign detail")?;

    Ok(())
}

/// Build run detail JSON.
async fn build_run_detail(
    registry: &Registry,
    run_id: &str,
    site_dir: &Path,
) -> Result<()> {
    let run = registry.get_run(run_id).await?
        .ok_or_else(|| anyhow::anyhow!("Run not found: {}", run_id))?;

    // Get top candidates (validated first, then research)
    let validated = registry.get_candidates_by_class(run_id, "validated").await?;
    let research = registry.get_candidates_by_class(run_id, "research").await?;

    let mut top_candidates: Vec<CandidateSummary> = validated.iter()
        .take(20)
        .map(candidate_to_summary)
        .collect();
    top_candidates.extend(research.iter()
        .take(20)
        .map(candidate_to_summary));

    // Sort by sharpe descending
    top_candidates.sort_by(|a, b| {
        b.oos_sharpe_net.unwrap_or(f32::NEG_INFINITY)
            .partial_cmp(&a.oos_sharpe_net.unwrap_or(f32::NEG_INFINITY))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    top_candidates.truncate(20);

    // Check for exports
    let export_dir = format!("artifacts/top_candidates/{}", run_id);
    let top1000_json = format!("{}/top1000.json", export_dir);
    let top1000_csv = format!("{}/top1000.csv", export_dir);
    let di_report = format!("artifacts/data_integrity/{}/report.json", run.campaign_id);

    let detail = RunDetail {
        schema_version: SITE_SCHEMA_VERSION.to_string(),
        run: RunInfo {
            run_id: run.run_id.clone(),
            campaign_id: run.campaign_id.clone(),
            seed: run.seed,
            status: format!("{:?}", run.status),
            started_at: run.started_at,
            completed_at: run.completed_at,
            duration_secs: run.duration_secs,
            artifact_path: run.artifact_path.clone(),
        },
        config_snapshot: None, // TODO: Load from artifact_path if available
        metrics: RunMetrics {
            generations_completed: run.generations_completed,
            total_evaluations: run.total_evaluations,
            data_integrity_verdict: run.data_integrity_verdict.clone(),
            data_integrity_score: run.data_integrity_score,
            best_oos_sharpe_net: run.best_oos_sharpe_net,
            best_pbo: run.best_pbo,
            research_candidates_count: research.len() as i32,
            validated_candidates_count: validated.len() as i32,
        },
        top_candidates,
        exports: RunExports {
            top1000_json: if Path::new(&top1000_json).exists() { Some(top1000_json) } else { None },
            top1000_csv: if Path::new(&top1000_csv).exists() { Some(top1000_csv) } else { None },
            data_integrity_report: if Path::new(&di_report).exists() { Some(di_report) } else { None },
        },
    };

    let path = site_dir.join(format!("run_{}.json", run_id));
    let json = serde_json::to_string_pretty(&detail)
        .context("Failed to serialize run detail")?;
    fs::write(&path, json)
        .context("Failed to write run detail")?;

    Ok(())
}

fn candidate_to_summary(c: &Candidate) -> CandidateSummary {
    CandidateSummary {
        candidate_id: c.candidate_id.clone(),
        genome_hash: c.genome_hash.clone(),
        rank: c.rank,
        candidate_class: c.candidate_class.clone(),
        oos_sharpe_net: c.oos_sharpe_net,
        pbo: c.pbo,
        oos_cagr_net: c.oos_cagr_net,
        gates_passed: c.gates_passed,
    }
}

