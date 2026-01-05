//! Factory promote command - Promote candidates from research to paper trading.

use anyhow::Result;
use tokio::runtime::Runtime;

use super::bundle::BundleGenerator;
use super::registry::{generate_promotion_id, Registry};

/// Promotion stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromotionStage {
    Research,
    Candidate,
    Paper,
}

impl PromotionStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Research => "research",
            Self::Candidate => "candidate",
            Self::Paper => "paper",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "research" => Some(Self::Research),
            "candidate" => Some(Self::Candidate),
            "paper" => Some(Self::Paper),
            _ => None,
        }
    }
}

/// Promotion criteria.
pub struct PromotionCriteria {
    pub min_oos_sharpe_net: f32,
    pub max_pbo: f32,
    pub min_stress_passed: i32,
    pub gates_required: bool,
    pub min_dsr: Option<f32>,
}

impl Default for PromotionCriteria {
    fn default() -> Self {
        Self {
            min_oos_sharpe_net: 0.5,
            max_pbo: 0.15,
            min_stress_passed: 4,
            gates_required: true,
            min_dsr: None,
        }
    }
}

/// Execute factory promote command.
pub fn execute_promote(
    run_id: Option<&str>,
    campaign_id: Option<&str>,
    top_n: usize,
    stage: &str,
    force: bool,
) -> Result<()> {
    let stage = PromotionStage::from_str(stage)
        .ok_or_else(|| anyhow::anyhow!("Invalid stage: {}. Use: research, candidate, paper", stage))?;

    let rt = Runtime::new()?;
    rt.block_on(async {
        let registry = Registry::connect().await?;

        // Get run IDs to process
        let run_ids: Vec<String> = if let Some(run) = run_id {
            vec![run.to_string()]
        } else if let Some(camp) = campaign_id {
            let runs = registry.list_runs(camp).await?;
            runs.iter()
                .filter(|r| r.status == "completed")
                .map(|r| r.run_id.clone())
                .collect()
        } else {
            return Err(anyhow::anyhow!("Provide --run or --campaign"));
        };

        if run_ids.is_empty() {
            println!("No completed runs found.");
            return Ok(());
        }

        // Get campaign info for config hash
        let first_run = registry.get_run(&run_ids[0]).await?
            .ok_or_else(|| anyhow::anyhow!("Run not found"))?;
        let campaign = registry.get_campaign(&first_run.campaign_id).await?
            .ok_or_else(|| anyhow::anyhow!("Campaign not found"))?;

        let criteria = PromotionCriteria::default();
        let bundle_gen = BundleGenerator::new("artifacts/candidates");

        let mut promoted = 0;
        let mut skipped_already = 0;
        let mut skipped_criteria = 0;

        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║                  PROMOTION PIPELINE                          ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Stage:        {}                                 ", stage.as_str());
        println!("║ Top N:        {}                                 ", top_n);
        println!("║ Runs:         {}                                 ", run_ids.len());
        println!("║ Criteria:                                                    ║");
        println!("║   OOS Sharpe >= {:.2}                                        ", criteria.min_oos_sharpe_net);
        println!("║   PBO <= {:.2}                                               ", criteria.max_pbo);
        println!("║   Stress >= {}                                               ", criteria.min_stress_passed);
        println!("║   Gates: {}                                                  ", if criteria.gates_required { "required" } else { "optional" });
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();

        for run_id in &run_ids {
            let run = registry.get_run(run_id).await?
                .ok_or_else(|| anyhow::anyhow!("Run not found: {}", run_id))?;

            // Check data integrity - skip runs without PASS verdict
            if run.data_integrity_verdict.as_deref() != Some("PASS") {
                println!("⚠️  Skipping run {}: data integrity not PASS ({:?})", run_id, run.data_integrity_verdict);
                continue;
            }

            let candidates = registry.get_top_candidates(run_id, top_n as i32).await?;

            for cand in candidates {
                // Check if already promoted
                if !force && registry.is_already_promoted(&cand.genome_hash, stage.as_str()).await? {
                    skipped_already += 1;
                    continue;
                }

                // Check criteria
                let meets_criteria = check_criteria(&cand, &criteria);
                if !meets_criteria {
                    skipped_criteria += 1;
                    continue;
                }

                // Generate bundle
                let bundle_path = bundle_gen.generate(
                    &cand,
                    &first_run.campaign_id,
                    run.seed,
                    &campaign.config_hash,
                    campaign.dataset_hash.as_deref(),
                    None, // Will try to find strategy.toml from artifacts
                    None, // Will use default execution config
                )?;

                // Register promotion
                let promotion_id = generate_promotion_id();
                registry.register_promotion(
                    &promotion_id,
                    &cand.candidate_id,
                    stage.as_str(),
                    None, // promoted_by
                    Some(&bundle_path),
                    None, // notes
                ).await?;

                println!("✓ Promoted: {} → {}", cand.candidate_id, bundle_path);
                promoted += 1;
            }
        }

        println!();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║                  PROMOTION SUMMARY                           ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Promoted:            {}                                      ", promoted);
        println!("║ Skipped (duplicate): {}                                      ", skipped_already);
        println!("║ Skipped (criteria):  {}                                      ", skipped_criteria);
        println!("╚══════════════════════════════════════════════════════════════╝");

        if promoted > 0 {
            println!("\nBundles created in: artifacts/candidates/");
            println!("View with: ls -la artifacts/candidates/");
        }

        Ok(())
    })
}

/// Hall of Fame promotion criteria (stricter than standard).
#[derive(Debug, Clone)]
pub struct HallOfFameCriteria {
    pub min_oos_sharpe_net: f32,
    pub max_pbo: f32,
    pub min_dsr: f32,
    pub max_drawdown_net: f32,
    pub gates_required: bool,
}

impl Default for HallOfFameCriteria {
    fn default() -> Self {
        Self {
            min_oos_sharpe_net: 0.5,
            max_pbo: 0.20,
            min_dsr: 0.4,
            max_drawdown_net: 0.30,
            gates_required: true,
        }
    }
}

impl HallOfFameCriteria {
    /// Create criteria from PromotionConfig (reads max_drawdown from TOML).
    pub fn from_promotion_config(config: &super::config::PromotionConfig) -> Self {
        Self {
            min_oos_sharpe_net: config.min_oos_sharpe_net as f32,
            max_pbo: config.max_pbo as f32,
            min_dsr: config.min_dsr.unwrap_or(0.4) as f32,
            // Convert negative drawdown to positive for comparison (e.g. -0.25 -> 0.25)
            max_drawdown_net: config.max_drawdown.abs() as f32,
            gates_required: config.gates_required,
        }
    }
}

/// Auto-promote candidates to Hall of Fame after a run completes.
/// Called automatically at the end of `factory run`.
pub async fn auto_promote_to_hall_of_fame(
    registry: &Registry,
    run_id: &str,
    market: &str,
    criteria: Option<HallOfFameCriteria>,
) -> Result<usize> {
    use tracing::info;
    
    let criteria = criteria.unwrap_or_default();
    let candidates = registry.get_top_candidates(run_id, 100).await?;
    
    let mut promoted = 0;
    let mut skipped_criteria = 0;
    let mut skipped_duplicate = 0;
    
    // Get git SHA for provenance
    let git_sha = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().chars().take(40).collect::<String>());
    
    for cand in candidates {
        // Check all criteria
        let sharpe = cand.oos_sharpe_net.unwrap_or(0.0);
        let pbo = cand.pbo.unwrap_or(1.0);
        let dsr = cand.dsr.unwrap_or(0.0);
        let max_dd = cand.max_drawdown_net.unwrap_or(-1.0).abs();
        let gates = cand.gates_passed.unwrap_or(false);
        
        let meets_sharpe = sharpe >= criteria.min_oos_sharpe_net;
        let meets_pbo = pbo <= criteria.max_pbo;
        let meets_dsr = dsr >= criteria.min_dsr;
        let meets_dd = max_dd <= criteria.max_drawdown_net;
        let meets_gates = !criteria.gates_required || gates;
        
        if !meets_sharpe || !meets_pbo || !meets_dsr || !meets_dd || !meets_gates {
            skipped_criteria += 1;
            continue;
        }
        
        // Check if already promoted
        if registry.is_already_promoted(&cand.genome_hash, "hall_of_fame").await? {
            skipped_duplicate += 1;
            continue;
        }
        
        // Promote!
        let promotion_id = generate_promotion_id();
        registry.register_hall_of_fame_promotion(
            &promotion_id,
            &cand.candidate_id,
            sharpe,
            pbo,
            Some(dsr),
            cand.max_drawdown_net,
            cand.oos_cagr_net,
            cand.stress_passed,
            cand.stress_total,
            gates,
            git_sha.as_deref(),
            market,
            &format!("Auto-promoted by Rust from run {}", run_id),
        ).await?;
        
        promoted += 1;
    }
    
    if promoted > 0 {
        info!(run_id, promoted, skipped_criteria, skipped_duplicate, 
              "Hall of Fame auto-promotion complete");
    }
    
    Ok(promoted)
}

fn check_criteria(cand: &super::registry::Candidate, criteria: &PromotionCriteria) -> bool {
    // Check OOS Sharpe
    if let Some(sharpe) = cand.oos_sharpe_net {
        if sharpe < criteria.min_oos_sharpe_net {
            return false;
        }
    } else {
        return false;
    }

    // Check PBO
    if let Some(pbo) = cand.pbo {
        if pbo > criteria.max_pbo {
            return false;
        }
    }

    // Check stress passed
    if let Some(passed) = cand.stress_passed {
        if passed < criteria.min_stress_passed {
            return false;
        }
    }

    // Check gates
    if criteria.gates_required {
        if cand.gates_passed != Some(true) {
            return false;
        }
    }

    // Check DSR if required
    if let Some(min_dsr) = criteria.min_dsr {
        if let Some(dsr) = cand.dsr {
            if dsr < min_dsr as f32 {
                return false;
            }
        }
    }

    true
}

