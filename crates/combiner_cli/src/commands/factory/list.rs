//! Factory list and show commands.

use anyhow::Result;
use tokio::runtime::Runtime;
use tracing::info;

use super::registry::Registry;

/// Execute factory list command.
pub fn execute_list(tag: Option<&str>) -> Result<()> {
    let rt = Runtime::new()?;
    rt.block_on(async {
        let registry = Registry::connect().await?;

        let campaigns = registry.list_campaigns(tag).await?;

        if campaigns.is_empty() {
            println!("No campaigns found.");
            if tag.is_some() {
                println!("Try without --tag filter.");
            }
            return Ok(());
        }

        println!("╔══════════════════════════════════════════════════════════════════════════════╗");
        println!("║                              CAMPAIGNS                                       ║");
        println!("╠════════════════╦════════════════════╦═════════╦═══════════╦═════════════════╣");
        println!("║ Campaign ID    ║ Name               ║ Status  ║ Seeds     ║ Created         ║");
        println!("╠════════════════╬════════════════════╬═════════╬═══════════╬═════════════════╣");

        for campaign in &campaigns {
            let name = if campaign.name.len() > 18 {
                format!("{}...", &campaign.name[..15])
            } else {
                format!("{:<18}", campaign.name)
            };

            let seeds_str = format!("{}/{}", 
                campaign.seeds.len(), 
                campaign.seeds.len()
            );

            let created = campaign.created_at.format("%Y-%m-%d %H:%M");

            println!(
                "║ {:<14} ║ {} ║ {:<7} ║ {:<9} ║ {:<15} ║",
                &campaign.campaign_id[..14.min(campaign.campaign_id.len())],
                name,
                campaign.status,
                seeds_str,
                created
            );
        }

        println!("╚════════════════╩════════════════════╩═════════╩═══════════╩═════════════════╝");
        println!("\nTotal: {} campaigns", campaigns.len());
        println!("\nUse 'combiner factory show <campaign_id>' for details.");

        Ok(())
    })
}

/// Execute factory show command.
pub fn execute_show(id: &str) -> Result<()> {
    let rt = Runtime::new()?;
    rt.block_on(async {
        let registry = Registry::connect().await?;

        // Try to find as campaign first
        if let Some(campaign) = registry.get_campaign(id).await? {
            show_campaign(&registry, &campaign).await?;
            return Ok(());
        }

        // Try to find as run
        if let Some(run) = registry.get_run(id).await? {
            show_run(&registry, &run).await?;
            return Ok(());
        }

        println!("No campaign or run found with ID: {}", id);
        Ok(())
    })
}

async fn show_campaign(registry: &Registry, campaign: &super::registry::Campaign) -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                      CAMPAIGN DETAILS                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ ID:          {}                            ", campaign.campaign_id);
    println!("║ Name:        {}                            ", campaign.name);
    println!("║ Status:      {}                            ", campaign.status);
    println!("║ Tag:         {}                            ", campaign.tag.as_deref().unwrap_or("-"));
    println!("║ Owner:       {}                            ", campaign.owner.as_deref().unwrap_or("-"));
    println!("║ Git Branch:  {}                            ", campaign.git_branch.as_deref().unwrap_or("-"));
    println!("║ Git SHA:     {}                            ", campaign.git_sha.as_deref().unwrap_or("-"));
    println!("║ Config Hash: {}                            ", campaign.config_hash);
    println!("║ Seeds:       {:?}                          ", campaign.seeds);
    println!("║ Created:     {}                            ", campaign.created_at.format("%Y-%m-%d %H:%M:%S"));
    println!("╚══════════════════════════════════════════════════════════════╝");

    // List runs
    let runs = registry.list_runs(&campaign.campaign_id).await?;

    if !runs.is_empty() {
        println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
        println!("║                              RUNS                                            ║");
        println!("╠════════════════╦═══════╦═══════════╦══════════╦═══════════╦═════════════════╣");
        println!("║ Run ID         ║ Seed  ║ Status    ║ Duration ║ OOS SR    ║ Candidates      ║");
        println!("╠════════════════╬═══════╬═══════════╬══════════╬═══════════╬═════════════════╣");

        for run in &runs {
            let duration = run.duration_secs
                .map(|d| format!("{}s", d))
                .unwrap_or_else(|| "-".to_string());

            let oos_sharpe = run.best_oos_sharpe_net
                .map(|s| format!("{:.2}", s))
                .unwrap_or_else(|| "-".to_string());

            let candidates = run.candidates_count
                .map(|c| c.to_string())
                .unwrap_or_else(|| "-".to_string());

            println!(
                "║ {:<14} ║ {:<5} ║ {:<9} ║ {:<8} ║ {:<9} ║ {:<15} ║",
                &run.run_id[..14.min(run.run_id.len())],
                run.seed,
                run.status,
                duration,
                oos_sharpe,
                candidates
            );
        }

        println!("╚════════════════╩═══════╩═══════════╩══════════╩═══════════╩═════════════════╝");
    }

    Ok(())
}

async fn show_run(registry: &Registry, run: &super::registry::Run) -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                        RUN DETAILS                           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Run ID:       {}                           ", run.run_id);
    println!("║ Campaign ID:  {}                           ", run.campaign_id);
    println!("║ Seed:         {}                           ", run.seed);
    println!("║ Status:       {}                           ", run.status);
    println!("║ Started:      {}                           ", run.started_at.format("%Y-%m-%d %H:%M:%S"));
    if let Some(ref completed) = run.completed_at {
        println!("║ Completed:    {}                           ", completed.format("%Y-%m-%d %H:%M:%S"));
    }
    if let Some(duration) = run.duration_secs {
        println!("║ Duration:     {}s                          ", duration);
    }
    if let Some(gens) = run.generations_completed {
        println!("║ Generations:  {}                           ", gens);
    }
    if let Some(evals) = run.total_evaluations {
        println!("║ Evaluations:  {}                           ", evals);
    }
    if let Some(sharpe) = run.best_oos_sharpe_net {
        println!("║ Best OOS SR:  {:.3}                         ", sharpe);
    }
    if let Some(pbo) = run.best_pbo {
        println!("║ Best PBO:     {:.3}                         ", pbo);
    }
    if let Some(ref path) = run.artifact_path {
        println!("║ Artifacts:    {}                           ", path);
    }
    if let Some(ref error) = run.error_message {
        println!("║ Error:        {}                           ", error);
    }
    println!("╚══════════════════════════════════════════════════════════════╝");

    // List candidates
    let candidates = registry.get_candidates(&run.run_id).await?;

    if !candidates.is_empty() {
        println!("\n╔══════════════════════════════════════════════════════════════════════════════════════╗");
        println!("║                                    CANDIDATES                                        ║");
        println!("╠══════╦════════════════╦═══════════╦═══════════╦═══════╦═════════╦══════════╦═════════╣");
        println!("║ Rank ║ Candidate ID   ║ OOS SR    ║ Gross SR  ║ PBO   ║ Stress  ║ Turnover ║ Gates   ║");
        println!("╠══════╬════════════════╬═══════════╬═══════════╬═══════╬═════════╬══════════╬═════════╣");

        for cand in &candidates {
            let oos = cand.oos_sharpe_net.map(|s| format!("{:.2}", s)).unwrap_or("-".into());
            let gross = cand.oos_sharpe_gross.map(|s| format!("{:.2}", s)).unwrap_or("-".into());
            let pbo = cand.pbo.map(|p| format!("{:.2}", p)).unwrap_or("-".into());
            let stress = match (cand.stress_passed, cand.stress_total) {
                (Some(p), Some(t)) => format!("{}/{}", p, t),
                _ => "-".into(),
            };
            let turnover = cand.turnover_annual.map(|t| format!("{:.1}x", t)).unwrap_or("-".into());
            let gates = cand.gates_passed.map(|g| if g { "PASS" } else { "FAIL" }).unwrap_or("-");

            println!(
                "║ {:<4} ║ {:<14} ║ {:<9} ║ {:<9} ║ {:<5} ║ {:<7} ║ {:<8} ║ {:<7} ║",
                cand.rank,
                &cand.candidate_id[..14.min(cand.candidate_id.len())],
                oos,
                gross,
                pbo,
                stress,
                turnover,
                gates
            );
        }

        println!("╚══════╩════════════════╩═══════════╩═══════════╩═══════╩═════════╩══════════╩═════════╝");
    }

    Ok(())
}

