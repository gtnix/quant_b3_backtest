//! Factory compare command - Compare candidates across multiple runs.

use anyhow::Result;
use tokio::runtime::Runtime;

use super::registry::Registry;

/// Execute factory compare command.
pub fn execute_compare(run_ids: &[String], top_n: usize) -> Result<()> {
    let rt = Runtime::new()?;
    rt.block_on(async {
        let registry = Registry::connect().await?;

        if run_ids.is_empty() {
            println!("No run IDs provided. Use --runs run1,run2,run3");
            return Ok(());
        }

        println!("╔══════════════════════════════════════════════════════════════════════════════════════════════════╗");
        println!("║                                    CANDIDATE COMPARISON                                          ║");
        println!("╠════════════════╦════════════════╦═══════╦═══════════╦═══════════╦═══════╦═════════╦══════════════╣");
        println!("║ Run ID         ║ Candidate ID   ║ Rank  ║ OOS SR    ║ Gross SR  ║ PBO   ║ Stress  ║ Gates        ║");
        println!("╠════════════════╬════════════════╬═══════╬═══════════╬═══════════╬═══════╬═════════╬══════════════╣");

        let mut all_candidates = Vec::new();

        for run_id in run_ids {
            let candidates = registry.get_top_candidates(run_id, top_n as i32).await?;
            
            for cand in candidates {
                all_candidates.push((run_id.clone(), cand));
            }
        }

        // Sort by OOS Sharpe (descending)
        all_candidates.sort_by(|a, b| {
            let a_sr = a.1.oos_sharpe_net.unwrap_or(f32::MIN);
            let b_sr = b.1.oos_sharpe_net.unwrap_or(f32::MIN);
            b_sr.partial_cmp(&a_sr).unwrap_or(std::cmp::Ordering::Equal)
        });

        for (run_id, cand) in &all_candidates {
            let oos = cand.oos_sharpe_net.map(|s| format!("{:.3}", s)).unwrap_or("-".into());
            let gross = cand.oos_sharpe_gross.map(|s| format!("{:.3}", s)).unwrap_or("-".into());
            let pbo = cand.pbo.map(|p| format!("{:.3}", p)).unwrap_or("-".into());
            let stress = match (cand.stress_passed, cand.stress_total) {
                (Some(p), Some(t)) => format!("{}/{}", p, t),
                _ => "-".into(),
            };
            let gates = cand.gates_passed.map(|g| if g { "PASS" } else { "FAIL" }).unwrap_or("-");

            println!(
                "║ {:<14} ║ {:<14} ║ {:<5} ║ {:<9} ║ {:<9} ║ {:<5} ║ {:<7} ║ {:<12} ║",
                &run_id[..14.min(run_id.len())],
                &cand.candidate_id[..14.min(cand.candidate_id.len())],
                cand.rank,
                oos,
                gross,
                pbo,
                stress,
                gates
            );
        }

        println!("╚════════════════╩════════════════╩═══════╩═══════════╩═══════════╩═══════╩═════════╩══════════════╝");
        println!("\nTotal: {} candidates from {} runs", all_candidates.len(), run_ids.len());

        // Summary statistics
        if !all_candidates.is_empty() {
            let avg_oos: f32 = all_candidates
                .iter()
                .filter_map(|(_, c)| c.oos_sharpe_net)
                .sum::<f32>()
                / all_candidates.len() as f32;

            let best_oos = all_candidates
                .iter()
                .filter_map(|(_, c)| c.oos_sharpe_net)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);

            let gates_passed = all_candidates
                .iter()
                .filter(|(_, c)| c.gates_passed == Some(true))
                .count();

            println!("\n╔══════════════════════════════════════════════════════════════╗");
            println!("║                         SUMMARY                              ║");
            println!("╠══════════════════════════════════════════════════════════════╣");
            println!("║ Average OOS Sharpe: {:.3}                                    ", avg_oos);
            println!("║ Best OOS Sharpe:    {:.3}                                    ", best_oos);
            println!("║ Gates Passed:       {} / {}                                  ", gates_passed, all_candidates.len());
            println!("╚══════════════════════════════════════════════════════════════╝");
        }

        Ok(())
    })
}
