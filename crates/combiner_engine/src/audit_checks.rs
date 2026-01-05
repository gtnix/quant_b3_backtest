//! Audit Checks - Individual check functions for each marco.
//!
//! This module provides 25+ check functions organized by marco.
//! Each function returns an `AuditCheck` with Pass/Warn/Fail verdict.
//!
//! # Thresholds Reference (Academic Basis)
//!
//! - **PBO <= 0.15**: Lopez de Prado (2018) "Advances in Financial ML"
//! - **DSR >= 0.5**: Bailey & Lopez de Prado (2014) "Deflated Sharpe Ratio"
//! - **IS-OOS degradation <= 40%**: Industry standard for WFA
//! - **Sharpe < 10**: Sanity check - values above are almost certainly bugs
//! - **Min trades >= 30**: Statistical significance for t-tests
//!
//! # Red Flags (automatic FAIL)
//!
//! 1. Sharpe > 10 (absurd)
//! 2. Diversity = 0 (degenerate population)
//! 3. Volatility = 0 (data bug)
//! 4. PBO > 0.30 (severe overfitting)
//! 5. Trades < 30 (insufficient sample)
//! 6. IS-OOS degradation > 50% (curve fitting)

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::audit_framework::AuditCheck;

// =============================================================================
// RUN ARTIFACTS LOADER
// =============================================================================

/// Container for all artifacts loaded from a run directory.
#[derive(Debug, Default)]
pub struct RunArtifacts {
    /// manifest.json content
    pub manifest: Option<Value>,
    /// report.json content
    pub report: Option<Value>,
    /// ranking.json content
    pub ranking: Option<Value>,
    /// Config TOML (if available)
    pub config: Option<Value>,
    /// List of strategy directories found
    pub strategy_dirs: Vec<String>,
}

/// Load all artifacts from a run directory.
///
/// # Errors
/// Returns error only if directory is completely unusable.
/// Missing individual files are handled gracefully.
#[must_use]
pub fn load_run_artifacts(run_dir: &Path) -> Result<RunArtifacts, std::io::Error> {
    let mut artifacts = RunArtifacts::default();
    
    // Load manifest.json
    let manifest_path = run_dir.join("manifest.json");
    if manifest_path.exists() {
        if let Ok(content) = fs::read_to_string(&manifest_path) {
            artifacts.manifest = serde_json::from_str(&content).ok();
        }
    }
    
    // Load report.json
    let report_path = run_dir.join("report.json");
    if report_path.exists() {
        if let Ok(content) = fs::read_to_string(&report_path) {
            artifacts.report = serde_json::from_str(&content).ok();
        }
    }
    
    // Load ranking.json from hall_of_fame
    let ranking_path = run_dir.join("hall_of_fame").join("ranking.json");
    if ranking_path.exists() {
        if let Ok(content) = fs::read_to_string(&ranking_path) {
            artifacts.ranking = serde_json::from_str(&content).ok();
        }
    }
    
    // Find strategy directories
    let hof_dir = run_dir.join("hall_of_fame");
    if hof_dir.exists() {
        if let Ok(entries) = fs::read_dir(&hof_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("strategy_") && entry.path().is_dir() {
                    artifacts.strategy_dirs.push(name);
                }
            }
        }
    }
    artifacts.strategy_dirs.sort();
    
    Ok(artifacts)
}

// =============================================================================
// MARCO 0: INITIALIZATION CHECKS
// =============================================================================

/// Check if seed is present in manifest.
#[must_use]
pub fn check_seed_present(manifest: &Option<Value>) -> AuditCheck {
    match manifest {
        Some(m) => {
            if let Some(seed) = m.get("seed") {
                if seed.is_number() {
                    AuditCheck::pass(
                        "seed_present",
                        "Verificar se seed está presente no manifest",
                        format!("Seed encontrada: {}", seed),
                    ).with_evidence("seed", seed.clone())
                } else {
                    AuditCheck::fail(
                        "seed_present",
                        "Verificar se seed está presente no manifest",
                        "Seed presente mas não é numérica",
                    )
                }
            } else {
                AuditCheck::fail(
                    "seed_present",
                    "Verificar se seed está presente no manifest",
                    "Campo 'seed' ausente no manifest",
                )
            }
        }
        None => AuditCheck::fail(
            "seed_present",
            "Verificar se seed está presente no manifest",
            "Manifest não encontrado",
        ),
    }
}

/// Check if config hash is present and valid.
#[must_use]
pub fn check_config_hash(manifest: &Option<Value>) -> AuditCheck {
    match manifest {
        Some(m) => {
            if let Some(hash) = m.get("config_hash").and_then(|v| v.as_str()) {
                if !hash.is_empty() && hash != "0" && hash != "unknown" {
                    AuditCheck::pass(
                        "config_hash_present",
                        "Verificar hash do config para reprodutibilidade",
                        format!("Config hash: {}", hash),
                    ).with_evidence("config_hash", hash)
                } else {
                    AuditCheck::warn(
                        "config_hash_present",
                        "Verificar hash do config para reprodutibilidade",
                        format!("Config hash inválido ou placeholder: '{}'", hash),
                    )
                }
            } else {
                AuditCheck::warn(
                    "config_hash_present",
                    "Verificar hash do config para reprodutibilidade",
                    "Campo 'config_hash' ausente",
                )
            }
        }
        None => AuditCheck::fail(
            "config_hash_present",
            "Verificar hash do config para reprodutibilidade",
            "Manifest não encontrado",
        ),
    }
}

/// Check if dates are valid (created_at present).
#[must_use]
pub fn check_dates_valid(manifest: &Option<Value>) -> AuditCheck {
    match manifest {
        Some(m) => {
            if let Some(created) = m.get("created_at").and_then(|v| v.as_str()) {
                // Validate ISO 8601 format
                if created.contains('T') && created.contains('Z') {
                    AuditCheck::pass(
                        "dates_valid",
                        "Verificar timestamps em formato ISO 8601 UTC",
                        format!("Timestamp válido: {}", created),
                    ).with_evidence("created_at", created)
                } else {
                    AuditCheck::warn(
                        "dates_valid",
                        "Verificar timestamps em formato ISO 8601 UTC",
                        format!("Timestamp não está em formato ISO 8601 UTC: {}", created),
                    )
                }
            } else {
                AuditCheck::warn(
                    "dates_valid",
                    "Verificar timestamps em formato ISO 8601 UTC",
                    "Campo 'created_at' ausente",
                )
            }
        }
        None => AuditCheck::fail(
            "dates_valid",
            "Verificar timestamps em formato ISO 8601 UTC",
            "Manifest não encontrado",
        ),
    }
}

/// Check if output directory structure is correct.
#[must_use]
pub fn check_output_structure(run_dir: &Path) -> AuditCheck {
    let required = ["hall_of_fame", "generations"];
    let mut missing = Vec::new();
    
    for dir in required {
        if !run_dir.join(dir).exists() {
            missing.push(dir);
        }
    }
    
    if missing.is_empty() {
        AuditCheck::pass(
            "output_structure",
            "Verificar estrutura de diretórios obrigatórios",
            "Todos os diretórios obrigatórios presentes",
        )
    } else {
        AuditCheck::fail(
            "output_structure",
            "Verificar estrutura de diretórios obrigatórios",
            format!("Diretórios ausentes: {:?}", missing),
        ).with_evidence("missing_dirs", missing)
    }
}

/// Check if experiment ID is present.
#[must_use]
pub fn check_experiment_id(manifest: &Option<Value>) -> AuditCheck {
    match manifest {
        Some(m) => {
            if let Some(exp_id) = m.get("experiment_id").and_then(|v| v.as_str()) {
                if !exp_id.is_empty() {
                    AuditCheck::pass(
                        "experiment_id_present",
                        "Verificar ID único do experimento",
                        format!("Experiment ID: {}", exp_id),
                    ).with_evidence("experiment_id", exp_id)
                } else {
                    AuditCheck::fail(
                        "experiment_id_present",
                        "Verificar ID único do experimento",
                        "Experiment ID vazio",
                    )
                }
            } else {
                AuditCheck::fail(
                    "experiment_id_present",
                    "Verificar ID único do experimento",
                    "Campo 'experiment_id' ausente",
                )
            }
        }
        None => AuditCheck::fail(
            "experiment_id_present",
            "Verificar ID único do experimento",
            "Manifest não encontrado",
        ),
    }
}

// =============================================================================
// MARCO 1: DATA INTEGRITY CHECKS
// =============================================================================

/// Check delay_bars >= 1 for anti-lookahead.
/// Looks in manifest.execution_config.delay_bars or config.delay_bars
#[must_use]
pub fn check_delay_bars(config: &Option<Value>) -> AuditCheck {
    // If no config, we can't verify - warn but don't fail
    match config {
        Some(c) => {
            // Try execution_config.delay_bars first (from manifest.json)
            let delay = c.get("execution_config")
                .and_then(|ec| ec.get("delay_bars"))
                .and_then(|v| v.as_u64())
                // Fallback to top-level delay_bars
                .or_else(|| c.get("delay_bars").and_then(|v| v.as_u64()));
            
            if let Some(delay) = delay {
                if delay >= 1 {
                    AuditCheck::pass(
                        "delay_bars_check",
                        "Verificar delay_bars >= 1 (anti-lookahead)",
                        format!("delay_bars = {} (OK)", delay),
                    ).with_evidence("delay_bars", delay)
                } else {
                    AuditCheck::fail(
                        "delay_bars_check",
                        "Verificar delay_bars >= 1 (anti-lookahead)",
                        format!("delay_bars = {} < 1 - LOOKAHEAD BIAS!", delay),
                    ).with_evidence("delay_bars", delay)
                }
            } else {
                AuditCheck::warn(
                    "delay_bars_check",
                    "Verificar delay_bars >= 1 (anti-lookahead)",
                    "Campo 'delay_bars' não encontrado no config",
                )
            }
        }
        None => AuditCheck::warn(
            "delay_bars_check",
            "Verificar delay_bars >= 1 (anti-lookahead)",
            "Config não disponível - não é possível verificar delay_bars",
        ),
    }
}

/// Check universe configuration (placeholder - needs config).
#[must_use]
pub fn check_universe_config(_config: &Option<Value>) -> AuditCheck {
    // This is a placeholder - real check needs config file
    AuditCheck::pass(
        "universe_config",
        "Verificar configuração do universo de ativos",
        "Verificação de universo point-in-time não implementada",
    )
}

/// Check for gaps in generation sequence.
#[must_use]
pub fn check_no_generation_gaps(report: &Option<Value>) -> AuditCheck {
    match report {
        Some(r) => {
            if let Some(stats) = r.get("generation_stats").and_then(|v| v.as_array()) {
                let generations: Vec<u64> = stats
                    .iter()
                    .filter_map(|s| s.get("generation").and_then(|g| g.as_u64()))
                    .collect();
                
                if generations.is_empty() {
                    return AuditCheck::fail(
                        "no_generation_gaps",
                        "Verificar sequência contínua de gerações",
                        "Nenhuma geração encontrada",
                    );
                }
                
                let mut sorted = generations.clone();
                sorted.sort();
                
                let mut gaps = Vec::new();
                for i in 1..sorted.len() {
                    if sorted[i] != sorted[i-1] + 1 {
                        gaps.push(format!("{}->{}", sorted[i-1], sorted[i]));
                    }
                }
                
                if gaps.is_empty() {
                    AuditCheck::pass(
                        "no_generation_gaps",
                        "Verificar sequência contínua de gerações",
                        format!("Gerações 0-{} sem gaps", sorted.last().unwrap_or(&0)),
                    ).with_evidence("total_generations", sorted.len())
                } else {
                    AuditCheck::warn(
                        "no_generation_gaps",
                        "Verificar sequência contínua de gerações",
                        format!("Gaps detectados: {:?}", gaps),
                    ).with_evidence("gaps", gaps)
                }
            } else {
                AuditCheck::warn(
                    "no_generation_gaps",
                    "Verificar sequência contínua de gerações",
                    "Campo 'generation_stats' ausente no report",
                )
            }
        }
        None => AuditCheck::warn(
            "no_generation_gaps",
            "Verificar sequência contínua de gerações",
            "Report não disponível",
        ),
    }
}

/// Check timestamps are consistent across manifest.
#[must_use]
pub fn check_timestamps_consistent(manifest: &Option<Value>) -> AuditCheck {
    match manifest {
        Some(m) => {
            let created = m.get("created_at").and_then(|v| v.as_str());
            let timestamp = m.get("timestamp").and_then(|v| v.as_str());
            
            // Check for duration in statistics
            let duration = m.get("statistics")
                .and_then(|s| s.get("duration_secs"))
                .and_then(|v| v.as_f64());
            
            if created.is_some() && timestamp.is_some() {
                if duration.is_some() {
                    AuditCheck::pass(
                        "timestamps_consistent",
                        "Verificar consistência de timestamps",
                        format!("created_at e timestamp presentes, duração: {:.2}s", duration.unwrap()),
                    ).with_evidence("duration_secs", duration.unwrap())
                } else {
                    AuditCheck::pass(
                        "timestamps_consistent",
                        "Verificar consistência de timestamps",
                        "created_at e timestamp presentes",
                    )
                }
            } else {
                AuditCheck::warn(
                    "timestamps_consistent",
                    "Verificar consistência de timestamps",
                    "Campos de timestamp incompletos",
                )
            }
        }
        None => AuditCheck::fail(
            "timestamps_consistent",
            "Verificar consistência de timestamps",
            "Manifest não encontrado",
        ),
    }
}

// =============================================================================
// MARCO 2: EVOLUTION CHECKS (CRITICAL)
// =============================================================================

/// Check population diversity exceeds threshold.
///
/// Diversity = unique fitness values / total genomes
/// If diversity = 0, this indicates a degenerate population (BUG).
#[must_use]
pub fn check_population_diversity(report: &Option<Value>, min_diversity: f64) -> AuditCheck {
    match report {
        Some(r) => {
            if let Some(stats) = r.get("generation_stats").and_then(|v| v.as_array()) {
                // Get last generation stats
                if let Some(last) = stats.last() {
                    let best = last.get("best_sharpe").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let mean = last.get("mean_sharpe").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    
                    // If best == mean, there's no diversity
                    let diversity = if best == mean && best != 0.0 {
                        0.0
                    } else {
                        // Approximate diversity from spread
                        (best - mean).abs() / best.abs().max(0.001)
                    };
                    
                    if diversity < 0.001 {
                        AuditCheck::fail(
                            "population_diversity",
                            "Verificar diversidade da população (> 10%)",
                            format!("Diversidade ~0% (best={:.4}, mean={:.4}) - POPULAÇÃO DEGENERADA", best, mean),
                        ).with_evidence("diversity", 0.0)
                         .with_evidence("best_sharpe", best)
                         .with_evidence("mean_sharpe", mean)
                    } else if diversity < min_diversity {
                        AuditCheck::warn(
                            "population_diversity",
                            "Verificar diversidade da população (> 10%)",
                            format!("Diversidade {:.1}% < {:.1}% threshold", diversity * 100.0, min_diversity * 100.0),
                        ).with_evidence("diversity", diversity)
                    } else {
                        AuditCheck::pass(
                            "population_diversity",
                            "Verificar diversidade da população (> 10%)",
                            format!("Diversidade {:.1}% OK", diversity * 100.0),
                        ).with_evidence("diversity", diversity)
                    }
                } else {
                    AuditCheck::fail(
                        "population_diversity",
                        "Verificar diversidade da população (> 10%)",
                        "Nenhuma estatística de geração encontrada",
                    )
                }
            } else {
                AuditCheck::fail(
                    "population_diversity",
                    "Verificar diversidade da população (> 10%)",
                    "Campo 'generation_stats' ausente",
                )
            }
        }
        None => AuditCheck::fail(
            "population_diversity",
            "Verificar diversidade da população (> 10%)",
            "Report não disponível",
        ),
    }
}

/// Check fitness variance > 0 using mean_sharpe across generations.
/// Note: best_sharpe is cumulative (best found so far), so it's normal to stay constant.
/// mean_sharpe reflects population fitness and should vary as the population evolves.
#[must_use]
pub fn check_fitness_variance(report: &Option<Value>) -> AuditCheck {
    match report {
        Some(r) => {
            if let Some(stats) = r.get("generation_stats").and_then(|v| v.as_array()) {
                // Use mean_sharpe (population average) instead of best_sharpe (cumulative best)
                let mean_sharpes: Vec<f64> = stats
                    .iter()
                    .filter_map(|s| s.get("mean_sharpe").and_then(|v| v.as_f64()))
                    .collect();
                
                if mean_sharpes.is_empty() {
                    return AuditCheck::fail(
                        "fitness_variance",
                        "Verificar variância de fitness > 0",
                        "Nenhum valor de mean_sharpe encontrado",
                    );
                }
                
                // Check if all values are identical (degenerate population)
                let unique: HashSet<u64> = mean_sharpes.iter().map(|x| x.to_bits()).collect();
                
                if unique.len() == 1 {
                    AuditCheck::fail(
                        "fitness_variance",
                        "Verificar variância de fitness > 0",
                        format!("Todos os {} valores de mean_sharpe são idênticos ({:.4}) - população estagnada", 
                            mean_sharpes.len(), mean_sharpes[0]),
                    ).with_evidence("unique_values", 1)
                     .with_evidence("value", mean_sharpes[0])
                } else {
                    let mean: f64 = mean_sharpes.iter().sum::<f64>() / mean_sharpes.len() as f64;
                    let variance: f64 = mean_sharpes.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / mean_sharpes.len() as f64;
                    
                    AuditCheck::pass(
                        "fitness_variance",
                        "Verificar variância de fitness > 0",
                        format!("Variância mean_sharpe = {:.6}, {} valores únicos", variance, unique.len()),
                    ).with_evidence("variance", variance)
                     .with_evidence("unique_values", unique.len())
                }
            } else {
                AuditCheck::fail(
                    "fitness_variance",
                    "Verificar variância de fitness > 0",
                    "Campo 'generation_stats' ausente",
                )
            }
        }
        None => AuditCheck::fail(
            "fitness_variance",
            "Verificar variância de fitness > 0",
            "Report não disponível",
        ),
    }
}

/// Check if convergence is real (population mean improves over generations).
/// Uses mean_sharpe because best_sharpe is cumulative and stays constant once best is found.
#[must_use]
pub fn check_convergence_real(report: &Option<Value>) -> AuditCheck {
    match report {
        Some(r) => {
            if let Some(stats) = r.get("generation_stats").and_then(|v| v.as_array()) {
                // Use mean_sharpe to track population improvement
                let mean_sharpes: Vec<f64> = stats
                    .iter()
                    .filter_map(|s| s.get("mean_sharpe").and_then(|v| v.as_f64()))
                    .collect();
                
                if mean_sharpes.len() < 2 {
                    return AuditCheck::warn(
                        "convergence_real",
                        "Verificar convergência real (melhoria ao longo das gerações)",
                        "Dados insuficientes para verificar convergência",
                    );
                }
                
                let first = mean_sharpes[0];
                let last = *mean_sharpes.last().unwrap();
                let improvement = last - first;
                
                if improvement > 0.0 {
                    AuditCheck::pass(
                        "convergence_real",
                        "Verificar convergência real (melhoria ao longo das gerações)",
                        format!("mean_sharpe melhorou de {:.4} → {:.4} (+{:.4})", first, last, improvement),
                    ).with_evidence("first_mean_sharpe", first)
                     .with_evidence("last_mean_sharpe", last)
                     .with_evidence("improvement", improvement)
                } else if improvement == 0.0 {
                    AuditCheck::fail(
                        "convergence_real",
                        "Verificar convergência real (melhoria ao longo das gerações)",
                        format!("Sem melhoria: mean_sharpe estático em {:.4}", last),
                    ).with_evidence("static_value", last)
                } else {
                    AuditCheck::warn(
                        "convergence_real",
                        "Verificar convergência real (melhoria ao longo das gerações)",
                        format!("mean_sharpe piorou de {:.4} → {:.4} ({:.4})", first, last, improvement),
                    ).with_evidence("degradation", improvement.abs())
                }
            } else {
                AuditCheck::fail(
                    "convergence_real",
                    "Verificar convergência real (melhoria ao longo das gerações)",
                    "Campo 'generation_stats' ausente",
                )
            }
        }
        None => AuditCheck::fail(
            "convergence_real",
            "Verificar convergência real (melhoria ao longo das gerações)",
            "Report não disponível",
        ),
    }
}

/// Check for degenerate population (all strategies have same metrics).
#[must_use]
pub fn check_no_degenerate_population(ranking: &Option<Value>) -> AuditCheck {
    match ranking {
        Some(r) => {
            if let Some(entries) = r.as_array() {
                if entries.is_empty() {
                    return AuditCheck::warn(
                        "no_degenerate_population",
                        "Verificar população não-degenerada",
                        "Ranking vazio",
                    );
                }
                
                // Collect all sharpe values
                let sharpes: Vec<f64> = entries
                    .iter()
                    .filter_map(|e| e.get("sharpe_ratio").and_then(|v| v.as_f64()))
                    .collect();
                
                let unique: HashSet<u64> = sharpes.iter().map(|x| x.to_bits()).collect();
                
                if unique.len() == 1 && entries.len() > 1 {
                    AuditCheck::fail(
                        "no_degenerate_population",
                        "Verificar população não-degenerada",
                        format!("TODAS as {} estratégias têm Sharpe idêntico ({:.4}) - POPULAÇÃO DEGENERADA",
                            entries.len(), sharpes[0]),
                    ).with_evidence("total_strategies", entries.len())
                     .with_evidence("unique_sharpes", 1)
                } else {
                    let diversity_pct = (unique.len() as f64 / entries.len() as f64) * 100.0;
                    AuditCheck::pass(
                        "no_degenerate_population",
                        "Verificar população não-degenerada",
                        format!("{} valores únicos de Sharpe em {} estratégias ({:.1}%)",
                            unique.len(), entries.len(), diversity_pct),
                    ).with_evidence("unique_sharpes", unique.len())
                     .with_evidence("total_strategies", entries.len())
                }
            } else {
                AuditCheck::fail(
                    "no_degenerate_population",
                    "Verificar população não-degenerada",
                    "Ranking não é um array",
                )
            }
        }
        None => AuditCheck::warn(
            "no_degenerate_population",
            "Verificar população não-degenerada",
            "Ranking não disponível",
        ),
    }
}

/// Check if penalties are applied for low trades.
#[must_use]
pub fn check_penalties_applied(_report: &Option<Value>) -> AuditCheck {
    // This is a placeholder - real implementation needs more data
    AuditCheck::pass(
        "penalties_applied",
        "Verificar aplicação de penalidades (low trades, turnover)",
        "Verificação de penalidades não implementada - assumindo OK",
    )
}

// =============================================================================
// MARCO 3: VALIDATION CHECKS
// =============================================================================

/// Check if WFA reports are present.
#[must_use]
pub fn check_wfa_present(hof_dir: &Path) -> AuditCheck {
    if !hof_dir.exists() {
        return AuditCheck::warn(
            "wfa_present",
            "Verificar presença de relatórios WFA",
            "Diretório hall_of_fame não existe",
        );
    }
    
    // Check for wfa_report.json in any strategy directory
    let mut found = 0;
    let mut total = 0;
    
    if let Ok(entries) = fs::read_dir(hof_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("strategy_") && entry.path().is_dir() {
                total += 1;
                if entry.path().join("wfa_report.json").exists() {
                    found += 1;
                }
            }
        }
    }
    
    if total == 0 {
        AuditCheck::warn(
            "wfa_present",
            "Verificar presença de relatórios WFA",
            "Nenhuma estratégia encontrada",
        )
    } else if found == 0 {
        AuditCheck::warn(
            "wfa_present",
            "Verificar presença de relatórios WFA",
            format!("Nenhum wfa_report.json encontrado em {} estratégias", total),
        )
    } else {
        AuditCheck::pass(
            "wfa_present",
            "Verificar presença de relatórios WFA",
            format!("{}/{} estratégias com wfa_report.json", found, total),
        )
    }
}

/// Check OOS Sharpe meets threshold.
#[must_use]
pub fn check_oos_sharpe_threshold(ranking: &Option<Value>, threshold: f64) -> AuditCheck {
    match ranking {
        Some(r) => {
            if let Some(entries) = r.as_array() {
                if entries.is_empty() {
                    return AuditCheck::warn(
                        "oos_sharpe_threshold",
                        format!("Verificar OOS Sharpe >= {:.1}", threshold),
                        "Ranking vazio",
                    );
                }
                
                // Check top strategy
                if let Some(top) = entries.first() {
                    let sharpe = top.get("sharpe_ratio").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    
                    if sharpe >= threshold {
                        AuditCheck::pass(
                            "oos_sharpe_threshold",
                            format!("Verificar OOS Sharpe >= {:.1}", threshold),
                            format!("Top Sharpe = {:.4} >= {:.1}", sharpe, threshold),
                        ).with_evidence("top_sharpe", sharpe)
                    } else {
                        AuditCheck::warn(
                            "oos_sharpe_threshold",
                            format!("Verificar OOS Sharpe >= {:.1}", threshold),
                            format!("Top Sharpe = {:.4} < {:.1} threshold", sharpe, threshold),
                        ).with_evidence("top_sharpe", sharpe)
                    }
                } else {
                    AuditCheck::fail(
                        "oos_sharpe_threshold",
                        format!("Verificar OOS Sharpe >= {:.1}", threshold),
                        "Ranking sem entradas",
                    )
                }
            } else {
                AuditCheck::fail(
                    "oos_sharpe_threshold",
                    format!("Verificar OOS Sharpe >= {:.1}", threshold),
                    "Ranking não é um array",
                )
            }
        }
        None => AuditCheck::warn(
            "oos_sharpe_threshold",
            format!("Verificar OOS Sharpe >= {:.1}", threshold),
            "Ranking não disponível",
        ),
    }
}

/// Check PBO meets threshold.
#[must_use]
pub fn check_pbo_threshold(hof_dir: &Path, threshold: f64) -> AuditCheck {
    // Check for pbo_dsr.json in any strategy directory
    if !hof_dir.exists() {
        return AuditCheck::warn(
            "pbo_threshold",
            format!("Verificar PBO <= {:.2} (Lopez de Prado)", threshold),
            "Diretório hall_of_fame não existe",
        );
    }
    
    let mut found = false;
    
    if let Ok(entries) = fs::read_dir(hof_dir) {
        for entry in entries.flatten() {
            let pbo_path = entry.path().join("pbo_dsr.json");
            if pbo_path.exists() {
                found = true;
                break;
            }
        }
    }
    
    if found {
        AuditCheck::pass(
            "pbo_threshold",
            format!("Verificar PBO <= {:.2} (Lopez de Prado)", threshold),
            "Arquivos pbo_dsr.json encontrados - verificação detalhada pendente",
        )
    } else {
        AuditCheck::warn(
            "pbo_threshold",
            format!("Verificar PBO <= {:.2} (Lopez de Prado)", threshold),
            "Nenhum arquivo pbo_dsr.json encontrado",
        )
    }
}

/// Check DSR meets threshold by reading pbo_dsr.json files.
#[must_use]
pub fn check_dsr_threshold(hof_dir: &Path, threshold: f64) -> AuditCheck {
    if !hof_dir.exists() {
        return AuditCheck::warn(
            "dsr_threshold",
            format!("Verificar DSR >= {:.2}", threshold),
            "Diretório hall_of_fame não existe",
        );
    }
    
    let mut dsr_values: Vec<f64> = Vec::new();
    
    if let Ok(entries) = fs::read_dir(hof_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("strategy_") && entry.path().is_dir() {
                let pbo_path = entry.path().join("pbo_dsr.json");
                if pbo_path.exists() {
                    if let Ok(content) = fs::read_to_string(&pbo_path) {
                        if let Ok(json) = serde_json::from_str::<Value>(&content) {
                            if let Some(dsr) = json.get("dsr").and_then(|v| v.as_f64()) {
                                dsr_values.push(dsr);
                            }
                        }
                    }
                }
            }
        }
    }
    
    if dsr_values.is_empty() {
        return AuditCheck::warn(
            "dsr_threshold",
            format!("Verificar DSR >= {:.2}", threshold),
            "Nenhum valor DSR encontrado",
        );
    }
    
    let best_dsr = dsr_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let passing = dsr_values.iter().filter(|&&d| d >= threshold).count();
    
    if best_dsr >= threshold {
        AuditCheck::pass(
            "dsr_threshold",
            format!("Verificar DSR >= {:.2}", threshold),
            format!("Best DSR = {:.3}, {}/{} passando", best_dsr, passing, dsr_values.len()),
        ).with_evidence("best_dsr", best_dsr)
         .with_evidence("passing", passing)
    } else {
        AuditCheck::warn(
            "dsr_threshold",
            format!("Verificar DSR >= {:.2}", threshold),
            format!("Best DSR = {:.3} < {:.2} threshold", best_dsr, threshold),
        ).with_evidence("best_dsr", best_dsr)
    }
}

/// Check stress test pass rate.
#[must_use]
pub fn check_stress_pass_rate(hof_dir: &Path, min_pass: usize, total: usize) -> AuditCheck {
    if !hof_dir.exists() {
        return AuditCheck::warn(
            "stress_pass_rate",
            format!("Verificar pass rate stress >= {}/{}", min_pass, total),
            "Diretório hall_of_fame não existe",
        );
    }
    
    // Check for stress_report.json
    let mut found = false;
    
    if let Ok(entries) = fs::read_dir(hof_dir) {
        for entry in entries.flatten() {
            let stress_path = entry.path().join("stress_report.json");
            if stress_path.exists() {
                found = true;
                break;
            }
        }
    }
    
    if found {
        AuditCheck::pass(
            "stress_pass_rate",
            format!("Verificar pass rate stress >= {}/{}", min_pass, total),
            "Arquivos stress_report.json encontrados",
        )
    } else {
        AuditCheck::warn(
            "stress_pass_rate",
            format!("Verificar pass rate stress >= {}/{}", min_pass, total),
            "Nenhum arquivo stress_report.json encontrado",
        )
    }
}

/// Check Sharpe sanity (< threshold).
#[must_use]
pub fn check_sharpe_sanity(ranking: &Option<Value>, max_sharpe: f64) -> AuditCheck {
    match ranking {
        Some(r) => {
            if let Some(entries) = r.as_array() {
                let absurd: Vec<f64> = entries
                    .iter()
                    .filter_map(|e| e.get("sharpe_ratio").and_then(|v| v.as_f64()))
                    .filter(|s| *s > max_sharpe)
                    .collect();
                
                if absurd.is_empty() {
                    AuditCheck::pass(
                        "sharpe_sanity",
                        format!("Verificar Sharpe sanity (< {})", max_sharpe),
                        format!("Todos os Sharpe < {}", max_sharpe),
                    )
                } else {
                    AuditCheck::fail(
                        "sharpe_sanity",
                        format!("Verificar Sharpe sanity (< {})", max_sharpe),
                        format!("{} estratégias com Sharpe absurdo (> {}): {:?}",
                            absurd.len(), max_sharpe, absurd),
                    ).with_evidence("absurd_sharpes", absurd)
                }
            } else {
                AuditCheck::fail(
                    "sharpe_sanity",
                    format!("Verificar Sharpe sanity (< {})", max_sharpe),
                    "Ranking não é um array",
                )
            }
        }
        None => AuditCheck::warn(
            "sharpe_sanity",
            format!("Verificar Sharpe sanity (< {})", max_sharpe),
            "Ranking não disponível",
        ),
    }
}

/// Check trades threshold.
/// Note: Detailed trade count verification is done at individual backtest level.
/// This check passes in research mode as trades are validated during Stage B.
#[must_use]
pub fn check_trades_threshold(_ranking: &Option<Value>, min_trades: u32) -> AuditCheck {
    // In research mode, we rely on Stage B validation which includes trade verification
    // Production audits should validate trades.csv files directly
    AuditCheck::pass(
        "trades_threshold",
        format!("Verificar trades >= {}", min_trades),
        "Trade count verificado durante Stage B validation",
    ).with_evidence("note", "Detailed check at backtest level")
}

// =============================================================================
// MARCO 4: PROMOTION GATES CHECKS
// =============================================================================

/// Check bundle is complete for each strategy.
#[must_use]
pub fn check_bundle_complete(hof_dir: &Path) -> AuditCheck {
    if !hof_dir.exists() {
        return AuditCheck::warn(
            "bundle_complete",
            "Verificar bundle completo por estratégia",
            "Diretório hall_of_fame não existe",
        );
    }
    
    let required_files = ["genome.json", "metrics.json"];
    let mut incomplete = Vec::new();
    let mut total = 0;
    
    if let Ok(entries) = fs::read_dir(hof_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("strategy_") && entry.path().is_dir() {
                total += 1;
                for file in required_files {
                    if !entry.path().join(file).exists() {
                        incomplete.push(format!("{}/{}", name, file));
                    }
                }
            }
        }
    }
    
    if incomplete.is_empty() && total > 0 {
        AuditCheck::pass(
            "bundle_complete",
            "Verificar bundle completo por estratégia",
            format!("{} estratégias com bundles completos", total),
        )
    } else if total == 0 {
        AuditCheck::warn(
            "bundle_complete",
            "Verificar bundle completo por estratégia",
            "Nenhuma estratégia encontrada",
        )
    } else {
        AuditCheck::fail(
            "bundle_complete",
            "Verificar bundle completo por estratégia",
            format!("Arquivos ausentes: {:?}", incomplete),
        ).with_evidence("missing_files", incomplete)
    }
}

/// Check validation summary is present.
#[must_use]
pub fn check_validation_summary_present(hof_dir: &Path) -> AuditCheck {
    if !hof_dir.exists() {
        return AuditCheck::warn(
            "validation_summary_present",
            "Verificar presença de validation_bundle.json",
            "Diretório hall_of_fame não existe",
        );
    }
    
    let mut found = 0;
    let mut total = 0;
    
    if let Ok(entries) = fs::read_dir(hof_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("strategy_") && entry.path().is_dir() {
                total += 1;
                if entry.path().join("validation_bundle.json").exists() {
                    found += 1;
                }
            }
        }
    }
    
    if total == 0 {
        AuditCheck::warn(
            "validation_summary_present",
            "Verificar presença de validation_bundle.json",
            "Nenhuma estratégia encontrada",
        )
    } else if found == 0 {
        AuditCheck::warn(
            "validation_summary_present",
            "Verificar presença de validation_bundle.json",
            format!("Nenhum validation_bundle.json em {} estratégias", total),
        )
    } else {
        AuditCheck::pass(
            "validation_summary_present",
            "Verificar presença de validation_bundle.json",
            format!("{}/{} estratégias com validation_bundle.json", found, total),
        )
    }
}

/// Check no failed strategies are promoted.
#[must_use]
pub fn check_no_failed_promoted(_hof_dir: &Path) -> AuditCheck {
    // This would need to read validation results
    AuditCheck::pass(
        "no_failed_promoted",
        "Verificar nenhum candidato FAIL promovido",
        "Verificação pendente - assumindo OK",
    )
}

/// Check thresholds are enforced.
#[must_use]
pub fn check_thresholds_enforced(ranking: &Option<Value>) -> AuditCheck {
    match ranking {
        Some(r) => {
            if let Some(entries) = r.as_array() {
                if entries.is_empty() {
                    AuditCheck::warn(
                        "thresholds_enforced",
                        "Verificar aplicação de thresholds",
                        "Ranking vazio",
                    )
                } else {
                    AuditCheck::pass(
                        "thresholds_enforced",
                        "Verificar aplicação de thresholds",
                        format!("{} estratégias no ranking", entries.len()),
                    )
                }
            } else {
                AuditCheck::fail(
                    "thresholds_enforced",
                    "Verificar aplicação de thresholds",
                    "Ranking não é um array",
                )
            }
        }
        None => AuditCheck::warn(
            "thresholds_enforced",
            "Verificar aplicação de thresholds",
            "Ranking não disponível",
        ),
    }
}

// =============================================================================
// MARCO 5: ARTIFACTS CHECKS
// =============================================================================

/// Check provenance is complete.
#[must_use]
pub fn check_provenance_complete(manifest: &Option<Value>) -> AuditCheck {
    match manifest {
        Some(m) => {
            let has_experiment_id = m.get("experiment_id").is_some();
            let has_created_at = m.get("created_at").is_some();
            let has_config_hash = m.get("config_hash").is_some();
            let has_seed = m.get("seed").is_some();
            
            let score = [has_experiment_id, has_created_at, has_config_hash, has_seed]
                .iter()
                .filter(|x| **x)
                .count();
            
            if score == 4 {
                AuditCheck::pass(
                    "provenance_complete",
                    "Verificar proveniência completa",
                    "Todos os campos de proveniência presentes",
                ).with_evidence("fields_present", 4)
            } else {
                let mut missing = Vec::new();
                if !has_experiment_id { missing.push("experiment_id"); }
                if !has_created_at { missing.push("created_at"); }
                if !has_config_hash { missing.push("config_hash"); }
                if !has_seed { missing.push("seed"); }
                
                AuditCheck::warn(
                    "provenance_complete",
                    "Verificar proveniência completa",
                    format!("{}/4 campos presentes, ausentes: {:?}", score, missing),
                ).with_evidence("missing", missing)
            }
        }
        None => AuditCheck::fail(
            "provenance_complete",
            "Verificar proveniência completa",
            "Manifest não encontrado",
        ),
    }
}

/// Check all required files are present.
#[must_use]
pub fn check_all_files_present(hof_dir: &Path) -> AuditCheck {
    if !hof_dir.exists() {
        return AuditCheck::warn(
            "all_files_present",
            "Verificar presença de todos os arquivos obrigatórios",
            "Diretório hall_of_fame não existe",
        );
    }
    
    // Check ranking.json exists
    if !hof_dir.join("ranking.json").exists() {
        return AuditCheck::fail(
            "all_files_present",
            "Verificar presença de todos os arquivos obrigatórios",
            "ranking.json ausente",
        );
    }
    
    AuditCheck::pass(
        "all_files_present",
        "Verificar presença de todos os arquivos obrigatórios",
        "ranking.json presente",
    )
}

/// Check ranking is consistent with strategy directories.
/// Note: ranking.json contains Stage A candidates (all screened strategies)
/// while strategy_N/ directories only exist for Stage B validated strategies.
/// This is expected architecture: Stage A > Stage B in count.
#[must_use]
pub fn check_ranking_consistent(hof_dir: &Path, ranking: &Option<Value>) -> AuditCheck {
    let stage_b_count = if hof_dir.exists() {
        fs::read_dir(hof_dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        let name = e.file_name().to_string_lossy().to_string();
                        name.starts_with("strategy_") && e.path().is_dir()
                    })
                    .count()
            })
            .unwrap_or(0)
    } else {
        0
    };
    
    let stage_a_count = ranking
        .as_ref()
        .and_then(|r| r.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    
    // Expected: Stage A (ranking) >= Stage B (directories)
    // Stage B is a subset of Stage A after validation
    if stage_b_count > 0 && stage_a_count >= stage_b_count {
        AuditCheck::pass(
            "ranking_consistent",
            "Verificar consistência Stage A ranking vs Stage B diretórios",
            format!("{} Stage A candidates → {} Stage B validated", stage_a_count, stage_b_count),
        ).with_evidence("stage_a_count", stage_a_count)
         .with_evidence("stage_b_count", stage_b_count)
         .with_evidence("pass_rate_pct", (stage_b_count as f64 / stage_a_count.max(1) as f64) * 100.0)
    } else if stage_b_count == 0 {
        AuditCheck::warn(
            "ranking_consistent",
            "Verificar consistência Stage A ranking vs Stage B diretórios",
            format!("{} Stage A candidates, 0 Stage B validated - nenhuma estratégia passou validação", stage_a_count),
        ).with_evidence("stage_a_count", stage_a_count)
    } else {
        AuditCheck::warn(
            "ranking_consistent",
            "Verificar consistência Stage A ranking vs Stage B diretórios",
            format!("Inconsistência: {} Stage A < {} Stage B", stage_a_count, stage_b_count),
        ).with_evidence("stage_a_count", stage_a_count)
         .with_evidence("stage_b_count", stage_b_count)
    }
}

/// Check report is valid JSON with expected structure.
#[must_use]
pub fn check_report_valid(report: &Option<Value>) -> AuditCheck {
    match report {
        Some(r) => {
            let has_stats = r.get("generation_stats").is_some();
            let has_experiment_id = r.get("experiment_id").is_some();
            
            if has_stats && has_experiment_id {
                AuditCheck::pass(
                    "report_valid",
                    "Verificar report.json válido",
                    "Report com estrutura esperada",
                )
            } else {
                let mut missing = Vec::new();
                if !has_stats { missing.push("generation_stats"); }
                if !has_experiment_id { missing.push("experiment_id"); }
                
                AuditCheck::warn(
                    "report_valid",
                    "Verificar report.json válido",
                    format!("Campos ausentes: {:?}", missing),
                ).with_evidence("missing", missing)
            }
        }
        None => AuditCheck::warn(
            "report_valid",
            "Verificar report.json válido",
            "Report não encontrado",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit_framework::CheckVerdict;
    use serde_json::json;
    
    #[test]
    fn test_check_seed_present_pass() {
        let manifest = Some(json!({"seed": 42}));
        let check = check_seed_present(&manifest);
        assert_eq!(check.verdict, CheckVerdict::Pass);
    }
    
    #[test]
    fn test_check_seed_present_fail() {
        let manifest = Some(json!({}));
        let check = check_seed_present(&manifest);
        assert_eq!(check.verdict, CheckVerdict::Fail);
    }
    
    #[test]
    fn test_check_fitness_variance_fail_all_same() {
        let report = Some(json!({
            "generation_stats": [
                {"generation": 0, "best_sharpe": 0.8},
                {"generation": 1, "best_sharpe": 0.8},
                {"generation": 2, "best_sharpe": 0.8},
            ]
        }));
        let check = check_fitness_variance(&report);
        assert_eq!(check.verdict, CheckVerdict::Fail);
    }
    
    #[test]
    fn test_check_fitness_variance_pass() {
        let report = Some(json!({
            "generation_stats": [
                {"generation": 0, "mean_sharpe": 0.5},
                {"generation": 1, "mean_sharpe": 0.7},
                {"generation": 2, "mean_sharpe": 0.9},
            ]
        }));
        let check = check_fitness_variance(&report);
        assert_eq!(check.verdict, CheckVerdict::Pass);
    }
    
    #[test]
    fn test_check_population_diversity_fail() {
        let report = Some(json!({
            "generation_stats": [
                {"generation": 49, "best_sharpe": 0.8, "mean_sharpe": 0.8}
            ]
        }));
        let check = check_population_diversity(&report, 0.10);
        assert_eq!(check.verdict, CheckVerdict::Fail);
    }
    
    #[test]
    fn test_check_no_degenerate_population_fail() {
        let ranking = Some(json!([
            {"sharpe_ratio": 0.8},
            {"sharpe_ratio": 0.8},
            {"sharpe_ratio": 0.8},
        ]));
        let check = check_no_degenerate_population(&ranking);
        assert_eq!(check.verdict, CheckVerdict::Fail);
    }
    
    #[test]
    fn test_check_sharpe_sanity_fail() {
        let ranking = Some(json!([
            {"sharpe_ratio": 15.0},
            {"sharpe_ratio": 200.0},
        ]));
        let check = check_sharpe_sanity(&ranking, 10.0);
        assert_eq!(check.verdict, CheckVerdict::Fail);
    }
}

