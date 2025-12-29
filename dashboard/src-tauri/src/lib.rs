//! Quant Dashboard Tauri Backend
//!
//! Provides commands for loading SCG artifacts, candidates, and backtest results.
//! Integrates with artifacts/site/*.json structure for institutional-grade UX.

use lru::LruCache;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::num::NonZeroUsize;
use std::path::PathBuf;

// =============================================================================
// ARTIFACT TYPES - Aligned with artifacts/site/*.json schema
// =============================================================================

/// Site index from artifacts/site/index.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteIndex {
    pub schema_version: String,
    pub generated_at: String,
    pub campaigns: Vec<CampaignSummary>,
}

/// Campaign summary from index.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignSummary {
    pub campaign_id: String,
    pub name: String,
    pub tag: String,
    pub status: String,
    pub runs_count: u32,
    pub created_at: String,
    pub detail_path: String,
}

/// Campaign detail from campaign_<id>.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignDetail {
    pub schema_version: String,
    pub campaign: CampaignInfo,
    pub runs: Vec<RunSummary>,
}

/// Campaign info section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignInfo {
    pub campaign_id: String,
    pub name: String,
    pub tag: String,
    pub owner: Option<String>,
    pub status: String,
    pub config_hash: Option<String>,
    pub git_sha: Option<String>,
    pub created_at: String,
    pub notes: Option<String>,
}

/// Run summary from campaign detail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub run_id: String,
    pub seed: u64,
    pub status: String,
    pub data_integrity_verdict: Option<String>,
    pub data_integrity_score: Option<f64>,
    pub candidates_count: Option<u32>,
    pub research_candidates_count: Option<u32>,
    pub validated_candidates_count: Option<u32>,
    pub best_oos_sharpe_net: Option<f64>,
    pub duration_secs: Option<u64>,
    pub detail_path: Option<String>,
    pub export_path: Option<String>,
}

/// Run detail from run_<id>.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunDetail {
    pub schema_version: String,
    pub run: RunInfo,
    pub config_snapshot: Option<serde_json::Value>,
    pub metrics: RunMetrics,
    pub top_candidates: Vec<TopCandidateEntry>,
    pub exports: RunExports,
}

/// Run info section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunInfo {
    pub run_id: String,
    pub campaign_id: String,
    pub seed: u64,
    pub status: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    pub duration_secs: Option<u64>,
    pub artifact_path: Option<String>,
}

/// Run metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetrics {
    pub total_evaluated: Option<u32>,
    pub research_candidates: Option<u32>,
    pub validated_candidates: Option<u32>,
    pub promoted_candidates: Option<u32>,
    pub best_oos_sharpe_net: Option<f64>,
    pub best_oos_cagr_net: Option<f64>,
    pub data_integrity_verdict: Option<String>,
    pub data_integrity_score: Option<f64>,
}

/// Top candidate entry from run detail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopCandidateEntry {
    pub rank: u32,
    pub candidate_id: String,
    pub candidate_class: String,
    pub display_name: Option<String>,
    pub oos_sharpe_net: Option<f64>,
    pub oos_cagr_net: Option<f64>,
    pub max_drawdown_net: Option<f64>,
    pub pbo: Option<f64>,
    pub dsr: Option<f64>,
    pub gates_passed: Option<bool>,
    pub stress_passed: Option<bool>,
    pub data_integrity_ok: Option<bool>,
}

/// Run exports section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunExports {
    pub top1000_json: Option<String>,
    pub top1000_csv: Option<String>,
    pub pareto_json: Option<String>,
}

/// Candidate list item for DataTable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateListItem {
    pub rank: u32,
    pub candidate_id: String,
    pub candidate_class: String,
    pub display_name: String,
    pub oos_sharpe_net: f64,
    pub oos_cagr_net: f64,
    pub max_drawdown_net: f64,
    pub pbo: f64,
    pub dsr: f64,
    pub gates_passed: bool,
    pub stress_passed: bool,
    pub data_integrity_ok: bool,
}

/// Strategy block info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineBlock {
    pub block_type: String,
    pub name: String,
    pub params: serde_json::Value,
}

/// Provenance info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub git_sha: Option<String>,
    pub dataset_hash: Option<String>,
    pub config_hash: Option<String>,
    pub run_id: Option<String>,
    pub campaign_id: Option<String>,
    pub seed: Option<u64>,
    pub created_at: Option<String>,
}

/// Execution config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    pub delay_bars: u32,
    pub bypass_for_debug: bool,
    pub slippage: SlippageConfig,
    pub fees: FeesConfig,
    pub fill_policy: Option<FillPolicyConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageConfig {
    pub slippage_type: String,
    pub bps: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeesConfig {
    pub tier: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillPolicyConfig {
    pub allow_partial: Option<bool>,
    pub max_participation: Option<f64>,
}

/// Full candidate detail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateDetailFull {
    pub candidate_id: String,
    pub display_name: String,
    pub candidate_class: String,
    
    // Strategy
    pub strategy_blocks: Vec<PipelineBlock>,
    pub strategy_toml: Option<String>,
    
    // Metrics
    pub oos_sharpe_net: Option<f64>,
    pub oos_cagr_net: Option<f64>,
    pub max_drawdown_net: Option<f64>,
    pub pbo: Option<f64>,
    pub dsr: Option<f64>,
    
    // Validation
    pub gates_passed: Option<bool>,
    pub stress_passed: Option<bool>,
    pub data_integrity_ok: Option<bool>,
    
    // Config
    pub execution_config: Option<ExecutionConfig>,
    
    // Provenance
    pub provenance: Option<Provenance>,
    
    // Paths
    pub bundle_path: Option<String>,
    pub strategy_toml_path: Option<String>,
    pub validation_summary_path: Option<String>,
}

/// Timeseries point from backtests/*/timeseries.csv
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeseriesPoint {
    pub date: String,
    pub equity: f64,
    pub drawdown: f64,
    pub exposure: Option<f64>,
    pub vol_exante: Option<f64>,
    pub vol_expost: Option<f64>,
}

/// Backtest result combining metadata, metrics, and timeseries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub available: bool,
    pub candidate_id: String,
    pub message: Option<String>,
    pub metadata: Option<BacktestMetadata>,
    pub metrics: Option<BacktestMetrics>,
    pub timeseries: Vec<TimeseriesPoint>,
    pub backtest_path: Option<String>,
}

/// Backtest metadata from metadata.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestMetadata {
    pub schema_version: Option<String>,
    pub run_id: String,
    pub config_hash: Option<String>,
    pub start_date: Option<String>,
    pub end_date: Option<String>,
}

/// Backtest metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestMetrics {
    pub total_return: Option<f64>,
    pub cagr: Option<f64>,
    pub sharpe: Option<f64>,
    pub sortino: Option<f64>,
    pub max_drawdown: Option<f64>,
    pub calmar: Option<f64>,
    pub volatility: Option<f64>,
    pub win_rate: Option<f64>,
    pub profit_factor: Option<f64>,
    pub total_trades: Option<u32>,
}

// =============================================================================
// LEGACY TYPES (for backward compatibility)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentListing {
    pub id: String,
    pub name: String,
    pub date: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScgReport {
    pub experiment_id: String,
    pub generations: Vec<GenerationData>,
    pub pareto_front: Vec<ParetoPoint>,
    pub best_candidates: Vec<CandidateInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationData {
    pub generation: u32,
    pub best_fitness: f64,
    pub avg_fitness: f64,
    pub population_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoPoint {
    pub id: String,
    pub sharpe: f64,
    pub cagr: f64,
    pub max_drawdown: f64,
    pub rank: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateInfo {
    pub id: String,
    pub strategy_type: String,
    pub params: serde_json::Value,
    pub metrics: CandidateMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateMetrics {
    pub sharpe: f64,
    pub cagr: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardOverview {
    pub total_experiments: u32,
    pub active_campaigns: u32,
    pub best_sharpe: f64,
    pub total_candidates: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityPoint {
    pub date: String,
    pub equity: f64,
    pub drawdown: f64,
}

// =============================================================================
// STATE MANAGEMENT
// =============================================================================

/// Artifact cache for efficient loading
pub struct ArtifactCache {
    pub artifacts_root: PathBuf,
    pub index: Option<SiteIndex>,
    pub campaigns: HashMap<String, CampaignDetail>,
    pub runs: HashMap<String, RunDetail>,
    pub candidates: LruCache<String, CandidateDetailFull>,
}

impl ArtifactCache {
    pub fn new(artifacts_root: PathBuf) -> Self {
        Self {
            artifacts_root,
            index: None,
            campaigns: HashMap::new(),
            runs: HashMap::new(),
            candidates: LruCache::new(NonZeroUsize::new(100).unwrap()),
        }
    }
}

/// Tauri state wrapper
pub struct ArtifactState {
    pub cache: Mutex<Option<ArtifactCache>>,
}

impl Default for ArtifactState {
    fn default() -> Self {
        Self {
            cache: Mutex::new(None),
        }
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Generate human-readable display name from strategy TOML
fn generate_display_name(strategy_toml: &str) -> String {
    let value: Result<toml::Value, _> = toml::from_str(strategy_toml);
    match value {
        Ok(v) => {
            let mut parts = Vec::new();
            
            // Selection block
            if let Some(sel) = v.get("selection") {
                if let Some(name) = sel.get("name").and_then(|n| n.as_str()) {
                    parts.push(format!("Sel:{}", name));
                }
            }
            
            // Entry block
            if let Some(entry) = v.get("entry") {
                if let Some(name) = entry.get("name").and_then(|n| n.as_str()) {
                    parts.push(format!("Entry:{}", name));
                }
            }
            
            // Exit block
            if let Some(exit) = v.get("exit") {
                if let Some(name) = exit.get("name").and_then(|n| n.as_str()) {
                    parts.push(format!("Exit:{}", name));
                }
            }
            
            if parts.is_empty() {
                "Unknown Strategy".to_string()
            } else {
                parts.join(" | ")
            }
        }
        Err(_) => "Unknown Strategy".to_string(),
    }
}

/// Parse strategy TOML into pipeline blocks
fn parse_strategy_blocks(strategy_toml: &str) -> Vec<PipelineBlock> {
    let value: Result<toml::Value, _> = toml::from_str(strategy_toml);
    let mut blocks = Vec::new();
    
    if let Ok(v) = value {
        for block_type in &["selection", "entry", "exit", "sizing", "risk"] {
            if let Some(block) = v.get(*block_type) {
                let name = block.get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("Unknown")
                    .to_string();
                    
                blocks.push(PipelineBlock {
                    block_type: block_type.to_string(),
                    name,
                    params: toml_to_json(block),
                });
            }
        }
    }
    
    blocks
}

/// Parse execution config TOML
fn parse_execution_config(content: &str) -> Result<ExecutionConfig, String> {
    let value: toml::Value = toml::from_str(content)
        .map_err(|e| format!("Failed to parse execution_config.toml: {}", e))?;

    let exec = value.get("execution").ok_or("Missing [execution] section")?;

    let delay_bars = exec
        .get("delay_bars")
        .and_then(|v| v.as_integer())
        .unwrap_or(1) as u32;
    let bypass_for_debug = exec
        .get("bypass_for_debug")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Parse slippage
    let slippage_section = exec.get("slippage");
    let slippage = SlippageConfig {
        slippage_type: slippage_section
            .and_then(|s| s.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("Constant")
            .to_string(),
        bps: slippage_section
            .and_then(|s| s.get("bps"))
            .and_then(|v| v.as_float()),
    };

    // Parse fees
    let fees_section = exec.get("fees");
    let fees = FeesConfig {
        tier: fees_section
            .and_then(|f| f.get("tier"))
            .and_then(|v| v.as_str())
            .unwrap_or("B3Retail")
            .to_string(),
    };

    // Parse fill policy
    let fill_policy = exec.get("fill_policy").map(|fp| FillPolicyConfig {
        allow_partial: fp.get("allow_partial").and_then(|v| v.as_bool()),
        max_participation: fp.get("max_participation").and_then(|v| v.as_float()),
    });

    Ok(ExecutionConfig {
        delay_bars,
        bypass_for_debug,
        slippage,
        fees,
        fill_policy,
    })
}

/// Convert TOML value to JSON value
fn toml_to_json(v: &toml::Value) -> serde_json::Value {
    match v {
        toml::Value::String(s) => serde_json::Value::String(s.clone()),
        toml::Value::Integer(i) => serde_json::Value::Number((*i).into()),
        toml::Value::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        toml::Value::Boolean(b) => serde_json::Value::Bool(*b),
        toml::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(toml_to_json).collect())
        }
        toml::Value::Table(tbl) => {
            let map: serde_json::Map<String, serde_json::Value> =
                tbl.iter().map(|(k, v)| (k.clone(), toml_to_json(v))).collect();
            serde_json::Value::Object(map)
        }
        toml::Value::Datetime(dt) => serde_json::Value::String(dt.to_string()),
    }
}

// =============================================================================
// TAURI COMMANDS - ARTIFACT INDEXER
// =============================================================================

/// Set the artifacts root path and initialize cache
#[tauri::command]
fn set_artifacts_root(
    path: String,
    state: tauri::State<'_, ArtifactState>,
) -> Result<String, String> {
    let root = PathBuf::from(&path);

    // Check if artifacts/ subdirectory exists
    let artifacts_path = if root.join("artifacts").exists() {
        root.join("artifacts")
    } else if root.ends_with("artifacts") {
        root.clone()
    } else {
        return Err(format!(
            "No artifacts directory found at {} or {}/artifacts",
            path, path
        ));
    };

    // Verify index.json exists
    let index_path = artifacts_path.join("site").join("index.json");
    if !index_path.exists() {
        return Err(format!(
            "No index.json found at {}",
            index_path.display()
        ));
    }

    let mut cache_guard = state.cache.lock();
    *cache_guard = Some(ArtifactCache::new(artifacts_path.clone()));

    Ok(artifacts_path.to_string_lossy().to_string())
}

/// Load the site index
#[tauri::command]
fn load_index(state: tauri::State<'_, ArtifactState>) -> Result<SiteIndex, String> {
    let mut cache_guard = state.cache.lock();
    let cache = cache_guard
        .as_mut()
        .ok_or("Artifacts root not set. Call set_artifacts_root first.")?;

    // Return cached if available
    if let Some(ref index) = cache.index {
        return Ok(index.clone());
    }

    // Load from file
    let index_path = cache.artifacts_root.join("site").join("index.json");
    let content = fs::read_to_string(&index_path)
        .map_err(|e| format!("Failed to read index.json: {}", e))?;

    let index: SiteIndex = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse index.json: {}", e))?;

    cache.index = Some(index.clone());
    Ok(index)
}

/// Load campaign detail
#[tauri::command]
fn load_campaign(
    campaign_id: String,
    state: tauri::State<'_, ArtifactState>,
) -> Result<CampaignDetail, String> {
    let mut cache_guard = state.cache.lock();
    let cache = cache_guard
        .as_mut()
        .ok_or("Artifacts root not set. Call set_artifacts_root first.")?;

    // Return cached if available
    if let Some(campaign) = cache.campaigns.get(&campaign_id) {
        return Ok(campaign.clone());
    }

    // Load from file
    let campaign_path = cache
        .artifacts_root
        .join("site")
        .join(format!("campaign_{}.json", campaign_id));

    let content = fs::read_to_string(&campaign_path)
        .map_err(|e| format!("Failed to read campaign file: {}", e))?;

    let campaign: CampaignDetail = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse campaign file: {}", e))?;

    cache.campaigns.insert(campaign_id, campaign.clone());
    Ok(campaign)
}

/// Load run detail
#[tauri::command]
fn load_run(
    run_id: String,
    state: tauri::State<'_, ArtifactState>,
) -> Result<RunDetail, String> {
    let mut cache_guard = state.cache.lock();
    let cache = cache_guard
        .as_mut()
        .ok_or("Artifacts root not set. Call set_artifacts_root first.")?;

    // Return cached if available
    if let Some(run) = cache.runs.get(&run_id) {
        return Ok(run.clone());
    }

    // Load from file
    let run_path = cache
        .artifacts_root
        .join("site")
        .join(format!("run_{}.json", run_id));

    let content = fs::read_to_string(&run_path)
        .map_err(|e| format!("Failed to read run file: {}", e))?;

    let run: RunDetail = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse run file: {}", e))?;

    cache.runs.insert(run_id, run.clone());
    Ok(run)
}

/// List candidates with optional filters
#[tauri::command]
fn list_candidates_v2(
    run_id: String,
    search: Option<String>,
    candidate_class: Option<String>,
    max_pbo: Option<f64>,
    limit: Option<usize>,
    state: tauri::State<'_, ArtifactState>,
) -> Result<Vec<CandidateListItem>, String> {
    let cache_guard = state.cache.lock();
    let cache = cache_guard
        .as_ref()
        .ok_or("Artifacts root not set. Call set_artifacts_root first.")?;

    // Try loading from top1000.csv first
    let csv_path = cache
        .artifacts_root
        .join("top_candidates")
        .join(&run_id)
        .join("top1000.csv");

    let mut candidates: Vec<CandidateListItem> = Vec::new();

    if csv_path.exists() {
        let mut rdr = csv::Reader::from_path(&csv_path)
            .map_err(|e| format!("Failed to read top1000.csv: {}", e))?;

        for (i, result) in rdr.records().enumerate() {
            if let Ok(record) = result {
                // Parse CSV fields
                let candidate_id = record.get(0).unwrap_or("").to_string();
                let candidate_class = record.get(1).unwrap_or("research").to_string();
                let oos_sharpe = record.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let oos_cagr = record.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let max_dd = record.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let pbo = record.get(5).and_then(|s| s.parse().ok()).unwrap_or(1.0);
                let dsr = record.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let gates = record.get(7).map(|s| s == "true").unwrap_or(false);
                let stress = record.get(8).map(|s| s == "true").unwrap_or(false);
                let integrity = record.get(9).map(|s| s == "true").unwrap_or(true);

                // Generate display name
                let display_name = format!(
                    "Strategy #{} (Sharpe: {:.2}, PBO: {:.2})",
                    i + 1,
                    oos_sharpe,
                    pbo
                );

                candidates.push(CandidateListItem {
                    rank: (i + 1) as u32,
                    candidate_id,
                    candidate_class,
                    display_name,
                    oos_sharpe_net: oos_sharpe,
                    oos_cagr_net: oos_cagr,
                    max_drawdown_net: max_dd,
                    pbo,
                    dsr,
                    gates_passed: gates,
                    stress_passed: stress,
                    data_integrity_ok: integrity,
                });
            }
        }
    }

    // Apply filters
    if let Some(ref search_query) = search {
        let query = search_query.to_lowercase();
        candidates.retain(|c| {
            c.display_name.to_lowercase().contains(&query)
                || c.candidate_id.to_lowercase().contains(&query)
        });
    }

    if let Some(ref class) = candidate_class {
        if !class.is_empty() {
            candidates.retain(|c| c.candidate_class == *class);
        }
    }

    if let Some(max) = max_pbo {
        candidates.retain(|c| c.pbo <= max);
    }

    // Apply limit
    if let Some(lim) = limit {
        candidates.truncate(lim);
    }

    Ok(candidates)
}

/// Load full candidate detail
#[tauri::command]
fn load_candidate_detail(
    candidate_id: String,
    state: tauri::State<'_, ArtifactState>,
) -> Result<CandidateDetailFull, String> {
    let mut cache_guard = state.cache.lock();
    let cache = cache_guard
        .as_mut()
        .ok_or("Artifacts root not set. Call set_artifacts_root first.")?;

    // Check cache
    if let Some(detail) = cache.candidates.get(&candidate_id) {
        return Ok(detail.clone());
    }

    // Load from bundle directory
    let bundle_path = cache.artifacts_root.join("candidates").join(&candidate_id);

    if !bundle_path.exists() {
        return Err(format!("Candidate bundle not found: {}", candidate_id));
    }

    // Load strategy.toml
    let strategy_path = bundle_path.join("strategy.toml");
    let strategy_toml = if strategy_path.exists() {
        fs::read_to_string(&strategy_path).ok()
    } else {
        None
    };

    let display_name = strategy_toml
        .as_ref()
        .map(|s| generate_display_name(s))
        .unwrap_or_else(|| format!("Candidate {}", candidate_id));

    let strategy_blocks = strategy_toml
        .as_ref()
        .map(|s| parse_strategy_blocks(s))
        .unwrap_or_default();

    // Load provenance
    let prov_path = bundle_path.join("provenance.json");
    let provenance: Option<Provenance> = if prov_path.exists() {
        fs::read_to_string(&prov_path)
            .ok()
            .and_then(|c| serde_json::from_str(&c).ok())
    } else {
        None
    };

    // Load execution config
    let exec_path = bundle_path.join("execution_config.toml");
    let execution_config: Option<ExecutionConfig> = if exec_path.exists() {
        fs::read_to_string(&exec_path)
            .ok()
            .and_then(|c| parse_execution_config(&c).ok())
    } else {
        None
    };

    // Load validation summary for metrics
    let val_path = bundle_path.join("validation_summary.json");
    let validation: Option<serde_json::Value> = if val_path.exists() {
        fs::read_to_string(&val_path)
            .ok()
            .and_then(|c| serde_json::from_str(&c).ok())
    } else {
        None
    };

    let detail = CandidateDetailFull {
        candidate_id: candidate_id.clone(),
        display_name,
        candidate_class: "research".to_string(),
        strategy_blocks,
        strategy_toml,
        oos_sharpe_net: validation.as_ref().and_then(|v| v["oos_sharpe_net"].as_f64()),
        oos_cagr_net: validation.as_ref().and_then(|v| v["oos_cagr_net"].as_f64()),
        max_drawdown_net: validation.as_ref().and_then(|v| v["max_drawdown_net"].as_f64()),
        pbo: validation.as_ref().and_then(|v| v["pbo"].as_f64()),
        dsr: validation.as_ref().and_then(|v| v["dsr"].as_f64()),
        gates_passed: validation.as_ref().and_then(|v| v["gates_passed"].as_bool()),
        stress_passed: validation.as_ref().and_then(|v| v["stress_passed"].as_bool()),
        data_integrity_ok: validation.as_ref().and_then(|v| v["data_integrity_ok"].as_bool()),
        execution_config,
        provenance,
        bundle_path: Some(bundle_path.to_string_lossy().to_string()),
        strategy_toml_path: if strategy_path.exists() {
            Some(strategy_path.to_string_lossy().to_string())
        } else {
            None
        },
        validation_summary_path: if val_path.exists() {
            Some(val_path.to_string_lossy().to_string())
        } else {
            None
        },
    };

    cache.candidates.put(candidate_id, detail.clone());
    Ok(detail)
}

/// Load backtest timeseries for a candidate
#[tauri::command]
fn load_backtest_series(
    candidate_id: String,
    state: tauri::State<'_, ArtifactState>,
) -> Result<BacktestResult, String> {
    let cache_guard = state.cache.lock();
    let cache = cache_guard
        .as_ref()
        .ok_or("Artifacts root not set. Call set_artifacts_root first.")?;

    // Look for backtest in multiple possible locations
    let possible_paths = vec![
        cache.artifacts_root.join("backtests").join(&candidate_id),
        cache.artifacts_root.parent().unwrap_or(&cache.artifacts_root).join("output").join("backtests").join(&candidate_id),
    ];

    for backtest_path in possible_paths {
        let ts_path = backtest_path.join("timeseries.csv");
        if ts_path.exists() {
            let mut timeseries = Vec::new();
            let mut rdr = csv::Reader::from_path(&ts_path)
                .map_err(|e| format!("Failed to read timeseries.csv: {}", e))?;

            for result in rdr.records() {
                if let Ok(record) = result {
                    timeseries.push(TimeseriesPoint {
                        date: record.get(0).unwrap_or("").to_string(),
                        equity: record.get(1).and_then(|s| s.parse().ok()).unwrap_or(1.0),
                        drawdown: record.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                        exposure: record.get(3).and_then(|s| s.parse().ok()),
                        vol_exante: record.get(4).and_then(|s| s.parse().ok()),
                        vol_expost: record.get(5).and_then(|s| s.parse().ok()),
                    });
                }
            }

            // Load metadata if exists
            let meta_path = backtest_path.join("metadata.json");
            let metadata: Option<BacktestMetadata> = if meta_path.exists() {
                fs::read_to_string(&meta_path)
                    .ok()
                    .and_then(|c| serde_json::from_str(&c).ok())
            } else {
                None
            };

            return Ok(BacktestResult {
                available: true,
                candidate_id,
                message: None,
                metadata,
                metrics: None,
                timeseries,
                backtest_path: Some(backtest_path.to_string_lossy().to_string()),
            });
        }
    }

    // No backtest found
    Ok(BacktestResult {
        available: false,
        candidate_id,
        message: Some("No backtest data found. Run replay to generate.".to_string()),
        metadata: None,
        metrics: None,
        timeseries: Vec::new(),
        backtest_path: None,
    })
}

/// Invalidate cache
#[tauri::command]
fn invalidate_cache(state: tauri::State<'_, ArtifactState>) -> Result<(), String> {
    let mut cache_guard = state.cache.lock();
    if let Some(ref mut cache) = *cache_guard {
        cache.index = None;
        cache.campaigns.clear();
        cache.runs.clear();
        cache.candidates.clear();
    }
    Ok(())
}

/// Get artifacts root
#[tauri::command]
fn get_artifacts_root(state: tauri::State<'_, ArtifactState>) -> Result<Option<String>, String> {
    let cache_guard = state.cache.lock();
    Ok(cache_guard
        .as_ref()
        .map(|c| c.artifacts_root.to_string_lossy().to_string()))
}

// =============================================================================
// FILE WATCHER
// =============================================================================

use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};
use std::sync::mpsc::channel;
use std::time::Duration;
use tauri::Emitter;

/// Watch artifacts directory for changes
#[tauri::command]
fn watch_artifacts(
    app: tauri::AppHandle,
    state: tauri::State<'_, ArtifactState>,
) -> Result<(), String> {
    let cache_guard = state.cache.lock();
    let artifacts_root = cache_guard
        .as_ref()
        .ok_or("Artifacts root not set")?
        .artifacts_root
        .clone();
    drop(cache_guard);

    std::thread::spawn(move || {
        let (tx, rx) = channel();
        let config = Config::default().with_poll_interval(Duration::from_secs(2));

        let mut watcher: RecommendedWatcher = match Watcher::new(tx, config) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("Failed to create watcher: {}", e);
                return;
            }
        };

        if let Err(e) = watcher.watch(&artifacts_root, RecursiveMode::Recursive) {
            eprintln!("Failed to watch directory: {}", e);
            return;
        }

        loop {
            match rx.recv() {
                Ok(_event) => {
                    // Emit event to frontend
                    let _ = app.emit("artifacts-changed", ());
                }
                Err(e) => {
                    eprintln!("Watch error: {}", e);
                    break;
                }
            }
        }
    });

    Ok(())
}

// =============================================================================
// LEGACY COMMANDS
// =============================================================================

#[tauri::command]
fn list_experiments() -> Result<Vec<ExperimentListing>, String> {
    // Mock data for legacy compatibility
    Ok(vec![
        ExperimentListing {
            id: "exp_001".to_string(),
            name: "IBOV Momentum".to_string(),
            date: "2024-01-15".to_string(),
            status: "completed".to_string(),
        },
    ])
}

#[tauri::command]
fn load_scg_report(experiment_id: String) -> Result<ScgReport, String> {
    Ok(ScgReport {
        experiment_id,
        generations: vec![],
        pareto_front: vec![],
        best_candidates: vec![],
    })
}

#[tauri::command]
fn get_dashboard_overview() -> Result<DashboardOverview, String> {
    Ok(DashboardOverview {
        total_experiments: 5,
        active_campaigns: 2,
        best_sharpe: 1.85,
        total_candidates: 1250,
    })
}

// =============================================================================
// TAURI APP ENTRY POINT
// =============================================================================

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::new().build())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(ArtifactState::default())
        .invoke_handler(tauri::generate_handler![
            // Artifact indexer commands
            set_artifacts_root,
            load_index,
            load_campaign,
            load_run,
            list_candidates_v2,
            load_candidate_detail,
            load_backtest_series,
            invalidate_cache,
            get_artifacts_root,
            watch_artifacts,
            // Legacy commands
            list_experiments,
            load_scg_report,
            get_dashboard_overview,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
