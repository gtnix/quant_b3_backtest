//! Campaign Configuration - Parser and hashing for reproducibility.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

// =============================================================================
// CAMPAIGN CONFIG
// =============================================================================

/// Complete campaign configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignConfig {
    /// Campaign metadata.
    pub campaign: CampaignMeta,
    /// Dataset configuration.
    #[serde(default)]
    pub dataset: DatasetConfig,
    /// Evolution configuration.
    #[serde(default)]
    pub evolution: EvolutionRef,
    /// Execution configuration.
    #[serde(default)]
    pub execution: ExecutionRef,
    /// Seed policy.
    #[serde(default)]
    pub seeds: SeedPolicy,
    /// Budget constraints.
    #[serde(default)]
    pub budget: BudgetConfig,
    /// Promotion thresholds.
    #[serde(default)]
    pub promotion: PromotionConfig,
    /// Data integrity configuration.
    #[serde(default)]
    pub data_integrity: DataIntegrityConfig,
}

/// Campaign metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignMeta {
    /// Campaign name (human-readable).
    pub name: String,
    /// Optional tag for filtering.
    #[serde(default)]
    pub tag: Option<String>,
    /// Owner/team name.
    #[serde(default)]
    pub owner: Option<String>,
    /// Notes/description.
    #[serde(default)]
    pub notes: Option<String>,
}

/// Dataset configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Market (BR, US).
    #[serde(default = "default_market")]
    pub market: String,
    /// Start date (YYYY-MM-DD).
    #[serde(default)]
    pub start_date: Option<String>,
    /// End date (YYYY-MM-DD).
    #[serde(default)]
    pub end_date: Option<String>,
    /// Universe identifier or path.
    #[serde(default)]
    pub universe: Option<String>,
    /// Path to data files (for hash computation).
    #[serde(default)]
    pub data_path: Option<String>,
}

fn default_market() -> String {
    "BR".to_string()
}

/// Evolution configuration reference.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvolutionRef {
    /// Path to base SCG config file.
    #[serde(default)]
    pub base_config: Option<String>,
    /// Override population size.
    #[serde(default)]
    pub population_size: Option<usize>,
    /// Override max generations.
    #[serde(default)]
    pub max_generations: Option<u32>,
    /// Override convergence generations.
    #[serde(default)]
    pub convergence_generations: Option<u32>,
}

/// Execution configuration reference.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionRef {
    /// Path to execution config file.
    #[serde(default)]
    pub config_path: Option<String>,
    /// Override slippage BPS.
    #[serde(default)]
    pub slippage_bps: Option<f64>,
    /// Override delay bars.
    #[serde(default)]
    pub delay_bars: Option<u8>,
}

/// Seed policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedPolicy {
    /// Number of seeds to run.
    #[serde(default = "default_seed_count")]
    pub count: usize,
    /// Base seed (seeds will be base, base+1, base+2, ...).
    #[serde(default = "default_base_seed")]
    pub base_seed: u64,
}

fn default_seed_count() -> usize {
    3
}

fn default_base_seed() -> u64 {
    42
}

impl Default for SeedPolicy {
    fn default() -> Self {
        Self {
            count: default_seed_count(),
            base_seed: default_base_seed(),
        }
    }
}

impl SeedPolicy {
    /// Generate the list of seeds.
    pub fn generate_seeds(&self) -> Vec<i32> {
        (0..self.count)
            .map(|i| (self.base_seed + i as u64) as i32)
            .collect()
    }
}

/// Budget constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// Maximum number of runs (same as seed count unless resumed).
    #[serde(default = "default_max_runs")]
    pub max_runs: usize,
    /// Top K candidates to validate per run.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Timeout per run in seconds.
    #[serde(default = "default_timeout")]
    pub timeout_per_run_secs: u64,
    /// Enable stress testing.
    #[serde(default = "default_stress_enabled")]
    pub stress_enabled: bool,
}

fn default_max_runs() -> usize {
    5
}

fn default_top_k() -> usize {
    10
}

fn default_timeout() -> u64 {
    3600
}

fn default_stress_enabled() -> bool {
    true
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            max_runs: default_max_runs(),
            top_k: default_top_k(),
            timeout_per_run_secs: default_timeout(),
            stress_enabled: default_stress_enabled(),
        }
    }
}

/// Promotion thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionConfig {
    /// Minimum OOS Sharpe (net of costs).
    #[serde(default = "default_min_sharpe")]
    pub min_oos_sharpe_net: f64,
    /// Maximum PBO.
    #[serde(default = "default_max_pbo")]
    pub max_pbo: f64,
    /// Minimum stress scenarios passed.
    #[serde(default = "default_min_stress")]
    pub min_stress_passed: usize,
    /// Require all gates to pass.
    #[serde(default = "default_gates_required")]
    pub gates_required: bool,
    /// Minimum DSR (optional).
    #[serde(default)]
    pub min_dsr: Option<f64>,
}

fn default_min_sharpe() -> f64 {
    0.5
}

fn default_max_pbo() -> f64 {
    0.15
}

fn default_min_stress() -> usize {
    4
}

fn default_gates_required() -> bool {
    true
}

impl Default for PromotionConfig {
    fn default() -> Self {
        Self {
            min_oos_sharpe_net: default_min_sharpe(),
            max_pbo: default_max_pbo(),
            min_stress_passed: default_min_stress(),
            gates_required: default_gates_required(),
            min_dsr: None,
        }
    }
}

/// Data integrity configuration for anti-lookahead and dataset validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataIntegrityConfig {
    /// Audit mode: "fast" (sampling) or "strict" (full scan).
    #[serde(default = "default_audit_mode")]
    pub mode: String,
    
    /// Maximum allowed gap in days without explanation.
    #[serde(default = "default_max_gap_days")]
    pub max_gap_days: u32,
    
    /// Threshold for detecting suspicious price jumps (percent).
    #[serde(default = "default_jump_threshold")]
    pub jump_threshold_pct: f64,
    
    /// Price adjustment type: "raw", "adjusted", or "total_return".
    #[serde(default = "default_price_adjustment")]
    pub price_adjustment: String,
    
    /// Universe type: "point_in_time", "static", or "unknown".
    #[serde(default = "default_universe_type")]
    pub universe_type: String,
    
    /// Enable data integrity check (default: true).
    #[serde(default = "default_integrity_enabled")]
    pub enabled: bool,
}

fn default_audit_mode() -> String {
    "fast".to_string()
}

fn default_max_gap_days() -> u32 {
    5
}

fn default_jump_threshold() -> f64 {
    30.0
}

fn default_price_adjustment() -> String {
    "adjusted".to_string()
}

fn default_universe_type() -> String {
    "unknown".to_string()
}

fn default_integrity_enabled() -> bool {
    true
}

impl Default for DataIntegrityConfig {
    fn default() -> Self {
        Self {
            mode: default_audit_mode(),
            max_gap_days: default_max_gap_days(),
            jump_threshold_pct: default_jump_threshold(),
            price_adjustment: default_price_adjustment(),
            universe_type: default_universe_type(),
            enabled: default_integrity_enabled(),
        }
    }
}

// =============================================================================
// LOADING AND HASHING
// =============================================================================

impl CampaignConfig {
    /// Load a campaign config from a TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read campaign config: {}", path.display()))?;

        let config: Self = toml::from_str(&content)
            .with_context(|| format!("Failed to parse campaign config: {}", path.display()))?;

        Ok(config)
    }

    /// Compute a deterministic hash of the configuration for reproducibility.
    pub fn config_hash(&self) -> String {
        let mut hasher = Sha256::new();

        // Include key config values that affect results
        hasher.update(self.campaign.name.as_bytes());
        hasher.update(self.dataset.market.as_bytes());
        if let Some(ref start) = self.dataset.start_date {
            hasher.update(start.as_bytes());
        }
        if let Some(ref end) = self.dataset.end_date {
            hasher.update(end.as_bytes());
        }
        if let Some(ref universe) = self.dataset.universe {
            hasher.update(universe.as_bytes());
        }
        if let Some(ref base) = self.evolution.base_config {
            hasher.update(base.as_bytes());
        }
        if let Some(pop) = self.evolution.population_size {
            hasher.update(&pop.to_le_bytes());
        }
        if let Some(gen) = self.evolution.max_generations {
            hasher.update(&gen.to_le_bytes());
        }
        if let Some(ref exec) = self.execution.config_path {
            hasher.update(exec.as_bytes());
        }

        let hash = hasher.finalize();
        format!("sha256:{}", hex::encode(&hash[..8]))
    }

    /// Compute a hash of the dataset for reproducibility.
    pub fn dataset_hash(&self) -> Option<String> {
        let data_path = self.dataset.data_path.as_ref()?;
        let path = Path::new(data_path);

        if !path.exists() {
            return None;
        }

        let mut hasher = Sha256::new();

        // For directories, hash the file listing
        if path.is_dir() {
            if let Ok(entries) = fs::read_dir(path) {
                let mut files: Vec<String> = entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .collect();
                files.sort();
                for file in files {
                    hasher.update(file.as_bytes());
                }
            }
        } else {
            // For single files, hash the content
            if let Ok(content) = fs::read(path) {
                hasher.update(&content);
            }
        }

        let hash = hasher.finalize();
        Some(format!("sha256:{}", hex::encode(&hash[..8])))
    }

    /// Get git branch name.
    pub fn git_branch() -> Option<String> {
        std::process::Command::new("git")
            .args(["rev-parse", "--abbrev-ref", "HEAD"])
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
                } else {
                    None
                }
            })
    }

    /// Get git SHA.
    pub fn git_sha() -> Option<String> {
        std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
                } else {
                    None
                }
            })
    }
}

/// Generate an example campaign config TOML.
pub fn generate_example_config(name: &str) -> String {
    format!(
        r#"# Strategy Factory Campaign Configuration
# Generated for: {name}

[campaign]
name = "{name}"
tag = "exploration"
owner = "quant_team"
notes = "Auto-generated campaign configuration"

[dataset]
market = "BR"
start_date = "2018-01-01"
end_date = "2024-12-01"
universe = "ibov"
# data_path = "data/ohlcv"  # Optional: for dataset hash

[evolution]
# base_config = "configs/optimization/scg_base.toml"
population_size = 100
max_generations = 50

[execution]
config_path = "configs/execution_institutional.toml"
# slippage_bps = 10.0  # Optional override
# delay_bars = 1       # Optional override

[seeds]
count = 3
base_seed = 42

[budget]
max_runs = 3
top_k = 10
timeout_per_run_secs = 3600
stress_enabled = true

[promotion]
min_oos_sharpe_net = 0.5
max_pbo = 0.15
min_stress_passed = 4
gates_required = true

[data_integrity]
mode = "fast"
max_gap_days = 5
jump_threshold_pct = 30.0
price_adjustment = "adjusted"
enabled = true
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_hash_deterministic() {
        let config1 = CampaignConfig {
            campaign: CampaignMeta {
                name: "test".to_string(),
                tag: None,
                owner: None,
                notes: None,
            },
            dataset: DatasetConfig::default(),
            evolution: EvolutionRef::default(),
            execution: ExecutionRef::default(),
            seeds: SeedPolicy::default(),
            budget: BudgetConfig::default(),
            promotion: PromotionConfig::default(),
            data_integrity: DataIntegrityConfig::default(),
        };

        let config2 = config1.clone();

        assert_eq!(config1.config_hash(), config2.config_hash());
    }

    #[test]
    fn test_generate_seeds() {
        let policy = SeedPolicy {
            count: 5,
            base_seed: 100,
        };

        let seeds = policy.generate_seeds();
        assert_eq!(seeds, vec![100, 101, 102, 103, 104]);
    }
}

