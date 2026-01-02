//! Candidate bundle generation - Create production-ready artifacts.

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use super::registry::Candidate;

/// Provenance information for reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub candidate_id: String,
    pub genome_hash: String,
    pub run_id: String,
    pub campaign_id: String,
    pub seed: i64,
    pub git_sha: Option<String>,
    pub git_branch: Option<String>,
    pub config_hash: String,
    pub dataset_hash: Option<String>,
    pub created_at: String,
    pub scg_version: String,
    pub original_report_path: Option<String>,
}

/// Validation summary for the bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub oos_sharpe_net: Option<f32>,
    pub oos_sharpe_gross: Option<f32>,
    pub pbo: Option<f32>,
    pub dsr: Option<f32>,
    pub stress_passed: Option<i32>,
    pub stress_total: Option<i32>,
    pub gates_passed: Option<bool>,
    pub turnover_annual: Option<f32>,
    pub capacity_usd: Option<f32>,
}

impl From<&Candidate> for ValidationSummary {
    fn from(c: &Candidate) -> Self {
        Self {
            oos_sharpe_net: c.oos_sharpe_net,
            oos_sharpe_gross: c.oos_sharpe_gross,
            pbo: c.pbo,
            dsr: c.dsr,
            stress_passed: c.stress_passed,
            stress_total: c.stress_total,
            gates_passed: c.gates_passed,
            turnover_annual: c.turnover_annual,
            capacity_usd: c.capacity_usd,
        }
    }
}

/// Bundle generation context.
pub struct BundleGenerator {
    artifacts_dir: String,
}

impl BundleGenerator {
    /// Create a new bundle generator.
    pub fn new(artifacts_dir: &str) -> Self {
        Self {
            artifacts_dir: artifacts_dir.to_string(),
        }
    }

    /// Generate a complete candidate bundle.
    pub fn generate(
        &self,
        candidate: &Candidate,
        campaign_id: &str,
        seed: i64,
        config_hash: &str,
        dataset_hash: Option<&str>,
        strategy_toml: Option<&str>,
        execution_toml: Option<&str>,
    ) -> Result<String> {
        // Create bundle directory
        let bundle_dir = format!("{}/{}", self.artifacts_dir, candidate.candidate_id);
        fs::create_dir_all(&bundle_dir)?;

        // 1. Write strategy.toml
        if let Some(toml) = strategy_toml {
            fs::write(format!("{}/strategy.toml", bundle_dir), toml)?;
        } else {
            // Try to find from artifact path
            let strategy_path = format!(
                "output/scg/{}/strategy_000.toml",
                candidate.run_id
            );
            if Path::new(&strategy_path).exists() {
                fs::copy(&strategy_path, format!("{}/strategy.toml", bundle_dir))?;
            }
        }

        // 2. Write execution_config.toml
        if let Some(toml) = execution_toml {
            fs::write(format!("{}/execution_config.toml", bundle_dir), toml)?;
        } else {
            // Write default execution config
            let default_exec = r#"# Execution Configuration (from campaign)

[execution]
delay_bars = 1
bypass_for_debug = false

[execution.slippage]
type = "Constant"
bps = 10.0

[execution.fees]
tier = "B3Retail"

[execution.fill_policy]
allow_partial = false
max_participation = 0.05
"#;
            fs::write(format!("{}/execution_config.toml", bundle_dir), default_exec)?;
        }

        // 3. Write validation_summary.json
        let summary = ValidationSummary::from(candidate);
        let summary_json = serde_json::to_string_pretty(&summary)?;
        fs::write(format!("{}/validation_summary.json", bundle_dir), summary_json)?;

        // 4. Write provenance.json
        let provenance = Provenance {
            candidate_id: candidate.candidate_id.clone(),
            genome_hash: candidate.genome_hash.clone(),
            run_id: candidate.run_id.clone(),
            campaign_id: campaign_id.to_string(),
            seed,
            git_sha: super::config::CampaignConfig::git_sha(),
            git_branch: super::config::CampaignConfig::git_branch(),
            config_hash: config_hash.to_string(),
            dataset_hash: dataset_hash.map(String::from),
            created_at: Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string(),
            scg_version: env!("CARGO_PKG_VERSION").to_string(),
            original_report_path: Some(format!(
                "output/scg/{}/final_report.json",
                candidate.run_id
            )),
        };
        let provenance_json = serde_json::to_string_pretty(&provenance)?;
        fs::write(format!("{}/provenance.json", bundle_dir), provenance_json)?;

        // 5. Write replay.sh
        let replay_script = format!(
            r#"#!/bin/bash
# Replay script for candidate {}
# Generated: {}
#
# This script re-runs the backtest with the exact same configuration
# to verify reproducibility.

set -e

# Navigate to project root (works from any directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

SEED={}
RUN_ID="{}"
CAMPAIGN_ID="{}"
STRATEGY_CONFIG="$SCRIPT_DIR/strategy.toml"

echo "Replaying candidate {}"
echo "  Seed: $SEED"
echo "  Run ID: $RUN_ID"
echo "  Project root: $PROJECT_ROOT"
echo "  Strategy config: $STRATEGY_CONFIG"

# Check if strategy.toml exists
if [ ! -f "$STRATEGY_CONFIG" ]; then
    echo "ERROR: strategy.toml not found in $SCRIPT_DIR"
    exit 1
fi

# Run the backtest using pre-compiled binary
./target/release/combiner run \
    --config "$STRATEGY_CONFIG" \
    --seed $SEED \
    --output output/replay/$RUN_ID

echo "Replay complete. Compare results in output/replay/$RUN_ID"
"#,
            candidate.candidate_id,
            Utc::now().format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string(),
            seed,
            candidate.run_id,
            campaign_id,
            candidate.candidate_id,
        );
        let replay_path = format!("{}/replay.sh", bundle_dir);
        fs::write(&replay_path, replay_script)?;

        // Make replay.sh executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&replay_path)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&replay_path, perms)?;
        }

        Ok(bundle_dir)
    }
}

