//! Strategy Factory Registry - PostgreSQL-backed experiment tracking.
//!
//! Provides CRUD operations for campaigns, runs, candidates, and promotions.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rustls::ClientConfig;
use serde::{Deserialize, Serialize};

use tokio_postgres::{Client, Row};
use tokio_postgres_rustls::MakeRustlsConnect;
use tracing::info;

/// Connection string environment variable name.
pub const DATABASE_URL_ENV: &str = "NEON_DATABASE_URL";

// =============================================================================
// DATA TYPES
// =============================================================================

/// Campaign record from the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Campaign {
    pub campaign_id: String,
    pub name: String,
    pub tag: Option<String>,
    pub owner: Option<String>,
    pub git_branch: Option<String>,
    pub git_sha: Option<String>,
    pub config_hash: String,
    pub dataset_hash: Option<String>,
    pub seeds: Vec<i32>,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub notes: Option<String>,
}

/// Run record from the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub run_id: String,
    pub campaign_id: String,
    pub seed: i64,
    pub status: String,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub duration_secs: Option<i32>,
    pub generations_completed: Option<i32>,
    pub total_evaluations: Option<i64>,
    pub artifact_path: Option<String>,
    pub error_message: Option<String>,
    pub best_oos_sharpe_net: Option<f32>,
    pub best_pbo: Option<f32>,
    pub candidates_count: Option<i32>,
    pub data_integrity_verdict: Option<String>,
    pub data_integrity_score: Option<f32>,
    pub data_integrity_report_path: Option<String>,
}

/// Candidate record from the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub candidate_id: String,
    pub run_id: String,
    pub genome_hash: String,
    pub rank: i32,
    pub oos_sharpe_net: Option<f32>,
    pub oos_sharpe_gross: Option<f32>,
    pub pbo: Option<f32>,
    pub dsr: Option<f32>,
    pub stress_passed: Option<i32>,
    pub stress_total: Option<i32>,
    pub gates_passed: Option<bool>,
    pub turnover_annual: Option<f32>,
    pub capacity_usd: Option<f32>,
    pub created_at: DateTime<Utc>,
}

/// Promotion record from the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Promotion {
    pub promotion_id: String,
    pub candidate_id: String,
    pub stage: String,
    pub promoted_at: DateTime<Utc>,
    pub promoted_by: Option<String>,
    pub bundle_path: Option<String>,
    pub notes: Option<String>,
}

/// Run status enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    Started,
    Completed,
    Failed,
    Cancelled,
}

impl RunStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Campaign status enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CampaignStatus {
    Created,
    Running,
    Completed,
    Failed,
}

impl CampaignStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

// =============================================================================
// REGISTRY CLIENT
// =============================================================================

/// Strategy Factory Registry client.
pub struct Registry {
    client: Client,
}

impl Registry {
    /// Connect to the PostgreSQL database using the connection string from environment.
    pub async fn connect() -> Result<Self> {
        let database_url = std::env::var(DATABASE_URL_ENV)
            .with_context(|| format!("Environment variable {} not set", DATABASE_URL_ENV))?;

        Self::connect_with_url(&database_url).await
    }

    /// Connect to the PostgreSQL database with a specific connection string.
    pub async fn connect_with_url(database_url: &str) -> Result<Self> {
        // Build rustls config with webpki roots
        let mut root_store = rustls::RootCertStore::empty();
        root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

        let tls_config = ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();

        let tls = MakeRustlsConnect::new(tls_config);

        let (client, connection) = tokio_postgres::connect(database_url, tls)
            .await
            .context("Failed to connect to PostgreSQL")?;

        // Spawn connection handler
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                eprintln!("PostgreSQL connection error: {}", e);
            }
        });

        info!("Connected to Strategy Factory registry");
        Ok(Self { client })
    }

    // =========================================================================
    // CAMPAIGN OPERATIONS
    // =========================================================================

    /// Register a new campaign.
    pub async fn register_campaign(
        &self,
        campaign_id: &str,
        name: &str,
        tag: Option<&str>,
        owner: Option<&str>,
        git_branch: Option<&str>,
        git_sha: Option<&str>,
        config_hash: &str,
        dataset_hash: Option<&str>,
        seeds: &[i32],
        notes: Option<&str>,
    ) -> Result<()> {
        self.client
            .execute(
                r#"
                INSERT INTO scg_campaigns 
                    (campaign_id, name, tag, owner, git_branch, git_sha, config_hash, dataset_hash, seeds, notes)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (campaign_id) DO UPDATE SET
                    updated_at = NOW(),
                    status = 'running'
                "#,
                &[
                    &campaign_id,
                    &name,
                    &tag,
                    &owner,
                    &git_branch,
                    &git_sha,
                    &config_hash,
                    &dataset_hash,
                    &seeds,
                    &notes,
                ],
            )
            .await
            .context("Failed to register campaign")?;

        info!(campaign_id, name, "Registered campaign");
        Ok(())
    }

    /// Get a campaign by ID.
    pub async fn get_campaign(&self, campaign_id: &str) -> Result<Option<Campaign>> {
        let row = self
            .client
            .query_opt(
                "SELECT * FROM scg_campaigns WHERE campaign_id = $1",
                &[&campaign_id],
            )
            .await
            .context("Failed to get campaign")?;

        Ok(row.map(|r| Self::row_to_campaign(&r)))
    }

    /// List all campaigns with optional tag filter.
    pub async fn list_campaigns(&self, tag: Option<&str>) -> Result<Vec<Campaign>> {
        let rows = if let Some(tag) = tag {
            self.client
                .query(
                    "SELECT * FROM scg_campaigns WHERE tag = $1 ORDER BY created_at DESC",
                    &[&tag],
                )
                .await?
        } else {
            self.client
                .query(
                    "SELECT * FROM scg_campaigns ORDER BY created_at DESC",
                    &[],
                )
                .await?
        };

        Ok(rows.iter().map(Self::row_to_campaign).collect())
    }

    /// Update campaign status.
    pub async fn update_campaign_status(
        &self,
        campaign_id: &str,
        status: CampaignStatus,
    ) -> Result<()> {
        self.client
            .execute(
                "UPDATE scg_campaigns SET status = $1, updated_at = NOW() WHERE campaign_id = $2",
                &[&status.as_str(), &campaign_id],
            )
            .await
            .context("Failed to update campaign status")?;

        Ok(())
    }

    fn row_to_campaign(row: &Row) -> Campaign {
        Campaign {
            campaign_id: row.get("campaign_id"),
            name: row.get("name"),
            tag: row.get("tag"),
            owner: row.get("owner"),
            git_branch: row.get("git_branch"),
            git_sha: row.get("git_sha"),
            config_hash: row.get("config_hash"),
            dataset_hash: row.get("dataset_hash"),
            seeds: row.get("seeds"),
            status: row.get("status"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
            notes: row.get("notes"),
        }
    }

    // =========================================================================
    // RUN OPERATIONS
    // =========================================================================

    /// Register a run start.
    pub async fn register_run_start(
        &self,
        run_id: &str,
        campaign_id: &str,
        seed: i64,
    ) -> Result<()> {
        self.client
            .execute(
                r#"
                INSERT INTO scg_runs (run_id, campaign_id, seed, status)
                VALUES ($1, $2, $3, 'started')
                ON CONFLICT (run_id) DO UPDATE SET
                    status = 'started',
                    started_at = NOW(),
                    completed_at = NULL,
                    error_message = NULL
                "#,
                &[&run_id, &campaign_id, &seed],
            )
            .await
            .context("Failed to register run start")?;

        info!(run_id, campaign_id, seed, "Registered run start");
        Ok(())
    }

    /// Register a run end (success or failure).
    pub async fn register_run_end(
        &self,
        run_id: &str,
        status: RunStatus,
        duration_secs: Option<i32>,
        generations_completed: Option<i32>,
        total_evaluations: Option<i64>,
        artifact_path: Option<&str>,
        error_message: Option<&str>,
        best_oos_sharpe_net: Option<f32>,
        best_pbo: Option<f32>,
        candidates_count: Option<i32>,
    ) -> Result<()> {
        self.client
            .execute(
                r#"
                UPDATE scg_runs SET
                    status = $1,
                    completed_at = NOW(),
                    duration_secs = $2,
                    generations_completed = $3,
                    total_evaluations = $4,
                    artifact_path = $5,
                    error_message = $6,
                    best_oos_sharpe_net = $7,
                    best_pbo = $8,
                    candidates_count = $9
                WHERE run_id = $10
                "#,
                &[
                    &status.as_str(),
                    &duration_secs,
                    &generations_completed,
                    &total_evaluations,
                    &artifact_path,
                    &error_message,
                    &best_oos_sharpe_net,
                    &best_pbo,
                    &candidates_count,
                    &run_id,
                ],
            )
            .await
            .context("Failed to register run end")?;

        info!(run_id, status = status.as_str(), "Registered run end");
        Ok(())
    }

    /// Get a run by ID.
    pub async fn get_run(&self, run_id: &str) -> Result<Option<Run>> {
        let row = self
            .client
            .query_opt("SELECT * FROM scg_runs WHERE run_id = $1", &[&run_id])
            .await
            .context("Failed to get run")?;

        Ok(row.map(|r| Self::row_to_run(&r)))
    }

    /// List runs for a campaign.
    pub async fn list_runs(&self, campaign_id: &str) -> Result<Vec<Run>> {
        let rows = self
            .client
            .query(
                "SELECT * FROM scg_runs WHERE campaign_id = $1 ORDER BY seed",
                &[&campaign_id],
            )
            .await
            .context("Failed to list runs")?;

        Ok(rows.iter().map(Self::row_to_run).collect())
    }

    /// Get incomplete runs (not completed/failed) for a campaign.
    pub async fn get_incomplete_seeds(&self, campaign_id: &str) -> Result<Vec<i64>> {
        // Get all expected seeds from campaign
        let campaign = self
            .get_campaign(campaign_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Campaign not found: {}", campaign_id))?;

        // Get completed seeds
        let completed_rows = self
            .client
            .query(
                "SELECT seed FROM scg_runs WHERE campaign_id = $1 AND status = 'completed'",
                &[&campaign_id],
            )
            .await?;

        let completed_seeds: std::collections::HashSet<i64> = completed_rows
            .iter()
            .map(|r| r.get::<_, i64>("seed"))
            .collect();

        // Return seeds not in completed
        let incomplete: Vec<i64> = campaign
            .seeds
            .iter()
            .map(|&s| s as i64)
            .filter(|s| !completed_seeds.contains(s))
            .collect();

        Ok(incomplete)
    }

    fn row_to_run(row: &Row) -> Run {
        Run {
            run_id: row.get("run_id"),
            campaign_id: row.get("campaign_id"),
            seed: row.get("seed"),
            status: row.get("status"),
            started_at: row.get("started_at"),
            completed_at: row.get("completed_at"),
            duration_secs: row.get("duration_secs"),
            generations_completed: row.get("generations_completed"),
            total_evaluations: row.get("total_evaluations"),
            artifact_path: row.get("artifact_path"),
            error_message: row.get("error_message"),
            best_oos_sharpe_net: row.get("best_oos_sharpe_net"),
            best_pbo: row.get("best_pbo"),
            candidates_count: row.get("candidates_count"),
            data_integrity_verdict: row.get("data_integrity_verdict"),
            data_integrity_score: row.get("data_integrity_score"),
            data_integrity_report_path: row.get("data_integrity_report_path"),
        }
    }

    // =========================================================================
    /// Register data integrity verdict for a run.
    pub async fn register_data_integrity(
        &self,
        run_id: &str,
        verdict: &str,
        score: f32,
        report_path: &str,
    ) -> Result<()> {
        self.client
            .execute(
                r#"
                UPDATE scg_runs SET
                    data_integrity_verdict = $1,
                    data_integrity_score = $2,
                    data_integrity_report_path = $3
                WHERE run_id = $4
                "#,
                &[&verdict, &score, &report_path, &run_id],
            )
            .await
            .context("Failed to register data integrity")?;
        Ok(())
    }

    // =========================================================================

    // CANDIDATE OPERATIONS
    // =========================================================================

    /// Register a candidate from a run.
    pub async fn register_candidate(
        &self,
        candidate_id: &str,
        run_id: &str,
        genome_hash: &str,
        rank: i32,
        oos_sharpe_net: Option<f32>,
        oos_sharpe_gross: Option<f32>,
        pbo: Option<f32>,
        dsr: Option<f32>,
        stress_passed: Option<i32>,
        stress_total: Option<i32>,
        gates_passed: Option<bool>,
        turnover_annual: Option<f32>,
        capacity_usd: Option<f32>,
    ) -> Result<()> {
        self.client
            .execute(
                r#"
                INSERT INTO scg_candidates 
                    (candidate_id, run_id, genome_hash, rank, oos_sharpe_net, oos_sharpe_gross,
                     pbo, dsr, stress_passed, stress_total, gates_passed, turnover_annual, capacity_usd)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (run_id, genome_hash) DO UPDATE SET
                    rank = $4,
                    oos_sharpe_net = $5,
                    oos_sharpe_gross = $6,
                    pbo = $7,
                    dsr = $8,
                    stress_passed = $9,
                    stress_total = $10,
                    gates_passed = $11,
                    turnover_annual = $12,
                    capacity_usd = $13
                "#,
                &[
                    &candidate_id,
                    &run_id,
                    &genome_hash,
                    &rank,
                    &oos_sharpe_net,
                    &oos_sharpe_gross,
                    &pbo,
                    &dsr,
                    &stress_passed,
                    &stress_total,
                    &gates_passed,
                    &turnover_annual,
                    &capacity_usd,
                ],
            )
            .await
            .context("Failed to register candidate")?;

        Ok(())
    }

    /// Get candidates for a run.
    pub async fn get_candidates(&self, run_id: &str) -> Result<Vec<Candidate>> {
        let rows = self
            .client
            .query(
                "SELECT * FROM scg_candidates WHERE run_id = $1 ORDER BY rank",
                &[&run_id],
            )
            .await
            .context("Failed to get candidates")?;

        Ok(rows.iter().map(Self::row_to_candidate).collect())
    }

    /// Get top N candidates for a run.
    pub async fn get_top_candidates(&self, run_id: &str, limit: i32) -> Result<Vec<Candidate>> {
        let rows = self
            .client
            .query(
                "SELECT * FROM scg_candidates WHERE run_id = $1 ORDER BY rank LIMIT $2",
                &[&run_id, &(limit as i64)],
            )
            .await
            .context("Failed to get top candidates")?;

        Ok(rows.iter().map(Self::row_to_candidate).collect())
    }

    fn row_to_candidate(row: &Row) -> Candidate {
        Candidate {
            candidate_id: row.get("candidate_id"),
            run_id: row.get("run_id"),
            genome_hash: row.get("genome_hash"),
            rank: row.get("rank"),
            oos_sharpe_net: row.get("oos_sharpe_net"),
            oos_sharpe_gross: row.get("oos_sharpe_gross"),
            pbo: row.get("pbo"),
            dsr: row.get("dsr"),
            stress_passed: row.get("stress_passed"),
            stress_total: row.get("stress_total"),
            gates_passed: row.get("gates_passed"),
            turnover_annual: row.get("turnover_annual"),
            capacity_usd: row.get("capacity_usd"),
            created_at: row.get("created_at"),
        }
    }

    // =========================================================================
    // PROMOTION OPERATIONS
    // =========================================================================

    /// Check if a candidate has already been promoted to a stage.
    pub async fn is_already_promoted(&self, genome_hash: &str, stage: &str) -> Result<bool> {
        let row = self
            .client
            .query_opt(
                r#"
                SELECT 1 FROM scg_promotions p
                JOIN scg_candidates c ON p.candidate_id = c.candidate_id
                WHERE c.genome_hash = $1 AND p.stage = $2
                "#,
                &[&genome_hash, &stage],
            )
            .await
            .context("Failed to check promotion")?;

        Ok(row.is_some())
    }

    /// Register a promotion.
    pub async fn register_promotion(
        &self,
        promotion_id: &str,
        candidate_id: &str,
        stage: &str,
        promoted_by: Option<&str>,
        bundle_path: Option<&str>,
        notes: Option<&str>,
    ) -> Result<()> {
        self.client
            .execute(
                r#"
                INSERT INTO scg_promotions 
                    (promotion_id, candidate_id, stage, promoted_by, bundle_path, notes)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (candidate_id, stage) DO NOTHING
                "#,
                &[
                    &promotion_id,
                    &candidate_id,
                    &stage,
                    &promoted_by,
                    &bundle_path,
                    &notes,
                ],
            )
            .await
            .context("Failed to register promotion")?;

        info!(promotion_id, candidate_id, stage, "Registered promotion");
        Ok(())
    }

    /// Get promotions for a stage.
    pub async fn list_promotions(&self, stage: Option<&str>) -> Result<Vec<Promotion>> {
        let rows = if let Some(stage) = stage {
            self.client
                .query(
                    "SELECT * FROM scg_promotions WHERE stage = $1 ORDER BY promoted_at DESC",
                    &[&stage],
                )
                .await?
        } else {
            self.client
                .query(
                    "SELECT * FROM scg_promotions ORDER BY promoted_at DESC",
                    &[],
                )
                .await?
        };

        Ok(rows.iter().map(Self::row_to_promotion).collect())
    }

    fn row_to_promotion(row: &Row) -> Promotion {
        Promotion {
            promotion_id: row.get("promotion_id"),
            candidate_id: row.get("candidate_id"),
            stage: row.get("stage"),
            promoted_at: row.get("promoted_at"),
            promoted_by: row.get("promoted_by"),
            bundle_path: row.get("bundle_path"),
            notes: row.get("notes"),
        }
    }

    // =========================================================================
    // COMPARISON OPERATIONS
    // =========================================================================

    /// Get top candidates from multiple runs for comparison.
    pub async fn compare_runs(&self, run_ids: &[&str], limit: i32) -> Result<Vec<Candidate>> {
        if run_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Build query with placeholders
        let placeholders: Vec<String> = (1..=run_ids.len()).map(|i| format!("${}", i)).collect();
        let query = format!(
            "SELECT * FROM scg_candidates WHERE run_id IN ({}) ORDER BY run_id, rank LIMIT ${}",
            placeholders.join(", "),
            run_ids.len() + 1
        );

        // Build params
        let mut params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = Vec::new();
        for run_id in run_ids {
            params.push(run_id);
        }
        let limit_i64 = (limit * run_ids.len() as i32) as i64;
        params.push(&limit_i64);

        let rows = self
            .client
            .query(&query, &params)
            .await
            .context("Failed to compare runs")?;

        Ok(rows.iter().map(Self::row_to_candidate).collect())
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Generate a unique run ID.
pub fn generate_run_id() -> String {
    format!("run_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..12])
}

/// Generate a unique campaign ID.
pub fn generate_campaign_id() -> String {
    format!("camp_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..12])
}

/// Generate a unique candidate ID.
pub fn generate_candidate_id() -> String {
    format!("cand_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..12])
}

/// Generate a unique promotion ID.
pub fn generate_promotion_id() -> String {
    format!("prom_{}", &uuid::Uuid::new_v4().to_string().replace('-', "")[..12])
}
