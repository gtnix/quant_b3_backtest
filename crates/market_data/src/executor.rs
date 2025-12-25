//! Stress download executor for full historical data ingestion.
//!
//! Executes download plans with rate limiting, resumption, and detailed reporting.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

use crate::brapi::BrapiClient;
use crate::db::{Database, DbError};
use crate::planner::{DownloadTask, StressPlan, TaskStatus};

// ============================================================================
// Executor Config
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorConfig {
    pub concurrency: usize,
    pub max_requests: Option<usize>,
    pub max_failures: usize,
    pub resume_from: Option<PathBuf>,
    pub dry_run: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            concurrency: 1,
            max_requests: None,
            max_failures: 50,
            resume_from: None,
            dry_run: false,
        }
    }
}

// ============================================================================
// Execution Result
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecutionResult {
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub skipped_tasks: usize,
    pub total_bars_inserted: usize,
    pub total_requests: usize,
    pub saved_requests: usize,
    pub duration_secs: f64,
    pub failures: Vec<TaskFailure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskFailure {
    pub task_id: usize,
    pub symbol: String,
    pub interval: String,
    pub range: String,
    pub error: String,
    pub http_status: Option<u16>,
    pub attempt: usize,
    pub timestamp: DateTime<Utc>,
}

// ============================================================================
// Stress Executor
// ============================================================================

pub struct StressExecutor {
    client: BrapiClient,
    db: Database,
    config: ExecutorConfig,
}

impl StressExecutor {
    pub fn new(client: BrapiClient, db: Database, config: ExecutorConfig) -> Self {
        Self { client, db, config }
    }

    /// Execute a stress plan.
    pub async fn execute(
        &self,
        plan: &mut StressPlan,
        output_dir: &Path,
    ) -> Result<ExecutionResult, DbError> {
        let started_at = Utc::now();
        info!(
            "Starting stress execution with {} tasks, concurrency={}",
            plan.tasks.len(),
            self.config.concurrency
        );

        if self.config.dry_run {
            info!("DRY RUN: No actual requests will be made");
            return Ok(ExecutionResult {
                started_at,
                completed_at: Some(Utc::now()),
                completed_tasks: 0,
                skipped_tasks: plan.tasks.len(),
                ..Default::default()
            });
        }

        // Counters
        let completed = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicUsize::new(0));
        let skipped = Arc::new(AtomicUsize::new(0));
        let bars_inserted = Arc::new(AtomicUsize::new(0));
        let requests_made = Arc::new(AtomicUsize::new(0));
        let saved_requests = Arc::new(AtomicUsize::new(0));

        let failures: Arc<tokio::sync::Mutex<Vec<TaskFailure>>> =
            Arc::new(tokio::sync::Mutex::new(Vec::new()));

        // Semaphore for concurrency control
        let semaphore = Arc::new(Semaphore::new(self.config.concurrency));
        let total_tasks = plan.tasks.len();

        // Execute tasks
        for task in plan.tasks.iter_mut() {
            // Check budget limit
            if let Some(max_req) = self.config.max_requests {
                if requests_made.load(Ordering::Relaxed) >= max_req {
                    info!("Budget limit reached ({} requests), stopping", max_req);
                    break;
                }
            }

            // Check failure threshold
            if failed.load(Ordering::Relaxed) >= self.config.max_failures {
                warn!(
                    "Max failures reached ({}), aborting",
                    self.config.max_failures
                );
                break;
            }

            // Skip already completed tasks (for resume)
            if task.status == TaskStatus::Completed || task.status == TaskStatus::Skipped {
                skipped.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            let permit = semaphore.clone().acquire_owned().await.unwrap();

            task.status = TaskStatus::InProgress;

            let result = self.execute_task(task).await;

            match result {
                Ok(bar_count) => {
                    task.status = TaskStatus::Completed;
                    completed.fetch_add(1, Ordering::Relaxed);
                    bars_inserted.fetch_add(bar_count, Ordering::Relaxed);
                    requests_made.fetch_add(1, Ordering::Relaxed);

                    debug!(
                        "Task {} completed: {} bars for {}@{}",
                        task.id, bar_count, task.symbol, task.interval
                    );
                }
                Err(e) => {
                    task.status = TaskStatus::Failed;
                    failed.fetch_add(1, Ordering::Relaxed);
                    requests_made.fetch_add(1, Ordering::Relaxed);

                    let failure = TaskFailure {
                        task_id: task.id,
                        symbol: task.symbol.clone(),
                        interval: task.interval.clone(),
                        range: task.range.clone(),
                        error: e.to_string(),
                        http_status: None,
                        attempt: 1,
                        timestamp: Utc::now(),
                    };

                    failures.lock().await.push(failure);
                    warn!("Task {} failed: {} - {}", task.id, task.symbol, e);
                }
            }

            // Log progress every 100 tasks
            let total = completed.load(Ordering::Relaxed) + failed.load(Ordering::Relaxed);
            if total % 100 == 0 && total > 0 {
                info!(
                    "Progress: {}/{} tasks ({} completed, {} failed)",
                    total,
                    total_tasks,
                    completed.load(Ordering::Relaxed),
                    failed.load(Ordering::Relaxed)
                );
            }

            drop(permit);
        }

        let completed_at = Utc::now();
        let duration = (completed_at - started_at).num_milliseconds() as f64 / 1000.0;

        let result = ExecutionResult {
            started_at,
            completed_at: Some(completed_at),
            completed_tasks: completed.load(Ordering::Relaxed),
            failed_tasks: failed.load(Ordering::Relaxed),
            skipped_tasks: skipped.load(Ordering::Relaxed),
            total_bars_inserted: bars_inserted.load(Ordering::Relaxed),
            total_requests: requests_made.load(Ordering::Relaxed),
            saved_requests: saved_requests.load(Ordering::Relaxed),
            duration_secs: duration,
            failures: failures.lock().await.clone(),
        };

        // Write result artifacts
        self.write_artifacts(&result, plan, output_dir)?;

        info!(
            "Execution complete in {:.2}s: {} completed, {} failed, {} skipped",
            duration, result.completed_tasks, result.failed_tasks, result.skipped_tasks
        );

        Ok(result)
    }

    /// Execute a single download task.
    async fn execute_task(&self, task: &DownloadTask) -> Result<usize, DbError> {
        // GATE CHECK: Validate ticker against provider_universe
        let is_active = self.db.is_ticker_active(&task.symbol).await?;
        if !is_active {
            debug!(
                "GATE BLOCKED: {} not ACTIVE in provider_universe",
                task.symbol
            );
            // Log divergence
            let _ = self
                .db
                .log_divergence(
                    &task.symbol,
                    "BLOCKED_NOT_ACTIVE",
                    false,
                    false,
                    None,
                    Some("Ticker not in provider_universe or not ACTIVE"),
                )
                .await;
            return Ok(0);
        }

        // Check watermark first (idempotency)
        let watermark = self.db.get_watermark(&task.symbol, &task.interval).await?;

        if let Some(wm) = &watermark {
            if let Some(last_ts) = wm.last_ts {
                let now = Utc::now();
                let gap_hours = (now - last_ts).num_hours();

                // Skip if data is fresh enough (less than 4 hours old for intraday, 1 day for daily)
                let threshold = if task.interval == "1d" { 24 } else { 4 };
                if gap_hours < threshold {
                    debug!(
                        "Skipping {} {} - data is fresh ({} hours old)",
                        task.symbol, task.interval, gap_hours
                    );
                    return Ok(0);
                }
            }
        }

        // Fetch data from API
        let bars = self
            .client
            .fetch_historical_bars(&task.symbol, &task.interval, &task.range)
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;

        let bar_count = bars.len();

        if bar_count == 0 {
            debug!(
                "No bars returned for {} {} {}",
                task.symbol, task.interval, task.range
            );
            return Ok(0);
        }

        // Get timestamps for watermark
        let first_ts = bars
            .first()
            .map(|b| DateTime::from_timestamp(b.date, 0))
            .flatten();
        let last_ts = bars
            .last()
            .map(|b| DateTime::from_timestamp(b.date, 0))
            .flatten();

        // Insert bars into database
        if task.interval == "1d" {
            self.db.upsert_ohlcv_batch(&task.symbol, &bars).await?;
        } else {
            self.db
                .upsert_ohlcv_intraday_batch(&task.symbol, &task.interval, &bars)
                .await?;
        }

        // Update watermark
        self.db
            .upsert_watermark(
                &task.symbol,
                &task.interval,
                first_ts,
                last_ts,
                bar_count as i32,
            )
            .await?;

        Ok(bar_count)
    }

    /// Write execution artifacts.
    fn write_artifacts(
        &self,
        result: &ExecutionResult,
        plan: &StressPlan,
        output_dir: &Path,
    ) -> Result<(), DbError> {
        std::fs::create_dir_all(output_dir).map_err(|e| DbError::Connection(e.to_string()))?;

        // Write manifest
        let manifest = IngestionRunManifest {
            version: env!("CARGO_PKG_VERSION").to_string(),
            git_hash: option_env!("GIT_HASH").map(String::from),
            started_at: result.started_at,
            completed_at: result.completed_at,
            config: self.config.clone(),
            plan_summary: plan.summary.clone(),
            result_summary: ResultSummary {
                completed_tasks: result.completed_tasks,
                failed_tasks: result.failed_tasks,
                skipped_tasks: result.skipped_tasks,
                total_bars_inserted: result.total_bars_inserted,
                total_requests: result.total_requests,
                saved_requests: result.saved_requests,
                duration_secs: result.duration_secs,
            },
        };

        let manifest_path = output_dir.join("ingestion_run_manifest.json");
        let manifest_json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| DbError::Connection(e.to_string()))?;
        std::fs::write(&manifest_path, manifest_json)
            .map_err(|e| DbError::Connection(e.to_string()))?;
        info!("Manifest written to {}", manifest_path.display());

        // Write failures
        if !result.failures.is_empty() {
            let failures_path = output_dir.join("failures.json");
            let failures_json = serde_json::to_string_pretty(&result.failures)
                .map_err(|e| DbError::Connection(e.to_string()))?;
            std::fs::write(&failures_path, failures_json)
                .map_err(|e| DbError::Connection(e.to_string()))?;
            info!("Failures written to {}", failures_path.display());
        }

        // Write updated plan (for resume)
        let plan_path = output_dir.join("stress_plan_updated.json");
        let plan_json =
            serde_json::to_string_pretty(plan).map_err(|e| DbError::Connection(e.to_string()))?;
        std::fs::write(&plan_path, plan_json).map_err(|e| DbError::Connection(e.to_string()))?;

        Ok(())
    }
}

// ============================================================================
// Manifest Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionRunManifest {
    pub version: String,
    pub git_hash: Option<String>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub config: ExecutorConfig,
    pub plan_summary: crate::planner::PlanSummary,
    pub result_summary: ResultSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultSummary {
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub skipped_tasks: usize,
    pub total_bars_inserted: usize,
    pub total_requests: usize,
    pub saved_requests: usize,
    pub duration_secs: f64,
}
