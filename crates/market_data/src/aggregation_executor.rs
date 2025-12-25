//! Aggregation Executor - executes aggregation plans with idempotency.
//!
//! Consumes aggregation_plan.json and fetches data with resume support.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;
use tracing::{debug, error, info, warn};

use crate::aggregation_planner::{AggregationPlan, AggregationTask, TaskReason};
use crate::brapi::BrapiClient;
use crate::db::{Database, DbError};

// ============================================================================
// Execution Result Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: usize,
    pub ticker: String,
    pub interval: String,
    pub status: TaskStatus,
    pub bars_fetched: usize,
    pub bars_inserted: usize,
    pub duration_ms: u64,
    pub error: Option<String>,
    pub saved_request: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Success,
    Skipped, // Already up-to-date
    Failed,
    RateLimited,
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskStatus::Success => write!(f, "SUCCESS"),
            TaskStatus::Skipped => write!(f, "SKIPPED"),
            TaskStatus::Failed => write!(f, "FAILED"),
            TaskStatus::RateLimited => write!(f, "RATE_LIMITED"),
        }
    }
}

// ============================================================================
// Execution Manifest
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionManifest {
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub plan_file: String,
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub skipped_tasks: usize,
    pub total_bars_fetched: usize,
    pub total_bars_inserted: usize,
    pub saved_requests: usize,
    pub total_duration_secs: f64,
    pub requests_made: usize,
    pub rate_limit_hits: usize,
    pub results: Vec<TaskResult>,
}

impl ExecutionManifest {
    pub fn write_json(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)?;
        info!("Execution manifest written to {}", path.display());
        Ok(())
    }

    pub fn write_failures_csv(&self, path: &Path) -> std::io::Result<()> {
        let failures: Vec<_> = self
            .results
            .iter()
            .filter(|r| r.status == TaskStatus::Failed)
            .collect();

        if failures.is_empty() {
            info!("No failures to report");
            return Ok(());
        }

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut content = "task_id,ticker,interval,error\n".to_string();
        for f in failures {
            content.push_str(&format!(
                "{},{},{},{}\n",
                f.task_id,
                f.ticker,
                f.interval,
                f.error.as_deref().unwrap_or("unknown"),
            ));
        }

        std::fs::write(path, content)?;
        info!("Failures CSV written to {}", path.display());
        Ok(())
    }

    pub fn write_success_csv(&self, path: &Path) -> std::io::Result<()> {
        let successes: Vec<_> = self
            .results
            .iter()
            .filter(|r| r.status == TaskStatus::Success)
            .collect();

        if successes.is_empty() {
            info!("No successes to report");
            return Ok(());
        }

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut content =
            "task_id,ticker,interval,bars_fetched,bars_inserted,duration_ms\n".to_string();
        for s in successes {
            content.push_str(&format!(
                "{},{},{},{},{},{}\n",
                s.task_id, s.ticker, s.interval, s.bars_fetched, s.bars_inserted, s.duration_ms,
            ));
        }

        std::fs::write(path, content)?;
        info!("Success CSV written to {}", path.display());
        Ok(())
    }
}

// ============================================================================
// Checkpoint for Resume (Optimized with HashSet for O(1) lookup)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionCheckpoint {
    pub plan_file: String,
    pub completed_task_ids: Vec<usize>, // For JSON serialization
    #[serde(skip)]
    completed_set: HashSet<usize>, // For O(1) lookup (rebuilt on load)
    pub last_updated: DateTime<Utc>,
}

impl ExecutionCheckpoint {
    pub fn new(plan_file: &str) -> Self {
        Self {
            plan_file: plan_file.to_string(),
            completed_task_ids: Vec::new(),
            completed_set: HashSet::new(),
            last_updated: Utc::now(),
        }
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut checkpoint: Self = serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        // Rebuild HashSet from Vec for O(1) lookups
        checkpoint.completed_set = checkpoint.completed_task_ids.iter().cloned().collect();
        Ok(checkpoint)
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    pub fn mark_completed(&mut self, task_id: usize) {
        if self.completed_set.insert(task_id) {
            self.completed_task_ids.push(task_id);
            self.last_updated = Utc::now();
        }
    }

    pub fn is_completed(&self, task_id: usize) -> bool {
        self.completed_set.contains(&task_id) // O(1) lookup!
    }

    pub fn completed_count(&self) -> usize {
        self.completed_set.len()
    }
}

// ============================================================================
// Aggregation Executor (Sequential - Single Connection)
// ============================================================================

pub struct AggregationExecutorConfig {
    pub concurrency: usize,
    pub budget: Option<usize>,
    pub max_failures: usize,
}

impl Default for AggregationExecutorConfig {
    fn default() -> Self {
        Self {
            concurrency: 1, // Sequential for DB safety
            budget: None,
            max_failures: 50,
        }
    }
}

/// Execute aggregation plan sequentially (single DB connection).
pub async fn execute_plan(
    db: &Database,
    client: &BrapiClient,
    plan: &AggregationPlan,
    plan_file: &str,
    checkpoint_path: Option<&Path>,
    config: AggregationExecutorConfig,
) -> Result<ExecutionManifest, DbError> {
    let started_at = Utc::now();
    info!("Starting aggregation execution: {} tasks", plan.total_tasks);

    // Load or create checkpoint
    let mut checkpoint = checkpoint_path
        .and_then(|p| ExecutionCheckpoint::load(p).ok())
        .unwrap_or_else(|| ExecutionCheckpoint::new(plan_file));

    let mut results: Vec<TaskResult> = Vec::new();
    let mut requests_made = 0usize;
    let mut rate_limit_hits = 0usize;
    let mut consecutive_failures = 0usize;

    // Process tasks sequentially
    for task in &plan.tasks {
        // Check budget
        if let Some(budget) = config.budget {
            if requests_made >= budget {
                warn!("Budget exhausted ({} requests), stopping execution", budget);
                break;
            }
        }

        // Check max failures
        if consecutive_failures >= config.max_failures {
            error!(
                "Too many consecutive failures ({}), aborting",
                consecutive_failures
            );
            break;
        }

        // Check checkpoint
        if checkpoint.is_completed(task.task_id) {
            debug!("Task {} already completed, skipping", task.task_id);
            results.push(TaskResult {
                task_id: task.task_id,
                ticker: task.ticker.clone(),
                interval: task.interval.clone(),
                status: TaskStatus::Skipped,
                bars_fetched: 0,
                bars_inserted: 0,
                duration_ms: 0,
                error: None,
                saved_request: true,
            });
            continue;
        }

        // Execute task
        let result = execute_single_task(db, client, task).await;

        match &result.status {
            TaskStatus::Success => {
                consecutive_failures = 0;
                requests_made += 1;
                checkpoint.mark_completed(task.task_id);
            }
            TaskStatus::Skipped => {
                consecutive_failures = 0;
            }
            TaskStatus::Failed => {
                consecutive_failures += 1;
                requests_made += 1;
            }
            TaskStatus::RateLimited => {
                rate_limit_hits += 1;
                consecutive_failures += 1;
                // On rate limit, wait before continuing
                tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
            }
        }

        results.push(result);

        // Save checkpoint periodically
        if let Some(cp_path) = checkpoint_path {
            if results.len() % 10 == 0 {
                let _ = checkpoint.save(cp_path);
            }
        }

        // Progress log every 50 tasks
        if results.len() % 50 == 0 {
            let elapsed_secs = (Utc::now() - started_at).num_seconds().max(1) as f64;
            let tasks_done = checkpoint.completed_count();
            let rate = tasks_done as f64 / (elapsed_secs / 60.0);
            let remaining = plan.total_tasks.saturating_sub(tasks_done);
            let eta_mins = if rate > 0.0 {
                remaining as f64 / rate
            } else {
                0.0
            };

            info!(
                "Progress: {}/{} ({:.1}%) | Rate: {:.1} tasks/min | ETA: {:.0} min | Saved: {}",
                tasks_done,
                plan.total_tasks,
                (tasks_done as f64 / plan.total_tasks as f64) * 100.0,
                rate,
                eta_mins,
                results.iter().filter(|r| r.saved_request).count()
            );
        }
    }

    // Final checkpoint save
    if let Some(cp_path) = checkpoint_path {
        let _ = checkpoint.save(cp_path);
    }

    let completed_at = Utc::now();
    let total_duration = (completed_at - started_at).num_milliseconds() as f64 / 1000.0;

    let completed_tasks = results
        .iter()
        .filter(|r| r.status == TaskStatus::Success)
        .count();
    let failed_tasks = results
        .iter()
        .filter(|r| r.status == TaskStatus::Failed)
        .count();
    let skipped_tasks = results
        .iter()
        .filter(|r| r.status == TaskStatus::Skipped)
        .count();
    let saved_requests = results.iter().filter(|r| r.saved_request).count();
    let total_bars_fetched: usize = results.iter().map(|r| r.bars_fetched).sum();
    let total_bars_inserted: usize = results.iter().map(|r| r.bars_inserted).sum();

    info!(
        "Execution complete: {}/{} tasks, {} bars inserted, {} saved requests",
        completed_tasks, plan.total_tasks, total_bars_inserted, saved_requests
    );

    Ok(ExecutionManifest {
        started_at,
        completed_at,
        plan_file: plan_file.to_string(),
        total_tasks: plan.total_tasks,
        completed_tasks,
        failed_tasks,
        skipped_tasks,
        total_bars_fetched,
        total_bars_inserted,
        saved_requests,
        total_duration_secs: total_duration,
        requests_made,
        rate_limit_hits,
        results,
    })
}

/// Execute a single task.
async fn execute_single_task(
    db: &Database,
    client: &BrapiClient,
    task: &AggregationTask,
) -> TaskResult {
    let start = std::time::Instant::now();
    debug!(
        "Executing task {}: {} {} ({:?})",
        task.task_id, task.ticker, task.interval, task.reason
    );

    // Check watermark first
    let watermark = db.get_watermark(&task.ticker, &task.interval).await;
    if let Ok(Some(wm)) = &watermark {
        if let Some(last_ts) = wm.last_ts {
            // If last_ts is recent enough, skip
            let staleness = (Utc::now() - last_ts).num_hours();
            if staleness < 4 && task.reason != TaskReason::InitialFill {
                debug!(
                    "Task {} skipped: data is fresh ({}h old)",
                    task.task_id, staleness
                );
                return TaskResult {
                    task_id: task.task_id,
                    ticker: task.ticker.clone(),
                    interval: task.interval.clone(),
                    status: TaskStatus::Skipped,
                    bars_fetched: 0,
                    bars_inserted: 0,
                    duration_ms: start.elapsed().as_millis() as u64,
                    error: None,
                    saved_request: true,
                };
            }
        }
    }

    // Fetch historical data
    let bars_result = client
        .fetch_historical(&task.ticker, &task.range_param, &task.interval)
        .await;

    match bars_result {
        Ok((bars, _metrics)) => {
            let bars_fetched = bars.len();

            if bars_fetched == 0 {
                return TaskResult {
                    task_id: task.task_id,
                    ticker: task.ticker.clone(),
                    interval: task.interval.clone(),
                    status: TaskStatus::Success,
                    bars_fetched: 0,
                    bars_inserted: 0,
                    duration_ms: start.elapsed().as_millis() as u64,
                    error: None,
                    saved_request: false,
                };
            }

            // Ensure instrument exists (for foreign key)
            if let Err(e) = db.ensure_instrument_exists(&task.ticker).await {
                warn!("Task {} failed to ensure instrument: {:?}", task.task_id, e);
            }

            // Persist bars
            let upsert_result = if task.interval == "1d" {
                db.upsert_ohlcv_batch(&task.ticker, &bars).await
            } else {
                db.upsert_ohlcv_intraday_batch(&task.ticker, &task.interval, &bars)
                    .await
            };

            match upsert_result {
                Ok(inserted) => {
                    // Update watermark
                    let first_ts = bars.first().and_then(|b| b.timestamp_utc());
                    let last_ts = bars.last().and_then(|b| b.timestamp_utc());

                    let _ = db
                        .upsert_watermark(
                            &task.ticker,
                            &task.interval,
                            first_ts,
                            last_ts,
                            inserted as i32,
                        )
                        .await;

                    TaskResult {
                        task_id: task.task_id,
                        ticker: task.ticker.clone(),
                        interval: task.interval.clone(),
                        status: TaskStatus::Success,
                        bars_fetched,
                        bars_inserted: inserted,
                        duration_ms: start.elapsed().as_millis() as u64,
                        error: None,
                        saved_request: false,
                    }
                }
                Err(e) => {
                    warn!("Task {} DB error: {:?}", task.task_id, e);
                    TaskResult {
                        task_id: task.task_id,
                        ticker: task.ticker.clone(),
                        interval: task.interval.clone(),
                        status: TaskStatus::Failed,
                        bars_fetched,
                        bars_inserted: 0,
                        duration_ms: start.elapsed().as_millis() as u64,
                        error: Some(format!("DB error: {:?}", e)),
                        saved_request: false,
                    }
                }
            }
        }
        Err(e) => {
            let error_str = format!("{}", e);
            let status = if error_str.contains("429") || error_str.to_lowercase().contains("rate") {
                TaskStatus::RateLimited
            } else {
                TaskStatus::Failed
            };

            warn!("Task {} API error: {}", task.task_id, e);
            TaskResult {
                task_id: task.task_id,
                ticker: task.ticker.clone(),
                interval: task.interval.clone(),
                status,
                bars_fetched: 0,
                bars_inserted: 0,
                duration_ms: start.elapsed().as_millis() as u64,
                error: Some(error_str),
                saved_request: false,
            }
        }
    }
}
