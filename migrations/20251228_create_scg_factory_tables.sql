-- Strategy Factory Schema
-- Migration: 20251228_create_scg_factory_tables
-- Description: Creates the core tables for the Strategy Factory registry
--
-- Tables:
--   - scg_campaigns: Campaign metadata and status
--   - scg_runs: Individual run records with metrics
--   - scg_candidates: Top candidates from each run
--   - scg_promotions: Promotion tracking
--
-- Usage:
--   psql "$NEON_DATABASE_URL" -f migrations/20251228_create_scg_factory_tables.sql

-- =============================================================================
-- CAMPAIGNS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS scg_campaigns (
    campaign_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    tag TEXT,
    owner TEXT,
    git_branch TEXT,
    git_sha TEXT,
    config_hash TEXT NOT NULL,
    dataset_hash TEXT,
    seeds INTEGER[] NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'created',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_scg_campaigns_tag ON scg_campaigns(tag);
CREATE INDEX IF NOT EXISTS idx_scg_campaigns_status ON scg_campaigns(status);
CREATE INDEX IF NOT EXISTS idx_scg_campaigns_config_hash ON scg_campaigns(config_hash);

COMMENT ON TABLE scg_campaigns IS 'SCG campaign metadata and status tracking';
COMMENT ON COLUMN scg_campaigns.campaign_id IS 'Unique campaign identifier (format: camp_XXXXXXXXXXXX)';
COMMENT ON COLUMN scg_campaigns.config_hash IS 'SHA256 hash of campaign configuration for reproducibility';
COMMENT ON COLUMN scg_campaigns.dataset_hash IS 'SHA256 hash of dataset for data integrity tracking';

-- =============================================================================
-- RUNS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS scg_runs (
    run_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES scg_campaigns(campaign_id) ON DELETE CASCADE,
    seed BIGINT NOT NULL,
    status TEXT NOT NULL DEFAULT 'started',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_secs INTEGER,
    generations_completed INTEGER,
    total_evaluations BIGINT,
    artifact_path TEXT,
    error_message TEXT,
    best_oos_sharpe_net REAL,
    best_pbo REAL,
    candidates_count INTEGER,
    data_integrity_verdict TEXT,
    data_integrity_score REAL,
    data_integrity_report_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_scg_runs_campaign ON scg_runs(campaign_id);
CREATE INDEX IF NOT EXISTS idx_scg_runs_status ON scg_runs(status);
CREATE INDEX IF NOT EXISTS idx_scg_runs_seed ON scg_runs(campaign_id, seed);

COMMENT ON TABLE scg_runs IS 'Individual SCG run records with metrics and data integrity';
COMMENT ON COLUMN scg_runs.run_id IS 'Unique run identifier (format: run_XXXXXXXXXXXX)';
COMMENT ON COLUMN scg_runs.data_integrity_verdict IS 'PASS/FAIL verdict from data integrity gate';

-- =============================================================================
-- CANDIDATES TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS scg_candidates (
    candidate_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES scg_runs(run_id) ON DELETE CASCADE,
    genome_hash TEXT NOT NULL,
    rank INTEGER NOT NULL,
    oos_sharpe_net REAL,
    oos_sharpe_gross REAL,
    pbo REAL,
    dsr REAL,
    stress_passed INTEGER,
    stress_total INTEGER,
    gates_passed BOOLEAN,
    turnover_annual REAL,
    capacity_usd REAL,
    oos_cagr_net REAL,
    max_drawdown_net REAL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(run_id, genome_hash)
);

CREATE INDEX IF NOT EXISTS idx_scg_candidates_run ON scg_candidates(run_id);
CREATE INDEX IF NOT EXISTS idx_scg_candidates_genome ON scg_candidates(genome_hash);
CREATE INDEX IF NOT EXISTS idx_scg_candidates_rank ON scg_candidates(run_id, rank);
CREATE INDEX IF NOT EXISTS idx_scg_candidates_sharpe ON scg_candidates(oos_sharpe_net DESC NULLS LAST);

COMMENT ON TABLE scg_candidates IS 'Top candidates from each SCG run with validation metrics';
COMMENT ON COLUMN scg_candidates.candidate_id IS 'Unique candidate identifier (format: cand_XXXXXXXXXXXX)';
COMMENT ON COLUMN scg_candidates.genome_hash IS 'SHA256 hash of strategy genome for deduplication';
COMMENT ON COLUMN scg_candidates.oos_sharpe_net IS 'Out-of-sample Sharpe ratio after costs';
COMMENT ON COLUMN scg_candidates.pbo IS 'Probability of Backtest Overfitting (lower is better)';

-- =============================================================================
-- PROMOTIONS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS scg_promotions (
    promotion_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES scg_candidates(candidate_id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    promoted_at TIMESTAMPTZ DEFAULT NOW(),
    promoted_by TEXT,
    bundle_path TEXT,
    notes TEXT,
    UNIQUE(candidate_id, stage)
);

CREATE INDEX IF NOT EXISTS idx_scg_promotions_candidate ON scg_promotions(candidate_id);
CREATE INDEX IF NOT EXISTS idx_scg_promotions_stage ON scg_promotions(stage);

COMMENT ON TABLE scg_promotions IS 'Promotion tracking for candidates through research/candidate/paper stages';
COMMENT ON COLUMN scg_promotions.stage IS 'Promotion stage: research, candidate, or paper';
COMMENT ON COLUMN scg_promotions.bundle_path IS 'Path to the promoted candidate bundle';

-- =============================================================================
-- VERIFY SCHEMA
-- =============================================================================
DO $$
BEGIN
    RAISE NOTICE 'SCG Factory schema created successfully';
    RAISE NOTICE 'Tables: scg_campaigns, scg_runs, scg_candidates, scg_promotions';
END $$;




