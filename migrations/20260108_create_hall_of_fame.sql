-- Global Hall of Fame Table
-- Migration: 20260108_create_hall_of_fame
-- Description: Creates a permanent, global hall_of_fame table for all-time best strategies
--
-- Usage:
--   psql "$DATABASE_URL" -f migrations/20260108_create_hall_of_fame.sql

-- =============================================================================
-- HALL OF FAME TABLE (Global, Permanent)
-- =============================================================================

CREATE TABLE IF NOT EXISTS hall_of_fame (
    id SERIAL PRIMARY KEY,
    genome_hash TEXT UNIQUE NOT NULL,
    candidate_id TEXT NOT NULL,
    
    -- Performance metrics (ordered by Sharpe)
    oos_sharpe_net REAL NOT NULL,
    oos_cagr_net REAL,
    max_drawdown_net REAL,
    pbo REAL,
    dsr REAL,
    
    -- Stress testing results
    stress_passed INTEGER,
    stress_total INTEGER,
    gates_passed BOOLEAN DEFAULT false,
    
    -- Provenance tracking
    run_id TEXT NOT NULL,
    campaign_id TEXT,
    promoted_at TIMESTAMPTZ DEFAULT NOW(),
    git_sha TEXT,
    market TEXT DEFAULT 'BR',
    
    -- Strategy payload
    strategy_toml TEXT,
    genome_json JSONB,
    
    -- Global rank (updated on insert/update)
    global_rank INTEGER
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- =============================================================================
-- FOREIGN KEY (single source of truth - no duplication in scg_promotions)
-- =============================================================================

ALTER TABLE hall_of_fame 
    ADD CONSTRAINT fk_hof_candidate 
    FOREIGN KEY (candidate_id) REFERENCES scg_candidates(candidate_id) 
    ON DELETE CASCADE;

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_hof_sharpe ON hall_of_fame(oos_sharpe_net DESC);
CREATE INDEX IF NOT EXISTS idx_hof_market ON hall_of_fame(market);
CREATE INDEX IF NOT EXISTS idx_hof_promoted_at ON hall_of_fame(promoted_at DESC);
CREATE INDEX IF NOT EXISTS idx_hof_global_rank ON hall_of_fame(global_rank ASC);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE hall_of_fame IS 'Global permanent Hall of Fame - all-time best strategies by Sharpe';
COMMENT ON COLUMN hall_of_fame.genome_hash IS 'Unique hash of the strategy genome for deduplication';
COMMENT ON COLUMN hall_of_fame.oos_sharpe_net IS 'Out-of-sample Sharpe ratio after costs - primary ranking metric';
COMMENT ON COLUMN hall_of_fame.global_rank IS 'Current global rank (1 = best). Updated when new strategies are added.';
COMMENT ON COLUMN hall_of_fame.market IS 'Target market: BR or US';

-- =============================================================================
-- FUNCTION: Update global ranks after insert/update
-- =============================================================================

CREATE OR REPLACE FUNCTION update_hall_of_fame_ranks()
RETURNS TRIGGER AS $$
BEGIN
    -- Update all ranks based on Sharpe (descending)
    WITH ranked AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY oos_sharpe_net DESC) as new_rank
        FROM hall_of_fame
    )
    UPDATE hall_of_fame h
    SET global_rank = r.new_rank
    FROM ranked r
    WHERE h.id = r.id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- TRIGGER: Auto-update ranks on changes
-- =============================================================================

DROP TRIGGER IF EXISTS trg_update_hof_ranks ON hall_of_fame;
CREATE TRIGGER trg_update_hof_ranks
    AFTER INSERT OR UPDATE OF oos_sharpe_net OR DELETE
    ON hall_of_fame
    FOR EACH STATEMENT
    EXECUTE FUNCTION update_hall_of_fame_ranks();

-- =============================================================================
-- VERIFY MIGRATION
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Hall of Fame table created successfully';
    RAISE NOTICE 'Features: permanent storage, auto-ranking by Sharpe, market filtering';
END $$;
