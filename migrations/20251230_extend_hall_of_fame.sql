-- Hall of Fame Extension for SCG Promotions
-- Migration: 20251230_extend_hall_of_fame
-- Description: Extends scg_promotions table with hall_of_fame metrics and class
--
-- Usage:
--   psql "$DATABASE_URL" -f migrations/20251230_extend_hall_of_fame.sql

-- =============================================================================
-- ADD HALL OF FAME COLUMNS TO SCG_PROMOTIONS
-- =============================================================================

-- Promotion class (standard | hall_of_fame)
ALTER TABLE scg_promotions 
  ADD COLUMN IF NOT EXISTS promotion_class TEXT DEFAULT 'standard';

-- Performance metrics for hall of fame entries
ALTER TABLE scg_promotions 
  ADD COLUMN IF NOT EXISTS oos_sharpe_net REAL,
  ADD COLUMN IF NOT EXISTS pbo REAL,
  ADD COLUMN IF NOT EXISTS dsr REAL,
  ADD COLUMN IF NOT EXISTS max_drawdown_net REAL,
  ADD COLUMN IF NOT EXISTS cagr_net REAL;

-- Validation metrics
ALTER TABLE scg_promotions 
  ADD COLUMN IF NOT EXISTS stress_passed INTEGER,
  ADD COLUMN IF NOT EXISTS stress_total INTEGER,
  ADD COLUMN IF NOT EXISTS gates_passed BOOLEAN;

-- Provenance tracking
ALTER TABLE scg_promotions 
  ADD COLUMN IF NOT EXISTS git_sha TEXT,
  ADD COLUMN IF NOT EXISTS config_hash TEXT,
  ADD COLUMN IF NOT EXISTS dataset_hash TEXT;

-- Market info
ALTER TABLE scg_promotions 
  ADD COLUMN IF NOT EXISTS market TEXT DEFAULT 'BR';

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_promotions_class 
  ON scg_promotions(promotion_class);

CREATE INDEX IF NOT EXISTS idx_promotions_hall_of_fame_sharpe 
  ON scg_promotions(oos_sharpe_net DESC NULLS LAST) 
  WHERE promotion_class = 'hall_of_fame';

CREATE INDEX IF NOT EXISTS idx_promotions_promoted_at 
  ON scg_promotions(promoted_at DESC);

CREATE INDEX IF NOT EXISTS idx_promotions_market 
  ON scg_promotions(market);

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON COLUMN scg_promotions.promotion_class IS 'Type of promotion: standard (manual) or hall_of_fame (auto-promoted by OMP)';
COMMENT ON COLUMN scg_promotions.oos_sharpe_net IS 'Out-of-sample Sharpe ratio after costs at promotion time';
COMMENT ON COLUMN scg_promotions.pbo IS 'Probability of Backtest Overfitting at promotion time';
COMMENT ON COLUMN scg_promotions.dsr IS 'Deflated Sharpe Ratio at promotion time';
COMMENT ON COLUMN scg_promotions.max_drawdown_net IS 'Maximum drawdown after costs';
COMMENT ON COLUMN scg_promotions.cagr_net IS 'Compound Annual Growth Rate after costs';
COMMENT ON COLUMN scg_promotions.git_sha IS 'Git commit SHA for reproducibility';
COMMENT ON COLUMN scg_promotions.config_hash IS 'Hash of campaign configuration';
COMMENT ON COLUMN scg_promotions.market IS 'Target market: BR or US';

-- =============================================================================
-- VERIFY MIGRATION
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Hall of Fame extension migration completed successfully';
    RAISE NOTICE 'New columns: promotion_class, oos_sharpe_net, pbo, dsr, max_drawdown_net, cagr_net, stress_passed, stress_total, gates_passed, git_sha, config_hash, dataset_hash, market';
END $$;


