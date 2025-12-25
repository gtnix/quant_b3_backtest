-- Migration: Create interest_rates table for point-in-time risk-free rates
-- ============================================================================
-- Status: STAGED ONLY - DO NOT APPLY until dividend pipeline completes
-- Author: AI Assistant (AO7-002 resolution)
-- Date: 2025-12-25
-- ============================================================================
--
-- Purpose:
--   Store historical risk-free rates for carry calculation (Technique 7).
--   Enables point-in-time backtesting without look-ahead bias.
--
-- Data Sources:
--   BR (SELIC): BCB SGS Serie 432 (SELIC Meta diária)
--               Endpoint: https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados?formato=json
--
--   US (T-Bill): FRED TB3MS (3-Month Treasury Bill Secondary Market Rate)
--                Endpoint: https://api.stlouisfed.org/fred/series/observations?series_id=TB3MS
--
-- Usage:
--   After applying this migration, implement ingestão jobs to populate data.
--   CarryFilter will use these rates when RiskFreeRateProvider is configured
--   to read from the database instead of using fallback values.
--
-- ============================================================================

-- Table for storing historical interest rates
CREATE TABLE IF NOT EXISTS interest_rates (
    id SERIAL PRIMARY KEY,
    
    -- Date the rate was effective (not publication date)
    rate_date DATE NOT NULL,
    
    -- Market region: BR (Brazil) or US (United States)
    region VARCHAR(2) NOT NULL CHECK (region IN ('BR', 'US')),
    
    -- Type of rate:
    --   SELIC: Brazilian Central Bank base rate (Serie 432 - meta)
    --   CDI: Brazilian interbank deposit rate (Serie 4389)
    --   TBILL_3M: US 3-Month Treasury Bill rate (FRED TB3MS)
    rate_type VARCHAR(20) NOT NULL CHECK (rate_type IN ('SELIC', 'CDI', 'TBILL_3M')),
    
    -- Annualized rate as decimal (e.g., 0.107500 = 10.75%)
    -- Constrained to valid range: 0% to 100%
    rate DECIMAL(10, 8) NOT NULL CHECK (rate >= 0 AND rate <= 1.0),
    
    -- Data source for audit trail
    source VARCHAR(50),
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Ensure one rate per date/region/type combination
    UNIQUE (rate_date, region, rate_type)
);

-- Index for efficient date-range queries by region
CREATE INDEX IF NOT EXISTS idx_interest_rates_date_region 
    ON interest_rates(rate_date DESC, region);

-- Index for lookups by rate type
CREATE INDEX IF NOT EXISTS idx_interest_rates_type 
    ON interest_rates(rate_type, rate_date DESC);

-- Composite index for efficient point-in-time queries by region and type
CREATE INDEX IF NOT EXISTS idx_interest_rates_pit 
    ON interest_rates(region, rate_type, rate_date DESC);

-- Comment for documentation
COMMENT ON TABLE interest_rates IS 
    'Historical risk-free rates for carry calculation (point-in-time). '
    'Used by CarryFilter to compute carry = dividend_yield - risk_free_rate. '
    'Sources: BCB SGS (BR), FRED (US).';

COMMENT ON COLUMN interest_rates.rate IS 
    'Annualized rate as decimal. Example: 0.1075 = 10.75% per year.';

COMMENT ON COLUMN interest_rates.rate_date IS 
    'Effective date of the rate, not the publication date. '
    'Use this for point-in-time queries in backtesting.';

-- ============================================================================
-- Sample data (for testing only - DO NOT use in production backfill)
-- ============================================================================
-- INSERT INTO interest_rates (rate_date, region, rate_type, rate, source) VALUES
--     ('2024-12-25', 'BR', 'SELIC', 0.1075, 'BCB_SGS_432'),
--     ('2024-12-25', 'US', 'TBILL_3M', 0.0435, 'FRED_TB3MS');
-- ============================================================================

