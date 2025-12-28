-- Migration: Add listing/delisting dates to provider_universe
-- Purpose: V2 Eligibility-Aware Universe - enables point-in-time eligibility checks
-- Date: 2025-12-28

-- Add listing/delisting dates to provider_universe
ALTER TABLE provider_universe 
ADD COLUMN IF NOT EXISTS listing_date DATE,
ADD COLUMN IF NOT EXISTS delisting_date DATE,
ADD COLUMN IF NOT EXISTS eligibility_source VARCHAR(20) DEFAULT 'UNKNOWN';

-- Index for efficient date range queries
CREATE INDEX IF NOT EXISTS idx_provider_universe_dates 
ON provider_universe (listing_date, delisting_date);

-- Comment for documentation
COMMENT ON COLUMN provider_universe.listing_date IS 'First trading date (IPO or first data point)';
COMMENT ON COLUMN provider_universe.delisting_date IS 'Last trading date (delisting date or NULL if still active)';
COMMENT ON COLUMN provider_universe.eligibility_source IS 'Source of eligibility data: DATA_DERIVED|PROVIDER_API|MANUAL';

