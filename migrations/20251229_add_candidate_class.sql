-- Migration: 20251229_add_candidate_class
-- Description: Add candidate classification for Stage A (research) vs Stage B (validated)
--
-- This enables persisting >= 1000 research candidates from Stage A evolution
-- while keeping Stage B validation for top candidates only.

-- Add candidate_class column
ALTER TABLE scg_candidates 
    ADD COLUMN IF NOT EXISTS candidate_class TEXT NOT NULL DEFAULT 'research';

-- Add rank_in_run for deterministic ordering within a run
ALTER TABLE scg_candidates 
    ADD COLUMN IF NOT EXISTS rank_in_run INTEGER;

-- Add source_stage to track origin (A = evolution, B = validated)
ALTER TABLE scg_candidates 
    ADD COLUMN IF NOT EXISTS source_stage TEXT NOT NULL DEFAULT 'A';

-- Create index for filtering by class
CREATE INDEX IF NOT EXISTS idx_scg_candidates_class 
    ON scg_candidates(candidate_class);

-- Create index for class + run combination (common query pattern)
CREATE INDEX IF NOT EXISTS idx_scg_candidates_run_class 
    ON scg_candidates(run_id, candidate_class);

-- Update existing candidates to be validated (they came from Stage B)
UPDATE scg_candidates 
SET candidate_class = 'validated', source_stage = 'B' 
WHERE candidate_class = 'research';

-- Add comments
COMMENT ON COLUMN scg_candidates.candidate_class IS 'research = Stage A bulk candidates, validated = Stage B top candidates';
COMMENT ON COLUMN scg_candidates.rank_in_run IS 'Deterministic rank within the run (1 = best)';
COMMENT ON COLUMN scg_candidates.source_stage IS 'A = from evolution HoF, B = from validation stage';

DO $$
BEGIN
    RAISE NOTICE 'Migration 20251229_add_candidate_class completed';
END $$;






















