/**
 * Ranking Logic - Multi-criteria ranking with explainability
 * 
 * Provides institutional-grade strategy ranking with:
 * - Multiple ranking methods
 * - Score calculation
 * - Explainability (why each strategy ranks where it does)
 */

// =============================================================================
// TYPES
// =============================================================================

export interface CandidateMetrics {
  candidateId: string;
  oosSharpeNet: number;
  oosCagrNet: number;
  maxDrawdownNet: number;
  pbo: number;
  dsr: number;
  gatesPassed: boolean;
  stressPassed: boolean;
  dataIntegrityOk: boolean;
}

export interface RankedCandidate extends CandidateMetrics {
  rank: number;
  score: number;
  rankReasons: string[];
}

export type RankingMethod = 'institutional' | 'pareto' | 'sharpe' | 'riskadjusted';

// =============================================================================
// SCORE FUNCTIONS
// =============================================================================

/**
 * Institutional score - Multi-criteria weighted
 * Used by NYC/Chicago prop shops
 */
function institutionalScore(c: CandidateMetrics): number {
  // Sharpe contributes 40%
  const sharpeScore = Math.min(c.oosSharpeNet / 2, 1) * 40;
  
  // PBO contributes 25% (lower is better)
  const pboScore = Math.max(0, (1 - c.pbo / 0.5)) * 25;
  
  // Stress testing contributes 20%
  const stressScore = c.stressPassed ? 20 : 0;
  
  // Gates passed contributes 15%
  const gatesScore = c.gatesPassed ? 15 : 0;
  
  return sharpeScore + pboScore + stressScore + gatesScore;
}

/**
 * Pareto score - Balance of return vs risk
 */
function paretoScore(c: CandidateMetrics): number {
  // Sharpe - penalized by drawdown
  return c.oosSharpeNet - Math.abs(c.maxDrawdownNet) * 0.05;
}

/**
 * Pure Sharpe score
 */
function sharpeScore(c: CandidateMetrics): number {
  return c.oosSharpeNet;
}

/**
 * Risk-adjusted score - Sharpe per unit of max drawdown
 */
function riskAdjustedScore(c: CandidateMetrics): number {
  if (c.maxDrawdownNet === 0) return 0;
  return (c.oosSharpeNet / Math.abs(c.maxDrawdownNet)) * 100;
}

// =============================================================================
// EXPLAINABILITY
// =============================================================================

/**
 * Generate human-readable reasons for why a candidate ranks well
 */
export function explainRank(c: CandidateMetrics): string[] {
  const reasons: string[] = [];
  
  // Sharpe-based
  if (c.oosSharpeNet >= 1.5) {
    reasons.push('Sharpe excepcional (≥1.5)');
  } else if (c.oosSharpeNet >= 1.0) {
    reasons.push('Sharpe excelente (≥1.0)');
  } else if (c.oosSharpeNet >= 0.7) {
    reasons.push('Sharpe bom (≥0.7)');
  }
  
  // PBO-based
  if (c.pbo <= 0.05) {
    reasons.push('Muito baixo risco de overfitting');
  } else if (c.pbo <= 0.10) {
    reasons.push('Baixo risco de overfitting');
  } else if (c.pbo <= 0.15) {
    reasons.push('PBO aceitável (<15%)');
  }
  
  // Stress testing
  if (c.stressPassed) {
    reasons.push('Passou testes de stress');
  }
  
  // Gates
  if (c.gatesPassed) {
    reasons.push('Passou todos os gates');
  }
  
  // Drawdown
  if (c.maxDrawdownNet > -10) {
    reasons.push('Drawdown muito baixo (<10%)');
  } else if (c.maxDrawdownNet > -15) {
    reasons.push('Drawdown controlado (<15%)');
  }
  
  // DSR
  if (c.dsr > 1.5) {
    reasons.push('DSR forte (>1.5)');
  } else if (c.dsr > 1.0) {
    reasons.push('DSR bom (>1.0)');
  }
  
  // CAGR
  if (c.oosCagrNet > 30) {
    reasons.push('CAGR alto (>30%)');
  } else if (c.oosCagrNet > 20) {
    reasons.push('CAGR sólido (>20%)');
  }
  
  // Data integrity
  if (c.dataIntegrityOk) {
    reasons.push('Dados validados');
  }
  
  return reasons.slice(0, 3); // Return top 3 reasons
}

// =============================================================================
// MAIN RANKING FUNCTION
// =============================================================================

/**
 * Rank candidates using the specified method
 * Returns candidates sorted by score with explainability
 */
export function rankCandidates(
  candidates: CandidateMetrics[],
  method: RankingMethod
): RankedCandidate[] {
  // Calculate scores
  const scored = candidates.map((c) => {
    let score: number;
    
    switch (method) {
      case 'institutional':
        score = institutionalScore(c);
        break;
      case 'pareto':
        score = paretoScore(c);
        break;
      case 'sharpe':
        score = sharpeScore(c);
        break;
      case 'riskadjusted':
        score = riskAdjustedScore(c);
        break;
      default:
        score = institutionalScore(c);
    }
    
    return {
      ...c,
      score,
      rankReasons: explainRank(c),
      rank: 0, // Will be set after sorting
    };
  });
  
  // Sort by score (descending)
  scored.sort((a, b) => b.score - a.score);
  
  // Assign ranks
  scored.forEach((c, i) => {
    c.rank = i + 1;
  });
  
  return scored;
}

// =============================================================================
// FILTERS
// =============================================================================

export interface CandidateFilters {
  minSharpe?: number;
  maxPbo?: number;
  requireGates?: boolean;
  requireStress?: boolean;
  maxDrawdown?: number;
}

/**
 * Filter candidates by criteria
 */
export function filterCandidates(
  candidates: CandidateMetrics[],
  filters: CandidateFilters
): CandidateMetrics[] {
  return candidates.filter((c) => {
    if (filters.minSharpe !== undefined && c.oosSharpeNet < filters.minSharpe) {
      return false;
    }
    if (filters.maxPbo !== undefined && c.pbo > filters.maxPbo) {
      return false;
    }
    if (filters.requireGates && !c.gatesPassed) {
      return false;
    }
    if (filters.requireStress && !c.stressPassed) {
      return false;
    }
    if (filters.maxDrawdown !== undefined && c.maxDrawdownNet < filters.maxDrawdown) {
      return false; // maxDrawdownNet is negative
    }
    return true;
  });
}

// =============================================================================
// PARETO FRONTIER
// =============================================================================

/**
 * Find Pareto-optimal candidates (non-dominated)
 * A candidate is Pareto-optimal if no other candidate is better in all dimensions
 */
export function findParetoFrontier(
  candidates: CandidateMetrics[]
): CandidateMetrics[] {
  const frontier: CandidateMetrics[] = [];
  
  for (const candidate of candidates) {
    let isDominated = false;
    
    for (const other of candidates) {
      if (other === candidate) continue;
      
      // Check if 'other' dominates 'candidate'
      // Domination: better or equal in all dimensions, strictly better in at least one
      const otherBetterSharpe = other.oosSharpeNet >= candidate.oosSharpeNet;
      const otherBetterDD = other.maxDrawdownNet >= candidate.maxDrawdownNet;
      const otherBetterPBO = other.pbo <= candidate.pbo;
      
      const otherStrictlyBetter = 
        other.oosSharpeNet > candidate.oosSharpeNet ||
        other.maxDrawdownNet > candidate.maxDrawdownNet ||
        other.pbo < candidate.pbo;
      
      if (otherBetterSharpe && otherBetterDD && otherBetterPBO && otherStrictlyBetter) {
        isDominated = true;
        break;
      }
    }
    
    if (!isDominated) {
      frontier.push(candidate);
    }
  }
  
  return frontier;
}

// =============================================================================
// COMPARISON
// =============================================================================

export interface ComparisonResult {
  winner: string | null; // null if tie
  dimension: string;
  difference: number;
  percentDiff: number;
}

/**
 * Compare two candidates across dimensions
 */
export function compareCandidates(
  a: CandidateMetrics,
  b: CandidateMetrics
): ComparisonResult[] {
  const results: ComparisonResult[] = [];
  
  // Sharpe
  results.push({
    winner: a.oosSharpeNet > b.oosSharpeNet ? a.candidateId : 
            b.oosSharpeNet > a.oosSharpeNet ? b.candidateId : null,
    dimension: 'Sharpe OOS',
    difference: a.oosSharpeNet - b.oosSharpeNet,
    percentDiff: ((a.oosSharpeNet - b.oosSharpeNet) / Math.abs(b.oosSharpeNet || 1)) * 100,
  });
  
  // PBO (lower is better)
  results.push({
    winner: a.pbo < b.pbo ? a.candidateId : 
            b.pbo < a.pbo ? b.candidateId : null,
    dimension: 'PBO',
    difference: b.pbo - a.pbo, // Inverted because lower is better
    percentDiff: ((b.pbo - a.pbo) / Math.abs(b.pbo || 1)) * 100,
  });
  
  // Max Drawdown (higher/less negative is better)
  results.push({
    winner: a.maxDrawdownNet > b.maxDrawdownNet ? a.candidateId : 
            b.maxDrawdownNet > a.maxDrawdownNet ? b.candidateId : null,
    dimension: 'Max Drawdown',
    difference: a.maxDrawdownNet - b.maxDrawdownNet,
    percentDiff: ((a.maxDrawdownNet - b.maxDrawdownNet) / Math.abs(b.maxDrawdownNet || 1)) * 100,
  });
  
  // CAGR
  results.push({
    winner: a.oosCagrNet > b.oosCagrNet ? a.candidateId : 
            b.oosCagrNet > a.oosCagrNet ? b.candidateId : null,
    dimension: 'CAGR',
    difference: a.oosCagrNet - b.oosCagrNet,
    percentDiff: ((a.oosCagrNet - b.oosCagrNet) / Math.abs(b.oosCagrNet || 1)) * 100,
  });
  
  return results;
}














