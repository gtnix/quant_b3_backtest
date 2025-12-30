/**
 * TooltipInfo - Informative tooltip component with structured content
 * 
 * Provides context-aware help for UI controls with:
 * - What: One-line description
 * - Impact: How it affects generation/validation
 * - When: When to adjust
 * - Example: Practical example
 */

import React, { useState } from 'react';
import { Info } from 'lucide-react';

// =============================================================================
// TYPES
// =============================================================================

export interface TooltipContent {
  what: string;
  impact: string;
  when: string;
  example: string;
}

export interface QuickTooltipContent {
  term: string;
  definition: string;
  formula?: string;
  benchmark?: string;
  interpretation?: string;
}

interface TooltipInfoProps {
  content: TooltipContent;
  children?: React.ReactNode;
}

interface SimpleTooltipProps {
  text: string;
  children: React.ReactNode;
}

interface QuickTooltipProps {
  termKey: keyof typeof QUANT_TOOLTIPS;
  position?: 'top' | 'bottom' | 'left' | 'right';
  size?: 'sm' | 'md';
}

// =============================================================================
// QUANT TOOLTIPS DATABASE - English explanations for all quant terms
// =============================================================================

export const QUANT_TOOLTIPS: Record<string, QuickTooltipContent> = {
  // ═══════════════════════════════════════════════════════════════════════════
  // PERFORMANCE METRICS
  // ═══════════════════════════════════════════════════════════════════════════
  sharpe: {
    term: 'Sharpe Ratio',
    definition: 'Risk-adjusted return measuring excess return per unit of volatility. Higher means better reward for the risk taken.',
    formula: '(Return - RiskFreeRate) / Volatility',
    benchmark: '≥1.0 good, ≥2.0 excellent',
    interpretation: 'A Sharpe of 1.5 means 1.5% extra return for each 1% of risk.'
  },
  sharpe_oos: {
    term: 'Out-of-Sample Sharpe',
    definition: 'Sharpe Ratio calculated on data the strategy never saw during optimization. The real test of strategy quality.',
    benchmark: 'Should be close to In-Sample Sharpe. Big drops suggest overfitting.'
  },
  sharpe_net: {
    term: 'Net Sharpe Ratio',
    definition: 'Sharpe Ratio after deducting all trading costs (fees, slippage). The actual performance you would achieve.',
    interpretation: 'Always use NET for trading decisions. GROSS is misleading.'
  },
  cagr: {
    term: 'CAGR (Compound Annual Growth Rate)',
    definition: 'Annualized return accounting for compounding. How much your investment grows per year on average.',
    formula: '(EndValue/StartValue)^(1/years) - 1',
    benchmark: '15%+ is strong for equities',
    interpretation: '15% CAGR doubles money in ~5 years. 25% in ~3 years.'
  },
  max_drawdown: {
    term: 'Maximum Drawdown (MDD)',
    definition: 'Largest peak-to-trough decline before recovery. The worst loss you would have experienced.',
    formula: 'Max((Peak - Trough) / Peak)',
    benchmark: '<20% conservative, <30% moderate',
    interpretation: '-25% MDD means at worst you were down 25% from your peak.'
  },
  volatility: {
    term: 'Volatility (Annualized)',
    definition: 'Standard deviation of returns annualized. Measures how much returns fluctuate around the mean.',
    formula: 'StdDev(DailyReturns) × √252',
    benchmark: '10-20% typical for equity strategies',
    interpretation: '15% vol means returns typically stay within ±15% of expected.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // RISK-ADJUSTED RATIOS
  // ═══════════════════════════════════════════════════════════════════════════
  sortino: {
    term: 'Sortino Ratio',
    definition: 'Like Sharpe but only penalizes downside volatility. Ignores upside "risk" which is actually good.',
    formula: '(Return - RiskFree) / DownsideDeviation',
    benchmark: '≥1.5 good, ≥2.0 excellent',
    interpretation: 'Better than Sharpe for strategies with asymmetric returns.'
  },
  calmar: {
    term: 'Calmar Ratio',
    definition: 'Annual return divided by maximum drawdown. Measures reward per unit of worst-case loss.',
    formula: 'CAGR / |MaxDrawdown|',
    benchmark: '≥1.0 good, ≥3.0 excellent',
    interpretation: 'Calmar 2.0 means you earned 2% return for each 1% of max loss risk.'
  },
  omega: {
    term: 'Omega Ratio',
    definition: 'Probability-weighted ratio of gains to losses above a threshold. Captures entire return distribution.',
    formula: '∫(gains above threshold) / ∫(losses below threshold)',
    benchmark: '≥1.5 good, ≥2.0 excellent',
    interpretation: 'Omega 1.8 means gains are 80% larger than losses on average.'
  },
  profit_factor: {
    term: 'Profit Factor',
    definition: 'Sum of all winning trades divided by sum of all losing trades. Simple profitability measure.',
    formula: 'Σ(Wins) / Σ(Losses)',
    benchmark: '≥1.5 good, ≥2.0 excellent',
    interpretation: 'PF 2.0 means you make $2 for every $1 you lose.'
  },
  win_rate: {
    term: 'Win Rate',
    definition: 'Percentage of trades that are profitable. Must be considered with average win/loss size.',
    formula: 'WinningTrades / TotalTrades × 100',
    benchmark: 'Depends on payoff ratio. 40% can be excellent with 3:1 reward/risk.',
    interpretation: 'High win rate with small wins can underperform low win rate with big wins.'
  },
  payoff_ratio: {
    term: 'Payoff Ratio (Reward/Risk)',
    definition: 'Average winning trade size divided by average losing trade size.',
    formula: 'AvgWin / AvgLoss',
    benchmark: '≥1.5 good, ≥2.0 excellent',
    interpretation: 'Combined with win rate determines if strategy is profitable.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // VALUE AT RISK
  // ═══════════════════════════════════════════════════════════════════════════
  var_95: {
    term: 'Value at Risk (95%)',
    definition: 'Maximum expected daily loss at 95% confidence. On 95% of days, losses won\'t exceed this.',
    formula: '5th percentile of daily returns',
    benchmark: 'Depends on risk tolerance',
    interpretation: 'VaR 2% means 1 in 20 days you might lose more than 2%.'
  },
  var_99: {
    term: 'Value at Risk (99%)',
    definition: 'Maximum expected daily loss at 99% confidence. More conservative than VaR95.',
    formula: '1st percentile of daily returns',
    interpretation: 'Captures more extreme tail events than VaR95.'
  },
  cvar_95: {
    term: 'CVaR / Expected Shortfall (95%)',
    definition: 'Average loss in the worst 5% of cases. Better captures tail risk than VaR.',
    formula: 'E[Loss | Loss > VaR95]',
    benchmark: 'Should be ~1.5x VaR for normal distribution',
    interpretation: 'If VaR is 2%, CVaR might be 3% - the average bad day loss.'
  },
  cvar_99: {
    term: 'CVaR / Expected Shortfall (99%)',
    definition: 'Average loss in the worst 1% of cases. Most conservative tail risk measure.',
    formula: 'E[Loss | Loss > VaR99]',
    interpretation: 'Use for stress testing worst-case scenarios.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // DISTRIBUTION STATISTICS
  // ═══════════════════════════════════════════════════════════════════════════
  skewness: {
    term: 'Skewness',
    definition: 'Asymmetry of return distribution. Positive = more extreme gains, Negative = more extreme losses.',
    formula: 'E[(X-μ)³] / σ³',
    benchmark: 'Positive is preferable (right-skewed gains)',
    interpretation: 'Negative skew common in selling options, positive in buying.'
  },
  kurtosis: {
    term: 'Excess Kurtosis',
    definition: 'Tail thickness compared to normal distribution. Positive = fat tails (more extreme events).',
    formula: 'E[(X-μ)⁴] / σ⁴ - 3',
    benchmark: '>0 is common in finance (fat tails)',
    interpretation: 'High kurtosis means black swan events are more likely than normal suggests.'
  },
  tail_ratio: {
    term: 'Tail Ratio',
    definition: 'Ratio of 95th percentile gain to 5th percentile loss. Measures upside vs downside extremes.',
    formula: 'Percentile95 / |Percentile5|',
    benchmark: '>1.0 means bigger upside than downside tails',
    interpretation: 'Tail ratio 1.5 means extreme gains are 50% larger than extreme losses.'
  },
  stability: {
    term: 'Timeseries Stability',
    definition: 'R² of linear regression on cumulative returns. Measures how steadily the strategy grows.',
    formula: 'R² of CumulativeReturns ~ Time',
    benchmark: '>0.9 very stable, <0.7 choppy',
    interpretation: 'High stability = consistent compounding. Low = volatile equity curve.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // DRAWDOWN METRICS
  // ═══════════════════════════════════════════════════════════════════════════
  longest_dd: {
    term: 'Longest Drawdown Duration',
    definition: 'Maximum number of days spent below the previous peak. How long until recovery.',
    benchmark: '<180 days preferred',
    interpretation: '400 days underwater is psychologically brutal even if eventually profitable.'
  },
  avg_dd_duration: {
    term: 'Average Drawdown Duration',
    definition: 'Mean time to recover from drawdowns. Shorter is better for capital efficiency.',
    benchmark: '<60 days is good',
    interpretation: 'Quick recoveries mean capital isn\'t stuck in losing positions.'
  },
  time_underwater: {
    term: 'Time Underwater',
    definition: 'Percentage of time spent below the previous peak. How often you\'re in drawdown.',
    benchmark: '<50% is good',
    interpretation: '70% underwater means you\'re usually losing - hard to hold.'
  },
  gain_to_pain: {
    term: 'Gain-to-Pain Ratio',
    definition: 'Sum of all returns divided by sum of absolute negative returns. Overall reward/suffering ratio.',
    formula: 'Σ(Returns) / Σ|NegativeReturns|',
    benchmark: '>1.0 required for profitability, >2.0 excellent',
    interpretation: 'G2P 1.5 means you make 1.5x more than you lose in total.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // VALIDATION METRICS
  // ═══════════════════════════════════════════════════════════════════════════
  pbo: {
    term: 'Probability of Backtest Overfitting (PBO)',
    definition: 'Statistical likelihood that a strategy is curve-fitted to historical noise rather than genuine patterns.',
    formula: 'Based on CPCV degradation distribution',
    benchmark: '<15% safe, <10% excellent',
    interpretation: 'PBO 8% = 8% chance this is just luck. 30% = very concerning.'
  },
  dsr: {
    term: 'Deflated Sharpe Ratio (DSR)',
    definition: 'Sharpe Ratio adjusted for multiple testing bias. Accounts for trying many strategies.',
    formula: 'SR × correction_factor(trials)',
    benchmark: '>0.5 after deflation is good',
    interpretation: 'If you tested 100 strategies, DSR adjusts for the "best" being lucky.'
  },
  t_stat: {
    term: 'T-Statistic',
    definition: 'Statistical significance of the Sharpe Ratio. Higher means less likely due to chance.',
    formula: 'SR × √(n/252)',
    benchmark: '≥2.0 for 95% confidence',
    interpretation: 't-stat 2.5 means <1% chance this Sharpe is random luck.'
  },
  p_value: {
    term: 'P-Value',
    definition: 'Probability that the observed returns could occur by chance. Lower is more significant.',
    formula: '2 × (1 - Φ(|t-stat|))',
    benchmark: '<0.05 for 95% confidence',
    interpretation: 'p-value 0.01 means only 1% chance this is luck.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // WALK-FORWARD ANALYSIS
  // ═══════════════════════════════════════════════════════════════════════════
  wfa: {
    term: 'Walk-Forward Analysis (WFA)',
    definition: 'Validation technique that trains on past data and tests on next period repeatedly. Simulates real trading.',
    interpretation: 'The gold standard for strategy validation. Tests how strategy adapts over time.'
  },
  is_oos: {
    term: 'In-Sample / Out-of-Sample',
    definition: 'IS = data used for optimization. OOS = data never seen during training. OOS is the real test.',
    interpretation: 'Good IS with bad OOS = overfitting. OOS performance is what you\'ll actually get.'
  },
  degradation_ratio: {
    term: 'Degradation Ratio',
    definition: 'How much OOS performance retains from IS. Measures overfitting severity.',
    formula: 'OOS_Sharpe / IS_Sharpe × 100%',
    benchmark: '>50% robust, <30% concerning',
    interpretation: '70% degradation = OOS keeps 70% of IS performance. Good sign.'
  },
  consistency_score: {
    term: 'Consistency Score',
    definition: 'Percentage of WFA folds that are profitable. Measures reliability across time periods.',
    benchmark: '>60% good, >80% excellent',
    interpretation: '75% consistency = profitable in 3 of 4 periods tested.'
  },
  wfa_window: {
    term: 'WFA Window Size',
    definition: 'Length of in-sample training period. Longer = more data but less tests.',
    benchmark: '12-24 months typical',
    interpretation: 'Balance between enough data to learn and enough tests to validate.'
  },
  wfa_step: {
    term: 'WFA Step Size',
    definition: 'How much to advance between WFA folds. Shorter = more tests but more overlap.',
    benchmark: '3-6 months typical',
    interpretation: 'Smaller steps give more data points but may be correlated.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // MONTE CARLO SIMULATION
  // ═══════════════════════════════════════════════════════════════════════════
  monte_carlo: {
    term: 'Monte Carlo Simulation',
    definition: 'Statistical technique using random sampling to estimate distributions of possible outcomes.',
    interpretation: 'Shows range of possible futures, not just one backtest path.'
  },
  bootstrap: {
    term: 'Bootstrap Resampling',
    definition: 'Technique that shuffles historical returns to generate alternate scenarios. Preserves statistical properties.',
    interpretation: 'Creates 1000s of possible equity curves from the same returns.'
  },
  block_size: {
    term: 'Block Size (Bootstrap)',
    definition: 'Size of return blocks when resampling. Preserves autocorrelation within blocks.',
    benchmark: '5-21 days typical',
    interpretation: 'Block=1 is IID (independent). Block=21 preserves monthly patterns.'
  },
  confidence_interval: {
    term: 'Confidence Interval',
    definition: 'Range of values likely to contain the true parameter with specified probability.',
    benchmark: '95% CI is standard',
    interpretation: '95% CI [0.8, 1.2] means true Sharpe is probably between 0.8 and 1.2.'
  },
  percentile_p5: {
    term: 'P5 (5th Percentile)',
    definition: 'Value below which 5% of outcomes fall. Represents worst-case scenario.',
    interpretation: 'Use P5 for pessimistic planning. 95% of outcomes are better than this.'
  },
  percentile_p50: {
    term: 'P50 (Median)',
    definition: 'Value where half of outcomes are above and half below. Robust central estimate.',
    interpretation: 'More robust than mean for skewed distributions.'
  },
  percentile_p95: {
    term: 'P95 (95th Percentile)',
    definition: 'Value below which 95% of outcomes fall. Represents optimistic scenario.',
    interpretation: 'Don\'t plan on P95 - only 5% chance of achieving it.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // GENETIC ALGORITHM / EVOLUTION
  // ═══════════════════════════════════════════════════════════════════════════
  genetic_algorithm: {
    term: 'Genetic Algorithm',
    definition: 'Optimization inspired by evolution. Strategies "breed" and mutate, fittest survive.',
    interpretation: 'Explores vast strategy spaces efficiently through natural selection.'
  },
  population: {
    term: 'Population Size',
    definition: 'Number of strategies evolving simultaneously. Larger = more diversity but slower.',
    benchmark: '100-200 typical',
    interpretation: 'Population 100 means 100 strategies compete each generation.'
  },
  generation: {
    term: 'Generation',
    definition: 'One cycle of evaluation, selection, and breeding. Evolution progresses through generations.',
    benchmark: '30-100 generations typical',
    interpretation: 'Each generation should show improvement in best/mean fitness.'
  },
  fitness: {
    term: 'Fitness Function',
    definition: 'Score determining which strategies survive and reproduce. Usually Sharpe or composite metric.',
    interpretation: 'Higher fitness = better strategy. Evolution maximizes this.'
  },
  pareto_frontier: {
    term: 'Pareto Frontier',
    definition: 'Set of strategies where none is better in all objectives. Represents optimal trade-offs.',
    interpretation: 'All frontier strategies are valid choices depending on your priorities.'
  },
  convergence: {
    term: 'Convergence',
    definition: 'When population stops improving significantly. May indicate optimal found or stuck.',
    interpretation: 'Early convergence may mean local optimum. Try larger population.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // STRESS TESTING
  // ═══════════════════════════════════════════════════════════════════════════
  stress_test: {
    term: 'Stress Testing',
    definition: 'Testing strategy against extreme historical scenarios like crashes and high volatility.',
    interpretation: 'Strategies passing stress tests are more likely to survive real crises.'
  },
  stress_scenario: {
    term: 'Stress Scenario',
    definition: 'Specific market condition used for stress testing (e.g., 2008 crisis, COVID crash).',
    interpretation: 'Each scenario tests a different type of market stress.'
  },
  stress_degradation: {
    term: 'Stress Degradation',
    definition: 'How much performance drops under stress conditions vs normal conditions.',
    benchmark: '<50% degradation is good',
    interpretation: '30% degradation = strategy keeps 70% of performance under stress.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // VALIDATION GATES
  // ═══════════════════════════════════════════════════════════════════════════
  gates_passed: {
    term: 'Validation Gates Passed',
    definition: 'Whether strategy passed all minimum thresholds for production readiness.',
    interpretation: 'Gates include Sharpe, PBO, stress tests, and consistency requirements.'
  },
  validated: {
    term: 'Validated Status',
    definition: 'Strategy passed all institutional validation criteria and is production-ready.',
    interpretation: 'Validated strategies have high confidence of real-world performance.'
  },
  research: {
    term: 'Research Status',
    definition: 'Strategy shows promise but hasn\'t passed all validation gates yet.',
    interpretation: 'Needs more testing or parameter adjustment before trading.'
  },
  cpcv: {
    term: 'CPCV (Combinatorial Purged Cross-Validation)',
    definition: 'Advanced validation generating multiple IS/OOS combinations to calculate PBO statistically.',
    interpretation: 'More rigorous than simple train/test split. Industry standard.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // EXECUTION & COSTS
  // ═══════════════════════════════════════════════════════════════════════════
  net_vs_gross: {
    term: 'NET vs GROSS',
    definition: 'NET = after all costs (fees, slippage). GROSS = before costs. Always use NET for decisions.',
    interpretation: 'A 2.0 Sharpe GROSS might be 0.5 NET after costs - big difference!'
  },
  slippage: {
    term: 'Slippage',
    definition: 'Difference between expected and actual execution price. Caused by latency and market impact.',
    benchmark: '1-5 bps for liquid stocks',
    interpretation: 'High frequency strategies are very sensitive to slippage.'
  },
  delay_bars: {
    term: 'Delay Bars',
    definition: 'Number of bars between signal and execution. Simulates real-world latency.',
    benchmark: '1 bar is conservative',
    interpretation: 'Delay=0 assumes instant execution - unrealistic for most traders.'
  },
  turnover: {
    term: 'Annual Turnover',
    definition: 'How many times the portfolio is completely replaced per year. Higher = more costs.',
    formula: 'Total traded value / Average portfolio value',
    benchmark: '<12x for cost efficiency',
    interpretation: 'Turnover 24x means trading twice per month on average.'
  },
  capacity: {
    term: 'Strategy Capacity',
    definition: 'Maximum capital the strategy can manage before market impact degrades returns.',
    interpretation: '$10M capacity means performance degrades above that AUM.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // PORTFOLIO & COMPARISON
  // ═══════════════════════════════════════════════════════════════════════════
  correlation_matrix: {
    term: 'Correlation Matrix',
    definition: 'Table showing pairwise correlations between strategy returns. Low correlation = better diversification.',
    benchmark: '<0.5 for good diversification',
    interpretation: 'Strategies with 0.2 correlation provide better combined performance.'
  },
  diversification_ratio: {
    term: 'Diversification Ratio',
    definition: 'Ratio of sum of individual volatilities to portfolio volatility. Measures diversification benefit.',
    formula: 'Σ(weights × volatilities) / PortfolioVolatility',
    benchmark: '>1.5 is good diversification',
    interpretation: 'Ratio 2.0 means diversification cuts risk in half.'
  },
  
  // ═══════════════════════════════════════════════════════════════════════════
  // MISC
  // ═══════════════════════════════════════════════════════════════════════════
  best_day: {
    term: 'Best Day Return',
    definition: 'Single best daily return in the backtest period.',
    interpretation: 'Shows upside potential. Very high values may indicate outlier dependence.'
  },
  worst_day: {
    term: 'Worst Day Return',
    definition: 'Single worst daily return in the backtest period. Your nightmare scenario.',
    interpretation: 'Can you emotionally handle this loss in a single day?'
  },
  best_month: {
    term: 'Best Month Return',
    definition: 'Single best monthly return in the backtest period.',
    interpretation: 'Strong months shouldn\'t be essential for overall profitability.'
  },
  worst_month: {
    term: 'Worst Month Return',
    definition: 'Single worst monthly return in the backtest period.',
    interpretation: 'Most investors review monthly - can you explain this to clients?'
  },
  rolling_sharpe: {
    term: 'Rolling Sharpe (252-day)',
    definition: 'Sharpe calculated over trailing 252 trading days, rolling forward. Shows stability.',
    interpretation: 'Wide swings in rolling Sharpe indicate regime sensitivity.'
  },
  rolling_volatility: {
    term: 'Rolling Volatility (252-day)',
    definition: 'Volatility calculated over trailing year, rolling forward. Shows risk variation.',
    interpretation: 'Spiking rolling vol during crises is normal but informative.'
  },
};

// =============================================================================
// LEGACY TOOLTIP DATABASE (for Cockpit page)
// =============================================================================

export const TOOLTIPS: Record<string, TooltipContent> = {
  // Compute Budget
  max_runtime: {
    what: 'Maximum time the system will use to discover strategies',
    impact: 'More time = more strategies evaluated = higher chance of finding good ones.',
    when: 'Increase for deeper exploration, decrease for quick tests',
    example: '15 min for initial exploration. 1h for deep analysis.',
  },
  population_size: {
    what: 'Number of strategies evolving simultaneously',
    impact: 'Larger population = more genetic diversity = finds solutions in larger spaces.',
    when: 'Increase if strategies converge too early',
    example: '100 for production, 200 for exhaustive exploration',
  },
  max_generations: {
    what: 'Maximum number of evolutionary cycles',
    impact: 'More generations = more strategy refinement. Diminishing returns after ~50.',
    when: 'Leave default or increase if runtime allows',
    example: '50 generations usually sufficient for convergence',
  },
  workers: {
    what: 'Parallel threads for strategy evaluation',
    impact: 'More workers = faster, but uses more CPU/memory. Ideal: physical cores.',
    when: 'Reduce if system becomes slow for other tasks',
    example: '8 workers on 8-core CPU uses 100% capacity',
  },
  seeds: {
    what: 'Seeds for experiment reproducibility',
    impact: 'Multiple seeds = more robust results (less luck dependence).',
    when: 'Use 3-5 seeds for institutional validation',
    example: '3 seeds = 3 independent runs, result is the average',
  },
  
  // Gates
  min_oos_sharpe: {
    what: 'Minimum Sharpe Ratio in Out-of-Sample period',
    impact: 'Gate filtering strategies with insufficient performance.',
    when: 'Adjust based on benchmark. More volatile markets may have lower thresholds.',
    example: 'Sharpe 0.5 = 50% more return than risk. 1.0 = excellent.',
  },
  max_pbo: {
    what: 'Probability of Backtest Overfitting',
    impact: 'Measures chance strategy is "lucky" vs genuinely good.',
    when: 'Keep ≤0.15 for production strategies',
    example: 'PBO 0.08 = 8% overfitting chance. 0.30 = concerning.',
  },
  min_stress_passed: {
    what: 'Minimum stress tests the strategy must pass',
    impact: 'Tests robustness in extreme historical scenarios.',
    when: 'Use 4+ for production. 0 for quick exploration.',
    example: '4 of 8 tests = strategy survives most crashes',
  },
  stress_testing: {
    what: 'Simulates extreme market scenarios',
    impact: 'Tests each strategy against 2x volatility, price gaps, prolonged drawdowns.',
    when: 'Always enable for production. Disable only for quick tests.',
    example: 'Strategy passing stress tests survived 2008 and COVID',
  },
  
  // Ranking
  ranking_institutional: {
    what: 'Multi-criteria weighted ranking (institutional standard)',
    impact: 'Weighs Sharpe OOS (40%), PBO (25%), stress (20%), gates (15%).',
    when: 'Use as default for production',
    example: 'Sharpe 1.2 + PBO 0.05 scores higher than Sharpe 1.5 + PBO 0.25',
  },
  ranking_pareto: {
    what: 'Pareto frontier (non-dominated strategies)',
    impact: 'Shows strategies optimal in at least one dimension.',
    when: 'Use to explore trade-offs (e.g., risk vs return)',
    example: '5 strategies on frontier = 5 valid choices depending on preference',
  },
  ranking_sharpe: {
    what: 'Orders only by Sharpe Ratio OOS NET',
    impact: 'Simple but may reward overfitting. Ignores PBO and stress.',
    when: 'Use for initial analysis or when PBO already validated',
    example: 'Top 1 by Sharpe may have high PBO - verify!',
  },
  ranking_riskadjusted: {
    what: 'Sharpe divided by Maximum Drawdown',
    impact: 'Penalizes strategies with large drops even if good Sharpe.',
    when: 'Use if drawdown is priority (loss aversion)',
    example: 'Sharpe 1.0 with DD 10% > Sharpe 1.5 with DD 30%',
  },
};

// =============================================================================
// COMPONENTS
// =============================================================================

export function TooltipInfo({ content, children }: TooltipInfoProps) {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <div className="relative inline-block">
      <button
        type="button"
        className="inline-flex items-center justify-center w-4 h-4 ml-1 text-xs text-cyan-400 hover:text-cyan-300 rounded-full border border-cyan-400/30 hover:border-cyan-400/60 transition-colors"
        onMouseEnter={() => setIsOpen(true)}
        onMouseLeave={() => setIsOpen(false)}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Mais informações"
      >
        ?
      </button>
      
      {isOpen && (
        <div className="absolute z-50 w-80 p-4 mt-2 left-0 bg-slate-900 border border-cyan-500/30 rounded-lg shadow-xl shadow-cyan-500/10">
          <div className="space-y-3 text-sm">
            <div>
              <span className="text-cyan-400 font-mono text-xs uppercase tracking-wider">O que é</span>
              <p className="text-slate-200 mt-1">{content.what}</p>
            </div>
            
            <div>
              <span className="text-amber-400 font-mono text-xs uppercase tracking-wider">Impacto</span>
              <p className="text-slate-300 mt-1">{content.impact}</p>
            </div>
            
            <div>
              <span className="text-emerald-400 font-mono text-xs uppercase tracking-wider">Quando ajustar</span>
              <p className="text-slate-300 mt-1">{content.when}</p>
            </div>
            
            <div className="pt-2 border-t border-slate-700">
              <span className="text-slate-500 font-mono text-xs uppercase tracking-wider">Exemplo</span>
              <p className="text-slate-400 mt-1 italic">{content.example}</p>
            </div>
          </div>
          
          {/* Arrow */}
          <div className="absolute -top-2 left-4 w-4 h-4 bg-slate-900 border-l border-t border-cyan-500/30 transform rotate-45" />
        </div>
      )}
      
      {children}
    </div>
  );
}

export function SimpleTooltip({ text, children }: SimpleTooltipProps) {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <div 
      className="relative inline-block"
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      {children}
      
      {isOpen && (
        <div className="absolute z-50 px-3 py-2 mt-1 left-1/2 transform -translate-x-1/2 bg-slate-800 border border-slate-600 rounded text-sm text-slate-200 whitespace-nowrap shadow-lg">
          {text}
          <div className="absolute -top-1 left-1/2 transform -translate-x-1/2 w-2 h-2 bg-slate-800 border-l border-t border-slate-600 rotate-45" />
        </div>
      )}
    </div>
  );
}

// =============================================================================
// HELPER COMPONENT
// =============================================================================

interface InfoIconProps {
  tooltipKey: keyof typeof TOOLTIPS;
}

export function InfoIcon({ tooltipKey }: InfoIconProps) {
  const content = TOOLTIPS[tooltipKey];
  if (!content) return null;
  
  return <TooltipInfo content={content} />;
}

export default TooltipInfo;




