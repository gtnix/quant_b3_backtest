import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { pool, readJsonFile, readTomlFile, readCsvFile, getArtifactsRoot, PROJECT_ROOT, generateDisplayName } from '../db.js';

const router = Router();

router.get('/candidates/recent', async (req, res) => {
  try {
    const result = await pool.query(`SELECT c.*, r.run_id, camp.name as campaign_name FROM scg_candidates c LEFT JOIN scg_runs r ON c.run_id = r.run_id LEFT JOIN scg_campaigns camp ON r.campaign_id = camp.campaign_id ORDER BY c.oos_sharpe_net DESC NULLS LAST, c.created_at DESC LIMIT $1`, [parseInt(req.query.limit) || 10]);
    res.json({ candidates: result.rows.map(c => ({ candidate_id: c.candidate_id, genome_hash: c.genome_hash || '', rank: c.rank_in_run || 1, display_name: `Strategy #${c.rank_in_run || 1} | ${(c.genome_hash || '').slice(-8)}`, oos_sharpe_net: c.oos_sharpe_net || 0, oos_cagr_net: c.oos_cagr_net || 0, max_drawdown_net: c.max_drawdown_net, pbo: c.pbo || 0, dsr: c.dsr || 0, gates_passed: c.gates_passed || false, run_id: c.run_id, campaign_name: c.campaign_name })), count: result.rows.length });
  } catch (e) { res.status(500).json({ error: e.message, candidates: [] }); }
});

router.get('/candidates/:runId', async (req, res) => {
  const csvPath = path.join(getArtifactsRoot(), 'top_candidates', req.params.runId, 'top1000.csv');
  if (fs.existsSync(csvPath)) {
    let cands = readCsvFile(csvPath).map((r, i) => ({ rank: parseInt(r.rank) || i + 1, candidate_id: r.candidate_id, genome_hash: r.genome_hash, display_name: `Strategy #${r.rank} | ${r.candidate_id.slice(-8)}`, candidate_class: r.gates_passed === 'true' ? 'validated' : 'research', oos_sharpe_net: parseFloat(r.oos_sharpe_net) || 0, oos_cagr_net: parseFloat(r.oos_cagr_net) || 0, max_drawdown_net: parseFloat(r.max_drawdown_net) || 0, pbo: parseFloat(r.pbo) || 0, dsr: parseFloat(r.dsr) || 0, stress_passed: parseInt(r.stress_passed) || 0, stress_total: parseInt(r.stress_total) || 0, gates_passed: r.gates_passed === 'true' }));
    if (req.query.search) { const q = req.query.search.toLowerCase(); cands = cands.filter(c => c.candidate_id.toLowerCase().includes(q) || c.display_name.toLowerCase().includes(q)); }
    if (req.query.candidate_class) cands = cands.filter(c => c.candidate_class === req.query.candidate_class);
    if (req.query.max_pbo) cands = cands.filter(c => c.pbo <= parseFloat(req.query.max_pbo));
    return res.json(cands.slice(0, parseInt(req.query.limit) || 100));
  }
  try {
    const result = await pool.query(`SELECT * FROM scg_candidates WHERE run_id = $1 ORDER BY rank_in_run ASC LIMIT $2`, [req.params.runId, parseInt(req.query.limit) || 100]);
    let cands = result.rows.map((c, i) => ({ rank: c.rank_in_run || i + 1, candidate_id: c.candidate_id, genome_hash: c.genome_hash, display_name: `Strategy #${c.rank_in_run || i + 1} | ${c.candidate_id.slice(-8)}`, candidate_class: c.candidate_class || (c.gates_passed ? 'validated' : 'research'), oos_sharpe_net: c.oos_sharpe_net || 0, oos_cagr_net: c.oos_cagr_net || 0, max_drawdown_net: c.max_drawdown_net, pbo: c.pbo || 0, dsr: c.dsr || 0, stress_passed: c.stress_passed || 0, stress_total: c.stress_total || 0, gates_passed: c.gates_passed || false, data_source: 'neon' }));
    if (req.query.search) { const q = req.query.search.toLowerCase(); cands = cands.filter(c => c.candidate_id.toLowerCase().includes(q) || c.display_name.toLowerCase().includes(q)); }
    if (req.query.candidate_class) cands = cands.filter(c => c.candidate_class === req.query.candidate_class);
    if (req.query.max_pbo) cands = cands.filter(c => c.pbo <= parseFloat(req.query.max_pbo));
    return res.json(cands);
  } catch (e) { res.status(404).json({ error: `Candidates for run ${req.params.runId} not found` }); }
});

router.get('/candidate/:candidateId', async (req, res) => {
  const bundlePath = path.join(getArtifactsRoot(), 'candidates', req.params.candidateId);
  if (!fs.existsSync(bundlePath)) {
    try {
      const result = await pool.query(`SELECT c.*, r.campaign_id, r.seed, camp.name as campaign_name, camp.tag as campaign_tag, camp.git_branch, camp.git_sha FROM scg_candidates c LEFT JOIN scg_runs r ON c.run_id = r.run_id LEFT JOIN scg_campaigns camp ON r.campaign_id = camp.campaign_id WHERE c.candidate_id = $1`, [req.params.candidateId]);
      if (result.rows.length > 0) {
        const c = result.rows[0];
        return res.json({ candidate_id: c.candidate_id, genome_hash: c.genome_hash || '', rank: c.rank_in_run || 0, candidate_class: c.candidate_class || (c.gates_passed ? 'validated' : 'research'), display_name: `Strategy #${c.rank_in_run || 1} | ${(c.genome_hash || '').slice(-8)}`, source_stage: c.source_stage || 'A', oos_sharpe_net: c.oos_sharpe_net || 0, oos_sharpe_gross: c.oos_sharpe_gross || c.oos_sharpe_net || 0, pbo: c.pbo || 0, dsr: c.dsr || 0, oos_cagr_net: c.oos_cagr_net || 0, max_drawdown_net: c.max_drawdown_net, turnover_annual: c.turnover_annual || 0, capacity_usd: c.capacity_usd, stress_passed: c.stress_passed || 0, stress_total: c.stress_total || 0, gates_passed: c.gates_passed || false, provenance: { run_id: c.run_id, campaign_id: c.campaign_id, campaign_name: c.campaign_name, seed: c.seed, git_branch: c.git_branch, git_sha: c.git_sha, created_at: c.created_at }, validation: { wfa_passed: c.gates_passed, cpcv_passed: c.gates_passed, pbo_passed: c.pbo <= 0.15 }, strategy: null, strategy_toml: null, execution: null, bundle_path: null, data_source: 'neon' });
      }
    } catch (e) { console.error('Neon query error:', e.message); }
    return res.status(404).json({ error: `Candidate ${req.params.candidateId} not found` });
  }
  const strategy = readTomlFile(path.join(bundlePath, 'strategy.toml'));
  const validation = readJsonFile(path.join(bundlePath, 'validation_summary.json')) || {};
  const provenance = readJsonFile(path.join(bundlePath, 'provenance.json')) || {};
  const execToml = readTomlFile(path.join(bundlePath, 'execution_config.toml'));
  let strategyTomlRaw = ''; try { strategyTomlRaw = fs.readFileSync(path.join(bundlePath, 'strategy.toml'), 'utf-8'); } catch (e) {}
  const displayName = generateDisplayName(strategy?.strategy ? strategy : { pipeline: strategy?.pipeline });
  const pipelineBlocks = (strategy?.pipeline || []).map(b => ({ block_type: b.type || 'unknown', block_id: b.block_id || 'unknown', enabled: b.enabled !== false, params: b.params || {} }));
  res.json({ candidate_id: req.params.candidateId, genome_hash: provenance.genome_hash || '', rank: 0, candidate_class: validation.gates_passed ? 'validated' : 'research', display_name: displayName, oos_sharpe_net: validation.oos_sharpe_net || 0, oos_sharpe_gross: validation.oos_sharpe_gross || 0, pbo: validation.pbo || 0, dsr: validation.dsr || 0, oos_cagr_net: validation.oos_cagr_net || 0, max_drawdown_net: validation.max_drawdown_net || 0, turnover_annual: validation.turnover_annual || 0, capacity_usd: validation.capacity_usd || null, stress_passed: validation.stress_passed || 0, stress_total: validation.stress_total || 0, gates_passed: validation.gates_passed || false, strategy: { id: strategy?.strategy?.id || req.params.candidateId, version: strategy?.strategy?.version || '1.0', description: strategy?.strategy?.description || '', author: strategy?.strategy?.author || 'SCG', pipeline: pipelineBlocks, rebalance: strategy?.rebalance || {}, constraints: strategy?.constraints || {} }, strategy_toml: strategyTomlRaw, execution: { delay_bars: execToml?.execution?.delay_bars || 1, bypass_for_debug: execToml?.execution?.bypass_for_debug || false, slippage: execToml?.execution?.slippage || { slippage_type: 'Constant', bps: 5 }, fees: execToml?.execution?.fees || { tier: 'B3Retail' } }, provenance: { git_sha: provenance.git_sha, dataset_hash: provenance.dataset_hash, config_hash: provenance.config_hash, run_id: provenance.run_id, campaign_id: provenance.campaign_id, seed: provenance.seed, created_at: provenance.created_at }, bundle_path: bundlePath });
});

router.get('/backtest/:candidateId', (req, res) => {
  const paths = [path.join(getArtifactsRoot(), 'backtests', req.params.candidateId, 'timeseries.csv'), path.join(getArtifactsRoot(), 'candidates', req.params.candidateId, 'backtest', 'timeseries.csv'), path.join(PROJECT_ROOT, 'output', 'backtests', req.params.candidateId, 'timeseries.csv')];
  for (const p of paths) {
    if (fs.existsSync(p)) {
      const ts = readCsvFile(p).map(r => ({ date: r.date, equity: parseFloat(r.equity) || 1, drawdown: parseFloat(r.drawdown) || 0, exposure: parseFloat(r.exposure) || null }));
      return res.json({ available: true, candidate_id: req.params.candidateId, timeseries: ts, backtest_path: path.dirname(p) });
    }
  }
  res.json({ available: false, candidate_id: req.params.candidateId, message: 'No backtest data found. Run replay to generate.', timeseries: [], backtest_path: null });
});

router.get('/candidate/:candidateId/simulated-equity', async (req, res) => {
  try {
    const result = await pool.query(`SELECT oos_cagr_net, oos_sharpe_net, max_drawdown_net, created_at FROM scg_candidates WHERE candidate_id = $1`, [req.params.candidateId]);
    if (result.rows.length === 0) return res.status(404).json({ error: `Candidate ${req.params.candidateId} not found` });
    const c = result.rows[0];
    const cagr = c.oos_cagr_net || 0.15, sharpe = c.oos_sharpe_net || 1.0, maxDD = Math.abs(c.max_drawdown_net) || 0.15;
    const annualVol = sharpe > 0.1 ? Math.abs(cagr) / sharpe : 0.20, dailyVol = annualVol / Math.sqrt(252), dailyReturn = cagr / 252;
    const numDays = parseInt(req.query.days) || 252;
    const startCapital = parseFloat(req.query.startCapital) || 100000;
    const timeseries = [];
    let equity = startCapital, peak = equity;
    const seed = req.params.candidateId.split('').reduce((a, b) => a + b.charCodeAt(0), 0);
    const seededRandom = () => { const x = Math.sin(seed + timeseries.length) * 10000; return x - Math.floor(x); };
    for (let i = 0; i < numDays; i++) {
      const date = new Date(Date.now() - (numDays - i) * 86400000).toISOString().slice(0, 10);
      if (i > 0) { const u1 = seededRandom(), u2 = seededRandom(); const z = Math.sqrt(-2 * Math.log(u1 + 0.0001)) * Math.cos(2 * Math.PI * u2); equity = equity * (1 + dailyReturn + dailyVol * z); if (equity > peak) peak = equity; if (equity < startCapital * 0.5) equity = startCapital * 0.5 + seededRandom() * startCapital * 0.1; }
      timeseries.push({ date, equity: Math.round(equity * 100) / 100, drawdown: peak > 0 ? Math.round((peak - equity) / peak * 10000) / 10000 : 0 });
    }
    res.json({ candidate_id: req.params.candidateId, data_source: 'simulated', simulation_params: { target_cagr: cagr, target_sharpe: sharpe, days: numDays, start_capital: startCapital }, timeseries });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

router.get('/candidate/:candidateId/pipeline', async (req, res) => {
  try {
    const result = await pool.query(`SELECT genome_hash, rank_in_run FROM scg_candidates WHERE candidate_id = $1`, [req.params.candidateId]);
    if (result.rows.length === 0) return res.status(404).json({ error: `Candidate ${req.params.candidateId} not found` });
    const hash = result.rows[0].genome_hash || req.params.candidateId;
    const hashNum = hash.split('').reduce((a, b) => a + b.charCodeAt(0), 0);
    const indicators = [{ name: 'RSI', params: { period: 14 + (hashNum % 10) } }, { name: 'MACD', params: { fast: 12, slow: 26, signal: 9 } }, { name: 'ATR', params: { period: 14 + (hashNum % 6) } }, { name: 'Bollinger', params: { period: 20, std: 2.0 } }];
    const signals = [{ name: 'MomentumBreakout', params: { lookback: 20 + (hashNum % 40) } }, { name: 'MeanReversion', params: { zscore: 2.0 } }, { name: 'TrendFollowing', params: { fast: 10, slow: 50 } }];
    const selectedInds = indicators.filter((_, i) => (hashNum + i) % 3 === 0).slice(0, 3);
    const pipeline = { version: '1.0.0', genome_hash: hash, blocks: [{ id: 'data', type: 'DataLoader', name: 'OHLCV', params: { asset: 'WINM25', timeframe: ['1m', '5m', '15m', '1h'][hashNum % 4] } }, { id: 'features', type: 'FeatureExtractor', name: 'Indicators', params: { indicators: selectedInds.map(i => i.name) }, children: selectedInds.map((ind, i) => ({ id: `ind_${i}`, type: 'Indicator', name: ind.name, params: ind.params })) }, { id: 'signal', type: 'SignalGenerator', ...signals[hashNum % signals.length] }, { id: 'sizing', type: 'PositionSizer', name: 'KellyFraction', params: { fraction: 0.25 } }, { id: 'risk', type: 'RiskManager', name: 'RiskStack', children: [{ name: 'StopLoss', params: { pct: 0.02 } }, { name: 'TakeProfit', params: { pct: 0.04 } }] }] };
    res.json({ candidate_id: req.params.candidateId, pipeline, data_source: 'generated' });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

router.get('/candidate/:candidateId/wfa', async (req, res) => {
  try {
    const result = await pool.query(`SELECT oos_sharpe_net, pbo, created_at FROM scg_candidates WHERE candidate_id = $1`, [req.params.candidateId]);
    if (result.rows.length === 0) return res.status(404).json({ error: `Candidate ${req.params.candidateId} not found` });
    const c = result.rows[0];
    const folds = 5;
    const hashNum = req.params.candidateId.split('').reduce((a, b) => a + b.charCodeAt(0), 0);
    const wfaFolds = Array.from({ length: folds }, (_, i) => ({ fold: i + 1, train_sharpe: (c.oos_sharpe_net || 1.5) * (0.9 + (((hashNum + i * 17) % 30) / 100)), oos_sharpe: (c.oos_sharpe_net || 1.2) * (0.85 + (((hashNum + i * 23) % 30) / 100)), is_oos: true, passed: true }));
    res.json({ candidate_id: req.params.candidateId, wfa_type: 'anchored', num_folds: folds, folds: wfaFolds, summary: { avg_oos_sharpe: c.oos_sharpe_net || 0, consistency: 0.85, passed: true }, data_source: 'simulated' });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

router.get('/candidate/:candidateId/stress', async (req, res) => {
  try {
    const result = await pool.query(`SELECT oos_sharpe_net, max_drawdown_net, stress_passed, stress_total FROM scg_candidates WHERE candidate_id = $1`, [req.params.candidateId]);
    if (result.rows.length === 0) return res.status(404).json({ error: `Candidate ${req.params.candidateId} not found` });
    const c = result.rows[0];
    const scenarios = [{ name: 'Flash Crash', description: 'Sudden 10% drop', severity: 'extreme', result: { max_dd: -0.15, recovery_days: 45, passed: true } }, { name: 'High Volatility', description: 'VIX spike to 40+', severity: 'high', result: { max_dd: -0.12, recovery_days: 30, passed: true } }, { name: 'Liquidity Crisis', description: 'Bid-ask spread 5x', severity: 'medium', result: { max_dd: -0.08, recovery_days: 20, passed: true } }];
    res.json({ candidate_id: req.params.candidateId, scenarios, summary: { passed: c.stress_passed || 0, total: c.stress_total || scenarios.length, pass_rate: c.stress_total ? (c.stress_passed / c.stress_total) : 1.0 }, data_source: 'simulated' });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

export default router;

