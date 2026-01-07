import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { pool, readJsonFile, getArtifactsRoot, setArtifactsRoot, PROJECT_ROOT } from '../db.js';
import { ompState } from '../state.js';

const router = Router();

router.get('/overview', async (req, res) => {
  try {
    const last24h = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
    const [hofResult, bestResult, totalResult, h24Result, campResult, genResult, recentResult] = await Promise.all([
      pool.query("SELECT COUNT(*) as count FROM scg_promotions WHERE promotion_class = 'hall_of_fame'"),
      pool.query(`SELECT MAX(oos_sharpe_net) as best_sharpe, MAX(oos_cagr_net) as best_cagr, MIN(max_drawdown_net) as worst_drawdown, AVG(oos_sharpe_net) as avg_sharpe FROM scg_candidates WHERE source_stage = 'stage_b' AND oos_sharpe_net IS NOT NULL AND oos_sharpe_net <= 10`),
      pool.query('SELECT COUNT(*) as count FROM scg_candidates'),
      pool.query('SELECT COUNT(*) as count FROM scg_candidates WHERE created_at > $1', [last24h]),
      pool.query(`SELECT COUNT(*) as total, SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed, SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running, SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed FROM scg_campaigns`),
      pool.query(`SELECT r.run_id, c.name as campaign_name, MAX(cand.oos_sharpe_net) as best_sharpe, AVG(cand.oos_sharpe_net) as mean_sharpe, COUNT(cand.candidate_id) as candidates_count, r.started_at FROM scg_runs r LEFT JOIN scg_campaigns c ON r.campaign_id = c.campaign_id LEFT JOIN scg_candidates cand ON cand.run_id = r.run_id WHERE cand.oos_sharpe_net IS NOT NULL AND cand.oos_sharpe_net <= 10 GROUP BY r.run_id, c.name, r.started_at ORDER BY r.started_at DESC LIMIT 50`),
      pool.query(`SELECT candidate_id, oos_sharpe_net, oos_cagr_net, max_drawdown_net, created_at FROM scg_candidates WHERE source_stage = 'stage_b' AND oos_sharpe_net IS NOT NULL AND oos_sharpe_net <= 10 ORDER BY created_at DESC LIMIT 100`)
    ]);
    
    const bs = bestResult.rows[0] || {};
    const cs = campResult.rows[0] || {};
    const genData = genResult.rows.map((r, i) => ({ generation: genResult.rows.length - i, bestSharpe: parseFloat(r.best_sharpe) || 0, meanSharpe: parseFloat(r.mean_sharpe) || 0, paretoSize: parseInt(r.candidates_count) || 0 })).reverse();
    let equity = 100000;
    const eqData = recentResult.rows.reverse().map(c => { equity *= (1 + (parseFloat(c.oos_cagr_net) || 0) / 252); return { time: new Date(c.created_at).toISOString().split('T')[0], value: equity }; });
    
    res.json({
      metrics: { totalReturn: parseFloat(bs.best_cagr) || 0, sharpeRatio: parseFloat(bs.best_sharpe) || 0, avgSharpeRatio: parseFloat(bs.avg_sharpe) || 0, maxDrawdown: parseFloat(bs.worst_drawdown) || 0, activeCandidates: parseInt(hofResult.rows[0].count) || 0, totalCandidates: parseInt(totalResult.rows[0].count) || 0, candidates24h: parseInt(h24Result.rows[0].count) || 0, currentGeneration: genData.length },
      campaigns: { total: parseInt(cs.total) || 0, completed: parseInt(cs.completed) || 0, running: parseInt(cs.running) || 0, failed: parseInt(cs.failed) || 0 },
      equityData: eqData.length > 0 ? eqData : Array.from({ length: 30 }, (_, i) => ({ time: new Date(Date.now() - (30 - i) * 86400000).toISOString().split('T')[0], value: 100000 })),
      generationData: genData, ompStatus: ompState.status, lastUpdated: new Date().toISOString()
    });
  } catch (e) { res.status(500).json({ error: 'Failed to fetch overview', details: e.message }); }
});

router.get('/index', async (req, res) => {
  const data = readJsonFile(path.join(getArtifactsRoot(), 'site', 'index.json'));
  if (data) return res.json(data);
  try {
    const result = await pool.query(`SELECT c.campaign_id, c.name, c.tag, c.status, c.created_at, COUNT(r.run_id) as runs_count FROM scg_campaigns c LEFT JOIN scg_runs r ON c.campaign_id = r.campaign_id GROUP BY c.campaign_id, c.name, c.tag, c.status, c.created_at ORDER BY c.created_at DESC`);
    res.json({ schema_version: '1.0', generated_at: new Date().toISOString(), campaigns: result.rows.map(c => ({ campaign_id: c.campaign_id, name: c.name, tag: c.tag || '', status: c.status || 'completed', runs_count: parseInt(c.runs_count) || 0, created_at: c.created_at })), data_source: 'neon' });
  } catch (e) { res.status(404).json({ error: 'Index not found' }); }
});

router.get('/campaigns', async (req, res) => {
  try {
    const result = await pool.query(`SELECT c.campaign_id, c.name, c.tag, c.status, c.owner, c.git_sha, c.git_branch, c.notes, c.created_at, COUNT(r.run_id) as runs_count, MAX(r.best_oos_sharpe_net) as best_sharpe FROM scg_campaigns c LEFT JOIN scg_runs r ON c.campaign_id = r.campaign_id GROUP BY c.campaign_id ORDER BY c.created_at DESC`);
    res.json({ campaigns: result.rows.map(c => ({ ...c, runs_count: parseInt(c.runs_count) || 0, best_sharpe: c.best_sharpe ? parseFloat(c.best_sharpe) : null })), count: result.rows.length, data_source: 'neon' });
  } catch (e) { res.status(500).json({ error: e.message, campaigns: [] }); }
});

router.get('/campaigns/:campaignId/runs', async (req, res) => {
  try {
    const result = await pool.query(`SELECT r.run_id, r.campaign_id, r.seed, r.status, r.started_at, r.completed_at, r.duration_secs, r.generations_completed, r.total_evaluations, r.best_oos_sharpe_net, COUNT(cand.candidate_id) as candidates_evaluated, SUM(CASE WHEN cand.gates_passed THEN 1 ELSE 0 END) as validated_count, MAX(cand.oos_cagr_net) as best_cagr FROM scg_runs r LEFT JOIN scg_candidates cand ON r.run_id = cand.run_id WHERE r.campaign_id = $1 GROUP BY r.run_id ORDER BY r.started_at DESC`, [req.params.campaignId]);
    res.json({ runs: result.rows.map(r => ({ ...r, candidates_evaluated: parseInt(r.candidates_evaluated) || 0, validated_count: parseInt(r.validated_count) || 0 })), count: result.rows.length });
  } catch (e) { res.status(500).json({ error: e.message, runs: [] }); }
});

router.get('/runs/recent', async (req, res) => {
  try {
    const result = await pool.query(`SELECT r.*, c.name as campaign_name, c.tag as campaign_tag, COUNT(cand.candidate_id) as candidates_count FROM scg_runs r LEFT JOIN scg_campaigns c ON r.campaign_id = c.campaign_id LEFT JOIN scg_candidates cand ON r.run_id = cand.run_id GROUP BY r.run_id, c.name, c.tag ORDER BY r.started_at DESC LIMIT $1`, [parseInt(req.query.limit) || 10]);
    res.json({ runs: result.rows.map(r => ({ ...r, candidates_count: parseInt(r.candidates_count) || 0 })), count: result.rows.length, data_source: 'neon' });
  } catch (e) { res.status(500).json({ error: e.message, runs: [] }); }
});

router.get('/campaign/:campaignId', async (req, res) => {
  const filePath = path.join(getArtifactsRoot(), 'site', `campaign_${req.params.campaignId}.json`);
  const data = readJsonFile(filePath);
  if (data) { const { runs, ...info } = data; return res.json({ schema_version: '1.0', campaign: info, runs: runs || [] }); }
  try {
    const cr = await pool.query(`SELECT * FROM scg_campaigns WHERE campaign_id = $1`, [req.params.campaignId]);
    if (cr.rows.length === 0) return res.status(404).json({ error: `Campaign ${req.params.campaignId} not found` });
    const rr = await pool.query(`SELECT r.*, COUNT(cand.candidate_id) as candidates_count, SUM(CASE WHEN cand.gates_passed THEN 1 ELSE 0 END) as validated_count FROM scg_runs r LEFT JOIN scg_candidates cand ON r.run_id = cand.run_id WHERE r.campaign_id = $1 GROUP BY r.run_id ORDER BY r.started_at DESC`, [req.params.campaignId]);
    res.json({ schema_version: '1.0', campaign: cr.rows[0], runs: rr.rows.map(r => ({ ...r, candidates_count: parseInt(r.candidates_count) || 0, validated_candidates_count: parseInt(r.validated_count) || 0 })), data_source: 'neon' });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

router.get('/run/:runId', async (req, res) => {
  try {
    const result = await pool.query(`SELECT r.*, c.name as campaign_name FROM scg_runs r LEFT JOIN scg_campaigns c ON r.campaign_id = c.campaign_id WHERE r.run_id = $1`, [req.params.runId]);
    if (result.rows.length === 0) return res.status(404).json({ error: `Run ${req.params.runId} not found` });
    const r = result.rows[0];
    res.json({ schema_version: '1.0', run: { run_id: r.run_id, campaign_id: r.campaign_id, seed: r.seed, status: r.status, started_at: r.started_at, completed_at: r.completed_at, duration_secs: r.duration_secs }, metrics: { total_evaluations: r.total_evaluations || 0, generations_completed: r.generations_completed || 0, best_oos_sharpe_net: r.best_oos_sharpe_net }, top_candidates: [], exports: {} });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

router.post('/set-root', (req, res) => {
  const { path: newPath } = req.body;
  let testPath = newPath;
  if (!fs.existsSync(path.join(testPath, 'site', 'index.json'))) testPath = path.join(newPath, 'artifacts');
  if (!fs.existsSync(path.join(testPath, 'site', 'index.json'))) return res.status(400).json({ error: `No artifacts found at ${newPath}` });
  setArtifactsRoot(testPath);
  res.json({ artifacts_root: testPath });
});

export default router;

