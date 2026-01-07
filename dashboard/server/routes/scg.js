import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { pool, PROJECT_ROOT, getArtifactsRoot } from '../db.js';
import { scgRuns, broadcastSSE } from '../state.js';
import { DATABASE_URL } from '../db.js';

const router = Router();

router.post('/scg/start', (req, res) => {
  const { maxRuntimeSeconds = 30, campaignConfig } = req.body;
  const runId = `run_${Date.now().toString(36)}`;
  const configPath = campaignConfig || path.join(PROJECT_ROOT, 'configs', 'campaigns', 'scg_quick_test.toml');
  const combinerPath = path.join(PROJECT_ROOT, 'target', 'release', 'combiner');
  
  if (!fs.existsSync(combinerPath)) return res.status(500).json({ error: 'Combiner binary not found. Run: cargo build --release -p combiner_cli' });
  if (!fs.existsSync(configPath)) return res.status(400).json({ error: `Config not found: ${configPath}` });
  
  const scgProcess = spawn(combinerPath, ['factory', 'run', '--campaign', configPath], { cwd: PROJECT_ROOT, env: { ...process.env, RUST_LOG: 'combiner=info', NEON_DATABASE_URL: process.env.DATABASE_URL || 'postgresql://neondb_owner:npg_HyU68iqJScrQ@ep-wild-cell-af18q8jx-pooler.c-2.us-west-2.aws.neon.tech/neondb?channel_binding=require&sslmode=require', BACKTEST_CLI_PATH: process.env.BACKTEST_CLI_PATH || path.join(PROJECT_ROOT, 'target/release/backtest') } });
  const runState = { runId, status: 'running', startTime: Date.now(), maxRuntimeSeconds, output: [], error: null, process: scgProcess };
  scgRuns.set(runId, runState);
  
  scgProcess.stdout.on('data', (data) => { runState.output.push(data.toString()); broadcastSSE('scg-progress', { run_id: runId, status: 'running', elapsed_secs: Math.floor((Date.now() - runState.startTime) / 1000), latest_log: runState.output.slice(-1)[0] }); });
  scgProcess.stderr.on('data', (data) => { runState.output.push(data.toString()); });
  scgProcess.on('close', (code) => { runState.status = code === 0 ? 'completed' : 'failed'; runState.exitCode = code; runState.endTime = Date.now(); broadcastSSE('scg-progress', { run_id: runId, status: runState.status, percent_complete: 100, exit_code: code }); });
  scgProcess.on('error', (err) => { runState.status = 'failed'; runState.error = err.message; broadcastSSE('scg-progress', { run_id: runId, status: 'failed', error_message: err.message }); });
  
  res.json({ runId, status: 'started' });
});

router.get('/scg/progress/:runId', (req, res) => {
  const runState = scgRuns.get(req.params.runId);
  if (!runState) return res.status(404).json({ error: `Run ${req.params.runId} not found` });
  const elapsedSeconds = Math.floor((Date.now() - runState.startTime) / 1000);
  let currentGeneration = 0, bestSharpe = null;
  for (const line of runState.output) {
    const genMatch = line.match(/Generation\s+(\d+)/i); if (genMatch) currentGeneration = parseInt(genMatch[1]);
    const sharpeMatch = line.match(/Best Sharpe[:\s]+(\d+\.?\d*)/i); if (sharpeMatch) bestSharpe = parseFloat(sharpeMatch[1]);
  }
  res.json({ run_id: req.params.runId, status: runState.status, percent_complete: Math.min((elapsedSeconds / runState.maxRuntimeSeconds) * 100, 100), elapsed_secs: elapsedSeconds, current_generation: currentGeneration, best_sharpe: bestSharpe, latest_log: runState.output.slice(-3).join('\n'), error_message: runState.error });
});

router.post('/scg/stop/:runId', (req, res) => {
  const runState = scgRuns.get(req.params.runId);
  if (!runState) return res.status(404).json({ error: `Run ${req.params.runId} not found` });
  if (runState.process) { runState.process.kill('SIGTERM'); runState.status = 'stopped'; }
  res.json({ runId: req.params.runId, status: 'stopped' });
});

router.get('/scg/active-runs', (req, res) => {
  const runs = [];
  for (const [id, state] of scgRuns) { if (state.status === 'running') runs.push({ run_id: id, started_at: new Date(state.startTime).toISOString(), elapsed_secs: Math.floor((Date.now() - state.startTime) / 1000) }); }
  res.json({ runs, count: runs.length });
});

router.get('/cockpit-candidates/:runId', async (req, res) => {
  try {
    const result = await pool.query(`SELECT * FROM scg_candidates WHERE run_id = $1 ORDER BY rank_in_run ASC LIMIT 50`, [req.params.runId]);
    if (result.rows.length > 0) {
      res.json({ candidates: result.rows.map(c => ({ candidate_id: c.candidate_id, rank: c.rank_in_run, oos_sharpe: c.oos_sharpe_net, oos_cagr: c.oos_cagr_net, max_dd: c.max_drawdown_net, pbo: c.pbo, gates_passed: c.gates_passed })), count: result.rows.length, source: 'neon' });
    } else {
      res.json({ candidates: [], count: 0, source: 'none' });
    }
  } catch (e) { res.status(500).json({ error: e.message, candidates: [] }); }
});

export default router;

