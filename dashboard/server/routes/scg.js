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

// Status do SCG overnight (roda fora do dashboard)
router.get('/scg/overnight-status', async (req, res) => {
  const { execSync, spawnSync } = await import('child_process');
  const pidFile = path.join(PROJECT_ROOT, 'logs', 'scg', 'scg.pid');
  const logDir = path.join(PROJECT_ROOT, 'logs', 'scg');
  const pendingDir = path.join(PROJECT_ROOT, 'output', 'scg', 'backtests', 'pending');
  
  let status = { running: false, pid: null, startTime: null, cpu: null, mem: null, lastGeneration: null, cleanups: 0, pendingFiles: 0, pendingSize: '0', errors: 0, latestLog: null };
  
  try {
    // Check PID file
    if (fs.existsSync(pidFile)) {
      const pid = fs.readFileSync(pidFile, 'utf8').trim();
      try {
        // Check if process is running
        process.kill(parseInt(pid), 0);
        status.running = true;
        status.pid = parseInt(pid);
        
        // Get CPU/MEM
        try {
          const ps = execSync(`ps -o %cpu,%mem,lstart -p ${pid} 2>/dev/null | tail -1`, { encoding: 'utf8' });
          const parts = ps.trim().split(/\s+/);
          status.cpu = parseFloat(parts[0]) || 0;
          status.mem = parseFloat(parts[1]) || 0;
          status.startTime = parts.slice(2).join(' ');
        } catch (e) {}
      } catch (e) {
        // Process not running
        status.running = false;
      }
    }
    
    // Check for any combiner process
    if (!status.running) {
      try {
        const pgrep = execSync('pgrep -f "combiner run" 2>/dev/null', { encoding: 'utf8' });
        const pids = pgrep.trim().split('\n').filter(p => p);
        if (pids.length > 0) {
          status.running = true;
          status.pid = parseInt(pids[0]);
        }
      } catch (e) {}
    }
    
    // Get latest log file
    if (fs.existsSync(logDir)) {
      const logs = fs.readdirSync(logDir).filter(f => f.startsWith('scg_') && f.endsWith('.log')).sort().reverse();
      if (logs.length > 0) {
        status.latestLog = logs[0];
        const logPath = path.join(logDir, logs[0]);
        const logContent = fs.readFileSync(logPath, 'utf8');
        
        // Parse last generation
        const genMatches = logContent.match(/Gen (\d+) ULTRA/g);
        if (genMatches && genMatches.length > 0) {
          const lastMatch = genMatches[genMatches.length - 1].match(/Gen (\d+)/);
          status.lastGeneration = parseInt(lastMatch[1]);
        }
        
        // Count cleanups
        status.cleanups = (logContent.match(/Incremental cleanup/g) || []).length;
        
        // Count errors
        status.errors = (logContent.match(/ERROR/g) || []).length;
      }
    }
    
    // Pending files
    if (fs.existsSync(pendingDir)) {
      try {
        const files = fs.readdirSync(pendingDir).filter(f => f.endsWith('.obfs'));
        status.pendingFiles = files.length;
        const du = execSync(`du -sh "${pendingDir}" 2>/dev/null`, { encoding: 'utf8' });
        status.pendingSize = du.split('\t')[0];
      } catch (e) {}
    }
    
    res.json(status);
  } catch (e) {
    res.status(500).json({ error: e.message, ...status });
  }
});

// Processo SCG persistente (sobrevive ao restart do server)
let scgOvernightProcess = null;
let scgOvernightRunId = null;

// Start SCG overnight from dashboard (direto via Node.js, sem scripts)
router.post('/scg/overnight-start', async (req, res) => {
  const { execSync } = await import('child_process');
  const { config = 'maxpower', ultra = true } = req.body;
  
  // Mapear config para arquivo
  const configMap = {
    'maxpower': 'scg_maxpower.toml',
    '1h_moderado': 'scg_1h_moderado.toml',
    '4h_agressivo': 'scg_4h_agressivo.toml',
  };
  const configFile = configMap[config] || `scg_${config}.toml`;
  const configPath = path.join(PROJECT_ROOT, 'configs', configFile);
  const combinerPath = path.join(PROJECT_ROOT, 'target', 'release', 'combiner');
  const logDir = path.join(PROJECT_ROOT, 'logs', 'scg');
  const pidFile = path.join(logDir, 'scg.pid');
  
  // Verificar binário
  if (!fs.existsSync(combinerPath)) {
    return res.status(500).json({ error: 'Combiner not found. Run: cargo build --release --bin combiner' });
  }
  if (!fs.existsSync(configPath)) {
    return res.status(400).json({ error: `Config not found: ${configPath}` });
  }
  
  // Verificar se já está rodando
  try {
    const pgrep = execSync('pgrep -f "combiner run"', { encoding: 'utf8' });
    if (pgrep.trim()) {
      return res.status(400).json({ error: 'SCG already running', pids: pgrep.trim().split('\n') });
    }
  } catch (e) { /* não está rodando */ }
  
  // Criar diretório de logs
  fs.mkdirSync(logDir, { recursive: true });
  
  // Timestamp para log
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const logFile = path.join(logDir, `scg_${timestamp}.log`);
  
  // Args do combiner
  const args = ['run', '--config', configPath];
  if (ultra) args.push('--ultra');
  
  // Iniciar processo detached
  const logFd = fs.openSync(logFile, 'a');
  const proc = spawn(combinerPath, args, {
    cwd: PROJECT_ROOT,
    detached: true,
    stdio: ['ignore', logFd, logFd],
    env: {
      ...process.env,
      RUST_LOG: 'combiner_engine=info,combiner_runner=info',
      BACKTEST_CLI_PATH: path.join(PROJECT_ROOT, 'target/release/backtest'),
    }
  });
  
  // Salvar PID
  fs.writeFileSync(pidFile, proc.pid.toString());
  proc.unref();
  fs.closeSync(logFd);
  
  scgOvernightProcess = proc;
  scgOvernightRunId = `overnight_${timestamp}`;
  
  res.json({ 
    status: 'started', 
    pid: proc.pid, 
    config: configFile,
    ultra,
    logFile: `logs/scg/scg_${timestamp}.log`,
    message: 'SCG started in background. Close browser/terminal - it will keep running.'
  });
});

// Stop SCG overnight (direto via Node.js)
router.post('/scg/overnight-stop', async (req, res) => {
  const { execSync } = await import('child_process');
  const pidFile = path.join(PROJECT_ROOT, 'logs', 'scg', 'scg.pid');
  
  let killed = [];
  
  // Tentar pelo PID file
  if (fs.existsSync(pidFile)) {
    const pid = fs.readFileSync(pidFile, 'utf8').trim();
    try {
      process.kill(parseInt(pid), 'SIGTERM');
      killed.push(parseInt(pid));
      fs.unlinkSync(pidFile);
    } catch (e) {}
  }
  
  // Matar qualquer combiner run
  try {
    const pgrep = execSync('pgrep -f "combiner run"', { encoding: 'utf8' });
    const pids = pgrep.trim().split('\n').filter(p => p);
    for (const pid of pids) {
      try {
        process.kill(parseInt(pid), 'SIGTERM');
        killed.push(parseInt(pid));
      } catch (e) {}
    }
  } catch (e) {}
  
  scgOvernightProcess = null;
  scgOvernightRunId = null;
  
  res.json({ status: 'stopped', killed, message: `Stopped ${killed.length} process(es)` });
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

// =========================================================================
// GLOBAL HALL OF FAME - Permanent, all-time best strategies
// =========================================================================

/**
 * GET /api/scg/hall-of-fame
 * Returns the top N all-time best strategies, permanently stored.
 * 
 * Query params:
 *   - limit: number (default 50, max 100)
 *   - market: string (default 'BR')
 */
router.get('/scg/hall-of-fame', async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 50, 100);
    const market = req.query.market || 'BR';
    
    const result = await pool.query(`
      SELECT 
        id,
        genome_hash,
        candidate_id,
        oos_sharpe_net,
        oos_cagr_net,
        max_drawdown_net,
        pbo,
        dsr,
        stress_passed,
        stress_total,
        gates_passed,
        run_id,
        campaign_id,
        promoted_at,
        git_sha,
        market,
        global_rank
      FROM hall_of_fame
      WHERE market = $1
      ORDER BY oos_sharpe_net DESC
      LIMIT $2
    `, [market, limit]);
    
    // Get total count for pagination info
    const countResult = await pool.query(
      'SELECT COUNT(*) as total FROM hall_of_fame WHERE market = $1',
      [market]
    );
    
    res.json({
      strategies: result.rows.map((r, idx) => ({
        rank: r.global_rank || idx + 1,
        genome_hash: r.genome_hash,
        candidate_id: r.candidate_id,
        oos_sharpe: r.oos_sharpe_net,
        oos_cagr: r.oos_cagr_net,
        max_drawdown: r.max_drawdown_net,
        pbo: r.pbo,
        dsr: r.dsr,
        stress_passed: r.stress_passed,
        stress_total: r.stress_total,
        gates_passed: r.gates_passed,
        run_id: r.run_id,
        campaign_id: r.campaign_id,
        promoted_at: r.promoted_at,
        git_sha: r.git_sha,
        market: r.market
      })),
      count: result.rows.length,
      total: parseInt(countResult.rows[0].total),
      market,
      source: 'hall_of_fame'
    });
  } catch (e) {
    // If table doesn't exist yet, return empty
    if (e.message.includes('does not exist')) {
      res.json({ 
        strategies: [], 
        count: 0, 
        total: 0, 
        market: req.query.market || 'BR',
        source: 'hall_of_fame',
        error: 'Hall of Fame table not yet created. Run migration first.'
      });
    } else {
      res.status(500).json({ error: e.message, strategies: [] });
    }
  }
});

/**
 * GET /api/scg/hall-of-fame/browse
 * Returns Hall of Fame strategies formatted for CandidateSelector component.
 * Same format as /candidates/recent for seamless integration.
 */
router.get('/scg/hall-of-fame/browse', async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 50, 100);
    const market = req.query.market || 'BR';
    
    const result = await pool.query(`
      SELECT h.*, camp.name as campaign_name
      FROM hall_of_fame h
      LEFT JOIN scg_runs r ON h.run_id = r.run_id
      LEFT JOIN scg_campaigns camp ON r.campaign_id = camp.campaign_id
      WHERE h.market = $1
      ORDER BY h.oos_sharpe_net DESC
      LIMIT $2
    `, [market, limit]);
    
    res.json({ 
      candidates: result.rows.map((h, idx) => ({ 
        candidate_id: h.candidate_id, 
        genome_hash: h.genome_hash || '', 
        rank: h.global_rank || idx + 1, 
        display_name: `🏆 #${h.global_rank || idx + 1} | ${(h.genome_hash || '').slice(-8)}`, 
        oos_sharpe_net: h.oos_sharpe_net || 0, 
        oos_cagr_net: h.oos_cagr_net || 0, 
        max_drawdown_net: h.max_drawdown_net, 
        pbo: h.pbo || 0, 
        dsr: h.dsr || 0, 
        gates_passed: h.gates_passed || false, 
        run_id: h.run_id, 
        campaign_name: h.campaign_name || 'Hall of Fame',
        source_stage: 'B',
        candidate_class: 'validated',
        stress_passed: h.stress_passed || 3,
        stress_total: h.stress_total || 3,
        data_source: 'hall_of_fame',
        promoted_at: h.promoted_at,
        market: h.market
      })), 
      count: result.rows.length,
      source: 'hall_of_fame'
    });
  } catch (e) {
    if (e.message.includes('does not exist')) {
      res.json({ candidates: [], count: 0, source: 'hall_of_fame', error: 'Table not created' });
    } else {
      res.status(500).json({ error: e.message, candidates: [] });
    }
  }
});

/**
 * GET /api/scg/hall-of-fame/stats
 * Returns summary statistics for the Hall of Fame.
 */
router.get('/scg/hall-of-fame/stats', async (req, res) => {
  try {
    const market = req.query.market || 'BR';
    
    const result = await pool.query(`
      SELECT 
        COUNT(*) as total_strategies,
        MAX(oos_sharpe_net) as best_sharpe,
        MIN(oos_sharpe_net) as worst_sharpe,
        AVG(oos_sharpe_net) as avg_sharpe,
        MAX(oos_cagr_net) as best_cagr,
        AVG(pbo) as avg_pbo,
        AVG(dsr) as avg_dsr,
        MIN(promoted_at) as oldest_entry,
        MAX(promoted_at) as newest_entry
      FROM hall_of_fame
      WHERE market = $1
    `, [market]);
    
    const stats = result.rows[0];
    
    res.json({
      market,
      total_strategies: parseInt(stats.total_strategies) || 0,
      best_sharpe: parseFloat(stats.best_sharpe) || 0,
      worst_sharpe: parseFloat(stats.worst_sharpe) || 0,
      avg_sharpe: parseFloat(stats.avg_sharpe) || 0,
      best_cagr: parseFloat(stats.best_cagr) || 0,
      avg_pbo: parseFloat(stats.avg_pbo) || 0,
      avg_dsr: parseFloat(stats.avg_dsr) || 0,
      oldest_entry: stats.oldest_entry,
      newest_entry: stats.newest_entry
    });
  } catch (e) {
    if (e.message.includes('does not exist')) {
      res.json({ 
        market: req.query.market || 'BR',
        total_strategies: 0,
        error: 'Hall of Fame table not yet created'
      });
    } else {
      res.status(500).json({ error: e.message });
    }
  }
});

/**
 * GET /api/scg/hall-of-fame/:genomeHash
 * Returns details for a specific strategy by genome hash.
 */
router.get('/scg/hall-of-fame/:genomeHash', async (req, res) => {
  try {
    const result = await pool.query(`
      SELECT 
        id,
        genome_hash,
        candidate_id,
        oos_sharpe_net,
        oos_cagr_net,
        max_drawdown_net,
        pbo,
        dsr,
        stress_passed,
        stress_total,
        gates_passed,
        run_id,
        campaign_id,
        promoted_at,
        git_sha,
        market,
        strategy_toml,
        genome_json,
        global_rank
      FROM hall_of_fame
      WHERE genome_hash = $1
    `, [req.params.genomeHash]);
    
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Strategy not found in Hall of Fame' });
    }
    
    const r = result.rows[0];
    res.json({
      rank: r.global_rank,
      genome_hash: r.genome_hash,
      candidate_id: r.candidate_id,
      oos_sharpe: r.oos_sharpe_net,
      oos_cagr: r.oos_cagr_net,
      max_drawdown: r.max_drawdown_net,
      pbo: r.pbo,
      dsr: r.dsr,
      stress_passed: r.stress_passed,
      stress_total: r.stress_total,
      gates_passed: r.gates_passed,
      run_id: r.run_id,
      campaign_id: r.campaign_id,
      promoted_at: r.promoted_at,
      git_sha: r.git_sha,
      market: r.market,
      strategy_toml: r.strategy_toml,
      genome_json: r.genome_json
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Global Hall of Fame status - diagnostic endpoint
router.get('/hof/status', async (req, res) => {
  try {
    // Get counts by market
    const countResult = await pool.query(`
      SELECT market, COUNT(*) as count, 
             MAX(oos_sharpe_net) as top_sharpe,
             MIN(oos_sharpe_net) as min_sharpe
      FROM hall_of_fame
      GROUP BY market
    `);
    
    const markets = {};
    for (const row of countResult.rows) {
      markets[row.market] = {
        count: parseInt(row.count),
        topSharpe: parseFloat(row.top_sharpe),
        minSharpe: parseFloat(row.min_sharpe),
        threshold: parseFloat(row.min_sharpe) // Entry threshold
      };
    }
    
    // Get total candidates in staging tables
    const candidatesResult = await pool.query(`
      SELECT candidate_class, source_stage, COUNT(*) as count
      FROM scg_candidates
      GROUP BY candidate_class, source_stage
    `);
    
    const candidates = {};
    for (const row of candidatesResult.rows) {
      const key = `${row.candidate_class}_${row.source_stage || 'unknown'}`;
      candidates[key] = parseInt(row.count);
    }
    
    res.json({
      connected: true,
      globalHallOfFame: markets,
      candidates,
      timestamp: new Date().toISOString()
    });
  } catch (e) {
    res.status(500).json({ 
      connected: false, 
      error: e.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Unified Hall of Fame - combines global DB + local files
router.get('/hof/unified', async (req, res) => {
  try {
    const { getUnifiedHallOfFame } = await import('../services/hofSync.js');
    const market = req.query.market || 'BR';
    const strategies = await getUnifiedHallOfFame(market);
    res.json({ 
      count: strategies.length,
      market,
      strategies 
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

export default router;

