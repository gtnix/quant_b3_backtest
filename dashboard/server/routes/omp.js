import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { execSync, spawn } from 'child_process';
import { pool, PROJECT_ROOT, getArtifactsRoot, getWorkspaceRoot, readTomlFile } from '../db.js';
import { ompState, loadOmpConfig, loadCampaignQueue, saveCampaignQueue, broadcastSSE, addActivityLog, getOmpLoopInterval, setOmpLoopInterval } from '../state.js';
import { DATABASE_URL } from '../db.js';
import { runSync, scanLocalStrategies, scanLocalStrategiesQuick } from '../services/hofSync.js';

const router = Router();

// Disk I/O history for pace calculation
const diskIoHistory = [];
const DISK_HISTORY_MAX = 60; // Keep 60 samples (1 min at 1s interval)
const MIN_DISK_FREE_GB = 1; // Stop if less than 1GB free

async function checkResources() {
  const cpus = os.cpus(), totalMem = os.totalmem(), freeMem = os.freemem();
  const loadAvg = os.loadavg()[0], cpuCount = cpus.length;
  const cpuUsage = Math.min((loadAvg / cpuCount) * 100, 100);
  const memoryUsagePct = ((totalMem - freeMem) / totalMem) * 100;
  const memoryAvailableMb = freeMem / (1024 * 1024);
  
  // Get disk free space
  let diskFreeGb = 100;
  let diskTotalGb = 100;
  try { 
    const df = execSync(`df -BG ${getArtifactsRoot()} | tail -1`, { encoding: 'utf-8' }); 
    const parts = df.trim().split(/\s+/);
    diskTotalGb = parseFloat(parts[1]?.replace('G', '')) || 100;
    diskFreeGb = parseFloat(parts[3]?.replace('G', '')) || 100; 
  } catch (e) {}
  
  // Track disk usage over time for pace calculation
  const now = Date.now();
  const diskUsedGb = diskTotalGb - diskFreeGb;
  diskIoHistory.push({ timestamp: now, usedGb: diskUsedGb });
  if (diskIoHistory.length > DISK_HISTORY_MAX) diskIoHistory.shift();
  
  // Calculate write rate (MB/s) from history
  let writeRateMbPerSec = 0;
  let writeAcceleration = 0;
  if (diskIoHistory.length >= 2) {
    const oldest = diskIoHistory[0];
    const newest = diskIoHistory[diskIoHistory.length - 1];
    const timeDiffSec = (newest.timestamp - oldest.timestamp) / 1000;
    if (timeDiffSec > 0) {
      const gbWritten = newest.usedGb - oldest.usedGb;
      writeRateMbPerSec = Math.max(0, (gbWritten * 1024) / timeDiffSec);
    }
    // Calculate acceleration (change in rate)
    if (diskIoHistory.length >= 10) {
      const mid = diskIoHistory[Math.floor(diskIoHistory.length / 2)];
      const t1 = (mid.timestamp - oldest.timestamp) / 1000;
      const t2 = (newest.timestamp - mid.timestamp) / 1000;
      if (t1 > 0 && t2 > 0) {
        const rate1 = ((mid.usedGb - oldest.usedGb) * 1024) / t1;
        const rate2 = ((newest.usedGb - mid.usedGb) * 1024) / t2;
        writeAcceleration = (rate2 - rate1) / ((t1 + t2) / 2);
      }
    }
  }
  
  // Estimate time until 1GB free (in hours)
  const availableToWrite = diskFreeGb - MIN_DISK_FREE_GB;
  let estimatedTimeToLimitHours = Infinity;
  if (writeRateMbPerSec > 0.001) {
    const mbToWrite = availableToWrite * 1024;
    estimatedTimeToLimitHours = mbToWrite / writeRateMbPerSec / 3600;
  }
  
  const config = ompState.config?.resource_limits || {};
  // Block start if < 2GB free OR estimated time < 30 min
  const hasDiskSpace = diskFreeGb >= 2 && (estimatedTimeToLimitHours > 0.5 || writeRateMbPerSec < 0.001);
  const canStart = cpuUsage < (config.max_cpu_util_pct || 90) && 
                   memoryAvailableMb > (config.min_mem_available_mb || 400) && 
                   hasDiskSpace && 
                   !ompState.currentCampaign;
  
  // Should we auto-stop? (< 1GB free or < 5 min estimated)
  const shouldAutoStop = diskFreeGb < MIN_DISK_FREE_GB || 
                         (writeRateMbPerSec > 0.001 && estimatedTimeToLimitHours < 0.083);
  
  ompState.resources = { 
    cpuUsage: Math.round(cpuUsage * 10) / 10, 
    memoryUsagePct: Math.round(memoryUsagePct * 10) / 10, 
    memoryAvailableMb: Math.round(memoryAvailableMb), 
    diskFreeGb: Math.round(diskFreeGb * 100) / 100,
    diskTotalGb: Math.round(diskTotalGb * 100) / 100,
    diskFreePct: Math.round((diskFreeGb / diskTotalGb) * 100),
    writeRateMbPerSec: Math.round(writeRateMbPerSec * 1000) / 1000,
    writeAcceleration: Math.round(writeAcceleration * 10000) / 10000,
    estimatedTimeToLimitHours: estimatedTimeToLimitHours === Infinity ? null : Math.round(estimatedTimeToLimitHours * 100) / 100,
    shouldAutoStop,
    canStartCampaign: canStart 
  };
  
  // Auto-stop if disk critically low
  if (shouldAutoStop && ompState.status === 'running') {
    console.log('\n⚠️ [OMP] Auto-stopping: disk space critical');
    addActivityLog('warning', `Auto-stop: disco < ${MIN_DISK_FREE_GB}GB livre`);
    // Trigger stop
    if (ompState.currentCampaign?.process) {
      ompState.currentCampaign.process.kill('SIGTERM');
    }
    ompState.status = 'offline';
    ompState.currentCampaign = null;
    broadcastSSE('omp-stopped', { reason: 'disk_low', stoppedAt: new Date().toISOString() });
  }
  
  return ompState.resources;
}

async function startQueuedCampaign(campaign) {
  const configPath = path.join(PROJECT_ROOT, campaign.config_path);
  const combinerPath = path.join(PROJECT_ROOT, 'target', 'release', 'combiner');
  if (!fs.existsSync(combinerPath)) { broadcastSSE('omp-error', { type: 'binary-missing', campaignId: campaign.id }); ompState.stats.campaignsFailed++; return null; }
  if (!fs.existsSync(configPath)) { broadcastSSE('omp-error', { type: 'config-missing', campaignId: campaign.id }); ompState.stats.campaignsFailed++; return null; }
  const runId = `run_${Date.now().toString(36)}`;
  console.log(`\n🚀 [OMP] Starting campaign: ${campaign.name} (${campaign.id})`);
  const scgProcess = spawn(combinerPath, ['factory', 'run', '--campaign', configPath], { cwd: PROJECT_ROOT, env: { ...process.env, RUST_LOG: 'combiner=info', NEON_DATABASE_URL: process.env.DATABASE_URL || 'postgresql://neondb_owner:npg_HyU68iqJScrQ@ep-wild-cell-af18q8jx-pooler.c-2.us-west-2.aws.neon.tech/neondb?channel_binding=require&sslmode=require', BACKTEST_CLI_PATH: process.env.BACKTEST_CLI_PATH || path.join(PROJECT_ROOT, 'target/release/backtest') } });
  const campaignState = { campaignId: campaign.id, campaignName: campaign.name, runId, market: campaign.market || 'br', status: 'running', startTime: Date.now(), output: [], process: scgProcess, currentGeneration: 0, bestSharpe: null, candidatesEvaluated: 0 };
  scgProcess.stdout.on('data', (data) => { campaignState.output.push(data.toString()); const line = data.toString(); const genMatch = line.match(/Generation\s+(\d+)/i); if (genMatch) campaignState.currentGeneration = parseInt(genMatch[1]); const sharpeMatch = line.match(/Best Sharpe[:\s]+(\d+\.?\d*)/i); if (sharpeMatch) campaignState.bestSharpe = parseFloat(sharpeMatch[1]); });
  scgProcess.stderr.on('data', (data) => { campaignState.output.push(data.toString()); });
  scgProcess.on('close', (code) => { campaignState.status = code === 0 ? 'completed' : 'failed'; campaignState.endTime = Date.now(); if (code === 0) ompState.stats.campaignsCompleted++; else ompState.stats.campaignsFailed++; ompState.currentCampaign = null; broadcastSSE('omp-campaign-completed', { campaignId: campaign.id, runId, status: campaignState.status, duration: (campaignState.endTime - campaignState.startTime) / 1000 }); });
  scgProcess.on('error', () => { campaignState.status = 'failed'; ompState.currentCampaign = null; });
  return campaignState;
}

async function ompLoop() {
  if (ompState.status !== 'running') return;
  ompState.loopCount++; ompState.lastLoop = new Date().toISOString();
  await checkResources();
  if (!ompState.currentCampaign && ompState.resources.canStartCampaign) {
    const queue = loadCampaignQueue();
    const next = queue.campaigns?.find(c => c.enabled);
    if (next) { ompState.currentCampaign = await startQueuedCampaign(next); addActivityLog('info', `Started campaign: ${next.name}`); }
  }
  broadcastSSE('omp-heartbeat', { loopCount: ompState.loopCount, resources: ompState.resources, currentCampaign: ompState.currentCampaign?.campaignName || null });
}

function getOmpStatus() {
  const currentCampaign = ompState.currentCampaign ? { 
    campaignId: ompState.currentCampaign.campaignId, 
    campaignName: ompState.currentCampaign.campaignName, 
    runId: ompState.currentCampaign.runId, 
    market: ompState.currentCampaign.market, 
    status: ompState.currentCampaign.status, 
    elapsedSecs: Math.floor((Date.now() - ompState.currentCampaign.startTime) / 1000), 
    currentGeneration: ompState.currentCampaign.currentGeneration, 
    bestSharpe: ompState.currentCampaign.bestSharpe,
    candidatesEvaluated: ompState.currentCampaign.candidatesEvaluated || 0
  } : null;
  
  return { 
    status: ompState.status, 
    startedAt: ompState.startedAt, 
    lastLoop: ompState.lastLoop, 
    loopCount: ompState.loopCount, 
    queueLength: ompState.queueLength, 
    lastPromotion: ompState.lastPromotion, 
    currentCampaign, 
    resources: ompState.resources, 
    stats: ompState.stats, 
    config: ompState.config ? { loopIntervalSecs: ompState.config.orchestrator?.loop_interval_secs || 30 } : null 
  };
}

router.get('/omp/status', async (req, res) => { 
  await checkResources(); // Always update resources on status fetch
  res.json(getOmpStatus()); 
});

router.post('/omp/start', (req, res) => {
  if (ompState.status === 'running') return res.status(400).json({ error: 'OMP is already running' });
  loadOmpConfig(); loadCampaignQueue();
  ompState.status = 'running'; ompState.startedAt = new Date().toISOString(); ompState.loopCount = 0;
  const intervalMs = (ompState.config?.orchestrator?.loop_interval_secs || 30) * 1000;
  setOmpLoopInterval(setInterval(ompLoop, intervalMs));
  ompLoop();
  console.log('\n🟢 [OMP] Mining started'); broadcastSSE('omp-started', { startedAt: ompState.startedAt });
  res.json({ status: 'started', startedAt: ompState.startedAt });
});

router.post('/omp/stop', (req, res) => {
  if (ompState.status === 'offline') return res.status(400).json({ error: 'OMP is not running' });
  const interval = getOmpLoopInterval(); if (interval) { clearInterval(interval); setOmpLoopInterval(null); }
  if (ompState.currentCampaign?.process) ompState.currentCampaign.process.kill('SIGTERM');
  ompState.status = 'offline'; ompState.currentCampaign = null;
  console.log('\n🔴 [OMP] Mining stopped'); broadcastSSE('omp-stopped', { stoppedAt: new Date().toISOString() });
  res.json({ status: 'stopped' });
});

router.post('/omp/pause', (req, res) => {
  if (ompState.status !== 'running') return res.status(400).json({ error: 'OMP is not running' });
  ompState.status = 'paused'; broadcastSSE('omp-paused', { pausedAt: new Date().toISOString() });
  res.json({ status: 'paused' });
});

router.post('/omp/resume', (req, res) => {
  if (ompState.status !== 'paused') return res.status(400).json({ error: 'OMP is not paused' });
  ompState.status = 'running'; broadcastSSE('omp-resumed', { resumedAt: new Date().toISOString() });
  res.json({ status: 'running' });
});

router.post('/omp/cleanup', async (req, res) => {
  if (ompState.status === 'running') return res.status(400).json({ error: 'Cannot cleanup while mining is running. Stop first.' });
  const results = { folders: false, database: false, errors: [] };
  try { const outputPath = path.join(PROJECT_ROOT, 'output', 'scg'); if (fs.existsSync(outputPath)) { fs.readdirSync(outputPath).forEach(f => fs.rmSync(path.join(outputPath, f), { recursive: true, force: true })); results.folders = true; } } catch (e) { results.errors.push(`Output folder: ${e.message}`); }
  try { await pool.query('TRUNCATE scg_candidates, scg_promotions, scg_runs, scg_campaigns RESTART IDENTITY CASCADE'); results.database = true; } catch (e) { results.errors.push(`Database: ${e.message}`); }
  ompState.stats = { candidatesGenerated: 0, promotions: 0, campaignsCompleted: 0, campaignsFailed: 0, backtestsExecuted: 0 }; ompState.activityLog = [];
  broadcastSSE('omp-cleanup', { results, timestamp: new Date().toISOString() });
  res.json({ success: results.folders && results.database, results, message: results.errors.length > 0 ? 'Cleanup completed with errors' : 'Cleanup completed successfully' });
});

router.get('/omp/queue', (req, res) => { res.json(loadCampaignQueue()); });

router.post('/omp/queue', (req, res) => {
  const { name, config_path, market, priority, enabled, repeat, tags } = req.body;
  if (!name || !config_path) return res.status(400).json({ error: 'name and config_path are required' });
  const queue = loadCampaignQueue();
  const campaign = { id: `camp_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`, name, config_path, market: market || 'br', priority: priority || queue.campaigns.length + 1, enabled: enabled !== false, repeat: repeat || false, tags: tags || [], created_at: new Date().toISOString() };
  queue.campaigns.push(campaign); saveCampaignQueue(queue);
  broadcastSSE('omp-queue-updated', { action: 'add', campaign });
  res.json(campaign);
});

router.patch('/omp/queue/:id', (req, res) => {
  const queue = loadCampaignQueue();
  const index = queue.campaigns.findIndex(c => c.id === req.params.id);
  if (index === -1) return res.status(404).json({ error: 'Campaign not found in queue' });
  queue.campaigns[index] = { ...queue.campaigns[index], ...req.body }; saveCampaignQueue(queue);
  broadcastSSE('omp-queue-updated', { action: 'update', campaign: queue.campaigns[index] });
  res.json(queue.campaigns[index]);
});

router.delete('/omp/queue/:id', (req, res) => {
  const queue = loadCampaignQueue();
  const index = queue.campaigns.findIndex(c => c.id === req.params.id);
  if (index === -1) return res.status(404).json({ error: 'Campaign not found in queue' });
  const removed = queue.campaigns.splice(index, 1)[0]; saveCampaignQueue(queue);
  broadcastSSE('omp-queue-updated', { action: 'remove', campaignId: req.params.id });
  res.json({ removed });
});

router.get('/omp/config', (req, res) => { loadOmpConfig(); res.json(ompState.config || {}); });
router.patch('/omp/config', (req, res) => { loadOmpConfig(); res.json({ message: 'Config reloaded', config: ompState.config }); });

router.get('/omp/promote-check', async (req, res) => {
  try {
    let query = `SELECT candidate_id, oos_sharpe_net, pbo, dsr, max_drawdown_net FROM scg_candidates`;
    const params = [];
    if (req.query.runId) { params.push(req.query.runId); query += ` WHERE run_id = $1`; }
    query += ` ORDER BY oos_sharpe_net DESC NULLS LAST LIMIT 100`;
    const result = await pool.query(query, params);
    const calcVar = (arr) => { if (arr.length < 2) return 0; const mean = arr.reduce((a, b) => a + b, 0) / arr.length; return arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length; };
    const sharpeVar = calcVar(result.rows.map(r => r.oos_sharpe_net).filter(v => v != null));
    const blocked = sharpeVar < 1e-6;
    res.json({ blocked, reason: blocked ? 'metrics_collapsed' : null, details: { sharpeVar: sharpeVar.toExponential(3), candidatesChecked: result.rows.length } });
  } catch (e) { res.status(500).json({ blocked: true, reason: 'error', error: e.message }); }
});

router.get('/omp/hall-of-fame', async (req, res) => {
  try {
    // First try database
    let query = `SELECT p.*, c.genome_hash, c.run_id, r.campaign_id, camp.name as campaign_name FROM scg_promotions p JOIN scg_candidates c ON p.candidate_id = c.candidate_id JOIN scg_runs r ON c.run_id = r.run_id LEFT JOIN scg_campaigns camp ON r.campaign_id = camp.campaign_id WHERE p.promotion_class = 'hall_of_fame'`;
    const params = [];
    if (req.query.market) { params.push(req.query.market); query += ` AND p.market = $${params.length}`; }
    query += ` ORDER BY p.oos_sharpe_net DESC NULLS LAST LIMIT $${params.length + 1}`; params.push(parseInt(req.query.limit) || 50);
    const result = await pool.query(query, params);
    
    if (result.rows.length > 0) {
      const genName = (e) => { const parts = [e.market?.toUpperCase() || 'BR']; if (e.campaign_name) parts.push(e.campaign_name.split('_')[0]?.slice(0, 12)); parts.push(`#${(e.genome_hash || e.candidate_id || '').slice(-6).toUpperCase()}`); return parts.join(' • ').slice(0, 48); };
      return res.json({ count: result.rows.length, entries: result.rows.map(r => ({ promotionId: r.promotion_id, candidateId: r.candidate_id, genomeHash: r.genome_hash, strategyName: genName(r), campaignId: r.campaign_id, campaignName: r.campaign_name, runId: r.run_id, market: r.market, promotedAt: r.promoted_at, metrics: { oosSharpeNet: r.oos_sharpe_net, pbo: r.pbo, dsr: r.dsr, maxDrawdownNet: r.max_drawdown_net, cagrNet: r.cagr_net }, validation: { stressPassed: r.stress_passed, stressTotal: r.stress_total, gatesPassed: r.gates_passed }, provenance: { gitSha: r.git_sha, configHash: r.config_hash }, notes: r.notes })) });
    }
    
    // Fallback: read from local files
    const entries = await readLocalHallOfFame(parseInt(req.query.limit) || 50);
    res.json({ count: entries.length, entries, source: 'local' });
  } catch (e) { 
    // If DB fails, try local fallback
    try {
      const entries = await readLocalHallOfFame(parseInt(req.query.limit) || 50);
      res.json({ count: entries.length, entries, source: 'local' });
    } catch (e2) {
      res.status(500).json({ error: e.message, entries: [] }); 
    }
  }
});

// Read Hall of Fame from local SCG output files
async function readLocalHallOfFame(limit = 50) {
  const scgDir = path.join(PROJECT_ROOT, 'output', 'scg');
  if (!fs.existsSync(scgDir)) return [];
  
  const runs = fs.readdirSync(scgDir).filter(d => d.startsWith('run_') && fs.statSync(path.join(scgDir, d)).isDirectory());
  const allEntries = [];
  
  for (const run of runs) {
    const hofDir = path.join(scgDir, run, 'hall_of_fame');
    if (!fs.existsSync(hofDir)) continue;
    
    const stratDirs = fs.readdirSync(hofDir).filter(d => d.startsWith('strategy_'));
    for (const stratDir of stratDirs) {
      const stratPath = path.join(hofDir, stratDir, 'strategy.toml');
      const metricsPath = path.join(hofDir, stratDir, 'metrics.obfs');
      
      if (!fs.existsSync(stratPath)) continue;
      
      try {
        const tomlContent = fs.readFileSync(stratPath, 'utf-8');
        const idMatch = tomlContent.match(/id\s*=\s*"([^"]+)"/);
        const descMatch = tomlContent.match(/description\s*=\s*"([^"]+)"/);
        const stratId = idMatch ? idMatch[1] : stratDir;
        
        // Parse metrics from OBFS (zstd compressed JSON)
        let sharpe = 0, cagr = 0, maxDd = 0, pbo = 0, dsr = 0;
        if (fs.existsSync(metricsPath)) {
          try {
            const { execSync } = await import('child_process');
            const metricsJson = execSync(`zstd -d -c "${metricsPath}"`, { encoding: 'utf-8', maxBuffer: 1024 * 1024 });
            const metrics = JSON.parse(metricsJson);
            sharpe = metrics.sharpe_ratio || 0;
            cagr = metrics.cagr || 0;
            maxDd = metrics.max_drawdown || 0;
            pbo = metrics.pbo || 0;
            dsr = metrics.dsr || 0;
          } catch (e) { /* failed to read metrics */ }
        }
        
        const genMatch = descMatch ? descMatch[1].match(/generation\s+(\d+)/i) : null;
        const gen = genMatch ? parseInt(genMatch[1]) : 0;
        
        // Extract rank from folder name
        const rankMatch = stratDir.match(/strategy_(\d+)/);
        const rank = rankMatch ? parseInt(rankMatch[1]) : 999;
        
        allEntries.push({
          candidateId: stratId,
          genomeHash: stratId.slice(-8),
          strategyName: `BR • MaxPower • #${stratId.slice(-6).toUpperCase()}`,
          runId: run,
          market: 'br',
          rank,
          generation: gen,
          promotedAt: fs.statSync(stratPath).mtime.toISOString(),
          metrics: { oosSharpeNet: sharpe, pbo, dsr, maxDrawdownNet: maxDd, cagrNet: cagr },
          validation: { stressPassed: 0, stressTotal: 5, gatesPassed: true },
          strategyPath: stratPath
        });
      } catch (e) { /* skip invalid */ }
    }
  }
  
  // Sort by sharpe (descending) and limit
  allEntries.sort((a, b) => b.metrics.oosSharpeNet - a.metrics.oosSharpeNet);
  return allEntries.slice(0, limit);
}

router.get('/omp/performance', async (req, res) => {
  const cc = ompState.currentCampaign;
  let evalPerSec = 0;
  if (cc) { const elapsedSecs = (Date.now() - cc.startTime) / 1000; evalPerSec = elapsedSecs > 0 ? cc.candidatesEvaluated / elapsedSecs : 0; }
  res.json({ current_run: cc ? { run_id: cc.runId, evaluations_per_second: Math.round(evalPerSec * 100) / 100, current_generation: cc.currentGeneration, best_sharpe: cc.bestSharpe, candidates_evaluated: cc.candidatesEvaluated, elapsed_seconds: Math.floor((Date.now() - cc.startTime) / 1000) } : null, system: { cpu_usage: ompState.resources.cpuUsage, memory_usage_pct: ompState.resources.memoryUsagePct, disk_free_gb: ompState.resources.diskFreeGb }, totals: { candidates_generated: ompState.stats.candidatesGenerated, backtests_executed: ompState.stats.backtestsExecuted, promotions: ompState.stats.promotions } });
});

router.get('/omp/stats', async (req, res) => {
  try {
    const [totalCands, cands24h, cands7d, hofCount, promo24h, promo7d] = await Promise.all([
      pool.query('SELECT COUNT(*) as count FROM scg_candidates'),
      pool.query("SELECT COUNT(*) as count FROM scg_candidates WHERE created_at > NOW() - INTERVAL '24 hours'"),
      pool.query("SELECT COUNT(*) as count FROM scg_candidates WHERE created_at > NOW() - INTERVAL '7 days'"),
      pool.query("SELECT COUNT(*) as count FROM scg_promotions WHERE promotion_class = 'hall_of_fame'"),
      pool.query("SELECT COUNT(*) as count FROM scg_promotions WHERE promoted_at > NOW() - INTERVAL '24 hours'"),
      pool.query("SELECT COUNT(*) as count FROM scg_promotions WHERE promoted_at > NOW() - INTERVAL '7 days'")
    ]);
    res.json({
      candidates: { total: parseInt(totalCands.rows[0].count) || 0, last24h: parseInt(cands24h.rows[0].count) || 0, last7d: parseInt(cands7d.rows[0].count) || 0 },
      promotions: { total: parseInt(hofCount.rows[0].count) || 0, last24h: parseInt(promo24h.rows[0].count) || 0, last7d: parseInt(promo7d.rows[0].count) || 0 },
      campaigns: { completed: ompState.stats.campaignsCompleted || 0, failed: ompState.stats.campaignsFailed || 0 },
      throughput: { candidatesPerMin: 0 },
      lastPromotion: null
    });
  } catch (e) { console.error('[stats]', e.message); res.json({ candidates: { total: 0, last24h: 0, last7d: 0 }, promotions: { total: 0, last24h: 0, last7d: 0 }, campaigns: { completed: 0, failed: 0 }, throughput: { candidatesPerMin: 0 }, lastPromotion: null }); }
});

router.get('/omp/activity', (req, res) => { res.json({ activity: ompState.activityLog.slice(0, parseInt(req.query.limit) || 100) }); });

// Hall of Fame sync endpoints
router.post('/omp/hof-sync', async (req, res) => {
  try {
    const result = await runSync();
    addActivityLog('info', `HoF sync: ${result.synced}/${result.total} strategies synced to Neon`);
    res.json({ success: true, ...result });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

router.get('/omp/hof-local', async (req, res) => {
  try {
    const strategies = await scanLocalStrategies();
    const limit = parseInt(req.query.limit) || 50;
    res.json({ 
      count: strategies.length, 
      entries: strategies.slice(0, limit).map((s, i) => ({
        rank: i + 1,
        candidateId: s.candidateId,
        genomeHash: s.genomeHash,
        generation: s.generation,
        strategyName: `BR • MaxPower • #${s.genomeHash.slice(-6).toUpperCase()}`,
        sharpe: s.sharpe,
        cagr: s.cagr,
        maxDrawdown: s.maxDd,
        pbo: s.pbo,
        dsr: s.dsr,
        runId: s.runId
      }))
    });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

export default router;

