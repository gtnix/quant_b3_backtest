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

// =============================================================================
// External CLI Process Detection
// =============================================================================
let externalProcess = null;
let lastStatusRead = 0;
let cachedLiveStatus = null;

// Read live status from logs (br.log, us.log)
function readLiveStatus() {
  const now = Date.now();
  if (cachedLiveStatus && (now - lastStatusRead) < 500) return cachedLiveStatus;
  
  try {
    const status = { genBR: 0, genUS: 0, bestSharpeBR: 0, bestSharpeUS: 0, throughputBR: 0, throughputUS: 0, errorsBR: 0, errorsUS: 0 };
    
    // Parse BR log
    const brLog = path.join(PROJECT_ROOT, 'logs', 'br.log');
    if (fs.existsSync(brLog)) {
      const tail = execSync(`tail -200 "${brLog}" 2>/dev/null`, { encoding: 'utf8', timeout: 2000 });
      const genMatches = tail.match(/Gen (\d+) Stage B/g);
      if (genMatches?.length > 0) status.genBR = parseInt(genMatches[genMatches.length - 1].match(/Gen (\d+)/)[1]);
      status.errorsBR = (tail.match(/ERROR/g) || []).length;
      const sharpeMatches = tail.match(/sharpe=([\d.]+)/g);
      if (sharpeMatches) status.bestSharpeBR = Math.max(...sharpeMatches.map(s => parseFloat(s.split('=')[1])));
      status.throughputBR = (tail.match(/Backtest successful/g) || []).length;
    }
    
    // Parse US log
    const usLog = path.join(PROJECT_ROOT, 'logs', 'us.log');
    if (fs.existsSync(usLog)) {
      const tail = execSync(`tail -200 "${usLog}" 2>/dev/null`, { encoding: 'utf8', timeout: 2000 });
      const genMatches = tail.match(/Gen (\d+) Stage B/g);
      if (genMatches?.length > 0) status.genUS = parseInt(genMatches[genMatches.length - 1].match(/Gen (\d+)/)[1]);
      status.errorsUS = (tail.match(/ERROR/g) || []).length;
      const sharpeMatches = tail.match(/sharpe=([\d.]+)/g);
      if (sharpeMatches) status.bestSharpeUS = Math.max(...sharpeMatches.map(s => parseFloat(s.split('=')[1])));
      status.throughputUS = (tail.match(/Backtest successful/g) || []).length;
    }
    
    status.generation = Math.max(status.genBR, status.genUS);
    status.best_sharpe = Math.max(status.bestSharpeBR, status.bestSharpeUS);
    status.candidates_evaluated = status.throughputBR + status.throughputUS;
    
    lastStatusRead = now;
    cachedLiveStatus = status;
    return status;
  } catch (e) { /* ignore */ }
  cachedLiveStatus = null;
  return null;
}

function detectExternalCombiner() {
  try {
    const ps = execSync('pgrep -a combiner 2>/dev/null || true', { encoding: 'utf-8', timeout: 2000 });
    if (ps.includes('combiner run') || ps.includes('combiner factory')) {
      const match = ps.match(/(\d+)\s+.*combiner\s+(run|factory)/);
      if (match) {
        const pid = parseInt(match[1]);
        const command = ps.trim().split('\n')[0];
        // Detect market from command (default_us.toml or radar_15d_us.toml = US)
        const market = command.includes('_us.toml') || command.includes('US') ? 'US' : 'BR';
        const isFactory = command.includes('factory');
        // Get process start time
        let startTime = Date.now();
        let elapsed = 0;
        try {
          const stat = execSync(`ps -p ${pid} -o etimes= 2>/dev/null || echo 0`, { encoding: 'utf-8' });
          elapsed = parseInt(stat.trim()) || 0;
          startTime = Date.now() - (elapsed * 1000);
        } catch {}
        // Read live status from status.json
        const liveStatus = readLiveStatus();
        return { pid, startTime, elapsed, command, market, isFactory, liveStatus };
      }
    }
    return null;
  } catch { return null; }
}

function syncExternalProcess() {
  const ext = detectExternalCombiner();
  const currentPid = ompState.currentCampaign?.pid;
  
  if (ext) {
    // External CLI is running
    const pidChanged = currentPid && currentPid !== ext.pid;
    
    if (!ompState.currentCampaign || (ompState.currentCampaign?.external && pidChanged)) {
      // New external process or PID changed (process restarted)
      if (pidChanged) {
        console.log(`[OMP] External process changed: PID ${currentPid} -> ${ext.pid}`);
      }
      externalProcess = ext;
      if (ompState.status === 'offline') {
        ompState.status = 'running';
      }
      ompState.startedAt = new Date(ext.startTime).toISOString();
      
      // Detect experimentId from most recent scg_* directory
      let experimentId = null;
      let artifactsPath = null;
      try {
        const scgDir = path.join(PROJECT_ROOT, 'output', 'scg');
        if (fs.existsSync(scgDir)) {
          const dirs = fs.readdirSync(scgDir)
            .filter(d => d.startsWith('scg_') && fs.statSync(path.join(scgDir, d)).isDirectory())
            .map(d => ({ name: d, mtime: fs.statSync(path.join(scgDir, d)).mtime }))
            .sort((a, b) => b.mtime - a.mtime);
          if (dirs.length > 0) {
            experimentId = dirs[0].name;
            artifactsPath = `output/scg/${experimentId}/`;
          }
        }
      } catch {}
      
      ompState.currentCampaign = {
        campaignId: 'external_cli',
        campaignName: ext.isFactory ? `Factory ${ext.market}` : `CLI ${ext.market}`,
        runId: ext.liveStatus?.runId || `cli_${ext.pid}`,
        experimentId,          // NEW: Detected from output directory
        artifactsPath,         // NEW: Path to outputs
        market: ext.market.toLowerCase(),
        status: 'running',
        startTime: ext.startTime,
        external: true,
        pid: ext.pid,
        isFactory: ext.isFactory
      };
      broadcastSSE('omp-external-detected', { pid: ext.pid, elapsed: ext.elapsed, experimentId });
    }
    
    // Always update metrics from live status file
    if (ompState.currentCampaign?.external && ext.liveStatus) {
      const ls = ext.liveStatus;
      ompState.currentCampaign.currentGeneration = ls.generation || 0;
      ompState.currentCampaign.bestSharpe = ls.best_sharpe || null;
      ompState.currentCampaign.meanSharpe = ls.mean_sharpe || 0;
      ompState.currentCampaign.diversity = ls.diversity || 0;
      ompState.currentCampaign.convergenceRate = ls.convergence_rate || 0;
      ompState.currentCampaign.stagnation = ls.stagnation || 0;
      ompState.currentCampaign.paretoSize = ls.pareto_size || 0;
      ompState.currentCampaign.hofSize = ls.hof_size || 0;
      ompState.currentCampaign.candidatesEvaluated = ls.candidates_evaluated || 0;
      ompState.currentCampaign.elapsedSecs = ls.elapsed_secs || ext.elapsed;
      if (ls.runId) ompState.currentCampaign.runId = ls.runId;
    } else if (ompState.currentCampaign?.external) {
      // No live status, just update elapsed time
      ompState.currentCampaign.elapsedSecs = ext.elapsed;
    }
  } else if (!ext && externalProcess) {
    // External process finished
    externalProcess = null;
    cachedLiveStatus = null;
    if (ompState.currentCampaign?.external) {
      ompState.currentCampaign = null;
      if (ompState.status === 'running' && !getOmpLoopInterval()) {
        ompState.status = 'offline';
      }
      broadcastSSE('omp-external-finished', { timestamp: new Date().toISOString() });
    }
  }
}

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
  const cc = ompState.currentCampaign;
  const currentCampaign = cc ? { 
    campaignId: cc.campaignId, 
    campaignName: cc.campaignName, 
    runId: cc.runId, 
    experimentId: cc.experimentId || null,   // NEW: Canonical experiment identifier
    artifactsPath: cc.artifactsPath || null, // NEW: Path to outputs
    market: cc.market, 
    markets: cc.markets || [cc.market],      // NEW: Array of active markets
    status: cc.status, 
    elapsedSecs: Math.floor((Date.now() - cc.startTime) / 1000), 
    currentGeneration: cc.currentGeneration || 0, 
    bestSharpe: cc.bestSharpe || null,
    candidatesEvaluated: cc.candidatesEvaluated || 0,
    external: cc.external || false,
    pid: cc.pid || null,
    // Evolution metrics
    paretoSize: cc.paretoSize || 0,
    validatedCount: cc.validatedCount || 0,
    validatedTotal: cc.validatedTotal || 0,
    hofSize: cc.hofSize || 0,
    meanSharpe: cc.meanSharpe || 0,
    diversity: cc.diversity || 0,
    convergenceRate: cc.convergenceRate || 0,
    stagnation: cc.stagnation || 0,
    // Per-market stats
    marketStats: cc.marketStats || {}
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
  syncExternalProcess(); // Detect CLI running externally
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
  
  // Kill all processes (multi-market support)
  if (ompState.currentCampaign?.processes) {
    for (const { market, process } of ompState.currentCampaign.processes) {
      if (process) {
        console.log(`   🔴 Stopping ${market} (PID=${process.pid})`);
        process.kill('SIGTERM');
      }
    }
  } else if (ompState.currentCampaign?.process) {
    ompState.currentCampaign.process.kill('SIGTERM');
  }
  
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

// =============================================================================
// Preflight Check - Validates all prerequisites before starting mining
// =============================================================================
function runPreflightChecks(markets = ['BR']) {
  const checks = [];
  const errors = [];
  const warnings = [];
  
  // 1. Combiner binary
  const combinerPath = path.join(PROJECT_ROOT, 'target', 'release', 'combiner');
  if (fs.existsSync(combinerPath)) {
    checks.push({ id: 'combiner', name: 'Combiner Binary', status: 'pass', path: combinerPath });
  } else {
    errors.push({ id: 'combiner', name: 'Combiner Binary', status: 'fail', 
      message: 'Binary não encontrado', fix: 'cargo build --release -p combiner_cli' });
  }
  
  // 2. Backtest binary (CRITICAL - was missing before!)
  const backtestPath = path.join(PROJECT_ROOT, 'target', 'release', 'backtest');
  if (fs.existsSync(backtestPath)) {
    checks.push({ id: 'backtest', name: 'Backtest Binary', status: 'pass', path: backtestPath });
  } else {
    errors.push({ id: 'backtest', name: 'Backtest Binary', status: 'fail',
      message: 'Binary não encontrado (CRÍTICO - combiner não funciona sem ele!)',
      fix: 'cargo build --release --bin backtest' });
  }
  
  // 3. Config files for each market
  for (const market of markets) {
    const configFile = market === 'US' ? 'default_us.toml' : 'default.toml';
    const configPath = path.join(PROJECT_ROOT, 'configs', configFile);
    if (fs.existsSync(configPath)) {
      checks.push({ id: `config_${market}`, name: `Config ${market}`, status: 'pass', path: configPath });
    } else {
      errors.push({ id: `config_${market}`, name: `Config ${market}`, status: 'fail',
        message: `Arquivo não encontrado: configs/${configFile}` });
    }
  }
  
  // 4. Market data files
  for (const market of markets) {
    const dataFile = market === 'US' ? 'market_data_us.csv' : 'market_data_ibov.csv';
    const dataPath = path.join(PROJECT_ROOT, 'data', dataFile);
    if (fs.existsSync(dataPath)) {
      const stats = fs.statSync(dataPath);
      const sizeMB = (stats.size / 1024 / 1024).toFixed(1);
      const age = Math.floor((Date.now() - stats.mtime.getTime()) / (1000 * 60 * 60 * 24));
      if (age > 7) {
        warnings.push({ id: `data_${market}`, name: `Data ${market}`, status: 'warn',
          message: `Dados com ${age} dias de idade`, size: `${sizeMB}MB` });
      } else {
        checks.push({ id: `data_${market}`, name: `Data ${market}`, status: 'pass', 
          path: dataPath, size: `${sizeMB}MB`, age: `${age}d` });
      }
    } else {
      errors.push({ id: `data_${market}`, name: `Data ${market}`, status: 'fail',
        message: `Arquivo não encontrado: data/${dataFile}`,
        fix: market === 'US' ? 'python -m datahub_us sync' : 'python -m datahub_b3 update' });
    }
  }
  
  // 5. Logs directory writable
  const logsDir = path.join(PROJECT_ROOT, 'logs');
  try {
    if (!fs.existsSync(logsDir)) fs.mkdirSync(logsDir, { recursive: true });
    const testFile = path.join(logsDir, '.preflight_test');
    fs.writeFileSync(testFile, 'test');
    fs.unlinkSync(testFile);
    checks.push({ id: 'logs_dir', name: 'Logs Directory', status: 'pass', path: logsDir });
  } catch (e) {
    errors.push({ id: 'logs_dir', name: 'Logs Directory', status: 'fail',
      message: `Sem permissão de escrita: ${logsDir}` });
  }
  
  // 6. Output directory writable
  const outputDir = path.join(PROJECT_ROOT, 'output', 'scg');
  try {
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });
    checks.push({ id: 'output_dir', name: 'Output Directory', status: 'pass', path: outputDir });
  } catch (e) {
    errors.push({ id: 'output_dir', name: 'Output Directory', status: 'fail',
      message: `Sem permissão de escrita: ${outputDir}` });
  }
  
  // 7. Disk space (warn if < 5GB free)
  try {
    const df = execSync(`df -BG "${PROJECT_ROOT}" | tail -1 | awk '{print $4}'`, { encoding: 'utf8' });
    const freeGB = parseInt(df.replace('G', ''));
    if (freeGB < 5) {
      warnings.push({ id: 'disk_space', name: 'Disk Space', status: 'warn',
        message: `Apenas ${freeGB}GB livres (recomendado: 5GB+)` });
    } else {
      checks.push({ id: 'disk_space', name: 'Disk Space', status: 'pass', free: `${freeGB}GB` });
    }
  } catch {}
  
  // 8. Database URL (for HoF sync)
  if (process.env.DATABASE_URL) {
    checks.push({ id: 'database', name: 'Database URL', status: 'pass' });
  } else {
    warnings.push({ id: 'database', name: 'Database URL', status: 'warn',
      message: 'DATABASE_URL não configurado - HoF não será sincronizado' });
  }
  
  // 9. No conflicting processes
  try {
    const ps = execSync('pgrep -c combiner 2>/dev/null || echo 0', { encoding: 'utf8' });
    const count = parseInt(ps.trim());
    if (count > 0) {
      warnings.push({ id: 'running_process', name: 'Existing Process', status: 'warn',
        message: `${count} processo(s) combiner já em execução` });
    } else {
      checks.push({ id: 'running_process', name: 'No Conflicts', status: 'pass' });
    }
  } catch {}
  
  const canStart = errors.length === 0;
  return { canStart, checks, errors, warnings, summary: {
    passed: checks.length,
    failed: errors.length,
    warnings: warnings.length
  }};
}

router.get('/omp/preflight', (req, res) => {
  const markets = req.query.markets ? req.query.markets.split(',') : ['BR', 'US'];
  const result = runPreflightChecks(markets);
  res.json(result);
});

// =============================================================================
// Simple Quick Start - No queue, just run! Supports BR+US simultaneous
// =============================================================================
router.post('/omp/quick-start', async (req, res) => {
  const mode = req.body.mode || 'quick';
  const indefinite = req.body.indefinite || false;
  const markets = req.body.markets || ['BR'];
  const templateSlugs = req.body.templateSlugs || [];
  const skipPreflight = req.body.skipPreflight || false;
  
  // Run preflight checks first (unless explicitly skipped)
  if (!skipPreflight) {
    const preflight = runPreflightChecks(markets);
    if (!preflight.canStart) {
      return res.status(400).json({ 
        error: 'Preflight check failed',
        preflight
      });
    }
  }
  
  if (ompState.currentCampaign) return res.status(400).json({ error: 'Campaign already running' });
  
  // Indefinite mode: 24h (86400 seg), senão usa os tempos padrão
  const runtimeMins = indefinite ? 1440 : (mode === 'full' ? 60 : 15);
  const totalCores = os.cpus().length;
  // Split workers between markets
  const workersPerMarket = mode === 'full' 
    ? Math.floor((totalCores - 2) / markets.length) 
    : Math.floor(totalCores / (2 * markets.length));
  
  const combinerPath = path.join(PROJECT_ROOT, 'target', 'release', 'combiner');
  if (!fs.existsSync(combinerPath)) {
    return res.status(400).json({ error: 'Combiner binary not found. Run: cargo build --release -p combiner_cli' });
  }
  
  // Validate all configs exist
  for (const market of markets) {
    const configFile = market === 'US' ? 'default_us.toml' : 'default.toml';
    const configPath = path.join(PROJECT_ROOT, 'configs', configFile);
    if (!fs.existsSync(configPath)) {
      return res.status(400).json({ error: `Config not found: configs/${configFile}` });
    }
  }
  
  // Generate experimentId in same format as Rust pipeline: scg_YYYYMMDD_HHMMSS
  const now = new Date();
  const experimentId = `scg_${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`;
  const artifactsPath = `output/scg/${experimentId}/`;
  
  const runId = `run_${Date.now().toString(36)}`;
  const cpuPercent = mode === 'full' ? '100%' : '50%';
  const marketLabel = markets.join('+');
  console.log(`\n🚀 [OMP] Iniciando: ${mode === 'full' ? 'NOITE' : 'DIA'} (${cpuPercent} CPU, ${indefinite ? 'indefinido' : runtimeMins + 'min'}, ${marketLabel})`);
  console.log(`   📁 Experiment: ${experimentId}`);
  
  // Spawn one process per market
  const processes = [];
  const marketStats = {}; // Track stats per market
  
  // Create temp configs dir if needed
  const tempConfigDir = path.join(PROJECT_ROOT, 'configs', '.runtime');
  if (!fs.existsSync(tempConfigDir)) {
    fs.mkdirSync(tempConfigDir, { recursive: true });
  }

  for (const market of markets) {
    const configFile = market === 'US' ? 'default_us.toml' : 'default.toml';
    const baseConfigPath = path.join(PROJECT_ROOT, 'configs', configFile);
    const seed = market === 'US' ? 43 : 42; // Different seeds for variety
    
    // If template slugs provided, create runtime config with filter
    let configPath = baseConfigPath;
    if (templateSlugs.length > 0) {
      const baseConfig = fs.readFileSync(baseConfigPath, 'utf8');
      const templateSlugsToml = `\n# Runtime template filter (auto-generated)\ntemplate_slugs = ${JSON.stringify(templateSlugs)}\n`;
      const runtimeConfig = baseConfig + templateSlugsToml;
      const runtimeConfigPath = path.join(tempConfigDir, `${runId}_${market}.toml`);
      fs.writeFileSync(runtimeConfigPath, runtimeConfig);
      configPath = runtimeConfigPath;
      console.log(`   📋 [${market}] Using ${templateSlugs.length} template filter`);
    }
    
    const args = ['run', '--config', configPath, '--ultra', '--seed', String(seed)];
    
    const proc = spawn(combinerPath, args, { 
      cwd: PROJECT_ROOT, 
      env: { 
        ...process.env, 
        RUST_LOG: 'combiner=info',
        SCG_MAX_RUNTIME_SECS: String(runtimeMins * 60),
        SCG_WORKERS: String(workersPerMarket),
        SCG_EXPERIMENT_ID: experimentId
      } 
    });
    
    processes.push({ market, process: proc });
    marketStats[market] = { generation: 0, bestSharpe: null, candidates: 0, hofSize: 0 };
    console.log(`   📊 [${market}] PID=${proc.pid}, workers=${workersPerMarket}, seed=${seed}`);
  }
  
  ompState.status = 'running';
  ompState.startedAt = new Date().toISOString();
  
  const modeName = mode === 'full' ? 'Noite (100% CPU)' : 'Dia (50% CPU)';
  
  ompState.currentCampaign = {
    campaignId: `quick_${mode}`,
    campaignName: `${modeName} • ${marketLabel}`,
    runId,
    experimentId,      // NEW: Canonical experiment identifier
    artifactsPath,     // NEW: Path to outputs (output/scg/<experimentId>/)
    market: marketLabel,
    markets, // Array of markets
    status: 'running',
    startTime: Date.now(),
    output: [],
    processes, // Array of {market, process}
    process: processes[0]?.process, // Primary process for backward compat
    marketStats, // Stats per market
    currentGeneration: 0,
    bestSharpe: null,
    candidatesEvaluated: 0,
    mode,
    indefinite,
    // Evolution metrics (aggregated from all markets)
    paretoSize: 0,
    validatedCount: 0,
    validatedTotal: 0,
    hofSize: 0,
    meanSharpe: 0,
    diversity: 0,
    convergenceRate: 0,
    stagnation: 0
  };
  
  const parseOutput = (line) => {
    if (!ompState.currentCampaign) return;
    ompState.currentCampaign.output.push(line);
    
    // Parse generation: "Gen 5 Stage B" or "Gen 5 ULTRA"
    const genMatch = line.match(/Gen\s+(\d+)\s+(?:Stage|ULTRA)/i);
    if (genMatch) ompState.currentCampaign.currentGeneration = parseInt(genMatch[1]);
    
    // Parse sharpe from backtest results only (not config thresholds like "Failures: sharpe=8")
    if (line.includes('Backtest successful')) {
      const sharpeMatch = line.match(/sharpe[=:\s]+(-?\d+\.?\d*)/i);
      if (sharpeMatch) {
        const sharpe = parseFloat(sharpeMatch[1]);
        const best = ompState.currentCampaign.bestSharpe;
        if (best === null || sharpe > best) {
          ompState.currentCampaign.bestSharpe = sharpe;
        }
      }
    }
    
    // Count candidates from "Backtest successful"
    if (line.includes('Backtest successful')) {
      ompState.currentCampaign.candidatesEvaluated++;
    }
    
    // Parse generation metrics: 
    // ULTRA mode: "Gen X ULTRA: ... | mean=0.123 div=0.045 conv=0.0012 stag=3"
    // Normal mode: "Gen 0: pareto=21, best_sharpe=0.645, best_cagr=24.9%, hof=12"
    if (line.includes('ULTRA:') || line.match(/Gen \d+:/)) {
      const paretoMatch = line.match(/pareto=(\d+)/);
      const hofMatch = line.match(/hof=(\d+)/);
      
      if (paretoMatch) ompState.currentCampaign.paretoSize = parseInt(paretoMatch[1]);
      if (hofMatch) ompState.currentCampaign.hofSize = parseInt(hofMatch[1]);
      
      // ULTRA-specific metrics
      if (line.includes('ULTRA:') && line.includes('|')) {
        const validatedMatch = line.match(/validated=(\d+)\/(\d+)/);
        const meanMatch = line.match(/mean=(-?\d+\.?\d*)/);
        const divMatch = line.match(/div=(-?\d+\.?\d*)/);
        const convMatch = line.match(/conv=(-?\d+\.?\d*)/);
        const stagMatch = line.match(/stag=(\d+)/);
        
        if (validatedMatch) {
          ompState.currentCampaign.validatedCount = parseInt(validatedMatch[1]);
          ompState.currentCampaign.validatedTotal = parseInt(validatedMatch[2]);
        }
        if (meanMatch) ompState.currentCampaign.meanSharpe = parseFloat(meanMatch[1]);
        if (divMatch) ompState.currentCampaign.diversity = parseFloat(divMatch[1]);
        if (convMatch) ompState.currentCampaign.convergenceRate = parseFloat(convMatch[1]);
        if (stagMatch) ompState.currentCampaign.stagnation = parseInt(stagMatch[1]);
      }
    }
  };
  
  // Track which processes have finished
  let finishedCount = 0;
  
  // Attach output handlers to each process
  for (const { market, process: proc } of processes) {
    const parseMarketOutput = (line) => {
      if (!ompState.currentCampaign) return;
      ompState.currentCampaign.output.push(`[${market}] ${line}`);
      
      const stats = ompState.currentCampaign.marketStats[market];
      if (!stats) return;
      
      // Parse generation
      const genMatch = line.match(/Gen\s+(\d+)\s+(?:Stage|ULTRA)/i);
      if (genMatch) stats.generation = parseInt(genMatch[1]);
      
      // Parse sharpe from backtest results
      if (line.includes('Backtest successful')) {
        const sharpeMatch = line.match(/sharpe[=:\s]+(-?\d+\.?\d*)/i);
        if (sharpeMatch) {
          const sharpe = parseFloat(sharpeMatch[1]);
          if (stats.bestSharpe === null || sharpe > stats.bestSharpe) {
            stats.bestSharpe = sharpe;
          }
        }
        stats.candidates++;
      }
      
      // Parse HoF size
      const hofMatch = line.match(/hof=(\d+)/);
      if (hofMatch) stats.hofSize = parseInt(hofMatch[1]);
      
      // Parse ULTRA-specific metrics
      if (line.includes('ULTRA:') && line.includes('|')) {
        const paretoMatch = line.match(/pareto=(\d+)/);
        const meanMatch = line.match(/mean=(-?\d+\.?\d*)/);
        const divMatch = line.match(/div=(-?\d+\.?\d*)/);
        const convMatch = line.match(/conv=(-?\d+\.?\d*)/);
        const stagMatch = line.match(/stag=(\d+)/);
        const validatedMatch = line.match(/validated=(\d+)\/(\d+)/);
        
        if (paretoMatch) stats.paretoSize = parseInt(paretoMatch[1]);
        if (meanMatch) stats.meanSharpe = parseFloat(meanMatch[1]);
        if (divMatch) stats.diversity = parseFloat(divMatch[1]);
        if (convMatch) stats.convergenceRate = parseFloat(convMatch[1]);
        if (stagMatch) stats.stagnation = parseInt(stagMatch[1]);
        if (validatedMatch) {
          stats.validatedCount = parseInt(validatedMatch[1]);
          stats.validatedTotal = parseInt(validatedMatch[2]);
        }
      }
      
      // Aggregate stats from all markets
      const allStats = Object.values(ompState.currentCampaign.marketStats);
      ompState.currentCampaign.currentGeneration = Math.max(...allStats.map(s => s.generation || 0));
      ompState.currentCampaign.candidatesEvaluated = allStats.reduce((sum, s) => sum + (s.candidates || 0), 0);
      ompState.currentCampaign.hofSize = allStats.reduce((sum, s) => sum + (s.hofSize || 0), 0);
      ompState.currentCampaign.paretoSize = allStats.reduce((sum, s) => sum + (s.paretoSize || 0), 0);
      ompState.currentCampaign.validatedCount = allStats.reduce((sum, s) => sum + (s.validatedCount || 0), 0);
      ompState.currentCampaign.validatedTotal = allStats.reduce((sum, s) => sum + (s.validatedTotal || 0), 0);
      ompState.currentCampaign.stagnation = Math.max(...allStats.map(s => s.stagnation || 0));
      
      // Average for mean metrics
      const validMeanSharpes = allStats.map(s => s.meanSharpe).filter(s => s !== undefined);
      ompState.currentCampaign.meanSharpe = validMeanSharpes.length > 0 
        ? validMeanSharpes.reduce((a, b) => a + b, 0) / validMeanSharpes.length 
        : 0;
      
      const validDiversities = allStats.map(s => s.diversity).filter(s => s !== undefined);
      ompState.currentCampaign.diversity = validDiversities.length > 0 
        ? validDiversities.reduce((a, b) => a + b, 0) / validDiversities.length 
        : 0;
        
      const validConvRates = allStats.map(s => s.convergenceRate).filter(s => s !== undefined);
      ompState.currentCampaign.convergenceRate = validConvRates.length > 0 
        ? validConvRates.reduce((a, b) => a + b, 0) / validConvRates.length 
        : 0;
      
      const sharpes = allStats.map(s => s.bestSharpe).filter(s => s !== null);
      ompState.currentCampaign.bestSharpe = sharpes.length > 0 ? Math.max(...sharpes) : null;
    };
    
    proc.stdout.on('data', (data) => {
      data.toString().split('\n').filter(l => l.trim()).forEach(parseMarketOutput);
    });
    
    proc.stderr.on('data', (data) => {
      data.toString().split('\n').filter(l => l.trim()).forEach(parseMarketOutput);
    });
    
    proc.on('close', (code) => {
      finishedCount++;
      console.log(`   ✅ [${market}] Finished (exit=${code}), ${finishedCount}/${processes.length}`);
      
      // Only cleanup state when ALL processes have finished
      if (finishedCount >= processes.length) {
        const cc = ompState.currentCampaign;
        if (cc) {
          cc.status = 'completed';
          cc.endTime = Date.now();
          ompState.stats.campaignsCompleted++;
          broadcastSSE('omp-campaign-completed', { 
            runId: cc.runId, 
            experimentId: cc.experimentId,
            artifactsPath: cc.artifactsPath,
            status: cc.status, 
            duration: (cc.endTime - cc.startTime) / 1000 
          });
        }
        ompState.currentCampaign = null;
        ompState.status = 'offline';
      }
    });
  }
  
  broadcastSSE('omp-started', { mode, runId, experimentId, artifactsPath, markets, startedAt: ompState.startedAt });
  res.json({ status: 'started', mode, runId, experimentId, artifactsPath, markets, runtimeMins, workersPerMarket });
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

// =============================================================================
// Diagnostics: Read failure breakdown and near-misses from experiment outputs
// =============================================================================
router.get('/omp/diagnostics/:experimentId', async (req, res) => {
  const { experimentId } = req.params;
  const artifactsDir = path.join(PROJECT_ROOT, 'output', 'scg', experimentId);
  
  if (!fs.existsSync(artifactsDir)) {
    return res.status(404).json({ error: `Experiment not found: ${experimentId}` });
  }
  
  const result = {
    experimentId,
    artifactsPath: `output/scg/${experimentId}/`,
    failedCandidates: [],
    failureBreakdown: {},
    nearMisses: [],
    files: []
  };
  
  // List available artifact files
  try {
    const files = fs.readdirSync(artifactsDir);
    result.files = files;
  } catch {}
  
  // Read failed_candidates.json if exists
  const failedPath = path.join(artifactsDir, 'failed_candidates.json');
  if (fs.existsSync(failedPath)) {
    try {
      const data = JSON.parse(fs.readFileSync(failedPath, 'utf8'));
      result.failedCandidates = data.candidates || data;
      
      // Compute failure breakdown (count by reason)
      const breakdown = {};
      for (const c of result.failedCandidates) {
        const reasons = c.failure_reasons || c.reasons || [];
        for (const reason of reasons) {
          breakdown[reason] = (breakdown[reason] || 0) + 1;
        }
      }
      result.failureBreakdown = breakdown;
      
      // Find near-misses: candidates closest to passing
      // Sort by number of failures (ascending) then by Sharpe (descending)
      const withScores = result.failedCandidates
        .filter(c => c.metrics?.sharpe != null || c.sharpe != null)
        .map(c => ({
          id: c.id || c.genome_hash,
          sharpe: c.metrics?.sharpe ?? c.sharpe,
          drawdown: c.metrics?.max_drawdown ?? c.max_drawdown,
          reasons: c.failure_reasons || c.reasons || [],
          reasonCount: (c.failure_reasons || c.reasons || []).length
        }))
        .sort((a, b) => a.reasonCount - b.reasonCount || (b.sharpe || 0) - (a.sharpe || 0));
      
      result.nearMisses = withScores.slice(0, 10);
    } catch (e) {
      result.parseError = e.message;
    }
  }
  
  // Read diagnostics files if they exist (from combiner diagnose)
  const diagFiles = ['br_diagnostic_report.json', 'br_failure_breakdown.json', 'br_near_miss.json'];
  for (const diagFile of diagFiles) {
    const diagPath = path.join(artifactsDir, diagFile);
    if (fs.existsSync(diagPath)) {
      try {
        result[diagFile.replace('.json', '')] = JSON.parse(fs.readFileSync(diagPath, 'utf8'));
      } catch {}
    }
  }
  
  res.json(result);
});

// List all experiments with basic metadata
router.get('/omp/runs', async (req, res) => {
  const scgDir = path.join(PROJECT_ROOT, 'output', 'scg');
  
  if (!fs.existsSync(scgDir)) {
    return res.json({ runs: [], count: 0 });
  }
  
  const runs = [];
  try {
    const dirs = fs.readdirSync(scgDir)
      .filter(d => (d.startsWith('scg_') || d.startsWith('run_')) && fs.statSync(path.join(scgDir, d)).isDirectory());
    
    for (const dir of dirs) {
      const dirPath = path.join(scgDir, dir);
      const stat = fs.statSync(dirPath);
      
      // Try to read manifest.json for metadata
      let manifest = null;
      const manifestPath = path.join(dirPath, 'manifest.json');
      if (fs.existsSync(manifestPath)) {
        try {
          manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
        } catch {}
      }
      
      // Count hall_of_fame entries
      let hofCount = 0;
      const hofDir = path.join(dirPath, 'hall_of_fame');
      if (fs.existsSync(hofDir)) {
        try {
          hofCount = fs.readdirSync(hofDir).filter(d => d.startsWith('strategy_')).length;
        } catch {}
      }
      
      runs.push({
        experimentId: dir,
        artifactsPath: `output/scg/${dir}/`,
        createdAt: stat.mtime.toISOString(),
        modifiedAt: stat.mtime.toISOString(),
        hofCount,
        manifest
      });
    }
    
    // Sort by date descending (most recent first)
    runs.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
    
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
  
  res.json({ runs, count: runs.length });
});

// Get details for a specific experiment
router.get('/omp/runs/:experimentId', async (req, res) => {
  const { experimentId } = req.params;
  const artifactsDir = path.join(PROJECT_ROOT, 'output', 'scg', experimentId);
  
  if (!fs.existsSync(artifactsDir)) {
    return res.status(404).json({ error: `Experiment not found: ${experimentId}` });
  }
  
  const result = {
    experimentId,
    artifactsPath: `output/scg/${experimentId}/`,
    files: [],
    hofStrategies: [],
    manifest: null
  };
  
  // List files
  try {
    result.files = fs.readdirSync(artifactsDir);
  } catch {}
  
  // Read manifest
  const manifestPath = path.join(artifactsDir, 'manifest.json');
  if (fs.existsSync(manifestPath)) {
    try {
      result.manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    } catch {}
  }
  
  // List HoF strategies
  const hofDir = path.join(artifactsDir, 'hall_of_fame');
  if (fs.existsSync(hofDir)) {
    try {
      const stratDirs = fs.readdirSync(hofDir).filter(d => d.startsWith('strategy_'));
      for (const stratDir of stratDirs) {
        const rankingPath = path.join(hofDir, 'ranking.obfs');
        const rankingJsonPath = path.join(hofDir, 'ranking.json');
        // Just list the directory names for now
        result.hofStrategies.push(stratDir);
      }
    } catch {}
  }
  
  res.json(result);
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
    // Force local source for testing if requested
    if (req.query.source === 'local') {
      const entries = await readLocalHallOfFame(parseInt(req.query.limit) || 50);
      return res.json({ count: entries.length, entries, source: 'local' });
    }
    // First try database with genome_json for identity extraction
    let query = `SELECT p.*, c.genome_hash, c.run_id, r.campaign_id, camp.name as campaign_name, h.genome_json FROM scg_promotions p JOIN scg_candidates c ON p.candidate_id = c.candidate_id JOIN scg_runs r ON c.run_id = r.run_id LEFT JOIN scg_campaigns camp ON r.campaign_id = camp.campaign_id LEFT JOIN hall_of_fame h ON c.genome_hash = h.genome_hash WHERE p.promotion_class = 'hall_of_fame'`;
    const params = [];
    if (req.query.market && req.query.market !== 'all') { 
      params.push(req.query.market.toUpperCase()); 
      query += ` AND UPPER(p.market) = $${params.length}`; 
    }
    query += ` ORDER BY p.oos_sharpe_net DESC NULLS LAST LIMIT $${params.length + 1}`; params.push(parseInt(req.query.limit) || 50);
    const result = await pool.query(query, params);
    
    if (result.rows.length > 0) {
      const genName = (e) => { const parts = [e.market?.toUpperCase() || 'BR']; if (e.campaign_name) parts.push(e.campaign_name.split('_')[0]?.slice(0, 12)); parts.push(`#${(e.genome_hash || e.candidate_id || '').slice(-6).toUpperCase()}`); return parts.join(' • ').slice(0, 48); };
      const extractIdentity = (r) => { try { const g = typeof r.genome_json === 'string' ? JSON.parse(r.genome_json) : r.genome_json; return g?.identity || null; } catch { return null; } };
      return res.json({ count: result.rows.length, entries: result.rows.map(r => ({ promotionId: r.promotion_id, candidateId: r.candidate_id, genomeHash: r.genome_hash, strategyName: genName(r), campaignId: r.campaign_id, campaignName: r.campaign_name, runId: r.run_id, market: r.market, promotedAt: r.promoted_at, metrics: { oosSharpeNet: r.oos_sharpe_net, pbo: r.pbo, dsr: r.dsr, maxDrawdownNet: r.max_drawdown_net, cagrNet: r.cagr_net }, validation: { stressPassed: r.stress_passed, stressTotal: r.stress_total, gatesPassed: r.gates_passed }, provenance: { gitSha: r.git_sha, configHash: r.config_hash }, notes: r.notes, identity: extractIdentity(r) })) });
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
  
  const runs = fs.readdirSync(scgDir).filter(d => (d.startsWith('run_') || d.startsWith('scg_')) && fs.statSync(path.join(scgDir, d)).isDirectory());
  const allEntries = [];
  
  for (const run of runs) {
    const hofDir = path.join(scgDir, run, 'hall_of_fame');
    if (!fs.existsSync(hofDir)) continue;
    
    const stratDirs = fs.readdirSync(hofDir).filter(d => d.startsWith('strategy_'));
    for (const stratDir of stratDirs) {
      // Support both strategy.toml and config.toml naming
      let stratPath = path.join(hofDir, stratDir, 'strategy.toml');
      if (!fs.existsSync(stratPath)) stratPath = path.join(hofDir, stratDir, 'config.toml');
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
        
        // Try to read identity from genome.obfs
        let identity = null;
        const genomePath = path.join(hofDir, stratDir, 'genome.obfs');
        if (fs.existsSync(genomePath)) {
          try {
            const { execSync } = await import('child_process');
            const genomeJson = execSync(`zstd -d -c "${genomePath}"`, { encoding: 'utf-8', maxBuffer: 1024 * 1024 });
            const genome = JSON.parse(genomeJson);
            identity = genome.identity || null;
          } catch (e) { /* failed to read genome */ }
        }
        
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
          strategyPath: stratPath,
          identity
        });
      } catch (e) { /* skip invalid */ }
    }
  }
  
  // Sort by sharpe (descending) and limit
  allEntries.sort((a, b) => b.metrics.oosSharpeNet - a.metrics.oosSharpeNet);
  return allEntries.slice(0, limit);
}

router.get('/omp/performance', async (req, res) => {
  syncExternalProcess(); // Update metrics from status.json
  const cc = ompState.currentCampaign;
  let evalPerSec = 0;
  if (cc) { 
    const elapsedSecs = cc.elapsedSecs || ((Date.now() - cc.startTime) / 1000); 
    evalPerSec = elapsedSecs > 0 ? (cc.candidatesEvaluated || 0) / elapsedSecs : 0; 
  }
  res.json({ 
    current_run: cc ? { 
      run_id: cc.runId, 
      evaluations_per_second: Math.round(evalPerSec * 100) / 100, 
      current_generation: cc.currentGeneration, 
      best_sharpe: cc.bestSharpe, 
      candidates_evaluated: cc.candidatesEvaluated, 
      elapsed_seconds: Math.floor((Date.now() - cc.startTime) / 1000),
      // Evolution metrics
      mean_sharpe: cc.meanSharpe || 0,
      diversity: cc.diversity || 0,
      convergence_rate: cc.convergenceRate || 0,
      stagnation: cc.stagnation || 0,
      pareto_size: cc.paretoSize || 0,
      validated_count: cc.validatedCount || 0,
      hof_size: cc.hofSize || 0
    } : null, 
    system: { 
      cpu_usage: ompState.resources.cpuUsage, 
      memory_usage_pct: ompState.resources.memoryUsagePct, 
      disk_free_gb: ompState.resources.diskFreeGb,
      disk_write_rate_mb: ompState.resources.writeRateMbPerSec || 0,
      disk_time_to_limit_hours: ompState.resources.estimatedTimeToLimitHours || Infinity
    }, 
    totals: { 
      candidates_generated: ompState.stats.candidatesGenerated, 
      backtests_executed: ompState.stats.backtestsExecuted, 
      promotions: ompState.stats.promotions 
    } 
  });
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

