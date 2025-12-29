/**
 * Quant Dashboard - API Server for Browser Testing
 * 
 * This server provides the same data as the Rust backend, allowing
 * the frontend to work in browser mode for testing and development.
 * 
 * Usage: node server.js
 * API runs on: http://localhost:3001
 */

import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { parse } from 'csv-parse/sync';
import toml from 'toml';

const app = express();
const PORT = 3001;

// CORS for frontend
app.use(cors());
app.use(express.json());

// Resolve artifacts path relative to project root
const PROJECT_ROOT = path.resolve(process.cwd(), '..');
let ARTIFACTS_ROOT = path.join(PROJECT_ROOT, 'artifacts');

console.log('🚀 Quant Dashboard API Server');
console.log(`📁 Project Root: ${PROJECT_ROOT}`);
console.log(`📦 Artifacts: ${ARTIFACTS_ROOT}`);

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function readJsonFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    return JSON.parse(content);
  } catch (error) {
    console.error(`Error reading ${filePath}:`, error.message);
    return null;
  }
}

function readTomlFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    return toml.parse(content);
  } catch (error) {
    console.error(`Error reading ${filePath}:`, error.message);
    return null;
  }
}

function readCsvFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    return parse(content, { columns: true, skip_empty_lines: true });
  } catch (error) {
    console.error(`Error reading ${filePath}:`, error.message);
    return [];
  }
}

function generateDisplayName(strategy) {
  if (!strategy || !strategy.pipeline) return 'Unknown Strategy';
  
  const parts = [];
  for (const block of strategy.pipeline) {
    if (block.type === 'selection') {
      parts.push(`Sel:${block.block_id}`);
    } else if (block.type === 'entry') {
      parts.push(`Entry:${block.block_id}`);
    } else if (block.type === 'exit') {
      parts.push(`Exit:${block.block_id}`);
    } else if (block.type === 'sizing') {
      parts.push(`Size:${block.block_id}`);
    }
  }
  
  return parts.length > 0 ? parts.join(' | ') : 'Unknown Strategy';
}

// =============================================================================
// API ENDPOINTS
// =============================================================================

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', artifacts_root: ARTIFACTS_ROOT });
});

// Set artifacts root
app.post('/api/set-root', (req, res) => {
  const { path: newPath } = req.body;
  
  // Check if path exists
  let testPath = newPath;
  if (!fs.existsSync(path.join(testPath, 'site', 'index.json'))) {
    testPath = path.join(newPath, 'artifacts');
  }
  
  if (!fs.existsSync(path.join(testPath, 'site', 'index.json'))) {
    return res.status(400).json({ error: `No artifacts found at ${newPath}` });
  }
  
  ARTIFACTS_ROOT = testPath;
  console.log(`📁 Artifacts root set to: ${ARTIFACTS_ROOT}`);
  res.json({ artifacts_root: ARTIFACTS_ROOT });
});

// Get site index
app.get('/api/index', (req, res) => {
  const indexPath = path.join(ARTIFACTS_ROOT, 'site', 'index.json');
  const data = readJsonFile(indexPath);
  
  if (!data) {
    return res.status(404).json({ error: 'Index not found' });
  }
  
  res.json(data);
});

// Get campaign detail
app.get('/api/campaign/:campaignId', (req, res) => {
  const { campaignId } = req.params;
  const filePath = path.join(ARTIFACTS_ROOT, 'site', `campaign_${campaignId}.json`);
  const data = readJsonFile(filePath);
  
  if (!data) {
    return res.status(404).json({ error: `Campaign ${campaignId} not found` });
  }
  
  res.json(data);
});

// Get run detail
app.get('/api/run/:runId', (req, res) => {
  const { runId } = req.params;
  const filePath = path.join(ARTIFACTS_ROOT, 'site', `run_${runId}.json`);
  const data = readJsonFile(filePath);
  
  if (!data) {
    return res.status(404).json({ error: `Run ${runId} not found` });
  }
  
  res.json(data);
});

// List candidates for a run
app.get('/api/candidates/:runId', (req, res) => {
  const { runId } = req.params;
  const { limit = 100, search, candidate_class, max_pbo } = req.query;
  
  // Try CSV first
  const csvPath = path.join(ARTIFACTS_ROOT, 'top_candidates', runId, 'top1000.csv');
  
  if (fs.existsSync(csvPath)) {
    let candidates = readCsvFile(csvPath);
    
    // Transform CSV data
    candidates = candidates.map((row, idx) => ({
      rank: parseInt(row.rank) || idx + 1,
      candidate_id: row.candidate_id,
      genome_hash: row.genome_hash,
      display_name: `Strategy #${row.rank} | ${row.candidate_id.slice(-8)}`,
      candidate_class: row.gates_passed === 'true' ? 'validated' : 'research',
      oos_sharpe_net: parseFloat(row.oos_sharpe_net) || 0,
      oos_cagr_net: parseFloat(row.oos_cagr_net) || 0,
      max_drawdown_net: parseFloat(row.max_drawdown_net) || 0,
      pbo: parseFloat(row.pbo) || 0,
      dsr: parseFloat(row.dsr) || 0,
      stress_passed: parseInt(row.stress_passed) || 0,
      stress_total: parseInt(row.stress_total) || 0,
      gates_passed: row.gates_passed === 'true',
      data_integrity_ok: true,
      created_at: row.created_at
    }));
    
    // Apply filters
    if (search) {
      const q = search.toLowerCase();
      candidates = candidates.filter(c => 
        c.candidate_id.toLowerCase().includes(q) ||
        c.display_name.toLowerCase().includes(q)
      );
    }
    
    if (candidate_class) {
      candidates = candidates.filter(c => c.candidate_class === candidate_class);
    }
    
    if (max_pbo) {
      candidates = candidates.filter(c => c.pbo <= parseFloat(max_pbo));
    }
    
    // Apply limit
    candidates = candidates.slice(0, parseInt(limit));
    
    return res.json(candidates);
  }
  
  res.status(404).json({ error: `Candidates for run ${runId} not found` });
});

// Get candidate detail
app.get('/api/candidate/:candidateId', (req, res) => {
  const { candidateId } = req.params;
  const bundlePath = path.join(ARTIFACTS_ROOT, 'candidates', candidateId);
  
  if (!fs.existsSync(bundlePath)) {
    // Return minimal info from CSV if bundle doesn't exist
    // Search through all top_candidates CSVs to find this candidate
    const topCandidatesDir = path.join(ARTIFACTS_ROOT, 'top_candidates');
    if (fs.existsSync(topCandidatesDir)) {
      const runDirs = fs.readdirSync(topCandidatesDir);
      for (const runDir of runDirs) {
        const csvPath = path.join(topCandidatesDir, runDir, 'top1000.csv');
        if (fs.existsSync(csvPath)) {
          const candidates = readCsvFile(csvPath);
          const found = candidates.find(c => c.candidate_id === candidateId);
          if (found) {
            return res.json({
              candidate_id: candidateId,
              genome_hash: found.genome_hash || '',
              rank: parseInt(found.rank) || 0,
              candidate_class: found.gates_passed === 'true' ? 'validated' : 'research',
              display_name: `Strategy #${found.rank} (No Bundle)`,
              oos_sharpe_net: parseFloat(found.oos_sharpe_net) || 0,
              oos_sharpe_gross: parseFloat(found.oos_sharpe_net) || 0,
              pbo: parseFloat(found.pbo) || 0,
              dsr: parseFloat(found.dsr) || 0,
              oos_cagr_net: parseFloat(found.oos_cagr_net) || 0,
              max_drawdown_net: parseFloat(found.max_drawdown_net) || 0,
              turnover_annual: parseFloat(found.turnover_annual) || 0,
              capacity_usd: found.capacity_usd ? parseFloat(found.capacity_usd) : null,
              stress_passed: parseInt(found.stress_passed) || 0,
              stress_total: parseInt(found.stress_total) || 0,
              gates_passed: found.gates_passed === 'true',
              strategy: null,
              strategy_toml: null,
              execution: null,
              provenance: { run_id: runDir.replace('run_', ''), created_at: found.created_at },
              bundle_path: null,
              bundle_missing: true,
              bundle_message: 'This candidate was not promoted to have a bundle. Run promotion or export to generate detailed artifacts.'
            });
          }
        }
      }
    }
    return res.status(404).json({ error: `Candidate ${candidateId} not found` });
  }
  
  // Load strategy.toml
  const strategyPath = path.join(bundlePath, 'strategy.toml');
  const strategy = readTomlFile(strategyPath);
  
  // Load validation_summary.json
  const validationPath = path.join(bundlePath, 'validation_summary.json');
  const validation = readJsonFile(validationPath) || {};
  
  // Load provenance.json
  const provenancePath = path.join(bundlePath, 'provenance.json');
  const provenance = readJsonFile(provenancePath) || {};
  
  // Load execution_config.toml
  const executionPath = path.join(bundlePath, 'execution_config.toml');
  const executionToml = readTomlFile(executionPath);
  const execution = executionToml?.execution || {};
  
  // Read strategy.toml as raw text
  let strategyTomlRaw = '';
  try {
    strategyTomlRaw = fs.readFileSync(strategyPath, 'utf-8');
  } catch (e) {}
  
  const displayName = generateDisplayName(strategy?.strategy ? strategy : { pipeline: strategy?.pipeline });
  
  // Build pipeline blocks from TOML
  const pipelineBlocks = (strategy?.pipeline || []).map(block => ({
    block_type: block.type || 'unknown',
    block_id: block.block_id || 'unknown',
    enabled: block.enabled !== false,
    params: block.params || {}
  }));
  
  res.json({
    candidate_id: candidateId,
    genome_hash: provenance.genome_hash || '',
    rank: 0,
    candidate_class: validation.gates_passed ? 'validated' : 'research',
    display_name: displayName,
    
    // Metrics
    oos_sharpe_net: validation.oos_sharpe_net || 0,
    oos_sharpe_gross: validation.oos_sharpe_gross || 0,
    pbo: validation.pbo || 0,
    dsr: validation.dsr || 0,
    oos_cagr_net: validation.oos_cagr_net || 0,
    max_drawdown_net: validation.max_drawdown_net || 0,
    turnover_annual: validation.turnover_annual || 0,
    capacity_usd: validation.capacity_usd || null,
    
    // Stress & Gates
    stress_passed: validation.stress_passed || 0,
    stress_total: validation.stress_total || 0,
    gates_passed: validation.gates_passed || false,
    
    // Strategy
    strategy: {
      id: strategy?.strategy?.id || candidateId,
      version: strategy?.strategy?.version || '1.0',
      description: strategy?.strategy?.description || '',
      author: strategy?.strategy?.author || 'SCG',
      pipeline: pipelineBlocks,
      rebalance: strategy?.rebalance || {},
      constraints: strategy?.constraints || {}
    },
    strategy_toml: strategyTomlRaw,
    
    // Execution
    execution: {
      delay_bars: execution.delay_bars || 1,
      bypass_for_debug: execution.bypass_for_debug || false,
      slippage: execution.slippage || { slippage_type: 'Constant', bps: 5 },
      fees: execution.fees || { tier: 'B3Retail' }
    },
    
    // Provenance
    provenance: {
      git_sha: provenance.git_sha || null,
      dataset_hash: provenance.dataset_hash || null,
      config_hash: provenance.config_hash || null,
      run_id: provenance.run_id || null,
      campaign_id: provenance.campaign_id || null,
      seed: provenance.seed || null,
      created_at: provenance.created_at || null
    },
    
    // Paths
    bundle_path: bundlePath,
    strategy_toml_path: strategyPath,
    validation_summary_path: validationPath
  });
});

// Get backtest timeseries
app.get('/api/backtest/:candidateId', (req, res) => {
  const { candidateId } = req.params;
  
  // Look for timeseries in various locations
  const possiblePaths = [
    path.join(ARTIFACTS_ROOT, 'backtests', candidateId, 'timeseries.csv'),
    path.join(ARTIFACTS_ROOT, 'candidates', candidateId, 'backtest', 'timeseries.csv'),
    path.join(PROJECT_ROOT, 'output', 'backtests', candidateId, 'timeseries.csv')
  ];
  
  for (const tsPath of possiblePaths) {
    if (fs.existsSync(tsPath)) {
      const timeseries = readCsvFile(tsPath).map(row => ({
        date: row.date,
        equity: parseFloat(row.equity) || 1,
        drawdown: parseFloat(row.drawdown) || 0,
        exposure: parseFloat(row.exposure) || null,
        vol_exante: parseFloat(row.vol_exante) || null,
        vol_expost: parseFloat(row.vol_expost) || null
      }));
      
      return res.json({
        available: true,
        candidate_id: candidateId,
        message: null,
        timeseries,
        backtest_path: path.dirname(tsPath)
      });
    }
  }
  
  // No backtest found - return graceful degradation
  res.json({
    available: false,
    candidate_id: candidateId,
    message: 'No backtest data found. Run replay to generate.',
    timeseries: [],
    backtest_path: null
  });
});

// =============================================================================
// START SERVER
// =============================================================================

app.listen(PORT, () => {
  console.log(`\n✅ API Server running at http://localhost:${PORT}`);
  console.log(`\n📊 Endpoints:`);
  console.log(`   GET  /api/health`);
  console.log(`   POST /api/set-root`);
  console.log(`   GET  /api/index`);
  console.log(`   GET  /api/campaign/:id`);
  console.log(`   GET  /api/run/:id`);
  console.log(`   GET  /api/candidates/:runId`);
  console.log(`   GET  /api/candidate/:id`);
  console.log(`   GET  /api/backtest/:id`);
  console.log(`\n🔗 Frontend: http://localhost:5173`);
});

