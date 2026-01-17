/**
 * Hall of Fame Sync Service - Optimized
 * 
 * This service handles Hall of Fame persistence with two data sources:
 * 1. Local files: output/scg/run_xxx/hall_of_fame/
 * 2. Global database: hall_of_fame table in Neon
 * 
 * The global hall_of_fame table is the source of truth for cumulative best strategies.
 * Local files are synced to scg_candidates and scg_promotions tables.
 */
import fs from 'fs';
import path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';
import pg from 'pg';

const execAsync = promisify(exec);
const { Pool } = pg;
const HOF_LIMIT = 50;
const PROJECT_ROOT = path.resolve(process.cwd(), '..');
const OUTPUT_DIR = path.join(PROJECT_ROOT, 'output/scg');

let pool = null;

function getPool() {
  if (!pool && process.env.DATABASE_URL) {
    pool = new Pool({
      connectionString: process.env.DATABASE_URL,
      ssl: { rejectUnauthorized: false },
      max: 5
    });
  }
  return pool;
}

// =============================================================================
// GLOBAL HALL OF FAME (from database - cumulative across all runs)
// =============================================================================

/**
 * Fetch the global Hall of Fame from the database.
 * This is the cumulative top 50 strategies across all runs.
 * @param {string} market - Market filter ('BR' or 'US')
 * @param {number} limit - Maximum entries to return
 * @returns {Promise<Array>} Array of HoF entries from database
 */
export async function fetchGlobalHallOfFame(market = 'BR', limit = HOF_LIMIT) {
  const p = getPool();
  if (!p) {
    console.warn('[HoF] DATABASE_URL not configured, returning empty global HoF');
    return [];
  }
  
  try {
    const result = await p.query(`
      SELECT 
        id,
        genome_hash,
        candidate_id,
        oos_sharpe_net as sharpe,
        oos_cagr_net as cagr,
        max_drawdown_net as max_dd,
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
        global_rank as rank
      FROM hall_of_fame
      WHERE market = $1
      ORDER BY oos_sharpe_net DESC
      LIMIT $2
    `, [market, limit]);
    
    console.log(`[HoF] Fetched ${result.rows.length} strategies from global hall_of_fame (market: ${market})`);
    return result.rows;
  } catch (err) {
    console.error('[HoF] Failed to fetch global hall_of_fame:', err.message);
    return [];
  }
}

/**
 * Get Hall of Fame status (count and top entry for each market).
 * @returns {Promise<Object>} Status object with counts and top entries
 */
export async function getHallOfFameStatus() {
  const p = getPool();
  if (!p) {
    return { connected: false, markets: {} };
  }
  
  try {
    const countResult = await p.query(`
      SELECT market, COUNT(*) as count, MAX(oos_sharpe_net) as top_sharpe
      FROM hall_of_fame
      GROUP BY market
    `);
    
    const markets = {};
    for (const row of countResult.rows) {
      markets[row.market] = {
        count: parseInt(row.count),
        topSharpe: parseFloat(row.top_sharpe)
      };
    }
    
    return { connected: true, markets };
  } catch (err) {
    console.error('[HoF] Failed to get status:', err.message);
    return { connected: false, error: err.message, markets: {} };
  }
}

/**
 * Get unified Hall of Fame from both local files AND global database.
 * Merges and deduplicates by genome_hash, keeping the best metrics.
 * @param {string} market - Market filter
 * @returns {Promise<Array>} Merged and deduplicated strategies
 */
export async function getUnifiedHallOfFame(market = 'BR') {
  // Fetch from both sources in parallel
  const [globalStrategies, localStrategies] = await Promise.all([
    fetchGlobalHallOfFame(market, HOF_LIMIT),
    scanLocalStrategies()
  ]);
  
  // Merge by genome_hash, keeping best sharpe
  const byHash = new Map();
  
  for (const s of globalStrategies) {
    byHash.set(s.genome_hash, {
      ...s,
      source: 'global'
    });
  }
  
  for (const s of localStrategies) {
    const existing = byHash.get(s.genomeHash);
    if (!existing || s.sharpe > existing.sharpe) {
      byHash.set(s.genomeHash, {
        ...s,
        genome_hash: s.genomeHash,
        source: 'local'
      });
    }
  }
  
  // Convert to array, sort by sharpe, limit
  const merged = Array.from(byHash.values())
    .sort((a, b) => (b.sharpe || 0) - (a.sharpe || 0))
    .slice(0, HOF_LIMIT);
  
  console.log(`[HoF] Unified: ${globalStrategies.length} global + ${localStrategies.length} local = ${merged.length} unique`);
  return merged;
}

async function decompressObfs(filepath) {
  try {
    if (!fs.existsSync(filepath)) return null;
    const { stdout } = await execAsync(`zstd -d -c "${filepath}"`, { maxBuffer: 10 * 1024 * 1024 });
    return JSON.parse(stdout);
  } catch { return null; }
}

function readStrategyToml(filepath) {
  try {
    return fs.existsSync(filepath) ? fs.readFileSync(filepath, 'utf8') : null;
  } catch { return null; }
}

function extractGeneration(candidateId) {
  const match = candidateId.match(/gen(\d+)/);
  return match ? parseInt(match[1], 10) : 0;
}

// Quick scan - only reads TOML and metrics (fast)
export function scanLocalStrategiesQuick() {
  const strategies = [];
  if (!fs.existsSync(OUTPUT_DIR)) return strategies;
  
  const runs = fs.readdirSync(OUTPUT_DIR).filter(d => d.startsWith('run_') || d.startsWith('scg_'));
  
  for (const run of runs) {
    const hofPath = path.join(OUTPUT_DIR, run, 'hall_of_fame');
    if (!fs.existsSync(hofPath)) continue;
    
    // Only check strategy_000 to strategy_049 (top 50 per run)
    for (let i = 0; i < 50; i++) {
      const slot = `strategy_${String(i).padStart(3, '0')}`;
      const stratDir = path.join(hofPath, slot);
      // Try both naming conventions
      let tomlPath = path.join(stratDir, 'strategy.toml');
      if (!fs.existsSync(tomlPath)) {
        tomlPath = path.join(stratDir, 'config.toml');
      }
      
      if (!fs.existsSync(tomlPath)) continue;
      
      const toml = readStrategyToml(tomlPath);
      if (!toml) continue;
      
      const idMatch = toml.match(/id\s*=\s*"([^"]+)"/);
      const candidateId = idMatch ? idMatch[1] : `unknown_${run}_${i}`;
      const genomeHash = candidateId.split('_').pop();
      
      strategies.push({
        candidateId,
        runId: run,
        genomeHash,
        generation: extractGeneration(candidateId),
        slot: i,
        stratDir,
        tomlPath
      });
    }
  }
  
  return strategies;
}

// Full scan with metrics (async, for sync to DB)
export async function scanLocalStrategies() {
  const quickList = scanLocalStrategiesQuick();
  const strategies = [];
  
  // Process in parallel batches of 10
  const batchSize = 10;
  for (let i = 0; i < quickList.length; i += batchSize) {
    const batch = quickList.slice(i, i + batchSize);
    const results = await Promise.all(batch.map(async (s) => {
      const metricsPath = path.join(s.stratDir, 'metrics.obfs');
      const metrics = await decompressObfs(metricsPath);
      if (!metrics) return null;
      
      return {
        ...s,
        sharpe: metrics.sharpe_ratio || 0,
        cagr: metrics.cagr || 0,
        maxDd: metrics.max_drawdown || 0,
        pbo: metrics.pbo || 0,
        dsr: metrics.dsr || 0
      };
    }));
    
    strategies.push(...results.filter(Boolean));
  }
  
  // Sort and dedupe
  strategies.sort((a, b) => b.sharpe - a.sharpe);
  
  const seen = new Set();
  const unique = [];
  for (const s of strategies) {
    const key = `${s.sharpe.toFixed(4)}_${s.cagr.toFixed(4)}_${s.maxDd.toFixed(4)}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(s);
    }
    if (unique.length >= HOF_LIMIT) break;
  }
  
  return unique;
}

async function loadFullStrategy(s) {
  const genomePath = path.join(s.stratDir, 'genome.obfs');
  const pboPath = path.join(s.stratDir, 'pbo_dsr.obfs');
  const wfaPath = path.join(s.stratDir, 'wfa_report.obfs');
  const stressPath = path.join(s.stratDir, 'stress_report.obfs');
  const validationPath = path.join(s.stratDir, 'validation_bundle.obfs');
  
  const [genome, pboDsr, wfa, stress, validation] = await Promise.all([
    decompressObfs(genomePath),
    decompressObfs(pboPath),
    decompressObfs(wfaPath),
    decompressObfs(stressPath),
    decompressObfs(validationPath)
  ]);
  
  const toml = readStrategyToml(s.tomlPath);
  const freqMatch = toml?.match(/frequency\s*=\s*"([^"]+)"/);
  const dayMatch = toml?.match(/day\s*=\s*"([^"]+)"/);
  
  return {
    ...s,
    strategyToml: toml,
    genome,
    wfa,
    stress,
    validation,
    pbo: pboDsr?.pbo || s.pbo,
    dsr: pboDsr?.dsr || s.dsr,
    rebalanceFreq: freqMatch?.[1] || 'weekly',
    rebalanceDay: dayMatch?.[1] || 'friday',
    pipelineBlocks: genome?.genes?.length || 0,
    stressPassed: stress?.passed || 0,
    stressTotal: stress?.total || 5
  };
}

async function ensureRunExists(client, runId) {
  await client.query(`
    INSERT INTO scg_runs (run_id, campaign_id, seed, status, machine_origin)
    VALUES ($1, 'camp_maxpower_local', 42, 'completed', 'local_sync')
    ON CONFLICT (run_id) DO NOTHING
  `, [runId]);
}

async function syncToDatabase(strategies) {
  const p = getPool();
  if (!p) throw new Error('DATABASE_URL not configured');
  
  const client = await p.connect();
  let synced = 0;
  
  try {
    await client.query('BEGIN');
    
    for (let i = 0; i < strategies.length; i++) {
      const s = await loadFullStrategy(strategies[i]);
      
      await ensureRunExists(client, s.runId);
      
      await client.query(`
        INSERT INTO scg_candidates (
          candidate_id, run_id, genome_hash, rank, rank_in_run,
          oos_sharpe_net, oos_cagr_net, max_drawdown_net, pbo, dsr,
          stress_passed, stress_total, gates_passed,
          candidate_class, strategy_name,
          genome_json, strategy_toml, wfa_report, stress_report, validation_bundle,
          rebalance_frequency, rebalance_day, pipeline_blocks, generation
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
        ON CONFLICT (candidate_id) DO UPDATE SET
          rank = EXCLUDED.rank,
          oos_sharpe_net = EXCLUDED.oos_sharpe_net,
          oos_cagr_net = EXCLUDED.oos_cagr_net,
          max_drawdown_net = EXCLUDED.max_drawdown_net,
          pbo = EXCLUDED.pbo,
          dsr = EXCLUDED.dsr,
          genome_json = EXCLUDED.genome_json,
          strategy_toml = EXCLUDED.strategy_toml,
          generation = EXCLUDED.generation
      `, [
        s.candidateId, s.runId, s.genomeHash, i + 1, s.slot,
        s.sharpe, s.cagr, s.maxDd, s.pbo, s.dsr,
        s.stressPassed, s.stressTotal, true,
        'elite', `BR • MaxPower • #${s.genomeHash.slice(-6).toUpperCase()}`,
        s.genome ? JSON.stringify(s.genome) : null, s.strategyToml,
        s.wfa ? JSON.stringify(s.wfa) : null,
        s.stress ? JSON.stringify(s.stress) : null,
        s.validation ? JSON.stringify(s.validation) : null,
        s.rebalanceFreq, s.rebalanceDay, s.pipelineBlocks, s.generation
      ]);
      
      await client.query(`
        INSERT INTO scg_promotions (
          promotion_id, candidate_id, stage, promoted_by, promotion_class,
          oos_sharpe_net, cagr_net, max_drawdown_net, pbo, dsr,
          stress_passed, stress_total, gates_passed, market, strategy_name
        ) VALUES ($1, $2, 'hall_of_fame', 'auto_sync', 'hall_of_fame', $3, $4, $5, $6, $7, $8, $9, true, 'BR', $10)
        ON CONFLICT (candidate_id, stage) DO UPDATE SET
          oos_sharpe_net = EXCLUDED.oos_sharpe_net,
          cagr_net = EXCLUDED.cagr_net,
          max_drawdown_net = EXCLUDED.max_drawdown_net,
          promoted_at = NOW()
      `, [
        `promo_${s.genomeHash}_hof`, s.candidateId,
        s.sharpe, s.cagr, s.maxDd, s.pbo, s.dsr,
        s.stressPassed, s.stressTotal,
        `BR • MaxPower • #${s.genomeHash.slice(-6).toUpperCase()}`
      ]);
      
      synced++;
    }
    
    await client.query('COMMIT');
    console.log(`[HoF Sync] Synced ${synced} strategies to Neon`);
    return synced;
  } catch (err) {
    await client.query('ROLLBACK');
    console.error('[HoF Sync] Error:', err.message);
    throw err;
  } finally {
    client.release();
  }
}

export async function runSync() {
  console.log('[HoF Sync] Scanning local strategies...');
  const strategies = await scanLocalStrategies();
  console.log(`[HoF Sync] Found ${strategies.length} unique strategies`);
  
  if (strategies.length === 0) return { synced: 0, total: 0, globalHof: 0 };
  
  const synced = await syncToDatabase(strategies);
  
  // Also sync to global hall_of_fame table
  const globalSynced = await syncToGlobalHallOfFame(strategies);
  
  return { synced, total: strategies.length, globalHof: globalSynced };
}

/**
 * Sync strategies to the GLOBAL hall_of_fame table.
 * Only inserts if strategy beats current threshold or HoF is not full.
 * @param {Array} strategies - Local strategies to potentially promote
 * @returns {Promise<number>} Number of strategies promoted
 */
async function syncToGlobalHallOfFame(strategies) {
  const p = getPool();
  if (!p) return 0;
  
  const market = 'BR'; // Default market
  let promoted = 0;
  
  try {
    // Get current HoF count and threshold
    const countResult = await p.query(
      'SELECT COUNT(*) as count FROM hall_of_fame WHERE market = $1',
      [market]
    );
    const currentCount = parseInt(countResult.rows[0]?.count || 0);
    
    // Get threshold (worst sharpe in top 50)
    let threshold = null;
    if (currentCount >= HOF_LIMIT) {
      const thresholdResult = await p.query(`
        SELECT oos_sharpe_net FROM hall_of_fame 
        WHERE market = $1 
        ORDER BY oos_sharpe_net DESC 
        OFFSET $2 LIMIT 1
      `, [market, HOF_LIMIT - 1]);
      threshold = thresholdResult.rows[0]?.oos_sharpe_net || 0;
    }
    
    for (const s of strategies) {
      // Skip if below threshold and HoF is full
      if (threshold !== null && s.sharpe <= threshold) continue;
      
      // Check if already in global HoF
      const existsResult = await p.query(
        'SELECT 1 FROM hall_of_fame WHERE genome_hash = $1',
        [s.genomeHash]
      );
      if (existsResult.rows.length > 0) continue;
      
      // Load full strategy data
      const full = await loadFullStrategy(s);
      
      // Insert into global hall_of_fame
      await p.query(`
        INSERT INTO hall_of_fame (
          genome_hash, candidate_id, oos_sharpe_net, oos_cagr_net,
          max_drawdown_net, pbo, dsr, stress_passed, stress_total,
          gates_passed, run_id, market, strategy_toml, genome_json
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        ON CONFLICT (genome_hash) DO UPDATE SET
          oos_sharpe_net = GREATEST(hall_of_fame.oos_sharpe_net, EXCLUDED.oos_sharpe_net),
          promoted_at = CASE 
            WHEN EXCLUDED.oos_sharpe_net > hall_of_fame.oos_sharpe_net THEN NOW()
            ELSE hall_of_fame.promoted_at
          END
      `, [
        s.genomeHash, s.candidateId, s.sharpe, s.cagr,
        s.maxDd, s.pbo || full.pbo, s.dsr || full.dsr,
        full.stressPassed, full.stressTotal,
        true, s.runId, market, full.strategyToml,
        full.genome ? JSON.stringify(full.genome) : null
      ]);
      
      promoted++;
    }
    
    // Prune if over limit
    if (promoted > 0) {
      await p.query(`
        DELETE FROM hall_of_fame
        WHERE id IN (
          SELECT id FROM hall_of_fame
          WHERE market = $1
          ORDER BY oos_sharpe_net ASC
          LIMIT GREATEST(0, (SELECT COUNT(*) FROM hall_of_fame WHERE market = $1) - $2)
        )
      `, [market, HOF_LIMIT]);
    }
    
    if (promoted > 0) {
      console.log(`[HoF Sync] Promoted ${promoted} strategies to global hall_of_fame`);
    }
    
    return promoted;
  } catch (err) {
    console.error('[HoF Sync] Failed to sync to global hall_of_fame:', err.message);
    return 0;
  }
}

let syncInterval = null;

export function startAutoSync(intervalMs = 5 * 60 * 1000) {
  if (syncInterval) return;
  console.log(`[HoF Sync] Auto-sync enabled (every ${intervalMs / 1000}s)`);
  syncInterval = setInterval(() => runSync().catch(e => console.error('[HoF Sync]', e.message)), intervalMs);
  // Delayed initial sync to not block server startup
  setTimeout(() => runSync().catch(e => console.error('[HoF Sync]', e.message)), 5000);
}

export function stopAutoSync() {
  if (syncInterval) {
    clearInterval(syncInterval);
    syncInterval = null;
  }
}
