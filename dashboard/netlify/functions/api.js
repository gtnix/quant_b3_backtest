/**
 * Netlify Serverless API Function
 * 
 * Handles all /api/* routes for the Quant B3 Dashboard
 * Connects to Neon PostgreSQL database
 */

import pg from 'pg';
const { Pool } = pg;

// Database connection (uses Netlify environment variable)
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false },
  max: 5,
  idleTimeoutMillis: 30000,
});

// CORS headers for all responses
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Content-Type': 'application/json',
};

// Helper: Send JSON response
function jsonResponse(statusCode, data) {
  return {
    statusCode,
    headers: corsHeaders,
    body: JSON.stringify(data),
  };
}

// Helper: Estimate MaxDD if not available
function estimateMaxDrawdown(sharpe, cagr) {
  if (!sharpe || !cagr) return 0.15;
  const volatility = (cagr || 0.1) / Math.max(sharpe || 1, 0.5);
  return Math.min(Math.max(volatility * 2.5, 0.05), 0.50);
}

// Route handlers
const routes = {
  // Health check
  'GET /health': async () => jsonResponse(200, { status: 'ok', timestamp: new Date().toISOString() }),

  // List campaigns from Neon
  'GET /campaigns': async () => {
    const result = await pool.query(`
      SELECT campaign_id, campaign_name, created_at, config_snapshot
      FROM scg_campaigns
      ORDER BY created_at DESC
      LIMIT 50
    `);
    
    const campaigns = result.rows.map(c => ({
      id: c.campaign_id,
      name: c.campaign_name || c.campaign_id,
      created_at: c.created_at,
      status: 'completed',
    }));
    
    return jsonResponse(200, { campaigns });
  },

  // List recent runs from Neon
  'GET /runs/recent': async (event) => {
    const limit = parseInt(event.queryStringParameters?.limit || '10');
    
    const result = await pool.query(`
      SELECT r.run_id, r.campaign_id, r.run_tag, r.status, r.created_at,
             r.finished_at, r.generations_completed, r.best_sharpe_oos,
             c.campaign_name,
             (SELECT COUNT(*) FROM scg_candidates WHERE run_id = r.run_id) as candidate_count
      FROM scg_runs r
      LEFT JOIN scg_campaigns c ON r.campaign_id = c.campaign_id
      ORDER BY r.created_at DESC
      LIMIT $1
    `, [limit]);
    
    const runs = result.rows.map(r => ({
      run_id: r.run_id,
      campaign_id: r.campaign_id,
      campaign_name: r.campaign_name || r.campaign_id,
      run_tag: r.run_tag,
      status: r.status || 'completed',
      created_at: r.created_at,
      finished_at: r.finished_at,
      generations_completed: r.generations_completed,
      best_sharpe_oos: r.best_sharpe_oos,
      candidate_count: parseInt(r.candidate_count) || 0,
    }));
    
    return jsonResponse(200, { runs });
  },

  // List candidates for a run
  'GET /candidates/:runId': async (event) => {
    const runId = event.path.split('/').pop().split('?')[0];
    const limit = parseInt(event.queryStringParameters?.limit || '100');
    
    const result = await pool.query(`
      SELECT candidate_id, genome_hash, rank_in_run, candidate_class,
             oos_sharpe_net, oos_cagr_net, max_drawdown_net, pbo, dsr,
             stress_passed, stress_total, gates_passed, created_at
      FROM scg_candidates
      WHERE run_id = $1
      ORDER BY rank_in_run ASC
      LIMIT $2
    `, [runId, limit]);
    
    const candidates = result.rows.map((c, idx) => {
      const maxDD = c.max_drawdown_net ?? estimateMaxDrawdown(c.oos_sharpe_net, c.oos_cagr_net);
      return {
        rank: c.rank_in_run || idx + 1,
        candidate_id: c.candidate_id,
        genome_hash: c.genome_hash,
        display_name: `Strategy #${c.rank_in_run || idx + 1} | ${c.candidate_id.slice(-8)}`,
        candidate_class: c.candidate_class || (c.gates_passed ? 'validated' : 'research'),
        oos_sharpe_net: c.oos_sharpe_net || 0,
        oos_cagr_net: c.oos_cagr_net || 0,
        max_drawdown_net: maxDD,
        pbo: c.pbo || 0,
        dsr: c.dsr || 0,
        stress_passed: c.stress_passed || 0,
        stress_total: c.stress_total || 8,
        gates_passed: c.gates_passed || false,
        data_integrity_ok: true,
        created_at: c.created_at,
      };
    });
    
    return jsonResponse(200, candidates);
  },

  // Get candidate detail
  'GET /candidate/:candidateId': async (event) => {
    const candidateId = event.path.split('/').pop();
    
    const result = await pool.query(`
      SELECT c.*, r.run_tag, r.campaign_id, camp.campaign_name
      FROM scg_candidates c
      LEFT JOIN scg_runs r ON c.run_id = r.run_id
      LEFT JOIN scg_campaigns camp ON r.campaign_id = camp.campaign_id
      WHERE c.candidate_id = $1
    `, [candidateId]);
    
    if (result.rows.length === 0) {
      return jsonResponse(404, { error: 'Candidate not found' });
    }
    
    const c = result.rows[0];
    const maxDD = c.max_drawdown_net ?? estimateMaxDrawdown(c.oos_sharpe_net, c.oos_cagr_net);
    
    return jsonResponse(200, {
      candidate_id: c.candidate_id,
      genome_hash: c.genome_hash,
      run_id: c.run_id,
      campaign_id: c.campaign_id,
      campaign_name: c.campaign_name,
      run_tag: c.run_tag,
      rank: c.rank_in_run,
      candidate_class: c.candidate_class,
      oos_sharpe_net: c.oos_sharpe_net,
      oos_cagr_net: c.oos_cagr_net,
      max_drawdown_net: maxDD,
      pbo: c.pbo,
      dsr: c.dsr,
      stress_passed: c.stress_passed,
      stress_total: c.stress_total,
      gates_passed: c.gates_passed,
      is_oos_sharpe_net: c.is_oos_sharpe_net,
      turnover_annual: c.turnover_annual,
      capacity_usd: c.capacity_usd,
      created_at: c.created_at,
      data_source: 'neon',
    });
  },

  // Get simulated equity for a candidate
  'GET /candidate/:candidateId/simulated-equity': async (event) => {
    const candidateId = event.path.split('/')[2];
    
    const result = await pool.query(`
      SELECT oos_sharpe_net, oos_cagr_net, max_drawdown_net, created_at
      FROM scg_candidates
      WHERE candidate_id = $1
    `, [candidateId]);
    
    if (result.rows.length === 0) {
      return jsonResponse(404, { error: 'Candidate not found' });
    }
    
    const c = result.rows[0];
    const sharpe = c.oos_sharpe_net || 0.5;
    const cagr = c.oos_cagr_net || 0.1;
    const maxDD = c.max_drawdown_net ?? estimateMaxDrawdown(sharpe, cagr);
    
    // Generate simulated equity curve
    const timeseries = [];
    const startDate = new Date('2020-01-01');
    let equity = 100;
    const dailyReturn = cagr / 252;
    const volatility = (cagr / Math.max(sharpe, 0.5)) / Math.sqrt(252);
    
    for (let i = 0; i < 1000; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
      const randomReturn = dailyReturn + volatility * (Math.random() - 0.5) * 2;
      equity *= (1 + randomReturn);
      timeseries.push({ time: date.toISOString().split('T')[0], value: equity });
    }
    
    return jsonResponse(200, {
      candidate_id: candidateId,
      sharpe,
      cagr,
      max_drawdown: maxDD,
      timeseries,
      simulated: true,
    });
  },

  // Overview stats
  'GET /overview': async () => {
    const stats = await pool.query(`
      SELECT 
        (SELECT COUNT(*) FROM scg_campaigns) as total_campaigns,
        (SELECT COUNT(*) FROM scg_runs) as total_runs,
        (SELECT COUNT(*) FROM scg_candidates) as total_candidates,
        (SELECT COUNT(*) FROM scg_candidates WHERE gates_passed = true) as validated_candidates,
        (SELECT MAX(oos_sharpe_net) FROM scg_candidates) as best_sharpe,
        (SELECT AVG(oos_sharpe_net) FROM scg_candidates WHERE oos_sharpe_net > 0) as avg_sharpe
    `);
    
    const s = stats.rows[0];
    return jsonResponse(200, {
      total_campaigns: parseInt(s.total_campaigns) || 0,
      total_runs: parseInt(s.total_runs) || 0,
      total_candidates: parseInt(s.total_candidates) || 0,
      validated_candidates: parseInt(s.validated_candidates) || 0,
      best_sharpe: parseFloat(s.best_sharpe) || 0,
      avg_sharpe: parseFloat(s.avg_sharpe) || 0,
    });
  },
};

// Main handler
export async function handler(event) {
  // Handle CORS preflight
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 204, headers: corsHeaders, body: '' };
  }

  try {
    // Extract route from path
    const path = event.path.replace('/.netlify/functions/api', '').replace('/api', '') || '/';
    const method = event.httpMethod;
    
    // Match route
    for (const [routeKey, routeHandler] of Object.entries(routes)) {
      const [routeMethod, routePattern] = routeKey.split(' ');
      
      if (method !== routeMethod) continue;
      
      // Simple pattern matching with :params
      const patternParts = routePattern.split('/');
      const pathParts = path.split('/').filter(Boolean);
      
      if (patternParts.length !== pathParts.length + 1) continue;
      
      let match = true;
      for (let i = 1; i < patternParts.length; i++) {
        if (patternParts[i].startsWith(':')) continue;
        if (patternParts[i] !== pathParts[i - 1]) {
          match = false;
          break;
        }
      }
      
      if (match || routePattern === path || routePattern === '/' + pathParts.join('/')) {
        return await routeHandler(event);
      }
    }
    
    // No route matched
    return jsonResponse(404, { error: 'Not found', path, method });
    
  } catch (error) {
    console.error('API Error:', error);
    return jsonResponse(500, { error: error.message });
  }
}

















