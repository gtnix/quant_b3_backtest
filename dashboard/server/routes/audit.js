import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { pool, readJsonFile, getArtifactsRoot } from '../db.js';

const router = Router();

router.get('/audits', async (req, res) => {
  try {
    const auditsPath = path.join(getArtifactsRoot(), 'audits');
    if (fs.existsSync(auditsPath)) {
      const dirs = fs.readdirSync(auditsPath).filter(d => d.startsWith('audit_'));
      const audits = dirs.map(d => {
        const summaryPath = path.join(auditsPath, d, 'summary.json');
        const summary = readJsonFile(summaryPath);
        return { run_id: d.replace('audit_', ''), path: path.join(auditsPath, d), has_summary: !!summary, ...summary };
      }).sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
      return res.json({ audits, count: audits.length, source: 'filesystem' });
    }
    res.json({ audits: [], count: 0, source: 'none' });
  } catch (e) { res.status(500).json({ error: e.message, audits: [] }); }
});

router.get('/audit/:runId', async (req, res) => {
  try {
    const auditPath = path.join(getArtifactsRoot(), 'audits', `audit_${req.params.runId}`);
    if (fs.existsSync(auditPath)) {
      const summary = readJsonFile(path.join(auditPath, 'summary.json'));
      const candidates = readJsonFile(path.join(auditPath, 'candidates.json'));
      const gates = readJsonFile(path.join(auditPath, 'gates.json'));
      const files = fs.readdirSync(auditPath);
      return res.json({ run_id: req.params.runId, path: auditPath, summary, candidates, gates, files, source: 'filesystem' });
    }
    // Try database
    const result = await pool.query(`SELECT r.*, c.name as campaign_name FROM scg_runs r LEFT JOIN scg_campaigns c ON r.campaign_id = c.campaign_id WHERE r.run_id = $1`, [req.params.runId]);
    if (result.rows.length > 0) {
      const r = result.rows[0];
      const cands = await pool.query(`SELECT candidate_id, rank_in_run, oos_sharpe_net, pbo, gates_passed FROM scg_candidates WHERE run_id = $1 ORDER BY rank_in_run LIMIT 50`, [req.params.runId]);
      return res.json({ run_id: req.params.runId, summary: { campaign_name: r.campaign_name, status: r.status, started_at: r.started_at, completed_at: r.completed_at, duration_secs: r.duration_secs, generations_completed: r.generations_completed, best_oos_sharpe_net: r.best_oos_sharpe_net }, candidates: cands.rows, source: 'neon' });
    }
    res.status(404).json({ error: `Audit ${req.params.runId} not found` });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

router.get('/stats/production', async (req, res) => {
  try {
    const [totalCands, hofCount, last24h, last7d] = await Promise.all([
      pool.query('SELECT COUNT(*) as count FROM scg_candidates'),
      pool.query("SELECT COUNT(*) as count FROM scg_promotions WHERE promotion_class = 'hall_of_fame'"),
      pool.query("SELECT COUNT(*) as count FROM scg_candidates WHERE created_at > NOW() - INTERVAL '24 hours'"),
      pool.query("SELECT COUNT(*) as count FROM scg_candidates WHERE created_at > NOW() - INTERVAL '7 days'")
    ]);
    res.json({ totalCandidates: parseInt(totalCands.rows[0].count) || 0, hallOfFameCount: parseInt(hofCount.rows[0].count) || 0, candidates24h: parseInt(last24h.rows[0].count) || 0, candidates7d: parseInt(last7d.rows[0].count) || 0 });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

export default router;

