import { Router } from 'express';
import { pool } from '../db.js';

const router = Router();

router.get('/strategies', async (req, res) => {
  try {
    const result = await pool.query(`
      SELECT id, slug, family_id, name, description, timeframe, bar_interval, position_type, 
             risk_profile, tooltip_short, tooltip_long, difficulty_level, tags, enabled, is_default, usage_count
      FROM strategy_templates WHERE enabled = true ORDER BY family_id, name
    `);
    res.json(result.rows);
  } catch (e) { res.status(500).json({ error: 'Failed to fetch strategies' }); }
});

router.get('/strategies/families', async (req, res) => {
  try {
    const result = await pool.query(`SELECT id, slug, name, description, icon, color, hypothesis, sort_order FROM strategy_families ORDER BY sort_order`);
    res.json(result.rows);
  } catch (e) { res.status(500).json({ error: 'Failed to fetch families' }); }
});

router.get('/strategies/:slug', async (req, res) => {
  try {
    const result = await pool.query(`
      SELECT s.*, f.name as family_name, f.color as family_color
      FROM strategy_templates s LEFT JOIN strategy_families f ON s.family_id = f.id WHERE s.slug = $1
    `, [req.params.slug]);
    if (result.rows.length === 0) return res.status(404).json({ error: 'Strategy not found' });
    res.json(result.rows[0]);
  } catch (e) { res.status(500).json({ error: 'Failed to fetch strategy' }); }
});

router.get('/catalogs', async (req, res) => {
  try {
    const result = await pool.query(`SELECT id, slug, name, description, icon, is_system, is_default FROM strategy_catalogs ORDER BY is_default DESC, name`);
    res.json(result.rows);
  } catch (e) { res.status(500).json({ error: 'Failed to fetch catalogs' }); }
});

router.get('/catalogs/:slug/strategies', async (req, res) => {
  try {
    const result = await pool.query(`
      SELECT s.id, s.slug, s.name, s.family_id, s.timeframe, s.risk_profile
      FROM strategy_templates s
      INNER JOIN catalog_strategies cs ON s.id = cs.strategy_id
      INNER JOIN strategy_catalogs c ON cs.catalog_id = c.id
      WHERE c.slug = $1 AND s.enabled = true ORDER BY cs.priority, s.name
    `, [req.params.slug]);
    res.json(result.rows);
  } catch (e) { res.status(500).json({ error: 'Failed to fetch catalog strategies' }); }
});

router.patch('/strategies/:slug/toggle', async (req, res) => {
  try {
    const result = await pool.query(`UPDATE strategy_templates SET enabled = NOT enabled, updated_at = NOW() WHERE slug = $1 RETURNING id, slug, enabled`, [req.params.slug]);
    if (result.rows.length === 0) return res.status(404).json({ error: 'Strategy not found' });
    res.json(result.rows[0]);
  } catch (e) { res.status(500).json({ error: 'Failed to toggle strategy' }); }
});

router.post('/strategies/:slug/use', async (req, res) => {
  try {
    await pool.query(`UPDATE strategy_templates SET usage_count = usage_count + 1, updated_at = NOW() WHERE slug = $1`, [req.params.slug]);
    res.json({ success: true });
  } catch (e) { res.status(500).json({ error: 'Failed to update usage' }); }
});

export default router;

