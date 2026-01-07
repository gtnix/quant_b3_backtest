import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import toml from 'toml';
import { getWorkspaceRoot } from '../db.js';

const router = Router();

router.get('/omp/universe/compatibility', (req, res) => {
  try {
    const matrixPath = path.join(getWorkspaceRoot(), 'configs', 'compatibility_matrix.toml');
    if (!fs.existsSync(matrixPath)) return res.json({ matrix: {}, source: 'not_found' });
    const content = fs.readFileSync(matrixPath, 'utf-8');
    const matrix = toml.parse(content);
    res.json({ matrix, source: 'file' });
  } catch (e) { res.status(500).json({ error: 'Failed to load compatibility matrix' }); }
});

router.get('/omp/universe/training-strategies', (req, res) => {
  try {
    const dir = path.join(getWorkspaceRoot(), 'configs', 'training_strategies');
    if (!fs.existsSync(dir)) return res.json({ strategies: [], source: 'not_found' });
    const files = fs.readdirSync(dir).filter(f => f.endsWith('.toml'));
    const strategies = files.map(f => { try { return { name: f.replace('.toml', ''), ...toml.parse(fs.readFileSync(path.join(dir, f), 'utf-8')) }; } catch (e) { return { name: f.replace('.toml', ''), error: true }; } });
    res.json({ strategies, count: strategies.length, source: 'file' });
  } catch (e) { res.status(500).json({ error: 'Failed to load training strategies' }); }
});

router.get('/omp/universe/training-tech', (req, res) => {
  try {
    const dir = path.join(getWorkspaceRoot(), 'configs', 'training_tech');
    if (!fs.existsSync(dir)) return res.json({ profiles: [], source: 'not_found' });
    const files = fs.readdirSync(dir).filter(f => f.endsWith('.toml'));
    const profiles = files.map(f => { try { return { name: f.replace('.toml', ''), ...toml.parse(fs.readFileSync(path.join(dir, f), 'utf-8')) }; } catch (e) { return { name: f.replace('.toml', ''), error: true }; } });
    res.json({ profiles, count: profiles.length, source: 'file' });
  } catch (e) { res.status(500).json({ error: 'Failed to load training tech profiles' }); }
});

router.get('/omp/universe/restrictions/:profileName', (req, res) => {
  try {
    const profilePath = path.join(getWorkspaceRoot(), 'configs', 'risk_profiles', `${req.params.profileName}.toml`);
    if (!fs.existsSync(profilePath)) return res.status(404).json({ error: `Risk profile '${req.params.profileName}' not found` });
    const content = fs.readFileSync(profilePath, 'utf-8');
    const restrictions = { allowed_strategy_families: ['swing', 'momentum', 'position'], max_parameters_to_optimize: 10, max_population_size: 200, max_generations: 150 };
    let inSection = false;
    for (const line of content.split('\n')) {
      const t = line.trim();
      if (t === '[universe_restrictions]') inSection = true;
      else if (t.startsWith('[') && t.endsWith(']')) inSection = false;
      else if (inSection && t.includes('=')) { const [k, v] = t.split('=').map(s => s.trim()); restrictions[k] = v.startsWith('[') ? JSON.parse(v.replace(/'/g, '"')) : (parseInt(v) || v); }
    }
    res.json({ profileName: req.params.profileName, restrictions, source: 'file' });
  } catch (e) { res.status(500).json({ error: 'Failed to load restrictions' }); }
});

router.get('/omp/universe/strategies', (req, res) => {
  try {
    const registryPath = path.join(getWorkspaceRoot(), 'configs', 'universe', 'strategy_registry.toml');
    if (!fs.existsSync(registryPath)) return res.json({ strategies: {}, metadata: { total_strategies: 0 }, source: 'not_found' });
    const registry = toml.parse(fs.readFileSync(registryPath, 'utf-8'));
    res.json({ strategies: registry.strategies || {}, metadata: registry.metadata || {}, index: registry.index || {}, source: 'file' });
  } catch (e) { res.status(500).json({ error: 'Failed to load strategy registry' }); }
});

router.get('/omp/universe/timeframe-profiles', (req, res) => {
  try {
    const profilesPath = path.join(getWorkspaceRoot(), 'configs', 'universe', 'timeframe_profiles.toml');
    if (!fs.existsSync(profilesPath)) return res.json({ profiles: {}, source: 'not_found' });
    const profiles = toml.parse(fs.readFileSync(profilesPath, 'utf-8'));
    res.json({ profiles: profiles.profiles || {}, regime_detection: profiles.regime_detection || {}, data_windows: profiles.data_windows || {}, source: 'file' });
  } catch (e) { res.status(500).json({ error: 'Failed to load timeframe profiles' }); }
});

router.get('/omp/universe/trading-modalities', (req, res) => {
  try {
    const modalitiesPath = path.join(getWorkspaceRoot(), 'configs', 'universe', 'trading_modalities.toml');
    if (!fs.existsSync(modalitiesPath)) return res.json({ families: {}, modalities: {}, source: 'not_found' });
    const modalities = toml.parse(fs.readFileSync(modalitiesPath, 'utf-8'));
    res.json({ families: modalities.families || {}, modalities: modalities.modalities || {}, compatibility: modalities.compatibility || {}, metadata: modalities.metadata || {}, source: 'file' });
  } catch (e) { res.status(500).json({ error: 'Failed to load trading modalities' }); }
});

export default router;

