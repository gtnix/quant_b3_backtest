import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { getArtifactsRoot, setArtifactsRoot, getWorkspaceRoot, setWorkspaceRoot, PROJECT_ROOT } from '../db.js';
import { sseClients, sseEventBuffer, sseEventId, ompState } from '../state.js';

const router = Router();

router.get('/events', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('Access-Control-Allow-Origin', '*');
  
  const lastEventId = req.headers['last-event-id'];
  if (lastEventId) {
    const startId = parseInt(lastEventId) + 1;
    const missedEvents = sseEventBuffer.filter(e => e.id >= startId);
    for (const event of missedEvents) res.write(`id: ${event.id}\ndata: ${JSON.stringify(event.data)}\n\n`);
  }
  
  res.write(`data: ${JSON.stringify({ type: 'connected', timestamp: Date.now() })}\n\n`);
  sseClients.add(res);
  
  const keepAlive = setInterval(() => { try { res.write(`data: ${JSON.stringify({ type: 'ping', timestamp: Date.now() })}\n\n`); } catch (e) { clearInterval(keepAlive); sseClients.delete(res); } }, 15000);
  req.on('close', () => { clearInterval(keepAlive); sseClients.delete(res); });
});

router.get('/poll-changes', async (req, res) => {
  const sinceTime = req.query.since ? parseInt(req.query.since) : Date.now() - 60000;
  const changes = [];
  const indexPath = path.join(getArtifactsRoot(), 'site', 'index.json');
  if (fs.existsSync(indexPath)) { const stat = fs.statSync(indexPath); if (stat.mtime.getTime() > sinceTime) changes.push({ type: 'index', path: indexPath, modified: stat.mtime.toISOString() }); }
  const siteDir = path.join(getArtifactsRoot(), 'site');
  if (fs.existsSync(siteDir)) {
    fs.readdirSync(siteDir).filter(f => f.endsWith('.json')).forEach(file => {
      const stat = fs.statSync(path.join(siteDir, file));
      if (stat.mtime.getTime() > sinceTime) changes.push({ type: file.startsWith('campaign_') ? 'campaign' : file.startsWith('run_') ? 'run' : 'other', path: path.join(siteDir, file), modified: stat.mtime.toISOString() });
    });
  }
  res.json({ changes, since: new Date(sinceTime).toISOString(), checked_at: new Date().toISOString(), has_changes: changes.length > 0 });
});

router.get('/artifacts-root', (req, res) => { res.json({ path: getArtifactsRoot() }); });
router.post('/artifacts-root', (req, res) => {
  const { path: newPath } = req.body;
  if (!newPath) return res.status(400).json({ error: 'path is required' });
  let testPath = newPath;
  if (!fs.existsSync(path.join(testPath, 'site'))) testPath = path.join(newPath, 'artifacts');
  if (fs.existsSync(testPath)) { setArtifactsRoot(testPath); return res.json({ path: testPath, status: 'updated' }); }
  res.status(400).json({ error: `Path not found: ${newPath}` });
});

router.get('/workspace-root', (req, res) => { res.json({ path: getWorkspaceRoot() }); });
router.post('/workspace-root', (req, res) => {
  const { path: newPath } = req.body;
  if (!newPath) return res.status(400).json({ error: 'path is required' });
  if (fs.existsSync(path.join(newPath, 'Cargo.toml'))) { setWorkspaceRoot(newPath); return res.json({ path: newPath, status: 'updated' }); }
  res.status(400).json({ error: `Invalid workspace: ${newPath}` });
});

router.post('/invalidate-cache', (req, res) => { res.json({ message: 'Cache invalidated', timestamp: new Date().toISOString() }); });

router.get('/evolution/state', (req, res) => {
  const cc = ompState.currentCampaign;
  if (!cc) return res.json({ status: 'no_data', message: 'No active campaign' });
  res.json({ status: 'running', run_id: cc.runId, campaign_name: cc.campaignName, current_generation: cc.currentGeneration, best_sharpe: cc.bestSharpe, candidates_evaluated: cc.candidatesEvaluated, elapsed_secs: Math.floor((Date.now() - cc.startTime) / 1000) });
});

export default router;

