/**
 * Quant Dashboard API Server
 * Modular Express server with route separation
 */

import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { PROJECT_ROOT, ARTIFACTS_ROOT, setArtifactsRoot, setWorkspaceRoot } from './server/db.js';
import { loadOmpConfig, loadCampaignQueue } from './server/state.js';

// Route imports
import healthRoutes from './server/routes/health.js';
import strategiesRoutes from './server/routes/strategies.js';
import campaignsRoutes from './server/routes/campaigns.js';
import candidatesRoutes from './server/routes/candidates.js';
import scgRoutes from './server/routes/scg.js';
import ompRoutes from './server/routes/omp.js';
import auditRoutes from './server/routes/audit.js';
import universeRoutes from './server/routes/universe.js';
import eventsRoutes from './server/routes/events.js';

const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json());

// Mount all routes
app.use('/api', healthRoutes);
app.use('/api', strategiesRoutes);
app.use('/api', campaignsRoutes);
app.use('/api', candidatesRoutes);
app.use('/api', scgRoutes);
app.use('/api', ompRoutes);
app.use('/api', auditRoutes);
app.use('/api', universeRoutes);
app.use('/api', eventsRoutes);

// Auto-detect paths on startup
function autoInitialize() {
  const possibleRoots = [PROJECT_ROOT, path.resolve(PROJECT_ROOT, '..'), path.resolve(PROJECT_ROOT, '../..')];
  for (const root of possibleRoots) {
    if (fs.existsSync(path.join(root, 'Cargo.toml'))) { setWorkspaceRoot(root); break; }
  }
  for (const artPath of [path.join(PROJECT_ROOT, 'artifacts'), ARTIFACTS_ROOT]) {
    if (fs.existsSync(path.join(artPath, 'site'))) { setArtifactsRoot(artPath); break; }
  }
}

// Initialize
loadOmpConfig();
loadCampaignQueue();
autoInitialize();

app.listen(PORT, () => {
  console.log(`\n🚀 Quant Dashboard API Server`);
  console.log(`   http://localhost:${PORT}`);
  console.log(`\n📊 Routes: health, strategies, campaigns, candidates, scg, omp, audit, universe, events`);
});
