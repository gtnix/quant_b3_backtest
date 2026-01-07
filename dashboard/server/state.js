import path from 'path';
import fs from 'fs';
import toml from 'toml';
import { getWorkspaceRoot, DATABASE_URL } from './db.js';

export { DATABASE_URL };

// SSE clients
export const sseClients = new Set();
export const sseEventBuffer = [];
export let sseEventId = 0;

// SCG tracking
export const scgRuns = new Map();

// OMP State
export const ompState = {
  status: 'offline',
  currentCampaign: null,
  queueLength: 0,
  startedAt: null,
  lastPromotion: null,
  lastLoop: null,
  loopCount: 0,
  stats: {
    candidatesGenerated: 0, candidatesGenerated24h: 0, candidatesGenerated7d: 0,
    backtestsExecuted: 0, backtestsExecuted24h: 0,
    promotions: 0, promotions24h: 0,
    campaignsCompleted: 0, campaignsFailed: 0,
    throughputPerMin: 0, gatesApprovalRate: 0,
  },
  throughputWindow: [],
  resources: { cpuUsage: 0, memoryUsagePct: 0, memoryAvailableMb: 0, diskFreeGb: 0, diskWritten24h: 0, writeRateMbPerSec: 0, writeAcceleration: 0, canStartCampaign: false },
  diskIoHistory: [],
  config: null,
  activityLog: [],
};

export let ompLoopInterval = null;
export function setOmpLoopInterval(v) { ompLoopInterval = v; }
export function getOmpLoopInterval() { return ompLoopInterval; }

const OMP_CONFIG_PATH = () => path.join(process.cwd(), 'omp_config.toml');
const QUEUE_PATH = () => path.join(process.cwd(), 'campaign_queue.json');

export function loadOmpConfig() {
  try {
    if (fs.existsSync(OMP_CONFIG_PATH())) {
      ompState.config = toml.parse(fs.readFileSync(OMP_CONFIG_PATH(), 'utf-8'));
      return ompState.config;
    }
  } catch (e) { console.error('[OMP] Config load failed:', e.message); }
  return null;
}

export function loadCampaignQueue() {
  try {
    if (fs.existsSync(QUEUE_PATH())) {
      const q = JSON.parse(fs.readFileSync(QUEUE_PATH(), 'utf-8'));
      ompState.queueLength = q.campaigns?.filter(c => c.enabled).length || 0;
      return q;
    }
  } catch (e) { console.error('[OMP] Queue load failed:', e.message); }
  return { version: '1.0', campaigns: [] };
}

export function saveCampaignQueue(queue) {
  try {
    queue.updated_at = new Date().toISOString();
    fs.writeFileSync(QUEUE_PATH(), JSON.stringify(queue, null, 2));
    ompState.queueLength = queue.campaigns?.filter(c => c.enabled).length || 0;
    return true;
  } catch (e) { return false; }
}

export function broadcastSSE(eventType, data) {
  sseEventId++;
  const eventData = { type: eventType, ...data, timestamp: Date.now() };
  const message = JSON.stringify(eventData);
  sseEventBuffer.push({ id: sseEventId, data: eventData });
  if (sseEventBuffer.length > 100) sseEventBuffer.shift();
  for (const client of sseClients) {
    try { client.write(`id: ${sseEventId}\ndata: ${message}\n\n`); }
    catch (e) { sseClients.delete(client); }
  }
}

export function addActivityLog(level, message, details = {}) {
  const entry = { timestamp: new Date().toISOString(), level, message, ...details };
  ompState.activityLog.unshift(entry);
  if (ompState.activityLog.length > 500) ompState.activityLog.pop();
  broadcastSSE('omp-activity', entry);
}

