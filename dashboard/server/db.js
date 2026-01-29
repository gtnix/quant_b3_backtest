import pg from 'pg';
import fs from 'fs';
import path from 'path';
import { parse } from 'csv-parse/sync';
import toml from 'toml';

const { Pool } = pg;

export const DATABASE_URL = process.env.DATABASE_URL;

if (!DATABASE_URL) {
  console.warn('[DB] WARNING: DATABASE_URL not configured. Database features will be unavailable.');
}
export const pool = DATABASE_URL ? new Pool({
  connectionString: DATABASE_URL,
  ssl: { rejectUnauthorized: false }
}) : null;

export const PROJECT_ROOT = path.resolve(process.cwd(), '..');
export let ARTIFACTS_ROOT = path.join(PROJECT_ROOT, 'artifacts');
export let WORKSPACE_ROOT = PROJECT_ROOT;

export function setArtifactsRoot(p) { ARTIFACTS_ROOT = p; }
export function setWorkspaceRoot(p) { WORKSPACE_ROOT = p; }
export function getArtifactsRoot() { return ARTIFACTS_ROOT; }
export function getWorkspaceRoot() { return WORKSPACE_ROOT; }

export function readJsonFile(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  } catch (e) {
    return null;
  }
}

export function readTomlFile(filePath) {
  try {
    return toml.parse(fs.readFileSync(filePath, 'utf-8'));
  } catch (e) {
    return null;
  }
}

export function readCsvFile(filePath) {
  try {
    return parse(fs.readFileSync(filePath, 'utf-8'), { columns: true, skip_empty_lines: true });
  } catch (e) {
    return [];
  }
}

export function generateDisplayName(strategy) {
  if (!strategy?.pipeline) return 'Unknown Strategy';
  const parts = strategy.pipeline.map(b => `${b.type?.charAt(0).toUpperCase()}:${b.block_id}`);
  return parts.length > 0 ? parts.join(' | ') : 'Unknown Strategy';
}

