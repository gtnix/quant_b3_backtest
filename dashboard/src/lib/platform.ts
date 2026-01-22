/**
 * Platform Configuration - Web Only
 */

export const platform = {
  isBrowser: true,
  isDev: import.meta.env?.DEV ?? false,
  isProd: import.meta.env?.PROD ?? true,
};

export const capabilities = {
  processSpawn: true,
  database: true,
  realTimeUpdates: true,
};

const isRemoteServer = (): boolean => {
  if (typeof window === 'undefined') return false;
  const hostname = window.location.hostname;
  return !hostname.includes('localhost') && !hostname.includes('127.0.0.1');
};

const getApiBase = (): string => {
  if (isRemoteServer()) {
    return `http://${window.location.hostname}:3001/api`;
  }
  return 'http://localhost:3001/api';
};

const getSseEndpoint = (): string => {
  if (isRemoteServer()) {
    return `http://${window.location.hostname}:3001/api/events`;
  }
  return 'http://localhost:3001/api/events';
};

export const config = {
  apiBase: getApiBase(),
  sseEndpoint: getSseEndpoint(),
  pollIntervalMs: 5000,
  scgPollIntervalMs: 1000,
};

export const features = {
  useSSE: true,
  usePolling: true,
  autoInitialize: true,
};

export default { platform, capabilities, config, features };

