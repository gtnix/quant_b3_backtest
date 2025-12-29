/**
 * Platform Detection & Capabilities
 * 
 * Provides consistent detection of running environment (Tauri vs Browser)
 * and available capabilities for each platform.
 */

// =============================================================================
// PLATFORM DETECTION
// =============================================================================

export const platform = {
  /** Running inside Tauri desktop app */
  isTauri: typeof window !== 'undefined' && '__TAURI__' in window,
  
  /** Running in browser (not Tauri) */
  isBrowser: typeof window !== 'undefined' && !('__TAURI__' in window),
  
  /** Running in Node.js (server-side) */
  isNode: typeof process !== 'undefined' && !!process.versions?.node,
  
  /** Running in development mode */
  isDev: import.meta.env?.DEV ?? false,
  
  /** Running in production mode */
  isProd: import.meta.env?.PROD ?? true,
};

// =============================================================================
// CAPABILITIES
// =============================================================================

export const capabilities = {
  /** Can open native file/folder dialogs */
  nativeDialog: platform.isTauri,
  
  /** Can watch file system for changes natively */
  fileWatcher: platform.isTauri,
  
  /** Can access local file system directly */
  directFS: platform.isTauri,
  
  /** Can spawn and monitor external processes */
  processSpawn: true, // Both modes can spawn via API
  
  /** Can access Neon database */
  database: true, // Both modes access Neon
  
  /** Real-time updates available */
  realTimeUpdates: true, // Tauri: native events, Browser: SSE
  
  /** Can export files to local filesystem */
  localExport: platform.isTauri,
};

// =============================================================================
// API CONFIGURATION
// =============================================================================

// Determine API base URL based on environment
const getApiBase = (): string => {
  // In production (Netlify), API is served from same domain via Functions
  if (platform.isProd) {
    return '/api';
  }
  // In development, use local Express server
  return 'http://localhost:3001/api';
};

const getSseEndpoint = (): string => {
  if (platform.isProd) {
    // SSE not supported in serverless - use polling
    return '';
  }
  return 'http://localhost:3001/api/events';
};

export const config = {
  /** Base URL for API calls (browser mode) */
  apiBase: getApiBase(),
  
  /** SSE endpoint for real-time updates (empty in production = use polling) */
  sseEndpoint: getSseEndpoint(),
  
  /** Polling interval for changes (browser mode fallback) */
  pollIntervalMs: 5000,
  
  /** SCG progress polling interval */
  scgPollIntervalMs: 1000,
};

// =============================================================================
// MODE DISPLAY
// =============================================================================

export function getModeDisplay(): { mode: string; description: string; icon: string } {
  if (platform.isTauri) {
    return {
      mode: 'Desktop',
      description: 'Running in Tauri desktop application with full capabilities',
      icon: '🖥️',
    };
  }
  
  return {
    mode: 'Browser',
    description: 'Running in browser mode. Data is fetched from Neon DB via API server.',
    icon: '🌐',
  };
}

// =============================================================================
// FEATURE FLAGS
// =============================================================================

export const features = {
  /** Show browser mode banner */
  showBrowserBanner: platform.isBrowser,
  
  /** Use SSE for real-time updates */
  useSSE: platform.isBrowser,
  
  /** Use polling as fallback for updates */
  usePolling: platform.isBrowser,
  
  /** Show folder path input instead of dialog */
  showPathInput: platform.isBrowser,
  
  /** Auto-initialize from API on startup */
  autoInitialize: platform.isBrowser,
};

export default {
  platform,
  capabilities,
  config,
  features,
  getModeDisplay,
};

