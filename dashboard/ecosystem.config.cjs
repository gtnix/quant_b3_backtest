/**
 * PM2 Ecosystem Configuration - Alpha Forge
 * 
 * Usage:
 *   pm2 start ecosystem.config.cjs
 *   pm2 reload ecosystem.config.cjs
 *   pm2 status
 *   pm2 logs
 */

const ALPHA_FORGE_ROOT = '/opt/alpha-forge';

module.exports = {
  apps: [
    {
      // API Server with OMP (Orquestrador de Mineração Perpétua)
      name: 'api-server',
      script: 'server.js',
      cwd: `${ALPHA_FORGE_ROOT}/dashboard`,
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_restarts: 10,
      min_uptime: '10s',
      restart_delay: 5000,
      max_memory_restart: '1G',
      
      // Environment variables
      env: {
        NODE_ENV: 'production',
        PORT: 3001,
        DATABASE_URL: process.env.DATABASE_URL || 'postgresql://neondb_owner:npg_HyU68iqJScrQ@ep-wild-cell-af18q8jx-pooler.c-2.us-west-2.aws.neon.tech/neondb?sslmode=require',
        OMP_ENABLED: 'true',
        // Binary paths for SCG/OMP
        COMBINER_PATH: `${ALPHA_FORGE_ROOT}/bin/combiner`,
        BACKTEST_PATH: `${ALPHA_FORGE_ROOT}/bin/backtest`,
        CONFIGS_PATH: `${ALPHA_FORGE_ROOT}/configs`,
      },
      
      // Logging
      log_file: `${ALPHA_FORGE_ROOT}/logs/api-server.log`,
      error_file: `${ALPHA_FORGE_ROOT}/logs/api-server-error.log`,
      out_file: `${ALPHA_FORGE_ROOT}/logs/api-server-out.log`,
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      
      // Resource limits
      node_args: '--max-old-space-size=512',
    },
    
    {
      // Frontend (Vite Preview)
      name: 'alpha-dashboard',
      script: 'npm',
      args: 'run preview -- --port 5173 --host',
      cwd: `${ALPHA_FORGE_ROOT}/dashboard`,
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_restarts: 5,
      restart_delay: 3000,
      
      env: {
        NODE_ENV: 'production',
      },
      
      log_file: `${ALPHA_FORGE_ROOT}/logs/dashboard.log`,
      error_file: `${ALPHA_FORGE_ROOT}/logs/dashboard-error.log`,
      out_file: `${ALPHA_FORGE_ROOT}/logs/dashboard-out.log`,
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    },
  ],
};





















