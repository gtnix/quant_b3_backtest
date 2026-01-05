/**
 * PM2 Ecosystem Configuration
 * 
 * Usage:
 *   pm2 start ecosystem.config.cjs
 *   pm2 reload ecosystem.config.cjs
 *   pm2 status
 *   pm2 logs
 */

module.exports = {
  apps: [
    {
      // API Server with OMP (Orquestrador de Mineração Perpétua)
      name: 'api-server',
      script: 'server.js',
      cwd: '/opt/alpha-forge/quant_b3_backtest/dashboard',
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
      },
      
      // Logging
      log_file: '/opt/alpha-forge/logs/api-server.log',
      error_file: '/opt/alpha-forge/logs/api-server-error.log',
      out_file: '/opt/alpha-forge/logs/api-server-out.log',
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
      cwd: '/opt/alpha-forge/quant_b3_backtest/dashboard',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_restarts: 5,
      restart_delay: 3000,
      
      env: {
        NODE_ENV: 'production',
      },
      
      log_file: '/opt/alpha-forge/logs/dashboard.log',
      error_file: '/opt/alpha-forge/logs/dashboard-error.log',
      out_file: '/opt/alpha-forge/logs/dashboard-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    },
  ],
};



















