/**
 * PM2 Local Config - Development
 * 
 * Usage:
 *   pm2 start pm2.local.config.cjs
 *   pm2 status
 *   pm2 logs
 *   pm2 stop all
 */

const path = require('path');
const DASHBOARD_ROOT = __dirname;

module.exports = {
  apps: [
    {
      name: 'quant-api',
      script: 'server.js',
      cwd: DASHBOARD_ROOT,
      instances: 1,
      autorestart: true,
      watch: false,
      max_restarts: 10,
      env: {
        NODE_ENV: 'development',
        PORT: 3001,
      },
    },
    {
      name: 'quant-dashboard',
      script: 'npm',
      args: 'run dev -- --host',
      cwd: DASHBOARD_ROOT,
      instances: 1,
      autorestart: true,
      watch: false,
    },
  ],
};
