/**
 * Alpha Forge - PM2 Ecosystem Configuration
 * Professional process management for backtest infrastructure
 */

module.exports = {
  apps: [
    {
      name: 'alpha-api',
      script: 'server.js',
      cwd: '/opt/alpha-forge/quant_b3_backtest/dashboard',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      env: {
        NODE_ENV: 'production',
        PORT: 3001,
        DATABASE_URL: 'postgresql://neondb_owner:npg_HyU68iqJScrQ@ep-wild-cell-af18q8jx-pooler.c-2.us-west-2.aws.neon.tech/neondb?sslmode=require',
        BRAPI_API_KEY: 'gNJ4vTTpjG8TZJJHGBqoV5'
      },
      error_file: '/opt/alpha-forge/logs/api-error.log',
      out_file: '/opt/alpha-forge/logs/api-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true
    },
    {
      name: 'alpha-dashboard',
      script: 'npx',
      args: 'vite preview --port 5173 --host 0.0.0.0',
      cwd: '/opt/alpha-forge/quant_b3_backtest/dashboard',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '300M',
      env: {
        NODE_ENV: 'production'
      },
      error_file: '/opt/alpha-forge/logs/dashboard-error.log',
      out_file: '/opt/alpha-forge/logs/dashboard-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true
    }
  ],

  deploy: {
    production: {
      user: 'root',
      host: '149.28.39.194',
      ref: 'origin/main',
      repo: 'git@github.com:gtnix/quant_b3_backtest.git',
      path: '/opt/alpha-forge/quant_b3_backtest',
      'pre-deploy-local': '',
      'post-deploy': 'cd dashboard && npm ci && npm run build && pm2 reload ecosystem.config.js --env production',
      'pre-setup': ''
    }
  }
};


















