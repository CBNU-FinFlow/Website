module.exports = {
  apps: [
    {
      name: 'finflow-frontend',
      script: 'npm',
      args: 'start',
      cwd: '/var/www/finflow',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '512M',
      env: {
        NODE_ENV: 'production',
        PORT: 3000
      },
      error_file: '/var/www/finflow/logs/frontend-error.log',
      out_file: '/var/www/finflow/logs/frontend-out.log',
      log_file: '/var/www/finflow/logs/frontend-combined.log'
    }
  ]
};
