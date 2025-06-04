module.exports = {
  apps: [
    {
      name: 'finflow-backend',
      script: 'venv/bin/python',
      args: 'rl_inference_server.py',
      cwd: '/var/www/finflow/scripts',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      env: {
        NODE_ENV: 'production',
        PORT: 8000,
        // 환경 변수 추가
        ENVIRONMENT: 'production',
        CORS_ORIGINS: 'https://finflow.reo91004.com,https://www.finflow.reo91004.com'
      },
      error_file: '/var/www/finflow/logs/backend-error.log',
      out_file: '/var/www/finflow/logs/backend-out.log',
      log_file: '/var/www/finflow/logs/backend-combined.log'
    }
  ]
};