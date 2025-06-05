#!/bin/bash

# FinFlow 배포 스크립트
set -e

PROJECT_DIR="/var/www/finflow"
BACKUP_DIR="/var/backups/finflow-$(date +%Y%m%d-%H%M%S)"

echo "🚀 FinFlow 배포 시작..."

# 1. 백업 생성
echo "📦 현재 버전 백업 중..."
sudo mkdir -p /var/backups
sudo cp -r $PROJECT_DIR $BACKUP_DIR

# 2. 코드 업데이트 (Git 사용하는 경우)
echo "📥 코드 업데이트 중..."
cd $PROJECT_DIR
git pull origin main  # 또는 사용하는 브랜치명

# 3. 백엔드 종속성 업데이트
echo "🐍 백엔드 종속성 업데이트 중..."
cd $PROJECT_DIR/scripts
source venv/bin/activate
pip install -r requirements.txt
deactivate

# 4. 프론트엔드 빌드
echo "⚛️ 프론트엔드 빌드 중..."
cd $PROJECT_DIR
npm install
npm run build

# 5. 서비스 재시작
echo "🔄 서비스 재시작 중..."

# PM2 프로세스 재시작
pm2 restart finflow-backend
pm2 restart finflow-frontend

# Nginx 설정 테스트 및 재시작
sudo nginx -t
sudo systemctl reload nginx

# 6. 헬스체크
echo "🏥 헬스체크 중..."
sleep 10

# 백엔드 헬스체크
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ 백엔드 서버 정상"
else
    echo "❌ 백엔드 서버 오류"
    exit 1
fi

# 프론트엔드 헬스체크
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ 프론트엔드 서버 정상"
else
    echo "❌ 프론트엔드 서버 오류"
    exit 1
fi

# 7. 배포 완료
echo "🎉 배포 완료!"
echo "🌐 사이트: https://finflow.reo91004.com"

# PM2 상태 확인
pm2 list

echo "📊 PM2 상태:"
pm2 monit
