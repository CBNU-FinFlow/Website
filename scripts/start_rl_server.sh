#!/bin/bash

# 강화학습 모델 추론 서버 시작 스크립트

echo "FinFlow RL 추론 서버 시작 중..."

# 필요한 디렉토리 생성
mkdir -p logs

# 현재 스크립트의 디렉토리로 이동
cd "$(dirname "$0")"

# 가상환경 활성화
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "가상환경을 찾을 수 없습니다. 먼저 setup.sh를 실행하세요."
    exit 1
fi

# 모델 디렉토리 확인
MODELS_DIR="models"
if [ ! -d "$MODELS_DIR" ]; then
    echo "models 디렉토리를 찾을 수 없습니다."
    echo "models 디렉토리를 생성하고 모델 파일을 넣어주세요."
    mkdir -p "$MODELS_DIR"
    exit 1
fi

# 서버 실행
echo "RL 추론 서버를 시작합니다..."
python rl_inference_server.py 