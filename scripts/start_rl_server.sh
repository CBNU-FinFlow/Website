#!/bin/bash

# 강화학습 모델 추론 서버 시작 스크립트

echo "FinFlow RL 추론 서버 시작 중..."

# 현재 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "스크립트 디렉토리: $SCRIPT_DIR"

# Python 가상환경 확인 및 생성
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Python 가상환경 생성 중..."
    python3 -m venv "$SCRIPT_DIR/venv"
fi

# 가상환경 활성화
echo "가상환경 활성화 중..."
source "$SCRIPT_DIR/venv/bin/activate"

# pip 업그레이드
echo "pip 업그레이드 중..."
python -m pip install --upgrade pip

# 패키지 설치
echo "필요한 패키지 설치 중..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# 설치 확인
if [ $? -ne 0 ]; then
    echo "패키지 설치에 실패했습니다. 개별 설치를 시도합니다..."
    pip install fastapi uvicorn pydantic numpy pandas yfinance requests
    echo "기본 패키지 설치 완료. torch는 별도로 설치합니다..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# finflow-rl 프로젝트 경로 확인
RL_PROJECT_PATH="../finflow-rl"
if [ ! -d "$RL_PROJECT_PATH" ]; then
    echo "경고: finflow-rl 프로젝트를 찾을 수 없습니다."
    echo "경로를 확인하거나 rl_inference_server.py에서 RL_PROJECT_PATH를 수정하세요."
fi

# 서버 시작
echo "추론 서버 시작 중... (포트 8000)"
echo "서버를 중지하려면 Ctrl+C를 누르세요."
python "$SCRIPT_DIR/rl_inference_server.py" 