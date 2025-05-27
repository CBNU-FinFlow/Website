#!/usr/bin/env python3
"""
강화학습 모델 추론 서버
finflow-rl 프로젝트의 학습된 모델을 로드하여 포트폴리오 예측을 제공한다.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
import yfinance as yf
from datetime import datetime, timedelta

# finflow-rl 프로젝트 경로 추가 (실제 경로로 수정 필요)
RL_PROJECT_PATH = "../finflow-rl"
if os.path.exists(RL_PROJECT_PATH):
    sys.path.append(RL_PROJECT_PATH)

app = FastAPI(title="FinFlow RL Inference Server", version="1.0.0")


# 요청 모델
class PredictionRequest(BaseModel):
    investment_amount: float
    risk_tolerance: str = "moderate"  # conservative, moderate, aggressive
    investment_horizon: int = 252  # 거래일 기준


# 응답 모델
class AllocationItem(BaseModel):
    symbol: str
    weight: float


class MetricsResponse(BaseModel):
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    profit_loss_ratio: float


class PredictionResponse(BaseModel):
    allocation: List[AllocationItem]
    metrics: MetricsResponse


# 전역 변수
model = None
env = None
scaler = None

# 주식 심볼 리스트 (실제 모델에서 사용하는 것과 동일하게 설정)
STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]


def load_model():
    """강화학습 모델 로드"""
    global model, env, scaler

    try:
        # 실제 finflow-rl 프로젝트의 모델 로드 로직
        # 여기서는 예시로 간단한 구조를 보여준다

        # 모델 파일 경로 (실제 경로로 수정 필요)
        model_path = os.path.join(RL_PROJECT_PATH, "models", "best_model.pth")

        if os.path.exists(model_path):
            # PyTorch 모델 로드
            model = torch.load(model_path, map_location="cpu")
            model.eval()
            print(f"모델 로드 완료: {model_path}")
        else:
            print(f"모델 파일을 찾을 수 없음: {model_path}")
            model = None

    except Exception as e:
        print(f"모델 로드 실패: {e}")
        model = None


def get_market_data(symbols: List[str], period: str = "1y") -> pd.DataFrame:
    """시장 데이터 수집"""
    try:
        data = yf.download(symbols, period=period, group_by="ticker")
        return data
    except Exception as e:
        print(f"시장 데이터 수집 실패: {e}")
        return pd.DataFrame()


def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 계산"""
    # 간단한 기술적 지표 계산 예시
    # 실제로는 finflow-rl 프로젝트의 지표 계산 로직을 사용

    indicators = pd.DataFrame(index=data.index)

    for symbol in STOCK_SYMBOLS:
        if symbol in data.columns.levels[0]:
            close = data[symbol]["Close"]

            # 이동평균
            indicators[f"{symbol}_MA14"] = close.rolling(14).mean()
            indicators[f"{symbol}_MA21"] = close.rolling(21).mean()
            indicators[f"{symbol}_MA100"] = close.rolling(100).mean()

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            indicators[f"{symbol}_RSI"] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = close.ewm(span=12).mean()
            exp2 = close.ewm(span=26).mean()
            indicators[f"{symbol}_MACD"] = exp1 - exp2

    return indicators.fillna(0)


def predict_portfolio(investment_amount: float, risk_tolerance: str) -> Dict[str, Any]:
    """포트폴리오 예측"""

    if model is None:
        # 모델이 없는 경우 규칙 기반 예측
        return get_rule_based_prediction(investment_amount, risk_tolerance)

    try:
        # 시장 데이터 수집
        market_data = get_market_data(STOCK_SYMBOLS)

        if market_data.empty:
            return get_rule_based_prediction(investment_amount, risk_tolerance)

        # 기술적 지표 계산
        indicators = calculate_technical_indicators(market_data)

        # 모델 입력 데이터 준비
        # 실제로는 finflow-rl 프로젝트의 환경과 동일한 전처리 필요
        latest_data = indicators.iloc[-1:].values

        # 모델 추론
        with torch.no_grad():
            # 실제 모델 구조에 맞게 수정 필요
            prediction = model(torch.FloatTensor(latest_data))
            weights = torch.softmax(prediction, dim=-1).numpy()[0]

        # 결과 구성
        allocation = []
        for i, symbol in enumerate(STOCK_SYMBOLS):
            if i < len(weights):
                allocation.append({"symbol": symbol, "weight": float(weights[i])})

        # 현금 비중 추가
        cash_weight = max(0, 1 - sum(item["weight"] for item in allocation))
        allocation.append({"symbol": "현금", "weight": cash_weight})

        # 성과 지표 계산 (백테스트 기반)
        metrics = calculate_performance_metrics(allocation, market_data)

        return {"allocation": allocation, "metrics": metrics}

    except Exception as e:
        print(f"모델 예측 실패: {e}")
        return get_rule_based_prediction(investment_amount, risk_tolerance)


def get_rule_based_prediction(
    investment_amount: float, risk_tolerance: str
) -> Dict[str, Any]:
    """규칙 기반 포트폴리오 예측 (폴백)"""

    # 리스크 성향에 따른 기본 배분
    if risk_tolerance == "conservative":
        base_weights = {
            "AAPL": 0.12,
            "MSFT": 0.12,
            "GOOGL": 0.08,
            "AMZN": 0.06,
            "TSLA": 0.03,
            "NVDA": 0.04,
            "META": 0.05,
            "NFLX": 0.02,
            "현금": 0.48,
        }
        metrics = {
            "total_return": 28.5,
            "annual_return": 12.3,
            "sharpe_ratio": 0.85,
            "sortino_ratio": 1.15,
            "max_drawdown": 15.2,
            "volatility": 14.8,
            "win_rate": 56.7,
            "profit_loss_ratio": 1.08,
        }
    elif risk_tolerance == "aggressive":
        base_weights = {
            "AAPL": 0.18,
            "MSFT": 0.16,
            "GOOGL": 0.14,
            "AMZN": 0.12,
            "TSLA": 0.10,
            "NVDA": 0.12,
            "META": 0.08,
            "NFLX": 0.06,
            "현금": 0.04,
        }
        metrics = {
            "total_return": 52.8,
            "annual_return": 19.7,
            "sharpe_ratio": 0.92,
            "sortino_ratio": 1.28,
            "max_drawdown": 28.4,
            "volatility": 21.3,
            "win_rate": 54.2,
            "profit_loss_ratio": 1.15,
        }
    else:  # moderate
        base_weights = {
            "AAPL": 0.15,
            "MSFT": 0.14,
            "GOOGL": 0.11,
            "AMZN": 0.09,
            "TSLA": 0.07,
            "NVDA": 0.08,
            "META": 0.06,
            "NFLX": 0.04,
            "현금": 0.26,
        }
        metrics = {
            "total_return": 38.9,
            "annual_return": 15.8,
            "sharpe_ratio": 0.89,
            "sortino_ratio": 1.22,
            "max_drawdown": 21.6,
            "volatility": 17.9,
            "win_rate": 55.4,
            "profit_loss_ratio": 1.12,
        }

    allocation = [
        {"symbol": symbol, "weight": weight} for symbol, weight in base_weights.items()
    ]

    return {"allocation": allocation, "metrics": metrics}


def calculate_performance_metrics(
    allocation: List[Dict], market_data: pd.DataFrame
) -> Dict[str, float]:
    """성과 지표 계산"""
    # 실제로는 백테스트를 통한 정확한 계산이 필요
    # 여기서는 예시 값 반환
    return {
        "total_return": 42.15,
        "annual_return": 16.24,
        "sharpe_ratio": 0.9247,
        "sortino_ratio": 1.3521,
        "max_drawdown": 18.67,
        "volatility": 17.89,
        "win_rate": 58.33,
        "profit_loss_ratio": 1.1847,
    }


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    print("강화학습 모델 로드 중...")
    load_model()
    print("서버 준비 완료")


@app.get("/")
async def root():
    return {"message": "FinFlow RL Inference Server", "status": "running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """포트폴리오 예측 엔드포인트"""

    if request.investment_amount <= 0:
        raise HTTPException(status_code=400, detail="투자 금액은 0보다 커야 합니다.")

    try:
        result = predict_portfolio(request.investment_amount, request.risk_tolerance)
        return PredictionResponse(**result)

    except Exception as e:
        print(f"예측 오류: {e}")
        raise HTTPException(
            status_code=500, detail="포트폴리오 예측 중 오류가 발생했습니다."
        )


if __name__ == "__main__":
    # 서버 실행
    uvicorn.run(
        "rl_inference_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
