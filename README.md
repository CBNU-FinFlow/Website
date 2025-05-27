# FinFlow UI

AI 기반 포트폴리오 리밸런싱 웹 애플리케이션

## 개요

FinFlow UI는 강화학습 모델을 활용하여 동적 포트폴리오 배분과 성과 예측을 제공하는 웹 애플리케이션이다.
[finflow-rl](https://github.com/reo91004/finflow-rl) 프로젝트에서 학습된 강화학습 모델을 통합하여 실시간 포트폴리오 최적화를 수행한다.

## 주요 기능

- **동적 포트폴리오 배분**: 강화학습 모델을 통한 실시간 자산 배분 최적화
- **성과 예측**: 백테스트 기반 수익률, 샤프비율, 최대낙폭 등 주요 지표 예측
- **리스크 관리**: 사용자 리스크 성향에 따른 맞춤형 포트폴리오 구성
- **실시간 분석**: 최신 시장 데이터와 기술적 지표를 활용한 분석

## 기술 스택

### 프론트엔드

- Next.js 15.2.4
- React 19
- TypeScript
- Tailwind CSS
- shadcn/ui

### 백엔드 (추론 서버)

- FastAPI
- PyTorch
- pandas, numpy
- yfinance (시장 데이터)

## 설치 및 실행

### 1. 프론트엔드 설정

```bash
# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

### 2. 강화학습 모델 추론 서버 설정

```bash
# 추론 서버 시작 (자동으로 가상환경 생성 및 패키지 설치)
./scripts/start_rl_server.sh
```

또는 수동으로:

```bash
cd scripts

# Python 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# 서버 실행
python rl_inference_server.py
```

### 3. finflow-rl 프로젝트 연동

1. [finflow-rl](https://github.com/reo91004/finflow-rl) 프로젝트를 클론한다:

```bash
git clone https://github.com/reo91004/finflow-rl.git
```

2. `scripts/models` 디렉토리에 모델 파일을 넣는다:

   - `best_model.pth` 파일을 `scripts/models` 디렉토리에 복사

3. RL 추론 서버를 실행한다:
   ```bash
   cd scripts
   ./start_rl_server.sh
   ```

## 사용법

1. 웹 브라우저에서 `http://localhost:3000` 접속
2. 투자 금액 입력
3. "지금 바로 시작하기" 버튼 클릭
4. AI 분석 완료 후 추천 포트폴리오 확인

## API 엔드포인트

### POST /api/portfolio

포트폴리오 예측 요청

**요청 본문:**

```json
{
	"investmentAmount": 1000000
}
```

**응답:**

```json
{
	"portfolioAllocation": [
		{
			"stock": "AAPL",
			"percentage": 15,
			"amount": 150000
		}
	],
	"performanceMetrics": [
		{
			"label": "연간 수익률",
			"portfolio": "16.24%",
			"spy": "0.00%",
			"qqq": "0.00%"
		}
	],
	"quickMetrics": {
		"annualReturn": "+16.2%",
		"sharpeRatio": "0.92",
		"maxDrawdown": "-18.7%",
		"volatility": "17.9%"
	}
}
```

## 강화학습 모델 통합

### 모델 서버 구조

- **포트**: 8000
- **엔드포인트**: `/predict`
- **폴백 모드**: 모델 로드 실패 시 규칙 기반 예측 제공

### 지원 기능

- 실시간 시장 데이터 수집 (yfinance)
- 기술적 지표 계산 (MACD, RSI, 이동평균)
- 리스크 성향별 포트폴리오 조정
- 성과 지표 백테스트

## 개발 정보

### 프로젝트 구조

```
finflow-ui/
├── app/                    # Next.js 앱 라우터
│   ├── api/portfolio/      # 포트폴리오 API
│   ├── page.tsx           # 메인 페이지
│   └── layout.tsx         # 레이아웃
├── components/            # UI 컴포넌트
├── lib/                   # 유틸리티 및 타입
├── scripts/               # 추론 서버 스크립트
│   ├── rl_inference_server.py
│   ├── requirements.txt
│   └── start_rl_server.sh
└── public/               # 정적 파일
```

### 환경 변수

현재 환경 변수는 필요하지 않지만, 프로덕션 환경에서는 다음을 고려할 수 있다:

- `RL_SERVER_URL`: 추론 서버 URL
- `MARKET_DATA_API_KEY`: 시장 데이터 API 키

## 배포

### Vercel 배포

```bash
npm run build
```

### 추론 서버 배포

Docker를 사용한 배포 예시:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY scripts/requirements.txt .
RUN pip install -r requirements.txt
COPY scripts/rl_inference_server.py .
EXPOSE 8000
CMD ["python", "rl_inference_server.py"]
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포된다.

## 기여

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 관련 프로젝트

- [finflow-rl](https://github.com/reo91004/finflow-rl): 강화학습 모델 학습 프로젝트
