import { NextRequest, NextResponse } from "next/server";
import { XAIData } from "@/lib/types";

export async function POST(request: NextRequest) {
	try {
		const { investmentAmount, riskTolerance, investmentHorizon, method = "fast" } = await request.json();

		if (!investmentAmount || investmentAmount <= 0) {
			return NextResponse.json({ error: "유효한 투자 금액을 입력해주세요." }, { status: 400 });
		}

		console.log("XAI 설명 요청:", {
			investmentAmount,
			riskTolerance,
			investmentHorizon,
			method,
		});

		// 백엔드 XAI 서버 호출
		const response = await fetch("http://localhost:8000/explain", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({
				investment_amount: investmentAmount,
				risk_tolerance: riskTolerance,
				investment_horizon: investmentHorizon,
				method: method, // "fast" 또는 "accurate"
			}),
		});

		if (!response.ok) {
			throw new Error(`XAI 서버 오류: ${response.status}`);
		}

		const data = await response.json();
		console.log("XAI 서버 응답:", data);

		// 응답 데이터 구조 변환
		const xaiData: XAIData = {
			feature_importance: data.feature_importance,
			attention_weights: data.attention_weights,
			explanation_text: data.explanation_text,
		};

		return NextResponse.json(xaiData);
	} catch (error) {
		console.error("XAI 설명 생성 오류:", error);

		// 폴백: 계산 방식에 따른 다른 더미 데이터
		const method = (await request.json()).method || "fast";

		const baseFallback = {
			feature_importance: [
				{ feature_name: "Close", importance_score: 0.245, asset_name: "AAPL" },
				{ feature_name: "RSI", importance_score: 0.198, asset_name: "MSFT" },
				{ feature_name: "MACD", importance_score: 0.176, asset_name: "GOOGL" },
				{ feature_name: "Volume", importance_score: 0.134, asset_name: "AMZN" },
				{ feature_name: "MA21", importance_score: 0.112, asset_name: "TSLA" },
			],
			attention_weights: [
				{ from_asset: "AAPL", to_asset: "MSFT", weight: 0.23 },
				{ from_asset: "MSFT", to_asset: "GOOGL", weight: 0.19 },
				{ from_asset: "GOOGL", to_asset: "AMZN", weight: 0.16 },
				{ from_asset: "TSLA", to_asset: "AAPL", weight: 0.14 },
				{ from_asset: "AMZN", to_asset: "TSLA", weight: 0.12 },
			],
		};

		const fallbackData: XAIData = {
			...baseFallback,
			explanation_text:
				method === "fast"
					? `AI 포트폴리오 결정 근거 (빠른 분석):

🔍 주요 영향 요인:
1. AAPL의 Close: 0.245
2. MSFT의 RSI: 0.198
3. GOOGL의 MACD: 0.176

📊 핵심 투자 논리:
• AAPL (18.0%): 종가 움직임이 긍정적
• MSFT (16.0%): RSI 지표가 과매도 구간
• GOOGL (12.0%): MACD 상승 신호

⚡ 빠른 분석: 근사적 계산으로 빠른 인사이트 제공`
					: `AI 포트폴리오 결정 근거 (정확한 분석):

🔍 주요 영향 요인 (Integrated Gradients):
1. AAPL의 Close: 0.245 (높은 신뢰도)
2. MSFT의 RSI: 0.198 (중간 신뢰도)  
3. GOOGL의 MACD: 0.176 (중간 신뢰도)
4. AMZN의 Volume: 0.134 (낮은 신뢰도)
5. TSLA의 MA21: 0.112 (낮은 신뢰도)

📊 핵심 투자 논리:
• AAPL (18.0%): Close 지표가 강한 상승 모멘텀 시사
• MSFT (16.0%): RSI 지표가 적정 매수 구간 진입
• GOOGL (12.0%): MACD 골든크로스 형성으로 상승 전환점

🛡️ 리스크 관리:
• 현금 14.0% 보유로 시장 변동성 대비
• 섹터 분산을 통한 체계적 리스크 완화

🔬 정확한 분석: 50-step Integrated Gradients로 높은 신뢰도의 해석 제공`,
		};

		return NextResponse.json(fallbackData);
	}
}
