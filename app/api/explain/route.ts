import { NextRequest, NextResponse } from "next/server";
import { XAIData } from "@/lib/types";

export async function POST(request: NextRequest) {
	try {
		const { investmentAmount, riskTolerance, investmentHorizon } = await request.json();

		if (!investmentAmount || investmentAmount <= 0) {
			return NextResponse.json({ error: "유효한 투자 금액을 입력해주세요." }, { status: 400 });
		}

		console.log("XAI 설명 요청:", { investmentAmount, riskTolerance, investmentHorizon });

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

		// 폴백: 더미 데이터 반환 (개발/테스트용)
		const fallbackData: XAIData = {
			feature_importance: [
				{ feature_name: "Close", importance_score: 0.245, asset_name: "AAPL" },
				{ feature_name: "RSI", importance_score: 0.198, asset_name: "MSFT" },
				{ feature_name: "MACD", importance_score: 0.176, asset_name: "GOOGL" },
				{ feature_name: "Volume", importance_score: 0.134, asset_name: "AMZN" },
				{ feature_name: "MA21", importance_score: 0.112, asset_name: "TSLA" },
				{ feature_name: "MA14", importance_score: 0.098, asset_name: "AMD" },
				{ feature_name: "High", importance_score: 0.087, asset_name: "JPM" },
				{ feature_name: "Low", importance_score: 0.076, asset_name: "JNJ" },
				{ feature_name: "Open", importance_score: 0.065, asset_name: "PG" },
				{ feature_name: "MA100", importance_score: 0.054, asset_name: "V" },
			],
			attention_weights: [
				{ from_asset: "AAPL", to_asset: "MSFT", weight: 0.23 },
				{ from_asset: "MSFT", to_asset: "GOOGL", weight: 0.19 },
				{ from_asset: "GOOGL", to_asset: "AMZN", weight: 0.16 },
				{ from_asset: "TSLA", to_asset: "AAPL", weight: 0.14 },
				{ from_asset: "AMZN", to_asset: "TSLA", weight: 0.12 },
				{ from_asset: "AMD", to_asset: "JPM", weight: 0.11 },
				{ from_asset: "JPM", to_asset: "JNJ", weight: 0.09 },
				{ from_asset: "JNJ", to_asset: "PG", weight: 0.08 },
				{ from_asset: "PG", to_asset: "V", weight: 0.07 },
				{ from_asset: "V", to_asset: "AAPL", weight: 0.06 },
			],
			explanation_text: `AI 포트폴리오 결정 근거:

🔍 주요 영향 요인:
1. AAPL의 Close: 0.245
2. MSFT의 RSI: 0.198
3. GOOGL의 MACD: 0.176
4. AMZN의 Volume: 0.134
5. TSLA의 MA21: 0.112

📊 핵심 투자 논리:
• AAPL (18.0%): Close 지표가 긍정적 신호를 보임
• MSFT (16.0%): RSI 지표가 긍정적 신호를 보임
• GOOGL (12.0%): MACD 지표가 긍정적 신호를 보임

🛡️ 리스크 관리:
• 현금 14.0% 보유로 변동성 완충

💡 AI 분석 요약:
현재 시장 상황에서 기술주 중심의 포트폴리오가 최적으로 판단됩니다. 
특히 AAPL의 주가 모멘텀과 MSFT의 과매도 신호가 주요 투자 근거가 되었습니다.`,
		};

		return NextResponse.json(fallbackData);
	}
}
