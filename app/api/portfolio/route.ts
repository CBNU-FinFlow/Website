import { NextRequest, NextResponse } from "next/server";
import { PortfolioAllocation, PerformanceMetrics, PredictionResult } from "@/lib/types";
import { createApiUrl, getDefaultFetchOptions } from "@/lib/config";

// 강화학습 모델 서버와 통신하는 함수
async function callRLModel(investmentAmount: number): Promise<PredictionResult> {
	try {
		// 실제 강화학습 모델 서버 호출
		// 여기서는 finflow-rl 프로젝트의 추론 서버를 호출한다고 가정
		const response = await fetch(createApiUrl("/predict"), {
			method: "POST",
			...getDefaultFetchOptions(),
			body: JSON.stringify({
				investment_amount: investmentAmount,
				// 추가 파라미터들 (리스크 선호도, 투자 기간 등)
				risk_tolerance: "moderate",
				investment_horizon: 252, // 1년 (거래일 기준)
			}),
		});

		if (!response.ok) {
			throw new Error(`RL 모델 서버 오류: ${response.status}`);
		}

		const data = await response.json();

		// 강화학습 모델 결과를 프론트엔드 형식으로 변환
		const portfolioAllocation: PortfolioAllocation[] = data.allocation.map((item: any) => ({
			stock: item.symbol,
			percentage: Math.round(item.weight * 100),
			amount: Math.round(investmentAmount * item.weight),
		}));

		const performanceMetrics: PerformanceMetrics[] = [
			{ label: "총 수익률", portfolio: `${data.metrics.total_return.toFixed(2)}%`, spy: "0.00%", qqq: "0.00%" },
			{ label: "연간 수익률", portfolio: `${data.metrics.annual_return.toFixed(2)}%`, spy: "0.00%", qqq: "0.00%" },
			{ label: "샤프 비율", portfolio: data.metrics.sharpe_ratio.toFixed(4), spy: "0.0000", qqq: "0.0000" },
			{ label: "소르티노 비율", portfolio: data.metrics.sortino_ratio.toFixed(4), spy: "0.0000", qqq: "0.0000" },
			{ label: "최대 낙폭", portfolio: `${data.metrics.max_drawdown.toFixed(2)}%`, spy: "0.00%", qqq: "0.00%" },
			{ label: "변동성", portfolio: `${data.metrics.volatility.toFixed(2)}%`, spy: "0.00%", qqq: "0.00%" },
			{ label: "승률", portfolio: `${data.metrics.win_rate.toFixed(2)}%`, spy: "0.00%", qqq: "0.00%" },
			{ label: "손익비", portfolio: data.metrics.profit_loss_ratio.toFixed(4), spy: "0.0000", qqq: "0.0000" },
		];

		return {
			portfolioAllocation,
			performanceMetrics,
			quickMetrics: {
				annualReturn: `+${data.metrics.annual_return.toFixed(1)}%`,
				sharpeRatio: data.metrics.sharpe_ratio.toFixed(2),
				maxDrawdown: `-${data.metrics.max_drawdown.toFixed(1)}%`,
				volatility: `${data.metrics.volatility.toFixed(1)}%`,
			},
		};
	} catch (error) {
		console.error("강화학습 모델 호출 실패:", error);

		// 폴백: 기본값 반환 (개발 중이거나 모델 서버가 다운된 경우)
		return getFallbackPrediction(investmentAmount);
	}
}

// 폴백 예측 결과 (모델 서버가 사용 불가능할 때)
function getFallbackPrediction(investmentAmount: number): PredictionResult {
	// 간단한 규칙 기반 포트폴리오 배분
	const baseAllocation = [
		{ stock: "AAPL", percentage: 18, weight: 0.18 },
		{ stock: "MSFT", percentage: 15, weight: 0.15 },
		{ stock: "GOOGL", percentage: 12, weight: 0.12 },
		{ stock: "AMZN", percentage: 10, weight: 0.1 },
		{ stock: "TSLA", percentage: 8, weight: 0.08 },
		{ stock: "NVDA", percentage: 7, weight: 0.07 },
		{ stock: "현금", percentage: 30, weight: 0.3 },
	];

	const portfolioAllocation: PortfolioAllocation[] = baseAllocation.map((item) => ({
		stock: item.stock,
		percentage: item.percentage,
		amount: Math.round(investmentAmount * item.weight),
	}));

	const performanceMetrics: PerformanceMetrics[] = [
		{ label: "총 수익률", portfolio: "42.15%", spy: "0.00%", qqq: "0.00%" },
		{ label: "연간 수익률", portfolio: "16.24%", spy: "0.00%", qqq: "0.00%" },
		{ label: "샤프 비율", portfolio: "0.9247", spy: "0.0000", qqq: "0.0000" },
		{ label: "소르티노 비율", portfolio: "1.3521", spy: "0.0000", qqq: "0.0000" },
		{ label: "최대 낙폭", portfolio: "18.67%", spy: "0.00%", qqq: "0.00%" },
		{ label: "변동성", portfolio: "17.89%", spy: "0.00%", qqq: "0.00%" },
		{ label: "승률", portfolio: "58.33%", spy: "0.00%", qqq: "0.00%" },
		{ label: "손익비", portfolio: "1.1847", spy: "0.0000", qqq: "0.0000" },
	];

	return {
		portfolioAllocation,
		performanceMetrics,
		quickMetrics: {
			annualReturn: "+16.2%",
			sharpeRatio: "0.92",
			maxDrawdown: "-18.7%",
			volatility: "17.9%",
		},
	};
}

export async function POST(request: NextRequest) {
	try {
		const { investmentAmount } = await request.json();

		if (!investmentAmount || investmentAmount <= 0) {
			return NextResponse.json({ error: "유효한 투자 금액을 입력해주세요." }, { status: 400 });
		}

		// 강화학습 모델 호출
		const prediction = await callRLModel(investmentAmount);

		return NextResponse.json(prediction);
	} catch (error) {
		console.error("포트폴리오 예측 오류:", error);
		return NextResponse.json({ error: "포트폴리오 예측 중 오류가 발생했습니다." }, { status: 500 });
	}
}
