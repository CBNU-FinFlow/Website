"use client";

import { useEffect, useState, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { ArrowLeft, Brain, Activity, BarChart3, TrendingUp, Target, DollarSign, Shield, PieChart, CheckCircle, Info, AlertTriangle, TrendingDown, Globe } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import PortfolioVisualization from "@/components/PortfolioVisualization";
import XAISection from "@/components/analysis/XAISection";
import MarketStatusHeader from "@/components/analysis/MarketStatusHeader";
import PerformanceChart from "@/components/analysis/PerformanceChart";
import PortfolioStatus from "@/components/analysis/PortfolioStatus";
import PortfolioHeatmap from "@/components/analysis/PortfolioHeatmap";
import PositionDetails from "@/components/analysis/PositionDetails";
import HelpTooltip from "@/components/ui/HelpTooltip";

// 모의 데이터 생성 함수
const generateMockData = (investmentAmount: string, riskTolerance: string, investmentHorizon: string) => {
	const amount = Number(investmentAmount);
	const risk = Number(riskTolerance);

	// 포트폴리오 배분 모의 데이터 (전체 주식 포함)
	const portfolioAllocation = [
		{ stock: "AAPL", company: "Apple Inc.", percentage: 18.2, amount: amount * 0.182 },
		{ stock: "MSFT", company: "Microsoft Corp.", percentage: 16.8, amount: amount * 0.168 },
		{ stock: "GOOGL", company: "Alphabet Inc.", percentage: 12.5, amount: amount * 0.125 },
		{ stock: "AMZN", company: "Amazon.com Inc.", percentage: 10.3, amount: amount * 0.103 },
		{ stock: "TSLA", company: "Tesla Inc.", percentage: 8.7, amount: amount * 0.087 },
		{ stock: "META", company: "Meta Platforms Inc.", percentage: 7.9, amount: amount * 0.079 },
		{ stock: "NVDA", company: "NVIDIA Corp.", percentage: 7.1, amount: amount * 0.071 },
		{ stock: "NFLX", company: "Netflix Inc.", percentage: 4.3, amount: amount * 0.043 },
		{ stock: "CRM", company: "Salesforce Inc.", percentage: 3.8, amount: amount * 0.038 },
		{ stock: "ORCL", company: "Oracle Corp.", percentage: 3.2, amount: amount * 0.032 },
		{ stock: "ADBE", company: "Adobe Inc.", percentage: 2.9, amount: amount * 0.029 },
		{ stock: "현금", company: "Cash", percentage: 4.3, amount: amount * 0.043 },
	];

	// 성과 지표 모의 데이터
	const performanceMetrics = [
		{ label: "연간 수익률", portfolio: "15.2%", spy: "10.5%", qqq: "13.2%" },
		{ label: "샤프비율", portfolio: "1.08", spy: "0.85", qqq: "0.95" },
		{ label: "최대 낙폭", portfolio: "-18.3%", spy: "-23.9%", qqq: "-29.1%" },
		{ label: "변동성", portfolio: "14.1%", spy: "16.2%", qqq: "19.1%" },
		{ label: "승률", portfolio: "72.3%", spy: "68.5%", qqq: "72.8%" },
	];

	// 빠른 지표
	const quickMetrics = {
		annualReturn: "15.2%",
		sharpeRatio: "1.08",
		maxDrawdown: "-18.3%",
		volatility: "14.1%",
	};

	return { portfolioAllocation, performanceMetrics, quickMetrics };
};

// 헬퍼 함수들
const formatAmount = (amount: string) => {
	return Number(amount).toLocaleString();
};

const getRiskLabel = (risk: string) => {
	const riskNum = Number(risk);
	if (riskNum <= 3) return "안전형";
	if (riskNum <= 6) return "중간형";
	return "공격형";
};

const getHorizonLabel = (horizon: string) => {
	const horizonNum = Number(horizon);
	if (horizonNum <= 6) return "단기 (6개월 이하)";
	if (horizonNum <= 24) return "중기 (2년 이하)";
	return "장기 (2년 이상)";
};

// 실제 분석 결과를 보여주는 컴포넌트
function AnalysisResultsContent() {
	const searchParams = useSearchParams();
	const router = useRouter();
	const [isLoading, setIsLoading] = useState(true);
	const [xaiData, setXaiData] = useState<any>(null);
	const [isLoadingXAI, setIsLoadingXAI] = useState(false);
	const [xaiProgress, setXaiProgress] = useState(0);

	// URL 파라미터에서 분석 데이터 가져오기
	const investmentAmount = searchParams.get("amount") || "1000000";
	const riskTolerance = searchParams.get("risk") || "5";
	const investmentHorizon = searchParams.get("horizon") || "12";

	// 모의 데이터 생성
	const { portfolioAllocation, performanceMetrics, quickMetrics } = generateMockData(investmentAmount, riskTolerance, investmentHorizon);

	// XAI 분석 핸들러
	const handleXAIAnalysis = async (analysisType: "fast" | "accurate") => {
		setIsLoadingXAI(true);
		setXaiProgress(0);

		try {
			// 진행률 시뮬레이션
			const steps = analysisType === "fast" ? 10 : 30;
			for (let i = 0; i <= steps; i++) {
				await new Promise((resolve) => setTimeout(resolve, analysisType === "fast" ? 200 : 100));
				setXaiProgress((i / steps) * 100);
			}

			// 모의 XAI 데이터 (올바른 형식)
			const mockXAIData = {
				explanation_text: `이 포트폴리오는 ${getRiskLabel(
					riskTolerance
				)} 투자 성향에 맞춰 구성되었다.\n\n주요 선택 요인:\n1. 기술주 중심의 성장 전략 (65.2%)\n2. 적정 수준의 위험 분산\n3. 높은 유동성 확보\n\n특히 Apple(18.2%)과 Microsoft(16.8%)의 비중이 높은 것은 안정적인 수익성과 지속적인 성장 가능성을 고려한 결과다. MACD와 RSI 지표가 강한 매수 신호를 보였으며, 높은 거래량과 함께 상승 모멘텀이 확인되었다.`,
				feature_importance: [
					{ feature_name: "MACD", asset_name: "AAPL", importance_score: 0.18 },
					{ feature_name: "RSI", asset_name: "MSFT", importance_score: 0.16 },
					{ feature_name: "Volume", asset_name: "GOOGL", importance_score: 0.14 },
					{ feature_name: "MA14", asset_name: "AMZN", importance_score: 0.12 },
					{ feature_name: "Close", asset_name: "TSLA", importance_score: 0.11 },
					{ feature_name: "High", asset_name: "META", importance_score: 0.09 },
					{ feature_name: "Open", asset_name: "NVDA", importance_score: 0.08 },
					{ feature_name: "MA21", asset_name: "NFLX", importance_score: 0.07 },
					{ feature_name: "Low", asset_name: "CRM", importance_score: 0.05 },
				],
				attention_weights: [
					{ from_asset: "AAPL", to_asset: "MSFT", weight: 0.95 },
					{ from_asset: "GOOGL", to_asset: "AMZN", weight: 0.87 },
					{ from_asset: "TSLA", to_asset: "META", weight: 0.73 },
					{ from_asset: "MSFT", to_asset: "NVDA", weight: 0.68 },
					{ from_asset: "AAPL", to_asset: "GOOGL", weight: 0.62 },
					{ from_asset: "META", to_asset: "NFLX", weight: 0.58 },
					{ from_asset: "AMZN", to_asset: "CRM", weight: 0.52 },
					{ from_asset: "NVDA", to_asset: "ORCL", weight: 0.48 },
				],
			};

			setXaiData(mockXAIData);
		} catch (error) {
			console.error("XAI 분석 오류:", error);
		} finally {
			setIsLoadingXAI(false);
			setXaiProgress(0);
		}
	};

	useEffect(() => {
		// 페이지 로딩 시뮬레이션
		setTimeout(() => {
			setIsLoading(false);
		}, 1000);
	}, []);

	if (isLoading) {
		return (
			<div className="fixed inset-0 bg-gray-50 flex items-center justify-center z-50">
				<div className="text-center space-y-4">
					<div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
					<p className="text-gray-600">결과를 준비하고 있습니다..</p>
				</div>
			</div>
		);
	}

	return (
		<div className="min-h-screen bg-gray-50">
			{/* 상단 헤더 */}
			<div className="bg-white border-b border-gray-200 sticky top-0 z-10">
				<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
					<div className="flex items-center justify-between h-16">
						<div className="flex items-center space-x-4">
							<Button variant="ghost" size="sm" onClick={() => router.push("/")} className="text-gray-600 hover:text-gray-900">
								<ArrowLeft className="w-4 h-4 mr-2" />
								홈으로 돌아가기
							</Button>
							<div className="flex items-center space-x-2">
								<div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
									<Activity className="w-4 h-4 text-white" />
								</div>
								<h1 className="text-xl font-bold text-gray-900">AI 포트폴리오 분석 결과</h1>
								<Badge className="bg-green-100 text-green-700 border-0">분석 완료</Badge>
							</div>
						</div>
						<Button
							onClick={() => router.push("/analysis/loading?amount=" + investmentAmount + "&risk=" + riskTolerance + "&horizon=" + investmentHorizon)}
							className="bg-blue-600 hover:bg-blue-700 text-white"
						>
							<Brain className="w-4 h-4 mr-2" />
							재분석
						</Button>
					</div>
				</div>
			</div>

			{/* 메인 콘텐츠 */}
			<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
				<div className="space-y-8">
					{/* 실시간 시장 상황 헤더 */}
					<MarketStatusHeader />

					{/* 포트폴리오 성과 대시보드 */}
					<div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
						{/* 메인 성과 차트 */}
						<PerformanceChart quickMetrics={quickMetrics} investmentAmount={investmentAmount} />

						{/* 실시간 포트폴리오 상태 */}
						<PortfolioStatus investmentAmount={investmentAmount} quickMetrics={quickMetrics} portfolioAllocation={portfolioAllocation} />
					</div>

					{/* 포트폴리오 히트맵 및 상세 배분 */}
					<div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
						{/* 포트폴리오 히트맵 */}
						<PortfolioHeatmap portfolioAllocation={portfolioAllocation} />

						{/* 상세 포지션 리스트 */}
						<PositionDetails portfolioAllocation={portfolioAllocation} />
					</div>

					{/* 고급 분석 메트릭 */}
					<div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
						<Card className="border border-gray-200 bg-white">
							<CardContent className="p-4">
								<div className="flex items-center justify-between mb-3">
									<div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center">
										<TrendingUp className="h-5 w-5 text-white" />
									</div>
									<div className="text-xs text-gray-500">vs 벤치마크</div>
								</div>
								<div className="space-y-1">
									<div className="flex items-center space-x-1">
										<div className="text-sm text-gray-600">정보 비율</div>
										<HelpTooltip
											title="정보 비율 (Information Ratio)"
											description="포트폴리오가 벤치마크를 얼마나 효율적으로 초과 수익을 달성하는지 측정하는 지표다. 초과 수익을 변동성으로 나눈 값으로, 높을수록 위험 대비 초과 수익이 우수하다는 의미다."
										/>
									</div>
									<div className="text-xl font-bold text-gray-900">0.45</div>
									<div className="text-xs text-green-600">+0.12 vs S&P500</div>
								</div>
							</CardContent>
						</Card>

						<Card className="border border-gray-200 bg-white">
							<CardContent className="p-4">
								<div className="flex items-center justify-between mb-3">
									<div className="w-10 h-10 bg-gradient-to-br from-green-500 to-green-600 rounded-lg flex items-center justify-center">
										<Target className="h-5 w-5 text-white" />
									</div>
									<div className="text-xs text-gray-500">리스크 조정</div>
								</div>
								<div className="space-y-1">
									<div className="flex items-center space-x-1">
										<div className="text-sm text-gray-600">트레이너 비율</div>
										<HelpTooltip
											title="트레이너 비율 (Treynor Ratio)"
											description="포트폴리오의 초과 수익을 시장 위험(베타)으로 나눈 지표다. 시장 위험 대비 얼마나 효과적으로 수익을 창출하는지 보여주며, 높을수록 시장 위험 대비 수익률이 우수하다."
										/>
									</div>
									<div className="text-xl font-bold text-gray-900">1.23</div>
									<div className="text-xs text-blue-600">우수</div>
								</div>
							</CardContent>
						</Card>

						<Card className="border border-gray-200 bg-white">
							<CardContent className="p-4">
								<div className="flex items-center justify-between mb-3">
									<div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg flex items-center justify-center">
										<BarChart3 className="h-5 w-5 text-white" />
									</div>
									<div className="text-xs text-gray-500">분산도</div>
								</div>
								<div className="space-y-1">
									<div className="flex items-center space-x-1">
										<div className="text-sm text-gray-600">상관계수</div>
										<HelpTooltip
											title="상관계수 (Correlation Coefficient)"
											description="포트폴리오 내 자산들이 서로 얼마나 비슷하게 움직이는지 나타내는 지표다. 0에 가까울수록 독립적이고, 1에 가까울수록 동조화된다. 적절한 분산을 위해서는 낮은 상관관계가 유리하다."
										/>
									</div>
									<div className="text-xl font-bold text-gray-900">0.78</div>
									<div className="text-xs text-orange-600">적정</div>
								</div>
							</CardContent>
						</Card>

						<Card className="border border-gray-200 bg-white">
							<CardContent className="p-4">
								<div className="flex items-center justify-between mb-3">
									<div className="w-10 h-10 bg-gradient-to-br from-red-500 to-red-600 rounded-lg flex items-center justify-center">
										<AlertTriangle className="h-5 w-5 text-white" />
									</div>
									<div className="text-xs text-gray-500">95% 신뢰구간</div>
								</div>
								<div className="space-y-1">
									<div className="flex items-center space-x-1">
										<div className="text-sm text-gray-600">VaR (1일)</div>
										<HelpTooltip
											title="VaR (Value at Risk)"
											description="95% 확률로 하루 동안 발생할 수 있는 최대 손실 금액을 나타낸다. 예를 들어 VaR이 -2.3%라면 95% 확률로 하루 손실이 2.3%를 넘지 않는다는 의미다. 리스크 관리의 핵심 지표다."
										/>
									</div>
									<div className="text-xl font-bold text-gray-900">-2.3%</div>
									<div className="text-xs text-red-600">-{(Number(investmentAmount) * 0.023).toLocaleString()}원</div>
								</div>
							</CardContent>
						</Card>
					</div>

					{/* 탭 메뉴 */}
					<Tabs defaultValue="overview" className="w-full">
						<TabsList className="grid w-full grid-cols-3">
							<TabsTrigger value="overview" className="flex items-center space-x-2">
								<BarChart3 className="h-4 w-4" />
								<span>포트폴리오 개요</span>
							</TabsTrigger>
							<TabsTrigger value="analysis" className="flex items-center space-x-2">
								<Activity className="h-4 w-4" />
								<span>상세 분석</span>
							</TabsTrigger>
							<TabsTrigger value="xai" className="flex items-center space-x-2">
								<Brain className="h-4 w-4" />
								<span>AI 설명</span>
							</TabsTrigger>
						</TabsList>

						<TabsContent value="overview" className="space-y-6 mt-6">
							{/* 투자 전략 요약 */}
							<Card className="border border-gray-200 bg-white">
								<CardHeader>
									<CardTitle className="flex items-center space-x-2">
										<Target className="h-5 w-5 text-blue-600" />
										<span>투자 전략 요약</span>
									</CardTitle>
									<CardDescription>AI가 분석한 최적 투자 전략</CardDescription>
								</CardHeader>
								<CardContent>
									<div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
										<div className="space-y-4">
											<div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
												<div className="flex items-center space-x-2 mb-2">
													<DollarSign className="h-4 w-4 text-blue-600" />
													<span className="text-sm font-medium text-blue-900">투자 정보</span>
												</div>
												<div className="space-y-2 text-sm">
													<div className="flex justify-between">
														<span className="text-gray-600">투자 금액</span>
														<span className="font-medium text-gray-900">{formatAmount(investmentAmount)}원</span>
													</div>
													<div className="flex justify-between">
														<span className="text-gray-600">투자 성향</span>
														<span className="font-medium text-gray-900">{getRiskLabel(riskTolerance)}</span>
													</div>
													<div className="flex justify-between">
														<span className="text-gray-600">투자 기간</span>
														<span className="font-medium text-gray-900">{getHorizonLabel(investmentHorizon)}</span>
													</div>
												</div>
											</div>
										</div>
										<div className="space-y-4">
											<div className="bg-green-50 p-4 rounded-lg border border-green-100">
												<div className="flex items-center space-x-2 mb-2">
													<TrendingUp className="h-4 w-4 text-green-600" />
													<span className="text-sm font-medium text-green-900">예상 성과</span>
												</div>
												<div className="space-y-2 text-sm">
													<div className="flex justify-between">
														<span className="text-gray-600">연간 수익률</span>
														<span className="font-medium text-green-600">{quickMetrics.annualReturn}</span>
													</div>
													<div className="flex justify-between">
														<span className="text-gray-600">예상 수익금</span>
														<span className="font-medium text-green-600">+{(Number(investmentAmount) * (parseFloat(quickMetrics.annualReturn.replace("%", "")) / 100)).toLocaleString()}원</span>
													</div>
													<div className="flex justify-between">
														<span className="text-gray-600">승률</span>
														<span className="font-medium text-green-600">{performanceMetrics.find((m) => m.label === "승률")?.portfolio || "72.3%"}</span>
													</div>
												</div>
											</div>
										</div>
										<div className="space-y-4">
											<div className="bg-purple-50 p-4 rounded-lg border border-purple-100">
												<div className="flex items-center space-x-2 mb-2">
													<Shield className="h-4 w-4 text-purple-600" />
													<span className="text-sm font-medium text-purple-900">리스크 관리</span>
												</div>
												<div className="space-y-2 text-sm">
													<div className="flex justify-between">
														<span className="text-gray-600">최대 낙폭</span>
														<span className="font-medium text-red-600">{quickMetrics.maxDrawdown}</span>
													</div>
													<div className="flex justify-between">
														<span className="text-gray-600">변동성</span>
														<span className="font-medium text-orange-600">{quickMetrics.volatility}</span>
													</div>
													<div className="flex justify-between">
														<span className="text-gray-600">샤프 비율</span>
														<span className="font-medium text-blue-600">{quickMetrics.sharpeRatio}</span>
													</div>
												</div>
											</div>
										</div>
									</div>
								</CardContent>
							</Card>

							{/* 섹터 및 지역 분산 */}
							<div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
								<Card className="border border-gray-200 bg-white">
									<CardHeader>
										<CardTitle className="flex items-center space-x-2">
											<PieChart className="h-5 w-5 text-blue-600" />
											<span>섹터 분산</span>
											<HelpTooltip
												title="섹터 분산 (Sector Diversification)"
												description="다양한 업종에 투자하여 특정 섹터의 위험을 분산시키는 전략이다. 기술주, 소비재, 헬스케어 등으로 나누어 투자함으로써 한 분야의 충격이 전체 포트폴리오에 미치는 영향을 줄일 수 있다."
											/>
										</CardTitle>
										<CardDescription>업종별 투자 비중 및 리스크 분산</CardDescription>
									</CardHeader>
									<CardContent>
										<div className="space-y-4">
											{/* 섹터 분산 차트 (도넛 차트 모의) */}
											<div className="relative h-40 flex items-center justify-center">
												<div className="w-28 h-28 rounded-full border-8 border-gray-200 relative">
													{/* 기술주 65.2% */}
													<div
														className="absolute inset-0 rounded-full"
														style={{
															background: `conic-gradient(#3B82F6 0deg ${65.2 * 3.6}deg, transparent ${65.2 * 3.6}deg)`,
														}}
													></div>
													{/* 소비재 18.4% */}
													<div
														className="absolute inset-0 rounded-full"
														style={{
															background: `conic-gradient(transparent 0deg ${65.2 * 3.6}deg, #10B981 ${65.2 * 3.6}deg ${(65.2 + 18.4) * 3.6}deg, transparent ${(65.2 + 18.4) * 3.6}deg)`,
														}}
													></div>
													{/* 헬스케어 12.1% */}
													<div
														className="absolute inset-0 rounded-full"
														style={{
															background: `conic-gradient(transparent 0deg ${(65.2 + 18.4) * 3.6}deg, #8B5CF6 ${(65.2 + 18.4) * 3.6}deg ${(65.2 + 18.4 + 12.1) * 3.6}deg, transparent ${
																(65.2 + 18.4 + 12.1) * 3.6
															}deg)`,
														}}
													></div>
													{/* 기타 4.3% */}
													<div
														className="absolute inset-0 rounded-full"
														style={{
															background: `conic-gradient(transparent 0deg ${(65.2 + 18.4 + 12.1) * 3.6}deg, #F59E0B ${(65.2 + 18.4 + 12.1) * 3.6}deg 360deg)`,
														}}
													></div>
													<div className="absolute inset-3 bg-white rounded-full flex items-center justify-center">
														<div className="text-center">
															<div className="text-lg font-bold text-gray-900">{portfolioAllocation.length}</div>
															<div className="text-xs text-gray-600">종목</div>
														</div>
													</div>
												</div>
											</div>

											<div className="space-y-3">
												<div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
													<div className="flex items-center space-x-2">
														<div className="w-3 h-3 bg-blue-500 rounded-full"></div>
														<span className="text-sm font-medium text-gray-700">기술주</span>
													</div>
													<div className="text-right">
														<div className="font-bold text-gray-900">65.2%</div>
														<div className="text-xs text-gray-500">고성장</div>
													</div>
												</div>
												<div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
													<div className="flex items-center space-x-2">
														<div className="w-3 h-3 bg-green-500 rounded-full"></div>
														<span className="text-sm font-medium text-gray-700">소비재</span>
													</div>
													<div className="text-right">
														<div className="font-bold text-gray-900">18.4%</div>
														<div className="text-xs text-gray-500">안정성</div>
													</div>
												</div>
												<div className="flex justify-between items-center p-3 bg-purple-50 rounded-lg">
													<div className="flex items-center space-x-2">
														<div className="w-3 h-3 bg-purple-500 rounded-full"></div>
														<span className="text-sm font-medium text-gray-700">헬스케어</span>
													</div>
													<div className="text-right">
														<div className="font-bold text-gray-900">12.1%</div>
														<div className="text-xs text-gray-500">방어적</div>
													</div>
												</div>
												<div className="flex justify-between items-center p-3 bg-orange-50 rounded-lg">
													<div className="flex items-center space-x-2">
														<div className="w-3 h-3 bg-orange-500 rounded-full"></div>
														<span className="text-sm font-medium text-gray-700">기타</span>
													</div>
													<div className="text-right">
														<div className="font-bold text-gray-900">4.3%</div>
														<div className="text-xs text-gray-500">분산</div>
													</div>
												</div>
											</div>
										</div>
									</CardContent>
								</Card>

								<Card className="border border-gray-200 bg-white">
									<CardHeader>
										<CardTitle className="flex items-center space-x-2">
											<Globe className="h-5 w-5 text-green-600" />
											<span>지역 분산</span>
											<HelpTooltip
												title="지역 분산 (Geographic Diversification)"
												description="여러 국가와 지역에 투자하여 특정 국가의 경제적, 정치적 위험을 분산시키는 전략이다. 환율 변동과 각국의 경제 사이클 차이를 이용해 전체적인 안정성을 높일 수 있다."
											/>
										</CardTitle>
										<CardDescription>지역별 투자 비중 및 통화 노출</CardDescription>
									</CardHeader>
									<CardContent>
										<div className="space-y-4">
											<div className="space-y-3">
												<div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
													<div className="flex items-center space-x-2">
														<div className="w-6 h-4 bg-blue-600 rounded-sm flex items-center justify-center text-white text-xs font-bold">US</div>
														<span className="text-sm font-medium text-gray-700">미국</span>
													</div>
													<div className="text-right">
														<div className="font-bold text-gray-900">78.5%</div>
														<div className="text-xs text-gray-500">USD</div>
													</div>
												</div>
												<div className="flex justify-between items-center p-3 bg-red-50 rounded-lg">
													<div className="flex items-center space-x-2">
														<div className="w-6 h-4 bg-red-600 rounded-sm flex items-center justify-center text-white text-xs font-bold">KR</div>
														<span className="text-sm font-medium text-gray-700">한국</span>
													</div>
													<div className="text-right">
														<div className="font-bold text-gray-900">15.2%</div>
														<div className="text-xs text-gray-500">KRW</div>
													</div>
												</div>
												<div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
													<div className="flex items-center space-x-2">
														<div className="w-6 h-4 bg-green-600 rounded-sm flex items-center justify-center text-white text-xs font-bold">EU</div>
														<span className="text-sm font-medium text-gray-700">유럽</span>
													</div>
													<div className="text-right">
														<div className="font-bold text-gray-900">6.3%</div>
														<div className="text-xs text-gray-500">EUR</div>
													</div>
												</div>
											</div>

											{/* 통화 헤지 정보 */}
											<div className="mt-6 p-4 bg-gray-50 rounded-lg">
												<h4 className="font-medium text-gray-900 mb-3">통화 헤지 전략</h4>
												<div className="space-y-2 text-sm">
													<div className="flex justify-between">
														<span className="text-gray-600">USD 헤지 비율</span>
														<span className="font-medium text-blue-600">65%</span>
													</div>
													<div className="flex justify-between">
														<span className="text-gray-600">통화 리스크</span>
														<span className="font-medium text-orange-600">중간</span>
													</div>
													<div className="flex justify-between">
														<span className="text-gray-600">헤지 비용</span>
														<span className="font-medium text-gray-600">연 0.8%</span>
													</div>
												</div>
											</div>
										</div>
									</CardContent>
								</Card>
							</div>

							{/* 리스크 분석 상세 */}
							<Card className="border border-gray-200 bg-white">
								<CardHeader>
									<CardTitle className="flex items-center space-x-2">
										<AlertTriangle className="h-5 w-5 text-red-600" />
										<span>리스크 분석 상세</span>
										<HelpTooltip
											title="리스크 분석 상세"
											description="포트폴리오가 직면할 수 있는 다양한 위험 요소들을 분석한다. 시장 리스크, 집중 리스크, 유동성 리스크 등을 종합적으로 평가하여 투자자가 감수해야 할 위험 수준을 파악할 수 있다."
										/>
									</CardTitle>
									<CardDescription>포트폴리오의 다양한 리스크 요소 분석</CardDescription>
								</CardHeader>
								<CardContent>
									<div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
										{/* 시장 리스크 */}
										<div className="space-y-4">
											<h4 className="font-medium text-gray-900 flex items-center space-x-2">
												<div className="w-3 h-3 bg-red-500 rounded-full"></div>
												<span>시장 리스크</span>
											</h4>
											<div className="space-y-3">
												<div className="bg-red-50 p-3 rounded-lg">
													<div className="text-sm text-gray-600 mb-1">베타 (시장 민감도)</div>
													<div className="text-lg font-bold text-red-600">1.12</div>
													<div className="text-xs text-gray-500">시장보다 12% 높은 변동성</div>
												</div>
												<div className="bg-red-50 p-3 rounded-lg">
													<div className="text-sm text-gray-600 mb-1">VaR (95%, 1일)</div>
													<div className="text-lg font-bold text-red-600">-2.3%</div>
													<div className="text-xs text-gray-500">-{(Number(investmentAmount) * 0.023).toLocaleString()}원</div>
												</div>
											</div>
										</div>

										{/* 집중 리스크 */}
										<div className="space-y-4">
											<h4 className="font-medium text-gray-900 flex items-center space-x-2">
												<div className="w-3 h-3 bg-orange-500 rounded-full"></div>
												<span>집중 리스크</span>
											</h4>
											<div className="space-y-3">
												<div className="bg-orange-50 p-3 rounded-lg">
													<div className="text-sm text-gray-600 mb-1">최대 종목 비중</div>
													<div className="text-lg font-bold text-orange-600">{Math.max(...portfolioAllocation.map((item) => item.percentage))}%</div>
													<div className="text-xs text-gray-500">{portfolioAllocation.find((item) => item.percentage === Math.max(...portfolioAllocation.map((i) => i.percentage)))?.stock}</div>
												</div>
												<div className="bg-orange-50 p-3 rounded-lg">
													<div className="text-sm text-gray-600 mb-1">섹터 집중도</div>
													<div className="text-lg font-bold text-orange-600">65.2%</div>
													<div className="text-xs text-gray-500">기술주 집중</div>
												</div>
											</div>
										</div>

										{/* 유동성 리스크 */}
										<div className="space-y-4">
											<h4 className="font-medium text-gray-900 flex items-center space-x-2">
												<div className="w-3 h-3 bg-blue-500 rounded-full"></div>
												<span>유동성 리스크</span>
											</h4>
											<div className="space-y-3">
												<div className="bg-blue-50 p-3 rounded-lg">
													<div className="text-sm text-gray-600 mb-1">평균 거래량</div>
													<div className="text-lg font-bold text-blue-600">높음</div>
													<div className="text-xs text-gray-500">대형주 중심</div>
												</div>
												<div className="bg-blue-50 p-3 rounded-lg">
													<div className="text-sm text-gray-600 mb-1">매도 소요시간</div>
													<div className="text-lg font-bold text-blue-600">1-2일</div>
													<div className="text-xs text-gray-500">예상 청산 기간</div>
												</div>
											</div>
										</div>
									</div>

									{/* 리스크 요약 */}
									<div className="mt-6 p-4 bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200 rounded-lg">
										<div className="flex items-start space-x-3">
											<Info className="h-5 w-5 text-amber-600 mt-0.5" />
											<div>
												<h4 className="font-medium text-amber-900 mb-2">리스크 요약</h4>
												<p className="text-sm text-amber-800 leading-relaxed">
													현재 포트폴리오는 <span className="font-medium">중간 수준의 리스크</span>를 가지고 있다. 기술주 집중도가 높아 시장 변동성에 민감하지만, 우수한 유동성과 분산 투자로 리스크가
													적절히 관리되고 있다. 정기적인 리밸런싱을 통해 리스크를 최적화할 수 있다.
												</p>
											</div>
										</div>
									</div>
								</CardContent>
							</Card>
						</TabsContent>

						<TabsContent value="analysis" className="space-y-6 mt-6">
							{/* 포트폴리오 시각화 */}
							<PortfolioVisualization portfolioAllocation={portfolioAllocation} performanceMetrics={performanceMetrics} showResults={true} />

							{/* 추가 분석 차트들 */}
							<div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
								{/* 수익률 분포 */}
								<Card className="border border-gray-200 bg-white">
									<CardHeader>
										<CardTitle className="flex items-center space-x-2">
											<BarChart3 className="h-5 w-5 text-blue-600" />
											<span>수익률 분포</span>
											<HelpTooltip
												title="수익률 분포 (Return Distribution)"
												description="포트폴리오의 월별 예상 수익률이 어떻게 분포되어 있는지 보여준다. 정규분포에 가까울수록 예측 가능하며, 꼬리가 두꺼울수록 극단적인 수익/손실 가능성이 높다. 평균과 표준편차를 통해 기대 수익과 변동성을 파악할 수 있다."
											/>
										</CardTitle>
										<CardDescription>월별 예상 수익률 분포</CardDescription>
									</CardHeader>
									<CardContent>
										<div className="h-56 bg-gray-50 rounded-lg p-4 relative">
											{/* 히스토그램 모의 */}
											<div className="flex items-end justify-center h-full space-x-2">
												{[-15, -10, -5, 0, 5, 10, 15, 20, 25].map((value, index) => {
													const height = Math.max(10, 80 - Math.abs(value - 8) * 3);
													const color = value < 0 ? "bg-red-400" : value < 10 ? "bg-yellow-400" : "bg-green-400";
													return (
														<div key={index} className="flex flex-col items-center">
															<div className={`${color} w-6 rounded-t`} style={{ height: `${height}px` }}></div>
															<div className="text-xs text-gray-600 mt-1">{value}%</div>
														</div>
													);
												})}
											</div>
											<div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm rounded-lg p-2 text-xs">
												<div className="text-gray-600">평균: +8.2%</div>
												<div className="text-gray-600">표준편차: 12.5%</div>
											</div>
										</div>
									</CardContent>
								</Card>

								{/* 드로우다운 분석 */}
								<Card className="border border-gray-200 bg-white">
									<CardHeader>
										<CardTitle className="flex items-center space-x-2">
											<TrendingDown className="h-5 w-5 text-red-600" />
											<span>드로우다운 분석</span>
											<HelpTooltip
												title="드로우다운 분석 (Drawdown Analysis)"
												description="포트폴리오가 고점에서 저점까지 하락한 최대 손실 구간을 분석한다. 최대 드로우다운은 투자 기간 중 겪을 수 있는 최악의 손실을 나타내며, 회복 기간은 원래 수준을 되찾는 데 걸리는 시간을 의미한다."
											/>
										</CardTitle>
										<CardDescription>최대 손실 구간 분석</CardDescription>
									</CardHeader>
									<CardContent>
										<div className="h-56 bg-gray-50 rounded-lg p-4 relative">
											{/* 드로우다운 차트 모의 */}
											<svg className="w-full h-full" viewBox="0 0 300 200">
												<path d="M 20 20 L 50 25 L 80 30 L 110 45 L 140 60 L 170 40 L 200 35 L 230 25 L 260 20" stroke="#EF4444" strokeWidth="2" fill="none" />
												<path d="M 20 20 L 50 25 L 80 30 L 110 45 L 140 60 L 170 40 L 200 35 L 230 25 L 260 20 L 260 180 L 20 180 Z" fill="#EF4444" opacity="0.2" />
											</svg>
											<div className="absolute bottom-4 left-4 space-y-1 text-xs">
												<div className="text-red-600 font-medium">최대 드로우다운: {quickMetrics.maxDrawdown}</div>
												<div className="text-gray-600">회복 기간: 약 4-6개월</div>
											</div>
										</div>
									</CardContent>
								</Card>
							</div>

							{/* 성과 비교 테이블 확장 */}
							<Card className="border border-gray-200 bg-white">
								<CardHeader>
									<CardTitle className="flex items-center space-x-2">
										<Activity className="h-5 w-5 text-green-600" />
										<span>벤치마크 대비 성과</span>
										<HelpTooltip
											title="벤치마크 대비 성과 (Benchmark Comparison)"
											description="포트폴리오의 성과를 주요 시장 지수들과 비교한 결과다. S&P 500, NASDAQ 등의 벤치마크 대비 얼마나 우수한 성과를 보이는지 확인할 수 있으며, 같은 위험 수준에서 더 높은 수익을 달성했는지 평가할 수 있다."
										/>
									</CardTitle>
									<CardDescription>주요 지수 및 ETF와의 성과 비교</CardDescription>
								</CardHeader>
								<CardContent>
									<div className="overflow-x-auto">
										<table className="w-full text-sm">
											<thead>
												<tr className="border-b border-gray-200">
													<th className="text-left py-3 px-4 font-semibold text-gray-700">지표</th>
													<th className="text-center py-3 px-4 font-semibold text-blue-600">내 포트폴리오</th>
													<th className="text-center py-3 px-4 font-semibold text-gray-500">S&P 500</th>
													<th className="text-center py-3 px-4 font-semibold text-gray-500">NASDAQ</th>
													<th className="text-center py-3 px-4 font-semibold text-gray-500">QQQ</th>
													<th className="text-center py-3 px-4 font-semibold text-gray-500">KOSPI</th>
												</tr>
											</thead>
											<tbody>
												{[
													{ label: "연간 수익률", portfolio: quickMetrics.annualReturn, sp500: "10.5%", nasdaq: "12.8%", qqq: "13.2%", kospi: "8.7%" },
													{ label: "샤프 비율", portfolio: quickMetrics.sharpeRatio, sp500: "0.85", nasdaq: "0.92", qqq: "0.95", kospi: "0.72" },
													{ label: "최대 낙폭", portfolio: quickMetrics.maxDrawdown, sp500: "-23.9%", nasdaq: "-28.2%", qqq: "-29.1%", kospi: "-31.2%" },
													{ label: "변동성", portfolio: quickMetrics.volatility, sp500: "16.2%", nasdaq: "18.7%", qqq: "19.1%", kospi: "22.4%" },
													{ label: "승률", portfolio: performanceMetrics.find((m) => m.label === "승률")?.portfolio || "72.3%", sp500: "68.5%", nasdaq: "71.2%", qqq: "72.8%", kospi: "64.1%" },
													{ label: "정보 비율", portfolio: "0.45", sp500: "0.00", nasdaq: "0.15", qqq: "0.18", kospi: "-0.12" },
												].map((metric, index) => (
													<tr key={index} className="hover:bg-gray-50">
														<td className="py-3 px-4 font-medium text-gray-900">{metric.label}</td>
														<td className="py-3 px-4 text-center font-bold text-blue-600">{metric.portfolio}</td>
														<td className="py-3 px-4 text-center text-gray-600">{metric.sp500}</td>
														<td className="py-3 px-4 text-center text-gray-600">{metric.nasdaq}</td>
														<td className="py-3 px-4 text-center text-gray-600">{metric.qqq}</td>
														<td className="py-3 px-4 text-center text-gray-600">{metric.kospi}</td>
													</tr>
												))}
											</tbody>
										</table>
									</div>

									<div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
										<div className="flex items-center space-x-2">
											<CheckCircle className="h-4 w-4 text-green-600" />
											<span className="text-sm text-green-800">내 포트폴리오는 주요 벤치마크 대비 우수한 리스크 조정 수익률을 보여준다.</span>
										</div>
									</div>
								</CardContent>
							</Card>

							{/* 시장 상관관계 분석 */}
							<Card className="border border-gray-200 bg-white">
								<CardHeader>
									<CardTitle className="flex items-center space-x-2">
										<Globe className="h-5 w-5 text-purple-600" />
										<span>시장 상관관계 분석</span>
										<HelpTooltip
											title="시장 상관관계 분석 (Market Correlation Analysis)"
											description="포트폴리오 내 자산들이 서로 얼마나 비슷하게 움직이는지 분석한다. 높은 상관관계는 집중 리스크를, 낮은 상관관계는 좋은 분산 효과를 의미한다. 상관계수가 0.7 이상이면 높은 상관관계, 0.3 이하면 낮은 상관관계로 분류한다."
										/>
									</CardTitle>
									<CardDescription>포트폴리오 내 자산들의 상관관계 매트릭스</CardDescription>
								</CardHeader>
								<CardContent>
									<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
										<div className="space-y-4">
											<h4 className="font-medium text-gray-900">높은 상관관계 (&gt;0.7)</h4>
											<div className="space-y-2">
												<div className="flex justify-between items-center p-2 bg-red-50 rounded border border-red-100">
													<span className="text-sm">AAPL ↔ MSFT</span>
													<span className="font-bold text-red-600">0.83</span>
												</div>
												<div className="flex justify-between items-center p-2 bg-red-50 rounded border border-red-100">
													<span className="text-sm">GOOGL ↔ META</span>
													<span className="font-bold text-red-600">0.76</span>
												</div>
											</div>
										</div>
										<div className="space-y-4">
											<h4 className="font-medium text-gray-900">낮은 상관관계 (&lt;0.3)</h4>
											<div className="space-y-2">
												<div className="flex justify-between items-center p-2 bg-green-50 rounded border border-green-100">
													<span className="text-sm">TSLA ↔ 현금</span>
													<span className="font-bold text-green-600">0.12</span>
												</div>
												<div className="flex justify-between items-center p-2 bg-green-50 rounded border border-green-100">
													<span className="text-sm">NFLX ↔ ORCL</span>
													<span className="font-bold text-green-600">0.24</span>
												</div>
											</div>
										</div>
									</div>
								</CardContent>
							</Card>

							{/* 리스크 분해 분석 */}
							<Card className="border border-gray-200 bg-white">
								<CardHeader>
									<CardTitle className="flex items-center space-x-2">
										<PieChart className="h-5 w-5 text-orange-600" />
										<span>리스크 분해 분석</span>
										<HelpTooltip
											title="리스크 분해 분석 (Risk Decomposition Analysis)"
											description="포트폴리오의 전체 리스크가 어디서 오는지 분석한다. 개별 종목의 리스크 기여도와 시장 리스크 vs 특정 리스크의 비중을 파악할 수 있다. 이를 통해 리스크 집중도를 확인하고 분산 투자 효과를 평가할 수 있다."
										/>
									</CardTitle>
									<CardDescription>포트폴리오 전체 리스크의 구성 요소별 기여도</CardDescription>
								</CardHeader>
								<CardContent>
									<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
										<div className="space-y-4">
											<h4 className="font-medium text-gray-900">개별 리스크 기여도</h4>
											<div className="space-y-3">
												<div className="flex justify-between items-center">
													<span className="text-sm text-gray-600">AAPL</span>
													<div className="flex items-center space-x-2">
														<div className="w-16 bg-gray-200 rounded-full h-2">
															<div className="bg-blue-600 h-2 rounded-full" style={{ width: "22%" }}></div>
														</div>
														<span className="text-sm font-medium">22%</span>
													</div>
												</div>
												<div className="flex justify-between items-center">
													<span className="text-sm text-gray-600">MSFT</span>
													<div className="flex items-center space-x-2">
														<div className="w-16 bg-gray-200 rounded-full h-2">
															<div className="bg-blue-600 h-2 rounded-full" style={{ width: "19%" }}></div>
														</div>
														<span className="text-sm font-medium">19%</span>
													</div>
												</div>
												<div className="flex justify-between items-center">
													<span className="text-sm text-gray-600">GOOGL</span>
													<div className="flex items-center space-x-2">
														<div className="w-16 bg-gray-200 rounded-full h-2">
															<div className="bg-blue-600 h-2 rounded-full" style={{ width: "15%" }}></div>
														</div>
														<span className="text-sm font-medium">15%</span>
													</div>
												</div>
												<div className="flex justify-between items-center">
													<span className="text-sm text-gray-600">기타</span>
													<div className="flex items-center space-x-2">
														<div className="w-16 bg-gray-200 rounded-full h-2">
															<div className="bg-blue-600 h-2 rounded-full" style={{ width: "44%" }}></div>
														</div>
														<span className="text-sm font-medium">44%</span>
													</div>
												</div>
											</div>
										</div>
										<div className="space-y-4">
											<h4 className="font-medium text-gray-900">리스크 유형별 분석</h4>
											<div className="space-y-3">
												<div className="p-3 bg-blue-50 rounded-lg border border-blue-100">
													<div className="text-sm text-gray-600 mb-1">시장 리스크</div>
													<div className="text-lg font-bold text-blue-600">78.2%</div>
													<div className="text-xs text-gray-500">전체 변동성 중 시장 요인</div>
												</div>
												<div className="p-3 bg-orange-50 rounded-lg border border-orange-100">
													<div className="text-sm text-gray-600 mb-1">특정 리스크</div>
													<div className="text-lg font-bold text-orange-600">21.8%</div>
													<div className="text-xs text-gray-500">개별 종목 고유 변동성</div>
												</div>
											</div>
										</div>
									</div>
								</CardContent>
							</Card>
						</TabsContent>

						<TabsContent value="xai" className="space-y-6 mt-6">
							<XAISection onXAIAnalysis={handleXAIAnalysis} isLoadingXAI={isLoadingXAI} xaiData={xaiData} xaiProgress={xaiProgress} />
						</TabsContent>
					</Tabs>
				</div>
			</div>
		</div>
	);
}

export default function AnalysisResultsPage() {
	return (
		<Suspense
			fallback={
				<div className="fixed inset-0 bg-gray-50 flex items-center justify-center z-50">
					<div className="text-center space-y-4">
						<div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
						<p className="text-gray-600">결과를 준비하고 있습니다..</p>
					</div>
				</div>
			}
		>
			<AnalysisResultsContent />
		</Suspense>
	);
}
