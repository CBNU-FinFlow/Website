"use client";

import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { PortfolioAllocation, PerformanceMetrics, QuickMetrics, XAIData } from "@/lib/types";
import PortfolioVisualization from "./PortfolioVisualization";
import XAIVisualization from "./XAIVisualization";
import { X, Brain, TrendingUp, DollarSign, PieChart, Calendar, ArrowLeft, BarChart3, Activity, Target, AlertTriangle, CheckCircle, Info, Maximize, Minimize } from "lucide-react";

interface AnalysisModalProps {
	isOpen: boolean;
	onClose: () => void;
	portfolioAllocation: PortfolioAllocation[];
	performanceMetrics: PerformanceMetrics[];
	quickMetrics: QuickMetrics;
	investmentAmount: string;
	riskTolerance: string;
	investmentHorizon: number[];
	onXAIAnalysis: (method: "fast" | "accurate") => void;
	xaiData: XAIData | null;
	isLoadingXAI: boolean;
	xaiProgress: number;
	showXAI: boolean;
	onBackToPortfolio: () => void;
}

export default function AnalysisModal({
	isOpen,
	onClose,
	portfolioAllocation,
	performanceMetrics,
	quickMetrics,
	investmentAmount,
	riskTolerance,
	investmentHorizon,
	onXAIAnalysis,
	xaiData,
	isLoadingXAI,
	xaiProgress,
	showXAI,
	onBackToPortfolio,
}: AnalysisModalProps) {
	const [isFullScreen, setIsFullScreen] = useState(false);

	const toggleFullScreen = () => {
		setIsFullScreen(!isFullScreen);
	};

	const getRiskLabel = (risk: string) => {
		const labels = {
			conservative: "보수적",
			moderate: "보통",
			aggressive: "적극적",
		};
		return labels[risk as keyof typeof labels] || risk;
	};

	const getHorizonLabel = (days: number) => {
		const months = Math.round(days / 21);
		if (months <= 3) return `단기 (${months}개월)`;
		if (months <= 6) return `중단기 (${months}개월)`;
		if (months <= 12) return `중기 (${months}개월)`;
		if (months <= 24) return `중장기 (${months}개월)`;
		return `장기 (${months}개월)`;
	};

	const formatAmount = (amount: string) => {
		return Number(amount).toLocaleString();
	};

	const getModalClassName = () => {
		const baseClasses = "bg-white rounded-lg shadow-lg";
		if (isFullScreen) {
			return `${baseClasses} dialog-fullscreen-max`;
		}
		return `${baseClasses} dialog-fullscreen`;
	};

	if (showXAI) {
		return (
			<Dialog open={isOpen} onOpenChange={onClose}>
				<DialogContent
					className={getModalClassName()}
					// 추가 props로 포지셔닝 강제
					style={{
						position: "fixed",
						top: isFullScreen ? "0" : "50%",
						left: isFullScreen ? "0" : "50%",
						transform: isFullScreen ? "none" : "translate(-50%, -50%)",
						width: isFullScreen ? "100vw" : "95vw",
						height: isFullScreen ? "100vh" : "90vh",
						maxWidth: "none",
						maxHeight: "none",
						margin: "0",
						padding: "0",
						borderRadius: isFullScreen ? "0" : "0.5rem",
					}}
				>
					<DialogTitle className="sr-only">AI 의사결정 분석</DialogTitle>
					<div className="flex flex-col h-full bg-white">
						{/* 헤더 */}
						<div className="flex items-center justify-between p-4 border-b border-gray-200 bg-white flex-shrink-0">
							<div className="flex items-center space-x-4">
								<Button variant="ghost" size="sm" onClick={onBackToPortfolio} className="text-gray-600 hover:text-gray-900">
									<ArrowLeft className="h-4 w-4 mr-2" />
									포트폴리오로 돌아가기
								</Button>
								<div className="flex items-center space-x-2">
									<div className="w-8 h-8 bg-purple-600 rounded-lg flex items-center justify-center">
										<Brain className="w-4 h-4 text-white" />
									</div>
									<h2 className="text-xl font-bold text-gray-900">AI 의사결정 분석</h2>
									<Badge className="bg-purple-100 text-purple-700 border-0">분석 완료</Badge>
								</div>
							</div>
							<div className="flex items-center space-x-2">
								<Button variant="ghost" size="sm" onClick={toggleFullScreen} className="text-gray-600 hover:text-gray-900">
									{isFullScreen ? <Minimize className="h-4 w-4" /> : <Maximize className="h-4 w-4" />}
								</Button>
								<Button variant="ghost" size="sm" onClick={onClose} className="text-gray-600 hover:text-gray-900">
									<X className="h-4 w-4" />
								</Button>
							</div>
						</div>

						{/* XAI 컨텐츠 */}
						<div className="flex-1 overflow-y-auto bg-gray-50" style={{ maxHeight: "calc(100% - 80px)" }}>
							<XAIVisualization xaiData={xaiData} isLoading={isLoadingXAI} progress={xaiProgress} onAnalyze={onXAIAnalysis} portfolioAllocation={portfolioAllocation} />
						</div>
					</div>
				</DialogContent>
			</Dialog>
		);
	}

	return (
		<Dialog open={isOpen} onOpenChange={onClose}>
			<DialogContent className={getModalClassName()}>
				<DialogTitle className="sr-only">AI 포트폴리오 분석 결과</DialogTitle>
				<div className="flex flex-col h-full bg-white">
					{/* 헤더 */}
					<div className="flex items-center justify-between p-4 border-b border-gray-200 bg-white flex-shrink-0">
						<div className="flex items-center space-x-4">
							<div className="flex items-center space-x-2">
								<div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
									<Activity className="w-4 h-4 text-white" />
								</div>
								<h2 className="text-xl font-bold text-gray-900">AI 포트폴리오 분석 결과</h2>
								<Badge className="bg-green-100 text-green-700 border-0">분석 완료</Badge>
							</div>
						</div>
						<div className="flex items-center space-x-2">
							<Button variant="ghost" size="sm" onClick={toggleFullScreen} className="text-gray-600 hover:text-gray-900">
								{isFullScreen ? <Minimize className="h-4 w-4" /> : <Maximize className="h-4 w-4" />}
							</Button>
							<Button variant="ghost" size="sm" onClick={onClose} className="text-gray-600 hover:text-gray-900">
								<X className="h-4 w-4" />
							</Button>
						</div>
					</div>

					{/* 메인 콘텐츠 */}
					<div className="flex-1 overflow-y-auto bg-gray-50" style={{ maxHeight: "calc(100% - 80px)" }}>
						<div className="p-6 space-y-6">
							{/* 포트폴리오 적용 시뮬레이션 */}
							<Card className="border border-gray-200 bg-white">
								<CardHeader className="pb-4">
									<CardTitle className="text-lg font-bold text-gray-900">포트폴리오 적용 시뮬레이션</CardTitle>
									<CardDescription className="text-gray-600">현재 포트폴리오를 1년간 적용했을 때 예상 결과</CardDescription>
								</CardHeader>
								<CardContent className="pt-0">
									<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
										<div className="space-y-4">
											<div className="flex justify-between py-2 border-b border-gray-100">
												<span className="text-gray-600">초기 투자금</span>
												<span className="font-semibold text-gray-900">{formatAmount(investmentAmount)}원</span>
											</div>
											<div className="flex justify-between py-2 border-b border-gray-100">
												<span className="text-gray-600">예상 수익률</span>
												<span className="font-semibold text-blue-600">{quickMetrics.annualReturn}</span>
											</div>
											<div className="flex justify-between py-2 border-b border-gray-100">
												<span className="text-gray-600">예상 총 자산</span>
												<span className="font-bold text-gray-900">{(Number(investmentAmount) * (1 + parseFloat(quickMetrics.annualReturn.replace("%", "")) / 100)).toLocaleString()}원</span>
											</div>
											<div className="flex justify-between py-2">
												<span className="text-gray-600">예상 수익금</span>
												<span className="font-bold text-green-600">+{(Number(investmentAmount) * (parseFloat(quickMetrics.annualReturn.replace("%", "")) / 100)).toLocaleString()}원</span>
											</div>
										</div>
										<div className="grid grid-cols-2 gap-4">
											<div className="bg-gray-50 p-4 rounded-lg text-center">
												<div className="text-xl font-bold text-gray-900">{quickMetrics.sharpeRatio}</div>
												<div className="text-sm text-gray-600">샤프 비율</div>
											</div>
											<div className="bg-gray-50 p-4 rounded-lg text-center">
												<div className="text-xl font-bold text-red-600">{quickMetrics.maxDrawdown}</div>
												<div className="text-sm text-gray-600">최대 낙폭</div>
											</div>
											<div className="bg-gray-50 p-4 rounded-lg text-center">
												<div className="text-xl font-bold text-gray-900">{quickMetrics.volatility}</div>
												<div className="text-sm text-gray-600">변동성</div>
											</div>
											<div className="bg-gray-50 p-4 rounded-lg text-center">
												<div className="text-xl font-bold text-green-600">{performanceMetrics.find((m) => m.label === "승률")?.portfolio || "N/A"}</div>
												<div className="text-sm text-gray-600">승률</div>
											</div>
										</div>
									</div>
								</CardContent>
							</Card>

							{/* 투자 정보 요약 */}
							<div className="grid grid-cols-1 md:grid-cols-4 gap-4">
								<Card className="border border-gray-200 bg-white">
									<CardContent className="p-4">
										<div className="flex items-center space-x-3">
											<div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
												<DollarSign className="h-5 w-5 text-white" />
											</div>
											<div>
												<span className="text-sm text-gray-600">투자 금액</span>
												<div className="text-lg font-bold text-gray-900">{formatAmount(investmentAmount)}원</div>
											</div>
										</div>
									</CardContent>
								</Card>
								<Card className="border border-gray-200 bg-white">
									<CardContent className="p-4">
										<div className="flex items-center space-x-3">
											<div className="w-10 h-10 bg-green-600 rounded-lg flex items-center justify-center">
												<PieChart className="h-5 w-5 text-white" />
											</div>
											<div>
												<span className="text-sm text-gray-600">투자 성향</span>
												<div className="text-lg font-bold text-gray-900">{getRiskLabel(riskTolerance)}</div>
											</div>
										</div>
									</CardContent>
								</Card>
								<Card className="border border-gray-200 bg-white">
									<CardContent className="p-4">
										<div className="flex items-center space-x-3">
											<div className="w-10 h-10 bg-purple-600 rounded-lg flex items-center justify-center">
												<Calendar className="h-5 w-5 text-white" />
											</div>
											<div>
												<span className="text-sm text-gray-600">투자 기간</span>
												<div className="text-lg font-bold text-gray-900">{getHorizonLabel(investmentHorizon[0])}</div>
											</div>
										</div>
									</CardContent>
								</Card>
								<Card className="border border-gray-200 bg-white">
									<CardContent className="p-4">
										<div className="flex items-center space-x-3">
											<div className="w-10 h-10 bg-orange-600 rounded-lg flex items-center justify-center">
												<TrendingUp className="h-5 w-5 text-white" />
											</div>
											<div>
												<span className="text-sm text-gray-600">예상 수익률</span>
												<div className="text-lg font-bold text-gray-900">{quickMetrics.annualReturn}</div>
											</div>
										</div>
									</CardContent>
								</Card>
							</div>

							{/* 탭 컨텐츠 */}
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

								{/* 포트폴리오 개요 */}
								<TabsContent value="overview" className="space-y-6 mt-6">
									<div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
										{/* 포트폴리오 배분 */}
										<Card className="lg:col-span-2 border border-gray-200 bg-white">
											<CardHeader>
												<CardTitle className="flex items-center space-x-2">
													<PieChart className="h-5 w-5 text-blue-600" />
													<span>포트폴리오 배분</span>
												</CardTitle>
												<CardDescription>종목별 투자 비중 및 금액</CardDescription>
											</CardHeader>
											<CardContent>
												<div className="space-y-3">
													{portfolioAllocation.map((item, index) => (
														<div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
															<div className="flex items-center space-x-3">
																<div className={`w-3 h-3 rounded-full`} style={{ backgroundColor: `hsl(${index * 45}, 70%, 50%)` }}></div>
																<div>
																	<span className="font-medium text-gray-900">{item.stock}</span>
																	<div className="text-sm text-gray-500">기술주</div>
																</div>
															</div>
															<div className="text-right">
																<div className="font-bold text-gray-900">{item.percentage}%</div>
																<div className="text-sm text-gray-600">{item.amount.toLocaleString()}원</div>
															</div>
														</div>
													))}
												</div>
											</CardContent>
										</Card>

										{/* 주요 지표 */}
										<Card className="border border-gray-200 bg-white">
											<CardHeader>
												<CardTitle className="flex items-center space-x-2">
													<Target className="h-5 w-5 text-green-600" />
													<span>주요 지표</span>
												</CardTitle>
												<CardDescription>포트폴리오 성과 지표</CardDescription>
											</CardHeader>
											<CardContent>
												<div className="space-y-3">
													<div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
														<span className="text-sm text-gray-600">연간 수익률</span>
														<span className="font-bold text-green-600">{quickMetrics.annualReturn}</span>
													</div>
													<div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
														<span className="text-sm text-gray-600">샤프 비율</span>
														<span className="font-bold text-blue-600">{quickMetrics.sharpeRatio}</span>
													</div>
													<div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
														<span className="text-sm text-gray-600">최대 낙폭</span>
														<span className="font-bold text-red-600">{quickMetrics.maxDrawdown}</span>
													</div>
													<div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
														<span className="text-sm text-gray-600">변동성</span>
														<span className="font-bold text-gray-900">{quickMetrics.volatility}</span>
													</div>
												</div>
											</CardContent>
										</Card>
									</div>

									{/* 리스크 관리 정보 */}
									<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
										<Card className="border border-gray-200 bg-white">
											<CardHeader>
												<CardTitle className="flex items-center space-x-2">
													<AlertTriangle className="h-5 w-5 text-red-600" />
													<span>리스크 분석</span>
												</CardTitle>
												<CardDescription>포트폴리오 위험 요소 분석</CardDescription>
											</CardHeader>
											<CardContent>
												<div className="space-y-3">
													<div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
														<span className="text-sm text-gray-600">VaR (95% 신뢰구간)</span>
														<span className="font-bold text-red-600">-{(Number(investmentAmount) * 0.032).toLocaleString()}원</span>
													</div>
													<div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
														<span className="text-sm text-gray-600">베타 (시장 대비)</span>
														<span className="font-bold text-blue-600">1.12</span>
													</div>
													<div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
														<span className="text-sm text-gray-600">상관계수 (S&P 500)</span>
														<span className="font-bold text-purple-600">0.78</span>
													</div>
													<div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
														<span className="text-sm text-gray-600">정보 비율</span>
														<span className="font-bold text-green-600">0.45</span>
													</div>
													<div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
														<div className="flex items-center space-x-2">
															<Info className="h-4 w-4 text-amber-600" />
															<span className="text-sm text-amber-800">중간 수준의 시장 위험</span>
														</div>
													</div>
												</div>
											</CardContent>
										</Card>

										<Card className="border border-gray-200 bg-white">
											<CardHeader>
												<CardTitle className="flex items-center space-x-2">
													<BarChart3 className="h-5 w-5 text-purple-600" />
													<span>섹터 분석</span>
												</CardTitle>
												<CardDescription>업종별 투자 비중</CardDescription>
											</CardHeader>
											<CardContent>
												<div className="space-y-3">
													<div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
														<div className="flex items-center space-x-2">
															<div className="w-3 h-3 bg-blue-500 rounded-full"></div>
															<span className="text-sm text-gray-700">기술주</span>
														</div>
														<span className="font-bold text-gray-900">65.2%</span>
													</div>
													<div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
														<div className="flex items-center space-x-2">
															<div className="w-3 h-3 bg-green-500 rounded-full"></div>
															<span className="text-sm text-gray-700">소비재</span>
														</div>
														<span className="font-bold text-gray-900">18.4%</span>
													</div>
													<div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
														<div className="flex items-center space-x-2">
															<div className="w-3 h-3 bg-purple-500 rounded-full"></div>
															<span className="text-sm text-gray-700">헬스케어</span>
														</div>
														<span className="font-bold text-gray-900">12.1%</span>
													</div>
													<div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
														<div className="flex items-center space-x-2">
															<div className="w-3 h-3 bg-orange-500 rounded-full"></div>
															<span className="text-sm text-gray-700">기타</span>
														</div>
														<span className="font-bold text-gray-900">4.3%</span>
													</div>
													<div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
														<div className="flex items-center space-x-2">
															<CheckCircle className="h-4 w-4 text-blue-600" />
															<span className="text-sm text-blue-800">기술주 중심의 성장형 포트폴리오</span>
														</div>
													</div>
												</div>
											</CardContent>
										</Card>
									</div>

									{/* 성과 비교 테이블 */}
									<Card className="border border-gray-200 bg-white">
										<CardHeader>
											<CardTitle className="flex items-center space-x-2">
												<BarChart3 className="h-5 w-5 text-gray-700" />
												<span>벤치마크 대비 성과</span>
											</CardTitle>
											<CardDescription>S&P 500, NASDAQ 대비 포트폴리오 성과 비교</CardDescription>
										</CardHeader>
										<CardContent>
											<div className="overflow-x-auto">
												<table className="w-full text-sm">
													<thead>
														<tr className="border-b border-gray-200">
															<th className="text-left py-3 font-semibold text-gray-700">지표</th>
															<th className="text-center py-3 font-semibold text-blue-600">AI 포트폴리오</th>
															<th className="text-center py-3 font-semibold text-gray-500">S&P 500</th>
															<th className="text-center py-3 font-semibold text-gray-500">NASDAQ</th>
														</tr>
													</thead>
													<tbody>
														{performanceMetrics.slice(0, 6).map((metric, index) => (
															<tr key={index} className="border-b border-gray-100">
																<td className="py-3 font-medium text-gray-900">{metric.label}</td>
																<td className="py-3 text-center font-semibold text-blue-600">{metric.portfolio}</td>
																<td className="py-3 text-center text-gray-600">{metric.spy}</td>
																<td className="py-3 text-center text-gray-600">{metric.qqq}</td>
															</tr>
														))}
													</tbody>
												</table>
											</div>
											<div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
												<div className="flex items-center space-x-2">
													<AlertTriangle className="h-4 w-4 text-amber-600" />
													<span className="text-sm text-amber-800">과거 성과는 미래 수익을 보장하지 않으며, 투자에는 원금 손실 위험이 있다.</span>
												</div>
											</div>
										</CardContent>
									</Card>
								</TabsContent>

								{/* 상세 분석 */}
								<TabsContent value="analysis" className="space-y-6 mt-6">
									<PortfolioVisualization portfolioAllocation={portfolioAllocation} performanceMetrics={performanceMetrics} showResults={true} />
								</TabsContent>

								{/* AI 설명 */}
								<TabsContent value="xai" className="space-y-6 mt-6">
									<Card className="border border-gray-200 bg-white">
										<CardHeader>
											<CardTitle className="flex items-center space-x-2">
												<Brain className="h-5 w-5 text-blue-600" />
												<span>AI 의사결정 분석</span>
											</CardTitle>
											<CardDescription>AI가 이 포트폴리오를 선택한 이유를 자세히 알아보자</CardDescription>
										</CardHeader>
										<CardContent className="p-8">
											<div className="text-center space-y-8">
												<div className="w-16 h-16 mx-auto bg-blue-600 rounded-lg flex items-center justify-center">
													<Brain className="w-8 h-8 text-white" />
												</div>
												<div className="space-y-3">
													<h3 className="text-2xl font-bold text-gray-900">AI 의사결정 과정 분석</h3>
													<p className="text-gray-600 max-w-2xl mx-auto leading-relaxed">
														AI가 어떤 요소들을 고려하여 이 포트폴리오를 구성했는지
														<br />
														상세한 분석을 제공한다. 투자 결정의 투명성을 높이고
														<br />
														<span className="text-blue-600 font-medium">신뢰할 수 있는 투자 근거</span>를 확인할 수 있다.
													</p>
												</div>

												<div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-2xl mx-auto">
													<div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
														<div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center mx-auto mb-4">
															<Activity className="w-6 h-6 text-white" />
														</div>
														<h4 className="font-bold text-gray-900 mb-2">빠른 분석</h4>
														<p className="text-sm text-gray-600 mb-4">주요 의사결정 요소와 기본적인 설명을 제공한다</p>
														<Button onClick={() => onXAIAnalysis("fast")} disabled={isLoadingXAI} className="w-full bg-blue-600 hover:bg-blue-700 text-white">
															<div className="flex items-center space-x-2">
																<Brain className="w-4 h-4" />
																<span>5-10초 분석</span>
															</div>
														</Button>
													</div>

													<div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
														<div className="w-12 h-12 bg-gray-700 rounded-lg flex items-center justify-center mx-auto mb-4">
															<Target className="w-6 h-6 text-white" />
														</div>
														<h4 className="font-bold text-gray-900 mb-2">정밀 분석</h4>
														<p className="text-sm text-gray-600 mb-4">상세한 특성 중요도와 종목별 근거를 분석한다</p>
														<Button onClick={() => onXAIAnalysis("accurate")} disabled={isLoadingXAI} variant="outline" className="w-full border-gray-300 text-gray-700 hover:bg-gray-50">
															<div className="flex items-center space-x-2">
																<Brain className="w-4 h-4" />
																<span>30초-2분 분석</span>
															</div>
														</Button>
													</div>
												</div>

												<div className="bg-gray-50 p-6 rounded-lg border border-gray-200 max-w-3xl mx-auto">
													<h4 className="font-bold text-gray-900 mb-4">분석 내용 미리보기</h4>
													<div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
														<div className="text-center">
															<div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center mx-auto mb-2">
																<BarChart3 className="w-4 h-4 text-white" />
															</div>
															<div className="font-medium text-gray-900">특성 중요도</div>
															<div className="text-gray-600">각 요소의 영향력</div>
														</div>
														<div className="text-center">
															<div className="w-8 h-8 bg-green-600 rounded-lg flex items-center justify-center mx-auto mb-2">
																<PieChart className="w-4 h-4 text-white" />
															</div>
															<div className="font-medium text-gray-900">종목별 근거</div>
															<div className="text-gray-600">선택 이유 설명</div>
														</div>
														<div className="text-center">
															<div className="w-8 h-8 bg-purple-600 rounded-lg flex items-center justify-center mx-auto mb-2">
																<Brain className="w-4 h-4 text-white" />
															</div>
															<div className="font-medium text-gray-900">AI 추론 과정</div>
															<div className="text-gray-600">의사결정 단계</div>
														</div>
													</div>
												</div>
											</div>
										</CardContent>
									</Card>
								</TabsContent>
							</Tabs>
						</div>
					</div>
				</div>
			</DialogContent>
		</Dialog>
	);
}
