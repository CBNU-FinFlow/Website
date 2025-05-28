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
import {
	X,
	Brain,
	TrendingUp,
	DollarSign,
	PieChart,
	Calendar,
	ArrowLeft,
	BarChart3,
	Activity,
	Target,
	AlertTriangle,
	CheckCircle,
	Info,
	Maximize,
	Minimize,
	Globe,
	Shield,
	TrendingDown,
} from "lucide-react";

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
							{/* 실시간 시장 상황 헤더 */}
							<div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-6 text-white">
								<div className="flex items-center justify-between mb-4">
									<div className="flex items-center space-x-3">
										<div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
										<span className="text-sm font-medium">실시간 시장 분석</span>
										<span className="text-xs bg-white/20 px-2 py-1 rounded">LIVE</span>
									</div>
									<div className="text-sm opacity-90">
										{new Date().toLocaleString("ko-KR", {
											month: "long",
											day: "numeric",
											hour: "2-digit",
											minute: "2-digit",
										})}
									</div>
								</div>
								<div className="grid grid-cols-4 gap-4">
									<div className="text-center">
										<div className="text-2xl font-bold">S&P 500</div>
										<div className="text-sm opacity-90">4,567.23</div>
										<div className="text-xs text-green-300">+0.85%</div>
									</div>
									<div className="text-center">
										<div className="text-2xl font-bold">NASDAQ</div>
										<div className="text-sm opacity-90">14,234.56</div>
										<div className="text-xs text-green-300">+1.23%</div>
									</div>
									<div className="text-center">
										<div className="text-2xl font-bold">VIX</div>
										<div className="text-sm opacity-90">18.45</div>
										<div className="text-xs text-red-300">+2.1%</div>
									</div>
									<div className="text-center">
										<div className="text-2xl font-bold">USD/KRW</div>
										<div className="text-sm opacity-90">1,345.67</div>
										<div className="text-xs text-red-300">-0.12%</div>
									</div>
								</div>
							</div>

							{/* 포트폴리오 성과 대시보드 */}
							<div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
								{/* 메인 성과 차트 */}
								<Card className="xl:col-span-2 border border-gray-200 bg-white">
									<CardHeader className="pb-4">
										<div className="flex items-center justify-between">
											<CardTitle className="text-lg font-bold text-gray-900">포트폴리오 성과 시뮬레이션</CardTitle>
											<div className="flex items-center space-x-2">
												<Badge className="bg-green-100 text-green-700 border-0">+{quickMetrics.annualReturn}</Badge>
												<div className="flex items-center space-x-1 text-xs text-gray-500">
													<div className="w-2 h-2 bg-green-500 rounded-full"></div>
													<span>실시간</span>
												</div>
											</div>
										</div>
										<CardDescription className="text-gray-600">1년간 예상 수익률 추이 및 벤치마크 비교</CardDescription>
									</CardHeader>
									<CardContent className="pt-0">
										{/* 모의 차트 영역 */}
										<div className="h-56 bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg p-4 relative overflow-hidden">
											{/* 차트 배경 그리드 */}
											<div className="absolute inset-0 opacity-20">
												{[...Array(12)].map((_, i) => (
													<div key={i} className="absolute border-l border-gray-300" style={{ left: `${(i + 1) * 8.33}%`, height: "100%" }}></div>
												))}
												{[...Array(6)].map((_, i) => (
													<div key={i} className="absolute border-t border-gray-300 w-full" style={{ top: `${(i + 1) * 16.67}%` }}></div>
												))}
											</div>

											{/* 모의 차트 라인 */}
											<svg className="w-full h-full absolute inset-0" viewBox="0 0 400 200">
												{/* 포트폴리오 라인 (상승 추세) */}
												<path d="M 20 160 Q 80 140 120 120 T 200 100 T 280 80 T 360 60" stroke="#3B82F6" strokeWidth="3" fill="none" className="drop-shadow-sm" />
												{/* 벤치마크 라인 (완만한 상승) */}
												<path d="M 20 160 Q 80 150 120 140 T 200 130 T 280 120 T 360 110" stroke="#6B7280" strokeWidth="2" fill="none" strokeDasharray="5,5" />
												{/* 포트폴리오 영역 채우기 */}
												<path d="M 20 160 Q 80 140 120 120 T 200 100 T 280 80 T 360 60 L 360 180 L 20 180 Z" fill="url(#portfolioGradient)" opacity="0.3" />
												<defs>
													<linearGradient id="portfolioGradient" x1="0%" y1="0%" x2="0%" y2="100%">
														<stop offset="0%" stopColor="#3B82F6" />
														<stop offset="100%" stopColor="#3B82F6" stopOpacity="0" />
													</linearGradient>
												</defs>
											</svg>

											{/* 차트 레이블 */}
											<div className="absolute bottom-2 left-4 flex items-center space-x-4 text-xs">
												<div className="flex items-center space-x-1">
													<div className="w-3 h-0.5 bg-blue-500"></div>
													<span className="text-gray-600">내 포트폴리오</span>
												</div>
												<div className="flex items-center space-x-1">
													<div className="w-3 h-0.5 bg-gray-500 border-dashed border-t"></div>
													<span className="text-gray-600">S&P 500</span>
												</div>
											</div>

											{/* 현재 값 표시 */}
											<div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm rounded-lg p-3 shadow-sm">
												<div className="text-lg font-bold text-gray-900">{(Number(investmentAmount) * (1 + parseFloat(quickMetrics.annualReturn.replace("%", "")) / 100)).toLocaleString()}원</div>
												<div className="text-sm text-green-600">+{(Number(investmentAmount) * (parseFloat(quickMetrics.annualReturn.replace("%", "")) / 100)).toLocaleString()}원</div>
											</div>
										</div>

										{/* 차트 하단 메트릭 */}
										<div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-4">
											<div className="text-center p-3 bg-gray-50 rounded-lg">
												<div className="text-lg font-bold text-blue-600">{quickMetrics.annualReturn}</div>
												<div className="text-xs text-gray-600">연간 수익률</div>
											</div>
											<div className="text-center p-3 bg-gray-50 rounded-lg">
												<div className="text-lg font-bold text-green-600">{quickMetrics.sharpeRatio}</div>
												<div className="text-xs text-gray-600">샤프 비율</div>
											</div>
											<div className="text-center p-3 bg-gray-50 rounded-lg">
												<div className="text-lg font-bold text-red-600">{quickMetrics.maxDrawdown}</div>
												<div className="text-xs text-gray-600">최대 낙폭</div>
											</div>
											<div className="text-center p-3 bg-gray-50 rounded-lg">
												<div className="text-lg font-bold text-purple-600">{quickMetrics.volatility}</div>
												<div className="text-xs text-gray-600">변동성</div>
											</div>
										</div>
									</CardContent>
								</Card>

								{/* 실시간 포트폴리오 상태 */}
								<Card className="border border-gray-200 bg-white">
									<CardHeader className="pb-4">
										<CardTitle className="flex items-center space-x-2">
											<div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
											<span>실시간 포트폴리오</span>
										</CardTitle>
										<CardDescription>현재 포지션 및 실시간 손익</CardDescription>
									</CardHeader>
									<CardContent className="pt-0">
										<div className="space-y-4">
											{/* 총 자산 */}
											<div className="bg-gradient-to-r from-blue-50 to-purple-50 p-4 rounded-lg border border-blue-100">
												<div className="text-sm text-gray-600 mb-1">총 자산 가치</div>
												<div className="text-2xl font-bold text-gray-900">{formatAmount(investmentAmount)}원</div>
												<div className="flex items-center space-x-2 mt-2">
													<div className="text-sm text-green-600 font-medium">+{quickMetrics.annualReturn} (1년 예상)</div>
													<div className="text-xs text-gray-500">vs 어제</div>
												</div>
											</div>

											{/* 포지션 요약 */}
											<div className="space-y-3">
												<div className="flex justify-between items-center text-sm">
													<span className="text-gray-600">활성 포지션</span>
													<span className="font-medium text-gray-900">{portfolioAllocation.length}개</span>
												</div>
												<div className="flex justify-between items-center text-sm">
													<span className="text-gray-600">현금 비중</span>
													<span className="font-medium text-gray-900">5.2%</span>
												</div>
												<div className="flex justify-between items-center text-sm">
													<span className="text-gray-600">베타</span>
													<span className="font-medium text-blue-600">1.12</span>
												</div>
												<div className="flex justify-between items-center text-sm">
													<span className="text-gray-600">알파</span>
													<span className="font-medium text-green-600">+2.3%</span>
												</div>
											</div>

											{/* 리스크 게이지 */}
											<div className="mt-4">
												<div className="flex justify-between items-center mb-2">
													<span className="text-sm text-gray-600">리스크 레벨</span>
													<span className="text-sm font-medium text-orange-600">중간</span>
												</div>
												<div className="w-full bg-gray-200 rounded-full h-2">
													<div className="bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 h-2 rounded-full relative">
														<div className="absolute top-0 left-1/2 w-3 h-3 bg-white border-2 border-orange-500 rounded-full transform -translate-x-1/2 -translate-y-0.5"></div>
													</div>
												</div>
											</div>
										</div>
									</CardContent>
								</Card>
							</div>

							{/* 포트폴리오 히트맵 및 상세 배분 */}
							<div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
								{/* 포트폴리오 히트맵 */}
								<Card className="border border-gray-200 bg-white">
									<CardHeader>
										<CardTitle className="flex items-center space-x-2">
											<div className="w-5 h-5 bg-gradient-to-br from-blue-500 to-purple-500 rounded"></div>
											<span>포트폴리오 히트맵</span>
										</CardTitle>
										<CardDescription>종목별 비중 및 성과 시각화</CardDescription>
									</CardHeader>
									<CardContent>
										<div className="h-80">
											{/* 개선된 히트맵 */}
											<div className="grid grid-cols-3 gap-3 h-full">
												{portfolioAllocation.map((item, index) => {
													const performance = (Math.random() - 0.5) * 10; // 모의 성과 데이터
													const isPositive = performance > 0;
													const intensity = Math.abs(performance) / 10;

													// 비중에 따른 크기 계산 (더 세밀하게)
													const sizeClass =
														item.percentage > 25 ? "col-span-2 row-span-2" : item.percentage > 15 ? "col-span-2 row-span-1" : item.percentage > 10 ? "col-span-1 row-span-2" : "col-span-1 row-span-1";

													return (
														<div
															key={index}
															className={`${
																isPositive ? "bg-green-500" : "bg-red-500"
															} ${sizeClass} rounded-lg p-3 flex flex-col justify-between text-white relative overflow-hidden transition-all hover:scale-105 cursor-pointer`}
															style={{
																opacity: 0.7 + intensity * 0.3,
																minHeight: "60px",
															}}
														>
															<div className="font-medium text-sm truncate">{item.stock}</div>
															<div className="space-y-1">
																<div className="font-bold text-lg">{item.percentage}%</div>
																<div className="text-xs opacity-90">
																	{performance > 0 ? "+" : ""}
																	{performance.toFixed(1)}%
																</div>
															</div>
															{/* 배경 패턴 */}
															<div className="absolute inset-0 opacity-10">
																<div className="w-full h-full bg-gradient-to-br from-white to-transparent"></div>
															</div>
															{/* 호버 효과 */}
															<div className="absolute inset-0 bg-white opacity-0 hover:opacity-10 transition-opacity"></div>
														</div>
													);
												})}
											</div>
										</div>
										<div className="mt-4 flex items-center justify-between text-xs text-gray-500">
											<div className="flex items-center space-x-2">
												<div className="w-3 h-3 bg-green-500 rounded"></div>
												<span>상승</span>
											</div>
											<div className="flex items-center space-x-2">
												<div className="w-3 h-3 bg-red-500 rounded"></div>
												<span>하락</span>
											</div>
											<div className="text-xs">크기 = 비중</div>
										</div>
									</CardContent>
								</Card>

								{/* 상세 포지션 리스트 */}
								<Card className="border border-gray-200 bg-white">
									<CardHeader>
										<CardTitle className="flex items-center space-x-2">
											<PieChart className="h-5 w-5 text-blue-600" />
											<span>포지션 상세</span>
										</CardTitle>
										<CardDescription>종목별 투자 현황 및 실시간 손익</CardDescription>
									</CardHeader>
									<CardContent>
										<div className="space-y-3 max-h-80 overflow-y-auto">
											{portfolioAllocation.map((item, index) => {
												const dailyChange = (Math.random() - 0.5) * 6; // 모의 일일 변동
												const changeColor = dailyChange > 0 ? "text-green-600" : "text-red-600";
												const bgColor = dailyChange > 0 ? "bg-green-50" : "bg-red-50";

												return (
													<div key={index} className={`flex items-center justify-between p-4 ${bgColor} rounded-lg hover:shadow-md transition-all cursor-pointer`}>
														<div className="flex items-center space-x-3">
															<div className="w-5 h-5 rounded-full flex-shrink-0" style={{ backgroundColor: `hsl(${index * 45}, 70%, 50%)` }}></div>
															<div>
																<div className="font-medium text-gray-900">{item.stock}</div>
																<div className="text-xs text-gray-500">기술주 • {item.percentage}% 비중</div>
															</div>
														</div>
														<div className="text-right">
															<div className="font-bold text-gray-900">{item.amount.toLocaleString()}원</div>
															<div className={`text-sm font-medium ${changeColor}`}>
																{dailyChange > 0 ? "+" : ""}
																{dailyChange.toFixed(2)}%
															</div>
														</div>
													</div>
												);
											})}
										</div>

										{/* 포지션 요약 */}
										<div className="mt-4 pt-4 border-t border-gray-200">
											<div className="grid grid-cols-2 gap-4 text-sm">
												<div className="text-center p-3 bg-blue-50 rounded-lg">
													<div className="font-bold text-blue-600">{portfolioAllocation.length}</div>
													<div className="text-xs text-gray-600">총 종목 수</div>
												</div>
												<div className="text-center p-3 bg-green-50 rounded-lg">
													<div className="font-bold text-green-600">{portfolioAllocation.reduce((sum, item) => sum + item.amount, 0).toLocaleString()}원</div>
													<div className="text-xs text-gray-600">총 투자금액</div>
												</div>
											</div>
										</div>
									</CardContent>
								</Card>
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
											<div className="text-sm text-gray-600">정보 비율</div>
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
											<div className="text-sm text-gray-600">트레이너 비율</div>
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
											<div className="text-sm text-gray-600">상관계수</div>
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
											<div className="text-sm text-gray-600">VaR (1일)</div>
											<div className="text-xl font-bold text-gray-900">-2.3%</div>
											<div className="text-xs text-red-600">-{(Number(investmentAmount) * 0.023).toLocaleString()}원</div>
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
																<span className="font-medium text-gray-900">{getHorizonLabel(investmentHorizon[0])}</span>
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
															현재 포트폴리오는 <span className="font-medium">중간 수준의 리스크</span>를 가지고 있다. 기술주 집중도가 높아 시장 변동성에 민감하지만, 우수한 유동성과 분산 투자로
															리스크가 적절히 관리되고 있다. 정기적인 리밸런싱을 통해 리스크를 최적화할 수 있다.
														</p>
													</div>
												</div>
											</div>
										</CardContent>
									</Card>
								</TabsContent>

								{/* 상세 분석 */}
								<TabsContent value="analysis" className="space-y-6 mt-6">
									<PortfolioVisualization portfolioAllocation={portfolioAllocation} performanceMetrics={performanceMetrics} showResults={true} />

									{/* 추가 분석 차트들 */}
									<div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
										{/* 수익률 분포 */}
										<Card className="border border-gray-200 bg-white">
											<CardHeader>
												<CardTitle className="flex items-center space-x-2">
													<BarChart3 className="h-5 w-5 text-blue-600" />
													<span>수익률 분포</span>
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
											</CardTitle>
											<CardDescription>주요 지수 및 ETF와의 성과 비교</CardDescription>
										</CardHeader>
										<CardContent>
											<div className="overflow-x-auto">
												<table className="w-full text-sm">
													<thead>
														<tr className="border-b border-gray-200">
															<th className="text-left py-3 px-4 font-medium text-gray-900">지표</th>
															<th className="text-center py-3 px-4 font-medium text-blue-600">내 포트폴리오</th>
															<th className="text-center py-3 px-4 font-medium text-gray-600">S&P 500</th>
															<th className="text-center py-3 px-4 font-medium text-gray-600">NASDAQ</th>
															<th className="text-center py-3 px-4 font-medium text-gray-600">QQQ ETF</th>
															<th className="text-center py-3 px-4 font-medium text-gray-600">KOSPI</th>
														</tr>
													</thead>
													<tbody className="divide-y divide-gray-100">
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
												<div className="w-16 h-16 mx-auto bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
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
													<div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-lg border border-blue-200 hover:shadow-lg transition-shadow">
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

													<div className="bg-gradient-to-br from-gray-50 to-gray-100 p-6 rounded-lg border border-gray-200 hover:shadow-lg transition-shadow">
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

												<div className="bg-gradient-to-r from-purple-50 to-blue-50 p-6 rounded-lg border border-purple-200 max-w-3xl mx-auto">
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

												{/* AI 분석 프로세스 설명 */}
												<div className="max-w-4xl mx-auto">
													<h4 className="font-bold text-gray-900 mb-6">AI 분석 프로세스</h4>
													<div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
														{[
															{ step: "1", title: "데이터 수집", desc: "시장 데이터, 재무 정보, 뉴스 분석", icon: "📊" },
															{ step: "2", title: "특성 추출", desc: "기술적/기본적 지표 계산", icon: "🔍" },
															{ step: "3", title: "모델 예측", desc: "강화학습 모델로 최적화", icon: "🤖" },
															{ step: "4", title: "포트폴리오 구성", desc: "리스크 조정 및 배분 결정", icon: "📈" },
														].map((process, index) => (
															<div key={index} className="text-center p-4 bg-white rounded-lg border border-gray-200">
																<div className="text-2xl mb-2">{process.icon}</div>
																<div className="font-medium text-gray-900 mb-1">단계 {process.step}</div>
																<div className="text-sm font-medium text-blue-600 mb-2">{process.title}</div>
																<div className="text-xs text-gray-600">{process.desc}</div>
															</div>
														))}
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
