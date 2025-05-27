"use client";

import { useState } from "react";
import { PortfolioAllocation, PerformanceMetrics, QuickMetrics } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { TrendingUp, BarChart3, Users, CheckCircle, Lock, User, AlertCircle, Calendar, Target, Brain } from "lucide-react";
import PortfolioVisualization from "@/components/PortfolioVisualization";
import XAIVisualization from "@/components/XAIVisualization";
import { XAIData } from "@/lib/types";

export default function FinFlowDemo() {
	const [investmentAmount, setInvestmentAmount] = useState("");
	const [displayAmount, setDisplayAmount] = useState(""); // 콤마가 포함된 표시용
	const [riskTolerance, setRiskTolerance] = useState("moderate");
	const [investmentHorizon, setInvestmentHorizon] = useState([252]); // 1년 = 252 거래일
	const [isAnalyzing, setIsAnalyzing] = useState(false);
	const [analysisProgress, setAnalysisProgress] = useState(0);
	const [analysisStep, setAnalysisStep] = useState("");
	const [showResults, setShowResults] = useState(false);
	const [portfolioAllocation, setPortfolioAllocation] = useState<PortfolioAllocation[]>([]);
	const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics[]>([]);
	const [quickMetrics, setQuickMetrics] = useState<QuickMetrics>({
		annualReturn: "",
		sharpeRatio: "",
		maxDrawdown: "",
		volatility: "",
	});
	const [error, setError] = useState<string>("");
	const [xaiData, setXaiData] = useState<XAIData | null>(null);
	const [isLoadingXAI, setIsLoadingXAI] = useState(false);
	const [showXAI, setShowXAI] = useState(false);
	const [xaiMethod, setXaiMethod] = useState<"fast" | "accurate">("fast");
	const [xaiProgress, setXaiProgress] = useState(0);

	// 투자 금액 포맷팅 함수
	const handleAmountChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		const value = e.target.value;
		// 숫자만 추출
		const numericValue = value.replace(/[^0-9]/g, "");

		// 원본 값 저장 (계산용)
		setInvestmentAmount(numericValue);

		// 콤마 포맷팅된 값 저장 (표시용)
		if (numericValue) {
			const formatted = Number(numericValue).toLocaleString();
			setDisplayAmount(formatted);
		} else {
			setDisplayAmount("");
		}
	};

	const getRiskLevel = (risk: string) => {
		const levels = {
			conservative: { label: "보수적", color: "text-blue-600", desc: "안정성 중심, 낮은 변동성" },
			moderate: { label: "보통", color: "text-green-600", desc: "균형잡힌 위험-수익" },
			aggressive: { label: "적극적", color: "text-red-600", desc: "고수익 추구, 높은 변동성" },
		};
		return levels[risk as keyof typeof levels];
	};

	const getHorizonLabel = (days: number) => {
		// 거래일을 월로 변환 (1개월 ≈ 21 거래일)
		const months = Math.round(days / 21);

		if (months <= 3) return `단기 (${months}개월)`;
		if (months <= 6) return `중단기 (${months}개월)`;
		if (months <= 12) return `중기 (${months}개월)`;
		if (months <= 24) return `중장기 (${months}개월)`;
		return `장기 (${months}개월)`;
	};

	const handleAnalysis = async () => {
		if (!investmentAmount || Number.parseInt(investmentAmount) <= 0) {
			setError("유효한 투자 금액을 입력해주세요.");
			return;
		}

		setIsAnalyzing(true);
		setShowResults(false); // 기존 결과 숨기기
		setShowXAI(false); // XAI 결과도 숨기기
		setXaiData(null); // XAI 데이터 초기화
		setError("");
		setAnalysisProgress(0);
		setAnalysisStep("시장 데이터를 수집하고 있습니다...");
		console.log("분석 시작:", {
			investmentAmount,
			riskTolerance,
			investmentHorizon: investmentHorizon[0],
		});

		try {
			// 단계별 분석 시뮬레이션
			const steps = [
				{ message: "시장 데이터를 수집하고 있습니다...", progress: 20, delay: 800 },
				{ message: "기술적 지표를 계산하고 있습니다...", progress: 40, delay: 1000 },
				{ message: "리스크 모델을 분석하고 있습니다...", progress: 60, delay: 1200 },
				{ message: "강화학습 모델을 추론하고 있습니다...", progress: 80, delay: 1500 },
				{ message: "포트폴리오를 최적화하고 있습니다...", progress: 95, delay: 800 },
			];

			for (const step of steps) {
				setAnalysisStep(step.message);
				setAnalysisProgress(step.progress);
				await new Promise((resolve) => setTimeout(resolve, step.delay));
			}

			const response = await fetch("http://localhost:8000/predict", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					investment_amount: Number.parseInt(investmentAmount),
					risk_tolerance: riskTolerance,
					investment_horizon: investmentHorizon[0],
				}),
			});

			if (!response.ok) {
				throw new Error("포트폴리오 예측에 실패했습니다.");
			}

			const data = await response.json();
			console.log("서버 응답 데이터:", data);

			// 서버 응답을 프론트엔드 형태로 변환
			const allocation = data.allocation.map((item: any) => ({
				stock: item.symbol,
				percentage: (item.weight * 100).toFixed(1),
				amount: Math.round(item.weight * Number.parseInt(investmentAmount)),
			}));
			console.log("변환된 allocation:", allocation);

			// 벤치마크 데이터 동적 생성 (실제 환경에서는 서버에서 받아와야 함)
			const generateBenchmarkData = () => {
				const spyBase = { totalReturn: 28.5, annualReturn: 12.3, sharpe: 0.825, sortino: 1.124, maxDrawdown: 15.2, volatility: 14.8, winRate: 56.7, profitLoss: 1.08 };
				const qqqBase = { totalReturn: 35.2, annualReturn: 15.8, sharpe: 0.892, sortino: 1.287, maxDrawdown: 21.6, volatility: 17.9, winRate: 54.2, profitLoss: 1.15 };

				// 약간의 변동성 추가 (±5% 범위)
				const addVariation = (value: number, isPercentage = true) => {
					const variation = (Math.random() - 0.5) * 0.1; // ±5% 변동
					const result = value * (1 + variation);
					return isPercentage ? result.toFixed(2) : result.toFixed(3);
				};

				return {
					spy: {
						totalReturn: `${addVariation(spyBase.totalReturn)}%`,
						annualReturn: `${addVariation(spyBase.annualReturn)}%`,
						sharpe: addVariation(spyBase.sharpe, false),
						sortino: addVariation(spyBase.sortino, false),
						maxDrawdown: `${addVariation(spyBase.maxDrawdown)}%`,
						volatility: `${addVariation(spyBase.volatility)}%`,
						winRate: `${addVariation(spyBase.winRate)}%`,
						profitLoss: addVariation(spyBase.profitLoss, false),
					},
					qqq: {
						totalReturn: `${addVariation(qqqBase.totalReturn)}%`,
						annualReturn: `${addVariation(qqqBase.annualReturn)}%`,
						sharpe: addVariation(qqqBase.sharpe, false),
						sortino: addVariation(qqqBase.sortino, false),
						maxDrawdown: `${addVariation(qqqBase.maxDrawdown)}%`,
						volatility: `${addVariation(qqqBase.volatility)}%`,
						winRate: `${addVariation(qqqBase.winRate)}%`,
						profitLoss: addVariation(qqqBase.profitLoss, false),
					},
				};
			};

			const benchmarks = generateBenchmarkData();

			const metrics = [
				{ label: "총 수익률", portfolio: `${data.metrics.total_return.toFixed(2)}%`, spy: benchmarks.spy.totalReturn, qqq: benchmarks.qqq.totalReturn },
				{ label: "연간 수익률", portfolio: `${data.metrics.annual_return.toFixed(2)}%`, spy: benchmarks.spy.annualReturn, qqq: benchmarks.qqq.annualReturn },
				{ label: "샤프 비율", portfolio: data.metrics.sharpe_ratio.toFixed(3), spy: benchmarks.spy.sharpe, qqq: benchmarks.qqq.sharpe },
				{ label: "소르티노 비율", portfolio: data.metrics.sortino_ratio.toFixed(3), spy: benchmarks.spy.sortino, qqq: benchmarks.qqq.sortino },
				{ label: "최대 낙폭", portfolio: `${data.metrics.max_drawdown.toFixed(2)}%`, spy: benchmarks.spy.maxDrawdown, qqq: benchmarks.qqq.maxDrawdown },
				{ label: "변동성", portfolio: `${data.metrics.volatility.toFixed(2)}%`, spy: benchmarks.spy.volatility, qqq: benchmarks.qqq.volatility },
				{ label: "승률", portfolio: `${data.metrics.win_rate.toFixed(1)}%`, spy: benchmarks.spy.winRate, qqq: benchmarks.qqq.winRate },
				{ label: "손익비", portfolio: data.metrics.profit_loss_ratio.toFixed(2), spy: benchmarks.spy.profitLoss, qqq: benchmarks.qqq.profitLoss },
			];

			const quick = {
				annualReturn: `${data.metrics.annual_return.toFixed(2)}%`,
				sharpeRatio: data.metrics.sharpe_ratio.toFixed(3),
				maxDrawdown: `${data.metrics.max_drawdown.toFixed(2)}%`,
				volatility: `${data.metrics.volatility.toFixed(2)}%`,
			};

			setPortfolioAllocation(allocation);
			setPerformanceMetrics(metrics);
			setQuickMetrics(quick);
			setShowResults(true);
			console.log("결과 설정 완료, showResults:", true);
		} catch (err) {
			setError(err instanceof Error ? err.message : "서버와 연결할 수 없습니다. 서버가 실행 중인지 확인해보세요.");
		} finally {
			setIsAnalyzing(false);
			setAnalysisProgress(0);
			setAnalysisStep("");
		}
	};

	// XAI 설명 가져오기 함수
	const handleXAIAnalysis = async (method: "fast" | "accurate" = xaiMethod) => {
		if (!investmentAmount) {
			setError("먼저 포트폴리오 분석을 완료해보세요.");
			return;
		}

		setIsLoadingXAI(true);
		setXaiProgress(0);
		setError("");

		// 예상 시간 계산
		const estimatedTime = method === "fast" ? "5-10초" : "30초-2분";
		console.log(`XAI 분석 시작 (${method} 모드, 예상 시간: ${estimatedTime})`);

		try {
			// 진행률 시뮬레이션 (실제 백엔드에서 WebSocket으로 받을 수도 있음)
			const progressInterval = setInterval(
				() => {
					setXaiProgress((prev) => {
						const increment = method === "fast" ? 10 : 5;
						return Math.min(prev + increment, 90);
					});
				},
				method === "fast" ? 500 : 2000
			);

			const response = await fetch("/api/explain", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					investmentAmount: Number.parseInt(investmentAmount),
					riskTolerance,
					investmentHorizon: investmentHorizon[0],
					method: method, // 계산 방식 전달
				}),
			});

			clearInterval(progressInterval);
			setXaiProgress(100);

			if (!response.ok) {
				throw new Error("XAI 분석에 실패했습니다.");
			}

			const data = await response.json();
			console.log("XAI 분석 결과:", data);

			setXaiData(data);
			setShowXAI(true);
		} catch (err) {
			setError(err instanceof Error ? err.message : "XAI 분석 중 오류가 발생했습니다.");
		} finally {
			setIsLoadingXAI(false);
			setXaiProgress(0);
		}
	};

	return (
		<div className="min-h-screen bg-gray-50">
			{/* Header */}
			<header className="bg-white border-b">
				<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
					<div className="flex justify-between items-center h-16">
						<div className="flex items-center space-x-8">
							<div className="text-xl font-bold text-blue-600">FinFlow</div>
							<nav className="hidden md:flex space-x-8">
								<a href="#" className="text-gray-900 hover:text-blue-600">
									My stocks
								</a>
								<a href="#" className="text-gray-500 hover:text-blue-600">
									Analytics
								</a>
							</nav>
						</div>
						<div className="flex items-center space-x-4">
							<Button variant="ghost" size="icon">
								<Lock className="h-4 w-4" />
							</Button>
							<Button variant="ghost" size="icon">
								<User className="h-4 w-4" />
							</Button>
						</div>
					</div>
				</div>
			</header>

			{/* Hero Section */}
			<section className="bg-white py-20">
				<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
					<div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
						<div>
							<h1 className="text-4xl font-bold text-gray-900 mb-6">AI 기반 포트폴리오 리밸런싱</h1>
							<p className="text-xl text-gray-600 mb-8">시장 상황과 재무 목표에 맞게 적응하는 강화학습 알고리즘으로 투자 전략을 최적화해드립니다.</p>

							<div className="space-y-6">
								{/* 투자 금액 */}
								<div>
									<Label htmlFor="investment" className="text-lg font-medium flex items-center">
										<Target className="w-5 h-5 mr-2" />
										투자 금액
									</Label>
									<div className="flex space-x-4 mt-2">
										<Input id="investment" type="text" placeholder="1,000,000" value={displayAmount} onChange={handleAmountChange} className="text-lg" />
										<span className="flex items-center text-lg text-gray-600">원</span>
									</div>
								</div>

								{/* 리스크 성향 */}
								<div>
									<Label className="text-lg font-medium flex items-center">
										<AlertCircle className="w-5 h-5 mr-2" />
										투자 성향
									</Label>
									<Select value={riskTolerance} onValueChange={setRiskTolerance}>
										<SelectTrigger className="w-full mt-2">
											<SelectValue />
										</SelectTrigger>
										<SelectContent>
											<SelectItem value="conservative">
												<div className="flex items-center justify-between w-full">
													<span>보수적</span>
													<span className="text-sm text-gray-500 ml-4">안정성 중심</span>
												</div>
											</SelectItem>
											<SelectItem value="moderate">
												<div className="flex items-center justify-between w-full">
													<span>보통</span>
													<span className="text-sm text-gray-500 ml-4">균형잡힌 투자</span>
												</div>
											</SelectItem>
											<SelectItem value="aggressive">
												<div className="flex items-center justify-between w-full">
													<span>적극적</span>
													<span className="text-sm text-gray-500 ml-4">고수익 추구</span>
												</div>
											</SelectItem>
										</SelectContent>
									</Select>
									<div className="mt-2">
										<Badge variant="outline" className={getRiskLevel(riskTolerance).color}>
											{getRiskLevel(riskTolerance).label}: {getRiskLevel(riskTolerance).desc}
										</Badge>
									</div>
								</div>

								{/* 투자 기간 */}
								<div>
									<Label className="text-lg font-medium flex items-center">
										<Calendar className="w-5 h-5 mr-2" />
										투자 기간
									</Label>
									<div className="mt-4 space-y-3">
										<Slider
											value={investmentHorizon}
											onValueChange={setInvestmentHorizon}
											max={756} // 3년
											min={63} // 3개월
											step={1} // 1일 단위로 부드럽게
											className="w-full"
										/>
										{/* <div className="flex justify-between text-sm text-gray-500 mt-2">
											<span>3개월</span>
											<span>1년</span>
											<span>2년</span>
											<span>3년</span>
										</div> */}
										<div className="flex items-center justify-between">
											<Badge variant="secondary">{getHorizonLabel(investmentHorizon[0])}</Badge>
											<span className="text-sm text-gray-500">{Math.round(investmentHorizon[0] / 21)}개월</span>
										</div>
									</div>
								</div>

								<Button onClick={handleAnalysis} disabled={!investmentAmount || isAnalyzing} className="w-full lg:w-auto" size="lg">
									{isAnalyzing ? "AI 분석 중입니다..." : "지금 바로 시작해보세요"}
								</Button>

								{error && (
									<div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
										<p className="text-red-600 text-sm flex items-center">
											<AlertCircle className="w-4 h-4 mr-2" />
											{error}
										</p>
									</div>
								)}
							</div>
						</div>

						{/* 오른쪽 영역 - 분석 진행 상황 또는 안내 */}
						<div className="bg-gray-100 rounded-lg p-8 flex items-center justify-center h-80">
							{isAnalyzing ? (
								<div className="text-center w-full">
									<h3 className="text-2xl font-bold text-gray-900 mb-4">AI가 시장을 분석하고 있습니다</h3>
									<p className="text-gray-600 mb-6">
										투자 성향: <Badge className="mx-1">{getRiskLevel(riskTolerance).label}</Badge>, 투자 기간: <Badge className="mx-1">{getHorizonLabel(investmentHorizon[0])}</Badge>
									</p>
									<Progress value={analysisProgress} className="w-full max-w-md mx-auto mb-4" />
									<div className="space-y-2 text-sm text-gray-500">
										<p className="font-medium text-blue-600">{analysisStep}</p>
										<p className="text-xs">예상 소요 시간: 약 5-7초</p>
									</div>
								</div>
							) : showResults ? (
								<div className="text-gray-400 text-center">
									<div className="w-24 h-24 mx-auto mb-4 bg-gray-200 rounded-full flex items-center justify-center">
										<CheckCircle className="w-12 h-12 text-green-500" />
									</div>
									<p className="text-lg font-medium text-gray-700">분석이 완료되었습니다!</p>
									<p className="text-sm mt-2">아래에서 상세 결과를 확인해보세요</p>
								</div>
							) : (
								<div className="text-gray-400 text-center">
									<div className="w-24 h-24 mx-auto mb-4 bg-gray-200 rounded-full flex items-center justify-center">
										<BarChart3 className="w-12 h-12" />
									</div>
									<p className="text-lg font-medium">포트폴리오 시각화</p>
									<p className="text-sm mt-2">분석 완료 후 차트가 표시됩니다</p>
								</div>
							)}
						</div>
					</div>
				</div>
			</section>

			{/* Results Section */}
			{showResults && (
				<section className="py-16 bg-gradient-to-br from-blue-50 to-indigo-100">
					<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
						<div className="text-center mb-12">
							<h2 className="text-4xl font-bold text-gray-900 mb-6">AI 추천 포트폴리오 결과</h2>
							<div className="flex justify-center space-x-4 mb-4">
								<Badge variant="outline" className="bg-white">
									{getRiskLevel(riskTolerance).label} 투자 성향
								</Badge>
								<Badge variant="outline" className="bg-white">
									{getHorizonLabel(investmentHorizon[0])}
								</Badge>
							</div>
							<p className="text-lg text-gray-600">
								총 투자금액: <span className="font-bold text-blue-600">{Number.parseInt(investmentAmount).toLocaleString()}원</span>
							</p>
						</div>

						{/* 시각화 차트 섹션 */}
						<div className="mb-12">
							<PortfolioVisualization portfolioAllocation={portfolioAllocation} performanceMetrics={performanceMetrics} showResults={showResults} />
						</div>

						{/* 요약 지표 카드들 */}
						<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
							<Card className="bg-white shadow-lg hover:shadow-xl transition-shadow">
								<CardContent className="p-6">
									<div className="flex items-center justify-between">
										<div>
											<p className="text-sm font-medium text-gray-600">연간 수익률</p>
											<p className="text-3xl font-bold text-green-600">{quickMetrics.annualReturn}</p>
										</div>
										<div className="p-3 bg-green-100 rounded-full">
											<TrendingUp className="h-6 w-6 text-green-600" />
										</div>
									</div>
								</CardContent>
							</Card>

							<Card className="bg-white shadow-lg hover:shadow-xl transition-shadow">
								<CardContent className="p-6">
									<div className="flex items-center justify-between">
										<div>
											<p className="text-sm font-medium text-gray-600">샤프 비율</p>
											<p className="text-3xl font-bold text-blue-600">{quickMetrics.sharpeRatio}</p>
										</div>
										<div className="p-3 bg-blue-100 rounded-full">
											<BarChart3 className="h-6 w-6 text-blue-600" />
										</div>
									</div>
								</CardContent>
							</Card>

							<Card className="bg-white shadow-lg hover:shadow-xl transition-shadow">
								<CardContent className="p-6">
									<div className="flex items-center justify-between">
										<div>
											<p className="text-sm font-medium text-gray-600">최대 낙폭</p>
											<p className="text-3xl font-bold text-red-600">{quickMetrics.maxDrawdown}</p>
										</div>
										<div className="p-3 bg-red-100 rounded-full">
											<AlertCircle className="h-6 w-6 text-red-600" />
										</div>
									</div>
								</CardContent>
							</Card>

							<Card className="bg-white shadow-lg hover:shadow-xl transition-shadow">
								<CardContent className="p-6">
									<div className="flex items-center justify-between">
										<div>
											<p className="text-sm font-medium text-gray-600">변동성</p>
											<p className="text-3xl font-bold text-purple-600">{quickMetrics.volatility}</p>
										</div>
										<div className="p-3 bg-purple-100 rounded-full">
											<Target className="h-6 w-6 text-purple-600" />
										</div>
									</div>
								</CardContent>
							</Card>
						</div>

						{/* 상세 성과 비교 테이블 */}
						<Card className="bg-white shadow-lg">
							<CardHeader>
								<CardTitle className="text-2xl">상세 성과 비교</CardTitle>
								<CardDescription className="text-lg">AI 포트폴리오 vs 벤치마크 지수 비교</CardDescription>
							</CardHeader>
							<CardContent>
								<div className="overflow-x-auto">
									<table className="w-full border-collapse">
										<thead>
											<tr className="border-b-2 border-gray-200">
												<th className="text-left py-4 px-6 font-semibold text-gray-700">지표</th>
												<th className="text-center py-4 px-6 font-semibold text-blue-600">AI 포트폴리오</th>
												<th className="text-center py-4 px-6 font-semibold text-gray-600">SPY</th>
												<th className="text-center py-4 px-6 font-semibold text-gray-600">QQQ</th>
											</tr>
										</thead>
										<tbody>
											{performanceMetrics.map((metric, index) => (
												<tr key={index} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
													<td className="py-4 px-6 font-medium text-gray-900">{metric.label}</td>
													<td className="py-4 px-6 text-center font-bold text-blue-600 text-lg">{metric.portfolio}</td>
													<td className="py-4 px-6 text-center text-gray-500">{metric.spy}</td>
													<td className="py-4 px-6 text-center text-gray-500">{metric.qqq}</td>
												</tr>
											))}
										</tbody>
									</table>
								</div>
								<p className="text-sm text-gray-500 mt-6 p-4 bg-gray-50 rounded-lg">
									<strong>참고:</strong> 과거 백테스트 기반 예상 수치이며, 실제 결과는 다를 수 있습니다. 투자 결정 시 다양한 요소를 종합적으로 고려해주세요.
								</p>
							</CardContent>
						</Card>

						{/* XAI 분석 선택 및 버튼 */}
						<div className="mt-8 bg-white rounded-xl p-6 shadow-lg border border-gray-100">
							<div className="text-center mb-6">
								<h3 className="text-xl font-bold text-gray-900 mb-2 flex items-center justify-center">
									<Brain className="w-6 h-6 mr-2 text-purple-600" />
									AI 의사결정 과정 분석
								</h3>
								<p className="text-gray-600">AI가 어떤 근거로 이 포트폴리오를 추천했는지 분석해드립니다</p>
							</div>

							{/* 계산 방식 선택 */}
							<div className="mb-6">
								<Label className="text-sm font-medium mb-3 block">계산 방식 선택</Label>
								<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
									<div
										className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${xaiMethod === "fast" ? "border-blue-500 bg-blue-50" : "border-gray-200 hover:border-gray-300"}`}
										onClick={() => setXaiMethod("fast")}
									>
										<div className="flex items-start space-x-3">
											<div className={`w-4 h-4 rounded-full border-2 mt-0.5 ${xaiMethod === "fast" ? "border-blue-500 bg-blue-500" : "border-gray-300"}`}>
												{xaiMethod === "fast" && <div className="w-2 h-2 bg-white rounded-full m-0.5" />}
											</div>
											<div>
												<h4 className="font-semibold text-gray-900">빠른 분석</h4>
												<p className="text-sm text-gray-600 mt-1">근사적 계산 방식 (5-10초)</p>
												<div className="mt-2">
													<Badge variant="secondary" className="text-xs">
														실시간 UX 최적화
													</Badge>
												</div>
											</div>
										</div>
									</div>

									<div
										className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${xaiMethod === "accurate" ? "border-purple-500 bg-purple-50" : "border-gray-200 hover:border-gray-300"}`}
										onClick={() => setXaiMethod("accurate")}
									>
										<div className="flex items-start space-x-3">
											<div className={`w-4 h-4 rounded-full border-2 mt-0.5 ${xaiMethod === "accurate" ? "border-purple-500 bg-purple-500" : "border-gray-300"}`}>
												{xaiMethod === "accurate" && <div className="w-2 h-2 bg-white rounded-full m-0.5" />}
											</div>
											<div>
												<h4 className="font-semibold text-gray-900">정확한 분석</h4>
												<p className="text-sm text-gray-600 mt-1">Integrated Gradients (30초-2분)</p>
												<div className="mt-2">
													<Badge variant="secondary" className="text-xs">
														연구/분석용 고정밀도
													</Badge>
												</div>
											</div>
										</div>
									</div>
								</div>
							</div>

							{/* 분석 버튼 및 진행률 */}
							{isLoadingXAI ? (
								<div className="text-center">
									<div className="mb-4">
										<div className="text-lg font-medium text-gray-700 mb-2">{xaiMethod === "fast" ? "빠른 분석" : "정확한 분석"} 진행 중입니다...</div>
										<Progress value={xaiProgress} className="w-full max-w-md mx-auto mb-2" />
										<p className="text-sm text-gray-500">
											{xaiProgress}% 완료 (예상 시간: {xaiMethod === "fast" ? "5-10초" : "30초-2분"})
										</p>
									</div>
									<div className="text-xs text-gray-400">{xaiMethod === "accurate" && "정확한 분석은 계산 시간이 오래 걸립니다. 잠시만 기다려주세요."}</div>
								</div>
							) : (
								<div className="text-center">
									<Button
										onClick={() => handleXAIAnalysis()}
										disabled={isLoadingXAI}
										size="lg"
										className={`${
											xaiMethod === "fast"
												? "bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700"
												: "bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700"
										} text-white border-0`}
									>
										<Brain className="w-5 h-5 mr-2" />
										{xaiMethod === "fast" ? "빠른 분석 시작하기" : "정확한 분석 시작하기"}
									</Button>

									<div className="mt-3 text-sm text-gray-500">예상 소요 시간: {xaiMethod === "fast" ? "5-10초" : "30초-2분"}</div>
								</div>
							)}

							{error && showResults && (
								<div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg max-w-md mx-auto">
									<p className="text-red-600 text-sm flex items-center justify-center">
										<AlertCircle className="w-4 h-4 mr-2" />
										{error}
									</p>
								</div>
							)}
						</div>
					</div>
				</section>
			)}

			{/* XAI 시각화 섹션 */}
			{showXAI && <XAIVisualization xaiData={xaiData} isLoading={isLoadingXAI} />}

			{/* Features Section */}
			<section className="bg-gray-100 py-16">
				<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
					<div className="text-center mb-12">
						<h2 className="text-3xl font-bold text-gray-900 mb-4">작동 방식</h2>
						<p className="text-xl text-gray-600">저희 플랫폼은 고급 강화학습 알고리즘을 사용하여 시장 상황을 지속적으로 분석하고 포트폴리오 배분을 최적화합니다.</p>
					</div>

					<div className="grid grid-cols-1 md:grid-cols-3 gap-8">
						<div className="text-center">
							<div className="bg-white rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
								<BarChart3 className="h-8 w-8 text-blue-600" />
							</div>
							<h3 className="text-xl font-bold mb-2">데이터 분석</h3>
							<p className="text-gray-600">AI가 과거 시장 데이터, 기술적 지표 및 포트폴리오 성과를 분석합니다.</p>
						</div>

						<div className="text-center">
							<div className="bg-white rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
								<TrendingUp className="h-8 w-8 text-blue-600" />
							</div>
							<h3 className="text-xl font-bold mb-2">강화학습</h3>
							<p className="text-gray-600">PPO 알고리즘이 시장 환경에 적응하며 최적의 자산 배분 전략을 학습합니다.</p>
						</div>

						<div className="text-center">
							<div className="bg-white rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
								<Users className="h-8 w-8 text-blue-600" />
							</div>
							<h3 className="text-xl font-bold mb-2">맞춤형 전략</h3>
							<p className="text-gray-600">사용자의 투자 성향, 기간, 금액을 고려해 개인화된 포트폴리오를 제안합니다.</p>
						</div>
					</div>
				</div>
			</section>

			{/* Platform Features */}
			<section className="py-16">
				<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
					<div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
						<div>
							<h2 className="text-3xl font-bold text-gray-900 mb-6">플랫폼의 주요 기능</h2>
							<p className="text-lg text-gray-600 mb-8">AI 기반 포트폴리오 리밸런서가 어떻게 투자 전략을 변화시킬 수 있는지 알아보실 수 있습니다.</p>

							<div className="space-y-4">
								<div className="flex items-center space-x-3">
									<CheckCircle className="h-5 w-5 text-green-500" />
									<span>강화학습 기반 자동화된 포트폴리오 최적화</span>
								</div>
								<div className="flex items-center space-x-3">
									<CheckCircle className="h-5 w-5 text-green-500" />
									<span>개인별 리스크 성향 및 투자 기간 고려</span>
								</div>
								<div className="flex items-center space-x-3">
									<CheckCircle className="h-5 w-5 text-green-500" />
									<span>실시간 기술적 지표 분석 및 적용</span>
								</div>
								<div className="flex items-center space-x-3">
									<CheckCircle className="h-5 w-5 text-green-500" />
									<span>벤치마크 대비 성과 비교 및 분석</span>
								</div>
							</div>
						</div>

						<div className="bg-gray-100 rounded-lg p-8">
							<div className="text-center text-gray-400">
								<TrendingUp className="h-32 w-32 mx-auto mb-4" />
								<p>성과 차트 시각화</p>
							</div>
						</div>
					</div>
				</div>
			</section>

			{/* Footer */}
			<footer className="bg-gray-900 text-white py-12">
				<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
					<div className="text-xl font-bold mb-4">FinFlow</div>
					<p className="text-gray-400">© 2024 FinFlow. 강화학습 기반 포트폴리오 최적화 플랫폼</p>
				</div>
			</footer>
		</div>
	);
}
