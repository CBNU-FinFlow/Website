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
import { TrendingUp, BarChart3, Users, CheckCircle, Lock, User, AlertCircle, Calendar, Target } from "lucide-react";

export default function FinFlowDemo() {
	const [investmentAmount, setInvestmentAmount] = useState("");
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

	const getRiskLevel = (risk: string) => {
		const levels = {
			conservative: { label: "보수적", color: "text-blue-600", desc: "안정성 중심, 낮은 변동성" },
			moderate: { label: "보통", color: "text-green-600", desc: "균형잡힌 위험-수익" },
			aggressive: { label: "적극적", color: "text-red-600", desc: "고수익 추구, 높은 변동성" },
		};
		return levels[risk as keyof typeof levels];
	};

	const getHorizonLabel = (days: number) => {
		if (days <= 90) return "단기 (3개월)";
		if (days <= 150) return "중단기 (6개월)";
		if (days <= 300) return "중기 (1년)";
		if (days <= 550) return "중장기 (2년)";
		return "장기 (3년+)";
	};

	const handleAnalysis = async () => {
		if (!investmentAmount || Number.parseInt(investmentAmount) <= 0) {
			setError("유효한 투자 금액을 입력해주세요.");
			return;
		}

		setIsAnalyzing(true);
		setShowResults(false); // 기존 결과 숨기기
		setError("");
		setAnalysisProgress(0);
		setAnalysisStep("시장 데이터 수집 중...");
		console.log("분석 시작:", {
			investmentAmount,
			riskTolerance,
			investmentHorizon: investmentHorizon[0],
		});

		try {
			// 단계별 분석 시뮬레이션
			const steps = [
				{ message: "시장 데이터 수집 중...", progress: 20, delay: 800 },
				{ message: "기술적 지표 계산 중...", progress: 40, delay: 1000 },
				{ message: "리스크 모델 분석 중...", progress: 60, delay: 1200 },
				{ message: "강화학습 모델 추론 중...", progress: 80, delay: 1500 },
				{ message: "포트폴리오 최적화 중...", progress: 95, delay: 800 },
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
			setError(err instanceof Error ? err.message : "서버와 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.");
		} finally {
			setIsAnalyzing(false);
			setAnalysisProgress(0);
			setAnalysisStep("");
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
							<p className="text-xl text-gray-600 mb-8">시장 상황과 재무 목표에 맞게 적응하는 강화학습 알고리즘으로 투자 전략을 최적화하세요.</p>

							<div className="space-y-6">
								{/* 투자 금액 */}
								<div>
									<Label htmlFor="investment" className="text-lg font-medium flex items-center">
										<Target className="w-5 h-5 mr-2" />
										투자 금액
									</Label>
									<div className="flex space-x-4 mt-2">
										<Input id="investment" type="number" placeholder="1000000" value={investmentAmount} onChange={(e) => setInvestmentAmount(e.target.value)} className="text-lg" />
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
									{isAnalyzing ? "AI 분석 중..." : "지금 바로 시작하기"}
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
						<div className="bg-gray-100 rounded-lg p-8 flex items-center justify-center h-80">
							<div className="text-gray-400 text-center">
								<BarChart3 className="h-24 w-24 mx-auto mb-4" />
								<p>포트폴리오 시각화</p>
							</div>
						</div>
					</div>
				</div>
			</section>

			{/* Analysis Progress */}
			{isAnalyzing && (
				<section className="bg-blue-50 py-12">
					<div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
						<h2 className="text-2xl font-bold text-gray-900 mb-4">AI가 시장을 분석하고 있습니다</h2>
						<p className="text-gray-600 mb-8">
							투자 성향: <Badge className="mx-1">{getRiskLevel(riskTolerance).label}</Badge>, 투자 기간: <Badge className="mx-1">{getHorizonLabel(investmentHorizon[0])}</Badge>
						</p>
						<Progress value={analysisProgress} className="w-full max-w-md mx-auto" />
						<div className="mt-4 space-y-2 text-sm text-gray-500">
							<p className="font-medium text-blue-600">{analysisStep}</p>
							<p className="text-xs">예상 소요 시간: 약 5-7초</p>
						</div>
					</div>
				</section>
			)}

			{/* Results Section */}
			{showResults && (
				<section className="py-16">
					<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
						<div className="text-center mb-8">
							<h2 className="text-3xl font-bold text-gray-900 mb-4">AI 추천 포트폴리오 결과</h2>
							<div className="flex justify-center space-x-4">
								<Badge variant="outline">{getRiskLevel(riskTolerance).label} 투자 성향</Badge>
								<Badge variant="outline">{getHorizonLabel(investmentHorizon[0])}</Badge>
							</div>
						</div>

						{/* Portfolio Allocation */}
						<div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
							<Card>
								<CardHeader>
									<CardTitle>추천 포트폴리오 배분</CardTitle>
									<CardDescription>총 투자금액: {Number.parseInt(investmentAmount).toLocaleString()}원</CardDescription>
								</CardHeader>
								<CardContent>
									<div className="space-y-4">
										{portfolioAllocation.map((item, index) => (
											<div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
												<div className="flex items-center space-x-3">
													<Badge variant={item.stock === "현금" ? "secondary" : "default"}>{item.stock}</Badge>
													<span className="font-medium">{item.percentage}%</span>
												</div>
												<span className="text-gray-600 font-medium">{item.amount.toLocaleString()}원</span>
											</div>
										))}
									</div>
								</CardContent>
							</Card>

							<Card>
								<CardHeader>
									<CardTitle>예상 성과 지표</CardTitle>
									<CardDescription>AI 모델 기반 예상 포트폴리오 성과</CardDescription>
								</CardHeader>
								<CardContent>
									<div className="grid grid-cols-2 gap-4">
										<div className="p-4 bg-green-50 rounded-lg">
											<p className="text-sm text-gray-600">연간 수익률</p>
											<p className="text-2xl font-bold text-green-600">{quickMetrics.annualReturn}</p>
										</div>
										<div className="p-4 bg-blue-50 rounded-lg">
											<p className="text-sm text-gray-600">샤프 비율</p>
											<p className="text-2xl font-bold text-blue-600">{quickMetrics.sharpeRatio}</p>
										</div>
										<div className="p-4 bg-red-50 rounded-lg">
											<p className="text-sm text-gray-600">최대 낙폭</p>
											<p className="text-2xl font-bold text-red-600">{quickMetrics.maxDrawdown}</p>
										</div>
										<div className="p-4 bg-purple-50 rounded-lg">
											<p className="text-sm text-gray-600">변동성</p>
											<p className="text-2xl font-bold text-purple-600">{quickMetrics.volatility}</p>
										</div>
									</div>
									<p className="text-xs text-gray-500 mt-4">*과거 백테스트 기반 예상 수치이며, 실제 결과는 다를 수 있습니다.</p>
								</CardContent>
							</Card>
						</div>

						{/* Performance Metrics Table */}
						<Card>
							<CardHeader>
								<CardTitle>상세 성과 비교</CardTitle>
								<CardDescription>AI 포트폴리오 vs 벤치마크 지수 비교</CardDescription>
							</CardHeader>
							<CardContent>
								<div className="overflow-x-auto">
									<table className="w-full border-collapse">
										<thead>
											<tr className="border-b">
												<th className="text-left py-3 px-4 font-medium">지표</th>
												<th className="text-center py-3 px-4 font-medium text-blue-600">AI 포트폴리오</th>
												<th className="text-center py-3 px-4 font-medium">SPY</th>
												<th className="text-center py-3 px-4 font-medium">QQQ</th>
											</tr>
										</thead>
										<tbody>
											{performanceMetrics.map((metric, index) => (
												<tr key={index} className="border-b hover:bg-gray-50">
													<td className="py-3 px-4 font-medium">{metric.label}</td>
													<td className="py-3 px-4 text-center font-bold text-blue-600">{metric.portfolio}</td>
													<td className="py-3 px-4 text-center text-gray-500">{metric.spy}</td>
													<td className="py-3 px-4 text-center text-gray-500">{metric.qqq}</td>
												</tr>
											))}
										</tbody>
									</table>
								</div>
							</CardContent>
						</Card>
					</div>
				</section>
			)}

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
							<p className="text-lg text-gray-600 mb-8">AI 기반 포트폴리오 리밸런서가 어떻게 투자 전략을 변화시킬 수 있는지 알아보세요.</p>

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
