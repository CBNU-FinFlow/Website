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
import {
	TrendingUp,
	BarChart3,
	Users,
	CheckCircle,
	Lock,
	User,
	AlertCircle,
	Calendar,
	Target,
	Brain,
	Menu,
	Bell,
	Search,
	ChevronDown,
	Activity,
	DollarSign,
	PieChart,
	TrendingDown,
} from "lucide-react";
import PortfolioVisualization from "@/components/PortfolioVisualization";
import AnalysisModal from "@/components/AnalysisModal";
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
	const [showModal, setShowModal] = useState(false); // 모달 표시 상태
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
			conservative: { label: "보수적", color: "text-blue-600", bgColor: "bg-blue-50", borderColor: "border-blue-200" },
			moderate: { label: "보통", color: "text-green-600", bgColor: "bg-green-50", borderColor: "border-green-200" },
			aggressive: { label: "적극적", color: "text-red-600", bgColor: "bg-red-50", borderColor: "border-red-200" },
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
			setShowModal(true); // 모달 열기
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
		const minDuration = method === "fast" ? 5000 : 15000; // 최소 대기 시간 (ms)
		console.log(`XAI 분석 시작 (${method} 모드, 예상 시간: ${estimatedTime})`);

		try {
			// 시작 시간 기록
			const startTime = Date.now();

			// 진행률 시뮬레이션 (실제 백엔드에서 WebSocket으로 받을 수도 있음)
			const progressInterval = setInterval(
				() => {
					setXaiProgress((prev) => {
						const increment = method === "fast" ? 8 : 3;
						return Math.min(prev + increment, 85);
					});
				},
				method === "fast" ? 600 : 1800
			);

			// 실제 API 호출
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

			if (!response.ok) {
				throw new Error("XAI 분석에 실패했습니다.");
			}

			const data = await response.json();
			console.log("XAI 분석 결과:", data);

			// 경과 시간 계산
			const elapsedTime = Date.now() - startTime;
			const remainingTime = Math.max(0, minDuration - elapsedTime);

			// 진행률을 90%로 설정하고 남은 시간 대기
			clearInterval(progressInterval);
			setXaiProgress(90);

			if (remainingTime > 0) {
				console.log(`최소 대기 시간 확보를 위해 ${remainingTime}ms 추가 대기`);

				// 남은 시간 동안 90%에서 100%로 천천히 증가
				const finalProgressInterval = setInterval(() => {
					setXaiProgress((prev) => Math.min(prev + 1, 99));
				}, remainingTime / 10);

				await new Promise((resolve) =>
					setTimeout(() => {
						clearInterval(finalProgressInterval);
						resolve(void 0);
					}, remainingTime)
				);
			}

			// 최종 완료
			setXaiProgress(100);

			// 완료 후 잠시 대기 (사용자 경험 향상)
			await new Promise((resolve) => setTimeout(resolve, 500));

			setXaiData(data);
		} catch (err) {
			setError(err instanceof Error ? err.message : "XAI 분석 중 오류가 발생했습니다.");
		} finally {
			setIsLoadingXAI(false);
			setXaiProgress(0);
		}
	};

	// 모달 닫기 핸들러
	const handleCloseModal = () => {
		setShowModal(false);
		setXaiData(null);
	};

	return (
		<div className="min-h-screen bg-gray-50">
			{/* Header - 간소화 */}
			<header className="bg-white border-b border-gray-200 sticky top-0 z-50">
				<div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
					<div className="flex justify-between items-center h-14">
						<div className="flex items-center space-x-6">
							<div className="flex items-center space-x-2">
								<div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
									<Activity className="w-4 h-4 text-white" />
								</div>
								<div className="text-lg font-bold text-gray-900">FinFlow</div>
								<Badge variant="outline" className="text-xs bg-blue-50 text-blue-600 border-blue-200">
									AI 투자
								</Badge>
							</div>
							<nav className="hidden md:flex space-x-6">
								<a href="#" className="text-gray-700 hover:text-blue-600 font-medium text-sm">
									포트폴리오
								</a>
								<a href="#" className="text-gray-500 hover:text-blue-600 font-medium text-sm">
									분석 리포트
								</a>
								<a href="#" className="text-gray-500 hover:text-blue-600 font-medium text-sm">
									투자 가이드
								</a>
							</nav>
						</div>
						<div className="flex items-center space-x-2">
							<Button variant="ghost" size="sm" className="text-gray-600 hover:text-gray-900">
								<Search className="h-4 w-4" />
							</Button>
							<Button variant="ghost" size="sm" className="text-gray-600 hover:text-gray-900 relative">
								<Bell className="h-4 w-4" />
								<span className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full"></span>
							</Button>
							<div className="w-7 h-7 bg-gray-200 rounded-full flex items-center justify-center">
								<User className="h-4 w-4 text-gray-600" />
							</div>
						</div>
					</div>
				</div>
			</header>

			{/* Hero Section - 간소화 */}
			<section className="bg-white">
				<div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
					<div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start">
						<div className="space-y-8">
							<div className="space-y-4">
								<div className="flex items-center space-x-2">
									<Badge className="bg-blue-100 text-blue-800 border-0">
										<Brain className="w-3 h-3 mr-1" />
										AI 포트폴리오
									</Badge>
									<Badge variant="outline" className="text-gray-600 border-gray-300">
										실시간 분석
									</Badge>
								</div>
								<h1 className="text-4xl font-bold text-gray-900 leading-tight">
									AI가 만드는
									<br />
									<span className="text-blue-600">스마트 투자</span>
								</h1>
								<p className="text-lg text-gray-600 leading-relaxed">
									강화학습 알고리즘이 시장 데이터를 실시간으로 분석하여
									<br />
									<span className="text-blue-700 font-medium">개인 맞춤형 포트폴리오</span>를 제안한다.
								</p>
							</div>

							<div className="space-y-6">
								{/* 투자 금액 */}
								<div className="space-y-3">
									<Label htmlFor="investment" className="text-base font-semibold text-gray-900 flex items-center">
										<DollarSign className="w-4 h-4 text-blue-600 mr-2" />
										투자 금액
									</Label>
									<div className="relative">
										<Input
											id="investment"
											type="text"
											placeholder="10,000,000"
											value={displayAmount}
											onChange={handleAmountChange}
											className="text-lg font-semibold h-12 pl-4 pr-12 border-2 border-gray-200 focus:border-blue-500 rounded-lg"
										/>
										<span className="absolute right-4 top-1/2 transform -translate-y-1/2 text-lg font-semibold text-gray-500">원</span>
									</div>
								</div>

								{/* 리스크 성향 */}
								<div className="space-y-3">
									<Label className="text-base font-semibold text-gray-900 flex items-center">
										<PieChart className="w-4 h-4 text-green-600 mr-2" />
										투자 성향
									</Label>
									<Select value={riskTolerance} onValueChange={setRiskTolerance}>
										<SelectTrigger className="w-full h-12 border-2 border-gray-200 focus:border-blue-500 rounded-lg">
											<SelectValue />
										</SelectTrigger>
										<SelectContent>
											<SelectItem value="conservative">
												<div className="flex items-center space-x-3">
													<div className="w-3 h-3 bg-blue-500 rounded-full"></div>
													<span>보수적 - 안정성 중심</span>
												</div>
											</SelectItem>
											<SelectItem value="moderate">
												<div className="flex items-center space-x-3">
													<div className="w-3 h-3 bg-green-500 rounded-full"></div>
													<span>보통 - 균형잡힌 위험-수익</span>
												</div>
											</SelectItem>
											<SelectItem value="aggressive">
												<div className="flex items-center space-x-3">
													<div className="w-3 h-3 bg-red-500 rounded-full"></div>
													<span>적극적 - 고수익 추구</span>
												</div>
											</SelectItem>
										</SelectContent>
									</Select>
									<div className={`inline-flex items-center px-3 py-2 rounded-lg text-sm font-medium bg-gray-100 text-gray-700 border border-gray-200`}>
										<div className={`w-2 h-2 rounded-full mr-2 ${riskTolerance === "conservative" ? "bg-blue-500" : riskTolerance === "moderate" ? "bg-green-500" : "bg-red-500"}`}></div>
										{getRiskLevel(riskTolerance).label} 투자 성향
									</div>
								</div>

								{/* 투자 기간 */}
								<div className="space-y-3">
									<Label className="text-base font-semibold text-gray-900 flex items-center">
										<Calendar className="w-4 h-4 text-purple-600 mr-2" />
										투자 기간
									</Label>
									<div className="bg-white p-4 rounded-lg border-2 border-gray-200">
										<Slider
											value={investmentHorizon}
											onValueChange={setInvestmentHorizon}
											max={756} // 3년
											min={63} // 3개월
											step={1}
											className="w-full"
										/>
										<div className="flex items-center justify-between mt-3">
											<Badge variant="secondary" className="bg-gray-100 text-gray-700 font-medium">
												{getHorizonLabel(investmentHorizon[0])}
											</Badge>
											<span className="text-sm text-gray-500 font-medium">{Math.round(investmentHorizon[0] / 21)}개월</span>
										</div>
									</div>
								</div>

								<Button onClick={handleAnalysis} disabled={!investmentAmount || isAnalyzing} className="w-full h-12 text-lg font-semibold bg-blue-600 hover:bg-blue-700 text-white rounded-lg">
									{isAnalyzing ? (
										<div className="flex items-center space-x-2">
											<div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
											<span>AI 분석 중...</span>
										</div>
									) : (
										<div className="flex items-center space-x-2">
											<Brain className="w-5 h-5" />
											<span>AI 포트폴리오 분석 시작</span>
										</div>
									)}
								</Button>

								{error && (
									<div className="p-4 bg-red-50 border border-red-200 rounded-lg">
										<p className="text-red-700 text-sm flex items-center font-medium">
											<AlertCircle className="w-4 h-4 mr-2" />
											{error}
										</p>
									</div>
								)}
							</div>
						</div>

						{/* 오른쪽 영역 - 간소화 */}
						<div className="bg-gray-50 rounded-lg p-8 border border-gray-200">
							{isAnalyzing ? (
								<div className="text-center space-y-6">
									<div className="w-16 h-16 mx-auto bg-blue-100 rounded-lg flex items-center justify-center">
										<Brain className="w-8 h-8 text-blue-600 animate-pulse" />
									</div>
									<div className="space-y-2">
										<h3 className="text-xl font-bold text-gray-900">AI 분석 진행 중</h3>
										<p className="text-gray-600">
											투자 성향: <span className="font-semibold text-blue-600">{getRiskLevel(riskTolerance).label}</span> · 투자 기간:{" "}
											<span className="font-semibold text-blue-600">{getHorizonLabel(investmentHorizon[0])}</span>
										</p>
									</div>
									<div className="space-y-3">
										<Progress value={analysisProgress} className="w-full h-2" />
										<div className="space-y-1">
											<p className="font-medium text-blue-600">{analysisStep}</p>
											<p className="text-sm text-gray-500">예상 소요 시간: 약 5-7초</p>
										</div>
									</div>
								</div>
							) : showResults ? (
								<div className="text-center space-y-4">
									<div className="w-16 h-16 mx-auto bg-green-100 rounded-lg flex items-center justify-center">
										<CheckCircle className="w-8 h-8 text-green-600" />
									</div>
									<div className="space-y-2">
										<h3 className="text-xl font-bold text-gray-900">분석 완료!</h3>
										<p className="text-gray-600">맞춤형 포트폴리오가 준비되었다</p>
									</div>
								</div>
							) : (
								<div className="text-center space-y-4">
									<div className="w-16 h-16 mx-auto bg-gray-200 rounded-lg flex items-center justify-center">
										<BarChart3 className="w-8 h-8 text-gray-400" />
									</div>
									<div className="space-y-2">
										<h3 className="text-xl font-bold text-gray-700">AI 포트폴리오 분석</h3>
										<p className="text-gray-500">투자 정보를 입력하고 분석을 시작하자</p>
									</div>
									<div className="grid grid-cols-2 gap-4 mt-6">
										<div className="bg-white p-4 rounded-lg border border-gray-200">
											<div className="text-2xl font-bold text-blue-600 mb-1">250+</div>
											<div className="text-sm text-gray-600">분석 종목</div>
										</div>
										<div className="bg-white p-4 rounded-lg border border-gray-200">
											<div className="text-2xl font-bold text-green-600 mb-1">98.5%</div>
											<div className="text-sm text-gray-600">정확도</div>
										</div>
									</div>
								</div>
							)}
						</div>
					</div>
				</div>
			</section>

			{/* 분석 결과 모달 */}
			<AnalysisModal
				isOpen={showModal}
				onClose={handleCloseModal}
				portfolioAllocation={portfolioAllocation}
				performanceMetrics={performanceMetrics}
				quickMetrics={quickMetrics}
				investmentAmount={investmentAmount}
				riskTolerance={riskTolerance}
				investmentHorizon={investmentHorizon}
				onXAIAnalysis={handleXAIAnalysis}
				xaiData={xaiData}
				isLoadingXAI={isLoadingXAI}
				xaiProgress={xaiProgress}
			/>

			{/* Features Section - 간소화 */}
			<section className="bg-white py-16">
				<div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
					<div className="text-center mb-12">
						<Badge className="bg-blue-100 text-blue-800 border-0 mb-4">
							<Activity className="w-3 h-3 mr-1" />
							AI 기술
						</Badge>
						<h2 className="text-3xl font-bold text-gray-900 mb-4">어떻게 작동하나?</h2>
						<p className="text-lg text-gray-600 max-w-3xl mx-auto">
							최신 강화학습 알고리즘이 시장 데이터를 실시간으로 분석하여
							<br />
							<span className="text-blue-700 font-medium">개인 맞춤형 포트폴리오 전략</span>을 제안한다.
						</p>
					</div>

					<div className="grid grid-cols-1 md:grid-cols-3 gap-8">
						<div className="text-center">
							<div className="w-16 h-16 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4">
								<BarChart3 className="h-8 w-8 text-blue-600" />
							</div>
							<h3 className="text-lg font-bold text-gray-900 mb-3">데이터 수집 & 분석</h3>
							<p className="text-gray-600">
								<span className="font-semibold text-blue-600">250개 이상</span>의 종목 데이터와 기술적 지표를
								<br />
								실시간으로 수집하고 분석한다.
							</p>
						</div>

						<div className="text-center">
							<div className="w-16 h-16 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-4">
								<Brain className="h-8 w-8 text-green-600" />
							</div>
							<h3 className="text-lg font-bold text-gray-900 mb-3">AI 학습 & 최적화</h3>
							<p className="text-gray-600">
								<span className="font-semibold text-green-600">PPO 강화학습</span> 알고리즘이 시장 환경에
								<br />
								적응하며 최적 전략을 학습한다.
							</p>
						</div>

						<div className="text-center">
							<div className="w-16 h-16 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-4">
								<Target className="h-8 w-8 text-purple-600" />
							</div>
							<h3 className="text-lg font-bold text-gray-900 mb-3">맞춤형 포트폴리오</h3>
							<p className="text-gray-600">
								개인의 <span className="font-semibold text-purple-600">투자 성향과 목표</span>에 맞는
								<br />
								최적화된 포트폴리오를 제안한다.
							</p>
						</div>
					</div>
				</div>
			</section>

			{/* Footer */}
			<footer className="bg-gray-900 text-white py-8">
				<div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
					<div className="text-lg font-bold mb-2">FinFlow</div>
					<p className="text-gray-400 text-sm">© 2025 FinFlow. 강화학습 기반 포트폴리오 최적화 플랫폼</p>
				</div>
			</footer>
		</div>
	);
}
