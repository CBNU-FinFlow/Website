"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { PortfolioAllocation, PerformanceMetrics, QuickMetrics, XAIData } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { BarChart3, CheckCircle, AlertCircle, Calendar, Target, Brain, Activity, DollarSign, PieChart, ArrowDown } from "lucide-react";
import AnalysisModal from "@/components/AnalysisModal";
import NavBar from "@/components/NavBar";
import { createApiUrl, getDefaultFetchOptions, config } from "@/lib/config";

export default function FinFlowDemo() {
	const router = useRouter();
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
	const [xaiMethod] = useState<"fast" | "accurate">("fast");
	const [xaiProgress, setXaiProgress] = useState(0);

	// 스크롤 관련 상태
	const [showScrollButton, setShowScrollButton] = useState(true);
	const [isFeaturesSectionVisible, setIsFeaturesSectionVisible] = useState(false);

	// ref 선언
	const featuresSectionRef = useRef<HTMLElement>(null);

	// 환경 설정 디버깅 (개발 시에만)
	useEffect(() => {
		if (config.environment === "development") {
			console.log("🔧 FinFlow 환경 설정:", config.debug);
			console.log("📡 API 기본 URL:", config.api.baseUrl);
		}
	}, []);

	// 스크롤 이벤트와 인터섹션 옵저버 설정
	useEffect(() => {
		const handleScroll = () => {
			const scrollPosition = window.scrollY;
			const windowHeight = window.innerHeight;

			// 스크롤 버튼 표시/숨김 (화면 높이의 80% 이상 스크롤하면 숨김)
			setShowScrollButton(scrollPosition < windowHeight * 0.8);
		};

		// 인터섹션 옵저버 설정
		const observerOptions = {
			threshold: 0.2,
			rootMargin: "0px 0px -100px 0px",
		};

		const observer = new IntersectionObserver((entries) => {
			entries.forEach((entry) => {
				if (entry.target === featuresSectionRef.current) {
					setIsFeaturesSectionVisible(entry.isIntersecting);
				}
			});
		}, observerOptions);

		// 관찰 시작
		if (featuresSectionRef.current) {
			observer.observe(featuresSectionRef.current);
		}

		// 스크롤 이벤트 리스너 추가
		window.addEventListener("scroll", handleScroll);
		handleScroll(); // 초기 상태 설정

		return () => {
			window.removeEventListener("scroll", handleScroll);
			observer.disconnect();
		};
	}, []);

	// 스크롤 함수
	const scrollToFeatures = () => {
		featuresSectionRef.current?.scrollIntoView({
			behavior: "smooth",
			block: "start",
		});
	};

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
			conservative: {
				label: "보수적",
				color: "text-blue-600",
				bgColor: "bg-blue-50",
				borderColor: "border-blue-200",
			},
			moderate: {
				label: "보통",
				color: "text-green-600",
				bgColor: "bg-green-50",
				borderColor: "border-green-200",
			},
			aggressive: {
				label: "적극적",
				color: "text-red-600",
				bgColor: "bg-red-50",
				borderColor: "border-red-200",
			},
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

		// Railway 스타일로 로딩 페이지로 이동
		const params = new URLSearchParams({
			amount: investmentAmount,
			risk: riskTolerance,
			horizon: investmentHorizon[0].toString(),
		});

		router.push(`/analysis/loading?${params.toString()}`);
	};

	// XAI 설명 가져오기 함수
	const handleXAIAnalysis = async (method: "fast" | "accurate" = xaiMethod) => {
		if (!investmentAmount) {
			setError("먼저 포트폴리오 분석을 완료해주세요.");
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
				throw new Error("XAI 분석에 실패했습니다. 다시 시도해주세요.");
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
			setError(err instanceof Error ? err.message : "XAI 분석 중 오류가 발생했습니다. 다시 시도해주세요.");
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
		<div className="min-h-screen">
			{/* Header - NavBar 컴포넌트로 분리 */}
			<NavBar />

			{/* Hero Section - 간소화 */}
			<section className="bg-white">
				<div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
					<div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
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
									<span className="text-blue-700 font-medium">개인 맞춤형 포트폴리오</span>를 제안합니다.
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
											<span className="text-sm text-gray-500 font-medium">
												{Math.round(investmentHorizon[0] / 21)}
												개월
											</span>
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
											<span className="font-semibold text-purple-600">{getHorizonLabel(investmentHorizon[0])}</span>
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
										<p className="text-gray-600">맞춤형 포트폴리오가 준비되었습니다.</p>
									</div>
								</div>
							) : (
								<div className="text-center space-y-4">
									<div className="w-16 h-16 mx-auto bg-gray-200 rounded-lg flex items-center justify-center">
										<BarChart3 className="w-8 h-8 text-gray-400" />
									</div>
									<div className="space-y-2">
										<h3 className="text-xl font-bold text-gray-700">AI 포트폴리오 분석</h3>
										<p className="text-gray-500">투자 정보를 입력하고 분석을 시작해주세요.</p>
									</div>
									<div className="grid grid-cols-2 gap-4 mt-6">
										<div className="bg-white p-4 rounded-lg border border-gray-200">
											<div className="text-2xl font-bold text-blue-600 mb-1">10+</div>
											<div className="text-sm text-gray-600">분석 종목</div>
										</div>
										<div className="bg-white p-4 rounded-lg border border-gray-200">
											<div className="text-2xl font-bold text-green-600 mb-1">98.5%</div>
											<div className="text-sm text-gray-600">만족도</div>
										</div>
									</div>
								</div>
							)}
						</div>
					</div>
				</div>
			</section>

			{/* 스크롤 버튼 */}
			{showScrollButton && (
				<div className="flex justify-center mt-12 flex-col items-center space-y-2 text-gray-600">
					<span className="text-sm font-semibold">더 자세히 알아보기</span>
					<button onClick={scrollToFeatures} className="p-0.5 bg-white text-slate-700 hover:bg-slate-100 transition-all duration-200 cursor-pointer rounded-full">
						<div className="flex items-center justify-center w-10 h-10 rounded-full">
							<ArrowDown className="w-4 h-4 animate-bounce" />
						</div>
					</button>
				</div>
			)}

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
			<section
				ref={featuresSectionRef}
				className={`py-16 transition-all duration-1000 ease-out ${isFeaturesSectionVisible ? "opacity-100 transform translate-y-0" : "opacity-0 transform translate-y-10"}`}
			>
				<div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
					<div className="text-center mb-12">
						<Badge className="bg-blue-100 text-blue-800 border-0 mb-4">
							<Activity className="w-3 h-3 mr-1" />
							AI 기술
						</Badge>
						<h2 className="text-3xl font-bold text-gray-900 mb-4">어떻게 작동하나요?</h2>
						<p className="text-lg text-gray-600 max-w-3xl mx-auto">
							최신 강화학습 알고리즘이 시장 데이터를 실시간으로 분석하여
							<br />
							<span className="text-blue-700 font-medium">개인 맞춤형 포트폴리오 전략</span>을 제안합니다.
						</p>
					</div>

					<div className="grid grid-cols-1 md:grid-cols-3 gap-8">
						<div className={`text-center transition-all duration-700 delay-200 ease-out ${isFeaturesSectionVisible ? "opacity-100 transform translate-y-0" : "opacity-0 transform translate-y-8"}`}>
							<div className="w-16 h-16 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4">
								<BarChart3 className="h-8 w-8 text-blue-600" />
							</div>
							<h3 className="text-lg font-bold text-gray-900 mb-3">데이터 수집 & 분석</h3>
							<p className="text-gray-600">
								<span className="font-semibold text-blue-600">250개 이상</span>
								의 종목 데이터와 기술적 지표를
								<br />
								실시간으로 수집하고 분석합니다.
							</p>
						</div>

						<div className={`text-center transition-all duration-700 delay-400 ease-out ${isFeaturesSectionVisible ? "opacity-100 transform translate-y-0" : "opacity-0 transform translate-y-8"}`}>
							<div className="w-16 h-16 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-4">
								<Brain className="h-8 w-8 text-green-600" />
							</div>
							<h3 className="text-lg font-bold text-gray-900 mb-3">AI 학습 & 최적화</h3>
							<p className="text-gray-600">
								<span className="font-semibold text-green-600">PPO 강화학습</span> 알고리즘이 시장 환경에
								<br />
								적응하며 최적 전략을 학습합니다.
							</p>
						</div>

						<div className={`text-center transition-all duration-700 delay-600 ease-out ${isFeaturesSectionVisible ? "opacity-100 transform translate-y-0" : "opacity-0 transform translate-y-8"}`}>
							<div className="w-16 h-16 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-4">
								<Target className="h-8 w-8 text-purple-600" />
							</div>
							<h3 className="text-lg font-bold text-gray-900 mb-3">맞춤형 포트폴리오</h3>
							<p className="text-gray-600">
								개인의 <span className="font-semibold text-purple-600">투자 성향과 목표</span>
								에 맞는
								<br />
								최적화된 포트폴리오를 제안합니다.
							</p>
						</div>
					</div>
				</div>
			</section>

			{/* Footer */}
			<footer className="bg-gray-900 text-white py-8">
				<div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
					<div className="text-lg font-bold mb-2">FinFlow</div>
					<p className="text-gray-400 text-sm">© 2025 FinFlow. 강화학습 기반 포트폴리오 최적화 플랫폼입니다.</p>
				</div>
			</footer>
		</div>
	);
}
