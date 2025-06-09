"use client";

import { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Brain, Activity, BarChart3, TrendingUp, Shield, CheckCircle } from "lucide-react";

const analysisSteps = [
	{
		id: 1,
		title: "시장 데이터 수집",
		description: "실시간 주식 데이터와 시장 지표를 수집하고 있습니다..",
		icon: Activity,
		duration: 2000,
		progress: 20,
	},
	{
		id: 2,
		title: "기술적 지표 계산",
		description: "RSI, MACD, 볼린저 밴드 등 기술적 지표를 계산하고 있습니다..",
		icon: BarChart3,
		duration: 2500,
		progress: 40,
	},
	{
		id: 3,
		title: "리스크 모델 분석",
		description: "포트폴리오 리스크와 변동성을 분석하고 있습니다..",
		icon: Shield,
		duration: 3000,
		progress: 60,
	},
	{
		id: 4,
		title: "AI 모델 추론",
		description: "강화학습 모델을 통해 최적 포트폴리오를 계산하고 있습니다..",
		icon: Brain,
		duration: 3500,
		progress: 80,
	},
	{
		id: 5,
		title: "포트폴리오 최적화",
		description: "투자 전략과 자산 배분을 최적화하고 있습니다..",
		icon: TrendingUp,
		duration: 2000,
		progress: 100,
	},
];

// 실제 로딩 화면을 보여주는 컴포넌트
function AnalysisLoadingContent() {
	const router = useRouter();
	const searchParams = useSearchParams();
	const [currentStep, setCurrentStep] = useState(0);
	const [progress, setProgress] = useState(0);
	const [isCompleted, setIsCompleted] = useState(false);

	// URL 파라미터에서 분석 데이터 가져오기
	const investmentAmount = searchParams.get("amount") || "1000000";
	const riskTolerance = searchParams.get("risk") || "5";
	const investmentHorizon = searchParams.get("horizon") || "12";

	useEffect(() => {
		let stepIndex = 0;
		let accumulatedTime = 0;

		const runAnalysis = async () => {
			for (const step of analysisSteps) {
				setCurrentStep(stepIndex);

				// 프로그레스 바 애니메이션
				const startProgress = stepIndex === 0 ? 0 : analysisSteps[stepIndex - 1].progress;
				const endProgress = step.progress;
				const duration = step.duration;
				const stepSize = (endProgress - startProgress) / (duration / 50);

				let currentProgress = startProgress;
				const progressInterval = setInterval(() => {
					currentProgress += stepSize;
					if (currentProgress >= endProgress) {
						currentProgress = endProgress;
						clearInterval(progressInterval);
					}
					setProgress(currentProgress);
				}, 50);

				await new Promise((resolve) => setTimeout(resolve, duration));
				stepIndex++;
			}

			setIsCompleted(true);

			// 2초 후 결과 페이지로 이동
			setTimeout(() => {
				router.push(`/analysis/results?amount=${investmentAmount}&risk=${riskTolerance}&horizon=${investmentHorizon}`);
			}, 2000);
		};

		runAnalysis();
	}, [router, investmentAmount, riskTolerance, investmentHorizon]);

	const currentStepData = analysisSteps[currentStep];

	return (
		<div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-4">
			<div className="max-w-2xl w-full space-y-8">
				{/* 헤더 */}
				<div className="text-center space-y-4">
					<div className="w-20 h-20 mx-auto bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center relative overflow-hidden">
						<Brain className="w-10 h-10 text-white relative z-10" />
						{/* 반짝이는 빛 효과 */}
						<div
							className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent transform -skew-x-12 -translate-x-full animate-[shimmer_2s_infinite]"
							style={{
								animation: "shimmer 2s infinite",
							}}
						/>
					</div>
					<h1 className="text-3xl font-bold text-gray-900">AI 포트폴리오 분석</h1>
					<p className="text-gray-600">투자 금액 {Number(investmentAmount).toLocaleString()}원을 분석하고 있습니다..</p>
				</div>

				{/* 커스텀 애니메이션 스타일 */}
				<style jsx>{`
					@keyframes shimmer {
						0% {
							transform: translateX(-100%) skewX(-12deg);
						}
						100% {
							transform: translateX(200%) skewX(-12deg);
						}
					}
				`}</style>

				{/* 프로그레스 바 */}
				<div className="space-y-4">
					<div className="flex justify-between items-center">
						<span className="text-sm font-medium text-gray-700">분석 진행률</span>
						<span className="text-sm font-medium text-blue-600">{Math.round(progress)}%</span>
					</div>
					<div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
						<div className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-300 ease-out" style={{ width: `${progress}%` }} />
					</div>
				</div>

				{/* 현재 단계 */}
				{!isCompleted && currentStepData && (
					<div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-100">
						<div className="flex items-center space-x-6">
							<div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
								<currentStepData.icon className="w-8 h-8 text-white" />
							</div>
							<div className="flex-1">
								<h3 className="text-xl font-bold text-gray-900 mb-2">{currentStepData.title}</h3>
								<p className="text-gray-600">{currentStepData.description}</p>
							</div>
						</div>
					</div>
				)}

				{/* 완료 상태 */}
				{isCompleted && (
					<div className="bg-white rounded-2xl p-8 shadow-lg border border-green-200">
						<div className="text-center space-y-4">
							<div className="w-16 h-16 mx-auto bg-green-500 rounded-full flex items-center justify-center">
								<CheckCircle className="w-8 h-8 text-white" />
							</div>
							<h3 className="text-xl font-bold text-green-900">분석 완료!</h3>
							<p className="text-gray-600">결과 페이지로 이동하고 있습니다.</p>
						</div>
					</div>
				)}

				{/* 단계별 표시 */}
				<div className="grid grid-cols-5 gap-4">
					{analysisSteps.map((step, index) => {
						const isActive = index === currentStep;
						const isStepCompleted = index < currentStep || isCompleted;
						const isPending = index > currentStep;

						return (
							<div key={step.id} className="text-center space-y-2">
								<div
									className={`w-12 h-12 mx-auto rounded-full flex items-center justify-center transition-all duration-300 ${
										isActive ? "bg-blue-500 text-white scale-110" : isStepCompleted ? "bg-green-500 text-white" : "bg-gray-200 text-gray-400"
									}`}
								>
									{isStepCompleted && index < currentStep ? <CheckCircle className="w-6 h-6" /> : <step.icon className="w-6 h-6" />}
								</div>
								<div className={`text-xs font-medium transition-colors duration-300 ${isActive ? "text-blue-600" : isStepCompleted ? "text-green-600" : "text-gray-400"}`}>{step.title}</div>
							</div>
						);
					})}
				</div>

				{/* 푸터 메시지 */}
				<div className="text-center text-sm text-gray-500">
					<p>분석이 완료되면 자동으로 결과 페이지로 이동합니다.</p>
				</div>
			</div>
		</div>
	);
}

export default function AnalysisLoadingPage() {
	return (
		<Suspense
			fallback={
				<div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-4">
					<div className="max-w-2xl w-full space-y-8">
						<div className="text-center space-y-4">
							<div className="w-20 h-20 mx-auto bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center">
								<Brain className="w-10 h-10 text-white" />
							</div>
							<h1 className="text-3xl font-bold text-gray-900">AI 포트폴리오 분석</h1>
							<p className="text-gray-600">분석을 준비하고 있습니다..</p>
						</div>
					</div>
				</div>
			}
		>
			<AnalysisLoadingContent />
		</Suspense>
	);
}
