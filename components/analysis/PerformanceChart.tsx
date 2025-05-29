"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import HelpTooltip from "@/components/ui/HelpTooltip";
import { QuickMetrics } from "@/lib/types";

interface PerformanceChartProps {
	quickMetrics: QuickMetrics;
	investmentAmount: string;
}

export default function PerformanceChart({ quickMetrics, investmentAmount }: PerformanceChartProps) {
	return (
		<Card className="xl:col-span-2 border border-gray-200 bg-white">
			<CardHeader className="pb-4">
				<div className="flex items-center justify-between">
					<div className="flex items-center space-x-2">
						<CardTitle className="text-lg font-bold text-gray-900">포트폴리오 성과 시뮬레이션</CardTitle>
						<HelpTooltip
							title="포트폴리오 성과 시뮬레이션"
							description="AI가 과거 데이터와 시장 분석을 바탕으로 예측한 포트폴리오의 1년간 수익률 추이다. 실제 수익률과 다를 수 있으며, 투자 참고용으로만 활용해야 한다. 벤치마크와의 비교를 통해 상대적 성과를 파악할 수 있다."
						/>
					</div>
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
	);
}
