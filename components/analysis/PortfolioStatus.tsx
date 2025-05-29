"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import HelpTooltip from "@/components/ui/HelpTooltip";
import { QuickMetrics, PortfolioAllocation } from "@/lib/types";
import { useState, useEffect } from "react";

interface PortfolioStatusProps {
	investmentAmount: string;
	quickMetrics: QuickMetrics;
	portfolioAllocation: PortfolioAllocation[];
}

export default function PortfolioStatus({ investmentAmount, quickMetrics, portfolioAllocation }: PortfolioStatusProps) {
	const [realTimeData, setRealTimeData] = useState({
		dailyChange: 0,
		beta: 1.0,
		alpha: 0,
		loading: false,
	});

	const formatAmount = (amount: string) => {
		return Number(amount).toLocaleString();
	};

	// 실제 포트폴리오 지표 계산
	const calculatePortfolioMetrics = () => {
		// 현금 비중 계산 - 안전한 처리
		const cashAllocation = portfolioAllocation.find((item) => item.stock === "현금");
		const cashPercentage = cashAllocation?.percentage ?? 0;

		// 활성 포지션 수 (현금 제외)
		const activePositions = portfolioAllocation.filter((item) => item.stock !== "현금").length;

		// 가중평균 베타 추정 (간단한 휴리스틱)
		let weightedBeta = 0;
		let totalWeight = 0;

		portfolioAllocation.forEach((item) => {
			if (item.stock !== "현금" && item.percentage) {
				// 기술주는 베타가 높고, 안정주는 베타가 낮다고 가정
				let estimatedBeta = 1.0;
				const stock = item.stock.toUpperCase();

				if (["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "AMD"].includes(stock)) {
					estimatedBeta = 1.3; // 기술주는 베타 높음
				} else if (["JPM", "JNJ", "PG", "V"].includes(stock)) {
					estimatedBeta = 0.8; // 안정주는 베타 낮음
				}

				const weight = (item.percentage || 0) / 100;
				weightedBeta += estimatedBeta * weight;
				totalWeight += weight;
			}
		});

		// 정규화
		if (totalWeight > 0) {
			weightedBeta = weightedBeta / totalWeight;
		} else {
			weightedBeta = 1.0; // 기본값
		}

		// 알파 추정 (연간 수익률에서 시장 수익률 초과분)
		const annualReturnStr = quickMetrics?.annualReturn || "0%";
		const annualReturn = parseFloat(annualReturnStr.replace("%", "")) || 0;
		const marketReturn = 8; // S&P 500 평균 수익률 가정
		const estimatedAlpha = annualReturn - weightedBeta * marketReturn;

		return {
			cashPercentage: Number(cashPercentage) || 0,
			activePositions: Number(activePositions) || 0,
			beta: Number(weightedBeta) || 1.0,
			alpha: Number(estimatedAlpha) || 0,
		};
	};

	// 실시간 데이터 모의 생성 (실제로는 WebSocket이나 실시간 API 사용)
	useEffect(() => {
		const updateRealTimeData = () => {
			// 일일 변동률 (-3% ~ +3% 범위에서 랜덤)
			const dailyChange = (Math.random() - 0.5) * 6;

			setRealTimeData((prev) => ({
				...prev,
				dailyChange,
				loading: false,
			}));
		};

		// 초기 데이터 설정
		updateRealTimeData();

		// 30초마다 업데이트 (실제 환경에서는 더 빈번할 수 있음)
		const interval = setInterval(updateRealTimeData, 30000);

		return () => clearInterval(interval);
	}, [portfolioAllocation]);

	const metrics = calculatePortfolioMetrics();

	// 리스크 레벨 계산
	const getRiskLevel = (beta: number) => {
		if (beta < 0.8) return { level: "낮음", color: "text-green-600", position: "25%" };
		if (beta < 1.2) return { level: "중간", color: "text-orange-600", position: "50%" };
		return { level: "높음", color: "text-red-600", position: "75%" };
	};

	const riskInfo = getRiskLevel(metrics.beta);

	return (
		<Card className="border border-gray-200 bg-white">
			<CardHeader className="pb-4">
				<div className="flex items-center space-x-2">
					<div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
					<span>실시간 포트폴리오</span>
					<HelpTooltip
						title="실시간 포트폴리오"
						description="현재 구성된 포트폴리오의 실시간 상태를 보여준다. 총 자산 가치, 활성 포지션 수, 리스크 지표 등을 통해 포트폴리오의 현재 상황을 한눈에 파악할 수 있다. 베타와 알파는 시장 대비 위험도와 초과 수익을 나타낸다."
					/>
				</div>
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
							<div className={`text-sm font-medium ${realTimeData.dailyChange >= 0 ? "text-green-600" : "text-red-600"}`}>
								{realTimeData.dailyChange >= 0 ? "+" : ""}
								{realTimeData.dailyChange.toFixed(2)}% (오늘)
							</div>
						</div>
					</div>

					{/* 포지션 요약 */}
					<div className="space-y-3">
						<div className="flex justify-between items-center text-sm">
							<span className="text-gray-600">활성 포지션</span>
							<span className="font-medium text-gray-900">{metrics.activePositions}개</span>
						</div>
						<div className="flex justify-between items-center text-sm">
							<span className="text-gray-600">현금 비중</span>
							<span className="font-medium text-gray-900">{metrics.cashPercentage.toFixed(1)}%</span>
						</div>
						<div className="flex justify-between items-center text-sm">
							<span className="text-gray-600">베타</span>
							<span className="font-medium text-blue-600">{metrics.beta.toFixed(2)}</span>
						</div>
						<div className="flex justify-between items-center text-sm">
							<span className="text-gray-600">알파</span>
							<span className={`font-medium ${metrics.alpha >= 0 ? "text-green-600" : "text-red-600"}`}>
								{metrics.alpha >= 0 ? "+" : ""}
								{metrics.alpha.toFixed(1)}%
							</span>
						</div>
					</div>

					{/* 리스크 게이지 */}
					<div className="mt-4">
						<div className="flex justify-between items-center mb-2">
							<span className="text-sm text-gray-600">리스크 레벨</span>
							<span className={`text-sm font-medium ${riskInfo.color}`}>{riskInfo.level}</span>
						</div>
						<div className="w-full bg-gray-200 rounded-full h-2 relative">
							<div className="bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 h-2 rounded-full"></div>
							<div className="absolute top-0 w-3 h-3 bg-white border-2 border-gray-700 rounded-full transform -translate-x-1/2 -translate-y-0.5" style={{ left: riskInfo.position }}></div>
						</div>
						<div className="flex justify-between text-xs text-gray-500 mt-1">
							<span>낮음</span>
							<span>중간</span>
							<span>높음</span>
						</div>
					</div>
				</div>
			</CardContent>
		</Card>
	);
}
