"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import HelpTooltip from "@/components/ui/HelpTooltip";
import { QuickMetrics, PortfolioAllocation } from "@/lib/types";

interface PortfolioStatusProps {
	investmentAmount: string;
	quickMetrics: QuickMetrics;
	portfolioAllocation: PortfolioAllocation[];
}

export default function PortfolioStatus({ investmentAmount, quickMetrics, portfolioAllocation }: PortfolioStatusProps) {
	const formatAmount = (amount: string) => {
		return Number(amount).toLocaleString();
	};

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
	);
}
