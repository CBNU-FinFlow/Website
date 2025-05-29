"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import HelpTooltip from "@/components/ui/HelpTooltip";
import { PortfolioAllocation } from "@/lib/types";

interface PortfolioHeatmapProps {
	portfolioAllocation: PortfolioAllocation[];
}

export default function PortfolioHeatmap({ portfolioAllocation }: PortfolioHeatmapProps) {
	return (
		<Card className="border border-gray-200 bg-white overflow-visible">
			<CardHeader>
				<div className="flex items-center space-x-2">
					<div className="w-5 h-5 bg-gradient-to-br from-blue-500 to-purple-500 rounded"></div>
					<span>포트폴리오 히트맵</span>
					<HelpTooltip
						title="포트폴리오 히트맵"
						description="각 종목의 투자 비중을 크기로, 수익률 성과를 색상으로 표현한 시각화다. 큰 사각형일수록 많은 비중을 차지하고, 초록색은 수익, 빨간색은 손실을 나타낸다. 포트폴리오의 구성과 성과를 한눈에 파악할 수 있다."
					/>
				</div>
				<CardDescription>종목별 비중 및 성과 시각화</CardDescription>
			</CardHeader>
			<CardContent className="overflow-visible">
				<div className="h-80 relative overflow-visible">
					{/* 트리맵 스타일 히트맵 - 완전히 채우기 */}
					<div className="w-full h-full bg-white rounded-lg overflow-hidden relative border border-gray-300">
						{/* 첫 번째 행 - 큰 종목들 (50% 높이) */}
						<div className="flex h-1/2">
							{portfolioAllocation.slice(0, 2).map((item, index) => {
								const performance = (Math.random() - 0.5) * 10;
								const isPositive = performance > 0;
								const widthPercent = (item.percentage / portfolioAllocation.slice(0, 2).reduce((sum, stock) => sum + stock.percentage, 0)) * 100;

								return (
									<div
										key={index}
										className={`${isPositive ? "bg-green-500" : "bg-red-500"} flex flex-col justify-center items-center text-white relative group cursor-pointer transition-all hover:brightness-110 ${
											index > 0 ? "border-l border-white" : ""
										}`}
										style={{ width: `${widthPercent}%` }}
									>
										<div className="text-center p-2 relative z-10">
											<div className="font-bold text-lg">{item.percentage}%</div>
											<div className="text-sm opacity-90">{item.stock}</div>
											<div className="text-xs opacity-75">
												{performance > 0 ? "+" : ""}
												{performance.toFixed(1)}%
											</div>
										</div>
									</div>
								);
							})}
						</div>

						{/* 두 번째 행 - 중간 종목들 (33% 높이) */}
						<div className="flex h-1/3 border-t border-white">
							{portfolioAllocation.slice(2, 5).map((item, index) => {
								const performance = (Math.random() - 0.5) * 10;
								const isPositive = performance > 0;
								const widthPercent = (item.percentage / portfolioAllocation.slice(2, 5).reduce((sum, stock) => sum + stock.percentage, 0)) * 100;

								return (
									<div
										key={index + 2}
										className={`${isPositive ? "bg-green-500" : "bg-red-500"} flex flex-col justify-center items-center text-white relative group cursor-pointer transition-all hover:brightness-110 ${
											index > 0 ? "border-l border-white" : ""
										}`}
										style={{ width: `${widthPercent}%` }}
									>
										<div className="text-center p-2 relative z-10">
											<div className="font-bold text-base">{item.percentage}%</div>
											<div className="text-xs opacity-90">{item.stock}</div>
											<div className="text-xs opacity-75">
												{performance > 0 ? "+" : ""}
												{performance.toFixed(1)}%
											</div>
										</div>
									</div>
								);
							})}
						</div>

						{/* 세 번째 행 - 작은 종목들 (17% 높이) */}
						<div className="flex h-1/6 border-t border-white">
							{portfolioAllocation.slice(5).map((item, index) => {
								const performance = (Math.random() - 0.5) * 10;
								const isPositive = performance > 0;
								const remainingStocks = portfolioAllocation.slice(5);
								const widthPercent = remainingStocks.length > 0 ? (item.percentage / remainingStocks.reduce((sum, stock) => sum + stock.percentage, 0)) * 100 : 0;

								return (
									<div
										key={index + 5}
										className={`${isPositive ? "bg-green-500" : "bg-red-500"} flex flex-col justify-center items-center text-white relative group cursor-pointer transition-all hover:brightness-110 ${
											index > 0 ? "border-l border-white" : ""
										}`}
										style={{ width: `${widthPercent}%` }}
									>
										<div className="text-center p-1 relative z-10">
											<div className="font-bold text-sm">{item.percentage}%</div>
											<div className="text-xs opacity-90 truncate">{item.stock.substring(0, 4)}</div>
										</div>
									</div>
								);
							})}
						</div>
					</div>

					{/* 모든 툴팁을 히트맵 외부에 배치 */}
					{portfolioAllocation.map((item, globalIndex) => {
						const performance = (Math.random() - 0.5) * 10;

						// 각 종목의 위치 계산
						let rowIndex, colIndex, rowHeight, colWidth;

						if (globalIndex < 2) {
							// 첫 번째 행
							rowIndex = 0;
							colIndex = globalIndex;
							rowHeight = 50; // h-1/2 = 50%
							colWidth = (item.percentage / portfolioAllocation.slice(0, 2).reduce((sum, stock) => sum + stock.percentage, 0)) * 100;
						} else if (globalIndex < 5) {
							// 두 번째 행
							rowIndex = 1;
							colIndex = globalIndex - 2;
							rowHeight = 33.33; // h-1/3 = 33.33%
							colWidth = (item.percentage / portfolioAllocation.slice(2, 5).reduce((sum, stock) => sum + stock.percentage, 0)) * 100;
						} else {
							// 세 번째 행
							rowIndex = 2;
							colIndex = globalIndex - 5;
							rowHeight = 16.67; // h-1/6 = 16.67%
							const remainingStocks = portfolioAllocation.slice(5);
							colWidth = remainingStocks.length > 0 ? (item.percentage / remainingStocks.reduce((sum, stock) => sum + stock.percentage, 0)) * 100 : 0;
						}

						// 위치 계산
						const topPercent = rowIndex === 0 ? 0 : rowIndex === 1 ? 50 : 83.33;
						const leftPercent =
							colIndex === 0
								? 0
								: globalIndex < 2
								? (portfolioAllocation[0].percentage / portfolioAllocation.slice(0, 2).reduce((sum, stock) => sum + stock.percentage, 0)) * 100
								: globalIndex < 5
								? (portfolioAllocation.slice(2, 2 + colIndex).reduce((sum, stock) => sum + stock.percentage, 0) / portfolioAllocation.slice(2, 5).reduce((sum, stock) => sum + stock.percentage, 0)) *
								  100
								: (portfolioAllocation.slice(5, 5 + colIndex).reduce((sum, stock) => sum + stock.percentage, 0) / portfolioAllocation.slice(5).reduce((sum, stock) => sum + stock.percentage, 0)) * 100;

						return (
							<div
								key={`tooltip-${globalIndex}`}
								className="absolute pointer-events-none"
								style={{
									top: `${topPercent}%`,
									left: `${leftPercent}%`,
									width: `${colWidth}%`,
									height: `${rowHeight}%`,
								}}
							>
								{/* 호버 감지 영역 */}
								<div className="w-full h-full group relative pointer-events-auto">
									{/* 툴팁 */}
									<div className="absolute -top-16 left-1/2 transform -translate-x-1/2 px-3 py-2 bg-gray-900 text-white text-sm rounded-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-[100] shadow-xl">
										<div className="font-medium">{item.stock}</div>
										<div className="text-xs text-gray-300">
											비중: {item.percentage}% | 금액: {item.amount.toLocaleString()}원
										</div>
										<div className="text-xs text-gray-300">
											일일 변동: {performance > 0 ? "+" : ""}
											{performance.toFixed(2)}%
										</div>
										<div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-gray-900"></div>
									</div>
								</div>
							</div>
						);
					})}
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
					<div className="text-xs">면적 = 비중</div>
				</div>
			</CardContent>
		</Card>
	);
}
