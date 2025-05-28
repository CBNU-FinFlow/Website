import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { PieChart } from "lucide-react";
import { PortfolioAllocation } from "@/lib/types";

interface PositionDetailsProps {
	portfolioAllocation: PortfolioAllocation[];
}

export default function PositionDetails({ portfolioAllocation }: PositionDetailsProps) {
	return (
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
	);
}
