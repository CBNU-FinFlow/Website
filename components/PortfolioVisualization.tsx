import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import HelpTooltip from "@/components/ui/HelpTooltip";
import { PortfolioAllocation, PerformanceMetrics, CorrelationData, RiskReturnData, PerformanceHistory, SectorAllocation } from "@/lib/types";
import StockChart from "./StockChart";
import CorrelationHeatmap from "./CorrelationHeatmap";
import RiskReturnScatter from "./RiskReturnScatter";

interface PortfolioVisualizationProps {
	portfolioAllocation: PortfolioAllocation[];
	performanceMetrics: PerformanceMetrics[];
	showResults: boolean;
}

const COLORS = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4", "#84CC16", "#F97316"];

export default function PortfolioVisualization({ portfolioAllocation, performanceMetrics, showResults }: PortfolioVisualizationProps) {
	// 파이 차트용 데이터 변환 (간소화)
	const pieData = portfolioAllocation.map((item, index) => ({
		name: item.stock,
		value: parseFloat(item.percentage.toString()),
		amount: item.amount,
		color: COLORS[index % COLORS.length],
	}));

	// 모의 데이터 생성 (실제 환경에서는 API에서 받아와야 함)
	const generateMockData = () => {
		// 성과 히스토리 데이터
		const performanceHistory: PerformanceHistory[] = [];
		const startDate = new Date();
		startDate.setMonth(startDate.getMonth() - 12);

		for (let i = 0; i < 252; i++) {
			// 1년간 거래일
			const date = new Date(startDate);
			date.setDate(date.getDate() + i);

			const portfolioReturn = Math.random() * 0.5 - 0.25 + (i * 0.05) / 252; // 연간 5% 상승 트렌드
			const spyReturn = Math.random() * 0.4 - 0.2 + (i * 0.08) / 252; // 연간 8% 상승 트렌드
			const qqqReturn = Math.random() * 0.6 - 0.3 + (i * 0.12) / 252; // 연간 12% 상승 트렌드

			performanceHistory.push({
				date: date.toISOString().split("T")[0],
				portfolio: portfolioReturn,
				spy: spyReturn,
				qqq: qqqReturn,
			});
		}

		// 상관관계 데이터
		const correlationData: CorrelationData[] = [];
		const stocks = portfolioAllocation.map((p) => p.stock);
		for (let i = 0; i < stocks.length; i++) {
			for (let j = i + 1; j < stocks.length; j++) {
				correlationData.push({
					stock1: stocks[i],
					stock2: stocks[j],
					correlation: (Math.random() - 0.5) * 1.8, // -0.9 ~ 0.9
				});
			}
		}

		// 리스크-수익률 데이터
		const riskReturnData: RiskReturnData[] = portfolioAllocation.map((item) => ({
			symbol: item.stock,
			risk: Math.random() * 25 + 5, // 5-30% 리스크
			return: Math.random() * 20 + 5, // 5-25% 수익률
			allocation: parseFloat(item.percentage.toString()),
		}));

		return { performanceHistory, correlationData, riskReturnData, stocks };
	};

	const { performanceHistory, correlationData, riskReturnData, stocks } = generateMockData();

	if (!showResults) {
		return (
			<div className="bg-gray-50 rounded-lg p-6 flex items-center justify-center h-64">
				<div className="text-gray-400 text-center">
					<div className="w-16 h-16 mx-auto mb-3 bg-gray-200 rounded-full flex items-center justify-center">
						<svg className="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
							<path
								fillRule="evenodd"
								d="M3 3a1 1 0 000 2v8a2 2 0 002 2h2.586l-1.293 1.293a1 1 0 101.414 1.414L10 15.414l2.293 2.293a1 1 0 001.414-1.414L12.414 15H15a2 2 0 002-2V5a1 1 0 100-2H3zm11.707 4.707a1 1 0 00-1.414-1.414L10 9.586 8.707 8.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
								clipRule="evenodd"
							/>
						</svg>
					</div>
					<p className="font-medium">포트폴리오 시각화</p>
					<p className="text-sm mt-1">분석 완료 후 차트가 표시됩니다</p>
				</div>
			</div>
		);
	}

	return (
		<div className="space-y-6">
			{/* 첫 번째 행: 성과 차트와 포트폴리오 배분 */}
			<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<StockChart data={performanceHistory} title="누적 수익률 비교" height={280} />

				{/* 간소화된 포트폴리오 배분 */}
				<div className="bg-white rounded-lg border border-gray-200 p-4">
					<div className="flex items-center space-x-2 mb-4">
						<h4 className="text-lg font-semibold text-gray-900">포트폴리오 배분</h4>
						<HelpTooltip
							title="포트폴리오 배분"
							description="AI가 추천한 최적 투자 비중을 원형 차트로 표현한다. 각 종목의 색상과 크기는 전체 포트폴리오에서 차지하는 비중을 나타내며, 분산 투자를 통해 리스크를 관리하면서 수익을 극대화하는 구성이다. 마우스를 올리면 상세 정보가 표시된다."
						/>
					</div>
					<div className="h-80 flex items-center justify-center group">
						<div className="relative transition-transform duration-300 group-hover:scale-110">
							<ResponsiveContainer width={300} height={300}>
								<PieChart>
									<Pie data={pieData} cx="50%" cy="50%" outerRadius={120} fill="#8884d8" dataKey="value" stroke="#fff" strokeWidth={2}>
										{pieData.map((entry, index) => (
											<Cell key={`cell-${index}`} fill={entry.color} />
										))}
									</Pie>
									<Tooltip
										formatter={(value: any, name: any, props: any) => [`${value}%`, name]}
										contentStyle={{
											backgroundColor: "#fff",
											border: "1px solid #e2e8f0",
											borderRadius: "8px",
											fontSize: "14px",
											padding: "16px",
											boxShadow: "0 10px 25px rgba(0, 0, 0, 0.1)",
										}}
										content={({ active, payload }) => {
											if (active && payload && payload.length) {
												const data = payload[0].payload;
												return (
													<div className="bg-white p-4 border border-gray-200 rounded-lg shadow-xl">
														<p className="font-bold text-lg text-gray-900 mb-2">{data.name}</p>
														<div className="space-y-2">
															<p className="text-sm text-gray-600">
																비중: <span className="font-semibold text-blue-600">{data.value}%</span>
															</p>
															<p className="text-sm text-gray-600">
																투자금액: <span className="font-semibold text-green-600">{data.amount.toLocaleString()}원</span>
															</p>
															<p className="text-sm text-gray-600">
																종목 유형: <span className="font-medium text-gray-800">{data.name === "현금" ? "안전자산" : "성장주"}</span>
															</p>
														</div>
													</div>
												);
											}
											return null;
										}}
									/>
								</PieChart>
							</ResponsiveContainer>
						</div>
					</div>
				</div>
			</div>

			{/* 두 번째 행: 리스크-수익률 분포와 상관관계 */}
			<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<RiskReturnScatter data={riskReturnData} />
				<CorrelationHeatmap data={correlationData} stocks={stocks} />
			</div>

			{/* 세 번째 행: 성과 비교 테이블 (간소화) */}
			<div className="bg-white rounded-lg border border-gray-200 p-4">
				<div className="flex items-center space-x-2 mb-4">
					<h4 className="text-lg font-semibold text-gray-900">주요 성과 지표</h4>
					<HelpTooltip
						title="주요 성과 지표"
						description="AI 포트폴리오와 주요 벤치마크들의 핵심 성과 지표를 비교한 표다. 연간 수익률, 샤프 비율, 최대 낙폭 등을 통해 위험 대비 수익률과 안정성을 평가할 수 있다. 파란색으로 표시된 값은 AI 포트폴리오의 성과다."
					/>
				</div>
				<div className="overflow-x-auto">
					<table className="w-full text-sm">
						<thead>
							<tr className="border-b border-gray-200">
								<th className="text-left py-2 font-semibold text-gray-700">지표</th>
								<th className="text-center py-2 font-semibold text-blue-600">AI 포트폴리오</th>
								<th className="text-center py-2 font-semibold text-gray-500">S&P 500</th>
								<th className="text-center py-2 font-semibold text-gray-500">NASDAQ</th>
							</tr>
						</thead>
						<tbody>
							{performanceMetrics.slice(0, 4).map((metric, index) => (
								<tr key={index} className="border-b border-gray-100">
									<td className="py-2 font-medium text-gray-900">{metric.label}</td>
									<td className="py-2 text-center font-semibold text-blue-600">{metric.portfolio}</td>
									<td className="py-2 text-center text-gray-600">{metric.spy}</td>
									<td className="py-2 text-center text-gray-600">{metric.qqq}</td>
								</tr>
							))}
						</tbody>
					</table>
				</div>
			</div>
		</div>
	);
}
