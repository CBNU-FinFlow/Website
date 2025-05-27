import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { PortfolioAllocation, PerformanceMetrics } from "@/lib/types";

interface PortfolioVisualizationProps {
	portfolioAllocation: PortfolioAllocation[];
	performanceMetrics: PerformanceMetrics[];
	showResults: boolean;
}

const COLORS = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4", "#84CC16", "#F97316"];

export default function PortfolioVisualization({ portfolioAllocation, performanceMetrics, showResults }: PortfolioVisualizationProps) {
	// 파이 차트용 데이터 변환
	const pieData = portfolioAllocation.map((item, index) => ({
		name: item.stock,
		value: parseFloat(item.percentage.toString()),
		amount: item.amount,
		color: COLORS[index % COLORS.length],
	}));

	// 성과 비교용 데이터 변환 (주요 지표만 선택)
	const performanceData = performanceMetrics
		.filter((metric) => ["연간 수익률", "샤프 비율", "최대 낙폭", "변동성"].includes(metric.label))
		.map((metric) => ({
			name: metric.label,
			portfolio: parseFloat(metric.portfolio.replace("%", "")),
			spy: parseFloat(metric.spy.replace("%", "")),
			qqq: parseFloat(metric.qqq.replace("%", "")),
		}));

	if (!showResults) {
		return (
			<div className="bg-gray-100 rounded-lg p-8 flex items-center justify-center h-80">
				<div className="text-gray-400 text-center">
					<div className="w-24 h-24 mx-auto mb-4 bg-gray-200 rounded-full flex items-center justify-center">
						<svg className="w-12 h-12" fill="currentColor" viewBox="0 0 20 20">
							<path
								fillRule="evenodd"
								d="M3 3a1 1 0 000 2v8a2 2 0 002 2h2.586l-1.293 1.293a1 1 0 101.414 1.414L10 15.414l2.293 2.293a1 1 0 001.414-1.414L12.414 15H15a2 2 0 002-2V5a1 1 0 100-2H3zm11.707 4.707a1 1 0 00-1.414-1.414L10 9.586 8.707 8.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
								clipRule="evenodd"
							/>
						</svg>
					</div>
					<p className="text-lg font-medium">포트폴리오 시각화</p>
					<p className="text-sm mt-2">분석 완료 후 차트가 표시됩니다</p>
				</div>
			</div>
		);
	}

	return (
		<div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
			{/* 포트폴리오 배분 파이 차트 */}
			<div className="bg-white rounded-xl p-8 shadow-lg border border-gray-100">
				<h3 className="text-2xl font-bold text-gray-900 mb-8 text-center">포트폴리오 배분</h3>
				<div className="space-y-8">
					<div className="h-80">
						<ResponsiveContainer width="100%" height="100%">
							<PieChart>
								<Pie
									data={pieData}
									cx="50%"
									cy="50%"
									labelLine={false}
									label={({ name, value }) => `${name}: ${value}%`}
									outerRadius={120}
									fill="#8884d8"
									dataKey="value"
									stroke="#fff"
									strokeWidth={2}
								>
									{pieData.map((entry, index) => (
										<Cell key={`cell-${index}`} fill={entry.color} />
									))}
								</Pie>
								<Tooltip
									formatter={(value: any, name: any, props: any) => [`${value}% (${props.payload.amount.toLocaleString()}원)`, name]}
									contentStyle={{
										backgroundColor: "#fff",
										border: "1px solid #e5e7eb",
										borderRadius: "8px",
										boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
									}}
								/>
							</PieChart>
						</ResponsiveContainer>
					</div>

					<div className="space-y-3">
						<h4 className="font-semibold text-gray-700 mb-4 text-center">배분 상세</h4>
						{pieData.map((item, index) => (
							<div key={index} className="flex items-center justify-between p-4 bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg hover:shadow-md transition-shadow">
								<div className="flex items-center space-x-3">
									<div className="w-5 h-5 rounded-full shadow-sm" style={{ backgroundColor: item.color }} />
									<span className="font-semibold text-gray-800">{item.name}</span>
								</div>
								<div className="text-right">
									<div className="font-bold text-lg text-gray-900">{item.value}%</div>
									<div className="text-sm text-gray-600">{item.amount.toLocaleString()}원</div>
								</div>
							</div>
						))}
					</div>
				</div>
			</div>

			{/* 성과 비교 바 차트 */}
			<div className="bg-white rounded-xl p-8 shadow-lg border border-gray-100">
				<h3 className="text-2xl font-bold text-gray-900 mb-8 text-center">성과 비교</h3>
				<div className="h-96">
					<ResponsiveContainer width="100%" height="100%">
						<BarChart data={performanceData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
							<CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
							<XAxis dataKey="name" tick={{ fontSize: 12 }} angle={-45} textAnchor="end" height={60} />
							<YAxis tick={{ fontSize: 12 }} />
							<Tooltip
								formatter={(value: any, name: any) => {
									const unit = name === "portfolio" || name === "spy" || name === "qqq" ? "%" : "";
									return [`${value}${unit}`, name === "portfolio" ? "AI 포트폴리오" : name.toUpperCase()];
								}}
								contentStyle={{
									backgroundColor: "#fff",
									border: "1px solid #e5e7eb",
									borderRadius: "8px",
									boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
								}}
							/>
							<Legend
								formatter={(value) => {
									if (value === "portfolio") return "AI 포트폴리오";
									return value.toUpperCase();
								}}
								wrapperStyle={{ paddingTop: "20px" }}
							/>
							<Bar dataKey="portfolio" fill="#3B82F6" name="portfolio" radius={[4, 4, 0, 0]} />
							<Bar dataKey="spy" fill="#10B981" name="spy" radius={[4, 4, 0, 0]} />
							<Bar dataKey="qqq" fill="#F59E0B" name="qqq" radius={[4, 4, 0, 0]} />
						</BarChart>
					</ResponsiveContainer>
				</div>
				<div className="mt-6 p-4 bg-blue-50 rounded-lg">
					<p className="text-sm text-blue-700 text-center">
						<strong>해석 가이드:</strong> 연간 수익률과 샤프 비율은 높을수록, 최대 낙폭과 변동성은 낮을수록 좋다.
					</p>
				</div>
			</div>
		</div>
	);
}
