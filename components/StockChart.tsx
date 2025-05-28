import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from "recharts";
import { StockPriceData, PerformanceHistory } from "@/lib/types";

interface StockChartProps {
	data: PerformanceHistory[];
	title: string;
	height?: number;
}

export default function StockChart({ data, title, height = 300 }: StockChartProps) {
	return (
		<div className="bg-white rounded-lg border border-gray-200 p-4">
			<h4 className="text-lg font-semibold text-gray-900 mb-4">{title}</h4>
			<div style={{ height }}>
				<ResponsiveContainer width="100%" height="100%">
					<AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
						<defs>
							<linearGradient id="portfolioGradient" x1="0" y1="0" x2="0" y2="1">
								<stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3} />
								<stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
							</linearGradient>
							<linearGradient id="spyGradient" x1="0" y1="0" x2="0" y2="1">
								<stop offset="5%" stopColor="#10B981" stopOpacity={0.2} />
								<stop offset="95%" stopColor="#10B981" stopOpacity={0} />
							</linearGradient>
							<linearGradient id="qqqGradient" x1="0" y1="0" x2="0" y2="1">
								<stop offset="5%" stopColor="#F59E0B" stopOpacity={0.2} />
								<stop offset="95%" stopColor="#F59E0B" stopOpacity={0} />
							</linearGradient>
						</defs>
						<CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
						<XAxis dataKey="date" tick={{ fontSize: 12 }} tickFormatter={(value) => new Date(value).toLocaleDateString("ko-KR", { month: "short", day: "numeric" })} />
						<YAxis tick={{ fontSize: 12 }} />
						<Tooltip
							formatter={(value: any, name: any) => [`${Number(value).toFixed(2)}%`, name === "portfolio" ? "AI 포트폴리오" : name.toUpperCase()]}
							labelFormatter={(value) => new Date(value).toLocaleDateString("ko-KR")}
							contentStyle={{
								backgroundColor: "#fff",
								border: "1px solid #e2e8f0",
								borderRadius: "6px",
								fontSize: "12px",
							}}
						/>
						<Area type="monotone" dataKey="portfolio" stroke="#3B82F6" strokeWidth={2} fill="url(#portfolioGradient)" />
						<Area type="monotone" dataKey="spy" stroke="#10B981" strokeWidth={1.5} fill="url(#spyGradient)" />
						<Area type="monotone" dataKey="qqq" stroke="#F59E0B" strokeWidth={1.5} fill="url(#qqqGradient)" />
					</AreaChart>
				</ResponsiveContainer>
			</div>
		</div>
	);
}
