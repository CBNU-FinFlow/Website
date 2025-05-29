import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import HelpTooltip from "@/components/ui/HelpTooltip";
import { RiskReturnData } from "@/lib/types";

interface RiskReturnScatterProps {
	data: RiskReturnData[];
}

export default function RiskReturnScatter({ data }: RiskReturnScatterProps) {
	const getColor = (allocation: number) => {
		if (allocation > 20) return "#EF4444"; // 빨간색 (높은 비중)
		if (allocation > 10) return "#F59E0B"; // 주황색 (중간 비중)
		return "#3B82F6"; // 파란색 (낮은 비중)
	};

	const getSize = (allocation: number) => {
		return Math.max(50, allocation * 10); // 최소 50, 최대 비중에 따라 크기 조정
	};

	return (
		<div className="bg-white rounded-lg border border-gray-200 p-4">
			<div className="flex items-center space-x-2 mb-4">
				<h4 className="text-lg font-semibold text-gray-900">리스크-수익률 분포</h4>
				<HelpTooltip
					title="리스크-수익률 분포"
					description="각 종목의 예상 수익률(세로축)과 리스크(가로축)를 점으로 표현한 산점도다. 원의 크기는 포트폴리오에서의 비중을 나타내며, 오른쪽 위(높은 수익률, 높은 리스크)에 위치할수록 공격적인 투자, 왼쪽 아래에 위치할수록 보수적인 투자를 의미한다."
				/>
			</div>
			<div style={{ height: 300 }}>
				<ResponsiveContainer width="100%" height="100%">
					<ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
						<CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
						<XAxis type="number" dataKey="risk" name="리스크" tick={{ fontSize: 12 }} label={{ value: "리스크 (%)", position: "insideBottom", offset: -10, style: { fontSize: "12px" } }} />
						<YAxis type="number" dataKey="return" name="수익률" tick={{ fontSize: 12 }} label={{ value: "수익률 (%)", angle: -90, position: "insideLeft", style: { fontSize: "12px" } }} />
						<Tooltip
							formatter={(value: any, name: any, props: any) => {
								if (name === "리스크") return [`${Number(value).toFixed(2)}%`, "리스크"];
								if (name === "수익률") return [`${Number(value).toFixed(2)}%`, "수익률"];
								return [value, name];
							}}
							labelFormatter={() => ""}
							content={({ active, payload }) => {
								if (active && payload && payload.length) {
									const data = payload[0].payload;
									return (
										<div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
											<p className="font-semibold text-gray-900">{data.symbol}</p>
											<p className="text-sm text-gray-600">수익률: {data.return.toFixed(2)}%</p>
											<p className="text-sm text-gray-600">리스크: {data.risk.toFixed(2)}%</p>
											<p className="text-sm text-gray-600">비중: {data.allocation.toFixed(1)}%</p>
										</div>
									);
								}
								return null;
							}}
						/>
						<Scatter name="종목" data={data} fill="#8884d8">
							{data.map((entry, index) => (
								<Cell key={`cell-${index}`} fill={getColor(entry.allocation)} r={getSize(entry.allocation) / 10} />
							))}
						</Scatter>
					</ScatterChart>
				</ResponsiveContainer>
			</div>
			<div className="mt-3 flex items-center justify-center space-x-4 text-xs text-gray-600">
				<div className="flex items-center space-x-1">
					<div className="w-3 h-3 bg-red-500 rounded-full"></div>
					<span>높은 비중 (20%+)</span>
				</div>
				<div className="flex items-center space-x-1">
					<div className="w-3 h-3 bg-amber-500 rounded-full"></div>
					<span>중간 비중 (10-20%)</span>
				</div>
				<div className="flex items-center space-x-1">
					<div className="w-3 h-3 bg-blue-500 rounded-full"></div>
					<span>낮은 비중 (10% 미만)</span>
				</div>
			</div>
		</div>
	);
}
