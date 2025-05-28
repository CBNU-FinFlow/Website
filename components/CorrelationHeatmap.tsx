import { CorrelationData } from "@/lib/types";

interface CorrelationHeatmapProps {
	data: CorrelationData[];
	stocks: string[];
}

export default function CorrelationHeatmap({ data, stocks }: CorrelationHeatmapProps) {
	// 상관관계 매트릭스 생성
	const matrix: number[][] = [];
	for (let i = 0; i < stocks.length; i++) {
		matrix[i] = [];
		for (let j = 0; j < stocks.length; j++) {
			if (i === j) {
				matrix[i][j] = 1;
			} else {
				const correlation = data.find((d) => (d.stock1 === stocks[i] && d.stock2 === stocks[j]) || (d.stock1 === stocks[j] && d.stock2 === stocks[i]));
				matrix[i][j] = correlation ? correlation.correlation : 0;
			}
		}
	}

	const getColor = (value: number) => {
		const intensity = Math.abs(value);
		if (value > 0) {
			return `rgba(239, 68, 68, ${intensity})`; // 빨간색 (양의 상관관계)
		} else {
			return `rgba(59, 130, 246, ${intensity})`; // 파란색 (음의 상관관계)
		}
	};

	return (
		<div className="bg-white rounded-lg border border-gray-200 p-4">
			<h4 className="text-lg font-semibold text-gray-900 mb-4">종목 간 상관관계</h4>
			<div className="overflow-x-auto">
				<table className="w-full">
					<thead>
						<tr>
							<th className="w-16"></th>
							{stocks.map((stock) => (
								<th key={stock} className="text-xs font-medium text-gray-600 p-1 text-center min-w-[60px]">
									{stock}
								</th>
							))}
						</tr>
					</thead>
					<tbody>
						{stocks.map((stock, i) => (
							<tr key={stock}>
								<td className="text-xs font-medium text-gray-600 p-1 text-right pr-2">{stock}</td>
								{matrix[i].map((value, j) => (
									<td key={j} className="p-1 text-center border border-gray-100" style={{ backgroundColor: getColor(value) }}>
										<span className="text-xs font-medium text-gray-900">{value.toFixed(2)}</span>
									</td>
								))}
							</tr>
						))}
					</tbody>
				</table>
			</div>
			<div className="mt-3 flex items-center justify-center space-x-4 text-xs text-gray-600">
				<div className="flex items-center space-x-1">
					<div className="w-3 h-3 bg-red-500 rounded"></div>
					<span>양의 상관관계</span>
				</div>
				<div className="flex items-center space-x-1">
					<div className="w-3 h-3 bg-blue-500 rounded"></div>
					<span>음의 상관관계</span>
				</div>
			</div>
		</div>
	);
}
