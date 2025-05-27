import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { XAIData } from "@/lib/types";
import { Brain, TrendingUp, Eye, Info } from "lucide-react";

interface XAIVisualizationProps {
	xaiData: XAIData | null;
	isLoading: boolean;
}

const FEATURE_COLORS: { [key: string]: string } = {
	Open: "#3B82F6",
	High: "#10B981",
	Low: "#F59E0B",
	Close: "#EF4444",
	Volume: "#8B5CF6",
	MACD: "#06B6D4",
	RSI: "#84CC16",
	MA14: "#F97316",
	MA21: "#EC4899",
	MA100: "#6B7280",
};

const ASSET_COLORS: { [key: string]: string } = {
	AAPL: "#007AFF",
	MSFT: "#00A4EF",
	AMZN: "#FF9900",
	GOOGL: "#4285F4",
	AMD: "#ED1C24",
	TSLA: "#CC0000",
	JPM: "#0066CC",
	JNJ: "#D50000",
	PG: "#005CA9",
	V: "#1434CB",
};

export default function XAIVisualization({ xaiData, isLoading }: XAIVisualizationProps) {
	if (isLoading) {
		return (
			<div className="bg-white rounded-xl p-8 shadow-lg border border-gray-100">
				<div className="text-center">
					<div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
					<p className="text-gray-600">AI 의사결정 과정을 분석하고 있습니다...</p>
				</div>
			</div>
		);
	}

	if (!xaiData) {
		return null;
	}

	// Feature Importance 차트용 데이터 변환
	const featureData = xaiData.feature_importance.map((item, index) => ({
		name: `${item.asset_name}-${item.feature_name}`,
		importance: item.importance_score,
		asset: item.asset_name,
		feature: item.feature_name,
		color: FEATURE_COLORS[item.feature_name] || "#6B7280",
	}));

	// Attention Weights 네트워크 시각화용 데이터
	const topAttentionWeights = xaiData.attention_weights.sort((a, b) => b.weight - a.weight).slice(0, 10); // 상위 10개만

	return (
		<section className="py-12 bg-gradient-to-br from-purple-50 to-blue-50">
			<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
				<div className="text-center mb-8">
					<h2 className="text-3xl font-bold text-gray-900 mb-4 flex items-center justify-center">
						<Brain className="w-8 h-8 mr-3 text-purple-600" />
						AI 의사결정 분석
					</h2>
					<p className="text-lg text-gray-600">강화학습 모델이 어떤 요소를 고려해 포트폴리오를 구성했는지 분석 결과입니다</p>
				</div>

				<div className="grid grid-cols-1 xl:grid-cols-2 gap-8 mb-8">
					{/* Feature Importance Chart */}
					<Card className="bg-white shadow-lg border border-gray-100">
						<CardHeader>
							<CardTitle className="text-xl flex items-center">
								<TrendingUp className="w-5 h-5 mr-2 text-blue-600" />
								영향도 분석
							</CardTitle>
							<CardDescription>각 기술적 지표가 포트폴리오 결정에 미친 영향력을 보여줍니다</CardDescription>
						</CardHeader>
						<CardContent>
							<div className="h-80">
								<ResponsiveContainer width="100%" height="100%">
									<BarChart data={featureData} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
										<CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
										<XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-45} textAnchor="end" height={80} />
										<YAxis tick={{ fontSize: 12 }} label={{ value: "중요도 점수", angle: -90, position: "insideLeft" }} />
										<Tooltip
											formatter={(value: any, name: any, props: any) => [`${(value * 100).toFixed(1)}%`, `${props.payload.asset} - ${props.payload.feature}`]}
											contentStyle={{
												backgroundColor: "#fff",
												border: "1px solid #e5e7eb",
												borderRadius: "8px",
												boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
											}}
										/>
										<Bar dataKey="importance" radius={[4, 4, 0, 0]}>
											{featureData.map((entry, index) => (
												<Cell key={`cell-${index}`} fill={entry.color} />
											))}
										</Bar>
									</BarChart>
								</ResponsiveContainer>
							</div>

							{/* 범례 */}
							<div className="mt-4 flex flex-wrap gap-2">
								{Object.entries(FEATURE_COLORS).map(([feature, color]) => (
									<Badge key={feature} variant="outline" className="text-xs">
										<div className="w-3 h-3 rounded-full mr-1" style={{ backgroundColor: color }} />
										{feature}
									</Badge>
								))}
							</div>
						</CardContent>
					</Card>

					{/* Attention Weights Visualization */}
					<Card className="bg-white shadow-lg border border-gray-100">
						<CardHeader>
							<CardTitle className="text-xl flex items-center">
								<Eye className="w-5 h-5 mr-2 text-green-600" />
								주목도 네트워크
							</CardTitle>
							<CardDescription>AI가 어떤 자산들 간의 관계에 주목했는지 보여줍니다</CardDescription>
						</CardHeader>
						<CardContent>
							<div className="space-y-4">
								{topAttentionWeights.map((weight, index) => (
									<div key={index} className="flex items-center justify-between p-4 bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg hover:shadow-md transition-shadow">
										<div className="flex items-center space-x-4">
											<div className="flex items-center space-x-2">
												<div className="w-4 h-4 rounded-full shadow-sm" style={{ backgroundColor: ASSET_COLORS[weight.from_asset] || "#6B7280" }} />
												<span className="font-semibold text-gray-800">{weight.from_asset}</span>
											</div>

											<div className="text-gray-400">→</div>

											<div className="flex items-center space-x-2">
												<div className="w-4 h-4 rounded-full shadow-sm" style={{ backgroundColor: ASSET_COLORS[weight.to_asset] || "#6B7280" }} />
												<span className="font-semibold text-gray-800">{weight.to_asset}</span>
											</div>
										</div>

										<div className="flex items-center space-x-3">
											<div className="w-20 bg-gray-200 rounded-full h-2">
												<div className="bg-blue-600 h-2 rounded-full transition-all duration-300" style={{ width: `${weight.weight * 100}%` }} />
											</div>
											<span className="text-sm font-bold text-blue-600 w-12 text-right">{(weight.weight * 100).toFixed(1)}%</span>
										</div>
									</div>
								))}
							</div>
						</CardContent>
					</Card>
				</div>

				{/* AI 설명 텍스트 */}
				<Card className="bg-white shadow-lg border border-gray-100">
					<CardHeader>
						<CardTitle className="text-xl flex items-center">
							<Info className="w-5 h-5 mr-2 text-orange-600" />
							AI 설명
						</CardTitle>
						<CardDescription>포트폴리오 구성에 대한 AI의 상세한 설명입니다</CardDescription>
					</CardHeader>
					<CardContent>
						<div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6">
							<pre className="text-gray-800 whitespace-pre-wrap font-medium leading-relaxed">{xaiData.explanation_text}</pre>
						</div>

						<div className="mt-6 p-4 bg-yellow-50 border-l-4 border-yellow-400 rounded-r-lg">
							<p className="text-sm text-yellow-800">
								<strong>참고:</strong> 이 설명은 AI 모델의 내부 계산을 바탕으로 생성되었으며, 실제 시장 상황과 다를 수 있습니다. 투자 결정시 참고 자료로만 활용해주세요.
							</p>
						</div>
					</CardContent>
				</Card>
			</div>
		</section>
	);
}
