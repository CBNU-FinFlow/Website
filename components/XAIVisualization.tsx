import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { XAIData, PortfolioAllocation } from "@/lib/types";
import { Brain, TrendingUp, Eye, Info } from "lucide-react";

interface XAIVisualizationProps {
	xaiData: XAIData | null;
	isLoading: boolean;
	progress: number;
	onAnalyze: (method: "fast" | "accurate") => void;
	portfolioAllocation: PortfolioAllocation[];
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

export default function XAIVisualization({ xaiData, isLoading, progress, onAnalyze, portfolioAllocation }: XAIVisualizationProps) {
	if (isLoading) {
		return (
			<div className="p-8">
				<div className="text-center space-y-6">
					<div className="w-16 h-16 mx-auto bg-purple-100 rounded-lg flex items-center justify-center">
						<Brain className="w-8 h-8 text-purple-600 animate-pulse" />
					</div>
					<div className="space-y-2">
						<h3 className="text-xl font-bold text-gray-900">AI 의사결정 분석 중</h3>
						<p className="text-gray-600">AI가 포트폴리오 구성 과정을 분석하고 있습니다</p>
					</div>
					<div className="space-y-3 max-w-md mx-auto">
						<Progress value={progress} className="w-full h-2" />
						<p className="text-sm text-gray-500">{progress}% 완료</p>
					</div>
				</div>
			</div>
		);
	}

	if (!xaiData) {
		return (
			<div className="p-8">
				<div className="text-center space-y-6">
					<div className="w-16 h-16 mx-auto bg-purple-100 rounded-lg flex items-center justify-center">
						<Brain className="w-8 h-8 text-purple-600" />
					</div>
					<div className="space-y-2">
						<h3 className="text-xl font-bold text-gray-900">AI 의사결정 분석</h3>
						<p className="text-gray-600">
							AI가 어떤 요소들을 고려하여 이 포트폴리오를 구성했는지
							<br />
							상세한 분석을 제공합니다.
						</p>
					</div>
					<div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-md mx-auto">
						<Button onClick={() => onAnalyze("fast")} className="h-12 bg-purple-600 hover:bg-purple-700 text-white">
							<div className="flex items-center space-x-2">
								<Brain className="w-4 h-4" />
								<span>빠른 분석</span>
							</div>
						</Button>
						<Button onClick={() => onAnalyze("accurate")} variant="outline" className="h-12 border-purple-600 text-purple-600 hover:bg-purple-50">
							<div className="flex items-center space-x-2">
								<Brain className="w-4 h-4" />
								<span>정밀 분석</span>
							</div>
						</Button>
					</div>
					<div className="text-xs text-gray-500 space-y-1">
						<p>• 빠른 분석: 주요 의사결정 요소와 기본적인 설명 (5-10초)</p>
						<p>• 정밀 분석: 상세한 특성 중요도와 종목별 근거 (30초-2분)</p>
					</div>
				</div>
			</div>
		);
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
		<div className="p-6 space-y-6">
			<div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
				{/* Feature Importance Chart */}
				<Card>
					<CardHeader>
						<CardTitle className="text-lg flex items-center">
							<TrendingUp className="w-4 h-4 mr-2 text-blue-600" />
							영향도 분석
						</CardTitle>
						<CardDescription className="text-sm">각 기술적 지표가 포트폴리오 결정에 미친 영향력</CardDescription>
					</CardHeader>
					<CardContent>
						<div className="h-64">
							<ResponsiveContainer width="100%" height="100%">
								<BarChart data={featureData} margin={{ top: 10, right: 20, left: 10, bottom: 60 }}>
									<CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
									<XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-45} textAnchor="end" height={60} />
									<YAxis tick={{ fontSize: 10 }} />
									<Tooltip
										formatter={(value: any, name: any, props: any) => [`${(value * 100).toFixed(1)}%`, `${props.payload.asset} - ${props.payload.feature}`]}
										contentStyle={{
											backgroundColor: "#fff",
											border: "1px solid #e5e7eb",
											borderRadius: "6px",
											fontSize: "12px",
										}}
									/>
									<Bar dataKey="importance" radius={[2, 2, 0, 0]}>
										{featureData.map((entry, index) => (
											<Cell key={`cell-${index}`} fill={entry.color} />
										))}
									</Bar>
								</BarChart>
							</ResponsiveContainer>
						</div>

						{/* 범례 */}
						<div className="mt-3 flex flex-wrap gap-1">
							{Object.entries(FEATURE_COLORS)
								.slice(0, 6)
								.map(([feature, color]) => (
									<Badge key={feature} variant="outline" className="text-xs">
										<div className="w-2 h-2 rounded-full mr-1" style={{ backgroundColor: color }} />
										{feature}
									</Badge>
								))}
						</div>
					</CardContent>
				</Card>

				{/* Attention Weights Visualization */}
				<Card>
					<CardHeader>
						<CardTitle className="text-lg flex items-center">
							<Eye className="w-4 h-4 mr-2 text-green-600" />
							주목도 네트워크
						</CardTitle>
						<CardDescription className="text-sm">AI가 주목한 자산들 간의 관계</CardDescription>
					</CardHeader>
					<CardContent>
						<div className="space-y-3 max-h-64 overflow-y-auto">
							{topAttentionWeights.slice(0, 6).map((weight, index) => (
								<div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
									<div className="flex items-center space-x-3">
										<div className="flex items-center space-x-1">
											<div className="w-3 h-3 rounded-full" style={{ backgroundColor: ASSET_COLORS[weight.from_asset] || "#6B7280" }} />
											<span className="font-medium text-gray-800 text-sm">{weight.from_asset}</span>
										</div>

										<div className="text-gray-400 text-sm">→</div>

										<div className="flex items-center space-x-1">
											<div className="w-3 h-3 rounded-full" style={{ backgroundColor: ASSET_COLORS[weight.to_asset] || "#6B7280" }} />
											<span className="font-medium text-gray-800 text-sm">{weight.to_asset}</span>
										</div>
									</div>

									<div className="flex items-center space-x-2">
										<div className="w-16 bg-gray-200 rounded-full h-1.5">
											<div className="bg-blue-600 h-1.5 rounded-full" style={{ width: `${weight.weight * 100}%` }} />
										</div>
										<span className="text-xs font-bold text-blue-600 w-10 text-right">{(weight.weight * 100).toFixed(1)}%</span>
									</div>
								</div>
							))}
						</div>
					</CardContent>
				</Card>
			</div>

			{/* AI 설명 텍스트 */}
			<Card>
				<CardHeader>
					<CardTitle className="text-lg flex items-center">
						<Info className="w-4 h-4 mr-2 text-orange-600" />
						AI 설명
					</CardTitle>
					<CardDescription className="text-sm">포트폴리오 구성에 대한 AI의 상세한 설명</CardDescription>
				</CardHeader>
				<CardContent>
					<div className="bg-blue-50 rounded-lg p-4">
						<pre className="text-gray-800 whitespace-pre-wrap text-sm leading-relaxed">{xaiData.explanation_text}</pre>
					</div>

					<div className="mt-4 p-3 bg-amber-50 border-l-4 border-amber-400 rounded-r-lg">
						<p className="text-xs text-amber-800">
							<strong>참고:</strong> 이 설명은 AI 모델의 내부 계산을 바탕으로 생성되었으며, 실제 시장 상황과 다를 수 있습니다.
						</p>
					</div>
				</CardContent>
			</Card>
		</div>
	);
}
