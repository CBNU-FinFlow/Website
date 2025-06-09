import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Brain, Activity, Target, BarChart3, PieChart, TrendingUp, Eye, Info } from "lucide-react";
import { XAIData } from "@/lib/types";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from "recharts";
import HelpTooltip from "@/components/ui/HelpTooltip";

interface XAISectionProps {
	onXAIAnalysis: (method: "fast" | "accurate") => void;
	isLoadingXAI: boolean;
	xaiData: XAIData | null;
	xaiProgress: number;
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
	META: "#1877F2",
	NVDA: "#76B900",
	NFLX: "#E50914",
	CRM: "#00A1E0",
	ORCL: "#F80000",
	ADBE: "#FF0000",
};

export default function XAISection({ onXAIAnalysis, isLoadingXAI, xaiData, xaiProgress }: XAISectionProps) {
	if (isLoadingXAI) {
		return (
			<Card className="border border-gray-200 bg-white">
				<CardHeader>
					<CardTitle className="flex items-center space-x-2">
						<Brain className="h-5 w-5 text-blue-600" />
						<span>AI 의사결정 분석</span>
					</CardTitle>
					<CardDescription>AI가 이 포트폴리오를 선택한 이유를 자세히 알아보세요.</CardDescription>
				</CardHeader>
				<CardContent className="p-8">
					<div className="text-center space-y-6">
						<div className="w-16 h-16 mx-auto bg-purple-100 rounded-lg flex items-center justify-center">
							<Brain className="w-8 h-8 text-purple-600 animate-pulse" />
						</div>
						<div className="space-y-2">
							<h3 className="text-xl font-bold text-gray-900">AI 의사결정 분석 중</h3>
							<p className="text-gray-600">AI가 포트폴리오 구성 과정을 분석하고 있습니다..</p>
						</div>
						<div className="space-y-3 max-w-md mx-auto">
							<Progress value={xaiProgress} className="w-full h-2" />
							<p className="text-sm text-gray-500">{xaiProgress}% 완료</p>
						</div>
					</div>
				</CardContent>
			</Card>
		);
	}

	if (xaiData) {
		const featureData = xaiData.feature_importance.map((item, index) => ({
			name: `${item.asset_name}-${item.feature_name}`,
			importance: item.importance_score,
			asset: item.asset_name,
			feature: item.feature_name,
			color: FEATURE_COLORS[item.feature_name] || "#6B7280",
		}));

		const topAttentionWeights = (xaiData.attention_weights || []).sort((a, b) => b.weight - a.weight).slice(0, 10);

		return (
			<Card className="border border-gray-200 bg-white">
				<CardHeader>
					<CardTitle className="flex items-center space-x-2">
						<Brain className="h-5 w-5 text-blue-600" />
						<span>AI 의사결정 분석 결과</span>
					</CardTitle>
					<CardDescription>AI가 이 포트폴리오를 구성한 상세한 분석 결과입니다.</CardDescription>
				</CardHeader>
				<CardContent className="p-6 space-y-6">
					<div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
						<Card>
							<CardHeader>
								<CardTitle className="text-lg flex items-center space-x-2">
									<TrendingUp className="w-4 h-4 text-blue-600" />
									<span>영향도 분석</span>
									<HelpTooltip
										title="영향도 분석 (Feature Importance Analysis)"
										description="AI 모델이 포트폴리오를 구성할 때 각 기술적 지표가 얼마나 중요하게 작용했는지 보여준다. 값이 높을수록 해당 지표가 투자 결정에 큰 영향을 미쳤다는 의미다. MACD, RSI, 거래량 등 다양한 기술적 분석 도구들의 상대적 중요도를 파악할 수 있다."
									/>
								</CardTitle>
								<CardDescription className="text-sm">각 기술적 지표가 포트폴리오 결정에 미친 영향력이다.</CardDescription>
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

						<Card>
							<CardHeader>
								<CardTitle className="text-lg flex items-center space-x-2">
									<Eye className="w-4 h-4 text-green-600" />
									<span>주목도 네트워크</span>
									<HelpTooltip
										title="주목도 네트워크 (Attention Network)"
										description="AI 모델이 종목 간의 상호작용을 얼마나 중요하게 고려했는지 나타낸다. 높은 주목도는 두 종목이 서로 강하게 연관되어 포트폴리오 결정에 함께 영향을 미쳤음을 의미한다. 이를 통해 AI가 어떤 종목들을 하나의 그룹으로 인식했는지 알 수 있다."
									/>
								</CardTitle>
								<CardDescription className="text-sm">AI가 주목한 자산들 간의 관계다.</CardDescription>
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

					<Card>
						<CardHeader>
							<CardTitle className="text-lg flex items-center space-x-2">
								<Info className="w-4 h-4 text-orange-600" />
								<span>AI 설명</span>
								<HelpTooltip
									title="AI 설명 (AI Explanation)"
									description="AI 모델이 왜 이런 포트폴리오를 구성했는지에 대한 자연어 설명이다. 기술적 분석 결과, 시장 상황, 투자 성향 등을 종합하여 사람이 이해하기 쉬운 형태로 투자 논리를 설명한다. 투자 결정의 투명성과 신뢰성을 높이는 핵심 기능이다."
								/>
							</CardTitle>
							<CardDescription className="text-sm">포트폴리오 구성에 대한 AI의 상세한 설명이다.</CardDescription>
						</CardHeader>
						<CardContent>
							<div className="bg-blue-50 rounded-lg p-4">
								<pre className="text-gray-800 whitespace-pre-wrap text-sm leading-relaxed">{xaiData.explanation_text}</pre>
							</div>
							<div className="mt-4 p-3 bg-amber-50 border-l-4 border-amber-400 rounded-r-lg">
								<p className="text-xs text-amber-800">
									<strong>참고:</strong> 이 설명은 AI 모델의 내부 계산을 바탕으로 생성되었으며, 실제 시장 상황과 다를 수 있다. 투자에는 원금 손실 위험이 있으므로 신중한 판단이 필요하다.
								</p>
							</div>
						</CardContent>
					</Card>

					<div className="flex justify-center pt-4">
						<div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-md">
							<Button onClick={() => onXAIAnalysis("fast")} className="h-12 bg-blue-600 hover:bg-blue-700 text-white">
								<div className="flex items-center space-x-2">
									<Brain className="w-4 h-4" />
									<span>빠른 재분석</span>
								</div>
							</Button>
							<Button onClick={() => onXAIAnalysis("accurate")} variant="outline" className="h-12 border-blue-600 text-blue-600 hover:bg-blue-50">
								<div className="flex items-center space-x-2">
									<Brain className="w-4 h-4" />
									<span>정밀 재분석</span>
								</div>
							</Button>
						</div>
					</div>
				</CardContent>
			</Card>
		);
	}

	return (
		<Card className="border border-gray-200 bg-white">
			<CardHeader>
				<CardTitle className="flex items-center space-x-2">
					<Brain className="h-5 w-5 text-blue-600" />
					<span>AI 의사결정 분석</span>
				</CardTitle>
				<CardDescription>AI가 이 포트폴리오를 선택한 이유를 자세히 알아보세요.</CardDescription>
			</CardHeader>
			<CardContent className="p-8">
				<div className="text-center space-y-8">
					<div className="w-16 h-16 mx-auto bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
						<Brain className="w-8 h-8 text-white" />
					</div>
					<div className="space-y-3">
						<h3 className="text-2xl font-bold text-gray-900">AI 의사결정 과정 분석</h3>
						<p className="text-gray-600 max-w-2xl mx-auto leading-relaxed">
							AI가 어떤 요소들을 고려하여 이 포트폴리오를 구성했는지
							<br />
							상세한 분석을 제공합니다. 투자 결정의 투명성을 높이고
							<br />
							<span className="text-blue-600 font-medium">신뢰할 수 있는 투자 근거</span>를 확인하실 수 있습니다.
						</p>
					</div>

					<div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-2xl mx-auto">
						<div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-lg border border-blue-200 hover:shadow-lg transition-shadow">
							<div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center mx-auto mb-4">
								<Activity className="w-6 h-6 text-white" />
							</div>
							<h4 className="font-bold text-gray-900 mb-2">빠른 분석</h4>
							<p className="text-sm text-gray-600 mb-4">주요 의사결정 요소와 기본적인 설명을 제공합니다.</p>
							<Button onClick={() => onXAIAnalysis("fast")} disabled={isLoadingXAI} className="w-full bg-blue-600 hover:bg-blue-700 text-white">
								<div className="flex items-center space-x-2">
									<Brain className="w-4 h-4" />
									<span>5-10초 분석</span>
								</div>
							</Button>
						</div>

						<div className="bg-gradient-to-br from-gray-50 to-gray-100 p-6 rounded-lg border border-gray-200 hover:shadow-lg transition-shadow">
							<div className="w-12 h-12 bg-gray-700 rounded-lg flex items-center justify-center mx-auto mb-4">
								<Target className="w-6 h-6 text-white" />
							</div>
							<h4 className="font-bold text-gray-900 mb-2">정밀 분석</h4>
							<p className="text-sm text-gray-600 mb-4">상세한 특성 중요도와 종목별 근거를 분석합니다.</p>
							<Button onClick={() => onXAIAnalysis("accurate")} disabled={isLoadingXAI} variant="outline" className="w-full border-gray-300 text-gray-700 hover:bg-gray-50">
								<div className="flex items-center space-x-2">
									<Brain className="w-4 h-4" />
									<span>30초-2분 분석</span>
								</div>
							</Button>
						</div>
					</div>

					<div className="bg-gradient-to-r from-purple-50 to-blue-50 p-6 rounded-lg border border-purple-200 max-w-3xl mx-auto">
						<h4 className="font-bold text-gray-900 mb-4">분석 내용 미리보기</h4>
						<div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
							<div className="text-center">
								<div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center mx-auto mb-2">
									<BarChart3 className="w-4 h-4 text-white" />
								</div>
								<div className="font-medium text-gray-900">특성 중요도</div>
								<div className="text-gray-600">각 요소의 영향력</div>
							</div>
							<div className="text-center">
								<div className="w-8 h-8 bg-green-600 rounded-lg flex items-center justify-center mx-auto mb-2">
									<PieChart className="w-4 h-4 text-white" />
								</div>
								<div className="font-medium text-gray-900">종목별 근거</div>
								<div className="text-gray-600">선택 이유 설명</div>
							</div>
							<div className="text-center">
								<div className="w-8 h-8 bg-purple-600 rounded-lg flex items-center justify-center mx-auto mb-2">
									<Brain className="w-4 h-4 text-white" />
								</div>
								<div className="font-medium text-gray-900">AI 추론 과정</div>
								<div className="text-gray-600">의사결정 단계</div>
							</div>
						</div>
					</div>

					<div className="max-w-4xl mx-auto">
						<h4 className="font-bold text-gray-900 mb-6">AI 분석 프로세스</h4>
						<div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
							{[
								{ step: "1", title: "데이터 수집", desc: "시장 데이터, 재무 정보, 뉴스 분석", icon: "📊" },
								{ step: "2", title: "특성 추출", desc: "기술적/기본적 지표 계산", icon: "🔍" },
								{ step: "3", title: "모델 예측", desc: "강화학습 모델로 최적화", icon: "🤖" },
								{ step: "4", title: "포트폴리오 구성", desc: "리스크 조정 및 배분 결정", icon: "📈" },
							].map((process, index) => (
								<div key={index} className="text-center p-4 bg-white rounded-lg border border-gray-200">
									<div className="text-2xl mb-2">{process.icon}</div>
									<div className="font-medium text-gray-900 mb-1">단계 {process.step}</div>
									<div className="text-sm font-medium text-blue-600 mb-2">{process.title}</div>
									<div className="text-xs text-gray-600">{process.desc}</div>
								</div>
							))}
						</div>
					</div>
				</div>
			</CardContent>
		</Card>
	);
}
