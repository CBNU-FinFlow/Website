import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, Activity, Target, BarChart3, PieChart } from "lucide-react";
import { XAIData } from "@/lib/types";

interface XAISectionProps {
	onXAIAnalysis: (method: "fast" | "accurate") => void;
	isLoadingXAI: boolean;
	xaiData: XAIData | null;
	xaiProgress: number;
}

export default function XAISection({ onXAIAnalysis, isLoadingXAI, xaiData, xaiProgress }: XAISectionProps) {
	return (
		<Card className="border border-gray-200 bg-white">
			<CardHeader>
				<CardTitle className="flex items-center space-x-2">
					<Brain className="h-5 w-5 text-blue-600" />
					<span>AI 의사결정 분석</span>
				</CardTitle>
				<CardDescription>AI가 이 포트폴리오를 선택한 이유를 자세히 알아보자</CardDescription>
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
							상세한 분석을 제공한다. 투자 결정의 투명성을 높이고
							<br />
							<span className="text-blue-600 font-medium">신뢰할 수 있는 투자 근거</span>를 확인할 수 있다.
						</p>
					</div>

					<div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-2xl mx-auto">
						<div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-lg border border-blue-200 hover:shadow-lg transition-shadow">
							<div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center mx-auto mb-4">
								<Activity className="w-6 h-6 text-white" />
							</div>
							<h4 className="font-bold text-gray-900 mb-2">빠른 분석</h4>
							<p className="text-sm text-gray-600 mb-4">주요 의사결정 요소와 기본적인 설명을 제공한다</p>
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
							<p className="text-sm text-gray-600 mb-4">상세한 특성 중요도와 종목별 근거를 분석한다</p>
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

					{/* AI 분석 프로세스 설명 */}
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
