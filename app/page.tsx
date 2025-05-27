"use client";

import { useState } from "react";
import { PortfolioAllocation, PerformanceMetrics, QuickMetrics } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { TrendingUp, BarChart3, Users, CheckCircle, Lock, User } from "lucide-react";

export default function FinFlowDemo() {
	const [investmentAmount, setInvestmentAmount] = useState("");
	const [isAnalyzing, setIsAnalyzing] = useState(false);
	const [showResults, setShowResults] = useState(false);
	const [portfolioAllocation, setPortfolioAllocation] = useState<PortfolioAllocation[]>([]);
	const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics[]>([]);
	const [quickMetrics, setQuickMetrics] = useState<QuickMetrics>({
		annualReturn: "",
		sharpeRatio: "",
		maxDrawdown: "",
		volatility: "",
	});
	const [error, setError] = useState<string>("");

	const handleAnalysis = async () => {
		if (!investmentAmount) return;

		setIsAnalyzing(true);
		setError("");

		try {
			const response = await fetch("/api/portfolio", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					investmentAmount: Number.parseInt(investmentAmount),
				}),
			});

			if (!response.ok) {
				throw new Error("포트폴리오 예측에 실패했습니다.");
			}

			const data = await response.json();

			setPortfolioAllocation(data.portfolioAllocation);
			setPerformanceMetrics(data.performanceMetrics);
			setQuickMetrics(data.quickMetrics);
			setShowResults(true);
		} catch (err) {
			setError(err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다.");
		} finally {
			setIsAnalyzing(false);
		}
	};

	return (
		<div className="min-h-screen bg-gray-50">
			{/* Header */}
			<header className="bg-white border-b">
				<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
					<div className="flex justify-between items-center h-16">
						<div className="flex items-center space-x-8">
							<div className="text-xl font-bold text-blue-600">FinFlow</div>
							<nav className="hidden md:flex space-x-8">
								<a href="#" className="text-gray-900 hover:text-blue-600">
									My stocks
								</a>
								<a href="#" className="text-gray-500 hover:text-blue-600">
									Analytics
								</a>
							</nav>
						</div>
						<div className="flex items-center space-x-4">
							<Button variant="ghost" size="icon">
								<Lock className="h-4 w-4" />
							</Button>
							<Button variant="ghost" size="icon">
								<User className="h-4 w-4" />
							</Button>
						</div>
					</div>
				</div>
			</header>

			{/* Hero Section */}
			<section className="bg-white py-20">
				<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
					<div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
						<div>
							<h1 className="text-4xl font-bold text-gray-900 mb-6">AI 기반 포트폴리오 리밸런싱</h1>
							<p className="text-xl text-gray-600 mb-8">시장 상황과 재무 목표에 맞게 적응하는 강화학습 알고리즘으로 투자 전략을 최적화하세요.</p>

							<div className="space-y-4">
								<Label htmlFor="investment" className="text-lg font-medium">
									투자 금액 입력
								</Label>
								<div className="flex space-x-4">
									<Input id="investment" type="number" placeholder="1000000" value={investmentAmount} onChange={(e) => setInvestmentAmount(e.target.value)} className="text-lg" />
									<span className="flex items-center text-lg text-gray-600">원</span>
								</div>
								<Button onClick={handleAnalysis} disabled={!investmentAmount || isAnalyzing} className="w-full lg:w-auto" size="lg">
									{isAnalyzing ? "AI 분석 중..." : "지금 바로 시작하기"}
								</Button>
								{error && (
									<div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
										<p className="text-red-600 text-sm">{error}</p>
									</div>
								)}
							</div>
						</div>
						<div className="bg-gray-100 rounded-lg p-8 flex items-center justify-center h-80">
							<div className="text-gray-400 text-center">
								<BarChart3 className="h-24 w-24 mx-auto mb-4" />
								<p>포트폴리오 시각화</p>
							</div>
						</div>
					</div>
				</div>
			</section>

			{/* Analysis Progress */}
			{isAnalyzing && (
				<section className="bg-blue-50 py-12">
					<div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
						<h2 className="text-2xl font-bold text-gray-900 mb-4">AI가 시장을 분석하고 있습니다</h2>
						<p className="text-gray-600 mb-8">실시간 데이터 수집 및 강화학습 모델 추론 중...</p>
						<Progress value={66} className="w-full max-w-md mx-auto" />
						<div className="mt-4 space-y-2 text-sm text-gray-500">
							<p>✓ 실시간 주가 데이터 수집 완료</p>
							<p>✓ 기술적 지표 계산 완료</p>
							<p>⏳ 포트폴리오 최적화 진행 중...</p>
						</div>
					</div>
				</section>
			)}

			{/* Results Section */}
			{showResults && (
				<section className="py-16">
					<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
						<h2 className="text-3xl font-bold text-center text-gray-900 mb-12">AI 추천 포트폴리오 결과</h2>

						{/* Portfolio Allocation */}
						<div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
							<Card>
								<CardHeader>
									<CardTitle>추천 포트폴리오 배분</CardTitle>
									<CardDescription>총 투자금액: {Number.parseInt(investmentAmount).toLocaleString()}원</CardDescription>
								</CardHeader>
								<CardContent>
									<div className="space-y-4">
										{portfolioAllocation.map((item, index) => (
											<div key={index} className="flex items-center justify-between">
												<div className="flex items-center space-x-3">
													<Badge variant={item.stock === "현금" ? "secondary" : "default"}>{item.stock}</Badge>
													<span className="font-medium">{item.percentage}%</span>
												</div>
												<span className="text-gray-600">{item.amount.toLocaleString()}원</span>
											</div>
										))}
									</div>
								</CardContent>
							</Card>

							<Card>
								<CardHeader>
									<CardTitle>성과 지표</CardTitle>
									<CardDescription>AI 리밸런싱 통한 성과 포트폴리오 성과</CardDescription>
								</CardHeader>
								<CardContent>
									<div className="grid grid-cols-2 gap-4">
										<div>
											<p className="text-sm text-gray-600">연간 수익률</p>
											<p className="text-2xl font-bold text-green-600">{quickMetrics.annualReturn}</p>
										</div>
										<div>
											<p className="text-sm text-gray-600">샤프 비율</p>
											<p className="text-2xl font-bold">{quickMetrics.sharpeRatio}</p>
										</div>
										<div>
											<p className="text-sm text-gray-600">최대 낙폭</p>
											<p className="text-2xl font-bold text-red-600">{quickMetrics.maxDrawdown}</p>
										</div>
										<div>
											<p className="text-sm text-gray-600">변동성</p>
											<p className="text-2xl font-bold">{quickMetrics.volatility}</p>
										</div>
									</div>
									<p className="text-xs text-gray-500 mt-4">*과거 백테스트 기간: 과거 성과가 미래 결과를 보장하지 않습니다.</p>
								</CardContent>
							</Card>
						</div>

						{/* Performance Metrics Table */}
						<Card>
							<CardHeader>
								<CardTitle>상세 성과 비교</CardTitle>
								<CardDescription>포트폴리오 vs 벤치마크 지수 비교</CardDescription>
							</CardHeader>
							<CardContent>
								<div className="overflow-x-auto">
									<table className="w-full border-collapse">
										<thead>
											<tr className="border-b">
												<th className="text-left py-3 px-4 font-medium">지표</th>
												<th className="text-center py-3 px-4 font-medium">포트폴리오</th>
												<th className="text-center py-3 px-4 font-medium">SPY</th>
												<th className="text-center py-3 px-4 font-medium">QQQ</th>
											</tr>
										</thead>
										<tbody>
											{performanceMetrics.map((metric, index) => (
												<tr key={index} className="border-b hover:bg-gray-50">
													<td className="py-3 px-4 font-medium">{metric.label}</td>
													<td className="py-3 px-4 text-center font-bold text-blue-600">{metric.portfolio}</td>
													<td className="py-3 px-4 text-center text-gray-500">{metric.spy}</td>
													<td className="py-3 px-4 text-center text-gray-500">{metric.qqq}</td>
												</tr>
											))}
										</tbody>
									</table>
								</div>
							</CardContent>
						</Card>
					</div>
				</section>
			)}

			{/* Features Section */}
			<section className="bg-gray-100 py-16">
				<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
					<div className="text-center mb-12">
						<h2 className="text-3xl font-bold text-gray-900 mb-4">작동 방식</h2>
						<p className="text-xl text-gray-600">저희 플랫폼은 고급 강화학습 알고리즘을 사용하여 시장 상황을 지속적으로 분석하고 포트폴리오 배분을 최적화합니다.</p>
					</div>

					<div className="grid grid-cols-1 md:grid-cols-3 gap-8">
						<div className="text-center">
							<div className="bg-white rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
								<BarChart3 className="h-8 w-8 text-blue-600" />
							</div>
							<h3 className="text-xl font-bold mb-2">데이터 분석</h3>
							<p className="text-gray-600">AI가 과거 시장 데이터, 경제 지표 및 포트폴리오 성과를 분석합니다.</p>
						</div>

						<div className="text-center">
							<div className="bg-white rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
								<TrendingUp className="h-8 w-8 text-blue-600" />
							</div>
							<h3 className="text-xl font-bold mb-2">강화학습</h3>
							<p className="text-gray-600">알고리즘은 이전 리밸런싱 결과에 대한 시장 반응을 학습하고 시간에 따라 적응합니다.</p>
						</div>

						<div className="text-center">
							<div className="bg-white rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
								<Users className="h-8 w-8 text-blue-600" />
							</div>
							<h3 className="text-xl font-bold mb-2">맞춤형 전략</h3>
							<p className="text-gray-600">주전한 사용자의 투자 성향을 고려해 조정됩니다.</p>
						</div>
					</div>
				</div>
			</section>

			{/* Platform Features */}
			<section className="py-16">
				<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
					<div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
						<div>
							<h2 className="text-3xl font-bold text-gray-900 mb-6">플랫폼의 주요 기능</h2>
							<p className="text-lg text-gray-600 mb-8">AI 기반 포트폴리오 리밸런서가 어떻게 투자 전략을 변화시킬 수 있는지 알아보세요.</p>

							<div className="space-y-4">
								<div className="flex items-center space-x-3">
									<CheckCircle className="h-5 w-5 text-green-500" />
									<span>자동화된 포트폴리오 리밸런싱 추천</span>
								</div>
								<div className="flex items-center space-x-3">
									<CheckCircle className="h-5 w-5 text-green-500" />
									<span>실시간 시장 데이터 통합</span>
								</div>
								<div className="flex items-center space-x-3">
									<CheckCircle className="h-5 w-5 text-green-500" />
									<span>고급 위험 평가 및 관리</span>
								</div>
								<div className="flex items-center space-x-3">
									<CheckCircle className="h-5 w-5 text-green-500" />
									<span>맞춤형 투자 매개변수</span>
								</div>
							</div>
						</div>

						<div className="bg-gray-100 rounded-lg p-8">
							<div className="text-center text-gray-400">
								<TrendingUp className="h-32 w-32 mx-auto mb-4" />
								<p>성과 차트 시각화</p>
							</div>
						</div>
					</div>
				</div>
			</section>

			{/* Footer */}
			<footer className="bg-gray-900 text-white py-12">
				<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
					<div className="text-xl font-bold mb-4">FinFlow</div>
					<p className="text-gray-400">© 2024 FinFlow. 강화학습 기반 포트폴리오 리스크 관리 플랫폼</p>
				</div>
			</footer>
		</div>
	);
}
