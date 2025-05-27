// 포트폴리오 배분 타입
export interface PortfolioAllocation {
	stock: string;
	percentage: number;
	amount: number;
}

// 성과 지표 타입
export interface PerformanceMetrics {
	label: string;
	portfolio: string;
	spy: string;
	qqq: string;
}

// 간단한 성과 지표 타입
export interface QuickMetrics {
	annualReturn: string;
	sharpeRatio: string;
	maxDrawdown: string;
	volatility: string;
}

// 예측 결과 타입
export interface PredictionResult {
	portfolioAllocation: PortfolioAllocation[];
	performanceMetrics: PerformanceMetrics[];
	quickMetrics: QuickMetrics;
}

// API 응답 타입
export interface ApiResponse<T> {
	data?: T;
	error?: string;
}
