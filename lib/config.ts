// 환경 변수를 통한 API 설정 관리

// 백엔드 API 서버 URL 설정
export const getApiBaseUrl = (): string => {
	// 1순위: 명시적 환경 변수
	if (process.env.NEXT_PUBLIC_API_BASE_URL) {
		return process.env.NEXT_PUBLIC_API_BASE_URL;
	}

	// 2순위: NODE_ENV 기반 자동 판단
	if (process.env.NODE_ENV === "production") {
		// API 서브도메인 대신 같은 도메인의 다른 포트 사용
		return "https://finflow.reo91004.com:8000";
	}

	// 3순위: 개발 환경 기본값
	return "http://localhost:8000";
};

// 환경 구분
export const getEnvironment = (): "development" | "production" => {
	// 1순위: 명시적 환경 변수
	if (process.env.NEXT_PUBLIC_ENVIRONMENT) {
		return process.env.NEXT_PUBLIC_ENVIRONMENT as "development" | "production";
	}

	// 2순위: NODE_ENV 기반 자동 판단
	return process.env.NODE_ENV === "production" ? "production" : "development";
};

// 프로덕션 환경 여부 확인
export const isProduction = (): boolean => {
	return getEnvironment() === "production";
};

// API 엔드포인트 생성 함수
export const createApiUrl = (endpoint: string): string => {
	const baseUrl = getApiBaseUrl();
	return `${baseUrl}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;
};

// 공통 fetch 옵션
export const getDefaultFetchOptions = () => ({
	headers: {
		"Content-Type": "application/json",
	},
});

// 환경별 설정
export const config = {
	api: {
		baseUrl: getApiBaseUrl(),
		timeout: isProduction() ? 30000 : 10000, // 프로덕션에서는 더 긴 타임아웃
	},
	environment: getEnvironment(),
	isProduction: isProduction(),
	// 환경 정보 출력 (디버깅용)
	debug: {
		nodeEnv: process.env.NODE_ENV,
		nextPublicApiUrl: process.env.NEXT_PUBLIC_API_BASE_URL,
		nextPublicEnv: process.env.NEXT_PUBLIC_ENVIRONMENT,
		detectedApiUrl: getApiBaseUrl(),
		detectedEnv: getEnvironment(),
	},
};
