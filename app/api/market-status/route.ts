import { NextRequest, NextResponse } from "next/server";
import { createApiUrl, getDefaultFetchOptions } from "@/lib/config";

export async function GET(request: NextRequest) {
	try {
		// 백엔드 서버로 요청 전달
		const response = await fetch(createApiUrl("/market-status"), {
			method: "GET",
			...getDefaultFetchOptions(),
		});

		if (!response.ok) {
			throw new Error(`백엔드 응답 오류: ${response.status}`);
		}

		const data = await response.json();

		return NextResponse.json(data);
	} catch (error) {
		console.error("시장 데이터 조회 오류:", error);
		return NextResponse.json({ error: "시장 데이터 조회 중 오류가 발생했습니다." }, { status: 500 });
	}
}
