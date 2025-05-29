import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
	try {
		const body = await request.json();

		// 백엔드 서버로 요청 전달
		const response = await fetch("http://localhost:8000/correlation-analysis", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify(body),
		});

		if (!response.ok) {
			throw new Error(`백엔드 응답 오류: ${response.status}`);
		}

		const data = await response.json();

		return NextResponse.json(data);
	} catch (error) {
		console.error("상관관계 분석 오류:", error);
		return NextResponse.json({ error: "상관관계 분석 중 오류가 발생했습니다." }, { status: 500 });
	}
}
