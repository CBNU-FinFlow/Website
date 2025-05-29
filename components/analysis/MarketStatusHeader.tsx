import { useState, useEffect } from "react";
import { MarketStatusResponse, MarketData } from "@/lib/types";

export default function MarketStatusHeader() {
	const [marketData, setMarketData] = useState<MarketData[]>([]);
	const [lastUpdated, setLastUpdated] = useState<string>("");
	const [loading, setLoading] = useState(false);

	// 실시간 시장 데이터 가져오기
	const fetchMarketData = async () => {
		try {
			setLoading(true);
			const response = await fetch("/api/market-status");

			if (response.ok) {
				const data: MarketStatusResponse = await response.json();
				setMarketData(data.market_data);
				setLastUpdated(data.last_updated);
			} else {
				console.error("시장 데이터 조회 실패");
			}
		} catch (error) {
			console.error("시장 데이터 가져오기 오류:", error);
		} finally {
			setLoading(false);
		}
	};

	// 컴포넌트 마운트 시 및 5분마다 데이터 갱신
	useEffect(() => {
		fetchMarketData();

		// 5분마다 갱신
		const interval = setInterval(fetchMarketData, 5 * 60 * 1000);

		return () => clearInterval(interval);
	}, []);

	const formatPrice = (price: number) => {
		if (price === 0) return "N/A";
		return price.toLocaleString(undefined, {
			minimumFractionDigits: 2,
			maximumFractionDigits: 2,
		});
	};

	const formatChange = (change: number, changePercent: number) => {
		if (change === 0 && changePercent === 0) return "N/A";

		const sign = changePercent >= 0 ? "+" : "";
		return `${sign}${changePercent.toFixed(2)}%`;
	};

	const getChangeColor = (changePercent: number) => {
		if (changePercent === 0) return "text-gray-300";
		return changePercent >= 0 ? "text-green-300" : "text-red-300";
	};

	return (
		<div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-6 text-white">
			<div className="flex items-center justify-between mb-4">
				<div className="flex items-center space-x-3">
					<div className={`w-3 h-3 rounded-full ${loading ? "bg-yellow-400" : "bg-green-400"} ${loading ? "" : "animate-pulse"}`}></div>
					<span className="text-sm font-medium">실시간 시장 분석</span>
					<span className="text-xs bg-white/20 px-2 py-1 rounded">LIVE</span>
				</div>
				<div className="text-sm opacity-90">
					{lastUpdated
						? new Date(lastUpdated).toLocaleString("ko-KR", {
								month: "long",
								day: "numeric",
								hour: "2-digit",
								minute: "2-digit",
						  })
						: new Date().toLocaleString("ko-KR", {
								month: "long",
								day: "numeric",
								hour: "2-digit",
								minute: "2-digit",
						  })}
				</div>
			</div>
			<div className="grid grid-cols-4 gap-4">
				{marketData.length > 0
					? marketData.map((market, index) => (
							<div key={index} className="text-center">
								<div className="text-xl font-bold">{market.name}</div>
								<div className="text-sm opacity-90">{formatPrice(market.price)}</div>
								<div className={`text-xs ${getChangeColor(market.change_percent)}`}>{formatChange(market.change, market.change_percent)}</div>
							</div>
					  ))
					: // 로딩 중이거나 데이터가 없을 때 스켈레톤 UI
					  [...Array(4)].map((_, index) => (
							<div key={index} className="text-center">
								<div className="text-xl font-bold">{loading ? <div className="animate-pulse bg-white/20 h-6 rounded"></div> : "N/A"}</div>
								<div className="text-sm opacity-90 mt-1">{loading ? <div className="animate-pulse bg-white/20 h-4 rounded"></div> : "데이터 없음"}</div>
								<div className="text-xs text-gray-300 mt-1">{loading ? <div className="animate-pulse bg-white/20 h-3 rounded"></div> : "N/A"}</div>
							</div>
					  ))}
			</div>
		</div>
	);
}
