export default function MarketStatusHeader() {
	return (
		<div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-6 text-white">
			<div className="flex items-center justify-between mb-4">
				<div className="flex items-center space-x-3">
					<div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
					<span className="text-sm font-medium">실시간 시장 분석</span>
					<span className="text-xs bg-white/20 px-2 py-1 rounded">LIVE</span>
				</div>
				<div className="text-sm opacity-90">
					{new Date().toLocaleString("ko-KR", {
						month: "long",
						day: "numeric",
						hour: "2-digit",
						minute: "2-digit",
					})}
				</div>
			</div>
			<div className="grid grid-cols-4 gap-4">
				<div className="text-center">
					<div className="text-2xl font-bold">S&P 500</div>
					<div className="text-sm opacity-90">4,567.23</div>
					<div className="text-xs text-green-300">+0.85%</div>
				</div>
				<div className="text-center">
					<div className="text-2xl font-bold">NASDAQ</div>
					<div className="text-sm opacity-90">14,234.56</div>
					<div className="text-xs text-green-300">+1.23%</div>
				</div>
				<div className="text-center">
					<div className="text-2xl font-bold">VIX</div>
					<div className="text-sm opacity-90">18.45</div>
					<div className="text-xs text-red-300">+2.1%</div>
				</div>
				<div className="text-center">
					<div className="text-2xl font-bold">USD/KRW</div>
					<div className="text-sm opacity-90">1,345.67</div>
					<div className="text-xs text-red-300">-0.12%</div>
				</div>
			</div>
		</div>
	);
}
