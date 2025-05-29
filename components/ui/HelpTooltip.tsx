"use client";

import { useState } from "react";
import { HelpCircle } from "lucide-react";

interface HelpTooltipProps {
	title: string;
	description: string;
	className?: string;
}

export default function HelpTooltip({ title, description, className = "" }: HelpTooltipProps) {
	const [isVisible, setIsVisible] = useState(false);

	return (
		<div className={`relative inline-block ${className}`}>
			<button
				type="button"
				className="inline-flex items-center justify-center w-4 h-4 text-gray-400 hover:text-blue-600 transition-colors"
				onMouseEnter={() => setIsVisible(true)}
				onMouseLeave={() => setIsVisible(false)}
				onFocus={() => setIsVisible(true)}
				onBlur={() => setIsVisible(false)}
			>
				<HelpCircle className="w-4 h-4" />
			</button>

			{isVisible && (
				<div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 z-50">
					<div className="bg-gray-900 text-white text-sm rounded-lg shadow-lg p-3 min-w-[250px] max-w-[350px]">
						<div className="font-medium mb-1">{title}</div>
						<div className="text-gray-300 text-xs leading-relaxed">{description}</div>
						{/* 화살표 */}
						<div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-[6px] border-r-[6px] border-t-[6px] border-l-transparent border-r-transparent border-t-gray-900"></div>
					</div>
				</div>
			)}
		</div>
	);
}
