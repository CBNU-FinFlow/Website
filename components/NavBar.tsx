"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Activity, Bell, Search, User } from "lucide-react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import classNames from "classnames";

export default function NavBar() {
    const pathname = usePathname();

    return (
        <header className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50">
            <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between items-center h-14">
                    <div className="flex items-center space-x-6">
                        <div className="flex items-center space-x-2">
                            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                                <Activity className="w-4 h-4 text-white" />
                            </div>
                            <div className="text-lg font-bold text-gray-900">
                                FinFlow
                            </div>
                            <Badge
                                variant="outline"
                                className="text-xs bg-blue-50 text-blue-600 border-blue-200"
                            >
                                AI 투자
                            </Badge>
                        </div>
                        <nav className="hidden md:flex space-x-6">
                            <Link
                                href="/"
                                className={classNames(
                                    "hover:text-gray-700 font-medium text-sm",
                                    pathname === "/"
                                        ? "text-gray-700"
                                        : "text-gray-500"
                                )}
                            >
                                홈
                            </Link>
                            <Link
                                href="/faq"
                                className={classNames(
                                    "hover:text-gray-700 font-medium text-sm",
                                    pathname === "/faq"
                                        ? "text-gray-700"
                                        : "text-gray-500"
                                )}
                            >
                                FAQ
                            </Link>
                        </nav>
                    </div>
                    <div className="flex items-center space-x-2">
                        <Button
                            variant="ghost"
                            size="sm"
                            className="text-gray-600 hover:text-gray-900"
                        >
                            <Search className="h-4 w-4" />
                        </Button>
                        <Button
                            variant="ghost"
                            size="sm"
                            className="text-gray-600 hover:text-gray-900 relative"
                        >
                            <Bell className="h-4 w-4" />
                            <span className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full"></span>
                        </Button>
                        <div className="w-7 h-7 bg-gray-200 rounded-full flex items-center justify-center">
                            <User className="h-4 w-4 text-gray-600" />
                        </div>
                    </div>
                </div>
            </div>
        </header>
    );
}
