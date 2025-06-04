// /app/layout.tsx

import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import localFont from 'next/font/local';
import './globals.css';

// 1) 기존 Google Fonts (선택 사항)
const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});
const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

// 2) Pretendard 로컬 폰트 등록
//    경로는 `./font/파일명.ttf` 로 지정합니다.
const pretendard = localFont({
  src: [
    { path: './font/Pretendard-Thin.ttf', weight: '100', style: 'normal' },
    {
      path: './font/Pretendard-ExtraLight.ttf',
      weight: '200',
      style: 'normal',
    },
    { path: './font/Pretendard-Light.ttf', weight: '300', style: 'normal' },
    { path: './font/Pretendard-Regular.ttf', weight: '400', style: 'normal' },
    { path: './font/Pretendard-Medium.ttf', weight: '500', style: 'normal' },
    { path: './font/Pretendard-SemiBold.ttf', weight: '600', style: 'normal' },
    { path: './font/Pretendard-Bold.ttf', weight: '700', style: 'normal' },
    { path: './font/Pretendard-ExtraBold.ttf', weight: '800', style: 'normal' },
    { path: './font/Pretendard-Black.ttf', weight: '900', style: 'normal' },
  ],
  variable: '--font-pretendard',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'My Next.js App',
  description: 'Pretendard 로컬 폰트 적용 예제',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko" suppressHydrationWarning>
      <body
        className={`
          ${pretendard.variable} 
          ${geistSans.variable} 
          ${geistMono.variable} 
          antialiased
        `}
      >
        {children}
      </body>
    </html>
  );
}
