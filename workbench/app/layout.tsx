import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SpineVision Agent Workbench",
  description: "腰椎 MRI 多 Agent 分析、证据检索与人工复核工作台",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
