import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "HyperNews — AI-Powered Personalized News",
  description: "Hyper-personalized news recommendations powered by RAG, Knowledge Graphs, and Reinforcement Learning.",
  openGraph: {
    title: "HyperNews",
    description: "Your mood-aware, context-intelligent news feed.",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="antialiased">{children}</body>
    </html>
  );
}
