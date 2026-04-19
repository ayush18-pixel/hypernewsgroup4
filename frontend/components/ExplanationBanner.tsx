"use client";

import { useEffect, useState } from "react";

interface Props {
  text: string;
  mode: string;
}

const MODE_ICONS: Record<string, string> = {
  rag: "🔍", rl: "🤖", cold_start: "✨",
};
const MODE_LABELS: Record<string, string> = {
  rag: "RAG Search", rl: "Reinforcement Learning", cold_start: "Personalising",
};

export default function ExplanationBanner({ text, mode }: Props) {
  const [displayed, setDisplayed] = useState("");

  useEffect(() => {
    setDisplayed("");
    let i = 0;
    const interval = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) clearInterval(interval);
    }, 14);
    return () => clearInterval(interval);
  }, [text]);

  return (
    <div className="explanation-banner fade-in-up" style={{ margin: "24px 0 0" }}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
        <span style={{ fontSize: 22, flexShrink: 0 }}>💡</span>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <span style={{ fontSize: 12, fontWeight: 700, color: "var(--accent-light)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
              AI Reasoning
            </span>
            {mode && (
              <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
                via {MODE_ICONS[mode]} {MODE_LABELS[mode]}
              </span>
            )}
          </div>
          <p style={{ margin: 0, fontSize: 14, color: "var(--text-secondary)", lineHeight: 1.65 }}>
            {displayed}
            <span style={{ opacity: displayed.length < text.length ? 1 : 0, animation: "pulse-glow 1s infinite" }}>|</span>
          </p>
        </div>
      </div>
    </div>
  );
}
