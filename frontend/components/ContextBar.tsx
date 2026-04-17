"use client";

import { useState } from "react";

interface Mood {
  key: string;
  emoji: string;
  label: string;
}

interface Props {
  userId: string;
  setUserId: (v: string) => void;
  mood: string;
  setMood: (v: string) => void;
  moods: Mood[];
  query: string;
  setQuery: (v: string) => void;
  onRefresh: () => void;
  loading: boolean;
  mode: string;
}

const TIME_LABELS: Record<number, string> = {
  5: "🌅 Morning", 12: "☀️ Afternoon", 17: "🌆 Evening", 21: "🌙 Night",
};

function getTimeLabel(): string {
  const h = new Date().getHours();
  if (h >= 5  && h < 12) return "🌅 Morning";
  if (h >= 12 && h < 17) return "☀️ Afternoon";
  if (h >= 17 && h < 21) return "🌆 Evening";
  return "🌙 Night";
}

const MODE_COLORS: Record<string, string> = {
  rag: "#6366f1", rl: "#22c55e", cold_start: "#f59e0b",
};

export default function ContextBar({
  userId, setUserId, mood, setMood, moods,
  query, setQuery, onRefresh, loading, mode,
}: Props) {
  const [editId, setEditId] = useState(false);

  return (
    <header style={{
      position: "sticky", top: 0, zIndex: 100,
      background: "rgba(10,14,26,0.85)",
      backdropFilter: "blur(20px)",
      borderBottom: "1px solid var(--glass-border)",
      padding: "0 24px",
    }}>
      {/* Top row */}
      <div style={{ display: "flex", alignItems: "center", gap: 16, height: 64 }}>
        {/* Brand */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
          <span style={{ fontSize: 22 }}>🧠</span>
          <span style={{ fontWeight: 800, fontSize: 18, background: "linear-gradient(135deg,#818cf8,#a78bfa)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            HyperNews
          </span>
        </div>

        <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 12 }}>
          {/* Search / RAG query */}
          <input
            type="text"
            placeholder="🔍  Search with AI (RAG mode)..."
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === "Enter" && onRefresh()}
            style={{
              flex: 1, background: "var(--glass-bg)", border: "1px solid var(--glass-border)",
              borderRadius: 10, padding: "8px 14px", fontSize: 13, color: "var(--text-primary)",
              outline: "none",
            }}
          />

          {/* User ID */}
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>User:</span>
            {editId ? (
              <input
                autoFocus
                value={userId}
                onChange={e => setUserId(e.target.value)}
                onBlur={() => setEditId(false)}
                onKeyDown={e => e.key === "Enter" && setEditId(false)}
                style={{ width: 110, background: "var(--glass-bg)", border: "1px solid var(--accent-light)", borderRadius: 6, padding: "4px 8px", fontSize: 12, color: "var(--text-primary)", outline: "none" }}
              />
            ) : (
              <button onClick={() => setEditId(true)} style={{ background: "var(--glass-bg)", border: "1px solid var(--glass-border)", borderRadius: 6, padding: "4px 10px", fontSize: 12, color: "var(--accent-light)", cursor: "pointer" }}>
                {userId}
              </button>
            )}
          </div>

          {/* Time of day */}
          <span style={{ fontSize: 12, color: "var(--text-muted)", flexShrink: 0 }}>{getTimeLabel()}</span>

          {/* Mode badge */}
          {mode && (
            <span style={{ fontSize: 11, fontWeight: 700, padding: "3px 8px", borderRadius: 6, background: `${MODE_COLORS[mode] || "#475569"}22`, color: MODE_COLORS[mode] || "#94a3b8", border: `1px solid ${MODE_COLORS[mode] || "#475569"}44` }}>
              {mode.toUpperCase()}
            </span>
          )}

          {/* Refresh */}
          <button
            onClick={onRefresh}
            disabled={loading}
            className="pulse-glow"
            style={{
              background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
              border: "none", borderRadius: 10, padding: "8px 18px",
              fontSize: 13, fontWeight: 700, color: "#fff",
              cursor: loading ? "wait" : "pointer",
              opacity: loading ? 0.7 : 1,
              transition: "all 0.2s",
              flexShrink: 0,
            }}
          >
            {loading ? "⟳ Loading..." : "🚀 Refresh"}
          </button>
        </div>
      </div>

      {/* Mood row */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, paddingBottom: 12, paddingTop: 2 }}>
        <span style={{ fontSize: 12, color: "var(--text-muted)", marginRight: 4 }}>Mood:</span>
        {moods.map(m => (
          <button
            key={m.key}
            className={`mood-pill${mood === m.key ? " active" : ""}`}
            onClick={() => { setMood(m.key); }}
          >
            {m.emoji} {m.label}
          </button>
        ))}
      </div>
    </header>
  );
}
