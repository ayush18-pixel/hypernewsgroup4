"use client";

import { useEffect, useState } from "react";

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
  onNewSession: () => void;
  loading: boolean;
  mode: string;
}

function getTimeLabel(): string {
  const hour = new Date().getHours();
  if (hour >= 5 && hour < 12) return "🌅 Morning";
  if (hour >= 12 && hour < 17) return "☀️ Afternoon";
  if (hour >= 17 && hour < 21) return "🌆 Evening";
  return "🌙 Night";
}

const MODE_COLORS: Record<string, string> = {
  rag: "#6366f1",
  rl: "#22c55e",
  cold_start: "#f59e0b",
};

export default function ContextBar({
  userId,
  setUserId,
  mood,
  setMood,
  moods,
  query,
  setQuery,
  onRefresh,
  onNewSession,
  loading,
  mode,
}: Props) {
  const [editId, setEditId] = useState(false);
  const [draftUserId, setDraftUserId] = useState(userId);

  useEffect(() => {
    setDraftUserId(userId);
  }, [userId]);

  const commitUserId = () => {
    const nextUserId = draftUserId.trim();
    if (nextUserId) {
      setUserId(nextUserId);
    }
    setEditId(false);
  };

  return (
    <header
      style={{
        position: "sticky",
        top: 0,
        zIndex: 100,
        background: "rgba(10,14,26,0.85)",
        backdropFilter: "blur(20px)",
        borderBottom: "1px solid var(--glass-border)",
        padding: "0 24px",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 16, minHeight: 64, padding: "10px 0" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
          <span style={{ fontSize: 22 }}>🧠</span>
          <span
            style={{
              fontWeight: 800,
              fontSize: 18,
              background: "linear-gradient(135deg,#818cf8,#a78bfa)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            HyperNews
          </span>
        </div>

        <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <input
            type="text"
            placeholder="Search with AI (RAG mode)..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onRefresh()}
            style={{
              flex: 1,
              minWidth: 240,
              background: "var(--glass-bg)",
              border: "1px solid var(--glass-border)",
              borderRadius: 10,
              padding: "8px 14px",
              fontSize: 13,
              color: "var(--text-primary)",
              outline: "none",
            }}
          />

          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>User:</span>
            {editId ? (
              <input
                autoFocus
                value={draftUserId}
                onChange={(e) => setDraftUserId(e.target.value)}
                onBlur={commitUserId}
                onKeyDown={(e) => e.key === "Enter" && commitUserId()}
                style={{
                  width: 150,
                  background: "var(--glass-bg)",
                  border: "1px solid var(--accent-light)",
                  borderRadius: 6,
                  padding: "4px 8px",
                  fontSize: 12,
                  color: "var(--text-primary)",
                  outline: "none",
                }}
              />
            ) : (
              <button
                onClick={() => {
                  setDraftUserId(userId);
                  setEditId(true);
                }}
                style={{
                  background: "var(--glass-bg)",
                  border: "1px solid var(--glass-border)",
                  borderRadius: 6,
                  padding: "4px 10px",
                  fontSize: 12,
                  color: "var(--accent-light)",
                  cursor: "pointer",
                }}
              >
                {userId || "loading..."}
              </button>
            )}
          </div>

          <span style={{ fontSize: 12, color: "var(--text-muted)", flexShrink: 0 }}>{getTimeLabel()}</span>

          {mode && (
            <span
              style={{
                fontSize: 11,
                fontWeight: 700,
                padding: "3px 8px",
                borderRadius: 6,
                background: `${MODE_COLORS[mode] || "#475569"}22`,
                color: MODE_COLORS[mode] || "#94a3b8",
                border: `1px solid ${MODE_COLORS[mode] || "#475569"}44`,
              }}
            >
              {mode.toUpperCase()}
            </span>
          )}

          <button
            onClick={onNewSession}
            disabled={loading}
            style={{
              background: "rgba(255,255,255,0.03)",
              border: "1px solid var(--glass-border)",
              borderRadius: 10,
              padding: "8px 14px",
              fontSize: 13,
              fontWeight: 700,
              color: "var(--text-secondary)",
              cursor: loading ? "wait" : "pointer",
              opacity: loading ? 0.7 : 1,
            }}
          >
            Fresh Session
          </button>

          <button
            onClick={onRefresh}
            disabled={loading}
            className="pulse-glow"
            style={{
              background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
              border: "none",
              borderRadius: 10,
              padding: "8px 18px",
              fontSize: 13,
              fontWeight: 700,
              color: "#fff",
              cursor: loading ? "wait" : "pointer",
              opacity: loading ? 0.7 : 1,
              transition: "all 0.2s",
              flexShrink: 0,
            }}
          >
            {loading ? "Loading..." : "Refresh"}
          </button>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, paddingBottom: 12, paddingTop: 2, flexWrap: "wrap" }}>
        <span style={{ fontSize: 12, color: "var(--text-muted)", marginRight: 4 }}>Mood:</span>
        {moods.map((moodOption) => (
          <button
            key={moodOption.key}
            className={`mood-pill${mood === moodOption.key ? " active" : ""}`}
            onClick={() => setMood(moodOption.key)}
          >
            {moodOption.emoji} {moodOption.label}
          </button>
        ))}
      </div>
    </header>
  );
}
