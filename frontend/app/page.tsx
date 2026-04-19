"use client";

import { useCallback, useEffect, useState } from "react";
import ContextBar from "@/components/ContextBar";
import ExplanationBanner from "@/components/ExplanationBanner";
import KnowledgeGraphPanel from "@/components/KnowledgeGraphPanel";
import NewsCard from "@/components/NewsCard";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Article {
  news_id: string;
  title: string;
  abstract: string;
  category: string;
  score?: number;
}

interface RecommendResponse {
  articles: Article[];
  explanation: string;
  mode: string;
}

const MOODS = [
  { key: "neutral", emoji: "😐", label: "Neutral" },
  { key: "curious", emoji: "🤔", label: "Curious" },
  { key: "happy", emoji: "😊", label: "Happy" },
  { key: "stressed", emoji: "😰", label: "Stressed" },
  { key: "tired", emoji: "😴", label: "Tired" },
];

function createSessionUserId() {
  if (typeof window !== "undefined" && window.crypto?.randomUUID) {
    return `session_${window.crypto.randomUUID().slice(0, 8)}`;
  }
  return `session_${Math.random().toString(36).slice(2, 10)}`;
}

export default function Home() {
  const [userId, setUserId] = useState("");
  const [mood, setMood] = useState("neutral");
  const [query, setQuery] = useState("");
  const [articles, setArticles] = useState<Article[]>([]);
  const [explanation, setExplanation] = useState("");
  const [mode, setMode] = useState("");
  const [loading, setLoading] = useState(false);
  const [feedbackMap, setFeedbackMap] = useState<Record<string, string>>({});
  const [toastMsg, setToastMsg] = useState("");

  useEffect(() => {
    const storedUserId = window.sessionStorage.getItem("hypernews_user_id");
    if (storedUserId) {
      setUserId(storedUserId);
      return;
    }
    const nextUserId = createSessionUserId();
    window.sessionStorage.setItem("hypernews_user_id", nextUserId);
    setUserId(nextUserId);
  }, []);

  useEffect(() => {
    if (!userId) return;
    window.sessionStorage.setItem("hypernews_user_id", userId);
    setFeedbackMap({});
  }, [userId]);

  const fetchRecommendations = useCallback(async () => {
    if (!userId) return;

    setLoading(true);
    try {
      const res = await fetch(`${API}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, mood, query: query || null, n: 8 }),
      });
      const data: RecommendResponse = await res.json();
      setArticles(data.articles || []);
      setExplanation(data.explanation || "");
      setMode(data.mode || "");
    } catch {
      setToastMsg("Cannot reach backend. Is FastAPI running on port 8000?");
    } finally {
      setLoading(false);
    }
  }, [userId, mood, query]);

  useEffect(() => {
    void fetchRecommendations();
  }, [fetchRecommendations]);

  const showToast = (msg: string) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(""), 2500);
  };

  const sendFeedback = async (articleId: string, action: string) => {
    setFeedbackMap((prev) => ({ ...prev, [articleId]: action }));
    showToast(action === "read_full" ? "Preference saved" : action === "save" ? "Saved" : "Skipped");
    try {
      await fetch(`${API}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, article_id: articleId, action }),
      });
      await fetchRecommendations();
    } catch {
      /* silent fail */
    }
  };

  const startFreshSession = async () => {
    if (userId) {
      try {
        await fetch(`${API}/reset/${userId}`, { method: "POST" });
      } catch {
        /* silent fail */
      }
    }

    const nextUserId = createSessionUserId();
    window.sessionStorage.setItem("hypernews_user_id", nextUserId);
    setArticles([]);
    setExplanation("");
    setMode("");
    setFeedbackMap({});
    setUserId(nextUserId);
    showToast("Fresh session started");
  };

  return (
    <div style={{ minHeight: "100vh" }}>
      <ContextBar
        userId={userId}
        setUserId={setUserId}
        mood={mood}
        setMood={setMood}
        moods={MOODS}
        query={query}
        setQuery={setQuery}
        onRefresh={fetchRecommendations}
        onNewSession={startFreshSession}
        loading={loading}
        mode={mode}
      />

      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "0 20px 60px" }}>
        {explanation && <ExplanationBanner text={explanation} mode={mode} />}
        <KnowledgeGraphPanel />

        {loading && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginTop: 24 }}>
            {[...Array(6)].map((_, i) => (
              <div
                key={i}
                className="glass-card fade-in-up"
                style={{
                  padding: 24,
                  height: 200,
                  animationDelay: `${i * 0.05}s`,
                  background:
                    "linear-gradient(90deg, rgba(255,255,255,0.03) 25%, rgba(255,255,255,0.06) 50%, rgba(255,255,255,0.03) 75%)",
                  backgroundSize: "200% 100%",
                  animation: "shimmer 1.5s infinite",
                }}
              />
            ))}
          </div>
        )}

        {!loading && articles.length > 0 && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(520px, 1fr))", gap: 20, marginTop: 24 }}>
            {articles.map((article, i) => (
              <div key={article.news_id} className="fade-in-up" style={{ animationDelay: `${i * 0.06}s` }}>
                <NewsCard
                  article={article}
                  feedbackState={feedbackMap[article.news_id] || null}
                  onFeedback={sendFeedback}
                />
              </div>
            ))}
          </div>
        )}

        {!loading && articles.length === 0 && !explanation && (
          <div style={{ textAlign: "center", marginTop: 80, color: "var(--text-muted)" }}>
            <div style={{ fontSize: 64, marginBottom: 16 }}>🧠</div>
            <h2 style={{ color: "var(--text-secondary)", marginBottom: 8 }}>Welcome to HyperNews</h2>
            <p>Your AI-powered, mood-aware news feed is ready.</p>
          </div>
        )}
      </main>

      {toastMsg && (
        <div
          style={{
            position: "fixed",
            bottom: 28,
            right: 28,
            background: "rgba(15,22,41,0.95)",
            border: "1px solid var(--glass-border)",
            borderRadius: 10,
            padding: "12px 20px",
            fontSize: 14,
            color: "var(--text-primary)",
            backdropFilter: "blur(12px)",
            zIndex: 9999,
            animation: "fadeInUp 0.3s ease",
          }}
        >
          {toastMsg}
        </div>
      )}
    </div>
  );
}
