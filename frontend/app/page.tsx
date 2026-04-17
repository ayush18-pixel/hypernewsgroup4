"use client";

import { useState, useEffect, useCallback } from "react";
import ContextBar from "@/components/ContextBar";
import NewsCard from "@/components/NewsCard";
import ExplanationBanner from "@/components/ExplanationBanner";
import KnowledgeGraphPanel from "@/components/KnowledgeGraphPanel";

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
  { key: "neutral",  emoji: "😐", label: "Neutral" },
  { key: "curious",  emoji: "🤔", label: "Curious" },
  { key: "happy",    emoji: "😊", label: "Happy" },
  { key: "stressed", emoji: "😰", label: "Stressed" },
  { key: "tired",    emoji: "😴", label: "Tired" },
];

export default function Home() {
  const [userId,       setUserId]       = useState("demo_user_1");
  const [mood,         setMood]         = useState("neutral");
  const [query,        setQuery]        = useState("");
  const [articles,     setArticles]     = useState<Article[]>([]);
  const [explanation,  setExplanation]  = useState("");
  const [mode,         setMode]         = useState("");
  const [loading,      setLoading]      = useState(false);
  const [feedbackMap,  setFeedbackMap]  = useState<Record<string,string>>({});
  const [toastMsg,     setToastMsg]     = useState("");

  const fetchRecommendations = useCallback(async () => {
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
      setToastMsg("❌ Cannot reach backend. Is FastAPI running on port 8000?");
    } finally {
      setLoading(false);
    }
  }, [userId, mood, query]);

  useEffect(() => { fetchRecommendations(); }, [fetchRecommendations]);

  const sendFeedback = async (articleId: string, action: string) => {
    setFeedbackMap(prev => ({ ...prev, [articleId]: action }));
    showToast(action === "read_full" ? "✅ Preference saved!" : action === "save" ? "⭐ Saved!" : "⏩ Skipped");
    try {
      await fetch(`${API}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, article_id: articleId, action }),
      });
    } catch { /* silent fail */ }
  };

  const showToast = (msg: string) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(""), 2500);
  };

  return (
    <div style={{ minHeight: "100vh" }}>
      {/* Context Bar */}
      <ContextBar
        userId={userId}
        setUserId={setUserId}
        mood={mood}
        setMood={setMood}
        moods={MOODS}
        query={query}
        setQuery={setQuery}
        onRefresh={fetchRecommendations}
        loading={loading}
        mode={mode}
      />

      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "0 20px 60px" }}>
        {/* Explanation Banner */}
        {explanation && <ExplanationBanner text={explanation} mode={mode} />}

        {/* Knowledge Graph Panel */}
        <KnowledgeGraphPanel />

        {/* Loading skeleton */}
        {loading && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginTop: 24 }}>
            {[...Array(6)].map((_, i) => (
              <div key={i} className="glass-card fade-in-up" style={{
                padding: 24, height: 200, animationDelay: `${i * 0.05}s`,
                background: "linear-gradient(90deg, rgba(255,255,255,0.03) 25%, rgba(255,255,255,0.06) 50%, rgba(255,255,255,0.03) 75%)",
                backgroundSize: "200% 100%", animation: "shimmer 1.5s infinite",
              }} />
            ))}
          </div>
        )}

        {/* Article Grid */}
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

        {/* Empty state */}
        {!loading && articles.length === 0 && !explanation && (
          <div style={{ textAlign: "center", marginTop: 80, color: "var(--text-muted)" }}>
            <div style={{ fontSize: 64, marginBottom: 16 }}>🧠</div>
            <h2 style={{ color: "var(--text-secondary)", marginBottom: 8 }}>Welcome to HyperNews</h2>
            <p>Your AI-powered, mood-aware news feed is ready.</p>
          </div>
        )}
      </main>

      {/* Toast */}
      {toastMsg && (
        <div style={{
          position: "fixed", bottom: 28, right: 28,
          background: "rgba(15,22,41,0.95)", border: "1px solid var(--glass-border)",
          borderRadius: 10, padding: "12px 20px", fontSize: 14, color: "var(--text-primary)",
          backdropFilter: "blur(12px)", zIndex: 9999,
          animation: "fadeInUp 0.3s ease",
        }}>
          {toastMsg}
        </div>
      )}
    </div>
  );
}
