"use client";

interface Article {
  news_id: string;
  title: string;
  abstract: string;
  category: string;
  score?: number;
}

interface Props {
  article: Article;
  feedbackState: string | null;
  onFeedback: (id: string, action: string) => void;
}

const CATEGORY_ICONS: Record<string, string> = {
  Technology: "💻", Sports: "⚽", Politics: "🏛️",
  Entertainment: "🎬", Science: "🔬", Business: "📈",
  Lifestyle: "🌿", Health: "❤️",
};

export default function NewsCard({ article, feedbackState, onFeedback }: Props) {
  const icon  = CATEGORY_ICONS[article.category] ?? "📰";
  const score = typeof article.score === "number" ? Math.min(1, Math.max(0, article.score / 5)) : 0;
  const done  = !!feedbackState;

  return (
    <div className="glass-card" style={{ padding: "20px 22px", height: "100%", display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span className={`badge badge-${article.category}`}>{icon} {article.category}</span>
        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
          Score: {(article.score ?? 0).toFixed(3)}
        </span>
      </div>

      {/* Title */}
      <h3 style={{
        margin: 0, fontSize: 16, fontWeight: 700,
        color: "var(--text-primary)", lineHeight: 1.45,
        display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden",
      }}>
        {article.title}
      </h3>

      {/* Abstract */}
      <p style={{
        margin: 0, fontSize: 13.5, color: "var(--text-secondary)", lineHeight: 1.6, flex: 1,
        display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden",
      }}>
        {article.abstract}
      </p>

      {/* Score bar */}
      <div className="score-bar-track">
        <div className="score-bar-fill" style={{ width: `${score * 100}%` }} />
      </div>

      {/* Actions */}
      <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
        <button
          className="action-btn action-btn-read"
          onClick={() => !done && onFeedback(article.news_id, "read_full")}
          disabled={done}
          style={{ opacity: done && feedbackState !== "read_full" ? 0.4 : 1 }}
        >
          {feedbackState === "read_full" ? "✅" : "👍"} Read
        </button>
        <button
          className="action-btn action-btn-skip"
          onClick={() => !done && onFeedback(article.news_id, "skip")}
          disabled={done}
          style={{ opacity: done && feedbackState !== "skip" ? 0.4 : 1 }}
        >
          {feedbackState === "skip" ? "✓" : "⏩"} Skip
        </button>
        <button
          className="action-btn action-btn-save"
          onClick={() => !done && onFeedback(article.news_id, "save")}
          disabled={done}
          style={{ opacity: done && feedbackState !== "save" ? 0.4 : 1 }}
        >
          {feedbackState === "save" ? "⭐" : "☆"} Save
        </button>
      </div>
    </div>
  );
}
