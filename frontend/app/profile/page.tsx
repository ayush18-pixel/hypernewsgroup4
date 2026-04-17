"use client";

import { useEffect, useState } from "react";
import InterestChart from "@/components/InterestChart";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Profile {
  user_id: string;
  mood: string;
  time_of_day: string;
  interests: Record<string, number>;
  articles_read: number;
  recent_clicks: string[];
  session_topics: string[];
}

export default function ProfilePage({ searchParams }: { searchParams: { user?: string } }) {
  const userId = searchParams.user || "demo_user_1";
  const [profile, setProfile] = useState<Profile | null>(null);

  useEffect(() => {
    fetch(`${API}/profile/${userId}`).then(r => r.json()).then(setProfile).catch(() => {});
  }, [userId]);

  return (
    <div style={{ maxWidth: 720, margin: "60px auto", padding: "0 24px" }}>
      <div style={{ marginBottom: 32 }}>
        <a href="/" style={{ color: "var(--accent-light)", fontSize: 13, textDecoration: "none" }}>← Back to Feed</a>
      </div>

      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        👤 {userId}
      </h1>
      <p style={{ color: "var(--text-muted)", marginBottom: 32, fontSize: 14 }}>Your personalized reading profile</p>

      {/* Stats row */}
      {profile && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginBottom: 32 }}>
          {[
            { label: "Articles Read", value: profile.articles_read },
            { label: "Current Mood", value: profile.mood },
            { label: "Time of Day", value: profile.time_of_day },
          ].map(stat => (
            <div key={stat.label} className="glass-card" style={{ padding: "18px 20px", textAlign: "center" }}>
              <div style={{ fontSize: 22, fontWeight: 800, color: "var(--accent-light)" }}>{stat.value}</div>
              <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>{stat.label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Interest chart */}
      <div className="glass-card" style={{ padding: "24px 20px" }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 20, color: "var(--text-primary)" }}>
          📊 Category Interest Profile
        </h2>
        <InterestChart userId={userId} />
      </div>

      {/* Recent topics */}
      {profile?.session_topics && profile.session_topics.length > 0 && (
        <div className="glass-card" style={{ padding: "20px", marginTop: 20 }}>
          <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 14, color: "var(--text-primary)" }}>
            🕐 Session Topics
          </h2>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {profile.session_topics.slice(-10).map((t, i) => (
              <span key={i} className="badge" style={{ background: "var(--glass-bg)", color: "var(--text-secondary)", border: "1px solid var(--glass-border)" }}>
                {t}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
