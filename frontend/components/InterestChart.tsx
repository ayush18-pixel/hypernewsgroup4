"use client";

import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import type { Formatter, NameType, ValueType } from "recharts/types/component/DefaultTooltipContent";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const CATEGORY_COLORS: Record<string, string> = {
  Technology: "#3b82f6", Sports: "#22c55e", Politics: "#ef4444",
  Entertainment: "#eab308", Science: "#a855f7", Business: "#14b8a6",
  Lifestyle: "#f97316", Health: "#ec4899",
};

interface Props {
  userId: string;
}

const formatInterestValue: Formatter<ValueType, NameType> = (value, _name) => {
  const rawValue = Array.isArray(value) ? value[0] : value;
  const numericValue =
    typeof rawValue === "number" && Number.isFinite(rawValue)
      ? rawValue
      : Number(rawValue ?? 0);
  const safeValue = Number.isFinite(numericValue) ? numericValue : 0;
  return [`${safeValue.toFixed(2)} pts`, "Interest Score"];
};

export default function InterestChart({ userId }: Props) {
  const [data, setData] = useState<{ name: string; value: number }[]>([]);

  useEffect(() => {
    if (!userId) return;
    fetch(`${API}/profile/${userId}`)
      .then(r => r.json())
      .then(profile => {
        const interests = profile.interests as Record<string, number>;
        const entries = Object.entries(interests)
          .map(([name, value]) => ({ name, value: Math.max(0, value) }))
          .sort((a, b) => b.value - a.value)
          .slice(0, 8);
        setData(entries);
      })
      .catch(() => {});
  }, [userId]);

  if (data.length === 0) {
    return (
      <div style={{ textAlign: "center", color: "var(--text-muted)", padding: "20px 0", fontSize: 13 }}>
        Read articles to build your interest profile 📊
      </div>
    );
  }

  return (
    <div style={{ width: "100%", height: 180 }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 16 }}>
          <XAxis type="number" hide />
          <YAxis
            type="category" dataKey="name" width={90}
            tick={{ fill: "#94a3b8", fontSize: 12 }}
            axisLine={false} tickLine={false}
          />
          <Tooltip
            contentStyle={{ background: "rgba(10,14,26,0.95)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: "#f1f5f9" }}
            formatter={formatInterestValue}
          />
          <Bar dataKey="value" radius={[0, 6, 6, 0]}>
            {data.map(entry => (
              <Cell key={entry.name} fill={CATEGORY_COLORS[entry.name] || "#6366f1"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
