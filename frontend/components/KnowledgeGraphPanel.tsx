"use client";

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { Maximize2, Minimize2 } from "lucide-react";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
  loading: () => (
    <div style={{ height: 350, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)" }}>
      Loading graph...
    </div>
  ),
});

interface GraphNode {
  id: string;
  label: string;
  type: string;
  degree?: number;
}

interface GraphLink {
  source: string;
  target: string;
  relation?: string;
}

interface GraphEntity {
  id: string;
  type: string;
  connections: number;
}

interface GraphResponse {
  error?: string;
  top_entities?: GraphEntity[];
  nodes?: GraphNode[];
  links?: GraphLink[];
}

interface GraphData {
  nodes: Array<GraphNode & { val: number; name: string; color: string; tooltip: string }>;
  links: GraphLink[];
}

const TYPE_COLORS: Record<string, string> = {
  article: "#64748b",
  category: "#ef4444",
  subcategory: "#f97316",
  PERSON: "#10b981",
  GPE: "#f59e0b",
  ORG: "#3b82f6",
  CONCEPT: "#8b5cf6",
  ENTITY: "#94a3b8",
  EVENT: "#ec4899",
  LOCATION: "#06b6d4",
};

export default function KnowledgeGraphPanel() {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [topEntities, setTopEntities] = useState<GraphEntity[]>([]);
  const [dimensions, setDimensions] = useState({ width: 800, height: 400 });
  const [isExpanded, setIsExpanded] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/graph`)
      .then((response) => response.json())
      .then((data: GraphResponse) => {
        if (data.error || !data.nodes?.length) return;
        setTopEntities(data.top_entities ?? []);
        setGraphData({
          nodes: data.nodes.map((node) => ({
            ...node,
            name: node.label,
            color: TYPE_COLORS[node.type] ?? "#94a3b8",
            tooltip: `${node.label} (${node.type})`,
            val:
              node.type === "article"
                ? 2.5
                : Math.max(3, Math.min(12, (node.degree ?? 1) / (node.type === "category" ? 1.3 : 1.8))),
          })),
          links: data.links ?? [],
        });
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const updateDimensions = () => {
      if (!containerRef.current) return;
      setDimensions({
        width: containerRef.current.clientWidth,
        height: isExpanded ? 620 : 360,
      });
    };

    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, [isExpanded]);

  if (!graphData || graphData.nodes.length === 0) return null;

  return (
    <div
      ref={containerRef}
      className="glass-card"
      style={{ marginTop: 24, padding: 16, position: "relative", transition: "all 0.3s ease" }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <h3 style={{ margin: 0, fontSize: 16, display: "flex", alignItems: "center", gap: 8 }}>
          Entity Knowledge Graph
          <span style={{ fontSize: 12, color: "var(--text-muted)", fontWeight: "normal" }}>
            (real category/article/entity edges from the dataset)
          </span>
        </h3>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          style={{ background: "transparent", border: "none", color: "var(--text-secondary)", cursor: "pointer" }}
        >
          {isExpanded ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
        </button>
      </div>

      <div style={{ borderRadius: 8, overflow: "hidden", background: "rgba(0,0,0,0.2)", border: "1px solid var(--glass-border)" }}>
        <ForceGraph2D
          width={dimensions.width}
          height={dimensions.height}
          graphData={graphData}
          nodeLabel="tooltip"
          nodeColor="color"
          nodeRelSize={5}
          linkColor={() => "rgba(255,255,255,0.11)"}
          backgroundColor="transparent"
          d3AlphaDecay={0.035}
          d3VelocityDecay={0.3}
          cooldownTicks={90}
        />
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 12, fontSize: 11, color: "var(--text-secondary)" }}>
        {[
          { color: TYPE_COLORS.category, label: "Category" },
          { color: TYPE_COLORS.article, label: "Article" },
          { color: TYPE_COLORS.PERSON, label: "Person" },
          { color: TYPE_COLORS.ORG, label: "Organization" },
          { color: TYPE_COLORS.GPE, label: "Location" },
          { color: TYPE_COLORS.CONCEPT, label: "Concept" },
        ].map(({ color, label }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: color }} />
            {label}
          </div>
        ))}
      </div>

      {topEntities.length > 0 && (
        <div style={{ marginTop: 14, display: "flex", flexWrap: "wrap", gap: 8 }}>
          {topEntities.slice(0, 10).map((entity) => (
            <span
              key={`${entity.id}-${entity.type}`}
              className="badge"
              style={{
                background: "rgba(255,255,255,0.04)",
                color: "var(--text-secondary)",
                border: "1px solid var(--glass-border)",
              }}
            >
              {entity.id} · {entity.connections}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
