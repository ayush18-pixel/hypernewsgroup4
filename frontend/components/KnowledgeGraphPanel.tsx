"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { Maximize2, Minimize2 } from "lucide-react";

// react-force-graph-2d accesses `window` at module evaluation time,
// so it MUST be imported with ssr:false — a plain "use client" is not enough.
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
  loading: () => (
    <div style={{ height: 350, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)" }}>
      Loading graph…
    </div>
  ),
});

interface GraphData {
  nodes: any[];
  links: any[];
}

export default function KnowledgeGraphPanel() {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 400 });
  const [isExpanded, setIsExpanded] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/graph`)
      .then(r => r.json())
      .then(data => {
        if (data.error) return;

        const nodes: any[] = [];
        const links: any[] = [];
        const addedNodes = new Set<string>();

        // Category nodes
        data.categories?.forEach((cat: string) => {
          if (!addedNodes.has(cat)) {
            nodes.push({ id: cat, name: cat, type: "category", val: 5 });
            addedNodes.add(cat);
          }
        });

        // Top entity nodes
        data.top_entities?.forEach((ent: any) => {
          if (!addedNodes.has(ent.id)) {
            nodes.push({
              id: ent.id,
              name: ent.id,
              type: ent.type,
              val: Math.max(2, Math.min(10, ent.connections / 2)),
            });
            addedNodes.add(ent.id);
          }
        });

        // Random edges from entities → categories for visualisation
        for (let i = 0; i < nodes.length; i++) {
          if (nodes[i].type !== "category" && Math.random() > 0.5) {
            const cats = data.categories ?? [];
            if (cats.length > 0) {
              links.push({
                source: nodes[i].id,
                target: cats[Math.floor(Math.random() * cats.length)],
              });
            }
          }
        }

        setGraphData({ nodes, links });
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const update = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: isExpanded ? 600 : 350,
        });
      }
    };
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, [isExpanded]);

  const getNodeColor = useCallback((node: any) => {
    const colors: Record<string, string> = {
      category:    "#ef4444",
      ORG:         "#3b82f6",
      PERSON:      "#10b981",
      GPE:         "#f59e0b",
      EVENT:       "#8b5cf6",
      PRODUCT:     "#ec4899",
      WORK_OF_ART: "#06b6d4",
    };
    return colors[node.type] ?? "#94a3b8";
  }, []);

  if (!graphData || graphData.nodes.length === 0) return null;

  return (
    <div
      ref={containerRef}
      className="glass-card"
      style={{ marginTop: 24, padding: 16, position: "relative", transition: "all 0.3s ease" }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <h3 style={{ margin: 0, fontSize: 16, display: "flex", alignItems: "center", gap: 8 }}>
          🕸️ Entity Knowledge Graph
          <span style={{ fontSize: 12, color: "var(--text-muted)", fontWeight: "normal" }}>
            (Top extracted entities from corpus)
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
          nodeLabel="name"
          nodeColor={getNodeColor}
          nodeRelSize={6}
          linkColor={() => "rgba(255,255,255,0.1)"}
          backgroundColor="transparent"
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          onNodeClick={(node) => console.log(node)}
        />
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 12, fontSize: 11, color: "var(--text-secondary)" }}>
        {[
          { color: "#ef4444", label: "Topic Category" },
          { color: "#3b82f6", label: "Organization"   },
          { color: "#10b981", label: "Person"         },
          { color: "#f59e0b", label: "Location"       },
        ].map(({ color, label }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: color }} />
            {label}
          </div>
        ))}
      </div>
    </div>
  );
}
