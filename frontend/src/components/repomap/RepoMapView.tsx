/**
 * RepoMapView — Community-aware repo knowledge graph visualization.
 */
import { useEffect, useRef, useState, useCallback } from "react";
import {
  CommunityDetail as CommunityDetailType,
  getCommunityDetail,
  getRepoMap,
  getSymbolDetail,
  RepoMapCommunity,
  RepoMapSummary,
  SymbolDetail as SymbolDetailType,
} from "../../api/client";
import { useStore } from "../../store/useStore";
import ProcessFlow from "./ProcessFlow";

const PALETTE = [
  "#00d4ff", "#7c3aed", "#10b981", "#f59e0b", "#ef4444",
  "#ec4899", "#3b82f6", "#14b8a6", "#f97316", "#8b5cf6",
  "#34d399", "#fbbf24", "#60a5fa", "#f472b6", "#a78bfa",
];

function colorForIdx(i: number): string {
  return PALETTE[i % PALETTE.length];
}
function colorForCommunity(
  id: string | null | undefined,
  communities: RepoMapCommunity[],
): string {
  const i = communities.findIndex((c) => c.id === id);
  return i >= 0 ? colorForIdx(i) : "#4b5563";
}

// ── Symbol detail ──────────────────────────────────────────────

interface SymbolDetailProps {
  symbolId: string;
  communities: RepoMapCommunity[];
  onClose: () => void;
}

function SymbolDetail({ symbolId, communities, onClose }: SymbolDetailProps) {
  const [detail, setDetail] = useState<SymbolDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getSymbolDetail(symbolId)
      .then((d) => {
        if (!cancelled) setDetail(d);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [symbolId]);

  return (
    <div style={detailBoxStyle}>
      <div style={detailHeaderStyle}>
        <span style={{ color: "#00d4ff", fontSize: 11, letterSpacing: 1 }}>SYMBOL</span>
        <button onClick={onClose} style={closeBtnStyle}>✕</button>
      </div>
      {loading && <DetailSkeleton />}
      {error && !loading && (
        <div style={{ ...dimText, color: "#f87171" }}>Failed to load: {error}</div>
      )}
      {detail && !loading && (
        <>
          <div style={{ color: "#e2e8f0", fontWeight: 600, fontSize: 13, marginBottom: 2 }}>
            {detail.name}
          </div>
          <div style={{ color: "#9ca3af", fontSize: 10, marginBottom: 4 }}>
            {detail.label}
          </div>
          <div style={{ color: "#9ca3af", fontSize: 10, marginBottom: 10, wordBreak: "break-all" }}>
            {detail.file_path}:{detail.start_line}
          </div>
          {detail.community && (
            <div style={{ marginBottom: 10 }}>
              <div style={sectionLabel}>COMMUNITY</div>
              <span
                style={{
                  color: colorForCommunity(detail.community, communities),
                  fontSize: 11,
                }}
              >
                {communities.find((c) => c.id === detail.community)?.heuristic_label ??
                  detail.community}
              </span>
            </div>
          )}
          {detail.callees && detail.callees.length > 0 && (
            <div style={{ marginBottom: 8 }}>
              <div style={sectionLabel}>CALLS ({detail.callees.length})</div>
              {detail.callees.slice(0, 10).map((c) => (
                <div key={c.id} style={relItemStyle("#10b981")}>→ {c.name}</div>
              ))}
            </div>
          )}
          {detail.callers && detail.callers.length > 0 && (
            <div>
              <div style={sectionLabel}>CALLED BY ({detail.callers.length})</div>
              {detail.callers.slice(0, 10).map((c) => (
                <div key={c.id} style={relItemStyle("#f59e0b")}>← {c.name}</div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ── Community detail ───────────────────────────────────────────

interface CommunityDetailProps {
  communityId: string;
  communities: RepoMapCommunity[];
  onClose: () => void;
  onSymbolClick: (id: string) => void;
}

function CommunityDetail({
  communityId,
  communities,
  onClose,
  onSymbolClick,
}: CommunityDetailProps) {
  const [detail, setDetail] = useState<CommunityDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const colour = colorForCommunity(communityId, communities);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getCommunityDetail(communityId)
      .then((d) => {
        if (!cancelled) setDetail(d);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [communityId]);

  return (
    <div style={detailBoxStyle}>
      <div style={detailHeaderStyle}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: colour }} />
          <span style={{ color: colour, fontSize: 11, letterSpacing: 1 }}>COMMUNITY</span>
        </div>
        <button onClick={onClose} style={closeBtnStyle}>✕</button>
      </div>
      {loading && <DetailSkeleton />}
      {error && !loading && (
        <div style={{ ...dimText, color: "#f87171" }}>Failed to load: {error}</div>
      )}
      {detail && !loading && (
        <>
          <div style={{ color: "#e2e8f0", fontWeight: 600, fontSize: 13, marginBottom: 6 }}>
            {detail.label}
          </div>
          <div style={{ display: "flex", gap: 16, marginBottom: 10 }}>
            <Stat label="Symbols" value={detail.symbol_count} />
            <Stat label="Cohesion" value={`${(detail.cohesion * 100).toFixed(0)}%`} />
            <Stat label="Edges" value={detail.internal_relationships?.length ?? 0} />
          </div>
          <div style={sectionLabel}>MEMBERS</div>
          <div style={{ maxHeight: 220, overflowY: "auto" }}>
            {detail.members?.slice(0, 40).map((m) => (
              <div
                key={m.id}
                onClick={() => onSymbolClick(m.id)}
                style={{
                  display: "flex", alignItems: "center", gap: 6,
                  padding: "3px 0", cursor: "pointer", borderRadius: 3,
                }}
                className="hover-row"
              >
                <span style={{
                  fontSize: 9, color: colour, background: colour + "22",
                  padding: "1px 4px", borderRadius: 3, flexShrink: 0,
                }}>
                  {m.label[0]}
                </span>
                <span style={{ fontSize: 11, color: "#e2e8f0", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {m.name}
                </span>
              </div>
            ))}
            {detail.members?.length > 40 && (
              <div style={{ color: "#9ca3af", fontSize: 10, paddingTop: 4 }}>
                +{detail.members.length - 40} more
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ color: "#00d4ff", fontSize: 14, fontWeight: 700 }}>{value}</div>
      <div style={{ color: "#9ca3af", fontSize: 9, letterSpacing: 0.5 }}>{label}</div>
    </div>
  );
}

/** Three-bar shimmer placeholder shown while a detail panel is loading. */
function DetailSkeleton() {
  return (
    <div aria-hidden style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={skeletonBar(40)} />
      <div style={skeletonBar(70)} />
      <div style={skeletonBar(55)} />
    </div>
  );
}

function skeletonBar(widthPct: number): React.CSSProperties {
  return {
    width: `${widthPct}%`,
    height: 10,
    borderRadius: 4,
    background:
      "linear-gradient(90deg, rgba(255,255,255,0.04) 0%, rgba(0,212,255,0.10) 50%, rgba(255,255,255,0.04) 100%)",
    backgroundSize: "200% 100%",
    animation: "skeleton-shimmer 1.4s linear infinite",
  };
}

// ── Force Graph Canvas ─────────────────────────────────────────

interface FGNode {
  id: string;
  label: string;
  symbolCount: number;
  colour: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
}

function ForceGraph({ data, onCommunityClick, selectedId }: {
  data: RepoMapSummary;
  onCommunityClick: (id: string) => void;
  selectedId: string | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const nodesRef = useRef<FGNode[]>([]);
  const edgesRef = useRef<{ source: string; target: string }[]>([]);
  const animRef = useRef<number>(0);
  const panRef = useRef({ x: 0, y: 0 });
  const scaleRef = useRef(1);
  const panningRef = useRef(false);
  const lastMouseRef = useRef({ x: 0, y: 0 });
  const hoverRef = useRef<string | null>(null);
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number; label: string; count: number } | null>(null);

  // Build nodes from summary
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const W = canvas.width || 600, H = canvas.height || 400;

    const nodes: FGNode[] = data.communities.map((c, i) => {
      const angle = (i / data.communities.length) * Math.PI * 2;
      const dist = Math.min(W, H) * 0.3;
      return {
        id: c.id,
        label: c.heuristic_label || c.label,
        symbolCount: c.symbol_count,
        colour: colorForIdx(i),
        x: W / 2 + Math.cos(angle) * dist + (Math.random() - 0.5) * 40,
        y: H / 2 + Math.sin(angle) * dist + (Math.random() - 0.5) * 40,
        vx: 0, vy: 0,
        r: Math.max(18, Math.min(52, 10 + Math.sqrt(c.symbol_count) * 4)),
      };
    });

    const edges: { source: string; target: string }[] = [];
    for (const proc of data.processes) {
      for (let i = 0; i < proc.communities.length - 1; i++) {
        edges.push({ source: proc.communities[i], target: proc.communities[i + 1] });
      }
    }

    nodesRef.current = nodes;
    edgesRef.current = edges;
    panRef.current = { x: 0, y: 0 };
    scaleRef.current = 1;
  }, [data]);

  // Simulation + draw loop
  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const resize = () => {
      canvas.width = wrap.clientWidth;
      canvas.height = wrap.clientHeight;
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    let tick = 0;
    const ctx = canvas.getContext("2d")!;

    const drawFrame = () => {
      const nodes = nodesRef.current;
      const edges = edgesRef.current;
      const W = canvas.width, H = canvas.height;
      const cx = W / 2, cy = H / 2;
      const alpha = Math.max(0, 1 - tick / 160);

      // Forces only while sim is running
      if (tick < 200 && nodes.length > 0) {
        for (const a of nodes) {
          a.vx += (cx - a.x) * 0.008 * alpha;
          a.vy += (cy - a.y) * 0.008 * alpha;
          for (const b of nodes) {
            if (b === a) continue;
            const dx = a.x - b.x, dy = a.y - b.y;
            const d = Math.sqrt(dx * dx + dy * dy) || 1;
            const minD = a.r + b.r + 24;
            if (d < minD * 2.2) {
              const f = ((minD * 2.2 - d) / d) * 0.25 * alpha;
              a.vx += dx * f; a.vy += dy * f;
            }
          }
        }
        for (const e of edges) {
          const src = nodes.find((n) => n.id === e.source);
          const tgt = nodes.find((n) => n.id === e.target);
          if (!src || !tgt) continue;
          const dx = tgt.x - src.x, dy = tgt.y - src.y;
          const f = 0.004 * alpha;
          src.vx += dx * f; src.vy += dy * f;
          tgt.vx -= dx * f; tgt.vy -= dy * f;
        }
        for (const n of nodes) {
          n.vx *= 0.72; n.vy *= 0.72;
          n.x += n.vx; n.y += n.vy;
          n.x = Math.max(n.r + 8, Math.min(W - n.r - 8, n.x));
          n.y = Math.max(n.r + 8, Math.min(H - n.r - 8, n.y));
        }
        tick++;
      }

      // Render
      ctx.clearRect(0, 0, W, H);
      ctx.save();
      ctx.translate(panRef.current.x, panRef.current.y);
      ctx.scale(scaleRef.current, scaleRef.current);

      // Edges
      for (const e of edges) {
        const src = nodes.find((n) => n.id === e.source);
        const tgt = nodes.find((n) => n.id === e.target);
        if (!src || !tgt) continue;
        ctx.beginPath();
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(tgt.x, tgt.y);
        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Nodes
      for (const n of nodes) {
        const isSelected = n.id === selectedId;
        const isHovered = n.id === hoverRef.current;

        // Glow
        const grd = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r * 2);
        grd.addColorStop(0, n.colour + (isSelected ? "55" : "33"));
        grd.addColorStop(1, "transparent");
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r * 2, 0, Math.PI * 2);
        ctx.fillStyle = grd;
        ctx.fill();

        // Circle
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = n.colour + (isSelected ? "44" : "22");
        ctx.fill();
        ctx.strokeStyle = n.colour;
        ctx.lineWidth = isSelected ? 2.5 : isHovered ? 2 : 1.5;
        ctx.stroke();

        // Label
        const fontSize = Math.max(9, Math.min(12, n.r * 0.42));
        ctx.font = `${fontSize}px "JetBrains Mono", monospace`;
        ctx.fillStyle = isSelected || isHovered ? "#e2e8f0" : "rgba(201,209,217,0.85)";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        // Trim label to fit
        let lbl = n.label.split("/").pop() ?? n.label;
        const maxW = n.r * 1.7;
        while (lbl.length > 3 && ctx.measureText(lbl).width > maxW) lbl = lbl.slice(0, -1);
        if (lbl !== (n.label.split("/").pop() ?? n.label)) lbl += "…";
        ctx.fillText(lbl, n.x, n.y - 4);

        ctx.font = `9px monospace`;
        ctx.fillStyle = "rgba(139,148,158,0.7)";
        ctx.fillText(`${n.symbolCount}`, n.x, n.y + 8);
      }

      ctx.restore();
      animRef.current = requestAnimationFrame(drawFrame);
    };

    animRef.current = requestAnimationFrame(drawFrame);
    return () => {
      cancelAnimationFrame(animRef.current);
      ro.disconnect();
    };
  }, [data, selectedId]);

  const worldPos = (clientX: number, clientY: number) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return {
      x: (clientX - rect.left - panRef.current.x) / scaleRef.current,
      y: (clientY - rect.top - panRef.current.y) / scaleRef.current,
    };
  };

  const hitNode = (wx: number, wy: number) =>
    nodesRef.current.find((n) => {
      const dx = n.x - wx, dy = n.y - wy;
      return Math.sqrt(dx * dx + dy * dy) <= n.r;
    }) ?? null;

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (panningRef.current) {
      panRef.current.x += e.clientX - lastMouseRef.current.x;
      panRef.current.y += e.clientY - lastMouseRef.current.y;
      lastMouseRef.current = { x: e.clientX, y: e.clientY };
      return;
    }
    const w = worldPos(e.clientX, e.clientY);
    const n = hitNode(w.x, w.y);
    hoverRef.current = n?.id ?? null;
    if (n) {
      const rect = canvasRef.current!.getBoundingClientRect();
      setHoverPos({ x: e.clientX - rect.left + 12, y: e.clientY - rect.top - 28, label: n.label, count: n.symbolCount });
    } else {
      setHoverPos(null);
    }
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    panningRef.current = true;
    lastMouseRef.current = { x: e.clientX, y: e.clientY };
  }, []);

  const handleMouseUp = useCallback((e: React.MouseEvent) => {
    const dx = Math.abs(e.clientX - lastMouseRef.current.x);
    const dy = Math.abs(e.clientY - lastMouseRef.current.y);
    panningRef.current = false;
    if (dx < 4 && dy < 4) {
      const w = worldPos(e.clientX, e.clientY);
      const n = hitNode(w.x, w.y);
      if (n) onCommunityClick(n.id);
    }
  }, [onCommunityClick]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const rect = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    panRef.current.x = mx - (mx - panRef.current.x) * factor;
    panRef.current.y = my - (my - panRef.current.y) * factor;
    scaleRef.current = Math.max(0.3, Math.min(3, scaleRef.current * factor));
  }, []);

  return (
    <div ref={wrapRef} style={{ position: "relative", width: "100%", height: "100%" }}>
      <canvas
        ref={canvasRef}
        style={{ display: "block", cursor: "pointer" }}
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => { panningRef.current = false; hoverRef.current = null; setHoverPos(null); }}
        onWheel={handleWheel}
      />
      {hoverPos && (
        <div style={{
          position: "absolute",
          left: hoverPos.x, top: hoverPos.y,
          background: "rgba(10,15,30,0.92)",
          border: "1px solid rgba(0,212,255,0.3)",
          borderRadius: 6, padding: "5px 10px",
          fontSize: 11, pointerEvents: "none",
          backdropFilter: "blur(8px)",
          zIndex: 10,
        }}>
          <span style={{ color: "#00d4ff", fontWeight: 600 }}>{hoverPos.label}</span>
          <span style={{ color: "#9ca3af", marginLeft: 8 }}>{hoverPos.count} symbols</span>
        </div>
      )}
    </div>
  );
}

// ── Main view ──────────────────────────────────────────────────

type Tab = "Graph" | "Processes";

export default function RepoMapView() {
  const [data, setData] = useState<RepoMapSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("Graph");
  const [search, setSearch] = useState("");

  const selectedCommunity = useStore((s) => s.selectedCommunity);
  const setSelectedCommunity = useStore((s) => s.setSelectedCommunity);
  const selectedSymbol = useStore((s) => s.selectedSymbol);
  const setSelectedSymbol = useStore((s) => s.setSelectedSymbol);

  useEffect(() => {
    getRepoMap()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const communities = data?.communities ?? [];
  const processes = data?.processes ?? [];
  const stats = data?.stats;

  const filteredCommunities = communities.filter((c) =>
    !search || (c.heuristic_label || c.label).toLowerCase().includes(search.toLowerCase())
  );
  const filteredProcesses = processes.filter((p) =>
    !search || p.label.toLowerCase().includes(search.toLowerCase())
  );

  if (loading) return (
    <div style={centerStyle}>
      <div style={{ color: "#00d4ff", fontSize: 13, fontFamily: "monospace" }}>
        Building repo map…
      </div>
    </div>
  );

  if (error) return (
    <div style={centerStyle}>
      <div style={{ color: "#ef4444", fontSize: 13, textAlign: "center" }}>
        {error.includes("503") || error.includes("not available")
          ? "Repo map not built yet. Index a repository first."
          : error}
      </div>
    </div>
  );

  if (!data) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      {/* Top bar */}
      <div style={{
        display: "flex", alignItems: "center", gap: 8, padding: "10px 16px",
        borderBottom: "1px solid rgba(255,255,255,0.06)", flexShrink: 0,
      }}>
        {(["Graph", "Processes"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: "3px 14px", borderRadius: 6, border: "1px solid",
              borderColor: tab === t ? "var(--cyber-cyan)" : "rgba(255,255,255,0.1)",
              background: tab === t ? "rgba(0,212,255,0.1)" : "transparent",
              color: tab === t ? "var(--cyber-cyan)" : "#9ca3af",
              cursor: "pointer", fontSize: 11,
              fontFamily: '"JetBrains Mono", monospace', letterSpacing: 1,
            }}
          >
            {t.toUpperCase()}
          </button>
        ))}

        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search…"
          style={{
            marginLeft: 8,
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 5, padding: "3px 10px",
            color: "#e2e8f0", fontSize: 11, outline: "none", width: 160,
          }}
        />

        <div style={{ marginLeft: "auto", color: "#9ca3af", fontSize: 11, fontFamily: "monospace" }}>
          {stats?.total_communities ?? communities.length} communities
          {" · "}
          {stats?.total_processes ?? processes.length} processes
          {" · "}
          {stats?.total_nodes ?? "?"} symbols
        </div>
      </div>

      {/* Body */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

        {/* Left sidebar */}
        <div style={{
          width: 220, flexShrink: 0, display: "flex", flexDirection: "column",
          borderRight: "1px solid rgba(255,255,255,0.06)", overflow: "hidden",
        }}>
          <div style={{ flex: 1, overflowY: "auto", padding: "8px 6px" }}>
            {tab === "Graph" && (
              <>
                <div style={listHeaderStyle}>
                  COMMUNITIES ({filteredCommunities.length})
                </div>
                {filteredCommunities.map((c, i) => {
                  const realIdx = communities.indexOf(c);
                  const colour = colorForIdx(realIdx);
                  const isActive = selectedCommunity === c.id;
                  return (
                    <div
                      key={c.id}
                      onClick={() => {
                        setSelectedSymbol(null);
                        setSelectedCommunity(isActive ? null : c.id);
                      }}
                      style={{
                        display: "flex", alignItems: "center", gap: 8,
                        padding: "6px 8px", borderRadius: 6, marginBottom: 2,
                        cursor: "pointer",
                        background: isActive ? colour + "18" : "transparent",
                        border: `1px solid ${isActive ? colour + "44" : "transparent"}`,
                      }}
                    >
                      <div style={{ width: 7, height: 7, borderRadius: "50%", background: colour, flexShrink: 0 }} />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ color: "#e2e8f0", fontSize: 12, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {c.heuristic_label || c.label}
                        </div>
                        <div style={{ color: "#9ca3af", fontSize: 10 }}>
                          {c.symbol_count} symbols · {(c.cohesion * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                  );
                })}
              </>
            )}

            {tab === "Processes" && (
              <>
                <div style={listHeaderStyle}>
                  PROCESSES ({filteredProcesses.length})
                </div>
                {filteredProcesses.map((p) => {
                  const colour = p.process_type === "cross_community" ? "#f97316" : "#00d4ff";
                  return (
                    <div
                      key={p.id}
                      style={{
                        display: "flex", alignItems: "flex-start", gap: 8,
                        padding: "6px 8px", borderRadius: 6, marginBottom: 2,
                      }}
                    >
                      <div style={{ width: 6, height: 6, borderRadius: "50%", background: colour, flexShrink: 0, marginTop: 4 }} />
                      <div>
                        <div style={{ color: "#e2e8f0", fontSize: 11, fontFamily: "monospace", wordBreak: "break-all" }}>
                          {p.label}
                        </div>
                        <div style={{ color: "#9ca3af", fontSize: 10 }}>
                          {p.step_count} steps · {p.process_type.replace("_", " ")}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </>
            )}
          </div>

          {/* Detail panel in sidebar bottom */}
          {(selectedCommunity || selectedSymbol) && (
            <div style={{ flexShrink: 0, borderTop: "1px solid rgba(255,255,255,0.06)", maxHeight: "55%", overflowY: "auto" }}>
              {selectedSymbol ? (
                <SymbolDetail
                  symbolId={selectedSymbol}
                  communities={communities}
                  onClose={() => setSelectedSymbol(null)}
                />
              ) : selectedCommunity ? (
                <CommunityDetail
                  communityId={selectedCommunity}
                  communities={communities}
                  onClose={() => setSelectedCommunity(null)}
                  onSymbolClick={(id) => { setSelectedSymbol(id); }}
                />
              ) : null}
            </div>
          )}
        </div>

        {/* Main area */}
        <div style={{ flex: 1, overflow: "hidden", position: "relative" }}>
          {tab === "Graph" && (
            communities.length === 0 ? (
              <div style={centerStyle}>
                <div style={{ color: "#9ca3af", fontSize: 13, textAlign: "center" }}>
                  No communities detected.<br />
                  <span style={{ fontSize: 11 }}>Index a larger repository to see clusters.</span>
                </div>
              </div>
            ) : (
              <ForceGraph
                data={data}
                onCommunityClick={(id) => {
                  setSelectedSymbol(null);
                  setSelectedCommunity(id === selectedCommunity ? null : id);
                }}
                selectedId={selectedCommunity}
              />
            )
          )}

          {tab === "Processes" && (
            <div style={{ height: "100%", overflowY: "auto", padding: 16 }}>
              <ProcessFlow processes={filteredProcesses} communities={communities} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Styles ─────────────────────────────────────────────────────

const centerStyle: React.CSSProperties = {
  display: "flex", alignItems: "center", justifyContent: "center",
  height: "100%",
};

const listHeaderStyle: React.CSSProperties = {
  color: "#9ca3af", fontSize: 10, letterSpacing: 1, padding: "4px 8px 6px",
  fontFamily: '"JetBrains Mono", monospace',
};

const detailBoxStyle: React.CSSProperties = {
  padding: "12px 12px",
};

const detailHeaderStyle: React.CSSProperties = {
  display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10,
};

const closeBtnStyle: React.CSSProperties = {
  background: "transparent", border: "none", color: "#9ca3af",
  cursor: "pointer", fontSize: 13, padding: "0 2px", lineHeight: 1,
};

const sectionLabel: React.CSSProperties = {
  color: "#9ca3af", fontSize: 9, letterSpacing: 1,
  fontFamily: "monospace", marginBottom: 4,
};

const dimText: React.CSSProperties = {
  color: "#9ca3af", fontSize: 12,
};

function relItemStyle(colour: string): React.CSSProperties {
  return {
    color: colour, fontSize: 11, padding: "2px 0",
    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
  };
}
