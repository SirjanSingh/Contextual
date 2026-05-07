/**
 * ProcessFlow — List and detail view for detected execution processes.
 */
import { useState, useEffect } from "react";
import { getProcessDetail } from "../../api/client";

const COMMUNITY_COLOURS = [
  "#00d4ff", "#7c3aed", "#10b981", "#f59e0b", "#ef4444",
  "#ec4899", "#3b82f6", "#14b8a6", "#f97316", "#8b5cf6",
];

function commColour(commId: string, communities: any[]): string {
  const idx = communities.findIndex((c) => c.id === commId);
  return COMMUNITY_COLOURS[idx % COMMUNITY_COLOURS.length] ?? "#4b5563";
}

interface Process {
  id: string;
  label: string;
  process_type: string;
  step_count: number;
  communities: string[];
}

interface ProcessFlowProps {
  processes: Process[];
  communities: any[];
}

function ProcessDetail({ processId, communities }: { processId: string; communities: any[] }) {
  const [detail, setDetail] = useState<any>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    getProcessDetail(processId)
      .then(setDetail)
      .catch(() => null)
      .finally(() => setLoaded(true));
  }, [processId]);

  if (!detail) return (
    <div style={{ color: "#9ca3af", fontSize: 11, padding: "8px 0" }}>Loading…</div>
  );

  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ display: "flex", gap: 6, alignItems: "center", overflowX: "auto", paddingBottom: 6 }}>
        {detail.steps.map((s: any, i: number) => {
          const stepComm = detail.communities?.[0];
          const colour = commColour(stepComm, communities);
          return (
            <div key={s.node_id} style={{ display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
              <div
                style={{
                  background: "rgba(0,212,255,0.08)",
                  border: `1px solid ${colour}44`,
                  borderRadius: 6,
                  padding: "4px 10px",
                  textAlign: "center",
                  minWidth: 80,
                }}
              >
                <div style={{ color: "#9ca3af", fontSize: 9, marginBottom: 2 }}>step {s.step}</div>
                <div style={{ color: "#e2e8f0", fontSize: 11, fontFamily: "monospace" }}>
                  {s.name.split(".").pop()}
                </div>
                <div style={{ color: "#9ca3af", fontSize: 9, marginTop: 2 }}>
                  {s.file_path.split("/").pop()}
                </div>
              </div>
              {i < detail.steps.length - 1 && (
                <div style={{ color: "#4b5563", fontSize: 14 }}>→</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function ProcessFlow({ processes, communities }: ProcessFlowProps) {
  const [expanded, setExpanded] = useState<string | null>(null);

  if (processes.length === 0) {
    return (
      <div style={{ color: "#9ca3af", textAlign: "center", paddingTop: 40, fontSize: 13 }}>
        No execution flows detected.<br />
        <span style={{ fontSize: 11 }}>Index a codebase with function calls to see flows.</span>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {processes.map((proc) => {
        const isOpen = expanded === proc.id;
        const typeColour = proc.process_type === "cross_community" ? "#7c3aed" : "#10b981";
        return (
          <div
            key={proc.id}
            className="glass-card"
            style={{ padding: 12, cursor: "pointer", borderRadius: 8 }}
            onClick={() => setExpanded(isOpen ? null : proc.id)}
          >
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <div style={{
                width: 6, height: 6, borderRadius: "50%",
                background: typeColour, flexShrink: 0,
              }} />
              <div style={{ flex: 1 }}>
                <div style={{ color: "#e2e8f0", fontSize: 13, fontFamily: "monospace" }}>
                  {proc.label}
                </div>
                <div style={{ color: "#9ca3af", fontSize: 10, marginTop: 2 }}>
                  {proc.step_count} steps · {proc.process_type.replace("_", " ")}
                  {proc.communities.length > 0 && (
                    <span> · </span>
                  )}
                  {proc.communities.map((cid) => {
                    const comm = communities.find((c) => c.id === cid);
                    return comm ? (
                      <span
                        key={cid}
                        style={{
                          color: commColour(cid, communities),
                          marginRight: 4,
                          fontSize: 10,
                        }}
                      >
                        {comm.heuristic_label}
                      </span>
                    ) : null;
                  })}
                </div>
              </div>
              <div style={{ color: "#9ca3af", fontSize: 11 }}>{isOpen ? "▲" : "▼"}</div>
            </div>

            {isOpen && (
              <ProcessDetail processId={proc.id} communities={communities} />
            )}
          </div>
        );
      })}
    </div>
  );
}
