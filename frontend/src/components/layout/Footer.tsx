/**
 * Footer stats bar with real-time metrics
 */
import { motion } from "framer-motion";
import { useStore } from "../../store/useStore";

export default function Footer() {
  const indexStatus = useStore((s) => s.indexStatus);
  const indexInfo = useStore((s) => s.indexInfo);
  const messages = useStore((s) => s.messages);

  const totalQueries = messages.filter((m) => m.role === "user").length;
  const chunks = (indexInfo.total_chunks as number) || 0;
  const files = (indexInfo.total_files as number) || 0;

  return (
    <motion.footer
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 1.2, duration: 0.6 }}
      className="glass-card"
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "8px 24px",
        borderRadius: "12px 12px 0 0",
        borderBottom: "none",
        zIndex: 50,
        fontFamily: '"JetBrains Mono", monospace',
        fontSize: "10px",
        color: "var(--text-dim)",
        letterSpacing: "1px",
      }}
    >
      <div style={{ display: "flex", gap: 24, alignItems: "center" }}>
        <Metric
          label="QUERIES"
          value={totalQueries}
          color="var(--cyber-cyan)"
        />
        <Metric label="FILES" value={files} color="var(--neural-purple)" />
        <Metric label="CHUNKS" value={chunks} color="var(--hot-magenta)" />
      </div>

      <div style={{ display: "flex", gap: 24, alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span>INDEX</span>
          <div
            style={{
              padding: "2px 8px",
              borderRadius: "3px",
              background:
                indexStatus === "ready"
                  ? "rgba(52, 211, 153, 0.15)"
                  : indexStatus === "building"
                    ? "rgba(251, 191, 36, 0.15)"
                    : "rgba(71, 85, 105, 0.15)",
              color:
                indexStatus === "ready"
                  ? "var(--success-green)"
                  : indexStatus === "building"
                    ? "var(--warning-amber)"
                    : "var(--text-dim)",
              fontSize: "9px",
              fontWeight: 600,
            }}
          >
            {indexStatus.toUpperCase()}
          </div>
        </div>
        <span style={{ color: "var(--text-dim)", fontSize: "9px" }}>
          CONTEXTUAL v1.0 — NEURAL CODE NEXUS
        </span>
      </div>
    </motion.footer>
  );
}

function Metric({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span>{label}</span>
      <span
        style={{
          color,
          fontWeight: 600,
          fontSize: "12px",
          textShadow: `0 0 8px ${color}44`,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </span>
    </div>
  );
}
