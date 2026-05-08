/**
 * Activity feed — terminal-style scrolling log
 */
import { motion } from "framer-motion";
import { useStore } from "../../store/useStore";

export default function ActivityFeed() {
  const activityLog = useStore((s) => s.activityLog);

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.9, duration: 0.6 }}
      className="glass-card scan-lines"
      style={{
        width: 260,
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        position: "relative",
        flexShrink: 0,
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "12px 16px 8px",
          borderBottom: "1px solid var(--glass-border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span
          style={{
            fontFamily: '"Orbitron", sans-serif',
            fontSize: "9px",
            letterSpacing: "2px",
            color: "var(--text-dim)",
          }}
        >
          ACTIVITY LOG
        </span>
        <div
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background:
              activityLog.length > 0
                ? "var(--success-green)"
                : "var(--text-dim)",
            boxShadow:
              activityLog.length > 0 ? "0 0 6px var(--success-green)" : "none",
          }}
        />
      </div>

      {/* Log entries */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "8px 12px",
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: "10px",
          lineHeight: 1.8,
          color: "var(--text-dim)",
        }}
      >
        {activityLog.length === 0 && (
          <div
            style={{
              textAlign: "center",
              padding: "20px 0",
              color: "var(--text-dim)",
              fontSize: "10px",
            }}
          >
            {"> System idle\n> Awaiting commands...".split("\n").map((l, i) => (
              <div key={i}>{l}</div>
            ))}
          </div>
        )}
        {activityLog.map((entry, i) => (
          <motion.div
            key={`${i}-${entry}`}
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            style={{
              borderBottom: "1px solid rgba(255,255,255,0.03)",
              paddingBottom: 4,
              marginBottom: 4,
              color: entry.includes("ERROR")
                ? "var(--error-red)"
                : entry.includes("success")
                  ? "var(--success-green)"
                  : "var(--text-dim)",
              wordBreak: "break-all",
            }}
          >
            <span style={{ color: "var(--cyber-cyan)", opacity: 0.4 }}>
              {"> "}
            </span>
            {entry}
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
