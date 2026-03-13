/**
 * Header HUD with glitchy animated logo and system status
 */
import { useEffect, useState, useRef } from "react";
import { motion } from "framer-motion";
import { useStore } from "../../store/useStore";

export default function Header() {
  const backendStatus = useStore((s) => s.backendStatus);
  const indexStatus = useStore((s) => s.indexStatus);
  const [time, setTime] = useState("");

  useEffect(() => {
    const tick = () =>
      setTime(new Date().toLocaleTimeString("en-US", { hour12: false }));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  const statusColor = {
    connecting: "#fbbf24",
    online: "#34d399",
    offline: "#ef4444",
  }[backendStatus];

  const indexColor = {
    none: "#475569",
    building: "#fbbf24",
    ready: "#34d399",
    error: "#ef4444",
  }[indexStatus];

  return (
    <header
      className="glass-card scan-lines"
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "12px 24px",
        borderRadius: "0 0 12px 12px",
        borderTop: "none",
        position: "relative",
        zIndex: 50,
      }}
    >
      {/* Logo */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.5, duration: 0.8, ease: "easeOut" }}
        style={{ display: "flex", alignItems: "center", gap: 12 }}
      >
        <div
          style={{
            width: 32,
            height: 32,
            border: "2px solid var(--cyber-cyan)",
            borderRadius: "6px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 0 12px var(--cyber-cyan-dim)",
            transform: "rotate(45deg)",
          }}
        >
          <span
            style={{
              transform: "rotate(-45deg)",
              color: "var(--cyber-cyan)",
              fontFamily: '"Orbitron", sans-serif',
              fontSize: "14px",
              fontWeight: 700,
            }}
          >
            C
          </span>
        </div>
        <h1
          className="glitch-text"
          data-text="CONTEXTUAL"
          style={{
            fontFamily: '"Orbitron", sans-serif',
            fontSize: "18px",
            fontWeight: 700,
            letterSpacing: "3px",
            color: "var(--cyber-cyan)",
            textShadow: "0 0 10px var(--cyber-cyan-dim)",
          }}
        >
          CONTEXTUAL
        </h1>
        <span
          style={{
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: "10px",
            color: "var(--text-dim)",
            letterSpacing: "2px",
            marginTop: 4,
          }}
        >
          NEURAL CODE NEXUS
        </span>
      </motion.div>

      {/* System Status HUD */}
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.7, duration: 0.8 }}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 24,
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: "11px",
          color: "var(--text-secondary)",
        }}
      >
        {/* System time */}
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ color: "var(--text-dim)" }}>SYS</span>
          <span
            style={{
              color: "var(--cyber-cyan)",
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {time}
          </span>
        </div>

        {/* Backend status */}
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div
            className={backendStatus === "online" ? "pulse-dot" : ""}
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: statusColor,
              boxShadow: `0 0 8px ${statusColor}`,
            }}
          />
          <span>{backendStatus.toUpperCase()}</span>
        </div>

        {/* Index status */}
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ color: "var(--text-dim)" }}>IDX</span>
          <div
            style={{
              width: 8,
              height: 8,
              borderRadius: "2px",
              background: indexColor,
              boxShadow: `0 0 8px ${indexColor}`,
            }}
          />
          <span>{indexStatus.toUpperCase()}</span>
        </div>

        {/* Model badge */}
        <div
          style={{
            padding: "3px 10px",
            border: "1px solid var(--neural-purple-dim)",
            borderRadius: "4px",
            color: "var(--neural-purple)",
            fontSize: "10px",
            letterSpacing: "1px",
          }}
        >
          GEMINI 2.0
        </div>
      </motion.div>
    </header>
  );
}
