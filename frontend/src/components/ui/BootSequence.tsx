/**
 * Boot sequence overlay — anime-inspired system initialization
 */
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const BOOT_LINES = [
  { text: "INITIALIZING NEURAL CODE NEXUS...", delay: 0 },
  { text: "LOADING CONTEXTUAL AI CORE v1.0", delay: 300 },
  { text: "ESTABLISHING GEMINI NEURAL LINK", delay: 600 },
  { text: "CALIBRATING VECTOR SPACE [768-DIM]", delay: 900 },
  { text: "FAISS INDEX ENGINE — STANDBY", delay: 1200 },
  { text: "HYBRID SEARCH MODULE — ACTIVE", delay: 1400 },
  { text: "AST PARSER — LOADED", delay: 1600 },
  { text: "RERANKER — ONLINE", delay: 1800 },
  { text: "QUERY EXPANSION ENGINE — READY", delay: 2000 },
  { text: "> ALL SYSTEMS NOMINAL", delay: 2400 },
  { text: "> NEURAL LINK READY", delay: 2700 },
];

export default function BootSequence({
  onComplete,
}: {
  onComplete: () => void;
}) {
  const [visibleLines, setVisibleLines] = useState<number[]>([]);
  const [complete, setComplete] = useState(false);

  useEffect(() => {
    const timeouts: ReturnType<typeof setTimeout>[] = [];

    BOOT_LINES.forEach((line, i) => {
      timeouts.push(
        setTimeout(() => {
          setVisibleLines((prev) => [...prev, i]);
        }, line.delay),
      );
    });

    // Complete after all lines
    timeouts.push(
      setTimeout(() => {
        setComplete(true);
      }, 3200),
    );

    timeouts.push(
      setTimeout(() => {
        onComplete();
      }, 3800),
    );

    return () => timeouts.forEach(clearTimeout);
  }, [onComplete]);

  return (
    <AnimatePresence>
      {!complete ? (
        <motion.div
          exit={{ opacity: 0, scale: 1.02 }}
          transition={{ duration: 0.6 }}
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 1000,
            background: "var(--deep-space)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 4,
          }}
        >
          {/* Scan lines */}
          <div
            className="scan-lines"
            style={{ position: "absolute", inset: 0 }}
          />

          {/* Logo */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            style={{ marginBottom: 32 }}
          >
            <div
              style={{
                fontFamily: '"Orbitron", sans-serif',
                fontSize: "32px",
                fontWeight: 900,
                letterSpacing: "8px",
                color: "var(--cyber-cyan)",
                textShadow:
                  "0 0 20px var(--cyber-cyan-dim), 0 0 40px rgba(0, 247, 255, 0.2)",
              }}
            >
              CONTEXTUAL
            </div>
            <div
              style={{
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: "10px",
                letterSpacing: "6px",
                color: "var(--text-dim)",
                textAlign: "center",
                marginTop: 4,
              }}
            >
              NEURAL CODE NEXUS
            </div>
          </motion.div>

          {/* Boot lines */}
          <div
            style={{
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: "11px",
              lineHeight: 1.8,
              color: "var(--text-dim)",
              width: 400,
            }}
          >
            {BOOT_LINES.map((line, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={visibleLines.includes(i) ? { opacity: 1, x: 0 } : {}}
                transition={{ duration: 0.2 }}
                style={{
                  color: line.text.startsWith(">")
                    ? "var(--success-green)"
                    : "var(--text-dim)",
                }}
              >
                <span style={{ color: "var(--cyber-cyan)", opacity: 0.4 }}>
                  {"// "}
                </span>
                {line.text}
              </motion.div>
            ))}
          </div>

          {/* Loading bar */}
          <div
            style={{
              width: 400,
              height: 2,
              background: "rgba(255,255,255,0.05)",
              borderRadius: 2,
              marginTop: 24,
              overflow: "hidden",
            }}
          >
            <motion.div
              initial={{ width: "0%" }}
              animate={{ width: "100%" }}
              transition={{ duration: 3, ease: "easeInOut" }}
              style={{
                height: "100%",
                background:
                  "linear-gradient(90deg, var(--cyber-cyan), var(--neural-purple))",
                boxShadow: "0 0 10px var(--cyber-cyan)",
              }}
            />
          </div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
