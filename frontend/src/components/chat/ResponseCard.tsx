/**
 * Response card with typewriter effect and syntax highlighting
 */
import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import type { ChatMessage } from "../../store/useStore";

/* ───── Typewriter hook ───── */
function useTypewriter(text: string, speed: number = 12) {
  const [displayed, setDisplayed] = useState("");
  const [done, setDone] = useState(false);

  useEffect(() => {
    setDisplayed("");
    setDone(false);
    let i = 0;
    const id = setInterval(() => {
      i += 1;
      if (i >= text.length) {
        setDisplayed(text);
        setDone(true);
        clearInterval(id);
      } else {
        setDisplayed(text.slice(0, i));
      }
    }, speed);
    return () => clearInterval(id);
  }, [text, speed]);

  return { displayed, done };
}

/* ───── Simple code block detector ───── */
function renderContent(text: string) {
  const parts = text.split(/(```[\s\S]*?```)/g);
  return parts.map((part, i) => {
    if (part.startsWith("```")) {
      const lines = part.split("\n");
      const lang = lines[0].replace("```", "").trim();
      const code = lines.slice(1, -1).join("\n");
      return (
        <div
          key={i}
          className="code-block-neon"
          style={{
            margin: "12px 0",
            padding: "14px 16px",
            position: "relative",
            overflow: "auto",
          }}
        >
          {lang && (
            <div
              style={{
                position: "absolute",
                top: 6,
                right: 10,
                fontSize: "9px",
                fontFamily: '"JetBrains Mono", monospace',
                color: "var(--neural-purple)",
                textTransform: "uppercase",
                letterSpacing: "1px",
              }}
            >
              {lang}
            </div>
          )}
          <pre
            style={{
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: "13px",
              lineHeight: "1.6",
              color: "var(--cyber-cyan)",
              margin: 0,
              whiteSpace: "pre-wrap",
            }}
          >
            {code}
          </pre>
        </div>
      );
    }

    // Bold text
    const boldParts = part.split(/(\*\*.*?\*\*)/g);
    return (
      <span key={i}>
        {boldParts.map((bp, j) => {
          if (bp.startsWith("**") && bp.endsWith("**")) {
            return (
              <strong key={j} style={{ color: "var(--cyber-cyan)" }}>
                {bp.slice(2, -2)}
              </strong>
            );
          }
          // Inline code
          const inlineParts = bp.split(/(`[^`]+`)/g);
          return inlineParts.map((ip, k) => {
            if (ip.startsWith("`") && ip.endsWith("`")) {
              return (
                <code
                  key={k}
                  style={{
                    background: "rgba(0, 247, 255, 0.08)",
                    padding: "1px 6px",
                    borderRadius: "4px",
                    fontFamily: '"JetBrains Mono", monospace',
                    fontSize: "0.9em",
                    color: "var(--cyber-cyan)",
                    border: "1px solid rgba(0, 247, 255, 0.1)",
                  }}
                >
                  {ip.slice(1, -1)}
                </code>
              );
            }
            return ip;
          });
        })}
      </span>
    );
  });
}

/* ───── Component ───── */
export default function ResponseCard({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const { displayed, done } = useTypewriter(message.content, isUser ? 0 : 8);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        marginBottom: 16,
      }}
    >
      <div
        className={isUser ? "glass-card" : "glass-card-glow"}
        style={{
          maxWidth: isUser ? "70%" : "85%",
          padding: "14px 18px",
          borderColor: isSystem
            ? "var(--error-red)"
            : isUser
              ? "var(--glass-border)"
              : "var(--cyber-cyan-dim)",
          boxShadow: isSystem ? "0 0 15px rgba(239, 68, 68, 0.15)" : undefined,
        }}
      >
        {/* Role label */}
        <div
          style={{
            fontFamily: '"Orbitron", sans-serif',
            fontSize: "9px",
            letterSpacing: "2px",
            marginBottom: 8,
            color: isUser
              ? "var(--hot-magenta)"
              : isSystem
                ? "var(--error-red)"
                : "var(--neural-purple)",
          }}
        >
          {isUser ? "OPERATOR" : isSystem ? "SYSTEM ALERT" : "CONTEXTUAL AI"}
        </div>

        {/* Content */}
        <div
          style={{
            fontFamily: isUser
              ? '"Inter", sans-serif'
              : '"JetBrains Mono", monospace',
            fontSize: isUser ? "14px" : "13px",
            lineHeight: "1.7",
            color: isSystem ? "var(--error-red)" : "var(--text-primary)",
            whiteSpace: "pre-wrap",
          }}
        >
          {renderContent(isUser ? message.content : displayed)}
          {!done && !isUser && <span className="typewriter-cursor" />}
        </div>

        {/* Sources */}
        {done && message.sources && message.sources.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            style={{
              marginTop: 12,
              paddingTop: 10,
              borderTop: "1px solid var(--glass-border)",
            }}
          >
            <div
              style={{
                fontFamily: '"Orbitron", sans-serif',
                fontSize: "8px",
                letterSpacing: "2px",
                color: "var(--text-dim)",
                marginBottom: 6,
              }}
            >
              SOURCES
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {message.sources.map((s, i) => (
                <span
                  key={i}
                  style={{
                    padding: "2px 8px",
                    borderRadius: "4px",
                    background: "rgba(139, 92, 246, 0.1)",
                    border: "1px solid rgba(139, 92, 246, 0.2)",
                    fontFamily: '"JetBrains Mono", monospace',
                    fontSize: "10px",
                    color: "var(--neural-purple)",
                  }}
                >
                  {s}
                </span>
              ))}
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}
