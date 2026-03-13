/**
 * Chat view — scrollable conversation container
 */
import { useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useStore } from "../../store/useStore";
import ResponseCard from "./ResponseCard";

export default function ChatView() {
  const messages = useStore((s) => s.messages);
  const isQuerying = useStore((s) => s.isQuerying);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div
      ref={scrollRef}
      style={{
        flex: 1,
        overflowY: "auto",
        padding: "20px",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {messages.length === 0 && (
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 16,
            opacity: 0.5,
          }}
        >
          <motion.div
            animate={{ rotate: [0, 360] }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            style={{
              width: 60,
              height: 60,
              border: "1px solid var(--cyber-cyan-dim)",
              borderRadius: "12px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <div
              style={{
                width: 30,
                height: 30,
                border: "1px solid var(--hot-magenta-dim)",
                borderRadius: "6px",
                transform: "rotate(45deg)",
              }}
            />
          </motion.div>
          <div
            style={{
              fontFamily: '"Orbitron", sans-serif',
              fontSize: "12px",
              letterSpacing: "3px",
              color: "var(--text-dim)",
            }}
          >
            AWAITING NEURAL QUERY
          </div>
          <div
            style={{
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: "11px",
              color: "var(--text-dim)",
              textAlign: "center",
              maxWidth: 400,
              lineHeight: 1.6,
            }}
          >
            Upload a repository and ask questions about your codebase. I can
            analyze architecture, trace function calls, and explain code
            patterns.
          </div>
        </div>
      )}

      <AnimatePresence>
        {messages.map((msg) => (
          <ResponseCard key={msg.id} message={msg} />
        ))}
      </AnimatePresence>

      {/* Thinking indicator */}
      {isQuerying && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "12px 16px",
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: "12px",
            color: "var(--neural-purple)",
          }}
        >
          <motion.span
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            ◉
          </motion.span>
          Neural pathways processing...
          <motion.div style={{ display: "flex", gap: 3 }}>
            {[0, 1, 2].map((i) => (
              <motion.span
                key={i}
                animate={{ opacity: [0.2, 1, 0.2] }}
                transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.2 }}
                style={{ color: "var(--cyber-cyan)" }}
              >
                ●
              </motion.span>
            ))}
          </motion.div>
        </motion.div>
      )}
    </div>
  );
}
