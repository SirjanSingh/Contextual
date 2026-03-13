/**
 * Query input with glassmorphic card, auto-expand, neon cursor
 */
import { useState, useRef, useCallback, KeyboardEvent } from "react";
import { motion } from "framer-motion";
import { useStore } from "../../store/useStore";
import { queryBackend } from "../../api/client";

export default function QueryInput() {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const addMessage = useStore((s) => s.addMessage);
  const setIsQuerying = useStore((s) => s.setIsQuerying);
  const isQuerying = useStore((s) => s.isQuerying);
  const indexStatus = useStore((s) => s.indexStatus);
  const addActivity = useStore((s) => s.addActivity);
  const triggerErrorShake = useStore((s) => s.triggerErrorShake);

  const handleSubmit = useCallback(async () => {
    const q = value.trim();
    if (!q || isQuerying) return;

    // Add user message
    addMessage({
      id: Date.now().toString(),
      role: "user",
      content: q,
      timestamp: Date.now(),
    });
    setValue("");
    setIsQuerying(true);
    addActivity(`Query submitted: "${q.slice(0, 50)}..."`);

    try {
      const result = await queryBackend(q);
      addMessage({
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: result.answer,
        sources: result.sources,
        timestamp: Date.now(),
      });
      addActivity(`Response received (${result.sources.length} sources)`);
    } catch (err: any) {
      triggerErrorShake();
      addMessage({
        id: (Date.now() + 1).toString(),
        role: "system",
        content: `⚠ NEURAL PATHWAY OVERLOAD — ${err.message || "Connection lost"}. Even AI needs a moment.`,
        timestamp: Date.now(),
      });
      addActivity(`ERROR: ${err.message}`);
    } finally {
      setIsQuerying(false);
    }
  }, [value, isQuerying]);

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const disabled = indexStatus !== "ready";

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 1, duration: 0.8, ease: "easeOut" }}
      className="glass-card-glow"
      style={{
        padding: "16px 20px",
        display: "flex",
        gap: 12,
        alignItems: "flex-end",
      }}
    >
      <div style={{ flex: 1, position: "relative" }}>
        <textarea
          ref={textareaRef}
          className="cyber-input"
          style={{
            width: "100%",
            resize: "none",
            minHeight: "44px",
            maxHeight: "120px",
            lineHeight: "1.5",
            fontSize: "14px",
          }}
          placeholder={
            disabled
              ? "// Upload a repository to begin..."
              : "// Ask about your codebase..."
          }
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            // Auto expand
            const el = e.target;
            el.style.height = "auto";
            el.style.height = Math.min(el.scrollHeight, 120) + "px";
          }}
          onKeyDown={handleKeyDown}
          disabled={disabled || isQuerying}
          rows={1}
          aria-label="Query input"
        />
        {isQuerying && (
          <div
            style={{
              position: "absolute",
              bottom: 4,
              right: 8,
              fontSize: "10px",
              fontFamily: '"JetBrains Mono", monospace',
              color: "var(--warning-amber)",
              animation: "pulse-glow 1.5s infinite",
            }}
          >
            PROCESSING...
          </div>
        )}
      </div>

      {/* Send button */}
      <motion.button
        whileHover={{ scale: 1.08, rotate: 15 }}
        whileTap={{ scale: 0.95 }}
        onClick={handleSubmit}
        disabled={!value.trim() || isQuerying || disabled}
        style={{
          width: 44,
          height: 44,
          border: "1px solid var(--cyber-cyan-dim)",
          borderRadius: "8px",
          background:
            value.trim() && !disabled
              ? "linear-gradient(135deg, rgba(0, 247, 255, 0.15), rgba(139, 92, 246, 0.15))"
              : "rgba(0, 0, 0, 0.3)",
          color:
            value.trim() && !disabled ? "var(--cyber-cyan)" : "var(--text-dim)",
          cursor: value.trim() && !disabled ? "pointer" : "not-allowed",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "18px",
          transition: "all 0.3s",
          boxShadow:
            value.trim() && !disabled
              ? "0 0 15px rgba(0, 247, 255, 0.2)"
              : "none",
          flexShrink: 0,
        }}
        aria-label="Send query"
      >
        ▶
      </motion.button>
    </motion.div>
  );
}
