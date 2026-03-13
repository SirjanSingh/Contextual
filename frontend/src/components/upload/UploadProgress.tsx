/**
 * Upload progress — 5-phase epic loading animation
 */
import { motion, AnimatePresence } from "framer-motion";
import { useStore } from "../../store/useStore";

const STAGES: Record<string, { label: string; color: string; phase: string }> =
  {
    uploading: {
      label: "UPLOADING REPOSITORY...",
      color: "var(--cyber-cyan)",
      phase: "PHASE 1 — DATA TRANSMISSION",
    },
    scanning: {
      label: "SCANNING FILE STRUCTURE...",
      color: "var(--cyber-cyan)",
      phase: "PHASE 2 — NEURAL SCAN",
    },
    parsing: {
      label: "PARSING CODE...",
      color: "var(--neural-purple)",
      phase: "PHASE 3 — AST DECOMPOSITION",
    },
    embedding: {
      label: "GENERATING EMBEDDINGS...",
      color: "var(--hot-magenta)",
      phase: "PHASE 4 — VECTOR SYNTHESIS",
    },
    indexing: {
      label: "BUILDING KNOWLEDGE GRAPH...",
      color: "var(--warning-amber)",
      phase: "PHASE 5 — NEURAL INDEXING",
    },
    complete: {
      label: "NEURAL LINK ESTABLISHED ✓",
      color: "var(--success-green)",
      phase: "COMPLETE",
    },
    error: {
      label: "NEURAL PATHWAY ERROR",
      color: "var(--error-red)",
      phase: "ERROR",
    },
  };

export default function UploadProgress() {
  const progress = useStore((s) => s.uploadProgress);
  const isUploading = useStore((s) => s.isUploading);

  if (!progress && !isUploading) return null;

  const stage = STAGES[progress?.stage || "uploading"] || STAGES.uploading;
  const pct = progress?.progress || 0;
  const isComplete = progress?.stage === "complete";
  const isError = progress?.stage === "error";

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        style={{
          position: "fixed",
          inset: 0,
          zIndex: 150,
          background: "rgba(10, 14, 26, 0.95)",
          backdropFilter: "blur(12px)",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 32,
        }}
      >
        {/* Scan lines overlay */}
        <div
          className="scan-lines"
          style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
        />

        {/* Circular progress ring */}
        <div style={{ position: "relative", width: 180, height: 180 }}>
          <svg
            width="180"
            height="180"
            viewBox="0 0 180 180"
            style={{ transform: "rotate(-90deg)" }}
          >
            {/* Background ring */}
            <circle
              cx="90"
              cy="90"
              r="80"
              fill="none"
              stroke="rgba(255,255,255,0.05)"
              strokeWidth="4"
            />
            {/* Progress ring */}
            <motion.circle
              cx="90"
              cy="90"
              r="80"
              fill="none"
              stroke={stage.color}
              strokeWidth="4"
              strokeLinecap="round"
              strokeDasharray={`${2 * Math.PI * 80}`}
              strokeDashoffset={`${2 * Math.PI * 80 * (1 - pct / 100)}`}
              style={{
                filter: `drop-shadow(0 0 8px ${stage.color})`,
                transition: "stroke-dashoffset 0.5s ease-out, stroke 0.3s",
              }}
            />
          </svg>
          {/* Center percentage */}
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <span
              style={{
                fontFamily: '"Orbitron", sans-serif',
                fontSize: "36px",
                fontWeight: 700,
                color: stage.color,
                textShadow: `0 0 15px ${stage.color}`,
              }}
            >
              {Math.round(pct)}
            </span>
            <span
              style={{
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: "10px",
                color: "var(--text-dim)",
                letterSpacing: "2px",
              }}
            >
              PERCENT
            </span>
          </div>
        </div>

        {/* Phase label */}
        <div style={{ textAlign: "center" }}>
          <motion.div
            key={stage.phase}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            style={{
              fontFamily: '"Orbitron", sans-serif',
              fontSize: "10px",
              letterSpacing: "3px",
              color: "var(--text-dim)",
              marginBottom: 8,
            }}
          >
            {stage.phase}
          </motion.div>
          <motion.div
            key={stage.label}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={isComplete ? "neon-cyan" : isError ? "neon-magenta" : ""}
            style={{
              fontFamily: '"Orbitron", sans-serif',
              fontSize: "18px",
              fontWeight: 600,
              letterSpacing: "3px",
              color: stage.color,
            }}
          >
            {stage.label}
          </motion.div>
        </div>

        {/* Current file */}
        {progress?.current_file && (
          <motion.div
            key={progress.current_file}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: "12px",
              color: "var(--text-secondary)",
              padding: "8px 20px",
              border: "1px solid var(--glass-border)",
              borderRadius: "6px",
              background: "rgba(0, 0, 0, 0.3)",
              maxWidth: 500,
              textAlign: "center",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {progress.current_file}
          </motion.div>
        )}

        {/* Stats row */}
        {progress && (
          <div
            style={{
              display: "flex",
              gap: 32,
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: "11px",
              color: "var(--text-dim)",
            }}
          >
            <Stat
              label="FILES"
              value={`${progress.files_processed}/${progress.total_files}`}
            />
            <Stat label="CHUNKS" value={String(progress.chunks_created || 0)} />
            <Stat
              label="ETA"
              value={
                progress.eta_seconds > 0
                  ? `${Math.round(progress.eta_seconds)}s`
                  : "—"
              }
            />
            <Stat
              label="ELAPSED"
              value={`${Math.round(progress.elapsed_seconds || 0)}s`}
            />
          </div>
        )}

        {/* Error messages */}
        {isError && progress?.errors?.length > 0 && (
          <div
            style={{
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: "11px",
              color: "var(--error-red)",
              maxWidth: 500,
              textAlign: "center",
            }}
          >
            {progress.errors.join(" | ")}
          </div>
        )}

        {/* Success burst */}
        {isComplete && (
          <motion.div
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: [0, 1.5, 1], opacity: [0, 1, 0.8] }}
            transition={{ duration: 1 }}
            style={{
              position: "absolute",
              width: 300,
              height: 300,
              borderRadius: "50%",
              border: "2px solid var(--success-green)",
              boxShadow: "0 0 60px rgba(52, 211, 153, 0.3)",
              pointerEvents: "none",
            }}
          />
        )}
      </motion.div>
    </AnimatePresence>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div
        style={{
          color: "var(--text-dim)",
          fontSize: "9px",
          letterSpacing: "2px",
          marginBottom: 2,
        }}
      >
        {label}
      </div>
      <div
        style={{
          color: "var(--text-secondary)",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </div>
    </div>
  );
}
