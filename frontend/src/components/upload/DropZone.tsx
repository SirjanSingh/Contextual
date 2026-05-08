/**
 * Drop zone — full-screen overlay on drag with hexagonal grid effect
 */
import { useCallback, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useStore } from "../../store/useStore";
import { uploadDirectory, connectProgressWS } from "../../api/client";

export default function DropZone() {
  const [isDragging, setIsDragging] = useState(false);
  const setIsUploading = useStore((s) => s.setIsUploading);
  const setUploadProgress = useStore((s) => s.setUploadProgress);
  const addActivity = useStore((s) => s.addActivity);
  const setIndexStatus = useStore((s) => s.setIndexStatus);

  useEffect(() => {
    const handleDragEnter = (e: DragEvent) => {
      e.preventDefault();
      if (e.dataTransfer?.types.includes("Files")) {
        setIsDragging(true);
      }
    };
    const handleDragLeave = (e: DragEvent) => {
      e.preventDefault();
      if (e.relatedTarget === null) setIsDragging(false);
    };
    const handleDragOver = (e: DragEvent) => {
      e.preventDefault();
    };
    const handleDrop = async (e: DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const files = e.dataTransfer?.files;
      if (!files?.length) return;

      await startUpload(files);
    };

    document.addEventListener("dragenter", handleDragEnter);
    document.addEventListener("dragleave", handleDragLeave);
    document.addEventListener("dragover", handleDragOver);
    document.addEventListener("drop", handleDrop);
    return () => {
      document.removeEventListener("dragenter", handleDragEnter);
      document.removeEventListener("dragleave", handleDragLeave);
      document.removeEventListener("dragover", handleDragOver);
      document.removeEventListener("drop", handleDrop);
    };
  }, []);

  const startUpload = useCallback(async (files: FileList | File[]) => {
    setIsUploading(true);
    addActivity(`Uploading ${files.length} files...`);
    setIndexStatus("building");

    try {
      const { upload_id } = await uploadDirectory(files);
      addActivity(`Upload started: ${upload_id}`);

      // Live progress feed via WebSocket.
      connectProgressWS(upload_id, (data) => {
        setUploadProgress(data);
        if (data.stage === "complete") {
          setIsUploading(false);
          setIndexStatus("ready", { ...data });
          addActivity("Repository indexed successfully!");
          setTimeout(() => setUploadProgress(null), 4000);
        } else if (data.stage === "error") {
          setIsUploading(false);
          setIndexStatus("error");
          addActivity(`Indexing error: ${data.errors?.join(", ")}`);
          setTimeout(() => setUploadProgress(null), 5000);
        }
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setIsUploading(false);
      setIndexStatus("error");
      addActivity(`Upload failed: ${message}`);
    }
  }, [addActivity, setIndexStatus, setIsUploading, setUploadProgress]);

  const handleBrowseClick = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.setAttribute("webkitdirectory", "");
    input.setAttribute("directory", "");
    input.multiple = true;
    input.onchange = (e) => {
      const files = (e.target as HTMLInputElement).files;
      if (files?.length) startUpload(files);
    };
    input.click();
  };

  return (
    <>
      {/* Always-visible small upload button */}
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={handleBrowseClick}
        style={{
          position: "fixed",
          top: 70,
          right: 24,
          zIndex: 60,
          padding: "8px 16px",
          border: "1px solid var(--neural-purple-dim)",
          borderRadius: "8px",
          background: "rgba(139, 92, 246, 0.1)",
          color: "var(--neural-purple)",
          fontFamily: '"Orbitron", sans-serif',
          fontSize: "10px",
          letterSpacing: "1.5px",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          gap: 6,
          transition: "all 0.3s",
        }}
      >
        <span style={{ fontSize: "14px" }}>⬡</span>
        UPLOAD REPO
      </motion.button>

      {/* Full-screen drag overlay */}
      <AnimatePresence>
        {isDragging && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            style={{
              position: "fixed",
              inset: 0,
              zIndex: 200,
              background: "rgba(10, 14, 26, 0.92)",
              backdropFilter: "blur(8px)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: 24,
            }}
          >
            {/* Hexagonal grid background */}
            <div
              style={{
                position: "absolute",
                inset: 0,
                backgroundImage: `radial-gradient(circle, rgba(0, 247, 255, 0.08) 1px, transparent 1px)`,
                backgroundSize: "30px 30px",
                animation: "hex-pulse 2s ease-in-out infinite",
              }}
            />

            {/* Pulsing border */}
            <motion.div
              animate={{
                boxShadow: [
                  "inset 0 0 30px rgba(0, 247, 255, 0.1)",
                  "inset 0 0 60px rgba(0, 247, 255, 0.2)",
                  "inset 0 0 30px rgba(0, 247, 255, 0.1)",
                ],
              }}
              transition={{ duration: 2, repeat: Infinity }}
              style={{
                position: "absolute",
                inset: 20,
                border: "2px dashed var(--cyber-cyan-dim)",
                borderRadius: "20px",
              }}
            />

            {/* Icon */}
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
              style={{
                width: 80,
                height: 80,
                border: "2px solid var(--cyber-cyan)",
                borderRadius: "16px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                boxShadow: "0 0 30px var(--cyber-cyan-dim)",
                position: "relative",
                zIndex: 1,
              }}
            >
              <span style={{ fontSize: "36px", transform: "rotate(-45deg)" }}>
                ⬡
              </span>
            </motion.div>

            {/* Text */}
            <div
              style={{ textAlign: "center", position: "relative", zIndex: 1 }}
            >
              <div
                className="glitch-text neon-cyan"
                data-text="INITIATING NEURAL SCAN..."
                style={{
                  fontFamily: '"Orbitron", sans-serif',
                  fontSize: "24px",
                  fontWeight: 700,
                  letterSpacing: "4px",
                  marginBottom: 12,
                }}
              >
                INITIATING NEURAL SCAN...
              </div>
              <div
                style={{
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: "12px",
                  color: "var(--text-secondary)",
                  letterSpacing: "1px",
                }}
              >
                DROP YOUR REPOSITORY TO BEGIN ANALYSIS
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
