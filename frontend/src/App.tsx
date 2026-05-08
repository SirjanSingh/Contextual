/**
 * App — Main application shell
 */
import { useEffect, useCallback, useState } from "react";
import { motion } from "framer-motion";
import Scene from "./components/three/Scene";
import Header from "./components/layout/Header";
import Footer from "./components/layout/Footer";
import ChatView from "./components/chat/ChatView";
import QueryInput from "./components/chat/QueryInput";
import DropZone from "./components/upload/DropZone";
import UploadProgress from "./components/upload/UploadProgress";
import ActivityFeed from "./components/sidebar/ActivityFeed";
import BootSequence from "./components/ui/BootSequence";
import RepoMapView from "./components/repomap/RepoMapView";
import { useStore } from "./store/useStore";
import { getHealth, getIndexStatus } from "./api/client";

export default function App() {
  const isBooting = useStore((s) => s.isBooting);
  const setIsBooting = useStore((s) => s.setIsBooting);
  const errorShake = useStore((s) => s.errorShake);
  const setBackendStatus = useStore((s) => s.setBackendStatus);
  const setBackendInfo = useStore((s) => s.setBackendInfo);
  const setIndexStatus = useStore((s) => s.setIndexStatus);
  const addActivity = useStore((s) => s.addActivity);
  const showActivityFeed = useStore((s) => s.showActivityFeed);
  const activeView = useStore((s) => s.activeView);

  // Health + index status check, retried until the backend responds.
  useEffect(() => {
    let cancelled = false;
    let attempts = 0;

    const isIndexStatus = (
      v: unknown,
    ): v is "none" | "building" | "ready" | "error" =>
      v === "none" || v === "building" || v === "ready" || v === "error";

    const probe = async () => {
      try {
        const health = await getHealth();
        if (cancelled) return;
        if (health.status === "ok") {
          setBackendStatus("online");
          setBackendInfo(health.model || "", health.embedding_model || "");
          addActivity("Backend connection established");
          if (health.model) addActivity(`Model: ${health.model}`);
        } else {
          setBackendStatus("offline");
        }
      } catch {
        if (cancelled) return;
        setBackendStatus("offline");
        // Retry with backoff up to ~30s before giving up silently.
        if (attempts < 6) {
          attempts += 1;
          setTimeout(probe, Math.min(1000 * 2 ** attempts, 8000));
          return;
        }
      }

      try {
        const idx = await getIndexStatus();
        if (cancelled) return;
        if (isIndexStatus(idx.status)) {
          setIndexStatus(idx.status, idx.info);
          if (idx.status === "ready") {
            addActivity("Index loaded — ready for queries");
          }
        }
      } catch {
        // status endpoint not yet up; non-fatal
      }
    };

    const timeout = setTimeout(probe, 3500);
    return () => {
      cancelled = true;
      clearTimeout(timeout);
    };
  }, [addActivity, setBackendInfo, setBackendStatus, setIndexStatus]);

  const handleBootComplete = useCallback(() => {
    setIsBooting(false);
  }, [setIsBooting]);

  return (
    <>
      {/* 3D Background */}
      <Scene />

      {/* Boot sequence overlay */}
      {isBooting && <BootSequence onComplete={handleBootComplete} />}

      {/* Drop zone (global drag listener) */}
      <DropZone />

      {/* Upload progress overlay */}
      <UploadProgress />

      {/* Main UI */}
      <motion.div
        className={errorShake ? "shake" : ""}
        initial={{ opacity: 0 }}
        animate={{ opacity: isBooting ? 0 : 1 }}
        transition={{ duration: 0.8, delay: 0.2 }}
        style={{
          position: "fixed",
          inset: 0,
          zIndex: 10,
          display: "flex",
          flexDirection: "column",
          pointerEvents: isBooting ? "none" : "auto",
        }}
      >
        <Header />

        {/* Main content area */}
        <div
          style={{
            flex: 1,
            display: "flex",
            overflow: "hidden",
            padding: "0 16px",
            gap: 12,
          }}
        >
          {/* Main panel */}
          <div
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              minWidth: 0,
              gap: 12,
              padding: "12px 0",
            }}
          >
            {activeView === "chat" ? (
              <>
                <div
                  className="glass-card"
                  style={{
                    flex: 1,
                    display: "flex",
                    flexDirection: "column",
                    overflow: "hidden",
                  }}
                >
                  <ChatView />
                </div>
                <QueryInput />
              </>
            ) : (
              <div
                className="glass-card"
                style={{
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                  overflow: "hidden",
                }}
              >
                <RepoMapView />
              </div>
            )}
          </div>

          {/* Activity feed (right panel) */}
          {showActivityFeed && (
            <div style={{ padding: "12px 0" }}>
              <ActivityFeed />
            </div>
          )}
        </div>

        <Footer />
      </motion.div>
    </>
  );
}
