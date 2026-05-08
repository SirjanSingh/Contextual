import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Backend URL is configurable so the frontend can target a remote server
 * when not running on localhost. Defaults to the FastAPI sidecar on :8360.
 *
 *   VITE_BACKEND_URL=http://localhost:8360   # dev (default)
 *   VITE_BACKEND_URL=https://api.example.com # production-style deploy
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backendUrl = env.VITE_BACKEND_URL || "http://localhost:8360";
  const wsUrl = backendUrl.replace(/^http/, "ws");

  const proxyTarget = {
    target: backendUrl,
    changeOrigin: true,
  };

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/query": proxyTarget,
        "/index": proxyTarget,
        "/upload": proxyTarget,
        "/health": proxyTarget,
        "/search": proxyTarget,
        "/context": proxyTarget,
        "/repository": proxyTarget,
        "/graph": proxyTarget,
        "/ws": {
          target: wsUrl,
          ws: true,
          changeOrigin: true,
        },
      },
    },
    define: {
      // Expose to runtime so client.ts can use it when not behind the dev proxy.
      __BACKEND_URL__: JSON.stringify(env.VITE_BACKEND_URL || ""),
    },
  };
});
