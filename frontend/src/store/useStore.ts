/**
 * Zustand store – global state for Contextual Neural Code Nexus
 */
import { create } from "zustand";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  sources?: string[];
  timestamp: number;
}

export interface UploadProgress {
  upload_id: string;
  stage: string;
  progress: number;
  total_files: number;
  files_processed: number;
  current_file: string;
  chunks_created: number;
  errors: string[];
  eta_seconds: number;
  elapsed_seconds: number;
}

interface AppState {
  // Connection
  backendStatus: "connecting" | "online" | "offline";
  setBackendStatus: (s: AppState["backendStatus"]) => void;

  // Index
  indexStatus: "none" | "building" | "ready" | "error";
  indexInfo: Record<string, unknown>;
  setIndexStatus: (
    s: AppState["indexStatus"],
    info?: Record<string, unknown>,
  ) => void;

  // Chat
  messages: ChatMessage[];
  isQuerying: boolean;
  addMessage: (msg: ChatMessage) => void;
  clearMessages: () => void;
  setIsQuerying: (v: boolean) => void;

  // Upload
  uploadProgress: UploadProgress | null;
  isUploading: boolean;
  showUploadZone: boolean;
  setUploadProgress: (p: UploadProgress | null) => void;
  setIsUploading: (v: boolean) => void;
  setShowUploadZone: (v: boolean) => void;

  // UI
  showRepoExplorer: boolean;
  showActivityFeed: boolean;
  isBooting: boolean;
  errorShake: boolean;
  toggleRepoExplorer: () => void;
  toggleActivityFeed: () => void;
  setIsBooting: (v: boolean) => void;
  triggerErrorShake: () => void;

  // Activity log
  activityLog: string[];
  addActivity: (msg: string) => void;
}

export const useStore = create<AppState>((set) => ({
  // Connection
  backendStatus: "connecting",
  setBackendStatus: (backendStatus) => set({ backendStatus }),

  // Index
  indexStatus: "none",
  indexInfo: {},
  setIndexStatus: (indexStatus, indexInfo) =>
    set({ indexStatus, ...(indexInfo ? { indexInfo } : {}) }),

  // Chat
  messages: [],
  isQuerying: false,
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  clearMessages: () => set({ messages: [] }),
  setIsQuerying: (isQuerying) => set({ isQuerying }),

  // Upload
  uploadProgress: null,
  isUploading: false,
  showUploadZone: false,
  setUploadProgress: (uploadProgress) => set({ uploadProgress }),
  setIsUploading: (isUploading) => set({ isUploading }),
  setShowUploadZone: (showUploadZone) => set({ showUploadZone }),

  // UI
  showRepoExplorer: false,
  showActivityFeed: true,
  isBooting: true,
  errorShake: false,
  toggleRepoExplorer: () =>
    set((s) => ({ showRepoExplorer: !s.showRepoExplorer })),
  toggleActivityFeed: () =>
    set((s) => ({ showActivityFeed: !s.showActivityFeed })),
  setIsBooting: (isBooting) => set({ isBooting }),
  triggerErrorShake: () => {
    set({ errorShake: true });
    setTimeout(() => set({ errorShake: false }), 300);
  },

  // Activity log
  activityLog: [],
  addActivity: (msg) =>
    set((s) => ({
      activityLog: [
        `[${new Date().toLocaleTimeString()}] ${msg}`,
        ...s.activityLog,
      ].slice(0, 100),
    })),
}));
