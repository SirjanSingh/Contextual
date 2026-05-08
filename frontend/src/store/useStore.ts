/**
 * Zustand store — global state for the Repo-Aware AI web app.
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

export type BackendStatus = "connecting" | "online" | "offline";
export type IndexStatus = "none" | "building" | "ready" | "error";
export type ActiveView = "chat" | "repomap";

interface AppState {
  // Connection
  backendStatus: BackendStatus;
  setBackendStatus: (s: BackendStatus) => void;

  // Backend metadata (from /health)
  model: string;
  embeddingModel: string;
  setBackendInfo: (model: string, embeddingModel: string) => void;

  // Index
  indexStatus: IndexStatus;
  indexInfo: Record<string, unknown>;
  setIndexStatus: (s: IndexStatus, info?: Record<string, unknown>) => void;

  // Chat
  messages: ChatMessage[];
  isQuerying: boolean;
  addMessage: (msg: ChatMessage) => void;
  clearMessages: () => void;
  setIsQuerying: (v: boolean) => void;

  // Upload
  uploadProgress: UploadProgress | null;
  isUploading: boolean;
  setUploadProgress: (p: UploadProgress | null) => void;
  setIsUploading: (v: boolean) => void;

  // UI
  showActivityFeed: boolean;
  isBooting: boolean;
  errorShake: boolean;
  toggleActivityFeed: () => void;
  setIsBooting: (v: boolean) => void;
  triggerErrorShake: () => void;

  // Activity log (capped to 100 entries)
  activityLog: string[];
  addActivity: (msg: string) => void;

  // View toggle
  activeView: ActiveView;
  setActiveView: (v: ActiveView) => void;

  // Repo map selection
  selectedCommunity: string | null;
  selectedSymbol: string | null;
  setSelectedCommunity: (id: string | null) => void;
  setSelectedSymbol: (id: string | null) => void;
}

export const useStore = create<AppState>((set) => ({
  backendStatus: "connecting",
  setBackendStatus: (backendStatus) => set({ backendStatus }),

  model: "",
  embeddingModel: "",
  setBackendInfo: (model, embeddingModel) => set({ model, embeddingModel }),

  indexStatus: "none",
  indexInfo: {},
  setIndexStatus: (indexStatus, indexInfo) =>
    set(indexInfo ? { indexStatus, indexInfo } : { indexStatus }),

  messages: [],
  isQuerying: false,
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  clearMessages: () => set({ messages: [] }),
  setIsQuerying: (isQuerying) => set({ isQuerying }),

  uploadProgress: null,
  isUploading: false,
  setUploadProgress: (uploadProgress) => set({ uploadProgress }),
  setIsUploading: (isUploading) => set({ isUploading }),

  showActivityFeed: true,
  isBooting: true,
  errorShake: false,
  toggleActivityFeed: () =>
    set((s) => ({ showActivityFeed: !s.showActivityFeed })),
  setIsBooting: (isBooting) => set({ isBooting }),
  triggerErrorShake: () => {
    set({ errorShake: true });
    setTimeout(() => set({ errorShake: false }), 300);
  },

  activityLog: [],
  addActivity: (msg) =>
    set((s) => ({
      activityLog: [
        `[${new Date().toLocaleTimeString()}] ${msg}`,
        ...s.activityLog,
      ].slice(0, 100),
    })),

  activeView: "chat",
  setActiveView: (activeView) => set({ activeView }),

  selectedCommunity: null,
  selectedSymbol: null,
  setSelectedCommunity: (selectedCommunity) => set({ selectedCommunity }),
  setSelectedSymbol: (selectedSymbol) => set({ selectedSymbol }),
}));
