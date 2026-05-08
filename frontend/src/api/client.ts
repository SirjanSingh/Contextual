/**
 * API client for the Repo-Aware AI backend.
 *
 * The base URL comes from the `VITE_BACKEND_URL` env var:
 *   - In dev (npm run dev), it's an empty string and Vite's proxy handles it.
 *   - In production builds, set VITE_BACKEND_URL to the absolute backend URL.
 */

declare const __BACKEND_URL__: string;

const BASE: string =
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (typeof __BACKEND_URL__ !== "undefined" ? __BACKEND_URL__ : "") || "";

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = (body && (body.detail || body.message)) || detail;
    } catch {
      // ignore
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json() as Promise<T>;
}

// ──────────────────────────────────────────────
// Health & status
// ──────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  version?: string;
  model: string;
  embedding_model: string;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`);
  return asJson<HealthResponse>(res);
}

export async function getIndexStatus(): Promise<{
  status: string;
  info: Record<string, unknown>;
}> {
  const res = await fetch(`${BASE}/index/status`);
  return asJson(res);
}

export async function rebuildIndex(): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/index/rebuild`, { method: "POST" });
  return asJson(res);
}

// ──────────────────────────────────────────────
// Query
// ──────────────────────────────────────────────

export interface QueryResponse {
  answer: string;
  sources: string[];
}

export async function queryBackend(question: string): Promise<QueryResponse> {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return asJson<QueryResponse>(res);
}

export type StreamEvent =
  | { kind: "chunk"; text: string }
  | { kind: "sources"; sources: string[] }
  | { kind: "done" }
  | { kind: "error"; message: string };

/**
 * Stream a query as a sequence of typed events. Mirrors the structure of
 * the VS Code extension's BackendClient.queryStream so behaviour is
 * predictable across surfaces.
 */
export async function* queryStream(
  question: string,
): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${BASE}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok || !res.body) {
    yield {
      kind: "error",
      message: `Stream failed: ${res.status} ${res.statusText}`,
    };
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let pendingEvent: string | null = null;
  let pendingDataLines: string[] = [];

  const flush = (): StreamEvent | null => {
    if (pendingDataLines.length === 0 && !pendingEvent) {
      return null;
    }
    const data = pendingDataLines.join("\n");
    const ev = pendingEvent;
    pendingEvent = null;
    pendingDataLines = [];

    if (ev === "sources") {
      try {
        return { kind: "sources", sources: JSON.parse(data) as string[] };
      } catch {
        return { kind: "error", message: `bad sources payload: ${data}` };
      }
    }
    if (ev === "done") return { kind: "done" };
    if (ev === "error") return { kind: "error", message: data };
    return { kind: "chunk", text: data };
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (line === "") {
          const ev = flush();
          if (ev) {
            yield ev;
            if (ev.kind === "done" || ev.kind === "error") return;
          }
        } else if (line.startsWith("event:")) {
          pendingEvent = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          pendingDataLines.push(line.slice(5).replace(/^\s/, ""));
        }
      }
    }
    const tail = flush();
    if (tail) yield tail;
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // ignore
    }
  }
}

// ──────────────────────────────────────────────
// Upload + WebSocket progress
// ──────────────────────────────────────────────

export interface UploadStartResponse {
  upload_id: string;
  total_files: number;
}

export async function uploadDirectory(
  files: FileList | File[],
): Promise<UploadStartResponse> {
  const formData = new FormData();
  for (const file of Array.from(files)) {
    const path = (file as File & { webkitRelativePath?: string })
      .webkitRelativePath || file.name;
    formData.append("files", file, path);
  }
  const res = await fetch(`${BASE}/upload/directory`, {
    method: "POST",
    body: formData,
  });
  return asJson<UploadStartResponse>(res);
}

export async function getUploadProgress(uploadId: string) {
  const res = await fetch(`${BASE}/upload/progress/${uploadId}`);
  return asJson(res);
}

export interface UploadProgressMessage {
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

export function connectProgressWS(
  uploadId: string,
  onMessage: (data: UploadProgressMessage) => void,
  onClose?: () => void,
): WebSocket {
  // If BASE is set, derive WS URL from it; otherwise use the dev proxy.
  let wsUrl: string;
  if (BASE) {
    wsUrl = BASE.replace(/^http/, "ws") + `/ws/indexing/${uploadId}`;
  } else {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    wsUrl = `${protocol}//${window.location.host}/ws/indexing/${uploadId}`;
  }
  const ws = new WebSocket(wsUrl);
  ws.onmessage = (e) => {
    try {
      onMessage(JSON.parse(e.data));
    } catch {
      // bad JSON; ignore
    }
  };
  ws.onclose = () => onClose?.();
  ws.onerror = () => onClose?.();
  return ws;
}

export async function clearRepository(): Promise<void> {
  const res = await fetch(`${BASE}/repository/clear`, { method: "DELETE" });
  if (!res.ok) {
    throw new Error(`Clear failed: ${res.status} ${res.statusText}`);
  }
}

// ──────────────────────────────────────────────
// Repo Map
// ──────────────────────────────────────────────

export interface RepoMapStats extends Record<string, unknown> {
  total_nodes?: number;
  total_relationships?: number;
  total_communities?: number;
  total_processes?: number;
}

export interface RepoMapCommunity {
  id: string;
  label: string;
  heuristic_label: string;
  cohesion: number;
  symbol_count: number;
}

export interface RepoMapProcess {
  id: string;
  label: string;
  process_type: string;
  step_count: number;
  communities: string[];
}

export interface RepoMapSummary {
  stats: RepoMapStats;
  communities: RepoMapCommunity[];
  processes: RepoMapProcess[];
  community_stats?: Record<string, unknown>;
  process_stats?: Record<string, unknown>;
}

export interface SymbolNode {
  id: string;
  label: string;
  name: string;
  file_path: string;
  start_line: number;
  language: string;
  is_exported: boolean;
  community?: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string;
  confidence?: number;
}

export interface SymbolDetail {
  id: string;
  label: string;
  name: string;
  file_path: string;
  start_line: number;
  end_line: number;
  community?: string;
  callers: Array<{ id: string; name: string; label: string; file_path: string }>;
  callees: Array<{ id: string; name: string; label: string; file_path: string }>;
}

export interface CommunityDetail {
  id: string;
  label: string;
  cohesion: number;
  symbol_count: number;
  members: SymbolNode[];
  internal_relationships: GraphEdge[];
}

export interface ProcessStep {
  step: number;
  node_id: string;
  name: string;
  file_path: string;
  label: string;
}

export interface ProcessDetail {
  id: string;
  label: string;
  process_type: string;
  step_count: number;
  communities: string[];
  steps: ProcessStep[];
}

export async function getRepoMap(): Promise<RepoMapSummary> {
  const res = await fetch(`${BASE}/graph/repo-map`);
  return asJson<RepoMapSummary>(res);
}

export async function getSymbolDetail(symbolId: string): Promise<SymbolDetail> {
  const res = await fetch(
    `${BASE}/graph/symbol/${encodeURIComponent(symbolId)}`,
  );
  return asJson<SymbolDetail>(res);
}

export async function getCommunityDetail(
  communityId: string,
): Promise<CommunityDetail> {
  const res = await fetch(
    `${BASE}/graph/community/${encodeURIComponent(communityId)}`,
  );
  return asJson<CommunityDetail>(res);
}

export async function getProcessDetail(
  processId: string,
): Promise<ProcessDetail> {
  const res = await fetch(
    `${BASE}/graph/process/${encodeURIComponent(processId)}`,
  );
  return asJson<ProcessDetail>(res);
}
