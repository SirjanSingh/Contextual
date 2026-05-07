/**
 * API client for Contextual backend
 */
const BASE = ""; // proxied by Vite in dev

export async function queryBackend(
  question: string,
): Promise<{ answer: string; sources: string[] }> {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Query failed");
  }
  return res.json();
}

export async function getHealth(): Promise<{
  status: string;
  model: string;
  embedding_model: string;
}> {
  const res = await fetch(`${BASE}/health`);
  return res.json();
}

export async function getIndexStatus(): Promise<{
  status: string;
  info: Record<string, unknown>;
}> {
  const res = await fetch(`${BASE}/index/status`);
  return res.json();
}

export async function rebuildIndex(): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/index/rebuild`, { method: "POST" });
  if (!res.ok) throw new Error("Rebuild failed");
  return res.json();
}

export async function uploadDirectory(
  files: FileList | File[],
): Promise<{ upload_id: string; total_files: number }> {
  const formData = new FormData();
  for (const file of Array.from(files)) {
    const path = (file as any).webkitRelativePath || file.name;
    formData.append("files", file, path);
  }
  const res = await fetch(`${BASE}/upload/directory`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

export async function getUploadProgress(uploadId: string) {
  const res = await fetch(`${BASE}/upload/progress/${uploadId}`);
  if (!res.ok) throw new Error("Progress fetch failed");
  return res.json();
}

export function connectProgressWS(
  uploadId: string,
  onMessage: (data: any) => void,
  onClose?: () => void,
) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(
    `${protocol}//${window.location.host}/ws/indexing/${uploadId}`,
  );
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose = () => onClose?.();
  ws.onerror = () => onClose?.();
  return ws;
}

export async function clearRepository(): Promise<void> {
  await fetch(`${BASE}/repository/clear`, { method: "DELETE" });
}

// ── Repo Map API ──────────────────────────────────────────────

export interface RepoMapSummary {
  stats: Record<string, unknown>;
  communities: Array<{
    id: string;
    label: string;
    heuristic_label: string;
    cohesion: number;
    symbol_count: number;
  }>;
  processes: Array<{
    id: string;
    label: string;
    process_type: string;
    step_count: number;
    communities: string[];
  }>;
  community_stats: Record<string, unknown>;
  process_stats: Record<string, unknown>;
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

export async function getRepoMap(): Promise<RepoMapSummary> {
  const res = await fetch(`${BASE}/graph/repo-map`);
  if (!res.ok) throw new Error(`Repo map not available: ${res.status}`);
  return res.json();
}

export async function getSymbolDetail(symbolId: string): Promise<{
  id: string;
  label: string;
  name: string;
  file_path: string;
  start_line: number;
  end_line: number;
  community?: string;
  callers: Array<{ id: string; name: string; label: string; file_path: string }>;
  callees: Array<{ id: string; name: string; label: string; file_path: string }>;
}> {
  const res = await fetch(`${BASE}/graph/symbol/${encodeURIComponent(symbolId)}`);
  if (!res.ok) throw new Error("Symbol not found");
  return res.json();
}

export async function getCommunityDetail(communityId: string): Promise<{
  id: string;
  label: string;
  cohesion: number;
  symbol_count: number;
  members: SymbolNode[];
  internal_relationships: GraphEdge[];
}> {
  const res = await fetch(`${BASE}/graph/community/${communityId}`);
  if (!res.ok) throw new Error("Community not found");
  return res.json();
}

export async function getProcessDetail(processId: string): Promise<{
  id: string;
  label: string;
  process_type: string;
  step_count: number;
  communities: string[];
  steps: Array<{ step: number; node_id: string; name: string; file_path: string; label: string }>;
}> {
  const res = await fetch(`${BASE}/graph/process/${processId}`);
  if (!res.ok) throw new Error("Process not found");
  return res.json();
}

export async function getNeighborhood(symbolId: string, hops = 2): Promise<{
  nodes: Array<SymbolNode & { is_focal: boolean }>;
  edges: GraphEdge[];
}> {
  const res = await fetch(
    `${BASE}/graph/neighborhood/${encodeURIComponent(symbolId)}?hops=${hops}`,
  );
  if (!res.ok) throw new Error("Neighborhood fetch failed");
  return res.json();
}
