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
