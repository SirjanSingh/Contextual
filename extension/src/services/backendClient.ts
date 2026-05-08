/**
 * HTTP + SSE client for the Python sidecar backend.
 *
 * All REST methods throw on non-2xx responses. The streaming method yields
 * structured events so callers don't have to parse SSE themselves.
 */

// ──────────────────────────────────────────────
// Response types
// ──────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  version: string;
  model: string;
  embedding_model: string;
}

export interface QueryResponse {
  answer: string;
  sources: string[];
}

export interface ChunkResult {
  source: string;
  text: string;
  start_char: number;
  end_char: number;
  score: number;
}

export interface SearchResponse {
  chunks: ChunkResult[];
}

export interface FileContextResponse {
  chunks: ChunkResult[];
  related_files: string[];
}

export interface IndexStatus {
  status: string;
  info: Record<string, unknown>;
}

// ── Repo Map types ────────────────────────────

export interface RepoMapCommunity {
  id: string;
  label: string;
  heuristic_label: string;
  symbol_count: number;
  cohesion: number;
}

export interface RepoMapProcess {
  id: string;
  label: string;
  process_type: string;
  step_count: number;
}

export interface RepoMapStats {
  total_nodes: number;
  total_relationships: number;
  total_communities: number;
  total_processes: number;
}

export interface RepoMapSummary {
  communities: RepoMapCommunity[];
  processes: RepoMapProcess[];
  stats: RepoMapStats;
}

export interface CommunityMember {
  id: string;
  name: string;
  label: string;
  file_path: string;
}

export interface CommunityRelationship {
  source: string;
  target: string;
  type: string;
  confidence: number;
}

export interface CommunityDetail {
  id: string;
  label: string;
  heuristic_label: string;
  symbol_count: number;
  cohesion: number;
  members: CommunityMember[];
  internal_relationships: CommunityRelationship[];
}

export interface ProcessStep {
  step: number;
  node_id: string;
  name: string;
  label: string;
  file_path: string;
}

export interface ProcessDetail {
  id: string;
  label: string;
  process_type: string;
  step_count: number;
  communities: string[];
  steps: ProcessStep[];
}

export interface SymbolRelationship {
  id: string;
  name: string;
  label: string;
  file_path: string;
  rel_type: string;
  confidence: number;
}

export interface SymbolDetail {
  id: string;
  name: string;
  label: string;
  file_path: string;
  start_line: number;
  end_line: number;
  is_exported: boolean;
  community_id: string | null;
  callers: SymbolRelationship[];
  callees: SymbolRelationship[];
}

// ──────────────────────────────────────────────
// Streaming events
// ──────────────────────────────────────────────

export type StreamEvent =
  | { kind: "chunk"; text: string }
  | { kind: "sources"; sources: string[] }
  | { kind: "done" }
  | { kind: "error"; message: string };

// ──────────────────────────────────────────────
// Client
// ──────────────────────────────────────────────

export class BackendClient {
  constructor(private baseUrl: string) {}

  private async _fetch<T>(
    path: string,
    method: "GET" | "POST" | "DELETE" = "GET",
    body?: unknown,
    timeoutMs = 10_000,
  ): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        method,
        headers: body ? { "Content-Type": "application/json" } : {},
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      if (!res.ok) {
        const text = await res.text().catch(() => res.statusText);
        throw new Error(`Backend ${method} ${path} → ${res.status}: ${text}`);
      }

      return res.json() as Promise<T>;
    } finally {
      clearTimeout(timer);
    }
  }

  // ── Lifecycle ──
  async health(): Promise<HealthResponse> {
    return this._fetch<HealthResponse>("/health", "GET", undefined, 5_000);
  }

  async indexStatus(): Promise<IndexStatus> {
    return this._fetch<IndexStatus>("/index/status");
  }

  async indexDirectory(
    repoPath: string,
  ): Promise<{ status: string; message: string }> {
    return this._fetch(
      "/index/directory",
      "POST",
      { repo_path: repoPath },
      60_000,
    );
  }

  async rebuildIndex(): Promise<void> {
    await this._fetch("/index/rebuild", "POST", undefined, 300_000);
  }

  // ── Query ──
  async query(question: string): Promise<QueryResponse> {
    return this._fetch<QueryResponse>("/query", "POST", { question }, 120_000);
  }

  /**
   * Stream an answer from /query/stream as a sequence of structured events.
   * The iterator completes after a `done` event or an `error` event.
   */
  async *queryStream(question: string): AsyncGenerator<StreamEvent> {
    const res = await fetch(`${this.baseUrl}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => res.statusText);
      yield { kind: "error", message: `Stream request failed: ${res.status} ${text}` };
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
          const sources = JSON.parse(data) as string[];
          return { kind: "sources", sources };
        } catch {
          return { kind: "error", message: `bad sources payload: ${data}` };
        }
      }
      if (ev === "done") {
        return { kind: "done" };
      }
      if (ev === "error") {
        return { kind: "error", message: data };
      }
      // Default event: a token chunk.
      return { kind: "chunk", text: data };
    };

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (line === "") {
            // Empty line marks the end of an SSE event.
            const ev = flush();
            if (ev) {
              yield ev;
              if (ev.kind === "done" || ev.kind === "error") {
                return;
              }
            }
          } else if (line.startsWith("event:")) {
            pendingEvent = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            pendingDataLines.push(line.slice(5).replace(/^\s/, ""));
          }
          // ignore other lines (id:, retry:, comments)
        }
      }
      // Final flush in case the stream ended without a trailing blank line.
      const tail = flush();
      if (tail) {
        yield tail;
      }
    } finally {
      try {
        reader.releaseLock();
      } catch {
        // ignore
      }
    }
  }

  // ── Search ──
  async search(query: string, topK = 8): Promise<ChunkResult[]> {
    const res = await this._fetch<SearchResponse>(
      "/search",
      "POST",
      { question: query, top_k: topK },
      10_000,
    );
    return res.chunks.slice(0, topK);
  }

  async fileContext(filePath: string): Promise<FileContextResponse> {
    const encoded = encodeURIComponent(filePath);
    return this._fetch<FileContextResponse>(
      `/context/file?file_path=${encoded}`,
      "GET",
      undefined,
      8_000,
    );
  }

  async clearRepository(): Promise<void> {
    await this._fetch("/repository/clear", "DELETE");
  }

  // ── Repo map ──
  async repoMapSummary(): Promise<RepoMapSummary> {
    return this._fetch<RepoMapSummary>(
      "/graph/repo-map",
      "GET",
      undefined,
      15_000,
    );
  }

  async communityDetail(id: string): Promise<CommunityDetail> {
    return this._fetch<CommunityDetail>(
      `/graph/community/${encodeURIComponent(id)}`,
    );
  }

  async processDetail(id: string): Promise<ProcessDetail> {
    return this._fetch<ProcessDetail>(
      `/graph/process/${encodeURIComponent(id)}`,
    );
  }

  async symbolDetail(id: string): Promise<SymbolDetail> {
    return this._fetch<SymbolDetail>(
      `/graph/symbol/${encodeURIComponent(id)}`,
    );
  }
}
