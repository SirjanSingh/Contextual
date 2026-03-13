/**
 * HTTP + WebSocket client for the Python sidecar backend.
 * All REST methods throw on non-2xx responses.
 */

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

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  chunkCount: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string;
}

export interface DependencyGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface Cluster {
  id: number;
  centroid_label: string;
  files: string[];
  size: number;
}

export interface ClusterResponse {
  clusters: Cluster[];
}

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
    await this._fetch("/index/rebuild", "POST", undefined, 120_000);
  }

  async query(question: string): Promise<QueryResponse> {
    return this._fetch<QueryResponse>("/query", "POST", { question }, 60_000);
  }

  /** SSE streaming query. Yields token chunks as strings. */
  async *queryStream(question: string): AsyncGenerator<string> {
    const res = await fetch(`${this.baseUrl}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok || !res.body) {
      throw new Error(`Stream request failed: ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          yield line.slice(6);
        }
      }
    }
  }

  async search(query: string, topK = 8): Promise<ChunkResult[]> {
    const res = await this._fetch<SearchResponse>(
      "/search",
      "POST",
      { question: query },
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

  async dependencyGraph(): Promise<DependencyGraph> {
    return this._fetch<DependencyGraph>("/graph/dependencies");
  }

  async semanticClusters(): Promise<ClusterResponse> {
    return this._fetch<ClusterResponse>("/graph/clusters");
  }
}
