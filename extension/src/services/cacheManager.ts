/**
 * Cache manager — bridges VS Code settings with backend config.
 * Reads repoAwareAI.* settings and exposes them for other services.
 */

import * as vscode from "vscode";
import { CONFIG_SECTION } from "../constants";

export interface ExtensionConfig {
  port: number;
  pythonPath: string;
  googleApiKey: string | undefined;
  topK: number;
  chunkSize: number;
  useReranker: boolean;
  useHybridSearch: boolean;
  useQueryExpansion: boolean;
  useCompression: boolean;
  autoIndex: boolean;
}

export function getExtensionConfig(): ExtensionConfig {
  const c = vscode.workspace.getConfiguration(CONFIG_SECTION);
  return {
    port: c.get<number>("port") ?? 8360,
    pythonPath: c.get<string>("pythonPath") ?? "python",
    googleApiKey: c.get<string>("googleApiKey") ?? process.env.GOOGLE_API_KEY,
    topK: c.get<number>("topK") ?? 6,
    chunkSize: c.get<number>("chunkSize") ?? 1800,
    useReranker: c.get<boolean>("useReranker") ?? true,
    useHybridSearch: c.get<boolean>("useHybridSearch") ?? true,
    useQueryExpansion: c.get<boolean>("useQueryExpansion") ?? true,
    useCompression: c.get<boolean>("useCompression") ?? true,
    autoIndex: c.get<boolean>("autoIndex") ?? true,
  };
}
