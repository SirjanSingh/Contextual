/**
 * Index manager — auto-index on file save, track status.
 */

import * as vscode from "vscode";
import { BackendClient } from "./backendClient";
import { AUTO_INDEX_DEBOUNCE_MS, CONFIG_SECTION } from "../constants";

const INDEXED_EXTENSIONS = new Set([
  ".py",
  ".ts",
  ".tsx",
  ".js",
  ".jsx",
  ".go",
  ".rs",
  ".java",
  ".cs",
  ".cpp",
  ".c",
  ".h",
  ".rb",
  ".php",
  ".kt",
  ".swift",
  ".md",
  ".txt",
]);

export class IndexManager {
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;
  private subscription: vscode.Disposable | null = null;

  constructor(private client: BackendClient) {}

  /** Set up file-save watcher. */
  enable(): void {
    if (this.subscription) {
      return;
    }
    this.subscription = vscode.workspace.onDidSaveTextDocument((doc) => {
      const config = vscode.workspace.getConfiguration(CONFIG_SECTION);
      if (!config.get<boolean>("autoIndex", true)) {
        return;
      }
      const ext = doc.fileName.slice(doc.fileName.lastIndexOf("."));
      if (!INDEXED_EXTENSIONS.has(ext)) {
        return;
      }
      this._scheduleRebuild();
    });
  }

  private _scheduleRebuild(): void {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }
    this.debounceTimer = setTimeout(async () => {
      try {
        await this.client.rebuildIndex();
      } catch (e) {
        // Fail silently on auto-index
      }
    }, AUTO_INDEX_DEBOUNCE_MS);
  }

  dispose(): void {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }
    this.subscription?.dispose();
  }
}
