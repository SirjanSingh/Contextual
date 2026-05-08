/**
 * Status bar item showing backend + index state.
 * - "$(loading~spin) Repo AI: Indexing..." while building
 * - "$(database) Repo AI: 1,247 chunks" when ready
 * - "$(warning) Repo AI: Error" on failure
 */

import * as vscode from "vscode";
import { BackendClient } from "../services/backendClient";
import { STATUS_POLL_IDLE_MS, STATUS_POLL_BUSY_MS } from "../constants";

export class StatusBar {
  private item: vscode.StatusBarItem;
  private timer: ReturnType<typeof setInterval> | null = null;

  constructor(private client: BackendClient) {
    this.item = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Left,
      100,
    );
    this.item.command = "workbench.action.quickOpen";
    this.item.text = "$(loading~spin) Repo AI: Starting...";
    this.item.tooltip = "Repo-Aware AI — click to run commands";
    this.item.show();
  }

  /** Start polling the backend for index status. */
  startPolling(): void {
    this._poll(); // Immediate first poll
    this.timer = setInterval(() => this._poll(), STATUS_POLL_IDLE_MS);
  }

  private async _poll(): Promise<void> {
    try {
      const status = await this.client.indexStatus();

      if (status.status === "building") {
        this.item.text = "$(loading~spin) Repo AI: Indexing...";
        this.item.backgroundColor = undefined;
        this._setFastPoll();
      } else if (status.status === "ready") {
        const chunks = (status.info as { chunk_count?: number }).chunk_count;
        const label =
          chunks !== undefined ? `${chunks.toLocaleString()} chunks` : "Ready";
        this.item.text = `$(database) Repo AI: ${label}`;
        this.item.backgroundColor = undefined;
        this._setSlowPoll();
      } else if (status.status === "error") {
        this.item.text = "$(warning) Repo AI: Error";
        this.item.backgroundColor = new vscode.ThemeColor(
          "statusBarItem.warningBackground",
        );
        this._setSlowPoll();
      } else {
        this.item.text = "$(circle-outline) Repo AI: Not indexed";
        this.item.backgroundColor = undefined;
        this._setSlowPoll();
      }
    } catch {
      this.item.text = "$(debug-disconnect) Repo AI: Offline";
      this.item.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.errorBackground",
      );
    }
  }

  private _fastMode = false;

  private _setFastPoll(): void {
    if (this._fastMode) {
      return;
    }
    this._fastMode = true;
    if (this.timer) {
      clearInterval(this.timer);
    }
    this.timer = setInterval(() => this._poll(), STATUS_POLL_BUSY_MS);
  }

  private _setSlowPoll(): void {
    if (!this._fastMode) {
      return;
    }
    this._fastMode = false;
    if (this.timer) {
      clearInterval(this.timer);
    }
    this.timer = setInterval(() => this._poll(), STATUS_POLL_IDLE_MS);
  }

  setIndexing(): void {
    this.item.text = "$(loading~spin) Repo AI: Indexing...";
  }

  dispose(): void {
    if (this.timer) {
      clearInterval(this.timer);
    }
    this.item.dispose();
  }
}
