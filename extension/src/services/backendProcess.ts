/**
 * Spawns and manages the Python sidecar backend process.
 * - Starts: python -m uvicorn repo_aware_ai.server:app --port {PORT} --host 127.0.0.1
 * - Sets: GOOGLE_API_KEY, RAI_PORT env vars
 * - Monitors: stdout/stderr → VS Code output channel
 * - Health check: polls /health every 500ms until ready (max 30s)
 * - Auto-restart: up to 3 times on unexpected crash
 */

import * as vscode from "vscode";
import * as cp from "child_process";
import * as path from "path";
import { BackendClient, HealthResponse } from "./backendClient";
import {
  HEALTH_POLL_INTERVAL_MS,
  BACKEND_STARTUP_TIMEOUT_MS,
  OUTPUT_CHANNEL_NAME,
  CONFIG_SECTION,
} from "../constants";

const MAX_RESTARTS = 3;

export class BackendProcess {
  private process: cp.ChildProcess | null = null;
  private outputChannel: vscode.OutputChannel;
  private restartCount = 0;
  private disposed = false;
  private port: number;
  private backendRoot: string;
  private dataDir: string;
  private client: BackendClient;

  constructor(backendRoot: string, port: number, dataDir: string) {
    this.backendRoot = backendRoot;
    this.port = port;
    this.dataDir = dataDir;
    this.client = new BackendClient(`http://127.0.0.1:${port}`);
    this.outputChannel = vscode.window.createOutputChannel(OUTPUT_CHANNEL_NAME);
  }

  get baseUrl(): string {
    return `http://127.0.0.1:${this.port}`;
  }

  getClient(): BackendClient {
    return this.client;
  }

  showOutput(): void {
    this.outputChannel.show(true);
  }

  /** Start the backend and wait until /health responds. */
  async start(apiKey: string): Promise<void> {
    if (this.process) {
      return; // Already running
    }
    await this._spawn(apiKey);
    await this._waitUntilHealthy();
  }

  private _spawn(apiKey: string): Promise<void> {
    const config = vscode.workspace.getConfiguration(CONFIG_SECTION);
    const pythonPath: string = config.get("pythonPath") ?? "python";

    const env: NodeJS.ProcessEnv = {
      ...process.env,
      GOOGLE_API_KEY: apiKey,
      RAI_PORT: String(this.port),
      RAI_DATA_DIR: this.dataDir,
    };

    this.outputChannel.appendLine(
      `[BackendProcess] Starting: ${pythonPath} -m uvicorn repo_aware_ai.server:app --port ${this.port} --host 127.0.0.1`,
    );
    this.outputChannel.appendLine(`[BackendProcess] Root: ${this.backendRoot}`);

    this.process = cp.spawn(
      pythonPath,
      [
        "-m",
        "uvicorn",
        "repo_aware_ai.server:app",
        "--port",
        String(this.port),
        "--host",
        "127.0.0.1",
        "--log-level",
        "info",
      ],
      {
        cwd: this.backendRoot,
        env,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );

    this.process.stdout?.on("data", (data: Buffer) => {
      this.outputChannel.append(data.toString());
    });

    this.process.stderr?.on("data", (data: Buffer) => {
      this.outputChannel.append(data.toString());
    });

    this.process.on("exit", (code, signal) => {
      if (this.disposed) {
        return;
      }
      this.outputChannel.appendLine(
        `[BackendProcess] Process exited (code=${code}, signal=${signal}), restarts=${this.restartCount}`,
      );
      this.process = null;

      if (this.restartCount < MAX_RESTARTS) {
        this.restartCount++;
        void vscode.window.showWarningMessage(
          `Repo AI backend crashed — restarting (${this.restartCount}/${MAX_RESTARTS})...`,
        );
        // Delay restart briefly
        setTimeout(() => {
          const key = _loadApiKey();
          if (key) {
            this._spawn(key).catch(console.error);
          }
        }, 2000);
      } else {
        void vscode.window.showErrorMessage(
          "Repo AI backend failed to stay running. Check the 'Repo AI Backend' output channel.",
        );
      }
    });

    this.process.on("error", (err) => {
      this.outputChannel.appendLine(
        `[BackendProcess] Spawn error: ${err.message}`,
      );
      void vscode.window.showErrorMessage(
        `Failed to start backend: ${err.message}`,
      );
    });

    return Promise.resolve();
  }

  private async _waitUntilHealthy(): Promise<void> {
    const deadline = Date.now() + BACKEND_STARTUP_TIMEOUT_MS;

    while (Date.now() < deadline) {
      try {
        const health: HealthResponse = await this.client.health();
        if (health.status === "ok") {
          this.outputChannel.appendLine(
            `[BackendProcess] Backend ready — model=${health.model}, version=${health.version}`,
          );
          this.restartCount = 0; // Reset on successful start
          return;
        }
      } catch {
        // Not ready yet — keep polling
      }
      await _sleep(HEALTH_POLL_INTERVAL_MS);
    }

    throw new Error(
      `Backend did not become healthy within ${BACKEND_STARTUP_TIMEOUT_MS / 1000}s. Check the 'Repo AI Backend' output channel.`,
    );
  }

  dispose(): void {
    this.disposed = true;
    if (this.process) {
      this.process.kill("SIGTERM");
      this.process = null;
    }
    this.outputChannel.dispose();
  }
}

function _sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Try to load API key from settings (SecretStorage preferred, falls back to config). */
function _loadApiKey(): string | undefined {
  const config = vscode.workspace.getConfiguration(CONFIG_SECTION);
  return config.get<string>("googleApiKey") ?? process.env.GOOGLE_API_KEY;
}
