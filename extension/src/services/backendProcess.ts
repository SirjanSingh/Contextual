/**
 * Spawns and manages the Python sidecar backend process.
 *
 * Expects an already-prepared Python interpreter (typically the venv created
 * by BackendBootstrap) that has `repo_aware_ai` installed and importable.
 *
 * - Runs: `<python> -m uvicorn repo_aware_ai.server:app --port {PORT} --host 127.0.0.1`
 * - Streams stdout/stderr to the supplied output channel.
 * - Polls /health until the server reports ok (max 30s).
 * - Restarts up to 3 times on unexpected crashes; after that, surfaces an error.
 */

import * as vscode from "vscode";
import * as cp from "child_process";

import {
  BACKEND_STARTUP_TIMEOUT_MS,
  CONFIG_SECTION,
  HEALTH_POLL_INTERVAL_MS,
} from "../constants";
import { BackendClient, HealthResponse } from "./backendClient";

const MAX_RESTARTS = 3;

export interface BackendProcessOptions {
  /** Absolute path to the Python interpreter (must have repo_aware_ai installed). */
  pythonExecutable: string;
  /** TCP port to listen on. */
  port: number;
  /** Path passed via RAI_DATA_DIR (FAISS cache root). */
  dataDir: string;
  /** Output channel for backend logs. Created externally so bootstrap shares it. */
  outputChannel: vscode.OutputChannel;
}

export class BackendProcess {
  private process: cp.ChildProcess | null = null;
  private restartCount = 0;
  private disposed = false;
  private apiKey: string | null = null;

  private readonly client: BackendClient;
  private readonly opts: BackendProcessOptions;

  constructor(opts: BackendProcessOptions) {
    this.opts = opts;
    this.client = new BackendClient(`http://127.0.0.1:${opts.port}`);
  }

  get baseUrl(): string {
    return `http://127.0.0.1:${this.opts.port}`;
  }

  getClient(): BackendClient {
    return this.client;
  }

  showOutput(): void {
    this.opts.outputChannel.show(true);
  }

  /** Start the backend and wait until /health responds. */
  async start(apiKey: string): Promise<void> {
    if (this.process) {
      return;
    }
    this.apiKey = apiKey;
    this._spawn();
    await this._waitUntilHealthy();
  }

  private _spawn(): void {
    if (!this.apiKey) {
      throw new Error("BackendProcess.start() must supply an API key before spawn.");
    }

    const env: NodeJS.ProcessEnv = {
      ...process.env,
      GOOGLE_API_KEY: this.apiKey,
      RAI_PORT: String(this.opts.port),
      RAI_DATA_DIR: this.opts.dataDir,
      // Force UTF-8 on Windows so non-ASCII filenames in repos don't crash
      // the uvicorn logger.
      PYTHONIOENCODING: "utf-8",
      PYTHONUNBUFFERED: "1",
    };

    const args = [
      "-m",
      "uvicorn",
      "repo_aware_ai.server:app",
      "--port",
      String(this.opts.port),
      "--host",
      "127.0.0.1",
      "--log-level",
      "info",
    ];

    this.opts.outputChannel.appendLine(
      `[BackendProcess] starting: ${this.opts.pythonExecutable} ${args.join(" ")}`,
    );

    this.process = cp.spawn(this.opts.pythonExecutable, args, {
      env,
      stdio: ["ignore", "pipe", "pipe"],
      shell: false,
    });

    this.process.stdout?.on("data", (data: Buffer) => {
      this.opts.outputChannel.append(data.toString());
    });
    this.process.stderr?.on("data", (data: Buffer) => {
      this.opts.outputChannel.append(data.toString());
    });

    this.process.on("exit", (code, signal) => {
      if (this.disposed) {
        return;
      }
      this.opts.outputChannel.appendLine(
        `[BackendProcess] exited (code=${code}, signal=${signal}); restartCount=${this.restartCount}`,
      );
      this.process = null;
      this._maybeRestart();
    });

    this.process.on("error", (err) => {
      this.opts.outputChannel.appendLine(
        `[BackendProcess] spawn error: ${err.message}`,
      );
      void vscode.window
        .showErrorMessage(
          `Repo AI: failed to start backend (${err.message})`,
          "Show logs",
        )
        .then((choice) => {
          if (choice === "Show logs") {
            this.showOutput();
          }
        });
    });
  }

  private _maybeRestart(): void {
    if (this.restartCount >= MAX_RESTARTS) {
      void vscode.window
        .showErrorMessage(
          "Repo AI backend kept crashing. Check the output channel.",
          "Show logs",
        )
        .then((choice) => {
          if (choice === "Show logs") {
            this.showOutput();
          }
        });
      return;
    }
    this.restartCount++;
    void vscode.window.showWarningMessage(
      `Repo AI backend crashed — restarting (${this.restartCount}/${MAX_RESTARTS})...`,
    );
    setTimeout(() => {
      if (!this.disposed) {
        this._spawn();
      }
    }, 2_000);
  }

  private async _waitUntilHealthy(): Promise<void> {
    const deadline = Date.now() + BACKEND_STARTUP_TIMEOUT_MS;
    while (Date.now() < deadline) {
      try {
        const health: HealthResponse = await this.client.health();
        if (health.status === "ok") {
          this.opts.outputChannel.appendLine(
            `[BackendProcess] healthy — model=${health.model} version=${health.version}`,
          );
          this.restartCount = 0;
          return;
        }
        // Backend started but reports a config error — fail fast.
        if (health.status?.startsWith("error")) {
          throw new Error(health.status);
        }
      } catch {
        // Not yet ready.
      }
      await _sleep(HEALTH_POLL_INTERVAL_MS);
    }
    throw new Error(
      `Backend did not become healthy within ${BACKEND_STARTUP_TIMEOUT_MS / 1000}s. Open the Repo AI Backend output channel for details.`,
    );
  }

  dispose(): void {
    this.disposed = true;
    if (this.process) {
      this.process.kill("SIGTERM");
      this.process = null;
    }
  }
}

function _sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Read API key from SecretStorage-fed setting or env. */
export function loadApiKey(
  context: vscode.ExtensionContext,
): Promise<string | undefined> {
  return Promise.resolve(
    vscode.workspace
      .getConfiguration(CONFIG_SECTION)
      .get<string>("googleApiKey") ?? process.env.GOOGLE_API_KEY,
  ).then(async (existing) => {
    if (existing) {
      return existing;
    }
    return await context.secrets.get("googleApiKey");
  });
}
