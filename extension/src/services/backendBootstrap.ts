/**
 * Backend bootstrap — ensures a Python venv with the `repo_aware_ai`
 * package installed, ready to be invoked as a sidecar process.
 *
 * Strategy:
 *   1. Locate a system Python interpreter (config override → python3 → python).
 *   2. Verify Python >= 3.10.
 *   3. Create a venv under globalStorageUri/venv if it doesn't exist.
 *   4. Upgrade pip, install `repo-aware-ai[all]` from local source (dev) or
 *      PyPI (production fallback).
 *   5. Skip steps 3-4 on subsequent activations if the venv already imports
 *      `repo_aware_ai` cleanly.
 *
 * All progress + output is streamed to a VS Code output channel so the user
 * can see what is happening; failures surface as actionable error notifications.
 */

import * as vscode from "vscode";
import * as cp from "child_process";
import * as fs from "fs";
import * as path from "path";
import { CONFIG_SECTION } from "../constants";

const MIN_PY_MAJOR = 3;
const MIN_PY_MINOR = 10;
const VENV_DIRNAME = "venv";

export interface BootstrappedBackend {
  /** Absolute path to the venv's Python interpreter. */
  pythonExecutable: string;
  /** Source dir we installed from, or `null` if installed from PyPI. */
  installSource: string | null;
}

export class BackendBootstrap {
  private outputChannel: vscode.OutputChannel;

  constructor(
    private context: vscode.ExtensionContext,
    outputChannel: vscode.OutputChannel,
  ) {
    this.outputChannel = outputChannel;
  }

  async ensure(
    progress: vscode.Progress<{ message?: string; increment?: number }>,
  ): Promise<BootstrappedBackend> {
    const venvDir = path.join(
      this.context.globalStorageUri.fsPath,
      VENV_DIRNAME,
    );
    fs.mkdirSync(this.context.globalStorageUri.fsPath, { recursive: true });

    const venvPython = this._venvPythonPath(venvDir);

    // Fast path: already installed and importable.
    if (fs.existsSync(venvPython)) {
      const ok = await this._canImport(venvPython);
      if (ok) {
        this._log(`[bootstrap] reusing venv at ${venvDir}`);
        return {
          pythonExecutable: venvPython,
          installSource: this._resolveInstallSource(),
        };
      }
      this._log(
        `[bootstrap] venv exists but repo_aware_ai not importable; reinstalling`,
      );
    }

    // 1. Find a system Python.
    progress.report({ message: "Locating Python 3.10+..." });
    const systemPython = await this._findSystemPython();
    this._log(`[bootstrap] using system Python: ${systemPython}`);

    // 2. Create venv.
    if (!fs.existsSync(venvPython)) {
      progress.report({ message: "Creating virtual environment..." });
      await this._exec(systemPython, ["-m", "venv", venvDir]);
      this._log(`[bootstrap] created venv at ${venvDir}`);
    }

    // 3. Upgrade pip, wheel, setuptools (silent).
    progress.report({ message: "Upgrading pip..." });
    await this._exec(venvPython, [
      "-m",
      "pip",
      "install",
      "--upgrade",
      "--quiet",
      "pip",
      "wheel",
      "setuptools",
    ]);

    // 4. Install the backend package.
    const installSource = this._resolveInstallSource();
    if (installSource) {
      progress.report({
        message: `Installing repo-aware-ai from ${path.basename(installSource)} (this may take a minute)...`,
      });
      this._log(`[bootstrap] pip install -e ${installSource}[all]`);
      await this._exec(venvPython, [
        "-m",
        "pip",
        "install",
        "-e",
        `${installSource}[all]`,
      ]);
    } else {
      progress.report({
        message: "Installing repo-aware-ai from PyPI (this may take a minute)...",
      });
      this._log(`[bootstrap] pip install repo-aware-ai[all]`);
      await this._exec(venvPython, [
        "-m",
        "pip",
        "install",
        "repo-aware-ai[all]",
      ]);
    }

    // 5. Final import check.
    const ok = await this._canImport(venvPython);
    if (!ok) {
      throw new Error(
        "Bootstrap finished but `repo_aware_ai` is not importable. See the Repo AI output channel for details.",
      );
    }

    return { pythonExecutable: venvPython, installSource };
  }

  // ────────────────────────────────────────────
  // Helpers
  // ────────────────────────────────────────────

  private _venvPythonPath(venvDir: string): string {
    return process.platform === "win32"
      ? path.join(venvDir, "Scripts", "python.exe")
      : path.join(venvDir, "bin", "python");
  }

  /** Find a Python interpreter on PATH that meets the version floor. */
  private async _findSystemPython(): Promise<string> {
    const config = vscode.workspace.getConfiguration(CONFIG_SECTION);
    const override = config.get<string>("pythonPath");
    const candidates: string[] = [];
    if (override && override !== "python") {
      candidates.push(override);
    }
    candidates.push("python3", "python");

    for (const candidate of candidates) {
      try {
        const out = await this._execCapture(candidate, ["--version"]);
        const m = out.match(/Python\s+(\d+)\.(\d+)/);
        if (!m) {
          continue;
        }
        const major = parseInt(m[1], 10);
        const minor = parseInt(m[2], 10);
        if (
          major > MIN_PY_MAJOR ||
          (major === MIN_PY_MAJOR && minor >= MIN_PY_MINOR)
        ) {
          return candidate;
        }
        this._log(
          `[bootstrap] ${candidate} reports Python ${major}.${minor} (need >= ${MIN_PY_MAJOR}.${MIN_PY_MINOR}); skipping`,
        );
      } catch {
        // not on PATH; try next
      }
    }

    throw new Error(
      `Python ${MIN_PY_MAJOR}.${MIN_PY_MINOR}+ is required but was not found. Install it from https://python.org and set ${CONFIG_SECTION}.pythonPath if needed.`,
    );
  }

  /**
   * Resolve the local source dir to install from. Looks for a pyproject.toml
   * next to the extension folder (dev mode) or under extension/backend
   * (bundled). Returns null to fall back to PyPI.
   */
  private _resolveInstallSource(): string | null {
    const ext = this.context.extensionPath;
    const candidates = [
      path.dirname(ext), // dev: extension/ is inside repo-aware-ai/
      ext, // dev alt: extensionPath IS the repo root
      path.join(ext, "backend"), // packaged: bundled sibling
    ];
    for (const c of candidates) {
      if (fs.existsSync(path.join(c, "pyproject.toml"))) {
        return c;
      }
    }
    return null;
  }

  private async _canImport(pythonExe: string): Promise<boolean> {
    try {
      await this._exec(pythonExe, [
        "-c",
        "import repo_aware_ai, importlib; importlib.import_module('repo_aware_ai.server')",
      ]);
      return true;
    } catch {
      return false;
    }
  }

  private _exec(command: string, args: string[]): Promise<void> {
    this._log(`$ ${command} ${args.join(" ")}`);
    return new Promise((resolve, reject) => {
      const child = cp.spawn(command, args, { shell: false });
      child.stdout?.on("data", (d) => this.outputChannel.append(d.toString()));
      child.stderr?.on("data", (d) => this.outputChannel.append(d.toString()));
      child.on("error", reject);
      child.on("exit", (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`${command} exited with code ${code}`));
        }
      });
    });
  }

  private _execCapture(command: string, args: string[]): Promise<string> {
    return new Promise((resolve, reject) => {
      cp.execFile(command, args, { timeout: 5_000 }, (err, stdout, stderr) => {
        if (err) {
          reject(err);
        } else {
          resolve((stdout || stderr || "").toString());
        }
      });
    });
  }

  private _log(line: string): void {
    this.outputChannel.appendLine(line);
  }
}
