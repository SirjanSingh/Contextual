/**
 * Main extension entry point.
 *
 * Activation order:
 *  1. Check for Google API key (prompt if missing)
 *  2. Check Python + install requirements if needed
 *  3. Spawn Python sidecar backend
 *  4. Wait for /health
 *  5. Index workspace via /index/directory
 *  6. Register all commands, providers, views
 *  7. Start status bar polling
 *
 * Deactivation: kill backend, dispose all providers
 */

import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";

import { BackendProcess } from "./services/backendProcess";
import { IndexManager } from "./services/indexManager";
import { getExtensionConfig } from "./services/cacheManager";
import { StatusBar } from "./views/statusBar";
import { ChatPanel } from "./views/chatPanel";
import { RepoMapPanel } from "./views/repoMapPanel";
import { CodeLensProvider } from "./providers/codeLensProvider";
import { HoverProvider } from "./providers/hoverProvider";
import { registerAskQuestion } from "./commands/askQuestion";
import { registerExplainSelection } from "./commands/explainSelection";
import { registerFindRelated } from "./commands/findRelated";
import { registerExplainRepo } from "./commands/explainRepo";
import { registerRebuildIndex } from "./commands/rebuildIndex";

let backendProcess: BackendProcess | null = null;

export async function activate(
  context: vscode.ExtensionContext,
): Promise<void> {
  // ── 1. Resolve backend root (extension install dir → parent of `extension/`) ──
  const extensionDir = context.extensionPath;
  // In dev: extensionDir = …/repo-aware-ai/extension → parent = repo-aware-ai
  // In production: extension ships with backend/ sibling folder
  const backendRoot = _resolveBackendRoot(extensionDir);

  // ── 2. Ensure API key ──
  const config = getExtensionConfig();
  let apiKey = config.googleApiKey;

  if (!apiKey) {
    apiKey = await _promptApiKey(context);
    if (!apiKey) {
      void vscode.window.showWarningMessage(
        "Repo AI: No Google API key set. Backend will not start.",
      );
      return;
    }
  }

  // ── 3. Ensure Python requirements ──
  await _ensureRequirements(backendRoot, config.pythonPath);

  // ── 4. Start the Python sidecar ──
  // globalStorageUri is VS Code's per-extension persistent storage — writable,
  // survives updates, and isolated from the user's workspace.
  const dataDir = context.globalStorageUri.fsPath;
  backendProcess = new BackendProcess(backendRoot, config.port, dataDir);
  context.subscriptions.push({ dispose: () => backendProcess?.dispose() });

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Repo AI",
      cancellable: false,
    },
    async (progress) => {
      progress.report({ message: "Starting backend..." });
      try {
        await backendProcess!.start(apiKey!);
      } catch (e) {
        void vscode.window.showErrorMessage(
          `Repo AI: Backend failed to start — ${String(e)}`,
        );
        backendProcess?.showOutput();
        return;
      }

      const client = backendProcess!.getClient();

      // ── 5. Index workspace ──
      const workspaceFolders = vscode.workspace.workspaceFolders;
      if (workspaceFolders && workspaceFolders.length > 0) {
        const rootPath = workspaceFolders[0].uri.fsPath;
        progress.report({ message: `Indexing ${path.basename(rootPath)}...` });
        try {
          await client.indexDirectory(rootPath);
        } catch (e) {
          void vscode.window.showWarningMessage(
            `Repo AI: Auto-index failed — ${String(e)}`,
          );
        }
      }

      // ── 6. Status bar ──
      const statusBar = new StatusBar(client);
      context.subscriptions.push({ dispose: () => statusBar.dispose() });
      statusBar.startPolling();

      // ── 7. Chat panel ──
      const { provider: chatProvider, disposable: chatDisposable } =
        ChatPanel.register(context, client);
      context.subscriptions.push(chatDisposable);

      const showChat = (
        question: string,
        answer: string,
        sources: string[],
      ) => {
        chatProvider.showAnswer(question, answer, sources);
      };

      // ── 8. Register commands ──
      context.subscriptions.push(
        registerAskQuestion(context, client, showChat),
        registerExplainSelection(context, client, showChat),
        registerFindRelated(context, client),
        registerExplainRepo(context, client),
        registerRebuildIndex(context, client, statusBar),

        // Open chat command
        vscode.commands.registerCommand("repoAwareAI.openChat", () => {
          vscode.commands.executeCommand("repoAwareAI.chatView.focus");
        }),

        // Open repo map command
        vscode.commands.registerCommand("repoAwareAI.openRepoMap", () => {
          RepoMapPanel.createOrShow(context, client);
        }),
      );

      // ── 9. Register CodeLens + Hover providers ──
      context.subscriptions.push(...CodeLensProvider.register(client));
      context.subscriptions.push(...HoverProvider.register(client));

      // ── 10. Auto-index on save ──
      const indexManager = new IndexManager(client);
      indexManager.enable();
      context.subscriptions.push({ dispose: () => indexManager.dispose() });
    },
  );
}

export function deactivate(): void {
  backendProcess?.dispose();
  backendProcess = null;
}

// ── Helpers ──

function _resolveBackendRoot(extensionDir: string): string {
  // Try extensionDir itself (when extensionPath == repo root in dev mode)
  if (fs.existsSync(path.join(extensionDir, "app", "server.py"))) {
    return extensionDir;
  }
  // Try ../  (dev workspace: extension/ is inside repo-aware-ai/)
  const parentDir = path.dirname(extensionDir);
  if (fs.existsSync(path.join(parentDir, "app", "server.py"))) {
    return parentDir;
  }
  // Try ./backend/  (packaged extension ships backend alongside)
  const bundledBackend = path.join(extensionDir, "backend");
  if (fs.existsSync(path.join(bundledBackend, "app", "server.py"))) {
    return bundledBackend;
  }
  // Fallback: workspace root
  const wsRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  return wsRoot ?? extensionDir;
}

async function _promptApiKey(
  context: vscode.ExtensionContext,
): Promise<string | undefined> {
  const key = await vscode.window.showInputBox({
    prompt: "Enter your Google API Key (for Gemini)",
    placeHolder: "AIza...",
    password: true,
    ignoreFocusOut: true,
  });
  if (!key) {
    return undefined;
  }
  // Store in SecretStorage (not settings.json)
  await context.secrets.store("googleApiKey", key);
  // Also set in config for this session
  await vscode.workspace
    .getConfiguration("repoAwareAI")
    .update("googleApiKey", key, vscode.ConfigurationTarget.Global);
  return key;
}

async function _ensureRequirements(
  backendRoot: string,
  pythonPath: string,
): Promise<void> {
  const reqFile = path.join(backendRoot, "requirements.txt");
  if (!fs.existsSync(reqFile)) {
    return; // Nothing to install
  }

  // Run pip install silently in background
  const { execFile } = await import("child_process");
  return new Promise((resolve) => {
    execFile(
      pythonPath,
      ["-m", "pip", "install", "-r", reqFile, "--quiet"],
      { cwd: backendRoot },
      () => resolve(), // ignore errors — user may have already installed
    );
  });
}
