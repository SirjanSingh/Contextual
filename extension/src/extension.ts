/**
 * Main extension entry point.
 *
 * Activation order:
 *   1. Resolve API key (SecretStorage, settings, env, or prompt user).
 *   2. Bootstrap a Python venv with the `repo_aware_ai` package installed.
 *   3. Spawn the backend sidecar and wait for /health.
 *   4. Auto-index the workspace.
 *   5. Register commands, providers, views.
 *   6. Start status bar polling.
 */

import * as path from "path";
import * as vscode from "vscode";

import { registerAskQuestion } from "./commands/askQuestion";
import { registerExplainRepo } from "./commands/explainRepo";
import { registerExplainSelection } from "./commands/explainSelection";
import { registerFindRelated } from "./commands/findRelated";
import { registerRebuildIndex } from "./commands/rebuildIndex";
import { OUTPUT_CHANNEL_NAME } from "./constants";
import { CodeLensProvider } from "./providers/codeLensProvider";
import { HoverProvider } from "./providers/hoverProvider";
import { BackendBootstrap } from "./services/backendBootstrap";
import { BackendProcess, loadApiKey } from "./services/backendProcess";
import { getExtensionConfig } from "./services/cacheManager";
import { IndexManager } from "./services/indexManager";
import { ChatPanel } from "./views/chatPanel";
import { RepoMapPanel } from "./views/repoMapPanel";
import { StatusBar } from "./views/statusBar";

let backendProcess: BackendProcess | null = null;
let outputChannel: vscode.OutputChannel | null = null;

export async function activate(
  context: vscode.ExtensionContext,
): Promise<void> {
  outputChannel = vscode.window.createOutputChannel(OUTPUT_CHANNEL_NAME);
  context.subscriptions.push(outputChannel);

  // Always-available command, even if startup fails.
  context.subscriptions.push(
    vscode.commands.registerCommand("repoAwareAI.showLogs", () => {
      outputChannel?.show(true);
    }),
  );

  // ── 1. API key ──
  const config = getExtensionConfig();
  let apiKey = config.googleApiKey || (await loadApiKey(context));
  if (!apiKey) {
    apiKey = await _promptApiKey(context);
  }
  if (!apiKey) {
    void vscode.window.showWarningMessage(
      "Repo AI: no Google API key set — run 'Repo AI: Set API Key' to start the backend.",
    );
    _registerSetApiKey(context);
    return;
  }

  // ── 2 + 3. Bootstrap Python + start backend ──
  let pythonExecutable: string;
  try {
    pythonExecutable = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "Repo AI",
        cancellable: false,
      },
      async (progress) => {
        const bootstrap = new BackendBootstrap(context, outputChannel!);
        progress.report({ message: "Setting up Python environment..." });
        const result = await bootstrap.ensure(progress);
        return result.pythonExecutable;
      },
    );
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    outputChannel.appendLine(`[activate] bootstrap failed: ${msg}`);
    void vscode.window
      .showErrorMessage(
        `Repo AI: setup failed — ${msg}`,
        "Show logs",
      )
      .then((choice) => {
        if (choice === "Show logs") {
          outputChannel?.show(true);
        }
      });
    return;
  }

  const dataDir = context.globalStorageUri.fsPath;
  backendProcess = new BackendProcess({
    pythonExecutable,
    port: config.port,
    dataDir,
    outputChannel,
  });
  context.subscriptions.push({
    dispose: () => backendProcess?.dispose(),
  });

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
        const msg = e instanceof Error ? e.message : String(e);
        void vscode.window
          .showErrorMessage(`Repo AI: backend failed to start — ${msg}`, "Show logs")
          .then((choice) => choice === "Show logs" && outputChannel?.show(true));
        return;
      }

      const client = backendProcess!.getClient();

      // ── 4. Auto-index workspace ──
      const workspaceFolders = vscode.workspace.workspaceFolders;
      if (workspaceFolders && workspaceFolders.length > 0) {
        const rootPath = workspaceFolders[0].uri.fsPath;
        progress.report({ message: `Indexing ${path.basename(rootPath)}...` });
        try {
          await client.indexDirectory(rootPath);
        } catch (e) {
          void vscode.window.showWarningMessage(
            `Repo AI: auto-index failed — ${String(e)}. Run 'Repo AI: Rebuild Index' once ready.`,
          );
        }
      }

      // ── 5. Status bar ──
      const statusBar = new StatusBar(client);
      context.subscriptions.push({ dispose: () => statusBar.dispose() });
      statusBar.startPolling();

      // ── 6. Chat panel ──
      const { provider: chatProvider, disposable: chatDisposable } =
        ChatPanel.register(context, client);
      context.subscriptions.push(chatDisposable);

      const showChat = (
        question: string,
        answer: string,
        sources: string[],
      ) => chatProvider.showAnswer(question, answer, sources);

      // ── 7. Commands ──
      context.subscriptions.push(
        registerAskQuestion(context, client, showChat),
        registerExplainSelection(context, client, showChat),
        registerFindRelated(context, client),
        registerExplainRepo(context, client),
        registerRebuildIndex(context, client, statusBar),
        vscode.commands.registerCommand("repoAwareAI.openChat", () =>
          vscode.commands.executeCommand("repoAwareAI.chatView.focus"),
        ),
        vscode.commands.registerCommand("repoAwareAI.openRepoMap", () =>
          RepoMapPanel.createOrShow(context, client),
        ),
        vscode.commands.registerCommand("repoAwareAI.setApiKey", async () => {
          const key = await _promptApiKey(context);
          if (key) {
            void vscode.window.showInformationMessage(
              "Repo AI: API key saved. Reload the window for it to take effect.",
            );
          }
        }),
      );

      // ── 8. Code intelligence providers ──
      context.subscriptions.push(...CodeLensProvider.register(client));
      context.subscriptions.push(...HoverProvider.register(client));

      // ── 9. Auto-index on save ──
      const indexManager = new IndexManager(client);
      indexManager.enable();
      context.subscriptions.push({ dispose: () => indexManager.dispose() });
    },
  );
}

export function deactivate(): void {
  backendProcess?.dispose();
  backendProcess = null;
  outputChannel?.dispose();
  outputChannel = null;
}

// ──────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────

async function _promptApiKey(
  context: vscode.ExtensionContext,
): Promise<string | undefined> {
  const key = await vscode.window.showInputBox({
    prompt: "Enter your Google API key for Gemini",
    placeHolder: "AIza...",
    password: true,
    ignoreFocusOut: true,
    validateInput: (v) =>
      v && v.trim().length > 8 ? null : "API key looks too short — paste the full value",
  });
  if (!key) {
    return undefined;
  }
  // Store ONLY in SecretStorage. We deliberately do NOT mirror into
  // settings.json — that would leak the key into settings exports / Settings
  // Sync, which the configuration description explicitly promises against.
  // `loadApiKey()` reads SecretStorage as its fallback, so this is sufficient.
  await context.secrets.store("googleApiKey", key);
  return key;
}

function _registerSetApiKey(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("repoAwareAI.setApiKey", async () => {
      const key = await _promptApiKey(context);
      if (key) {
        void vscode.window.showInformationMessage(
          "Repo AI: API key saved. Reload the window to start the backend.",
        );
      }
    }),
  );
}
