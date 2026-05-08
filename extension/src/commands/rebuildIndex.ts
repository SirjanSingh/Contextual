/** rebuildIndex — forces a full re-index via /index/rebuild with progress notification */

import * as vscode from "vscode";
import { BackendClient } from "../services/backendClient";
import { StatusBar } from "../views/statusBar";

export function registerRebuildIndex(
  context: vscode.ExtensionContext,
  client: BackendClient,
  statusBar: StatusBar,
): vscode.Disposable {
  return vscode.commands.registerCommand(
    "repoAwareAI.rebuildIndex",
    async () => {
      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: "Repo AI",
          cancellable: false,
        },
        async (progress) => {
          progress.report({ message: "Rebuilding index..." });
          statusBar.setIndexing();
          try {
            await client.rebuildIndex();
            void vscode.window.showInformationMessage(
              "Repo AI: Index rebuilt successfully.",
            );
          } catch (e) {
            void vscode.window.showErrorMessage(
              `Repo AI: Rebuild failed — ${String(e)}`,
            );
          }
        },
      );
    },
  );
}
