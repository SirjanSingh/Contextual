/**
 * findRelated — Ctrl+Shift+R → finds semantically related files/chunks.
 * The "killer demo" — instant semantic similarity with QuickPick UI.
 */

import * as vscode from "vscode";
import { BackendClient } from "../services/backendClient";

export function registerFindRelated(
  context: vscode.ExtensionContext,
  client: BackendClient,
): vscode.Disposable {
  return vscode.commands.registerCommand(
    "repoAwareAI.findRelated",
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        void vscode.window.showWarningMessage("Repo AI: Open a file first.");
        return;
      }

      // Use selection if any, otherwise use first 800 chars of document
      const query = editor.selection.isEmpty
        ? editor.document.getText().slice(0, 800)
        : editor.document.getText(editor.selection);

      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: "Repo AI",
          cancellable: false,
        },
        async (progress) => {
          progress.report({ message: "Finding related code..." });
          try {
            const chunks = await client.search(query, 12);

            if (!chunks.length) {
              void vscode.window.showInformationMessage(
                "Repo AI: No related code found.",
              );
              return;
            }

            // Build QuickPick items
            const items = chunks.map((c) => ({
              label: `$(file-code) ${vscode.workspace.asRelativePath(c.source)}`,
              description: `score: ${c.score.toFixed(3)}`,
              detail: c.text.split("\n")[0].trim().slice(0, 100),
              chunk: c,
            }));

            const picked = await vscode.window.showQuickPick(items, {
              title: "Related Code",
              placeHolder: "Select a chunk to navigate to",
              matchOnDescription: true,
              matchOnDetail: true,
            });

            if (!picked) {
              return;
            }

            // Open file at chunk position. Backend always emits POSIX
            // separators, so normalise once and try workspace-relative first.
            const sourcePath = picked.chunk.source.replace(/\\/g, "/");
            const folders = vscode.workspace.workspaceFolders ?? [];

            let fileUri: vscode.Uri | undefined;
            for (const folder of folders) {
              const candidate = vscode.Uri.joinPath(folder.uri, sourcePath);
              try {
                await vscode.workspace.fs.stat(candidate);
                fileUri = candidate;
                break;
              } catch {
                // not in this folder; try next
              }
            }
            if (!fileUri) {
              const uris = await vscode.workspace.findFiles(
                `**/${sourcePath}`,
                "**/node_modules/**",
                1,
              );
              fileUri = uris[0];
            }
            if (!fileUri) {
              void vscode.window.showWarningMessage(
                `Repo AI: could not open ${sourcePath}`,
              );
              return;
            }

            const doc = await vscode.workspace.openTextDocument(fileUri);
            const ed = await vscode.window.showTextDocument(doc, {
              preview: true,
            });
            const pos = doc.positionAt(
              Math.min(picked.chunk.start_char, doc.getText().length),
            );
            ed.revealRange(
              new vscode.Range(pos, pos),
              vscode.TextEditorRevealType.InCenter,
            );
            ed.selection = new vscode.Selection(pos, pos);
          } catch (e) {
            void vscode.window.showErrorMessage(`Repo AI: ${String(e)}`);
          }
        },
      );
    },
  );
}
