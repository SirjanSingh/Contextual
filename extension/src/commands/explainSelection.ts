/** explainSelection — right-click → explain selected code in codebase context */

import * as vscode from "vscode";
import { BackendClient } from "../services/backendClient";

export function registerExplainSelection(
  context: vscode.ExtensionContext,
  client: BackendClient,
  showChat: (question: string, answer: string, sources: string[]) => void,
): vscode.Disposable {
  return vscode.commands.registerCommand(
    "repoAwareAI.explainSelection",
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor || editor.selection.isEmpty) {
        void vscode.window.showWarningMessage(
          "Repo AI: Select some code first.",
        );
        return;
      }

      const selection = editor.document.getText(editor.selection);
      const filePath = editor.document.fileName;
      const question = `Explain what this code does and how it fits in the codebase:\n\nFile: ${filePath}\n\`\`\`\n${selection}\n\`\`\``;

      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: "Repo AI",
          cancellable: false,
        },
        async (progress) => {
          progress.report({ message: "Explaining selection..." });
          try {
            const res = await client.query(question);
            showChat(
              `Explain selection from ${vscode.workspace.asRelativePath(filePath)}`,
              res.answer,
              res.sources,
            );
          } catch (e) {
            void vscode.window.showErrorMessage(`Repo AI: ${String(e)}`);
          }
        },
      );
    },
  );
}
