/** askQuestion command — Ctrl+Shift+A → InputBox → answer in Chat panel */

import * as vscode from "vscode";
import { BackendClient } from "../services/backendClient";

export function registerAskQuestion(
  context: vscode.ExtensionContext,
  client: BackendClient,
  showChat: (question: string, answer: string, sources: string[]) => void,
): vscode.Disposable {
  return vscode.commands.registerCommand(
    "repoAwareAI.askQuestion",
    async () => {
      const question = await vscode.window.showInputBox({
        prompt: "Ask about your codebase...",
        placeHolder: "e.g. How does the indexing pipeline work?",
      });
      if (!question) {
        return;
      }

      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: "Repo AI",
          cancellable: false,
        },
        async (progress) => {
          progress.report({ message: "Searching codebase..." });
          try {
            const res = await client.query(question);
            showChat(question, res.answer, res.sources);
          } catch (e) {
            void vscode.window.showErrorMessage(`Repo AI: ${String(e)}`);
          }
        },
      );
    },
  );
}
