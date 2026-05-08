/**
 * Hover provider — shows RAG-powered hover cards when hovering function/class names.
 * Includes 2s timeout + LRU cache.
 */

import * as vscode from "vscode";
import { BackendClient } from "../services/backendClient";
import { HOVER_TIMEOUT_MS, HOVER_CACHE_SIZE } from "../constants";

export class HoverProvider implements vscode.HoverProvider {
  // Simple LRU cache (key = symbol name)
  private cache = new Map<string, vscode.Hover>();

  constructor(private client: BackendClient) {}

  static register(client: BackendClient): vscode.Disposable[] {
    const provider = new HoverProvider(client);
    return [
      vscode.languages.registerHoverProvider(
        [
          "python",
          "typescript",
          "javascript",
          "typescriptreact",
          "javascriptreact",
          "go",
          "rust",
          "java",
          "csharp",
        ],
        provider,
      ),
    ];
  }

  async provideHover(
    document: vscode.TextDocument,
    position: vscode.Position,
    token: vscode.CancellationToken,
  ): Promise<vscode.Hover | null> {
    const wordRange = document.getWordRangeAtPosition(position, /[\w.]+/);
    if (!wordRange) {
      return null;
    }

    const word = document.getText(wordRange);
    if (word.length < 3 || /^\d+$/.test(word)) {
      return null; // Skip short words and pure numbers
    }

    // Cached?
    if (this.cache.has(word)) {
      return this.cache.get(word)!;
    }

    try {
      const chunks = await Promise.race([
        this.client.search(word, 5),
        new Promise<null>((_, reject) =>
          setTimeout(() => reject(new Error("timeout")), HOVER_TIMEOUT_MS),
        ),
      ]);

      if (!chunks || token.isCancellationRequested) {
        return null;
      }

      if (chunks.length === 0) {
        return null;
      }

      const fileSet = new Set(chunks.map((c) => c.source));
      const topChunk = chunks[0];
      const snippet = topChunk.text.split("\n").slice(0, 3).join("\n").trim();

      const md = new vscode.MarkdownString("", true);
      md.isTrusted = true;
      md.appendMarkdown(`**Repo AI: \`${word}\`**\n\n`);
      md.appendMarkdown(
        `**Used in:** ${[...fileSet].slice(0, 3).join(", ")}\n\n`,
      );
      if (snippet) {
        md.appendMarkdown(`**Context:**\n`);
        md.appendCodeblock(snippet, document.languageId);
      }

      const hover = new vscode.Hover(md, wordRange);

      // LRU eviction
      if (this.cache.size >= HOVER_CACHE_SIZE) {
        const firstKey = this.cache.keys().next().value;
        if (firstKey !== undefined) {
          this.cache.delete(firstKey);
        }
      }
      this.cache.set(word, hover);

      return hover;
    } catch {
      return null; // Timeout or error — fail silently
    }
  }
}
