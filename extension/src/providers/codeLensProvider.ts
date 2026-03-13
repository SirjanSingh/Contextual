/**
 * CodeLens provider — shows "🔍 N related chunks across M files" above each function/class.
 * Only fires after document is idle for 2s. Caches per-file, invalidates on save.
 */

import * as vscode from "vscode";
import { BackendClient } from "../services/backendClient";
import { CODELENS_DEBOUNCE_MS, CODELENS_MAX_SYMBOLS } from "../constants";

const SUPPORTED_LANGUAGES = [
  "python",
  "typescript",
  "javascript",
  "typescriptreact",
  "javascriptreact",
  "go",
  "rust",
  "java",
  "csharp",
  "cpp",
  "c",
];

interface CachedLens {
  lenses: vscode.CodeLens[];
  version: number;
}

export class CodeLensProvider implements vscode.CodeLensProvider {
  private cache = new Map<string, CachedLens>();
  private debounceTimers = new Map<string, ReturnType<typeof setTimeout>>();
  private _onDidChangeCodeLenses = new vscode.EventEmitter<void>();
  readonly onDidChangeCodeLenses = this._onDidChangeCodeLenses.event;

  constructor(private client: BackendClient) {}

  static register(client: BackendClient): vscode.Disposable[] {
    const provider = new CodeLensProvider(client);
    const disposables: vscode.Disposable[] = [];

    for (const lang of SUPPORTED_LANGUAGES) {
      disposables.push(
        vscode.languages.registerCodeLensProvider({ language: lang }, provider),
      );
    }

    // Invalidate cache on save
    disposables.push(
      vscode.workspace.onDidSaveTextDocument((doc) => {
        provider.cache.delete(doc.uri.toString());
        provider._onDidChangeCodeLenses.fire();
      }),
    );

    disposables.push(provider._onDidChangeCodeLenses);
    return disposables;
  }

  provideCodeLenses(
    document: vscode.TextDocument,
    token: vscode.CancellationToken,
  ): vscode.CodeLens[] | Thenable<vscode.CodeLens[]> {
    const key = document.uri.toString();
    const cached = this.cache.get(key);

    if (cached && cached.version === document.version) {
      return cached.lenses;
    }

    // Return empty immediately; schedule async computation with debounce
    this._scheduleCompute(document);
    return [];
  }

  private _scheduleCompute(document: vscode.TextDocument): void {
    const key = document.uri.toString();

    const existing = this.debounceTimers.get(key);
    if (existing) {
      clearTimeout(existing);
    }

    const timer = setTimeout(async () => {
      this.debounceTimers.delete(key);
      await this._computeLenses(document);
    }, CODELENS_DEBOUNCE_MS);

    this.debounceTimers.set(key, timer);
  }

  private async _computeLenses(document: vscode.TextDocument): Promise<void> {
    try {
      // Use VS Code symbol API
      const symbols = await vscode.commands.executeCommand<
        vscode.DocumentSymbol[]
      >("vscode.executeDocumentSymbolProvider", document.uri);

      if (!symbols || symbols.length === 0) {
        return;
      }

      // Flatten and limit
      const flat = flattenSymbols(symbols)
        .filter(
          (s) =>
            s.kind === vscode.SymbolKind.Function ||
            s.kind === vscode.SymbolKind.Method ||
            s.kind === vscode.SymbolKind.Class,
        )
        .slice(0, CODELENS_MAX_SYMBOLS);

      // Batch search for all symbols
      const lenses: vscode.CodeLens[] = [];

      for (const sym of flat) {
        const query = sym.name;
        try {
          const chunks = await this.client.search(query, 6);
          if (chunks.length === 0) {
            continue;
          }

          const fileSet = new Set(chunks.map((c) => c.source));
          const label = `🔍 ${chunks.length} related chunks across ${fileSet.size} file${fileSet.size > 1 ? "s" : ""}`;

          lenses.push(
            new vscode.CodeLens(sym.range, {
              title: label,
              command: "repoAwareAI.findRelated",
              tooltip: `Find code related to ${sym.name}`,
            }),
          );
        } catch {
          // Skip symbols that fail
        }
      }

      this.cache.set(document.uri.toString(), {
        lenses,
        version: document.version,
      });
      this._onDidChangeCodeLenses.fire();
    } catch {
      // Fail silently
    }
  }

  dispose(): void {
    for (const t of this.debounceTimers.values()) {
      clearTimeout(t);
    }
    this._onDidChangeCodeLenses.dispose();
  }
}

function flattenSymbols(
  symbols: vscode.DocumentSymbol[],
): vscode.DocumentSymbol[] {
  const result: vscode.DocumentSymbol[] = [];
  for (const s of symbols) {
    result.push(s);
    if (s.children?.length) {
      result.push(...flattenSymbols(s.children));
    }
  }
  return result;
}
