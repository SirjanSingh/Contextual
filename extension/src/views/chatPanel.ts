/**
 * Chat panel — WebviewViewProvider for the sidebar chat UI.
 *
 * Plain HTML + vanilla JS to keep the extension bundle small.
 *
 * Features:
 *   - Live token streaming via SSE from /query/stream.
 *   - Click a source citation to jump to the file at the chunk's start line.
 *   - Surfaces backend errors as inline messages instead of swallowing them.
 */

import * as vscode from "vscode";
import { BackendClient } from "../services/backendClient";

interface OutgoingMessage {
  type: "ask" | "clear" | "openSource";
  question?: string;
  source?: string;
}

export class ChatPanel implements vscode.WebviewViewProvider {
  private view?: vscode.WebviewView;

  constructor(
    private context: vscode.ExtensionContext,
    private client: BackendClient,
  ) {}

  static register(
    context: vscode.ExtensionContext,
    client: BackendClient,
  ): { provider: ChatPanel; disposable: vscode.Disposable } {
    const provider = new ChatPanel(context, client);
    const disposable = vscode.window.registerWebviewViewProvider(
      "repoAwareAI.chatView",
      provider,
      { webviewOptions: { retainContextWhenHidden: true } },
    );
    return { provider, disposable };
  }

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken,
  ): void {
    this.view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.html = this._getHtml();

    webviewView.webview.onDidReceiveMessage(
      async (msg: OutgoingMessage) => {
        if (msg.type === "ask" && msg.question) {
          await this._handleQuestionStream(msg.question);
        } else if (msg.type === "clear") {
          this._post({ type: "clear" });
        } else if (msg.type === "openSource" && msg.source) {
          await this._openSource(msg.source);
        }
      },
    );
  }

  /** Show a question+answer pair from outside (e.g. askQuestion command). */
  showAnswer(question: string, answer: string, sources: string[]): void {
    this._ensureVisible();
    this._post({ type: "answer", question, answer, sources });
  }

  private _ensureVisible(): void {
    this.view?.show(true);
  }

  /** Stream the answer token-by-token and finish with a sources event. */
  private async _handleQuestionStream(question: string): Promise<void> {
    this._post({ type: "stream-start", question });
    try {
      const stream = this.client.queryStream(question);
      for await (const event of stream) {
        if (event.kind === "chunk") {
          this._post({ type: "stream-token", text: event.text });
        } else if (event.kind === "sources") {
          this._post({ type: "stream-sources", sources: event.sources });
        } else if (event.kind === "error") {
          this._post({ type: "error", message: event.message });
          return;
        }
      }
      this._post({ type: "stream-end" });
    } catch (e) {
      this._post({ type: "error", message: String(e) });
    }
  }

  /**
   * Open a source citation like `path/to/file.py:1234-2000`. The numbers are
   * character offsets — we convert them to a line position once the file is
   * open.
   */
  private async _openSource(source: string): Promise<void> {
    const match = source.match(/^(.*?)(?::(\d+)-(\d+))?$/);
    if (!match) {
      return;
    }
    const relPath = match[1].replace(/\\/g, "/");
    const startChar = match[2] ? parseInt(match[2], 10) : 0;

    const folders = vscode.workspace.workspaceFolders;
    if (!folders || folders.length === 0) {
      return;
    }

    let uri: vscode.Uri | undefined;
    // Try relative-to-workspace first (fast, no filesystem walk).
    for (const folder of folders) {
      const candidate = vscode.Uri.joinPath(folder.uri, relPath);
      try {
        await vscode.workspace.fs.stat(candidate);
        uri = candidate;
        break;
      } catch {
        // not in this workspace folder
      }
    }

    // Fall back to a glob search by filename.
    if (!uri) {
      const matches = await vscode.workspace.findFiles(
        `**/${relPath}`,
        "**/node_modules/**",
        1,
      );
      uri = matches[0];
    }

    if (!uri) {
      void vscode.window.showWarningMessage(`Repo AI: could not open ${relPath}`);
      return;
    }

    const doc = await vscode.workspace.openTextDocument(uri);
    const editor = await vscode.window.showTextDocument(doc, { preview: true });
    const pos = doc.positionAt(Math.min(startChar, doc.getText().length));
    editor.revealRange(
      new vscode.Range(pos, pos),
      vscode.TextEditorRevealType.InCenter,
    );
    editor.selection = new vscode.Selection(pos, pos);
  }

  private _post(msg: unknown): void {
    this.view?.webview.postMessage(msg);
  }

  private _getHtml(): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Repo AI Chat</title>
  <style>
    :root {
      --bg: var(--vscode-sideBar-background, #1e1e1e);
      --fg: var(--vscode-foreground, #cccccc);
      --input-bg: var(--vscode-input-background, #2d2d2d);
      --input-fg: var(--vscode-input-foreground, #cccccc);
      --btn-bg: var(--vscode-button-background, #0e639c);
      --btn-fg: var(--vscode-button-foreground, #ffffff);
      --border: var(--vscode-input-border, #3c3c3c);
      --accent: var(--vscode-textLink-foreground, #3794ff);
      --msg-bg: var(--vscode-editor-inactiveSelectionBackground, #2a2d2e);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--fg);
      font-family: var(--vscode-font-family, "Segoe UI", sans-serif);
      font-size: var(--vscode-font-size, 13px);
      display: flex; flex-direction: column;
      height: 100vh; overflow: hidden;
    }
    #header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 8px 12px; border-bottom: 1px solid var(--border);
    }
    #header h1 { font-size: 13px; font-weight: 600; opacity: 0.9; }
    #clear-btn {
      background: transparent; border: none; color: var(--fg);
      cursor: pointer; opacity: 0.6; font-size: 16px; line-height: 1;
    }
    #clear-btn:hover { opacity: 1; }
    #messages {
      flex: 1; overflow-y: auto; padding: 12px;
      display: flex; flex-direction: column; gap: 12px;
    }
    .msg { border-radius: 6px; padding: 10px 12px; }
    .msg.user { background: var(--btn-bg); color: var(--btn-fg); align-self: flex-end; max-width: 85%; }
    .msg.assistant { background: var(--msg-bg); align-self: flex-start; max-width: 100%; }
    .msg.thinking { opacity: 0.6; font-style: italic; }
    .msg.error { background: #5a1a1a; color: #f48771; }
    .sources { margin-top: 8px; font-size: 11px; opacity: 0.7; }
    .sources a { color: var(--accent); text-decoration: none; cursor: pointer; }
    .sources a:hover { text-decoration: underline; }
    pre { white-space: pre-wrap; word-break: break-word; font-family: var(--vscode-editor-font-family, monospace); font-size: 12px; }
    .cursor { display: inline-block; width: 6px; height: 1em; background: var(--accent); margin-left: 2px; vertical-align: text-bottom; animation: blink 1s steps(2) infinite; }
    @keyframes blink { 50% { opacity: 0; } }
    #input-area {
      padding: 10px; border-top: 1px solid var(--border);
      display: flex; gap: 8px;
    }
    #question {
      flex: 1; background: var(--input-bg); color: var(--input-fg);
      border: 1px solid var(--border); border-radius: 4px;
      padding: 6px 10px; font-size: 13px; resize: none;
      font-family: inherit; min-height: 36px; max-height: 120px;
    }
    #question:focus { outline: 1px solid var(--accent); border-color: var(--accent); }
    #send-btn {
      background: var(--btn-bg); color: var(--btn-fg);
      border: none; border-radius: 4px; padding: 6px 14px;
      cursor: pointer; font-size: 13px; white-space: nowrap;
    }
    #send-btn:hover { opacity: 0.85; }
    #send-btn:disabled { opacity: 0.5; cursor: default; }
    .empty-state {
      display: flex; flex-direction: column; align-items: center;
      justify-content: center; height: 100%; opacity: 0.5; gap: 8px;
    }
    .empty-state .icon { font-size: 32px; }
  </style>
</head>
<body>
  <div id="header">
    <h1>🧠 Repo AI Chat</h1>
    <button id="clear-btn" title="Clear chat">✕</button>
  </div>
  <div id="messages">
    <div class="empty-state">
      <div class="icon">💬</div>
      <div>Ask anything about your codebase</div>
      <div style="font-size:11px">Ctrl+Shift+A from anywhere</div>
    </div>
  </div>
  <div id="input-area">
    <textarea id="question" rows="1" placeholder="Ask about your codebase..."></textarea>
    <button id="send-btn">Send</button>
  </div>

  <script>
    const vscode = acquireVsCodeApi();
    const messages = document.getElementById('messages');
    const questionEl = document.getElementById('question');
    const sendBtn = document.getElementById('send-btn');
    const clearBtn = document.getElementById('clear-btn');
    let hasMessages = false;
    let activeAssistant = null; // { pre, sourcesDiv, cursor }

    function clearEmptyState() {
      if (!hasMessages) {
        messages.innerHTML = '';
        hasMessages = true;
      }
    }

    function escapeHtml(s) {
      return String(s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function renderUserMessage(question) {
      const userDiv = document.createElement('div');
      userDiv.className = 'msg user';
      userDiv.textContent = question;
      messages.appendChild(userDiv);
    }

    function startStreaming(question) {
      clearEmptyState();
      renderUserMessage(question);

      const aiDiv = document.createElement('div');
      aiDiv.className = 'msg assistant';
      const pre = document.createElement('pre');
      const cursor = document.createElement('span');
      cursor.className = 'cursor';
      pre.appendChild(cursor);
      aiDiv.appendChild(pre);
      messages.appendChild(aiDiv);
      messages.scrollTop = messages.scrollHeight;

      activeAssistant = { aiDiv, pre, cursor, sourcesDiv: null };
    }

    function appendToken(text) {
      if (!activeAssistant) return;
      const node = document.createTextNode(text);
      activeAssistant.pre.insertBefore(node, activeAssistant.cursor);
      messages.scrollTop = messages.scrollHeight;
    }

    function attachSources(sources) {
      if (!activeAssistant) return;
      if (!sources || !sources.length) return;
      const sourcesDiv = document.createElement('div');
      sourcesDiv.className = 'sources';
      sourcesDiv.innerHTML = '📎 ' + sources.slice(0, 5).map((s, i) => {
        const label = (s.split('/').pop() || s).replace(/:.*$/, '');
        return '<a data-source-i="' + i + '">' + escapeHtml(label) + '</a>';
      }).join(' · ');
      sourcesDiv.querySelectorAll('a[data-source-i]').forEach((el) => {
        el.addEventListener('click', () => {
          const idx = parseInt(el.getAttribute('data-source-i'), 10);
          vscode.postMessage({ type: 'openSource', source: sources[idx] });
        });
      });
      activeAssistant.aiDiv.appendChild(sourcesDiv);
      activeAssistant.sourcesDiv = sourcesDiv;
    }

    function endStream() {
      if (activeAssistant && activeAssistant.cursor) {
        activeAssistant.cursor.remove();
      }
      activeAssistant = null;
      sendBtn.disabled = false;
    }

    function renderError(msg) {
      if (activeAssistant && activeAssistant.cursor) {
        activeAssistant.cursor.remove();
      }
      const e = document.createElement('div');
      e.className = 'msg error';
      e.textContent = '⚠ ' + msg;
      messages.appendChild(e);
      messages.scrollTop = messages.scrollHeight;
      activeAssistant = null;
      sendBtn.disabled = false;
    }

    /** Static answer (e.g. arrived from showAnswer). */
    function renderAnswer({ question, answer, sources }) {
      clearEmptyState();
      renderUserMessage(question);
      const aiDiv = document.createElement('div');
      aiDiv.className = 'msg assistant';
      const pre = document.createElement('pre');
      pre.textContent = answer;
      aiDiv.appendChild(pre);
      messages.appendChild(aiDiv);
      activeAssistant = { aiDiv, pre, cursor: null, sourcesDiv: null };
      attachSources(sources);
      activeAssistant = null;
      messages.scrollTop = messages.scrollHeight;
    }

    function ask() {
      const q = questionEl.value.trim();
      if (!q) return;
      questionEl.value = '';
      sendBtn.disabled = true;
      vscode.postMessage({ type: 'ask', question: q });
    }

    sendBtn.addEventListener('click', ask);
    questionEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        ask();
      }
    });
    clearBtn.addEventListener('click', () => {
      messages.innerHTML = '<div class="empty-state"><div class="icon">💬</div><div>Ask anything about your codebase</div></div>';
      hasMessages = false;
      activeAssistant = null;
      vscode.postMessage({ type: 'clear' });
    });

    window.addEventListener('message', (event) => {
      const msg = event.data;
      switch (msg.type) {
        case 'stream-start': startStreaming(msg.question); break;
        case 'stream-token': appendToken(msg.text); break;
        case 'stream-sources': attachSources(msg.sources); break;
        case 'stream-end': endStream(); break;
        case 'answer': renderAnswer(msg); sendBtn.disabled = false; break;
        case 'error': renderError(msg.message); break;
        case 'clear':
          messages.innerHTML = '<div class="empty-state"><div class="icon">💬</div><div>Ask anything about your codebase</div></div>';
          hasMessages = false;
          activeAssistant = null;
          break;
      }
    });
  </script>
</body>
</html>`;
  }
}
