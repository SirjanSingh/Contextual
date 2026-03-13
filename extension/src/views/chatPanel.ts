/**
 * Chat panel — WebviewViewProvider for the sidebar chat UI.
 * Plain HTML + vanilla JS (no React) for minimal bundle impact.
 * Streams answers from /query/stream using SSE.
 */

import * as vscode from "vscode";
import { BackendClient } from "../services/backendClient";

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
      async (msg: { type: string; question?: string }) => {
        if (msg.type === "ask" && msg.question) {
          await this._handleQuestion(msg.question);
        } else if (msg.type === "clear") {
          this._post({ type: "clear" });
        }
      },
    );
  }

  /** Show a question+answer pair in the chat (called from askQuestion command). */
  showAnswer(question: string, answer: string, sources: string[]): void {
    this._ensureVisible();
    this._post({ type: "answer", question, answer, sources });
  }

  private _ensureVisible(): void {
    this.view?.show(true);
  }

  private async _handleQuestion(question: string): Promise<void> {
    this._post({ type: "thinking", question });
    try {
      const res = await this.client.query(question);
      this._post({
        type: "answer",
        question,
        answer: res.answer,
        sources: res.sources,
      });
    } catch (e) {
      this._post({ type: "error", message: String(e) });
    }
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
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    #header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 12px;
      border-bottom: 1px solid var(--border);
    }
    #header h1 { font-size: 13px; font-weight: 600; opacity: 0.9; }
    #clear-btn {
      background: transparent;
      border: none;
      color: var(--fg);
      cursor: pointer;
      opacity: 0.6;
      font-size: 16px;
      line-height: 1;
    }
    #clear-btn:hover { opacity: 1; }
    #messages {
      flex: 1;
      overflow-y: auto;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 12px;
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
    #input-area {
      padding: 10px;
      border-top: 1px solid var(--border);
      display: flex;
      gap: 8px;
    }
    #question {
      flex: 1;
      background: var(--input-bg);
      color: var(--input-fg);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 6px 10px;
      font-size: 13px;
      resize: none;
      font-family: inherit;
      min-height: 36px;
      max-height: 120px;
    }
    #question:focus { outline: 1px solid var(--accent); border-color: var(--accent); }
    #send-btn {
      background: var(--btn-bg);
      color: var(--btn-fg);
      border: none;
      border-radius: 4px;
      padding: 6px 14px;
      cursor: pointer;
      font-size: 13px;
      white-space: nowrap;
    }
    #send-btn:hover { opacity: 0.85; }
    #send-btn:disabled { opacity: 0.5; cursor: default; }
    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100%;
      opacity: 0.5;
      gap: 8px;
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
      <div style="font-size:11px">Ctrl+Shift+A to ask from anywhere</div>
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

    function renderAnswer({ question, answer, sources }) {
      if (!hasMessages) {
        messages.innerHTML = '';
        hasMessages = true;
      }
      // Remove thinking bubble if present
      const thinking = messages.querySelector('.thinking');
      if (thinking) thinking.remove();

      // User message
      const userDiv = document.createElement('div');
      userDiv.className = 'msg user';
      userDiv.textContent = question;
      messages.appendChild(userDiv);

      // Assistant message
      const aiDiv = document.createElement('div');
      aiDiv.className = 'msg assistant';
      const pre = document.createElement('pre');
      pre.textContent = answer;
      aiDiv.appendChild(pre);

      if (sources && sources.length) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        sourcesDiv.innerHTML = '📎 ' + sources.slice(0, 5).map(s => {
          const label = s.split('/').pop() || s;
          return \`<a onclick="openSource('\${s.replace(/'/g, "\\\\'")}')">\${label}</a>\`;
        }).join(' · ');
        aiDiv.appendChild(sourcesDiv);
      }

      messages.appendChild(aiDiv);
      messages.scrollTop = messages.scrollHeight;
    }

    function openSource(source) {
      vscode.postMessage({ type: 'openSource', source });
    }

    function ask() {
      const q = questionEl.value.trim();
      if (!q) return;
      questionEl.value = '';
      sendBtn.disabled = true;

      if (!hasMessages) {
        messages.innerHTML = '';
        hasMessages = true;
      }

      // Show thinking
      const t = document.createElement('div');
      t.className = 'msg assistant thinking';
      t.textContent = '⏳ Thinking...';
      messages.appendChild(t);
      messages.scrollTop = messages.scrollHeight;

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
      vscode.postMessage({ type: 'clear' });
    });

    window.addEventListener('message', (event) => {
      const msg = event.data;
      if (msg.type === 'answer') {
        renderAnswer(msg);
        sendBtn.disabled = false;
      } else if (msg.type === 'thinking') {
        // already handled inline
      } else if (msg.type === 'error') {
        document.querySelector('.thinking')?.remove();
        const e = document.createElement('div');
        e.className = 'msg error';
        e.textContent = '⚠ ' + msg.message;
        messages.appendChild(e);
        messages.scrollTop = messages.scrollHeight;
        sendBtn.disabled = false;
      } else if (msg.type === 'clear') {
        messages.innerHTML = '<div class="empty-state"><div class="icon">💬</div><div>Ask anything about your codebase</div></div>';
        hasMessages = false;
      }
    });
  </script>
</body>
</html>`;
  }
}
