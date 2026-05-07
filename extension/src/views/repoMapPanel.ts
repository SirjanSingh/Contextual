/**
 * Repo Map panel — community-aware codebase visualization.
 * Shows communities as force-directed bubbles, processes as step flows.
 */

import * as vscode from "vscode";
import { BackendClient } from "../services/backendClient";

export class RepoMapPanel {
  private static current: RepoMapPanel | undefined;
  private readonly panel: vscode.WebviewPanel;
  private disposables: vscode.Disposable[] = [];

  static createOrShow(
    context: vscode.ExtensionContext,
    client: BackendClient,
  ): void {
    if (RepoMapPanel.current) {
      RepoMapPanel.current.panel.reveal(vscode.ViewColumn.One);
      return;
    }
    const panel = vscode.window.createWebviewPanel(
      "repoAwareAI.repoMap",
      "Repo Map",
      vscode.ViewColumn.One,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
      },
    );
    RepoMapPanel.current = new RepoMapPanel(panel, client);
  }

  private constructor(
    panel: vscode.WebviewPanel,
    private client: BackendClient,
  ) {
    this.panel = panel;
    panel.webview.options = { enableScripts: true };
    panel.webview.html = this._getHtml();

    this._loadRepoMap();

    panel.webview.onDidReceiveMessage(
      async (msg: { type: string; id?: string; file?: string }) => {
        switch (msg.type) {
          case "refresh":
            await this._loadRepoMap();
            break;
          case "loadCommunity":
            if (msg.id) await this._loadCommunity(msg.id);
            break;
          case "loadProcess":
            if (msg.id) await this._loadProcess(msg.id);
            break;
          case "loadSymbol":
            if (msg.id) await this._loadSymbol(msg.id);
            break;
          case "openFile":
            if (msg.file) await this._openFile(msg.file);
            break;
        }
      },
      null,
      this.disposables,
    );

    panel.onDidDispose(() => this.dispose(), null, this.disposables);
  }

  private async _loadRepoMap(): Promise<void> {
    try {
      const summary = await this.client.repoMapSummary();
      this.panel.webview.postMessage({ type: "repoMapData", summary });
    } catch (e) {
      this.panel.webview.postMessage({ type: "error", message: String(e) });
    }
  }

  private async _loadCommunity(id: string): Promise<void> {
    try {
      const detail = await this.client.communityDetail(id);
      this.panel.webview.postMessage({ type: "communityDetail", detail });
    } catch (e) {
      this.panel.webview.postMessage({ type: "error", message: String(e) });
    }
  }

  private async _loadProcess(id: string): Promise<void> {
    try {
      const detail = await this.client.processDetail(id);
      this.panel.webview.postMessage({ type: "processDetail", detail });
    } catch (e) {
      this.panel.webview.postMessage({ type: "error", message: String(e) });
    }
  }

  private async _loadSymbol(id: string): Promise<void> {
    try {
      const detail = await this.client.symbolDetail(id);
      this.panel.webview.postMessage({ type: "symbolDetail", detail });
    } catch (e) {
      this.panel.webview.postMessage({ type: "error", message: String(e) });
    }
  }

  private async _openFile(file: string): Promise<void> {
    try {
      const uris = await vscode.workspace.findFiles(
        `**/${file}`,
        "**/node_modules/**",
        1,
      );
      const uri = uris.length > 0 ? uris[0] : vscode.Uri.file(file);
      await vscode.window.showTextDocument(uri);
    } catch {
      void vscode.window.showWarningMessage(`Could not open: ${file}`);
    }
  }

  private _getHtml(): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Repo Map</title>
  <style>
    *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
    html, body { width: 100%; height: 100%; overflow: hidden; background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif; font-size: 13px; }

    /* Layout */
    #app { display: flex; flex-direction: column; height: 100vh; }
    #topbar {
      display: flex; align-items: center; gap: 10px; padding: 8px 14px;
      background: #161b22; border-bottom: 1px solid #30363d; flex-shrink: 0;
    }
    #main { display: flex; flex: 1; overflow: hidden; }
    #sidebar {
      width: 220px; flex-shrink: 0; background: #161b22;
      border-right: 1px solid #30363d; display: flex; flex-direction: column; overflow: hidden;
    }
    #canvas-wrap { flex: 1; position: relative; overflow: hidden; }
    #detail-panel {
      width: 260px; flex-shrink: 0; background: #161b22;
      border-left: 1px solid #30363d; overflow-y: auto; display: none;
    }
    #detail-panel.visible { display: block; }

    /* Topbar */
    #title { font-size: 15px; font-weight: 700; color: #58a6ff; }
    #stats { font-size: 11px; color: #8b949e; }
    #search-input {
      margin-left: auto; background: #0d1117; border: 1px solid #30363d;
      border-radius: 5px; padding: 4px 10px; color: #c9d1d9; font-size: 12px;
      outline: none; width: 180px;
    }
    #search-input:focus { border-color: #58a6ff; }
    #refresh-btn {
      background: #1f6feb; border: none; border-radius: 4px;
      color: white; padding: 5px 10px; cursor: pointer; font-size: 12px;
    }
    #refresh-btn:hover { background: #388bfd; }

    /* Sidebar tabs */
    .tab-bar {
      display: flex; border-bottom: 1px solid #30363d; flex-shrink: 0;
    }
    .tab {
      flex: 1; padding: 8px 4px; text-align: center; cursor: pointer;
      font-size: 11px; letter-spacing: 0.5px; color: #8b949e;
      border-bottom: 2px solid transparent; transition: color 0.15s;
    }
    .tab:hover { color: #c9d1d9; }
    .tab.active { color: #58a6ff; border-bottom-color: #58a6ff; }

    /* Sidebar list */
    #sidebar-list { flex: 1; overflow-y: auto; padding: 6px 0; }
    .list-item {
      padding: 6px 12px; cursor: pointer; display: flex;
      align-items: center; gap: 8px; border-radius: 4px; margin: 1px 4px;
      transition: background 0.1s;
    }
    .list-item:hover { background: #21262d; }
    .list-item.active { background: #1c2a3a; }
    .list-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .list-label { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; }
    .list-badge {
      font-size: 10px; background: #21262d; color: #8b949e;
      padding: 1px 5px; border-radius: 10px; flex-shrink: 0;
    }
    .list-type { font-size: 10px; color: #8b949e; flex-shrink: 0; }

    /* Canvas */
    #canvas { display: block; width: 100%; height: 100%; }
    #loading-msg {
      position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
      color: #58a6ff; font-size: 14px;
    }
    #hover-tip {
      position: absolute; pointer-events: none; background: #161b22;
      border: 1px solid #30363d; border-radius: 6px; padding: 6px 10px;
      font-size: 11px; max-width: 220px; opacity: 0; transition: opacity 0.15s;
    }

    /* Detail panel */
    .detail-header {
      padding: 12px 14px; border-bottom: 1px solid #30363d;
      display: flex; align-items: center; gap: 8px;
    }
    .detail-title { font-size: 13px; font-weight: 600; color: #c9d1d9; flex: 1; word-break: break-all; }
    .detail-close {
      cursor: pointer; color: #8b949e; font-size: 16px; line-height: 1;
      padding: 0 2px;
    }
    .detail-close:hover { color: #c9d1d9; }
    .detail-section { padding: 10px 14px; border-bottom: 1px solid #21262d; }
    .detail-section-title { font-size: 10px; color: #8b949e; letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 6px; }
    .detail-stat { display: flex; justify-content: space-between; font-size: 12px; margin: 3px 0; }
    .detail-stat-val { color: #58a6ff; font-weight: 600; }
    .member-item {
      display: flex; align-items: center; gap: 6px; padding: 4px 0;
      cursor: pointer; border-radius: 3px;
    }
    .member-item:hover { color: #58a6ff; }
    .member-name { font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .member-label { font-size: 10px; color: #8b949e; flex-shrink: 0; }
    .file-path { font-size: 10px; color: #8b949e; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

    /* Process flow */
    .process-flow { display: flex; align-items: center; flex-wrap: wrap; gap: 4px; padding: 4px 0; }
    .flow-step {
      background: #21262d; border: 1px solid #30363d; border-radius: 4px;
      padding: 3px 8px; font-size: 11px; cursor: pointer; white-space: nowrap;
    }
    .flow-step:hover { border-color: #58a6ff; color: #58a6ff; }
    .flow-arrow { color: #8b949e; font-size: 12px; }
    .flow-step.entry { border-color: #3fb950; color: #3fb950; }
    .flow-step.terminal { border-color: #f78166; color: #f78166; }

    /* Empty state */
    .empty-state { padding: 20px 14px; text-align: center; color: #8b949e; font-size: 12px; }
  </style>
</head>
<body>
<div id="app">
  <div id="topbar">
    <div id="title">Repo Map</div>
    <div id="stats">Loading...</div>
    <input id="search-input" placeholder="Search..." />
    <button id="refresh-btn" onclick="onRefresh()">Refresh</button>
  </div>
  <div id="main">
    <div id="sidebar">
      <div class="tab-bar">
        <div class="tab active" id="tab-communities" onclick="switchTab('communities')">Communities</div>
        <div class="tab" id="tab-processes" onclick="switchTab('processes')">Processes</div>
      </div>
      <div id="sidebar-list"></div>
    </div>
    <div id="canvas-wrap">
      <canvas id="canvas"></canvas>
      <div id="loading-msg">Loading repo map...</div>
      <div id="hover-tip"></div>
    </div>
    <div id="detail-panel">
      <div id="detail-content"></div>
    </div>
  </div>
</div>

<script>
const vscode = acquireVsCodeApi();
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const loadingMsg = document.getElementById('loading-msg');
const hoverTip = document.getElementById('hover-tip');
const statsEl = document.getElementById('stats');
const sidebarList = document.getElementById('sidebar-list');
const detailPanel = document.getElementById('detail-panel');
const detailContent = document.getElementById('detail-content');
const searchInput = document.getElementById('search-input');

// Palette
const PALETTE = [
  '#58a6ff', '#3fb950', '#d29922', '#f78166', '#bc8cff',
  '#76e3ea', '#ffa657', '#ff7b72', '#7ee787', '#a5d6ff',
  '#ffa8a3', '#cae8ff', '#d2a8ff', '#ffdf5d', '#56d364',
];

let summary = null;
let activeTab = 'communities';
let selectedId = null;
let searchQ = '';

// Canvas state
let nodes = []; // { id, label, x, y, vx, vy, r, color, data }
let edges = []; // { source, target }
let panX = 0, panY = 0, scale = 1;
let isPanning = false, lastPanX = 0, lastPanY = 0;
let animFrame = null;
let simRunning = false;

// ── Data Loading ──────────────────────────────────────────

function onRefresh() { vscode.postMessage({ type: 'refresh' }); }

function onRepoMapData(s) {
  summary = s;
  loadingMsg.style.display = 'none';
  const { stats } = s;
  statsEl.textContent =
    \`\${stats.total_nodes} symbols · \${stats.total_communities} communities · \${stats.total_processes} processes\`;
  renderSidebar();
  buildCommunityGraph();
}

// ── Sidebar ────────────────────────────────────────────────

function switchTab(tab) {
  activeTab = tab;
  document.getElementById('tab-communities').classList.toggle('active', tab === 'communities');
  document.getElementById('tab-processes').classList.toggle('active', tab === 'processes');
  renderSidebar();
}

function renderSidebar() {
  if (!summary) return;
  sidebarList.innerHTML = '';
  const q = searchQ.toLowerCase();

  if (activeTab === 'communities') {
    const list = summary.communities.filter(c =>
      !q || c.heuristic_label.toLowerCase().includes(q) || c.label.toLowerCase().includes(q)
    );
    if (!list.length) {
      sidebarList.innerHTML = '<div class="empty-state">No communities found</div>';
      return;
    }
    list.forEach((c, i) => {
      const color = PALETTE[i % PALETTE.length];
      const el = document.createElement('div');
      el.className = 'list-item' + (selectedId === c.id ? ' active' : '');
      el.innerHTML =
        \`<div class="list-dot" style="background:\${color}"></div>
        <div class="list-label" title="\${c.heuristic_label}">\${c.heuristic_label}</div>
        <div class="list-badge">\${c.symbol_count}</div>\`;
      el.onclick = () => { selectedId = c.id; renderSidebar(); vscode.postMessage({ type: 'loadCommunity', id: c.id }); highlightNode(c.id); };
      sidebarList.appendChild(el);
    });
  } else {
    const list = summary.processes.filter(p =>
      !q || p.label.toLowerCase().includes(q)
    );
    if (!list.length) {
      sidebarList.innerHTML = '<div class="empty-state">No processes found</div>';
      return;
    }
    list.forEach(p => {
      const el = document.createElement('div');
      el.className = 'list-item' + (selectedId === p.id ? ' active' : '');
      const typeColor = p.process_type === 'cross_community' ? '#ffa657' : '#58a6ff';
      el.innerHTML =
        \`<div class="list-dot" style="background:\${typeColor}"></div>
        <div class="list-label" title="\${p.label}">\${p.label}</div>
        <div class="list-type">\${p.step_count}s</div>\`;
      el.onclick = () => { selectedId = p.id; renderSidebar(); vscode.postMessage({ type: 'loadProcess', id: p.id }); };
      sidebarList.appendChild(el);
    });
  }
}

// ── Detail Panel ───────────────────────────────────────────

function showDetail(html) {
  detailContent.innerHTML = html;
  detailPanel.classList.add('visible');
}

function hideDetail() {
  detailPanel.classList.remove('visible');
  selectedId = null;
  renderSidebar();
}

function onCommunityDetail(detail) {
  const colorIdx = (summary?.communities || []).findIndex(c => c.id === detail.id);
  const color = PALETTE[colorIdx >= 0 ? colorIdx % PALETTE.length : 0];

  const membersHtml = detail.members.slice(0, 40).map(m =>
    \`<div class="member-item" onclick="vscode.postMessage({type:'loadSymbol',id:'\${esc(m.id)}'})" title="\${esc(m.file_path)}">
      <span class="member-label">\${esc(m.label)}</span>
      <span class="member-name">\${esc(m.name)}</span>
    </div>\`
  ).join('');
  const moreCount = detail.members.length - 40;

  showDetail(\`
    <div class="detail-header">
      <div class="list-dot" style="background:\${color};width:10px;height:10px;border-radius:50%;flex-shrink:0"></div>
      <div class="detail-title">\${esc(detail.heuristic_label)}</div>
      <div class="detail-close" onclick="hideDetail()">&#x2715;</div>
    </div>
    <div class="detail-section">
      <div class="detail-section-title">Stats</div>
      <div class="detail-stat"><span>Symbols</span><span class="detail-stat-val">\${detail.symbol_count}</span></div>
      <div class="detail-stat"><span>Cohesion</span><span class="detail-stat-val">\${(detail.cohesion * 100).toFixed(1)}%</span></div>
      <div class="detail-stat"><span>Internal edges</span><span class="detail-stat-val">\${detail.internal_relationships.length}</span></div>
    </div>
    <div class="detail-section">
      <div class="detail-section-title">Symbols (\${detail.members.length})</div>
      \${membersHtml}
      \${moreCount > 0 ? \`<div style="font-size:11px;color:#8b949e;padding-top:4px">+\${moreCount} more</div>\` : ''}
    </div>
  \`);
}

function onProcessDetail(detail) {
  const stepsHtml = detail.steps.map((s, i) => {
    const isFirst = i === 0;
    const isLast = i === detail.steps.length - 1;
    const cls = isFirst ? 'entry' : isLast ? 'terminal' : '';
    const arrow = i < detail.steps.length - 1 ? '<span class="flow-arrow">&#x2192;</span>' : '';
    return \`<span class="flow-step \${cls}" onclick="openSymbolFile('\${esc(s.file_path)}')" title="\${esc(s.file_path)}">\${esc(s.name)}</span>\${arrow}\`;
  }).join('');

  const typeColor = detail.process_type === 'cross_community' ? '#ffa657' : '#58a6ff';
  showDetail(\`
    <div class="detail-header">
      <div class="list-dot" style="background:\${typeColor};width:10px;height:10px;border-radius:50%;flex-shrink:0"></div>
      <div class="detail-title">\${esc(detail.label)}</div>
      <div class="detail-close" onclick="hideDetail()">&#x2715;</div>
    </div>
    <div class="detail-section">
      <div class="detail-section-title">Stats</div>
      <div class="detail-stat"><span>Type</span><span class="detail-stat-val">\${detail.process_type}</span></div>
      <div class="detail-stat"><span>Steps</span><span class="detail-stat-val">\${detail.step_count}</span></div>
      <div class="detail-stat"><span>Communities</span><span class="detail-stat-val">\${detail.communities.length}</span></div>
    </div>
    <div class="detail-section">
      <div class="detail-section-title">Execution Flow</div>
      <div class="process-flow">\${stepsHtml}</div>
    </div>
  \`);
}

function onSymbolDetail(detail) {
  const callersHtml = detail.callers.slice(0, 15).map(r =>
    \`<div class="member-item" onclick="openSymbolFile('\${esc(r.file_path)}')" title="\${esc(r.file_path)}">
      <span class="member-name">\${esc(r.name)}</span>
      <span class="member-label" style="margin-left:auto;color:#8b949e">\${(r.confidence * 100).toFixed(0)}%</span>
    </div>\`
  ).join('');
  const calleesHtml = detail.callees.slice(0, 15).map(r =>
    \`<div class="member-item" onclick="openSymbolFile('\${esc(r.file_path)}')" title="\${esc(r.file_path)}">
      <span class="member-name">\${esc(r.name)}</span>
      <span class="member-label" style="margin-left:auto;color:#8b949e">\${(r.confidence * 100).toFixed(0)}%</span>
    </div>\`
  ).join('');

  showDetail(\`
    <div class="detail-header">
      <div class="detail-title">\${esc(detail.name)}</div>
      <div class="detail-close" onclick="hideDetail()">&#x2715;</div>
    </div>
    <div class="detail-section">
      <div class="detail-section-title">Info</div>
      <div class="detail-stat"><span>Type</span><span class="detail-stat-val">\${esc(detail.label)}</span></div>
      <div class="detail-stat"><span>Lines</span><span class="detail-stat-val">\${detail.start_line}-\${detail.end_line}</span></div>
      <div class="detail-stat"><span>Exported</span><span class="detail-stat-val">\${detail.is_exported ? 'Yes' : 'No'}</span></div>
      <div class="file-path" style="margin-top:4px">\${esc(detail.file_path)}</div>
      <button style="margin-top:8px;background:#1f6feb;border:none;border-radius:4px;color:white;padding:4px 10px;cursor:pointer;font-size:11px"
        onclick="openSymbolFile('\${esc(detail.file_path)}')">Open File</button>
    </div>
    \${detail.callers.length ? \`<div class="detail-section">
      <div class="detail-section-title">Called By (\${detail.callers.length})</div>
      \${callersHtml}
    </div>\` : ''}
    \${detail.callees.length ? \`<div class="detail-section">
      <div class="detail-section-title">Calls (\${detail.callees.length})</div>
      \${calleesHtml}
    </div>\` : ''}
  \`);
}

function openSymbolFile(path) {
  vscode.postMessage({ type: 'openFile', file: path });
}

function esc(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ── Force Graph ────────────────────────────────────────────

function buildCommunityGraph() {
  if (!summary) return;
  const wrap = document.getElementById('canvas-wrap');
  const W = wrap.clientWidth, H = wrap.clientHeight;
  canvas.width = W;
  canvas.height = H;

  nodes = summary.communities.map((c, i) => {
    const angle = (i / summary.communities.length) * Math.PI * 2;
    const dist = Math.min(W, H) * 0.3;
    return {
      id: c.id,
      label: c.heuristic_label,
      x: W / 2 + Math.cos(angle) * dist,
      y: H / 2 + Math.sin(angle) * dist,
      vx: 0, vy: 0,
      r: Math.max(18, Math.min(50, 10 + c.symbol_count * 0.5)),
      color: PALETTE[i % PALETTE.length],
      data: c,
    };
  });

  // Edges: communities that share processes
  edges = [];
  if (summary.processes) {
    for (const p of summary.processes) {
      if (p.process_type === 'cross_community' && summary.communities.length > 1) {
        // Just connect first two communities as placeholder since we don't have per-process community list here
        // Real edges come from community detail
      }
    }
  }

  panX = 0; panY = 0; scale = 1;
  startSim();
}

function startSim() {
  simRunning = true;
  let tick = 0;
  function step() {
    if (!simRunning) return;
    simulateTick(tick++);
    draw();
    if (tick < 200) animFrame = requestAnimationFrame(step);
    else { simRunning = false; draw(); }
  }
  if (animFrame) cancelAnimationFrame(animFrame);
  animFrame = requestAnimationFrame(step);
}

function simulateTick(tick) {
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2;
  const alpha = Math.max(0.01, 1 - tick / 180);

  for (let i = 0; i < nodes.length; i++) {
    const a = nodes[i];
    // Gravity toward center
    a.vx += (cx - a.x) * 0.01 * alpha;
    a.vy += (cy - a.y) * 0.01 * alpha;

    // Repulsion from other nodes
    for (let j = i + 1; j < nodes.length; j++) {
      const b = nodes[j];
      const dx = a.x - b.x, dy = a.y - b.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const minDist = a.r + b.r + 20;
      if (dist < minDist * 2) {
        const force = (minDist * 2 - dist) / dist * 0.3 * alpha;
        a.vx += dx * force; a.vy += dy * force;
        b.vx -= dx * force; b.vy -= dy * force;
      }
    }
  }

  for (const n of nodes) {
    n.vx *= 0.7; n.vy *= 0.7;
    n.x += n.vx; n.y += n.vy;
    // Boundary
    n.x = Math.max(n.r + 10, Math.min(W - n.r - 10, n.x));
    n.y = Math.max(n.r + 10, Math.min(H - n.r - 10, n.y));
  }
}

function draw() {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.save();
  ctx.translate(panX, panY);
  ctx.scale(scale, scale);

  const q = searchQ.toLowerCase();

  // Draw edges
  for (const e of edges) {
    const src = nodes.find(n => n.id === e.source);
    const tgt = nodes.find(n => n.id === e.target);
    if (!src || !tgt) continue;
    ctx.beginPath();
    ctx.moveTo(src.x, src.y);
    ctx.lineTo(tgt.x, tgt.y);
    ctx.strokeStyle = 'rgba(88,166,255,0.2)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Draw nodes
  for (const n of nodes) {
    const isSelected = n.id === selectedId;
    const matchSearch = !q || n.label.toLowerCase().includes(q);
    const alpha = matchSearch ? 1 : 0.25;

    // Outer glow for selected
    if (isSelected) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r + 6, 0, Math.PI * 2);
      ctx.fillStyle = n.color + '33';
      ctx.fill();
    }

    // Circle
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    ctx.fillStyle = hexWithAlpha(n.color, 0.18 * alpha);
    ctx.fill();
    ctx.strokeStyle = hexWithAlpha(n.color, (isSelected ? 1 : 0.7) * alpha);
    ctx.lineWidth = isSelected ? 2 : 1.5;
    ctx.stroke();

    // Label
    if (n.r > 14 || isSelected) {
      ctx.fillStyle = \`rgba(201,209,217,\${alpha})\`;
      ctx.font = \`\${Math.max(9, Math.min(13, n.r * 0.45))}px 'Segoe UI', sans-serif\`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const maxW = n.r * 1.6;
      const words = n.label.split('/').pop() || n.label;
      ctx.fillText(trimText(ctx, words, maxW), n.x, n.y);

      // Symbol count badge
      ctx.font = '9px "Segoe UI", sans-serif';
      ctx.fillStyle = \`rgba(139,148,158,\${alpha})\`;
      ctx.fillText(n.data.symbol_count + ' syms', n.x, n.y + n.r * 0.55);
    }
  }

  ctx.restore();
}

function hexWithAlpha(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return \`rgba(\${r},\${g},\${b},\${alpha})\`;
}

function trimText(ctx, text, maxW) {
  if (ctx.measureText(text).width <= maxW) return text;
  let t = text;
  while (t.length > 3 && ctx.measureText(t + '...').width > maxW) t = t.slice(0, -1);
  return t + '...';
}

function highlightNode(id) {
  selectedId = id;
  draw();
}

// ── Canvas Interactions ───────────────────────────────────

function canvasToWorld(cx, cy) {
  return { x: (cx - panX) / scale, y: (cy - panY) / scale };
}

function nodeAtPoint(wx, wy) {
  for (const n of nodes) {
    const dx = n.x - wx, dy = n.y - wy;
    if (Math.sqrt(dx * dx + dy * dy) <= n.r) return n;
  }
  return null;
}

canvas.addEventListener('mousedown', e => {
  isPanning = true; lastPanX = e.clientX; lastPanY = e.clientY;
});
canvas.addEventListener('mouseup', e => {
  if (!isPanning) return;
  const dx = e.clientX - lastPanX, dy = e.clientY - lastPanY;
  if (Math.abs(dx) < 3 && Math.abs(dy) < 3) {
    // Click
    const rect = canvas.getBoundingClientRect();
    const w = canvasToWorld(e.clientX - rect.left, e.clientY - rect.top);
    const n = nodeAtPoint(w.x, w.y);
    if (n) {
      selectedId = n.id;
      renderSidebar();
      draw();
      if (activeTab === 'communities') {
        vscode.postMessage({ type: 'loadCommunity', id: n.id });
      }
    }
  }
  isPanning = false;
});
canvas.addEventListener('mousemove', e => {
  if (isPanning) {
    panX += e.clientX - lastPanX;
    panY += e.clientY - lastPanY;
    lastPanX = e.clientX; lastPanY = e.clientY;
    draw();
    return;
  }
  // Hover
  const rect = canvas.getBoundingClientRect();
  const w = canvasToWorld(e.clientX - rect.left, e.clientY - rect.top);
  const n = nodeAtPoint(w.x, w.y);
  if (n) {
    hoverTip.innerHTML = \`<strong>\${esc(n.label)}</strong><br>\${n.data.symbol_count} symbols · cohesion \${(n.data.cohesion * 100).toFixed(0)}%\`;
    hoverTip.style.left = (e.clientX + 12) + 'px';
    hoverTip.style.top = (e.clientY - 30) + 'px';
    hoverTip.style.opacity = '1';
    canvas.style.cursor = 'pointer';
  } else {
    hoverTip.style.opacity = '0';
    canvas.style.cursor = 'default';
  }
});
canvas.addEventListener('mouseleave', () => { hoverTip.style.opacity = '0'; isPanning = false; });
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const factor = e.deltaY > 0 ? 0.9 : 1.1;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  panX = mx - (mx - panX) * factor;
  panY = my - (my - panY) * factor;
  scale = Math.max(0.3, Math.min(3, scale * factor));
  draw();
}, { passive: false });

window.addEventListener('resize', () => {
  const wrap = document.getElementById('canvas-wrap');
  canvas.width = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
  draw();
});

// ── Search ─────────────────────────────────────────────────

searchInput.addEventListener('input', () => {
  searchQ = searchInput.value;
  renderSidebar();
  draw();
});

// ── Messages ───────────────────────────────────────────────

window.addEventListener('message', e => {
  const msg = e.data;
  if (msg.type === 'repoMapData') onRepoMapData(msg.summary);
  else if (msg.type === 'communityDetail') onCommunityDetail(msg.detail);
  else if (msg.type === 'processDetail') onProcessDetail(msg.detail);
  else if (msg.type === 'symbolDetail') onSymbolDetail(msg.detail);
  else if (msg.type === 'error') {
    loadingMsg.textContent = 'Error: ' + msg.message;
    loadingMsg.style.display = 'block';
  }
});
</script>
</body>
</html>`;
  }

  dispose(): void {
    RepoMapPanel.current = undefined;
    this.panel.dispose();
    while (this.disposables.length) {
      this.disposables.pop()?.dispose();
    }
  }
}
