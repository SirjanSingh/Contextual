/**
 * Repo Map panel — interactive 3D codebase visualization using Three.js.
 * Opens as a full editor panel (not sidebar — needs space for 3D view).
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

    // Load graph data when panel becomes visible
    this._loadGraph();

    panel.webview.onDidReceiveMessage(
      async (msg: { type: string; file?: string }) => {
        if (msg.type === "openFile" && msg.file) {
          try {
            const uris = await vscode.workspace.findFiles(
              `**/${msg.file}`,
              "**/node_modules/**",
              1,
            );
            const uri = uris.length > 0 ? uris[0] : vscode.Uri.file(msg.file);
            await vscode.window.showTextDocument(uri);
          } catch {
            void vscode.window.showWarningMessage(
              `Could not open: ${msg.file}`,
            );
          }
        } else if (msg.type === "refresh") {
          await this._loadGraph();
        }
      },
      null,
      this.disposables,
    );

    panel.onDidDispose(() => this.dispose(), null, this.disposables);
  }

  private async _loadGraph(): Promise<void> {
    try {
      const [graph, clusters] = await Promise.all([
        this.client.dependencyGraph(),
        this.client.semanticClusters(),
      ]);
      this.panel.webview.postMessage({ type: "graphData", graph, clusters });
    } catch (e) {
      this.panel.webview.postMessage({ type: "error", message: String(e) });
    }
  }

  private _getHtml(): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Repo Map</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif; overflow: hidden; }
    #canvas { width: 100vw; height: 100vh; display: block; }
    #overlay {
      position: fixed; top: 16px; left: 16px; right: 16px;
      display: flex; align-items: center; gap: 12px; pointer-events: none; z-index: 10;
    }
    #title { font-size: 18px; font-weight: 700; color: #58a6ff; text-shadow: 0 0 12px #58a6ff44; }
    #stats { font-size: 12px; opacity: 0.7; }
    #search {
      margin-left: auto; pointer-events: all;
      background: #161b22; border: 1px solid #30363d; border-radius: 6px;
      padding: 6px 12px; color: #c9d1d9; font-size: 13px; outline: none; width: 200px;
    }
    #search:focus { border-color: #58a6ff; }
    #refresh-btn {
      pointer-events: all; background: #1f6feb; border: none; border-radius: 4px;
      color: white; padding: 6px 12px; cursor: pointer; font-size: 12px;
    }
    #refresh-btn:hover { background: #388bfd; }
    #tooltip {
      position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
      background: #161b22; border: 1px solid #30363d; border-radius: 8px;
      padding: 10px 16px; font-size: 12px; max-width: 300px; text-align: center;
      opacity: 0; transition: opacity 0.2s; pointer-events: none;
    }
    #loading {
      position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
      font-size: 16px; color: #58a6ff;
    }
    #legend {
      position: fixed; bottom: 16px; right: 16px;
      font-size: 11px; opacity: 0.6; line-height: 1.8;
    }
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <div id="overlay">
    <div id="title">🗺 Repo Map</div>
    <div id="stats">Loading...</div>
    <input id="search" placeholder="Search files..." />
    <button id="refresh-btn" onclick="refresh()">↻ Refresh</button>
  </div>
  <div id="tooltip"></div>
  <div id="loading">Loading dependency graph...</div>
  <div id="legend">
    ● Size = chunk count<br>
    ── = import dependency<br>
    Click node to open file
  </div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script>
    const vscode = acquireVsCodeApi();
    const canvas = document.getElementById('canvas');
    const tooltip = document.getElementById('tooltip');
    const statsEl = document.getElementById('stats');
    const searchEl = document.getElementById('search');
    const loadingEl = document.getElementById('loading');

    // Three.js setup
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x0d1117, 0.002);

    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 2000);
    camera.position.set(0, 0, 300);

    // Lights
    scene.add(new THREE.AmbientLight(0x404080, 0.5));
    const pointLight = new THREE.PointLight(0x58a6ff, 2, 500);
    pointLight.position.set(0, 100, 100);
    scene.add(pointLight);

    // Mouse controls (simple orbit)
    let isDragging = false, prevX = 0, prevY = 0;
    let rotX = 0, rotY = 0, zoom = 300;
    const pivot = new THREE.Group();
    scene.add(pivot);

    canvas.addEventListener('mousedown', e => { isDragging = true; prevX = e.clientX; prevY = e.clientY; });
    canvas.addEventListener('mouseup', () => isDragging = false);
    canvas.addEventListener('mousemove', e => {
      if (!isDragging) { checkHover(e); return; }
      rotY += (e.clientX - prevX) * 0.5;
      rotX += (e.clientY - prevY) * 0.5;
      prevX = e.clientX; prevY = e.clientY;
      pivot.rotation.y = THREE.MathUtils.degToRad(rotY);
      pivot.rotation.x = THREE.MathUtils.degToRad(rotX);
    });
    canvas.addEventListener('wheel', e => {
      zoom = Math.max(50, Math.min(800, zoom + e.deltaY * 0.3));
      camera.position.z = zoom;
    });
    canvas.addEventListener('click', handleClick);

    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    const nodeObjects = [];
    const clusterColors = [
      0x58a6ff, 0x3fb950, 0xd29922, 0xf78166, 0xbc8cff,
      0x76e3ea, 0xffa657, 0xff7b72
    ];

    function buildGraph(graph, clusters) {
      loadingEl.style.display = 'none';
      // Clear existing
      while (pivot.children.length) pivot.remove(pivot.children[0]);
      nodeObjects.length = 0;

      const fileCluster = {};
      if (clusters && clusters.clusters) {
        for (const c of clusters.clusters) {
          for (const f of c.files) fileCluster[f] = c.id;
        }
      }

      const nodes = graph.nodes || [];
      const edges = graph.edges || [];
      statsEl.textContent = \`\${nodes.length} files · \${edges.length} deps\`;

      // Force-directed layout (simple approximation with golden angle)
      const phi = Math.PI * (3 - Math.sqrt(5));
      const radius = Math.max(80, nodes.length * 3);

      nodes.forEach((node, i) => {
        const y = 1 - (i / (nodes.length - 1 || 1)) * 2;
        const r = Math.sqrt(1 - y * y);
        const theta = phi * i;
        const pos = new THREE.Vector3(
          Math.cos(theta) * r * radius,
          y * radius,
          Math.sin(theta) * r * radius,
        );

        const size = Math.max(2, Math.min(12, node.chunkCount * 0.5));
        const geo = new THREE.SphereGeometry(size, 16, 16);
        const clusterId = fileCluster[node.id] ?? 0;
        const color = clusterColors[clusterId % clusterColors.length];
        const mat = new THREE.MeshPhongMaterial({
          color,
          emissive: color,
          emissiveIntensity: 0.2,
          transparent: true,
          opacity: 0.85,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.copy(pos);
        mesh.userData = { node, originalColor: color, pos };
        pivot.add(mesh);
        nodeObjects.push(mesh);
      });

      // Draw edges (lines)
      const nodePos = {};
      nodeObjects.forEach(m => { nodePos[m.userData.node.id] = m.position; });

      const lineMat = new THREE.LineBasicMaterial({ color: 0x30363d, transparent: true, opacity: 0.3 });
      for (const edge of edges) {
        const from = nodePos[edge.source];
        const to = nodePos[edge.target];
        if (!from || !to) continue;
        const geo = new THREE.BufferGeometry().setFromPoints([from, to]);
        pivot.add(new THREE.Line(geo, lineMat));
      }
    }

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    function checkHover(e) {
      mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(nodeObjects);
      if (hits.length > 0) {
        const node = hits[0].object.userData.node;
        tooltip.innerHTML = \`<strong>\${node.label}</strong><br>\${node.chunkCount} chunks\`;
        tooltip.style.opacity = '1';
        canvas.style.cursor = 'pointer';
      } else {
        tooltip.style.opacity = '0';
        canvas.style.cursor = 'default';
      }
    }

    function handleClick(e) {
      mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(nodeObjects);
      if (hits.length > 0) {
        const node = hits[0].object.userData.node;
        vscode.postMessage({ type: 'openFile', file: node.id });
      }
    }

    searchEl.addEventListener('input', () => {
      const q = searchEl.value.toLowerCase();
      nodeObjects.forEach(m => {
        const label = (m.userData.node.label || '').toLowerCase();
        const match = !q || label.includes(q);
        m.material.opacity = match ? 0.85 : 0.1;
        m.material.emissiveIntensity = match ? (q ? 0.6 : 0.2) : 0;
      });
    });

    function refresh() { vscode.postMessage({ type: 'refresh' }); }

    // Animation loop
    function animate() {
      requestAnimationFrame(animate);
      if (!isDragging) {
        pivot.rotation.y += 0.001; // Slow auto-rotation
      }
      renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('message', (event) => {
      const msg = event.data;
      if (msg.type === 'graphData') {
        buildGraph(msg.graph, msg.clusters);
      } else if (msg.type === 'error') {
        loadingEl.textContent = '⚠ ' + msg.message;
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
