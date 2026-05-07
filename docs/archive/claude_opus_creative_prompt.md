# Creative Frontend Prompt for Claude Opus 4.6

## Context & Mission

You are tasked with creating an exceptionally creative, visually stunning testing interface for **Contextual** - an advanced RAG-based code intelligence system that understands repositories at a deep level. This is not just another chatbot interface - this is a portal into the neural network of code understanding.

Think **Ghost in the Shell's cyberspace**, **Sword Art Online's UI overlays**, or **Psycho-Pass's Sibyl System interface**. The user should feel like they're interfacing with an omniscient AI entity that sees through the fabric of their codebase.41.

---

## Theme Direction: "Neural Code Nexus"

**Primary Inspiration:** Blend these aesthetics:

- **Ghost in the Shell**: Digital rain, hexagonal data streams, holographic projections, cyan/magenta neon accents
- **Cyberpunk 2077**: Glitch effects, chromatic aberration, scan lines, brutalist UI elements
- **Evangelion**: Angular geometric shapes, warning overlays, progress indicators with urgency
- **Tron Legacy**: Illuminated grid systems, particle trails, de-rezzed transitions

**Color Palette:**

- Primary: Deep space black (#0a0e1a) with subtle grid overlay
- Accent 1: Electric cyan (#00f7ff) - for active elements, data streams
- Accent 2: Hot magenta (#ff00aa) - for highlights, errors, emphasis
- Accent 3: Neural purple (#8b5cf6) - for AI responses, thinking states
- Accent 4: Warning amber (#fbbf24) - for system states
- Subtle: Translucent glass morphism with 5-10% opacity

---

## Three.js Core Features (MANDATORY)

### 1. **Animated 3D Background Scene**

Create a living, breathing 3D environment:

- **Particle system** representing code fragments floating in space
- **Neural network visualization**: Interconnected nodes forming/dissolving as the AI processes
- **Code matrix rain**: Actual code snippets from the repository falling like Matrix rain (extract from context)
- **Geometric wireframe structures** that rotate slowly - think DNA helix mixed with circuit boards
- **Dynamic camera** that subtly orbits/drifts (not static)
- **Depth-of-field blur** to add cinematic quality
- **Bloom/glow effects** on all neon elements

### 2. **Interactive 3D Elements**

- **Repository visualizer**: 3D graph showing file/folder structure as an explorable node graph
- **Query particles**: User queries spawn particles that travel through the neural network
- **Response materialization**: AI responses "compile" from floating data fragments
- **Gesture controls**: Mouse movement affects camera angle/particle flow

### 3. **Advanced WebGL Shaders**

- Custom fragment shaders for:
  - Holographic scan line effects
  - Data stream distortion waves
  - Chromatic aberration on hover states
  - CRT screen curvature (subtle)
  - Glitch artifacts during "thinking" states

---

## UI/UX Requirements

### Main Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│  [CONTEXTUAL LOGO - Glitchy animated]    [System Status HUD] │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│              ╔═══ 3D NEURAL BACKGROUND ═══╗                  │
│              ║    (Particle Systems,      ║                  │
│              ║     Code Rain, Network)    ║                  │
│   ┌──────────╫────────────────────────────╫────────┐        │
│   │          ║                            ║        │        │
│   │  Repo    ║    QUERY INPUT ZONE        ║  Live  │        │
│   │  Tree    ║   (Floating Glass Card)    ║  Feed  │        │
│   │  (3D)    ║                            ║  (Logs)│        │
│   │          ║   [AI Response Display]    ║        │        │
│   │          ║   (Animated Type-in)       ║        │        │
│   └──────────╫────────────────────────────╫────────┘        │
│              ║                            ║                  │
│              ╚════════════════════════════╝                  │
│                                                               │
│  [Performance Metrics]  [Model: Gemini]  [Index Status: ●]  │
└─────────────────────────────────────────────────────────────┘
```

### Key UI Components

1. **Query Input Area**
   - Floating glass-morphic card with subtle border glow
   - Holographic placeholder text that glitches
   - Auto-expanding textarea with neon cursor
   - Voice input visualizer (animated waveform)
   - Send button: 3D polygon that rotates on hover

2. **Response Display**
   - Typewriter effect with character-by-character reveal
   - Code blocks with syntax highlighting + glow effects
   - Inline annotations that expand on hover (3D transform)
   - "Thinking..." state shows neural network pulsing
   - Response cards with slide-in animation from particle formation

3. **Repository Explorer (Left Panel)**
   - 3D tree visualization (Three.js Force Graph)
   - Files as glowing nodes, connections as light trails
   - Click to explore, zoom in/out with mouse wheel
   - Heatmap overlay showing file relevance to query
   - Search spotlight effect that sweeps through nodes

4. **Live Activity Feed (Right Panel)**
   - Scrolling terminal-style log with scan lines
   - Query history with mini-previews
   - System events (indexing, embedding) with progress bars
   - Connection status with animated pulse
   - Particle trails connecting related queries

5. **Header HUD**
   - Animated logo with glitch effect on page load
   - System status indicators (CPU, memory as circular gauges)
   - Model selector dropdown with holographic options
   - Settings gear icon that opens radial menu (3D)

6. **Footer Stats Bar**
   - Real-time metrics: response time, tokens, chunks retrieved
   - Animated bar graphs with neon fills
   - Index health indicator with warning states
   - Last sync timestamp with countdown timer

---

## Advanced Animations & Interactions

### Micro-Animations (CRITICAL FOR "WOW" FACTOR)

1. **On Page Load:**
   - Logo materializes from particles (2s)
   - 3D background elements phase in with stagger
   - UI cards "compile" with digital assembly effect
   - Sound design: synthetic boot-up sequence (optional)

2. **On Directory Upload (NEW - CRITICAL):**
   - Drag enter: Full screen hexagonal grid overlay fades in (0.3s)
   - Drop: Files convert to data packets, shoot toward center (0.5s)
   - Upload start: Background shifts to blue, particle velocity increases
   - Progress: Neural network nodes light up in sequence matching progress %
   - Scanning: Horizontal scan bars sweep screen (Evangelion-style, 2s loop)
   - Parsing: Code fragments explode from files, orbit central sphere
   - Embedding: Fragments compress into glowing points, arrange in 3D space
   - Complete: Particle supernova from center, shake screen subtly (0.2s)
   - Transition: Fade to main UI with camera zoom effect (1s)

3. **On Query Submit:**
   - Input text converts to particles, shoots into 3D space
   - Neural network lights up showing "processing paths"
   - Thinking state: pulsing orb in center, shader wave effects
   - Background intensifies (more particles, faster movement)

4. **On Response Arrival:**
   - Particles coalesce into response card
   - Text types in with subtle glow trail
   - Code blocks slide in with backdrop blur
   - Success particle burst effect

5. **On Hover:**
   - UI elements: lift with 3D transform, glow intensifies, subtle rotation
   - Buttons: morph shape, emit particles
   - Cards: holographic scan line sweeps across
   - Links: lightning arc effect between characters

6. **On Error:**
   - Screen shake, red chromatic aberration
   - Error message glitches in with static effect
   - Background shifts to warning color temporarily
   - Particle system turns red briefly

### Performance Optimization

- Use `requestAnimationFrame` for all animations
- Implement level-of-detail (LOD) for 3D models
- Particle pooling to avoid GC pauses
- Lazy load non-critical shaders
- Debounce expensive operations
- Target 60fps on mid-range hardware

---

## Tech Stack Requirements

### Core Technologies

- **Three.js**: 3D rendering engine
- **React Three Fiber**: React renderer for Three.js (optional but recommended)
- **@react-three/drei**: Helpers for R3F
- **@react-three/postprocessing**: Bloom, glitch, CRT effects
- **Framer Motion**: UI component animations
- **GSAP**: Timeline-based complex animations
- **Tailwind CSS**: Base styling (with heavy customization)
- **React**: Component framework
- **TypeScript**: Type safety

### Additional Libraries

- **three-nebula**: Advanced particle systems
- **lamina**: Shader gradient materials
- **maath**: Math utilities for 3D
- **leva**: Debug GUI (for development)
- **react-syntax-highlighter**: Code display with glow theme
- **react-markdown**: Response formatting
- **zustand** or **jotai**: State management (lightweight)

### Shader Resources

- Custom GLSL shaders for unique effects
- ShaderToy inspiration for background
- Vertex displacement for dynamic geometry

---

## Functional Requirements

### Backend Integration

The frontend must connect to the existing Python FastAPI backend:

```javascript
// API Endpoints to integrate
POST / query; // Send user question
GET / index / status; // Get indexing status
POST / index / rebuild; // Trigger re-indexing
GET / health; // System health check
WS / ws; // WebSocket for live updates (optional)

// NEW ENDPOINTS NEEDED FOR DIRECTORY UPLOAD
POST / upload / directory; // Upload directory (multipart/form-data)
POST / upload / files; // Upload multiple files
GET / upload / progress; // Get upload/indexing progress
WS / ws / indexing; // Real-time indexing progress stream
DELETE / repository / clear; // Clear current repository
```

### Features to Implement

1. **🔥 EPIC Directory Upload System (NEW - CRITICAL):**

   **Upload Interface:**
   - **Drag & Drop Zone** - Full-screen overlay when dragging files
     - Hexagonal grid pattern lights up on hover
     - Pulsing border with cyber-glow effect
     - "INITIATING NEURAL SCAN..." text with glitch effect
     - Ghost in the Shell-style data acceptance animation
   - **File Browser Button** - Floating orb that opens native file picker
     - Supports: Single folder selection, Multiple files, or Directory tree
     - Icon: 3D rotating folder with holographic shimmer
     - On click: Emanates concentric scan rings
   - **Upload Modal/Overlay:**
     - Glass-morphic card with blueprint-style file tree preview
     - File count, total size with animated counter
     - "Analyze Repository" primary CTA button
     - Option to exclude patterns (.git, node_modules, etc.) with neon checkboxes

   **🎬 EPIC LOADING ANIMATIONS (Make this UNFORGETTABLE):**

   _Phase 1: Upload (0-30%)_
   - **Visual:** Files appear as data packets traveling through light tubes (Tron-style)
   - **3D Effect:** Each file generates a particle trail that feeds into central neural core
   - **Text:** "UPLOADING REPOSITORY... [N] FILES TRANSMITTED"
   - **Sound:** Pulsing data transfer whoosh (optional)
   - **Background:** Blue electric arcs between upload nodes

   _Phase 2: Scanning (30-50%)_
   - **Visual:** Evangelion-style hexagonal scan grid sweeps across screen
   - **3D Effect:** Repository visualizes as wireframe structure being "compiled"
   - **Shader:** Holographic scan lines revealing code fragments
   - **Text:** "SCANNING FILE STRUCTURE... ANALYZING DEPENDENCIES"
   - **Animation:** Random code snippets flash across screen (Matrix-style)

   _Phase 3: Parsing & Chunking (50-80%)_
   - **Visual:** Files explode into glowing fragments (AST nodes)
   - **3D Effect:** Code chunks orbit around central sphere, get "absorbed"
   - **Particle System:** Each chunk processed emits particle burst
   - **Text:** "PARSING CODE... [N/M] FILES PROCESSED"
   - **Progress Bar:** Neural network growth visualization (synapses forming)

   _Phase 4: Embedding & Indexing (80-95%)_
   - **Visual:** Ghost in the Shell brain-dive sequence inspired
   - **3D Effect:** Chunks transform into glowing embeddings in vector space
   - **Shader:** Data streams flowing into neural network nodes
   - **Text:** "GENERATING EMBEDDINGS... BUILDING KNOWLEDGE GRAPH"
   - **Animation:** 3D constellation of vectors forming connections

   _Phase 5: Finalization (95-100%)_
   - **Visual:** System pulse, all elements glow bright then normalize
   - **3D Effect:** Repository graph "locks in" with satisfying snap
   - **Particle Burst:** Celebratory explosion of particles
   - **Text:** "REPOSITORY INDEXED ✓ NEURAL LINK ESTABLISHED"
   - **Sound:** Success chime (Evangelion-style confirmation beep)

   **Real-Time Progress Display:**
   - Circular progress ring with percentage in center (neon glow pulse)
   - Stage indicator with animated transitions
   - Live file counter: "Processing: utils.py (145/892)"
   - ETA with countdown timer (warning amber color)
   - Estimated tokens/chunks in sidebar
   - Scrolling log of files being processed (terminal-style)
   - "Cancel" button with warning state (red chromatic aberration on hover)

   **Error States (Cyberpunk Style):**
   - **File too large:** "⚠️ FILE SIZE EXCEEDS NEURAL CAPACITY - COMPRESSION REQUIRED"
   - **Unsupported format:** "❌ UNKNOWN DATA FORMAT - DECODER NOT FOUND"
   - **Upload failed:** Screen glitch, red error particles, retry button pulsing
   - **Permission denied:** Lock icon with electric shock effect

   **Success State:**
   - Full-screen particle explosion in cyan/purple
   - "REPOSITORY SYNCHRONIZED" in large holographic text
   - Auto-transition to main chat interface (3s delay)
   - Summary card showing: Files indexed, Chunks created, Vector database size

2. **Intelligent Query Input:**
   - Multi-line support for complex questions
   - Syntax detection (highlight code in queries)
   - Recent queries dropdown with search
   - Query templates/examples

3. **Rich Response Display:**
   - Markdown rendering with custom theme
   - Code syntax highlighting (Prism with neon theme)
   - Collapsible sections for long responses
   - Copy buttons with success animations
   - File references as clickable links

4. **Repository Context:**
   - Show which files were retrieved for context
   - Similarity scores visualization
   - Click file to view snippet in modal
   - Toggle between different retrieval modes

5. **System Control:**
   - Index repository button (with progress)
   - Clear conversation history
   - Export chat as markdown/JSON
   - Theme customization panel

6. **Error Handling:**
   - Graceful degradation if 3D fails
   - Retry logic with exponential backoff
   - User-friendly error messages with anime references
   - Offline detection with reconnect

---

## Non-AI-Generated Uniqueness Markers

To ensure this doesn't look like typical AI-generated work:

1. **Asymmetric Layouts:** Not everything centered - break the grid intentionally
2. **Custom Cursor:** Replace cursor with animated crosshair or targeting reticle
3. **Easter Eggs:** Hidden animations (Konami code triggers special mode)
4. **Dynamic Typography:** Font weight/size changes based on system load
5. **Unexpected Interactions:** Drag to rotate 3D scene, right-click for context menu
6. **Sound Design:** Subtle UI sounds (whooshes, clicks, hums) - toggleable
7. **Glitch Art:** Intentional "bugs" that look artistic
8. **Personality:** Error messages with character ("Neural pathway overload - even I need coffee")
9. **Progressive Enhancement:** Features unlock as user explores
10. **Signature Detail:** Add a small "Designed for Contextual" badge with unique animation

---

## Code Quality Standards

- **Clean Architecture:** Separate concerns (components, hooks, utils, shaders)
- **Performance Monitoring:** Built-in FPS counter, memory usage display
- **Accessibility:** Keyboard navigation, ARIA labels, reduced motion mode
- **Responsive:** Mobile-friendly (simplified 3D on mobile)
- **Documentation:** Comments explaining shader math, animation timing
- **Error Boundaries:** React error boundaries with fallback UI
- **Testing:** Unit tests for utils, E2E for critical flows

---

## Backend Implementation Requirements (NEW)

### Python/FastAPI Backend Changes Needed

**1. New API Endpoints:**

```python
# app/api/upload.py (NEW FILE)

from fastapi import APIRouter, UploadFile, File, WebSocket
from typing import List
import asyncio

router = APIRouter(prefix="/upload", tags=["upload"])

@router.post("/directory")
async def upload_directory(files: List[UploadFile] = File(...)):
    """
    Accept multiple files representing a directory structure.
    Files should include relative paths in metadata.

    Returns:
    - upload_id: UUID for tracking progress
    - total_files: Number of files received
    - estimated_time: ETA for indexing
    """
    pass

@router.post("/files")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload individual files without directory structure"""
    pass

@router.get("/progress/{upload_id}")
async def get_progress(upload_id: str):
    """
    Get current progress of upload/indexing operation.

    Returns:
    - stage: "uploading" | "scanning" | "parsing" | "embedding" | "indexing" | "complete"
    - progress: 0-100
    - current_file: Filename currently being processed
    - files_processed: N/M count
    - eta_seconds: Estimated time remaining
    - errors: List of any errors encountered
    """
    pass

@router.websocket("/ws/indexing")
async def websocket_indexing_progress(websocket: WebSocket):
    """
    Stream real-time indexing progress.
    Emit events: file_uploaded, file_parsed, chunk_created, embedding_generated, etc.
    """
    await websocket.accept()
    # Stream progress updates
    pass

@router.delete("/clear")
async def clear_repository():
    """Clear current repository and vector database"""
    pass
```

**2. File Handling Logic:**

```python
# app/services/upload_handler.py (NEW FILE)

import tempfile
import shutil
from pathlib import Path

async def handle_directory_upload(files: List[UploadFile]):
    """
    1. Create temporary directory
    2. Save uploaded files preserving structure
    3. Return path to temp directory
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="repo_upload_"))

    for file in files:
        # Extract relative path from filename or metadata
        relative_path = file.filename  # Frontend should send full relative path
        target_path = temp_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Save file
        with open(target_path, "wb") as f:
            content = await file.read()
            f.write(content)

    return str(temp_dir)

async def process_uploaded_repository(repo_path: str, progress_callback=None):
    """
    1. Scan directory structure
    2. Parse files (call existing chunker)
    3. Generate embeddings (call existing embedder)
    4. Index in vector DB (call existing indexer)
    5. Report progress via callback/websocket
    """
    # Integrate with existing main.py indexing logic
    # Add progress reporting at each stage
    pass
```

**3. Progress Tracking:**

```python
# app/services/progress_tracker.py (NEW FILE)

from typing import Dict, Callable
import asyncio

class UploadProgress:
    def __init__(self, upload_id: str):
        self.upload_id = upload_id
        self.stage = "uploading"  # uploading, scanning, parsing, embedding, indexing, complete
        self.progress = 0  # 0-100
        self.total_files = 0
        self.files_processed = 0
        self.current_file = ""
        self.errors = []
        self.start_time = time.time()

    def update(self, **kwargs):
        """Update progress and notify listeners"""
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.notify_listeners()

    def to_dict(self):
        return {
            "stage": self.stage,
            "progress": self.progress,
            "current_file": self.current_file,
            "files_processed": f"{self.files_processed}/{self.total_files}",
            "eta_seconds": self.calculate_eta(),
            "errors": self.errors
        }

# Global progress tracker
progress_store: Dict[str, UploadProgress] = {}
```

**4. WebSocket Integration:**

```python
# app/api/websocket.py (MODIFY EXISTING)

@app.websocket("/ws/indexing/{upload_id}")
async def websocket_endpoint(websocket: WebSocket, upload_id: str):
    await websocket.accept()

    while True:
        progress = progress_store.get(upload_id)
        if progress:
            await websocket.send_json(progress.to_dict())

            if progress.stage == "complete":
                await websocket.close()
                break

        await asyncio.sleep(0.5)  # Send updates every 500ms
```

**5. Integration with Existing Indexer:**

```python
# Modify app/indexer.py to accept progress callback

def index_repository(repo_path: str, progress_callback=None):
    """
    Existing indexing logic, but add progress reporting:

    - After scanning: callback("scanning", files_found)
    - After each file parsed: callback("parsing", current_file, N/M)
    - After each embedding: callback("embedding", progress_percent)
    - After indexing: callback("complete")
    """
    pass
```

**6. Frontend JavaScript Upload Logic:**

```javascript
// utils/upload.ts (NEW FILE)

export async function uploadDirectory(directory: FileList | File[]) {
  const formData = new FormData();

  // Preserve directory structure in file paths
  for (const file of directory) {
    // Use webkitRelativePath for directory uploads
    const relativePath = file.webkitRelativePath || file.name;
    formData.append('files', file, relativePath);
  }

  const response = await fetch('/api/upload/directory', {
    method: 'POST',
    body: formData,
  });

  const { upload_id } = await response.json();

  // Connect to WebSocket for progress
  const ws = new WebSocket(`ws://localhost:8000/ws/indexing/${upload_id}`);

  return { upload_id, ws };
}

export function useUploadProgress(ws: WebSocket) {
  const [progress, setProgress] = useState(null);

  useEffect(() => {
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setProgress(data);

      // Trigger animations based on stage
      if (data.stage === "scanning") triggerScanAnimation();
      if (data.stage === "embedding") triggerEmbeddingAnimation();
      // etc.
    };
  }, [ws]);

  return progress;
}
```

**7. Storage Considerations:**

- Save uploaded files to temporary directory
- After indexing, optionally keep or delete original files
- Store vector embeddings in ChromaDB/existing vector store
- Add configuration for max upload size, allowed file types

---

## Deliverables

1. **Full React/Three.js Application**
   - All source code with clear folder structure
   - Package.json with all dependencies
   - Environment variables template (.env.example)

2. **README Documentation**
   - Setup instructions
   - Feature showcase with screenshots
   - Performance optimization tips
   - Customization guide

3. **Visual Assets**
   - Custom shaders (well-commented GLSL)
   - 3D models (if any, optimized GLTF/GLB)
   - Texture maps for effects
   - Font files (Google Fonts or custom)

4. **Integration Guide**
   - How to connect to existing backend
   - API client setup
   - WebSocket integration (if needed)
   - Deployment instructions

---

## Inspiration References

**Visual Style:**

- Ghost in the Shell (1995) - Opening sequence UI
- Cyberpunk 2077 - Braindance interface
- Neon Genesis Evangelion - NERV terminal displays
- Psycho-Pass - Sibyl System holographics
- Blade Runner 2049 - Holographic UI elements
- Tron Legacy - Grid and light cycles
- The Matrix - Code rain and construct loading

**Three.js Examples to Study:**

- https://threejs.org/examples/#webgl_points_waves
- https://threejs.org/examples/#webgl_postprocessing_unreal_bloom
- Bruno Simon's portfolio (creative interaction)
- Awwwards winners with Three.js

**Shader Resources:**

- ShaderToy: Digital rain effects
- The Book of Shaders: Noise and patterns
- Three.js shader examples

---

## Final Instructions

You are Claude Opus 4.6 with extended thinking capabilities. Use your reasoning to:

1. **Think deeply** about how each animation contributes to the "neural code intelligence" narrative
2. **Balance** visual complexity with usability - it should be stunning but functional
3. **Optimize** ruthlessly - beauty means nothing at 15fps
4. **Innovate** beyond typical WebGL demos - add unexpected touches
5. **Polish** every detail - cursor styles, loading states, transitions
6. **Test** your imagination - what would make YOU say "wow, this is unique"?

**Remember:** This interface represents an AI that understands code at a fundamental level. Every visual element should reinforce that this is not just a chatbot - it's a sentient being exploring the graph of software architecture.

Make it **unforgettable**. Make it **alive**. Make it look like no AI tool the user has ever seen.

---

## Success Criteria

When complete, the user should:

- ✨ Feel genuine excitement when the page loads
- 🎯 Intuitively understand how to interact despite unique design
- 🚀 Be proud to showcase this to others
- 💎 See no resemblance to generic AI chat interfaces
- 🔮 Feel like they're using technology from the future
- 🎨 Want to stare at it even when not using it

**Now, bring this vision to life with code that matches the ambition.**
