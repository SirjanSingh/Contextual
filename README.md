# Repo-Aware AI Assistant (Google API)

A "talk to your codebase" tool:

- Indexes a repository locally
- Retrieves relevant code chunks (RAG)
- Uses Google Gemini for intelligent answers
- Returns answers with file references

## Tech Stack

- **LLM**: Google Gemini API (`gemini-2.0-flash-exp`)
- **Embeddings**: Google AI (`text-embedding-004`, 768 dims)
- **Vector Store**: FAISS (CPU, local)
- **Language**: Python 3.10+

## Setup

### 1. Get Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy the key

### 2. Configure Environment

```powershell
# Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Note: This installs google-genai (new package), not google-generativeai (deprecated)
```

### 3. Set API Key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here
```

Or set it as an environment variable:

```powershell
$env:GOOGLE_API_KEY = "your_api_key_here"
```

## Run

Index a repo and start a Q&A loop:

```powershell
python main.py --repo "C:\path\to\repo"
```

Force rebuild the index:

```powershell
python main.py --repo "C:\path\to\repo" --rebuild
```

## How It Works (Pipeline)

1. **Loader**: Reads code files from disk, ignores junk directories
2. **Chunker**: Splits files into overlapping chunks (1800 chars, 250 overlap)
3. **Embedder**: Converts chunks to 768-dim vectors via Google AI API
4. **Indexer**: Stores vectors + metadata in FAISS; caches on disk
5. **Retriever**: Finds top-k relevant chunks per question
6. **LLM**: Sends context + question to Gemini for answer generation
7. **Answer**: Prints answer + sources like `path:start-end`

## Options

| Flag            | Default      | Description                     |
| --------------- | ------------ | ------------------------------- |
| `--repo`        | required     | Path to target repository       |
| `--cache`       | `data/index` | Cache directory for FAISS index |
| `--rebuild`     | false        | Force rebuild the index         |
| `--topk`        | 6            | Number of chunks to retrieve    |
| `--temperature` | 0.2          | LLM temperature                 |
| `--chunk_size`  | 1800         | Chunk size in characters        |
| `--overlap`     | 250          | Chunk overlap in characters     |

## Environment Variables

| Variable          | Required | Default              | Description            |
| ----------------- | -------- | -------------------- | ---------------------- |
| `GOOGLE_API_KEY`  | Yes      | -                    | Your Google API key    |
| `GEMINI_MODEL`    | No       | `gemini-1.5-flash`   | Gemini model to use    |
| `EMBEDDING_MODEL` | No       | `text-embedding-004` | Embedding model to use |

## Notes

- The cache is stored under `data/index/<repo_id>/`
- First run requires internet to call Google APIs
- Subsequent runs use cached embeddings (fast)
- API costs are minimal (~$0.001 per 1000 chunks)

## Project Structure

```
repo-aware-ai/
├── app/
│   ├── config.py      # API configuration
│   ├── loader.py      # Load repository files
│   ├── chunker.py     # Split into chunks
│   ├── embedder.py    # Google embeddings
│   ├── indexer.py     # FAISS index management
│   ├── retriever.py   # Chunk retrieval
│   ├── llm.py         # Gemini LLM client
│   ├── qa.py          # QA engine orchestration
│   └── debug.py       # Debug logging
├── data/index/        # Cached FAISS indices
├── main.py            # CLI entry point
├── requirements.txt   # Python dependencies
├── .env.example       # Environment template
└── README.md          # This file
```

## Next Steps

- [ ] FastAPI backend + UI
- [ ] VS Code extension
- [ ] Multi-repository support
- [ ] Streaming responses
