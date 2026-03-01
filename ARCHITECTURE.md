# Architecture

## Overview

The system processes a user's Claude.ai conversation history through an ML pipeline (fetch, scan, embed, cluster, label, organize) and presents the results as an interactive 3D knowledge graph with generated notes and a skill tree.

```
User's browser (Next.js on Vercel)
  │
  ├── Supabase (Google OAuth + skill_progress table)
  │
  └── FastAPI backend (Azure VM, Docker)
        ├── Claude.ai API (fetch conversations)
        ├── NVIDIA NIM API (embeddings + Mistral LLM)
        ├── Exa API (web search for notes)
        ├── GLiNER inference server (laptop GPU, Tailscale)
        └── Discord webhook (alerts)
```

## Repos & Where They Live

| Repo | Location (local) | Deployed to | Purpose |
|------|-------------------|-------------|---------|
| `Mistral` | `~/coding/Mistral` | Azure VM (`4.251.193.23:8000`) via Docker | FastAPI backend — pipeline, APIs |
| `hackathon-web-app` | `~/coding/hackathon-web-app` | Vercel (`claude-ux.vercel.app`) | Next.js frontend — dashboard, graph, skill tree |
| `gliner-inference-server` | `~/coding/gliner-inference-server` | Laptop GPU (`100.67.243.94:8081`) via Tailscale | GLiNER-PII model serving |

## Backend (Mistral)

### Stack
- **Runtime:** Python 3.12, FastAPI, uvicorn
- **Container:** `python:3.12-slim` Docker image on Azure VM
- **CI/CD:** GitHub Actions → SSH → `docker build` + `docker run` on push to `main`

### Directory Structure

```
Mistral/
├── main.py                     # FastAPI app, mounts all routers
├── config.py                   # Loads env vars (Supabase, API keys)
├── auth.py                     # Supabase SSR auth (extracts user_id from cookies)
├── store.py                    # In-memory per-user data store
├── routers/
│   ├── pipeline.py             # POST /start, GET /stream/{run_id}, POST /continue
│   ├── graph.py                # GET /graph-data (3D force graph nodes/links)
│   ├── topics.py               # GET /tree, GET /topic/{label}
│   ├── notes.py                # GET /notes/{label}, GET /notes
│   ├── skills.py               # GET /skills, POST /skills/{label}/unlock
│   └── cookies.py              # POST /count-conversations, POST /get-cookies
├── services/
│   ├── pipeline/
│   │   ├── pipeline.py         # 7-stage ML pipeline orchestration
│   │   ├── events.py           # Per-run state & SSE event queue
│   │   └── embed_cache.py      # File cache: .embed_cache/{uuid}/{hash}.npy
│   ├── privacy/
│   │   └── scanner.py          # Remote GLiNER-PII scanning via HTTP
│   ├── claude_fetcher/
│   │   ├── master.py           # ClaudeFetcher (curl_cffi, Chrome impersonation)
│   │   └── fetch_all.py        # Sync CLI utilities
│   ├── notes_agent.py          # Mistral + Exa agentic loop for note generation
│   ├── discord.py              # Fire-and-forget Discord webhook alerts
│   └── supabase_client.py      # skill_progress CRUD
├── models/
│   └── schemas.py              # Pydantic types (ScanResult, GraphData, etc.)
├── Dockerfile                  # Multi-layer: heavy deps (cached) → light deps → app
├── requirements.txt            # Light deps (fastapi, httpx, openai, pandas, etc.)
├── requirements-heavy.txt      # ML deps (numpy, scikit-learn, umap-learn, hdbscan)
└── .github/workflows/deploy.yml
```

### Pipeline Stages

The pipeline runs as a background task, streaming progress via SSE:

| # | Phase | What it does | External service |
|---|-------|-------------|------------------|
| 1 | **Fetch** | Paginated fetch of Claude.ai conversations (curl_cffi, Chrome impersonation) | Claude.ai API |
| 2 | **Scan** | PII detection on sampled messages (first 3 + last 3 per convo) | GLiNER server (laptop) |
| 3 | **Review** | Pause for user to exclude flagged PII categories | — |
| 4 | **Embed** | Embed all messages with baai/bge-m3, file-cached per conversation | NVIDIA NIM |
| 5 | **Segment** | DeepTiling — cosine similarity depth scoring to split conversations | — (local) |
| 6 | **Cluster** | BERTopic: UMAP → HDBSCAN → c-TF-IDF | — (local) |
| 7 | **Label** | LLM labels each cluster from keywords + example segments | Mistral via NIM |
| 8 | **Hierarchy** | LLM organizes labels into root → subcategory → topic tree | Mistral via NIM |

Labeling and hierarchy run concurrently via a producer-consumer pattern.

After pipeline completes, note generation fires in the background (Mistral + Exa web search).

### Deployment

Push to `main` triggers GitHub Actions:
1. SSH into Azure VM (`4.251.193.23`)
2. `git fetch` + `git reset --hard` in `/home/azureuser/final_production/Mistral`
3. `docker build -t mistral-backend .`
4. `docker run -d --name mistral-backend --env-file .env -p 8000:8000 mistral-backend`
5. `docker image prune -f`

## Frontend (hackathon-web-app)

### Stack
- **Framework:** Next.js 16, React 19, TypeScript
- **Styling:** Tailwind CSS 4, shadcn/ui, Radix UI
- **3D:** Three.js + 3d-force-graph
- **Auth:** Supabase Google OAuth
- **Deploy:** Vercel (`claude-ux.vercel.app`)

### Key Pages
- `/` — Landing page
- `/login` — Google OAuth via Supabase
- `/dashboard` — Main workspace: pipeline controls + 3D constellation graph + topic detail panel
- `/skill-tree` — 2D canvas skill constellation with unlock mechanics

### Backend Communication
- **REST:** All endpoints via `fetch()` with `credentials: "include"` for cookie auth
- **SSE:** `GET /api/pipeline/stream/{runId}` for real-time pipeline progress
- **Proxy:** `/api/proxy/*` rewrites to `https://aclo.ai/*` (next.config.ts)

### Key Hooks
- `usePipelineSSE` — EventSource connection, parses progress events
- `useGraphData` — Builds progressive graph state from SSE node/snapshot events
- `useNotes` — Polls note generation status, handles ZIP download

## GLiNER Inference Server (gliner-inference-server)

### Stack
- **Runtime:** Python, FastAPI, uvicorn
- **Model:** `nvidia/gliner-PII` on GPU (RTX 4060)
- **Network:** Tailscale VPN (`100.67.243.94:8081`)

### Endpoints
- `GET /health` → `{"status": "ok", "model": "nvidia/gliner-PII", "device": "cuda"}`
- `POST /predict` → accepts `{"texts": {"uuid": "text", ...}}`, returns `{"results": {"uuid": ["PERSON", "EMAIL"], ...}}`

### Graceful Degradation
If the server is off or unreachable, the backend skips PII scanning and the pipeline continues normally.

## External Services

| Service | Used for | Auth |
|---------|----------|------|
| **NVIDIA NIM** (`integrate.api.nvidia.com`) | Embeddings (baai/bge-m3) + LLM (mistral-large) | `NVIDIA_API_KEY` |
| **Mistral Cloud** | Note generation agent | `MISTRAL_API_KEY` |
| **Exa** | Web search during note generation | `EXA_API_KEY` |
| **Claude.ai** | Fetching user's conversation history | User's `sessionKey` cookie |
| **Supabase** (`koiaoajdcnxsarfpsfau.supabase.co`) | Auth (Google OAuth) + `skill_progress` table | Service role key + publishable key |
| **Discord webhook** | Fire-and-forget observability alerts | Hardcoded URL in `discord.py` |

## Environment Variables (Backend .env)

```
NVIDIA_API_KEY=...
MISTRAL_API_KEY=...
EXA_API_KEY=...
FRONTEND_URL=https://claude-ux.vercel.app
ALLOWED_ORIGINS=https://claude-ux.vercel.app,http://localhost:3000
NEXT_PUBLIC_SUPABASE_URL=https://koiaoajdcnxsarfpsfau.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
GLINER_SERVER_URL=http://100.67.243.94:8081
```

## Network Diagram

```
┌──────────────────────────────────────────────────────┐
│  User's Browser                                      │
│  claude-ux.vercel.app (Next.js on Vercel)            │
└──────────┬───────────────────────────────────────────┘
           │ HTTPS (REST + SSE)
           ▼
┌──────────────────────────────────────────────────────┐
│  Azure VM  4.251.193.23:8000                         │
│  Docker: mistral-backend (FastAPI)                   │
│                                                      │
│  ┌─ routers ──────────────────────────────────────┐  │
│  │  pipeline, graph, topics, notes, skills        │  │
│  └────────────────────────────────────────────────┘  │
│  ┌─ services ─────────────────────────────────────┐  │
│  │  pipeline (embed, segment, cluster, label)     │  │
│  │  claude_fetcher (conversation fetch)           │  │
│  │  privacy/scanner (remote GLiNER client)        │  │
│  │  notes_agent (Mistral + Exa)                   │  │
│  └────────────────────────────────────────────────┘  │
└──┬──────────┬──────────┬──────────┬──────────────────┘
   │          │          │          │
   │ Tailscale│ HTTPS    │ HTTPS    │ HTTPS
   ▼          ▼          ▼          ▼
 Laptop    Claude.ai  NVIDIA NIM  Supabase / Exa / Discord
 GLiNER     (fetch)   (embed+LLM)
 :8081
```
