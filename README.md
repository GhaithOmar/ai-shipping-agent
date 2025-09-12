# AI Shipping Agent 🚀
![CI](https://github.com/GhaithOmar/ai-shipping-agent/actions/workflows/ci.yml/badge.svg)


A production-leaning AI Agent project built step by step.  
This repo will grow into a full **LLM-powered support assistant** using **LangChain/LangGraph**, **FastAPI**, **Docker**, and a lightly fine-tuned LLM (LoRA).

---

## ✅ Day 1 — FastAPI Skeleton

Today we set up the project foundation:

- Created a **FastAPI app** with two endpoints:
  - `GET /health` → simple health check.
  - `POST /chat` → stub endpoint that echoes back the user’s message.
- Installed core dependencies (`fastapi`, `uvicorn`, `pydantic`, `python-dotenv`).
- Created a **virtual environment** for isolated development.
- Saved dependencies into `backend/requirements.txt`.

---

## 🔧 Quickstart (local)

### 1. Clone repo & enter folder
```bash
git clone <your-repo-url>
cd ai-shipping-agent
```

### 2. Create & activate virtual environment (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Run the FastAPI server
```bash
uvicorn backend.main:app --reload
```

### 5. Test it
Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

- Try **GET /health** → returns:
  ```json
  {"status": "ok"}
  ```

- Try **POST /chat** with:
  ```json
  { "message": "hello world" }
  ```
  → returns:
  ```json
  { "answer": "Echo: hello world", "citations": [] }
  ```

---

## 📂 Project Structure (Day 1)

```text
ai-shipping-agent/
├── backend/
│   ├── main.py
│   └── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Next Steps (Day 2 Preview)

- Collect 20–40 FAQ pages (DHL, Aramex, UPS, etc.) into `rag/data/`.
- Build an ingestion script to **chunk → embed → store in Qdrant**.
- Add `/search` endpoint to test retrieval.

---

## 📚 Day 2 — Ingestion & Search

Today we added the first Retrieval-Augmented Generation (RAG) layer:

- Wrote ingestion pipeline (`rag/ingest.py`):
  - Loads `.md` files from `rag/data/` (skips `README.md`).
  - Splits into overlapping character chunks.
  - Embeds with **BAAI/bge-m3** and stores in local **Qdrant**.
- Added `/search` endpoint → validates retrieval end-to-end.
- Indexed **synthetic shipping FAQs & policies** (14 chunks).

### Notes
- The dataset is **synthetic**, generated with the help of ChatGPT.  
- Uses placeholder carriers (*Shipping Company A/B/C*) to avoid trademark/legal issues.  
- Purpose: demo MVP pipeline with realistic structure, but no real data.

## Day 3 — Tiny LoRA (Production Path)

**Data (v0.2):**
- Sources: Bitext Customer Support (HF), Customer Support on Twitter (Kaggle).
- Processing: PII scrub, carrier aliasing (Shipping_A/B/C), de-twitterization (strip @handles, URLs, ^tags, “DM us”), near-dup dedup, shipping-intent filter.
- Rebalanced ratios: Bitext 85%, Kaggle 5%, Synthetic 10%.
- Format: JSONL with {"input", "assistant_response"} (≤1024 tokens).

**Training:**
- Base: Llama 3.1 8B Instruct (4-bit).
- LoRA: r=8, alpha=16, dropout=0.1, targets: attention only (q/k/v/o).
- Trainer: TRL SFT (legacy API) with HF DataCollatorForLanguageModeling.
- Output: adapter-only saved under /MyDrive/ai_shipping_agent/lora-llama3.1-8b.

**Eval:**
- System prefix enforces: ask-before-answer, no links, concise steps, defer facts to RAG.
- Decoding guardrails: blocked URL/handles/tags, conservative max_new_tokens.
- Result: clean tone without Twitter artifacts; still requests IDs and avoids live-claiming.

## 🧪 Day 4 — Fine-tune & Inference Integration

**Data v0.2 (20k examples):**
- Built new supervised set with ~20,000 pairs (17,600 train + 2,400 val).
- Auto-split logged by the builder and stored to `data/v0.2/` (`mini_sft.jsonl`, `mini_sft_val.jsonl`).

**Training:**
- LoRA/PEFT SFT using TRL on Colab L4 (24GB), 1 epoch to reduce risk of overfitting.
- Best run saved; eval loss ≈ **0.77**. Adapters pushed to the Hugging Face Hub (user: **GhaithOmar**).

**Inference / Guardrails:**
- Implemented `infer_guarded` with deterministic decoding, link suppression, and intent-aware “missing ID” nudge.
- Added a small rule set to scrub social artifacts and enforce concise bullets.

**Backend wiring:**
- Created `backend/generation.py` and updated `backend/main.py` to load base + adapter on startup and expose `/chat` with RAG context.
- Added `tests/smoke/smoke.py` to sanity-check loading, search, and a sample round trip.

**Artifacts to check in:**
- `configs/data_prep_v02.yaml` (metadata bump to v0.2).
- `data/v0.2/mini_sft.jsonl` and `mini_sft_val.jsonl`.
- `backend/generation.py`, updated `backend/main.py`.
- `tests/smoke/smoke.py`.

---

## 📂 Project Structure (Day 4)
```text
ai-shipping-agent/
├── backend/
│   ├── generation.py
│   ├── main.py
│   ├── requirements.txt
│   └── search.py
├── configs/
│   ├── data_prep_v01.yaml
│   └── data_prep_v02.yaml
├── data/
│   ├── v0.1/
│   │   ├── manifest.json
│   │   ├── mini_sft.jsonl
│   │   └── mini_sft_val.jsonl
│   └── v0.2/
│       ├── manifest.json
│       ├── mini_sft.jsonl
│       └── mini_sft_val.jsonl
├── notebooks/
│   ├── Day3/Tiny_LoRA_Smoke-Train.ipynb
│   └── Day4/Full_LoRA_Train.ipynb
├── qdrant_db/
│   └── collection/shipping_kb/storage.sqlite
├── rag/
│   ├── ingest.py
│   └── data/
│       ├── common_policies.md
│       ├── company_a_faq.md
│       ├── company_b_faq.md
│       ├── company_c_faq.md
│       ├── tracking_status_reference.md
│       └── README.md
├── scripts/
│   └── data_prep/build_sft.py
└── tests/
    └── smoke/smoke.py
```

## Day 5 — Agent Integration (LangChain + LangGraph)

**What we added**
- `backend/tools/`:
  - `search_kb.py`: Qdrant retriever (prefers embedded `qdrant_db/`, soft‑fails to `[]` if unavailable) + `format_citations`.
  - `parse_tracking.py`: regex extractor for carrier aliases + tracking IDs.
  - `estimate_eta.py`: rough ETA rules (business days) with GCC/EU region tweaks.
- `backend/agent/`:
  - `graph.py`: minimal LangGraph app with nodes:
    `history_read → understand → tool_router → respond → history_write`.
    Reuses the existing `infer_guarded` for final text.
  - `memory.py`: in‑process short‑term memory (last ~6 turns).
- `backend/main.py`:
  - Agent toggle: `POST /chat?agent=1` uses the LangGraph path; default `POST /chat` keeps legacy RAG path.
  - Streaming: `POST /chat/stream` streams the agent’s answer line‑by‑line.
- Tests:
  - `tests/agent/test_tools.py`, `tests/agent/test_api_agent.py` + smoke tests; pass even without a model/Qdrant server.

```powershell
uvicorn backend.main:app --reload
# Agent path (JSON)
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat?agent=1" `
  -ContentType "application/json" -Body '{"message":"When will my package arrive JO to AE express?"}'

# Legacy search
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/search" `
  -ContentType "application/json" -Body '{"query":"change address","k":3}'

# Streaming (arrives as a single string in PowerShell)
$body = '{"message":"Track order 12345 with Shipping_A and give last two scans."}'
(Invoke-WebRequest -Method Post -Uri "http://127.0.0.1:8000/chat/stream" `
  -ContentType "application/json" -Body $body).Content
```

📂 Project Structure (Day 5 — after Agent Integration)
```text
ai-shipping-agent/
├── backend/
│   ├── main.py              # FastAPI app; /chat (legacy + agent toggle), /chat/stream
│   ├── generation.py        # Guarded inference (LoRA adapter + safety rules)
│   ├── settings.py          # Pydantic settings loader (.env support)
│   ├── search.py            # Legacy RAG search helper (kept for compatibility)
│   ├── agent/
│   │   ├── graph.py         # LangGraph state machine (understand → tool_router → respond + memory)
│   │   └── memory.py        # Short-term in-process memory (last 6 turns)
│   └── tools/
│       ├── search_kb.py     # Qdrant semantic search retriever + citation formatter
│       ├── parse_tracking.py# Regex parser for carrier + tracking IDs
│       └── estimate_eta.py  # Simple ETA rules (business-day estimates)
├── rag/
│   └── ingest.py            # Ingestion pipeline → builds qdrant_db/ from markdown data
├── qdrant_db/               # Embedded Qdrant collection (local persistence)
├── tests/
│   ├── agent/               # Unit tests (tools + API agent branch)
│   └── smoke/               # Smoke tests (model, RAG, agent path)
├── configs/                 # Dataset prep configs (v0.1, v0.2 YAMLs)
├── data/                    # Training/eval datasets (v0.1, v0.2 JSONL)
├── notebooks/               # Colab notebooks (LoRA training runs)
├── scripts/                 # Data prep scripts
├── README.md
└── .env.example             # Example environment variables
**Run (PowerShell)**

```
## 🧭 Day 6 — Agent polish, new tool, streaming & citations

**What we shipped today**

- **Agent as the smooth default**: `/chat` now uses the LangGraph path (toggle with `AGENT_ENABLE` or `?agent=0` to force legacy).
- **New tool — `rate_quote`**: quick, explainable label price estimate with zone/service rules.
- **Stronger `parse_tracking`**: accepts hyphenated / spaced / mixed IDs, normalizes to canonical form.
- **Citations normalized**: both paths return `["source#chunk", ...]` (e.g., `company_a_faq.md#12`).  
- **Streaming**:
  - `/chat/stream` (agent): chunked SSE (client contract identical to token streams).
  - `/chat/stream?agent=0` (legacy): **token SSE** when a model is loaded, **offline chunked** fallback otherwise.
- **Resilience**: all endpoints degrade gracefully even if no model/GPU is available.
- **Re‑ingest**: Qdrant payload now includes `source` (filename) and `chunk_id` (per‑file index) for traceable citations.

---

## 📂 Project Structure (Day 6)

```text
ai-shipping-agent/
├── backend/
│   ├── main.py                 # FastAPI app: /health, /chat, /chat/stream (SSE)
│   ├── generation.py           # Guarded inference + token streaming helper
│   ├── settings.py             # Env-driven config (AGENT_ENABLE, QDRANT_*)
│   ├── search.py               # Legacy retriever (returns source + chunk_id)
│   ├── agent/
│   │   ├── graph.py            # LangGraph: understand → tool_router → respond (+ memory)
│   │   └── memory.py           # Short-term memory (last N turns)
│   └── tools/
│       ├── search_kb.py        # Qdrant retriever; emits KBHit with source + chunk_id
│       ├── parse_tracking.py   # Robust tracking parser (hyphens/spaces → normalized)
│       ├── estimate_eta.py     # Handbook ETA rules (business days, regions)
│       └── rate_quote.py       # NEW: handbook label price estimate
├── rag/
│   └── ingest.py               # Markdown → chunks → BGE-M3 → Qdrant (writes source + chunk_id)
├── qdrant_db/                  # Embedded Qdrant store (created by ingest)
├── tests/
│   ├── agent/                  # Unit tests (graph/tools)
│   └── smoke/                  # Smoke tests (API/e2e)
├── .env.example
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment

Create `.env` (or copy from `.env.example`) and adjust if needed:

```env
# Toggle default path (true = agent by default)
AGENT_ENABLE=true

# Optional model settings (leave empty if no GPU; app will degrade gracefully)
BASE_MODEL=
ADAPTER_ID=
FALLBACK_BASE=Qwen/Qwen2.5-3B-Instruct
HUGGINGFACE_TOKEN=

# Qdrant (embedded)
QDRANT_PATH=qdrant_db
QDRANT_COLLECTION=shipping_kb
```

> **Tip:** If you don’t have a capable GPU, keep model variables empty. The agent and legacy paths still return helpful, handbook‑based replies with citations.

---

## 📦 Ingest the Knowledge Base

```powershell
# PowerShell
Remove-Item -Recurse -Force .\qdrant_db\ -ErrorAction SilentlyContinue
python .\rag\ingest.py
```

```bash
# Bash
rm -rf qdrant_db
python rag/ingest.py
```

Quick sanity check (optional):
```powershell
python - << 'PY'
from backend.search import search
print([ (h["source"], h.get("chunk_id")) for h in search("return policy", k=3) ])
PY
```

---

## ▶️ Run the API

```powershell
# PowerShell
uvicorn backend.main:app --reload --port 8000
```

```bash
# Bash
uvicorn backend.main:app --reload --port 8000
```

Health:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```
```bash
curl -s http://127.0.0.1:8000/health
```

---

## 🧪 Quick Chat Tests

**Agent (default)**
```powershell
$body = @{ message = "Track order 12345678 with Shipping_A"; top_k = 2 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method Post -ContentType "application/json" -Body $body
```

```bash
curl -s -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"Track order 12345678 with Shipping_A","top_k":2}'
```

**Force legacy**
```powershell
$body = @{ message = "Return policy for fragile items"; top_k = 3 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat?agent=0" -Method Post -ContentType "application/json" -Body $body
```

```bash
curl -s -X POST "http://127.0.0.1:8000/chat?agent=0" \
  -H "Content-Type: application/json" \
  -d '{"message":"Return policy for fragile items","top_k":3}'
```

You should see a JSON response like:
```json
{
  "answer": "- ...",
  "citations": ["company_a_faq.md#6", "company_b_faq.md#9"]
}
```

---

## 📡 Streaming (SSE)

**Agent (chunked SSE output)**
```powershell
$body = @{ message = "Track order 12345678 with Shipping_A"; top_k = 2 } | ConvertTo-Json
$r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/chat/stream" -Method Post -ContentType "application/json" -Body $body
$r.Content
```

```bash
curl -N -X POST "http://127.0.0.1:8000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"message":"Track order 12345678 with Shipping_A","top_k":2}'
```

**Legacy (token SSE if model is loaded, otherwise offline-chunked SSE)**
```powershell
$body = @{ message = "Return policy for fragile items"; top_k = 3 } | ConvertTo-Json
$r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/chat/stream?agent=0" -Method Post -ContentType "application/json" -Body $body
$r.Content
```

```bash
curl -N -X POST "http://127.0.0.1:8000/chat/stream?agent=0" \
  -H "Content-Type: application/json" \
  -d '{"message":"Return policy for fragile items","top_k":3}'
```

---

## 🧰 Tests

```powershell
# Unit tests
python -m pytest -q tests/agent

# Full suite
python -m pytest -q

# (If API is running, the embedded DB may be locked on Windows; either stop Uvicorn
# or isolate a test store)
$env:QDRANT_PATH = "qdrant_db_test"
Remove-Item -Recurse -Force .\qdrant_db_test -ErrorAction SilentlyContinue
python -m pytest -q
Remove-Item -Recurse -Force .\qdrant_db_test -ErrorAction SilentlyContinue
```

```bash
# Bash
python -m pytest -q
```

---

## 🧯 Troubleshooting

- **Empty citations on agent path** → Re‑ingest with the updated payload (must contain `source` + `chunk_id`).  
- **“Model not loaded” in legacy streaming** → expected without GPU; endpoint streams offline chunks instead of failing.  
- **PowerShell quoting issues** → prefer `Invoke-RestMethod` or here‑strings for JSON; avoid multi‑line `curl.exe` unless you double‑escape.  
- **Weird characters (–, →)** → console encoding; server normalizes to ASCII in responses.

---

## 🔒 Limitations (Day 6)

- **Data realism**: KB docs are synthetic and for demo only; citations are precise (`source#chunk`) but not from real carrier content.  
- **Model hosting**: LoRA adapters are not deployed by default; legacy token streaming requires a local/remote model.  
- **Evaluation**: smoke tests verify routing/guardrails; no large‑scale RAGAS or human eval yet.  
- **Streaming in agent path**: chunked SSE today (client‑compatible); full token streaming through the graph can be added later if needed.

## 🐳 Day 7 — Dockerization & Deployment Prep

Today we containerized the AI Shipping Agent for reproducibility and easier sharing.

### What We Did
- **Dockerfile**: multi-stage build
  - Builder → pre-builds wheels from `requirements.txt`.
  - Runtime → lightweight `python:3.11-slim` with just runtime deps.
  - Entrypoint script (`docker/entrypoint.sh`) runs ingestion once, then launches FastAPI.
- **docker-compose.yml**: 
  - Mounts `qdrant_db/` volume for persistence.
  - Exposes FastAPI on port `8000`.
  - Loads `.env` for Hugging Face token, agent toggle, etc.
- **Environment (`.env`)**:
  - `AGENT_ENABLE` → agent vs legacy path.
  - `QDRANT_PATH` → vector DB location inside container.
  - `HOST`, `PORT` for FastAPI.
  - Optional `BASE_MODEL`, `ADAPTER_ID`, `HF_TOKEN`.
- **Warmup Script**:
  - Added `warm_agent.py` for one-off model+adapter load test.
- **Smoke Tests**:
  - `docker compose up` launches container; `/docs` accessible.
  - Verified retrieval and agent endpoints with PowerShell + Bash.
  - Confirmed fallback to smaller base (`TinyLlama-1.1B`) works on CPU-only machines.

### 📂 Project Structure (Day 7)
```text
ai-shipping-agent/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── generation.py        # Model loading + guarded inference
│   ├── settings.py          # Env-based settings
│   ├── search.py            # Legacy retriever
│   ├── agent/
│   │   ├── graph.py         # LangGraph agent
│   │   └── memory.py        # Short-term memory
│   └── tools/
│       ├── search_kb.py     # Qdrant retriever
│       ├── parse_tracking.py# Tracking ID parser
│       ├── estimate_eta.py  # ETA rules
│       └── rate_quote.py    # Rate quoting tool
├── rag/
│   └── ingest.py            # Markdown → embeddings → Qdrant
├── qdrant_db/               # Local vector store (mounted in container)
├── docker/
│   └── entrypoint.sh        # Boot logic
├── tests/
│   ├── conftest.py
│   ├── test_agent_graph.py
│   └── smoke/…
├── warm_agent.py            # Model warmup helper
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```
##🔒 Limitations (Day 7)

- **Compute limits**: On laptops without GPU, large models may OOM. Fallback base (TinyLlama-1.1B) keeps tests runnable.

- **Network reliance**: First container run downloads models from Hugging Face (can be slow/unreliable).

- **Adapters**: LoRA adapters not bundled into image; require token+download.

- **Windows quirks**: PowerShell needs Invoke-RestMethod/Invoke-WebRequest instead of curl; CRLF vs LF warnings expected.

## 🚀 Day 8 — CI, Releases & GHCR

[![CI](https://github.com/GhaithOmar/ai-shipping-agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/GhaithOmar/ai-shipping-agent/actions/workflows/ci.yml)

Today we made the project feel “production-ready”: automated lint/tests, deterministic Docker builds in CI, and one-command versioned releases.

### What We Did
- **CI Pipeline (GitHub Actions)**
  - Lint & format: `ruff` + `black` (fail fast on style/correctness).
  - **Offline** smoke tests (no HF downloads) to keep CI fast/reliable.
  - **Docker Build (no push)** job to ensure the image builds on clean runners.
  - **Publish on tag** to **GHCR** (`ghcr.io/ghaithomar/ai-shipping-agent:vX.Y.Z` + `:latest`) and auto **GitHub Release** notes.
- **Safer Docker build**
  - `.dockerignore` tuned to exclude caches, notebooks, and local DBs.
  - `.env.example` generated in-image (no secrets; predictable boot).
  - Health check hits `/health` so containers report **healthy**.
- **Docs**
  - Added run commands (Docker & source), release steps, and troubleshooting.

---

### How to Run (Docker)
```bash
# Pull the latest published image
docker pull ghcr.io/ghaithomar/ai-shipping-agent:latest

# Run the API at http://127.0.0.1:8000
docker run --rm -p 8000:8000 ghcr.io/ghaithomar/ai-shipping-agent:latest
```

**Windows (PowerShell) quick checks**
```powershell
# health
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get

# chat (non-streaming)
$body = @{ message = "Track order 12345678 with Shipping_A"; top_k = 2 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method Post -ContentType "application/json" -Body $body

# streaming (SSE text payload)
$r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/chat/stream" -Method Post -ContentType "application/json" -Body $body
$r.Content
```

**Bash**
```bash
# agent (default)
curl -s -X POST "http://127.0.0.1:8000/chat"   -H "Content-Type: application/json"   -d '{"message":"Track order 12345678 with Shipping_A","top_k":2}'

# legacy (force no agent)
curl -s -X POST "http://127.0.0.1:8000/chat?agent=0"   -H "Content-Type: application/json"   -d '{"message":"Return policy for fragile items","top_k":3}'

# streaming (agent)
curl -N -X POST "http://127.0.0.1:8000/chat/stream"   -H "Content-Type: application/json"   -d '{"message":"Track order 12345678 with Shipping_A","top_k":2}'
```

---

### Run from Source (dev)
```bash
# 1) Create your local env file
cp .env.example .env   # (Windows: copy .env.example .env)

# 2) (Optional) offline mode to avoid HF downloads
echo "AGENT_OFFLINE=1" >> .env

# 3) Create venv & install deps
python -m venv .venv && source .venv/bin/activate   # (Windows: .\.venv\Scripts\Activate.ps1)
pip install -U pip && pip install -r requirements.txt

# 4) Start API
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

### Cut a Release (publishes Docker image to GHCR)
This repo auto-publishes `ghcr.io/ghaithomar/ai-shipping-agent` when you push a tag that starts with `v`.

```bash
git pull --rebase origin main
git tag -a v0.8.3 -m "Day 8: CI + GHCR publish"
git push origin v0.8.3
```

- After the first publish, set **Packages → ai-shipping-agent → Settings → Visibility: Public**.
- Pull & run:
  ```bash
  docker pull ghcr.io/ghaithomar/ai-shipping-agent:v0.8.3
  docker run --rm -p 8000:8000 ghcr.io/ghaithomar/ai-shipping-agent:v0.8.3
  ```

---

### 📂 Project Structure (Day 8)
```text
ai-shipping-agent/
├── backend/
│   ├── main.py              # FastAPI app (/health, /chat, /chat/stream)
│   ├── generation.py        # Model loading + guarded inference
│   ├── settings.py          # Env-based settings
│   ├── search.py            # Legacy retriever
│   ├── agent/
│   │   ├── graph.py         # LangGraph agent
│   │   └── memory.py        # Short-term memory
│   └── tools/
│       ├── search_kb.py     # Qdrant retriever
│       ├── parse_tracking.py# Tracking ID parser
│       ├── estimate_eta.py  # ETA rules
│       └── rate_quote.py    # Rate quoting tool
├── rag/
│   └── ingest.py            # Markdown → embeddings → Qdrant
├── qdrant_db/               # Local vector store (generated; not in image)
├── docker/
│   └── entrypoint.sh        # Boot logic (ingest once → uvicorn)
├── tests/
│   └── smoke/test_api_smoke.py  # Offline, self-contained API smoke
├── .github/workflows/ci.yml # Lint/tests → docker build → publish on tag → release
├── .dockerignore            # Slimmer, safer build context
├── .env.example             # Safe defaults (generated in image too)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

### 🔒 Limitations (Day 8)
- **Offline smoke** keeps CI fast but doesn’t measure model quality; full eval (e.g., RAGAS) is out of scope today.
- **No live tracking**: answers are handbook/RAG-based; agent refuses to share direct links.
- **Image excludes local data** (Qdrant DB, caches, notebooks) by design; they’re mounted at runtime.



## 🔮 Limitations & Future Work

This project was built as a **portfolio showcase** to demonstrate the end-to-end design of an AI shipping support agent. While it integrates retrieval, fine-tuned generation, and agent orchestration, there are some known limitations and clear paths for future work.

### 📉 Current Limitations
- **Data realism**  
  - FAQ knowledge base content is **synthetic**, generated to resemble real carrier FAQs and policies.  
  - Fine-tuning data combines public **Customer Support Twitter (Kaggle)** and **Bitext datasets**, with preprocessing for PII scrub + carrier aliasing.  
  - These datasets approximate real support conversations but are **not production carrier data**.

- **Model deployment**  
  - The LoRA-fine-tuned **Llama-3.1-8B Instruct** is tested successfully offline (Colab / local GPU).  
  - Due to GPU cost constraints, the LoRA adapter is **not hosted live**. Online demos may instead route to a base model API (e.g., Together/Fireworks) while preserving the pipeline.

- **Evaluation**  
  - Current testing relies on smoke tests and qualitative inspection.  
  - No large-scale **RAGAS** or structured human evaluation has been performed yet.

### 🚀 Future Work
- **Data improvements** → Incorporate real carrier datasets (FAQs, scan events, policies) for higher realism.  
- **Scalable deployment** → Host fine-tuned model + retriever via GPU cloud infra (AWS/GCP/RunPod) with Docker/K8s.  
- **Evaluation framework** → Add automated metrics (context precision/recall, answer faithfulness) and human eval.  
- **Additional tools** → Extend agent with rate quote, pickup scheduling, and customs clearance calculators.  
- **Monitoring & guardrails** → Add observability, tracing, and stronger safety filters for production reliability.  

---

## 📌 Project Status

This repository represents a **research prototype and learning project**, not a production-ready system.  
It demonstrates the following end-to-end skills:  
- Data curation & preprocessing (synthetic + public datasets).  
- LoRA fine-tuning on Llama-3.1-8B with guardrails.  
- RAG integration with Qdrant + semantic search.  
- Agent orchestration with LangGraph and tool-calling.  
- FastAPI backend + Docker packaging.  

⚠️ **Disclaimer:** Since the project uses synthetic and public datasets, and the fine-tuned model is not hosted live, results should be considered illustrative only.  
