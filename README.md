# AI Shipping Agent 🚀

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

**Next (Day 4):**
- Scale data to 10–30k (Bitext 70–85%, Kaggle 5–15%, Synthetic 10–20%).
- 1–2 short epochs; keep RAG as ground-truth at inference.
- Extended eval (20–30 prompts) and integrate adapter into backend.
