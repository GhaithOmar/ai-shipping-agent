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
