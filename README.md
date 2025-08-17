AI Shipping Agent 🚀

A production-leaning AI Agent project built step by step.
This repo will grow into a full LLM-powered support assistant using LangChain/LangGraph, FastAPI, Docker, and a lightly fine-tuned LLM (LoRA).

✅ Day 1 — FastAPI Skeleton

Today we set up the project foundation:

Created a FastAPI app with two endpoints:

GET /health → simple health check.

POST /chat → stub endpoint that echoes back the user’s message.

Installed core dependencies (fastapi, uvicorn, pydantic, python-dotenv).

Created a virtual environment for isolated development.

Saved dependencies into backend/requirements.txt.

🔧 Quickstart (local)
1. Clone repo & enter folder
git clone <your-repo-url>
cd ai-shipping-agent

2. Create & activate virtual environment

PowerShell

python -m venv .venv
.venv\Scripts\Activate.ps1

3. Install dependencies
pip install -r backend/requirements.txt

4. Run the FastAPI server
uvicorn backend.main:app --reload

5. Test it

Open http://127.0.0.1:8000/docs

Try GET /health → returns:

{"status": "ok"}


Try POST /chat with:

{ "message": "hello world" }


→ returns:

{ "answer": "Echo: hello world", "citations": [] }