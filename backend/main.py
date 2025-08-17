from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Shipping Agent")

class ChatRequest(BaseModel):
	message: str


@app.get("/health")
def health():
	return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
	return {"answer": f"Echo: {req.message}", "citations": []}
	