from fastapi import FastAPI
from pydantic import BaseModel

from typing import List
from backend.search import search as vector_search

app = FastAPI(title="AI Shipping Agent")

class ChatRequest(BaseModel):
	message: str

class SearchRequest(BaseModel):
	query: str 
	k: int = 5

class SearchHit(BaseModel):
	text: str 
	source: str 
	score: float 

class SearchResponse(BaseModel):
	results: List[SearchHit]

@app.get("/health")
def health():
	return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
	return {"answer": f"Echo: {req.message}", "citations": []}


@app.post("/search", response_model=SearchResponse)
def search_api(req: SearchRequest):
    hits = vector_search(req.query, req.k)
    return {"results": hits}
