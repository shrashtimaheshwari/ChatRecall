from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Group Chat Semantic Search")

class SearchQuery(BaseModel):
    query: str
    mode: Optional[str] = "semantic"  # 'semantic', 'attributed', or 'temporal'
    sender: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    context_n: Optional[int] = 3

class Message(BaseModel):
    id: str
    sender: str
    timestamp: str
    text: str

class ContextMessage(Message):
    is_match: bool

class SearchResult(BaseModel):
    match: Message
    context: List[ContextMessage]

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

@app.get("/health")
def health_check():
    return {"status": "ok"}

import search as search_module

@app.post("/search", response_model=List[SearchResult])
def search(query: SearchQuery):
    engine = search_module.get_engine()
    
    raw_results = engine.search(
        query=query.query,
        mode=query.mode,
        sender=query.sender,
        start_date=query.start_date,
        end_date=query.end_date,
        top_k=15,
        context_n=query.context_n
    )
    
    formatted_results = []
    for r in raw_results:
        msg = Message(
            id=r["match"]["id"],
            sender=r["match"]["sender"],
            timestamp=r["match"]["timestamp"],
            text=r["match"]["text"]
        )
        
        ctx_list = []
        for c in r["context"]:
            ctx_list.append(ContextMessage(
                id=c["id"],
                sender=c["sender"],
                timestamp=c["timestamp"],
                text=c["text"],
                is_match=c["is_match"]
            ))
            
        formatted_results.append(SearchResult(
            match=msg,
            context=ctx_list
        ))
        
    return formatted_results
