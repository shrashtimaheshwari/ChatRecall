from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Group Chat Semantic Search")

class SearchQuery(BaseModel):
    query: str
    mode: Optional[str] = "semantic"  # 'semantic', 'attributed', or 'temporal'

class Message(BaseModel):
    id: str
    sender: str
    timestamp: str
    text: str

class SearchResult(BaseModel):
    match: Message
    context_before: List[Message]
    context_after: List[Message]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/search", response_model=List[SearchResult])
def search(query: SearchQuery):
    # Hardcoded dummy response for Phase 1 stub
    dummy_message = Message(
        id="msg_001",
        sender="Shrashti",
        timestamp="2026-09-08T10:00:00Z",
        text="chalo Manali fix hai"
    )
    
    dummy_context_before = [
        Message(
            id="msg_000",
            sender="Priya",
            timestamp="2026-09-08T09:55:00Z",
            text="what did we decide on the trip?"
        )
    ]
    
    dummy_context_after = [
        Message(
            id="msg_002",
            sender="Shrey",
            timestamp="2026-09-08T10:05:00Z",
            text="Done! I'll book the tickets."
        )
    ]
    
    result = SearchResult(
        match=dummy_message,
        context_before=dummy_context_before,
        context_after=dummy_context_after
    )
    
    return [result]
