from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, Any]]] = []

class Citation(BaseModel):
    page: int
    snippet: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    metadata: Dict[str, Any]