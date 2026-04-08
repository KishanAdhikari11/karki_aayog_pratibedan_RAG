from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    embedding_source: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    embedding_source: Optional[Dict[str, Any]] = None


class EmbeddingModelError(Exception):
    error: str


class ErrorResponseSchema(BaseModel):
    description: str
