"""
Shared data schemas used across the entire application.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class Document(BaseModel):
    """Raw document produced by a loader before chunking."""
    text: str
    metadata: dict  # Required keys: source (str), document_id (str). Optional: page (int)


class Chunk(BaseModel):
    """A single text chunk with full provenance metadata."""
    text: str
    metadata: dict
    # Metadata keys:
    #   document_id: str
    #   source: str       — filename/path of originating document
    #   page: int | None  — page number if applicable
    #   chunk_id: str     — globally unique ID (uuid4)
    #   chunk_index: int  — 0-based position within the parent document


class RetrievalResult(BaseModel):
    """A ranked chunk returned by any retriever (BM25, semantic, or hybrid)."""
    chunk_id: str
    text: str
    score: float       # Higher is better
    source: str
    page: Optional[int] = None
    document_id: str


class Source(BaseModel):
    """A citation attached to a final answer."""
    source: str
    page: Optional[int] = None
    chunk_id: str


class LatencyMetrics(BaseModel):
    """Per-stage latency in milliseconds."""
    stt_ms: Optional[float] = None
    embedding_ms: float = 0.0
    bm25_ms: float = 0.0
    vector_ms: float = 0.0
    rrf_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0
