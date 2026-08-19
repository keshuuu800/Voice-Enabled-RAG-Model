"""Response schemas for all API endpoints."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel
from app.schemas.common import Source, LatencyMetrics


class ChunkDetail(BaseModel):
    chunk_id: str
    source: str
    page: Optional[int] = None
    text: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]
    chunks: Optional[list[ChunkDetail]] = None
    latency: LatencyMetrics


class VoiceQueryResponse(BaseModel):
    transcript: str
    answer: str
    sources: list[Source]
    chunks: Optional[list[ChunkDetail]] = None
    latency: LatencyMetrics


class HealthResponse(BaseModel):
    status: str
    services: dict[str, str]
    version: str = "1.0.0"


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
