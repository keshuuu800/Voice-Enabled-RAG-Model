"""
FastAPI dependency injection providers.
All services are accessed from app.state (set during lifespan).
"""
from fastapi import Request
from app.rag.pipeline import RAGPipeline
from app.services.speech_service import SpeechService
from app.services.metrics_service import MetricsService

def get_rag_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.rag_pipeline

def get_speech_service(request: Request) -> SpeechService:
    return request.app.state.speech_service

def get_metrics_service(request: Request) -> MetricsService:
    return request.app.state.metrics_service
