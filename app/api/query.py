import time
import logging
from fastapi import APIRouter, Depends, HTTPException
from app.schemas.query import QueryRequest
from app.schemas.response import QueryResponse
from app.schemas.common import LatencyMetrics, Source
from app.core.dependencies import get_rag_pipeline, get_metrics_service
from app.rag.pipeline import RAGPipeline
from app.services.metrics_service import MetricsService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post('/query', response_model=QueryResponse)
async def text_query(request: QueryRequest, pipeline: RAGPipeline = Depends(get_rag_pipeline), metrics: MetricsService = Depends(get_metrics_service)):
    """
    Text query endpoint. Takes a text question, runs full RAG pipeline.
    Same pipeline used by voice-query (no duplication).
    """
    try:
        result = pipeline.answer_query(request.query)
        
        latency_bd = result.get('latency_breakdown', {})
        latency = LatencyMetrics(
            embedding_ms=latency_bd.get('embedding_ms', 0.0),
            bm25_ms=latency_bd.get('bm25_ms', 0.0),
            vector_ms=latency_bd.get('vector_ms', 0.0),
            rrf_ms=latency_bd.get('rrf_ms', 0.0),
            llm_ms=latency_bd.get('llm_ms', 0.0),
            total_ms=latency_bd.get('total_ms', 0.0)
        )
        
        metrics.record_text_query(latency.total_ms, latency_bd)
        
        return QueryResponse(
            query=request.query,
            answer=result['answer'],
            sources=result.get('sources', []),
            chunks=result.get('chunks', []),
            latency=latency
        )
    except Exception as e:
        logger.error(f'Query failed: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail='Query processing failed. Please try again.')

@router.get('/metrics')
async def get_metrics(metrics: MetricsService = Depends(get_metrics_service)):
    """Return latency analytics: P50, P70, P90, avg, min, max."""
    return metrics.get_stats()
