from fastapi import APIRouter, Request
from app.schemas.response import HealthResponse

router = APIRouter()

@router.get('/health', response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint. Returns status of all services."""
    state = request.app.state
    
    services = {}
    
    # Vector store
    try:
        count = state.vector_store.count()
        services['vector_store'] = f'ok ({count} chunks)'
    except Exception:
        services['vector_store'] = 'unavailable'
    
    # BM25
    try:
        ready = state.bm25_index.is_ready()
        services['bm25'] = 'ok' if ready else 'not_indexed'
    except Exception:
        services['bm25'] = 'unavailable'
    
    # Embedding model
    try:
        _ = state.embedding_service
        services['embedding_model'] = 'ok'
    except Exception:
        services['embedding_model'] = 'unavailable'
    
    # Speech service
    try:
        available = state.speech_service.is_available()
        services['speech_service'] = 'ok (api_key_set)' if available else 'ok (no_api_key_mock_mode)'
    except Exception:
        services['speech_service'] = 'unavailable'
    
    all_ok = 'unavailable' not in services.values()
    
    return HealthResponse(
        status='ok' if all_ok else 'degraded',
        services=services
    )
