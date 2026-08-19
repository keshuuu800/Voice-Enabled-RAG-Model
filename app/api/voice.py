import logging
import time
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from app.schemas.response import VoiceQueryResponse
from app.schemas.common import LatencyMetrics
from app.core.dependencies import get_rag_pipeline, get_speech_service, get_metrics_service
from app.rag.pipeline import RAGPipeline
from app.services.speech_service import SpeechService
from app.services.metrics_service import MetricsService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post('/voice-query', response_model=VoiceQueryResponse)
async def voice_query(
    audio: UploadFile = File(...),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    speech: SpeechService = Depends(get_speech_service),
    metrics: MetricsService = Depends(get_metrics_service)
):
    """
    Voice query endpoint.
    1. Accept multipart audio file
    2. Transcribe via Sarvam Saaras v3
    3. Run the same RAG pipeline as /api/query
    """
    total_start = time.perf_counter()
    
    # Validate audio
    if audio.size is not None and audio.size > 25 * 1024 * 1024:  # 25MB limit
        raise HTTPException(status_code=413, detail='Audio file too large (max 25MB)')
    
    allowed_types = {'audio/wav', 'audio/mpeg', 'audio/webm', 'audio/ogg', 'audio/mp4', 'application/octet-stream'}
    if audio.content_type and audio.content_type not in allowed_types:
        logger.warning(f'Unusual audio content type: {audio.content_type}. Proceeding anyway.')
    
    try:
        audio_data = await audio.read()
        
        # Transcribe
        stt_result = await speech.transcribe(audio_data, audio.filename or 'audio.wav')
        transcript = stt_result['text']
        stt_ms = stt_result['latency_ms']
        
        if not transcript or not transcript.strip():
            raise HTTPException(status_code=400, detail='Could not transcribe audio. Please speak clearly and try again.')
        
        # Run same RAG pipeline — no duplication
        rag_result = pipeline.answer_query(transcript)
        
        latency_bd = rag_result.get('latency_breakdown', {})
        total_ms = (time.perf_counter() - total_start) * 1000
        
        latency = LatencyMetrics(
            stt_ms=stt_ms,
            embedding_ms=latency_bd.get('embedding_ms', 0.0),
            bm25_ms=latency_bd.get('bm25_ms', 0.0),
            vector_ms=latency_bd.get('vector_ms', 0.0),
            rrf_ms=latency_bd.get('rrf_ms', 0.0),
            llm_ms=latency_bd.get('llm_ms', 0.0),
            total_ms=total_ms
        )
        
        metrics.record_voice_query(total_ms, latency_bd)
        
        return VoiceQueryResponse(
            transcript=transcript,
            answer=rag_result['answer'],
            sources=rag_result.get('sources', []),
            chunks=rag_result.get('chunks', []),
            latency=latency
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Voice query failed: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail='Voice query processing failed. Please try again.')
