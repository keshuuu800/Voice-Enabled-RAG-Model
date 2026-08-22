"""
HH-Goa Voice RAG API — FastAPI application entry point.

Services are initialized ONCE during startup via the lifespan context manager
and stored on app.state for dependency injection.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.api import query as query_router
from app.api import voice as voice_router
from app.api import ingest as ingest_router
from app.api import health as health_router
from app.services.speech_service import SpeechService
from app.services.metrics_service import MetricsService
from retrieval.embeddings import EmbeddingService
from retrieval.chroma_store import VectorStore
from retrieval.bm25 import BM25Index
from retrieval.hybrid import HybridRetriever
from app.rag.pipeline import RAGPipeline
from generation.llm import get_llm_provider


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup, clean up on shutdown."""
    setup_logging()
    logger = logging.getLogger(__name__)
    settings = get_settings()
    
    logger.info('=== HH-Goa Voice RAG Starting Up ===')
    
    # Store settings on app.state for convenience
    app.state.settings = settings
    
    # Initialize embedding model (loads BAAI/bge-m3)
    logger.info(f'Loading embedding model: {settings.embedding_model}')
    embedding_service = EmbeddingService(
        model_name=settings.embedding_model,
        device=settings.embedding_device
    )
    app.state.embedding_service = embedding_service
    
    # Initialize vector store (connects to existing ChromaDB or creates new)
    logger.info(f'Connecting to ChromaDB at: {settings.chroma_path}')
    vector_store = VectorStore(
        persist_path=settings.chroma_path,
        collection_name=settings.chroma_collection_name
    )
    app.state.vector_store = vector_store
    
    # Initialize BM25 index (load from disk if available)
    logger.info('Loading BM25 index...')
    bm25_index = BM25Index()
    loaded = bm25_index.load(settings.bm25_path)
    if not loaded:
        logger.warning('No BM25 index found at disk. Run ingestion first: python -m ingestion.pipeline')
    app.state.bm25_index = bm25_index
    
    # Initialize hybrid retriever
    hybrid_retriever = HybridRetriever(embedding_service, vector_store, bm25_index)
    app.state.hybrid_retriever = hybrid_retriever
    
    # Initialize LLM provider
    logger.info(f'Initializing LLM provider: {settings.llm_provider}')
    llm_provider = get_llm_provider(settings)
    app.state.llm_provider = llm_provider
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(hybrid_retriever, llm_provider, settings)
    app.state.rag_pipeline = rag_pipeline
    
    # Initialize speech service
    speech_service = SpeechService()
    app.state.speech_service = speech_service
    
    # Initialize metrics service
    metrics_service = MetricsService()
    app.state.metrics_service = metrics_service
    
    logger.info('=== All services initialized. Ready to serve ===')
    logger.info(f'ChromaDB chunks: {vector_store.count()}')
    logger.info(f'BM25 ready: {bm25_index.is_ready()}')
    
    # Detect embedding dimension mismatch (e.g., switching from local 384-dim to Gemini 768-dim).
    # If stored embeddings have a different dimension than what the current backend produces,
    # ChromaDB will crash on the first query. Wipe and let users re-upload to fix cleanly.
    if vector_store.count() > 0:
        try:
            probe = vector_store.collection.get(limit=1, include=["embeddings"])
            stored_dim = len(probe["embeddings"][0]) if probe and probe.get("embeddings") else None
            sample_emb = embedding_service.embed_query("test")
            current_dim = len(sample_emb)
            if stored_dim and stored_dim != current_dim:
                logger.warning(
                    f"Embedding dimension mismatch: stored={stored_dim}, current={current_dim}. "
                    "Wiping ChromaDB collection. Users must re-upload documents."
                )
                vector_store.delete_collection()
                bm25_index._index = None
                bm25_index._chunks = []
                bm25_index._tokenized_corpus = []
        except Exception as dim_err:
            logger.warning(f"Could not verify embedding dimensions: {dim_err}")
    
    yield
    
    logger.info('=== Shutting down ===')


app = FastAPI(
    title='HH-Goa Voice RAG API',
    description='Low-latency voice-enabled document-grounded RAG assistant for the Harmonious Homes Goa scheme',
    version='1.0.0',
    lifespan=lifespan
)

# CORS — allow localhost for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Register API routers FIRST
app.include_router(health_router.router, prefix='/api', tags=['Health'])
app.include_router(query_router.router, prefix='/api', tags=['Query'])
app.include_router(voice_router.router, prefix='/api', tags=['Voice'])
app.include_router(ingest_router.router, prefix='/api', tags=['Ingestion'])

@app.get('/api')
async def api_info():
    return {'name': 'HH-Goa Voice RAG API', 'version': '1.0.0', 'docs': '/docs'}

# Mount frontend directory to serve static assets directly at root level
frontend_path = Path(__file__).parent.parent / 'frontend'
if frontend_path.exists():
    app.mount('/', StaticFiles(directory=str(frontend_path), html=True), name='frontend')
