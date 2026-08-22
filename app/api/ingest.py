import os
import logging
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from app.schemas.query import IngestRequest
from app.schemas.response import IngestResponse
from ingestion.pipeline import IngestionPipeline
from ingestion.loaders import SUPPORTED_EXTENSIONS

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit

# The ONLY directory that holds user-uploaded files
RAW_DIR = Path("./data/raw")


@router.post('/ingest', response_model=IngestResponse)
async def ingest_documents(body: IngestRequest, request: Request):
    """
    Ingest documents from a directory into ChromaDB and BM25 index.
    """
    state = request.app.state
    data_dir = body.data_dir
    
    try:
        ingestion = IngestionPipeline()
        chunks = ingestion.ingest_directory(data_dir)
        
        if not chunks:
            return IngestResponse(
                status='warning',
                documents_loaded=0,
                chunks_created=0,
                message=f'No supported documents found in {data_dir}'
            )
        
        # Clear and rebuild vector store
        state.vector_store.delete_collection()
        embeddings = state.embedding_service.embed_documents([c.text for c in chunks])
        state.vector_store.add_chunks(chunks, embeddings)
        
        # Rebuild BM25 index
        state.bm25_index.build(chunks)
        state.bm25_index.save(state.settings.bm25_path)
        
        unique_sources = len(set(c.metadata.get('document_id', '') for c in chunks))
        logger.info(f'Ingestion complete: {unique_sources} documents, {len(chunks)} chunks')
        
        return IngestResponse(
            status='success',
            documents_loaded=unique_sources,
            chunks_created=len(chunks),
            message=f'Successfully ingested {unique_sources} documents into {len(chunks)} chunks'
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f'Directory not found: {data_dir}')
    except Exception as e:
        logger.error(f'Ingestion failed: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'Ingestion failed: {str(e)}')


@router.post('/upload-document', response_model=IngestResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Upload a document (PDF, TXT, MD, JSON, CSV up to 10 MB).
    Processes, chunks, embeds with BGE-M3, and ADDS to ChromaDB + BM25 index.
    Only user-uploaded files in data/raw are ever indexed.
    """
    state = request.app.state
    
    # 1. Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    
    # 2. Read content and check size (<= 10 MB)
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents) / (1024*1024):.1f} MB). Maximum allowed size is 10 MB."
        )
    
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    
    # 3. Save to data/raw — the ONLY place user documents live
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RAW_DIR / file.filename
    
    try:
        with open(save_path, "wb") as f:
            f.write(contents)
        logger.info(f"Saved uploaded file to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file on server: {str(e)}")
    
    # 4. Process ONLY the new file through ingestion pipeline
    try:
        ingestion = IngestionPipeline()
        new_chunks = ingestion.ingest_file(str(save_path))
        
        if not new_chunks:
            raise HTTPException(
                status_code=400,
                detail=f"No valid text could be extracted from '{file.filename}'."
            )
        
        # 5. ADD new chunks to ChromaDB (incremental — no wipe)
        embeddings = state.embedding_service.embed_documents([c.text for c in new_chunks])
        state.vector_store.add_chunks(new_chunks, embeddings)
        
        # 6. Rebuild BM25 by merging existing in-memory chunks with the new ones.
        #    This avoids re-ingesting all files (which spikes memory on low-RAM hosts).
        existing_chunks = state.bm25_index._chunks if state.bm25_index.is_ready() else []
        all_chunks = existing_chunks + new_chunks
        state.bm25_index.build(all_chunks)
        state.bm25_index.save(state.settings.bm25_path)
        
        logger.info(f"Successfully processed '{file.filename}': {len(new_chunks)} new chunks added")
        
        return IngestResponse(
            status='success',
            documents_loaded=1,
            chunks_created=len(new_chunks),
            message=f"Successfully ingested '{file.filename}' ({len(new_chunks)} chunks indexed and ready for queries!)"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process uploaded file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.delete('/clear-knowledge', response_model=IngestResponse)
async def clear_knowledge_base(request: Request):
    """
    Wipe the entire knowledge base (ChromaDB + BM25 + uploaded files in data/raw).
    Use when you want a fresh start.
    """
    state = request.app.state
    import shutil

    try:
        # Clear ChromaDB collection
        state.vector_store.delete_collection()
        logger.info("ChromaDB collection cleared.")

        # Reset BM25
        state.bm25_index._index = None
        state.bm25_index._chunks = []
        state.bm25_index._tokenized_corpus = []
        logger.info("BM25 index reset.")

        # Delete all files in data/raw
        if RAW_DIR.exists():
            shutil.rmtree(str(RAW_DIR))
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("data/raw cleared.")

        return IngestResponse(
            status='success',
            documents_loaded=0,
            chunks_created=0,
            message="Knowledge base cleared. Upload new documents to get started."
        )
    except Exception as e:
        logger.error(f"Failed to clear knowledge base: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear knowledge base: {str(e)}")
