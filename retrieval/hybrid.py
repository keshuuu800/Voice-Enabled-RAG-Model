"""
Hybrid retriever: BGE-M3 semantic + BM25 keyword, fused via RRF.
"""
import time
import logging
from typing import Optional
from retrieval.embeddings import EmbeddingService
from retrieval.chroma_store import VectorStore
from retrieval.bm25 import BM25Index
from retrieval.rrf import reciprocal_rank_fusion
from app.schemas.common import RetrievalResult
from app.core.config import get_settings

class HybridRetriever:
    """
    Combines semantic and BM25 search using Reciprocal Rank Fusion.
    """
    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore, bm25_index: BM25Index):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

    def retrieve(
        self,
        query: str,
        semantic_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        final_top_k: Optional[int] = None,
        rrf_k: Optional[int] = None
    ) -> tuple[list[RetrievalResult], dict[str, float]]:
        """
        Execute hybrid retrieval and return fused results along with latency metrics.
        """
        semantic_top_k = semantic_top_k or self.settings.semantic_top_k
        bm25_top_k = bm25_top_k or self.settings.bm25_top_k
        final_top_k = final_top_k or self.settings.final_top_k
        rrf_k = rrf_k or self.settings.rrf_k

        # 1. Embed query
        query_embedding, embedding_ms = self.embedding_service.query_with_latency(query)

        # 2. BM25 search
        t0 = time.perf_counter()
        bm25_results = self._run_bm25(query, bm25_top_k)
        bm25_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Vector search
        t0 = time.perf_counter()
        vector_results = self._run_vector(query_embedding, semantic_top_k)
        vector_ms = (time.perf_counter() - t0) * 1000.0

        # 4. RRF
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion([vector_results, bm25_results], k=rrf_k, top_k=final_top_k)
        rrf_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Log
        self.logger.info(f'Hybrid retrieval: semantic={len(vector_results)}, bm25={len(bm25_results)}, fused={len(fused)}')

        # 6. Return
        latency_dict = {
            'embedding_ms': embedding_ms,
            'bm25_ms': bm25_ms,
            'vector_ms': vector_ms,
            'rrf_ms': rrf_ms
        }
        return fused, latency_dict

    def _run_bm25(self, query: str, top_k: int) -> list[RetrievalResult]:
        if not self.bm25_index.is_ready():
            self.logger.warning("BM25 index is not ready. Returning empty list.")
            return []
        return self.bm25_index.search(query, top_k)

    def _run_vector(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        if self.vector_store.is_empty():
            self.logger.warning("Vector store is empty. Returning empty list.")
            return []
        return self.vector_store.search(query_embedding, top_k)
