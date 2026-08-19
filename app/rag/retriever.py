"""
Retriever wrapper — used by API endpoints to access hybrid retrieval.
"""
from retrieval.hybrid import HybridRetriever
from app.schemas.common import RetrievalResult
from app.core.config import get_settings

class RetrieverService:
    def __init__(self, hybrid_retriever: HybridRetriever):
        self.retriever = hybrid_retriever
        self.settings = get_settings()
    
    def retrieve(self, query: str) -> tuple[list[RetrievalResult], dict]:
        return self.retriever.retrieve(
            query,
            semantic_top_k=getattr(self.settings, 'semantic_top_k', 10),
            bm25_top_k=getattr(self.settings, 'bm25_top_k', 10),
            final_top_k=getattr(self.settings, 'final_top_k', 5),
            rrf_k=getattr(self.settings, 'rrf_k', 60)
        )
