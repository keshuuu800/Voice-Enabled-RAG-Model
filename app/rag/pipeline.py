"""
RAG Pipeline — orchestrates retrieval, guardrails, context building, and generation.

Usage:
    pipeline = RAGPipeline(hybrid_retriever, llm_provider, settings)
    result = pipeline.answer_query("What is the objective?")
    # result: {answer, sources, latency_breakdown: dict, retrieved_count: int, chunks: list[dict]}
"""
import time
import logging
from typing import Optional
from app.schemas.common import RetrievalResult, Source, LatencyMetrics
from app.rag.guardrails import (
    check_relevance, detect_injection_in_context, sanitize_query,
    check_query_not_empty, REFUSAL_MESSAGE
)
from app.rag.context_builder import ContextBuilder
from generation.prompts import build_rag_prompt
from retrieval.hybrid import HybridRetriever
from generation.llm import LLMProvider
from app.core.config import get_settings, Settings

class RAGPipeline:
    def __init__(self, hybrid_retriever: HybridRetriever, llm_provider: LLMProvider, settings: Optional[Settings] = None):
        self.hybrid_retriever = hybrid_retriever
        self.llm_provider = llm_provider
        self.settings = settings or get_settings()
        self.context_builder = ContextBuilder()
        self.logger = logging.getLogger(__name__)

    def answer_query(self, query: str) -> dict:
        total_start = time.perf_counter()
        query = sanitize_query(query)
        
        if not check_query_not_empty(query):
            total_ms = (time.perf_counter() - total_start) * 1000
            return {
                'answer': REFUSAL_MESSAGE,
                'sources': [],
                'chunks': [],
                'latency_breakdown': {'embedding_ms': 0.0, 'bm25_ms': 0.0, 'vector_ms': 0.0, 'rrf_ms': 0.0, 'llm_ms': 0.0, 'total_ms': total_ms},
                'retrieved_count': 0,
                'query': query
            }
            
        semantic_top_k = getattr(self.settings, 'semantic_top_k', 10)
        bm25_top_k = getattr(self.settings, 'bm25_top_k', 10)
        final_top_k = getattr(self.settings, 'final_top_k', 5)
        rrf_k = getattr(self.settings, 'rrf_k', 60)
        min_relevance_score = getattr(self.settings, 'min_relevance_score', 0.0)

        results, latency_dict = self.hybrid_retriever.retrieve(
            query,
            semantic_top_k=semantic_top_k,
            bm25_top_k=bm25_top_k,
            final_top_k=final_top_k,
            rrf_k=rrf_k
        )
        
        if not check_relevance(results, min_relevance_score):
            total_ms = (time.perf_counter() - total_start) * 1000
            latency_dict['llm_ms'] = 0.0
            latency_dict['total_ms'] = total_ms
            return {
                'answer': REFUSAL_MESSAGE,
                'sources': [],
                'chunks': [],
                'latency_breakdown': latency_dict,
                'retrieved_count': 0,
                'query': query
            }
            
        detect_injection_in_context(results)
        
        context_str = self.context_builder.build(results)
        system_prompt, user_message = build_rag_prompt(query, context_str)
        
        answer, llm_ms = self.llm_provider.generate_with_latency(system_prompt, user_message)
        sources = self.context_builder.extract_sources(results)
        
        total_ms = (time.perf_counter() - total_start) * 1000
        
        latency_breakdown = {
            'embedding_ms': latency_dict.get('embedding_ms', 0.0),
            'bm25_ms': latency_dict.get('bm25_ms', 0.0),
            'vector_ms': latency_dict.get('vector_ms', 0.0),
            'rrf_ms': latency_dict.get('rrf_ms', 0.0),
            'llm_ms': llm_ms,
            'total_ms': total_ms
        }
        
        chunk_details = [
            {
                'chunk_id': r.chunk_id,
                'source': r.source,
                'page': r.page,
                'text': r.text,
                'score': r.score
            }
            for r in results
        ]
        
        self.logger.info(f'Query answered in {total_ms:.1f}ms | chunks={len(results)} | llm_ms={llm_ms:.1f}')
        
        return {
            'answer': answer,
            'sources': sources,
            'chunks': chunk_details,
            'latency_breakdown': latency_breakdown,
            'retrieved_count': len(results),
            'query': query
        }
