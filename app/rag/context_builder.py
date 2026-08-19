"""
Builds the context string passed to the LLM from retrieval results.
"""
from app.schemas.common import RetrievalResult, Source
from generation.prompts import format_context, truncate_context

class ContextBuilder:
    def build(self, results: list[RetrievalResult], max_chars: int = 12000) -> str:
        """Format retrieved chunks into LLM-ready context string."""
        context = format_context(results)
        return truncate_context(context, max_chars)
    
    def extract_sources(self, results: list[RetrievalResult]) -> list[Source]:
        """Deduplicated source list for the API response."""
        seen_chunk_ids = set()
        sources = []
        for r in results:
            if r.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(r.chunk_id)
                sources.append(Source(
                    source=r.source,
                    page=r.page,
                    chunk_id=r.chunk_id
                ))
        return sources
