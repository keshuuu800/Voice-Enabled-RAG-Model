"""
Guardrails for the RAG pipeline.
Protects against: out-of-domain queries, prompt injection, empty retrieval.
"""
import re
import logging
from app.schemas.common import RetrievalResult

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = "I couldn't find that information in the provided documents."

# Patterns that suggest prompt injection attempts in retrieved content
INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?(previous|above|prior)\s+instructions',
    r'reveal\s+(your\s+)?(system\s+)?prompt',
    r'you\s+are\s+now\s+a',
    r'forget\s+(everything|all)',
    r'act\s+as\s+(if\s+you\s+are|a)',
    r'new\s+instructions?\s*:',
    r'disregard\s+(your\s+)?(previous|all)',
]

def check_relevance(results: list[RetrievalResult], min_score: float) -> bool:
    """Return True if there is at least one result above the minimum relevance threshold."""
    if not results:
        return False
    return any(r.score >= min_score for r in results)

def detect_injection_in_context(results: list[RetrievalResult]) -> bool:
    """Return True if any retrieved chunk contains suspected prompt injection patterns."""
    for r in results:
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, r.text, re.IGNORECASE):
                logger.warning(f"Suspected prompt injection detected in chunk {r.chunk_id}")
                return True
    return False

def sanitize_query(query: str) -> str:
    """Basic sanitization of the user query."""
    if not query:
        return ""
    cleaned = re.sub(r'\s+', ' ', query.strip())
    return cleaned[:2000]

def check_query_not_empty(query: str) -> bool:
    """Return True if query has meaningful content."""
    return bool(query and query.strip() and len(query.strip()) >= 2)
