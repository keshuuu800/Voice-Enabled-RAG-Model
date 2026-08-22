"""
Embedding service — supports two backends:
  1. Gemini API  (EMBEDDING_BACKEND=gemini, default on Render — zero local RAM)
  2. SentenceTransformers (EMBEDDING_BACKEND=local — for local dev / GPU machines)

Set GEMINI_API_KEY in environment to use the Gemini backend.
"""
import time
import logging
import numpy as np
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Unified embedding service. Picks backend from settings:
      - 'gemini' → google.generativeai embeddings (API-based, no local model)
      - 'local'  → sentence-transformers (local model, needs ~300 MB RAM)
    """

    def __init__(self, model_name: str = None, device: str = None):
        settings = get_settings()
        self.backend = settings.embedding_backend   # 'gemini' | 'local'
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self._model = None  # lazy-loaded for local backend

        if self.backend == "gemini":
            self._init_gemini(settings)
        else:
            self._init_local()

    # ── Backend init ──────────────────────────────────────────────────────────

    def _init_gemini(self, settings):
        import google.generativeai as genai
        api_key = settings.gemini_api_key or settings.llm_api_key
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY must be set when EMBEDDING_BACKEND=gemini. "
                "Get a free key at https://aistudio.google.com"
            )
        genai.configure(api_key=api_key)
        self._genai = genai
        # gemini-embedding-004: 768-dim, free tier generous
        self._gemini_model = "models/text-embedding-004"
        logger.info(f"Embedding backend: Gemini API ({self._gemini_model}) — no local model loaded")

    def _init_local(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"Embedding backend: local SentenceTransformer ({self.model_name} on {self.device})")

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.backend == "gemini":
            return self._gemini_embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")
        return self._local_embed(texts)

    def embed_query(self, query: str) -> list[float]:
        if self.backend == "gemini":
            return self._gemini_embed_batch([query], task_type="RETRIEVAL_QUERY")[0]
        return self._local_embed([query])[0]

    def embed_with_latency(self, texts: list[str]) -> tuple[list[list[float]], float]:
        t0 = time.perf_counter()
        embeddings = self.embed_documents(texts)
        return embeddings, (time.perf_counter() - t0) * 1000.0

    def query_with_latency(self, query: str) -> tuple[list[float], float]:
        t0 = time.perf_counter()
        embedding = self.embed_query(query)
        return embedding, (time.perf_counter() - t0) * 1000.0

    # ── Private helpers ───────────────────────────────────────────────────────

    def _gemini_embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        """
        Embed texts using Gemini API. Batches up to 100 texts per call.
        """
        results = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._genai.embed_content(
                model=self._gemini_model,
                content=batch,
                task_type=task_type,
            )
            # response['embedding'] is a list of vectors when content is a list
            embeddings = response.get("embedding", [])
            if embeddings and not isinstance(embeddings[0], list):
                # Single text returned as flat list — wrap it
                embeddings = [embeddings]
            results.extend(embeddings)
        return results

    def _local_embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()
