"""
Embedding service using BAAI/bge-m3.
The model is loaded ONCE at initialization and reused for all requests.
"""
import time
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings

class EmbeddingService:
    """
    Service for generating embeddings using a SentenceTransformer model.
    """
    def __init__(self, model_name: str = None, device: str = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        
        self.logger = logging.getLogger(__name__)
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.logger.info(f"Embedding model loaded: {self.model_name} on {self.device}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of documents.
        """
        if not texts:
            return []
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.
        """
        embedding = self.model.encode([query], normalize_embeddings=True)[0]
        return embedding.tolist()

    def embed_with_latency(self, texts: list[str]) -> tuple[list[list[float]], float]:
        """
        Embed a list of documents and measure the latency.
        """
        t0 = time.perf_counter()
        embeddings = self.embed_documents(texts)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return embeddings, latency_ms

    def query_with_latency(self, query: str) -> tuple[list[float], float]:
        """
        Embed a single query and measure the latency.
        """
        t0 = time.perf_counter()
        embedding = self.embed_query(query)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return embedding, latency_ms
