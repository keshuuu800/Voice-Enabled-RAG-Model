"""
BM25 lexical retrieval index.
Built from chunks and persisted to disk. Never rebuilt per query.
"""
import os
import pickle
import logging
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from typing import Optional
from app.schemas.common import Chunk, RetrievalResult

class BM25Index:
    """
    BM25 lexical retrieval index.
    """
    def __init__(self):
        self._index: Optional[BM25Okapi] = None
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self.logger = logging.getLogger(__name__)

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.split(r'[\s\.,;:!?()\[\]{}\'"\-_/\\]+', text.lower())
        return [t for t in tokens if len(t) >= 2]

    def build(self, chunks: list[Chunk]) -> None:
        """Build the index from a list of chunks."""
        self._chunks = chunks
        self._tokenized_corpus = [self._tokenize(chunk.text) for chunk in chunks]
        self._index = BM25Okapi(self._tokenized_corpus)
        self.logger.info(f"BM25 index built with {len(self._chunks)} documents")

    def save(self, directory: str) -> None:
        """Save the index to disk."""
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, 'bm25_index.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': self._index,
                'chunks': self._chunks,
                'corpus': self._tokenized_corpus
            }, f)
        self.logger.info(f"BM25 index saved to {filepath}")

    def load(self, directory: str) -> bool:
        """Load the index from disk."""
        filepath = os.path.join(directory, 'bm25_index.pkl')
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self._index = data['index']
                self._chunks = data['chunks']
                self._tokenized_corpus = data['corpus']
            self.logger.info(f"BM25 index loaded: {len(self._chunks)} documents")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load BM25 index from {filepath}: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Search the index for the top_k most relevant chunks."""
        if not self.is_ready():
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        max_score = max(scores) if len(scores) > 0 else 0
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score == 0:
                continue
                
            normalized_score = score / max_score if max_score > 0 else 0.0
            chunk = self._chunks[idx]
            
            res = RetrievalResult(
                chunk_id=chunk.metadata.get('chunk_id', ''),
                text=chunk.text,
                score=normalized_score,
                source=chunk.metadata.get('source', ''),
                page=chunk.metadata.get('page', None),
                document_id=chunk.metadata.get('document_id', '')
            )
            results.append(res)
            
        return results

    def is_ready(self) -> bool:
        """Check if the index is built and ready."""
        return self._index is not None and len(self._chunks) > 0
