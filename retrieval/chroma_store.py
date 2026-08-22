"""
ChromaDB vector store — persistent, initialized once at startup.
"""
import logging
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.schemas.common import Chunk, RetrievalResult
from typing import Optional

class VectorStore:
    """
    Persistent ChromaDB vector store for storing and searching chunk embeddings.
    """
    def __init__(self, persist_path: str, collection_name: str = 'documents'):
        self.logger = logging.getLogger(__name__)
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            metadata={"hnsw:space": "cosine"}
        )
        self.logger.info(f"ChromaDB initialized. Collection: {self.collection_name}, Count: {self.count()}")

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """
        Batch upsert chunks and their embeddings into ChromaDB.
        """
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            ids = [chunk.metadata['chunk_id'] for chunk in batch_chunks]
            documents = [chunk.text for chunk in batch_chunks]
            
            metadatas = []
            for chunk in batch_chunks:
                meta = {}
                for k, v in chunk.metadata.items():
                    if v is None:
                        meta[k] = ""
                    elif isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    else:
                        meta[k] = str(v)
                        
                # Ensure page is int
                if 'page' not in meta or not isinstance(meta['page'], int):
                    meta['page'] = int(meta.get('page', 0)) if str(meta.get('page', 0)).isdigit() else 0
                metadatas.append(meta)
            
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=batch_embeddings
            )
        self.logger.info(f"Added {len(chunks)} chunks to ChromaDB")

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[RetrievalResult]:
        """
        Search for the top_k most similar chunks to the query embedding.
        """
        n_results = min(top_k, self.count())
        if n_results == 0:
            return []
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        retrieval_results = []
        if not results['ids'] or not results['ids'][0]:
            return []
            
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            text = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            score = 1.0 - distance
            score = max(0.0, min(1.0, score))
            
            res = RetrievalResult(
                chunk_id=doc_id,
                text=text,
                score=score,
                source=metadata.get('source', ''),
                page=metadata.get('page', None),
                document_id=metadata.get('document_id', '')
            )
            retrieval_results.append(res)
            
        retrieval_results.sort(key=lambda x: x.score, reverse=True)
        return retrieval_results

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        return self.collection.count()

    def is_empty(self) -> bool:
        """Check if the collection is empty."""
        return self.count() == 0

    def delete_collection(self) -> None:
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, 
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"Deleted and recreated collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
