import logging
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.schemas.common import Document, Chunk

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Breaks down documents into smaller chunks for vector embedding."""

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 300):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a single Document into Chunks."""
        splits = self.splitter.split_text(document.text)
        chunks = []
        
        for idx, split_text in enumerate(splits):
            if not split_text or len(split_text) < 20:
                continue
                
            chunk_meta = {
                "document_id": document.metadata.get("document_id"),
                "source": document.metadata.get("source"),
                "page": document.metadata.get("page"),
                "chunk_id": str(uuid.uuid4()),
                "chunk_index": idx
            }
            
            chunks.append(Chunk(
                text=split_text,
                metadata=chunk_meta
            ))
            
        return chunks

    def chunk_batch(self, documents: list[Document]) -> list[Chunk]:
        """Process a list of Documents and return all Chunks."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk(doc))
            
        logger.info(f"Total chunks generated: {len(all_chunks)}")
        return all_chunks
