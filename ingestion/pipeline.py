import logging
from pathlib import Path
from ingestion.loaders import get_loader
from ingestion.cleaner import TextCleaner
from ingestion.chunker import DocumentChunker
from app.schemas.common import Document, Chunk
from app.core.config import get_settings

class IngestionPipeline:
    """Orchestrates the ingestion, cleaning, and chunking of data."""

    def __init__(self):
        settings = get_settings()
        self.cleaner = TextCleaner()
        self.chunker = DocumentChunker(settings.chunk_size, settings.chunk_overlap)
        self.logger = logging.getLogger(__name__)

    def ingest_file(self, file_path: str) -> list[Chunk]:
        """Load, clean, and chunk a single file."""
        try:
            loader = get_loader(file_path)
            documents = loader.load(file_path)
            
            valid_docs = []
            for doc in documents:
                doc.text = self.cleaner.clean(doc.text)
                if self.cleaner.is_meaningful(doc.text):
                    valid_docs.append(doc)
                    
            chunks = self.chunker.chunk_batch(valid_docs)
            self.logger.info(f"File {file_path} loaded: {len(valid_docs)} valid documents, {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            self.logger.error(f"Error ingesting {file_path}: {e}", exc_info=True)
            return []

    def ingest_directory(self, data_dir: str) -> list[Chunk]:
        """Scan directory for supported files and ingest them."""
        dir_path = Path(data_dir)
        supported_exts = {'.pdf', '.txt', '.md', '.json', '.csv'}
        all_chunks = []
        file_count = 0

        if not dir_path.exists() or not dir_path.is_dir():
            self.logger.error(f"Directory not found: {data_dir}")
            return []

        for file_path in dir_path.rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                if file_path.suffix.lower() in supported_exts:
                    self.logger.info(f"Processing file: {file_path}")
                    chunks = self.ingest_file(str(file_path))
                    all_chunks.extend(chunks)
                    file_count += 1

        self.logger.info(f"Ingestion complete: {file_count} files processed, {len(all_chunks)} total chunks.")
        return all_chunks


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/raw"
    pipeline = IngestionPipeline()
    chunks = pipeline.ingest_directory(data_dir)
    
    print(f"\nTotal chunks: {len(chunks)}")
    if chunks:
        print(f"Sample chunk:\n{chunks[0].text[:200]}")
        print(f"Metadata: {chunks[0].metadata}")
