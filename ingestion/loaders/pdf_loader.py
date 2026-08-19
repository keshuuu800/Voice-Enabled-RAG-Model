"""PDF document loader using pypdf."""
import logging
from pathlib import Path
from ingestion.loaders.base import BaseLoader
from app.schemas.common import Document

logger = logging.getLogger(__name__)

_PDF_EXTENSIONS = {".pdf"}


class PDFLoader(BaseLoader):
    """Loads PDF files page-by-page, preserving page numbers."""

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in _PDF_EXTENSIONS

    def load(self, file_path: str) -> list[Document]:
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.error("pypdf not installed. Run: pip install pypdf")
            return []

        documents: list[Document] = []
        file_path = str(file_path)
        doc_id = self._make_document_id(file_path)
        source = Path(file_path).name

        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logger.info(f"PDF '{source}': {total_pages} pages")

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num} of '{source}': {e}")
                    text = ""

                if not text or len(text.strip()) < 50:
                    text = f"[Page {page_num}: minimal or scanned content — no extractable text]"

                documents.append(Document(
                    text=text,
                    metadata={
                        "source": source,
                        "page": page_num,
                        "document_id": doc_id,
                    },
                ))

        except Exception as e:
            logger.error(f"Failed to read PDF '{file_path}': {e}")

        return documents
