"""Plain-text and Markdown document loader."""
import logging
from pathlib import Path
from ingestion.loaders.base import BaseLoader
from app.schemas.common import Document

logger = logging.getLogger(__name__)

_TXT_EXTENSIONS = {".txt", ".md"}


class TXTLoader(BaseLoader):
    """Loads plain text and Markdown files as a single Document."""

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in _TXT_EXTENSIONS

    def load(self, file_path: str) -> list[Document]:
        file_path = str(file_path)
        source = Path(file_path).name
        doc_id = self._make_document_id(file_path)

        # Try UTF-8 first, fall back to latin-1 for legacy documents
        for encoding in ("utf-8", "latin-1"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            logger.error(f"Could not decode '{source}' with utf-8 or latin-1")
            return []

        if not text.strip():
            logger.warning(f"'{source}' is empty — skipping")
            return []

        logger.info(f"TXT/MD '{source}': {len(text)} chars loaded")
        return [Document(
            text=text,
            metadata={"source": source, "page": None, "document_id": doc_id},
        )]
