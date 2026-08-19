"""CSV document loader using pandas — groups rows into batched Documents."""
import logging
from pathlib import Path
from ingestion.loaders.base import BaseLoader
from app.schemas.common import Document

logger = logging.getLogger(__name__)

_ROWS_PER_DOC = 50  # Group this many rows into one Document to avoid tiny fragments


class CSVLoader(BaseLoader):
    """Loads CSV files. Each batch of rows becomes one Document."""

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".csv"

    def load(self, file_path: str) -> list[Document]:
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas not installed. Run: pip install pandas")
            return []

        file_path = str(file_path)
        source = Path(file_path).name
        doc_id = self._make_document_id(file_path)

        for encoding in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Failed to read CSV '{source}': {e}")
                return []
        else:
            logger.error(f"Could not decode CSV '{source}'")
            return []

        if df.empty:
            logger.warning(f"CSV '{source}' is empty")
            return []

        documents: list[Document] = []
        columns = list(df.columns)

        # Convert rows to human-readable strings and batch them
        row_texts: list[str] = []
        for _, row in df.iterrows():
            parts = [f"{col}: {row[col]}" for col in columns if pd.notna(row[col])]
            row_texts.append(" | ".join(parts))

        # Group into batches
        for batch_idx, start in enumerate(range(0, len(row_texts), _ROWS_PER_DOC)):
            batch = row_texts[start : start + _ROWS_PER_DOC]
            text = "\n".join(batch)
            documents.append(Document(
                text=text,
                metadata={
                    "source": source,
                    "page": batch_idx + 1,
                    "document_id": doc_id,
                },
            ))

        logger.info(f"CSV '{source}': {len(df)} rows → {len(documents)} document(s)")
        return documents
