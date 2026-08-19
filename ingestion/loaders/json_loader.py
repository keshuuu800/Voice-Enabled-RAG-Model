"""JSON document loader — handles arrays, objects, and primitive values."""
import json
import logging
from pathlib import Path
from ingestion.loaders.base import BaseLoader
from app.schemas.common import Document

logger = logging.getLogger(__name__)


class JSONLoader(BaseLoader):
    """Loads JSON files. Handles list-of-dicts, single dict, or scalar."""

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".json"

    def load(self, file_path: str) -> list[Document]:
        file_path = str(file_path)
        source = Path(file_path).name
        doc_id = self._make_document_id(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in '{source}': {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to read '{source}': {e}")
            return []

        documents: list[Document] = []

        if isinstance(data, list):
            for idx, item in enumerate(data):
                text = self._to_text(item)
                if text.strip():
                    documents.append(Document(
                        text=text,
                        metadata={"source": source, "page": idx + 1, "document_id": doc_id},
                    ))
        elif isinstance(data, dict):
            # Try to extract a 'text' or 'content' key; else dump the whole dict
            if "text" in data:
                text = str(data["text"])
            elif "content" in data:
                text = str(data["content"])
            else:
                text = json.dumps(data, ensure_ascii=False, indent=2)
            documents.append(Document(
                text=text,
                metadata={"source": source, "page": None, "document_id": doc_id},
            ))
        else:
            # Scalar (string, number, bool)
            documents.append(Document(
                text=str(data),
                metadata={"source": source, "page": None, "document_id": doc_id},
            ))

        logger.info(f"JSON '{source}': {len(documents)} document(s) loaded")
        return documents

    @staticmethod
    def _to_text(item: object) -> str:
        """Convert a JSON item to a readable string."""
        if isinstance(item, dict):
            # Check for text/content keys first
            if "text" in item:
                return str(item["text"])
            if "content" in item:
                return str(item["content"])
            # Format as 'Key: Value | Key: Value ...' for readability
            parts = [f"{k}: {v}" for k, v in item.items()]
            return " | ".join(parts)
        elif isinstance(item, str):
            return item
        else:
            return json.dumps(item, ensure_ascii=False)
