"""Abstract base class for all document loaders."""
from abc import ABC, abstractmethod
from pathlib import Path
from app.schemas.common import Document


class BaseLoader(ABC):
    """Every loader must implement load() and supports()."""

    @abstractmethod
    def load(self, file_path: str) -> list[Document]:
        """Load a file and return a list of normalized Documents.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            list[Document]: One or more documents extracted from the file.
                            Never returns None; returns [] on failure.
        """
        ...

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Return True if this loader can handle the given file path."""
        ...

    def _make_document_id(self, file_path: str) -> str:
        """Generate a stable document_id from the file stem."""
        return Path(file_path).stem
