"""Export all loaders and provide a factory function."""
from ingestion.loaders.base import BaseLoader
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.txt_loader import TXTLoader
from ingestion.loaders.json_loader import JSONLoader
from ingestion.loaders.csv_loader import CSVLoader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".json", ".csv"}

__all__ = ['BaseLoader', 'PDFLoader', 'TXTLoader', 'JSONLoader', 'CSVLoader', 'get_loader', 'SUPPORTED_EXTENSIONS']

def get_loader(file_path: str) -> BaseLoader:
    """Return the correct loader for the given file extension."""
    loaders = [PDFLoader(), TXTLoader(), JSONLoader(), CSVLoader()]
    
    for loader in loaders:
        if loader.supports(file_path):
            return loader
            
    raise ValueError(f"No suitable loader found for file: {file_path}")
