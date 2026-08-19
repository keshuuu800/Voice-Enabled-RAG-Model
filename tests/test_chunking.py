"""
Tests for the chunking pipeline.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.cleaner import TextCleaner
from ingestion.chunker import DocumentChunker
from app.schemas.common import Document

def test_cleaner_removes_extra_whitespace():
    cleaner = TextCleaner()
    result = cleaner.clean('Hello   \n\n\n\nWorld')
    assert '\n\n\n' not in result
    assert 'Hello' in result
    assert 'World' in result

def test_cleaner_preserves_content():
    cleaner = TextCleaner()
    text = 'The scheme objective is affordable housing.'
    result = cleaner.clean(text)
    assert 'affordable housing' in result

def test_cleaner_handles_empty():
    cleaner = TextCleaner()
    result = cleaner.clean('')
    assert result == ''

def test_chunker_creates_chunks():
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    doc = Document(
        text='This is sentence one. ' * 30,
        metadata={'source': 'test.txt', 'document_id': 'test', 'page': None}
    )
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    assert all(c.text for c in chunks)

def test_chunk_metadata_preserved():
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    doc = Document(
        text='Content here. ' * 30,
        metadata={'source': 'test.txt', 'document_id': 'doc_001', 'page': 5}
    )
    chunks = chunker.chunk(doc)
    for i, chunk in enumerate(chunks):
        assert chunk.metadata['source'] == 'test.txt'
        assert chunk.metadata['document_id'] == 'doc_001'
        assert chunk.metadata['page'] == 5
        assert chunk.metadata['chunk_index'] == i
        assert 'chunk_id' in chunk.metadata

def test_chunk_sizes_within_bounds():
    chunk_size = 500
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=100)
    doc = Document(
        text='Word ' * 1000,
        metadata={'source': 'test.txt', 'document_id': 'test', 'page': None}
    )
    chunks = chunker.chunk(doc)
    # Chunks should generally be within bounds (may slightly exceed for short overlap)
    for chunk in chunks:
        assert len(chunk.text) <= chunk_size + 200, f'Chunk too large: {len(chunk.text)}'

def test_short_doc_produces_one_chunk():
    chunker = DocumentChunker(chunk_size=2000, chunk_overlap=300)
    doc = Document(
        text='This is a very short document.',
        metadata={'source': 'short.txt', 'document_id': 'short', 'page': None}
    )
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
