"""
Tests for BM25 retrieval and RRF.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.bm25 import BM25Index
from retrieval.rrf import reciprocal_rank_fusion
from app.schemas.common import Chunk, RetrievalResult
import uuid

def make_chunk(text: str, source: str = 'test.txt', idx: int = 0) -> Chunk:
    return Chunk(
        text=text,
        metadata={
            'chunk_id': str(uuid.uuid4()),
            'document_id': source.split('.')[0],
            'source': source,
            'page': None,
            'chunk_index': idx
        }
    )

def test_bm25_finds_relevant_document():
    index = BM25Index()
    chunks = [
        make_chunk('The HH-Goa scheme provides affordable housing for EWS families.', 'scheme.txt', 0),
        make_chunk('The weather in Goa is tropical and humid during monsoons.', 'weather.txt', 1),
        make_chunk('The application deadline is March 2025 for all categories.', 'deadline.txt', 2),
    ]
    index.build(chunks)
    
    results = index.search('affordable housing scheme', top_k=3)
    assert len(results) > 0
    # The housing document should rank first
    assert 'scheme.txt' in results[0].source or 'housing' in results[0].text.lower()

def test_bm25_returns_empty_for_no_match():
    index = BM25Index()
    chunks = [make_chunk('Document about housing in Goa.', 'doc.txt', 0)]
    index.build(chunks)
    
    results = index.search('xyzzy quux frobnicate', top_k=5)
    # All scores should be 0, so results list should be empty
    assert results == [] or all(r.score == 0 for r in results)

def test_bm25_not_ready_before_build():
    index = BM25Index()
    assert not index.is_ready()
    results = index.search('anything', top_k=5)
    assert results == []

def test_rrf_combines_lists():
    def make_result(chunk_id, score, text='text'):
        return RetrievalResult(
            chunk_id=chunk_id, text=text, score=score,
            source='doc.txt', page=None, document_id='doc'
        )
    
    list1 = [make_result('a', 0.9), make_result('b', 0.8), make_result('c', 0.7)]
    list2 = [make_result('b', 0.95), make_result('a', 0.6), make_result('d', 0.5)]
    
    results = reciprocal_rank_fusion([list1, list2], k=60, top_k=3)
    
    assert len(results) <= 3
    # 'a' and 'b' appear in both lists, should score higher than 'c' or 'd'
    top_ids = [r.chunk_id for r in results]
    assert 'a' in top_ids
    assert 'b' in top_ids

def test_rrf_handles_empty_list():
    results = reciprocal_rank_fusion([], k=60, top_k=5)
    assert results == []

def test_rrf_handles_one_list():
    def make_result(chunk_id):
        return RetrievalResult(
            chunk_id=chunk_id, text='text', score=0.5,
            source='doc.txt', page=None, document_id='doc'
        )
    results = reciprocal_rank_fusion([[make_result('x'), make_result('y')]], k=60, top_k=2)
    assert len(results) == 2
