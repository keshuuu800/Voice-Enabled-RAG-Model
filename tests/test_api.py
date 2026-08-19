"""
API integration tests using FastAPI TestClient.
These tests run against the full app with mocked services.
"""
import pytest
import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from app.main import app
from app.schemas.common import Source, LatencyMetrics

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_app_state():
    """Mock all app.state services."""
    mock_pipeline = MagicMock()
    mock_pipeline.answer_query.return_value = {
        'answer': 'The HH-Goa scheme provides affordable housing.',
        'sources': [Source(source='scheme_overview.txt', page=1, chunk_id='test-id')],
        'latency_breakdown': {
            'embedding_ms': 15.0, 'bm25_ms': 2.0, 'vector_ms': 5.0,
            'rrf_ms': 0.5, 'llm_ms': 80.0, 'total_ms': 102.5
        },
        'retrieved_count': 3,
        'query': 'What is the scheme?'
    }
    
    mock_vector_store = MagicMock()
    mock_vector_store.count.return_value = 42
    
    mock_bm25 = MagicMock()
    mock_bm25.is_ready.return_value = True
    
    mock_embedding = MagicMock()
    mock_speech = MagicMock()
    mock_speech.is_available.return_value = False
    
    mock_metrics = MagicMock()
    mock_metrics.get_stats.return_value = {'request_count': 0}
    mock_metrics.record_text_query.return_value = None
    
    app.state.rag_pipeline = mock_pipeline
    app.state.vector_store = mock_vector_store
    app.state.bm25_index = mock_bm25
    app.state.embedding_service = mock_embedding
    app.state.speech_service = mock_speech
    app.state.metrics_service = mock_metrics
    app.state.settings = MagicMock()
    
    return app

@pytest.fixture
def client(mock_app_state):
    with TestClient(mock_app_state, raise_server_exceptions=True) as c:
        yield c

# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health_endpoint(client):
    resp = client.get('/api/health')
    assert resp.status_code == 200
    data = resp.json()
    assert 'status' in data
    assert 'services' in data

def test_query_endpoint_returns_answer(client):
    resp = client.post('/api/query', json={'query': 'What is the objective?'})
    assert resp.status_code == 200
    data = resp.json()
    assert 'answer' in data
    assert 'sources' in data
    assert 'latency' in data
    assert data['answer'] != ''

def test_query_endpoint_rejects_empty_query(client):
    resp = client.post('/api/query', json={'query': ''})
    assert resp.status_code == 422  # Pydantic validation error

def test_query_endpoint_rejects_whitespace(client):
    resp = client.post('/api/query', json={'query': '   '})
    assert resp.status_code == 422

def test_metrics_endpoint(client):
    resp = client.get('/api/metrics')
    assert resp.status_code == 200

def test_api_root(client):
    resp = client.get('/api')
    assert resp.status_code == 200
    data = resp.json()
    assert 'name' in data

def test_upload_document_rejects_unsupported_format(client):
    file_bytes = io.BytesIO(b"dummy data")
    resp = client.post(
        '/api/upload-document',
        files={'file': ('document.exe', file_bytes, 'application/octet-stream')}
    )
    assert resp.status_code == 400
    assert 'Unsupported file format' in resp.json()['detail']

def test_upload_document_rejects_large_file(client):
    large_bytes = io.BytesIO(b"A" * (11 * 1024 * 1024))
    resp = client.post(
        '/api/upload-document',
        files={'file': ('large_doc.txt', large_bytes, 'text/plain')}
    )
    assert resp.status_code == 413
    assert 'File too large' in resp.json()['detail']
