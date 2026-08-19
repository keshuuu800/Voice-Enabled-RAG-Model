import json
import sys
import httpx
import numpy as np
from pathlib import Path

# Load test queries
TEST_QUERIES_PATH = Path(__file__).parent / 'test_queries.json'

def evaluate(base_url: str = 'http://localhost:8000'):
    with open(TEST_QUERIES_PATH) as f:
        queries = json.load(f)
    
    print(f'\n{"="*60}')
    print(f'Retrieval Evaluation')
    print(f'{"="*60}')
    print(f'Test queries: {len(queries)}')
    print()
    
    hits = []
    reciprocal_ranks = []
    
    with httpx.Client(base_url=base_url, timeout=30) as client:
        for q in queries:
            query = q['query']
            expected_sources = [s.lower() for s in q.get('expected_sources', [])]
            
            try:
                resp = client.post('/api/query', json={'query': query})
                if resp.status_code != 200:
                    print(f'  SKIP (HTTP {resp.status_code}): {query[:50]}')
                    continue
                
                data = resp.json()
                returned_sources = [s['source'].lower() for s in data.get('sources', [])]
                
                # Hit@K: did at least one expected source appear in results?
                hit = any(exp in ' '.join(returned_sources) for exp in expected_sources)
                hits.append(1 if hit else 0)
                
                # MRR: reciprocal rank of first hit
                rr = 0.0
                for rank, src in enumerate(returned_sources, 1):
                    if any(exp in src for exp in expected_sources):
                        rr = 1.0 / rank
                        break
                reciprocal_ranks.append(rr)
                
                status = '✓ HIT' if hit else '✗ MISS'
                print(f'  {status} | {query[:50]}')
                if not hit:
                    print(f'        Expected: {expected_sources}')
                    print(f'        Got:      {returned_sources[:3]}')
            except Exception as e:
                print(f'  ERROR: {e}')
    
    if hits:
        print(f'\n{"="*60}')
        print(f'Results')
        print(f'{"-"*60}')
        print(f'  Hit@K:  {np.mean(hits):.3f} ({sum(hits)}/{len(hits)})')
        print(f'  MRR:    {np.mean(reciprocal_ranks):.3f}')
        print(f'{"="*60}\n')

if __name__ == '__main__':
    evaluate()
