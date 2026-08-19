import argparse
import time
import json
import sys
import numpy as np
import httpx

TEST_QUERIES = [
    'What is the objective of the HH-Goa scheme?',
    'Who is eligible for the EWS category?',
    'What documents are required to apply?',
    'How much subsidy does an LIG applicant receive?',
    'How can I apply for the scheme?',
    'What is the income limit for EWS?',
    'Which talukas are covered under the scheme?',
    'What is the total budget of HH-Goa?',
    'What is the construction standard?',
    'How do I raise a grievance?',
]

def run_benchmark(base_url: str, n_requests: int):
    print(f'\n{"="*60}')
    print(f'HH-Goa Voice RAG — Latency Benchmark')
    print(f'{"="*60}')
    print(f'Target: {base_url}')
    print(f'Requests: {n_requests}')
    print()
    
    latencies = []
    stage_latencies = {k: [] for k in ['embedding_ms', 'bm25_ms', 'vector_ms', 'rrf_ms', 'llm_ms']}
    errors = 0
    
    with httpx.Client(base_url=base_url, timeout=60) as client:
        for i in range(n_requests):
            query = TEST_QUERIES[i % len(TEST_QUERIES)]
            try:
                start = time.perf_counter()
                resp = client.post('/api/query', json={'query': query})
                wall_time = (time.perf_counter() - start) * 1000
                
                if resp.status_code == 200:
                    data = resp.json()
                    reported_ms = data.get('latency', {}).get('total_ms', wall_time)
                    latencies.append(reported_ms)
                    for k in stage_latencies:
                        v = data.get('latency', {}).get(k, 0)
                        if v:
                            stage_latencies[k].append(v)
                    print(f'  [{i+1:3d}/{n_requests}] {reported_ms:7.1f}ms | {query[:45]}')
                else:
                    errors += 1
                    print(f'  [{i+1:3d}/{n_requests}] ERROR {resp.status_code}')
            except Exception as e:
                errors += 1
                print(f'  [{i+1:3d}/{n_requests}] EXCEPTION: {e}')
    
    if not latencies:
        print('No successful requests. Cannot generate report.')
        return
    
    arr = np.array(latencies)
    print(f'\n{"="*60}')
    print(f'Latency Report (Total End-to-End)')
    print(f'{"-"*60}')
    print(f'  Successful requests: {len(latencies)}')
    print(f'  Failed requests:     {errors}')
    print(f'  P50:    {np.percentile(arr, 50):8.1f} ms')
    print(f'  P70:    {np.percentile(arr, 70):8.1f} ms')
    print(f'  P90:    {np.percentile(arr, 90):8.1f} ms')
    print(f'  Avg:    {np.mean(arr):8.1f} ms')
    print(f'  Min:    {np.min(arr):8.1f} ms')
    print(f'  Max:    {np.max(arr):8.1f} ms')
    
    print(f'\nPer-Stage Averages')
    print(f'{"-"*60}')
    for stage, vals in stage_latencies.items():
        if vals:
            print(f'  {stage:15s}: {np.mean(vals):7.1f} ms avg')
    print(f'{"="*60}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:8000')
    parser.add_argument('--n', type=int, default=20)
    args = parser.parse_args()
    run_benchmark(args.url, args.n)
