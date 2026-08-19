"""
Metrics collection for latency analytics.
Stores request latencies in memory for P50/P70/P90 calculation.
"""
import time
import threading
import numpy as np
from collections import deque
from typing import Optional

class MetricsService:
    def __init__(self, max_history: int = 1000):
        self._lock = threading.Lock()
        self._text_query_latencies: deque = deque(maxlen=max_history)
        self._voice_query_latencies: deque = deque(maxlen=max_history)
        self._request_count = 0
    
    def record_text_query(self, total_ms: float, latency_breakdown: dict) -> None:
        with self._lock:
            self._text_query_latencies.append(total_ms)
            self._request_count += 1
    
    def record_voice_query(self, total_ms: float, latency_breakdown: dict) -> None:
        with self._lock:
            self._voice_query_latencies.append(total_ms)
            self._request_count += 1
    
    def get_stats(self) -> dict:
        """Return P50, P70, P90, avg, min, max for text and voice queries."""
        with self._lock:
            result = {'request_count': self._request_count}
            for name, data in [('text_query', self._text_query_latencies), ('voice_query', self._voice_query_latencies)]:
                if data:
                    arr = np.array(list(data))
                    result[name] = {
                        'count': len(arr),
                        'p50_ms': float(np.percentile(arr, 50)),
                        'p70_ms': float(np.percentile(arr, 70)),
                        'p90_ms': float(np.percentile(arr, 90)),
                        'avg_ms': float(np.mean(arr)),
                        'min_ms': float(np.min(arr)),
                        'max_ms': float(np.max(arr))
                    }
                else:
                    result[name] = {'count': 0, 'message': 'No requests yet'}
            return result
