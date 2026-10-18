"""In-memory sliding-window rate limiter.

State lives in the process, so limits apply per instance. That is enough for a
single Render service; a multi-instance deployment would need Redis instead.
"""

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float, clock=time.monotonic):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Record a hit for `key`. Return None if allowed, else seconds to wait."""
        if self.max_requests <= 0:
            return None

        now = self._clock()

        with self._lock:
            hits = self._hits[key]

            while hits and now - hits[0] >= self.window_seconds:
                hits.popleft()

            if len(hits) >= self.max_requests:
                return max(self.window_seconds - (now - hits[0]), 0.0)

            hits.append(now)
            return None
