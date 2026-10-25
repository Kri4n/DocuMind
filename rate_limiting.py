from collections import defaultdict, deque
from threading import Lock
from time import monotonic


class RequestRateLimiter:
    def __init__(self, requests: int, window_seconds: int) -> None:
        self.requests = requests
        self.window_seconds = window_seconds
        self._requests_by_client: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()
        self._last_cleanup = monotonic()

    def retry_after(self, client_id: str) -> int | None:
        now = monotonic()
        with self._lock:
            if now - self._last_cleanup >= self.window_seconds:
                expired = [
                    known_client
                    for known_client, client_requests in self._requests_by_client.items()
                    if not client_requests
                    or now - client_requests[-1] >= self.window_seconds
                ]
                for known_client in expired:
                    del self._requests_by_client[known_client]
                self._last_cleanup = now

            requests = self._requests_by_client[client_id]
            while requests and now - requests[0] >= self.window_seconds:
                requests.popleft()
            if len(requests) >= self.requests:
                return max(1, int(self.window_seconds - (now - requests[0])))
            requests.append(now)
            return None
