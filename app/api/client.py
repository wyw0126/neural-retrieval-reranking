"""Small synchronous HTTP client for the retrieval API."""
from time import perf_counter
import httpx


class RetrievalClient:
    def __init__(self, base_url="http://127.0.0.1:8000", timeout=30.0):
        self.http = httpx.Client(base_url=base_url, timeout=timeout, trust_env=False)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.http.close()

    def request(self, method, path, *, expected=200, **kwargs):
        started = perf_counter()
        response = self.http.request(method, path, **kwargs)
        elapsed = (perf_counter() - started) * 1000
        if response.status_code != expected:
            raise RuntimeError(f"{method} {path}: expected {expected}, got {response.status_code}")
        data = response.json()
        rid = response.headers.get("x-request-id")
        if not rid:
            raise RuntimeError("Missing request ID")
        server_ms = float(response.headers["x-process-time-ms"])
        return data, {"client_ms": elapsed, "server_ms": server_ms, "request_id": rid}

    def search(self, query, retrieve_top_k=100, rerank_top_k=10, use_reranker=True):
        return self.request("POST", "/search", json=dict(query=query,
            retrieve_top_k=retrieve_top_k, rerank_top_k=rerank_top_k,
            use_reranker=use_reranker))
