"""Exercise RAG against an existing retrieval HTTP service without loading FAISS twice."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from app.api.client import RetrievalClient
from app.config import load_settings
from app.llm.client import LLMClient
from app.rag.chain import RAGEngine
from app.schemas import SearchResponse


class HTTPBackend:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def search(self, query, **kwargs):
        with RetrievalClient(base_url=self.base_url) as client:
            data, _ = client.search(query, **kwargs)
        return SearchResponse.model_validate(data)


async def run(args):
    settings = load_settings()
    async with LLMClient(settings) as llm:
        engine = RAGEngine(settings, HTTPBackend(args.base_url), llm)
        result = await engine.answer(args.query, retrieve_top_k=args.retrieve_top_k,
                                     rerank_top_k=args.top_k, use_reranker=not args.no_rerank)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
    if args.expect_answer and result.answer.insufficient_evidence:
        raise SystemExit("RAG smoke: model abstained, inspect evidence")
    print("RAG Chain Smoke PASSED", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--retrieve-top-k", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--expect-answer", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except httpx.ConnectError:
        raise SystemExit("Cannot connect to retrieval service; verify /health and start uvicorn if needed.") from None


if __name__ == "__main__":
    main()
