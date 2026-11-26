"""Command-line smoke test for the persistent two-stage retrieval engine."""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query", action="append", help="Repeat for multiple queries")
    mode.add_argument("--interactive", action="store_true")
    parser.add_argument("--retrieve-top-k", type=int)
    parser.add_argument("--top-k", type=int, help="Number of returned documents")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="JSON Lines output on stdout")
    parser.add_argument("--text-limit", type=int, default=300)
    args = parser.parse_args(argv)
    for key in ("retrieve_top_k", "top_k"):
        value = getattr(args, key)
        if value is not None and not 1 <= value <= 200:
            parser.error(f"{key} must be between 1 and 200")
    if args.repeat < 1 or args.warmup < 0 or args.text_limit < 0:
        parser.error("repeat must be positive; warmup and text-limit must be nonnegative")
    if args.query and any(not q.strip() for q in args.query):
        parser.error("query must not be empty")
    return args


def percentile(values, p):
    """Nearest-rank percentile, including endpoints."""
    if not values or not 0 <= p <= 100:
        raise ValueError("Invalid percentile input")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered) / 100) - 1)]


def summarize(records):
    if not records:
        return {"requests": 0, "timings_ms": {}}
    return {
        "requests": len(records),
        "timings_ms": {
            key: {
                "mean": statistics.mean(row[key] for row in records),
                "p50": percentile([row[key] for row in records], 50),
                "p95": percentile([row[key] for row in records], 95),
                "p99": percentile([row[key] for row in records], 99),
            }
            for key in records[0]
        },
    }


def emit(payload, json_mode=False):
    if json_mode:
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def validate_response(response, k, n, rerank):
    hits = response.results
    if not 0 <= response.candidate_count <= k:
        raise RuntimeError("Unexpected candidate count")
    if len(hits) != min(n, response.candidate_count):
        raise RuntimeError("Unexpected result count")
    if len({h.passage_id for h in hits}) != len(hits):
        raise RuntimeError("Duplicate result IDs")
    for rank, hit in enumerate(hits, 1):
        if (hit.rank != rank or not 1 <= hit.retrieval_rank <= response.candidate_count
            or not hit.text.strip() or not math.isfinite(hit.retrieval_score)):
            raise RuntimeError("Invalid result")
        if rerank:
            if hit.rerank_score is None or not math.isfinite(hit.rerank_score):
                raise RuntimeError("Invalid rerank score")
        elif hit.rerank_score is not None or hit.rank != hit.retrieval_rank:
            raise RuntimeError("Invalid dense-only result")
    scores = [h.rerank_score if rerank else h.retrieval_score for h in hits]
    if any(a < b for a, b in zip(scores, scores[1:])):
        raise RuntimeError("Results not descending")
    if any(not math.isfinite(v) or v < 0 for v in response.timings_ms.values()):
        raise RuntimeError("Invalid timing")


def run_query(engine, query, k, n, rerank, args):
    for _ in range(args.warmup):
        engine.search(query, retrieve_top_k=k, rerank_top_k=n, use_reranker=rerank)
    records = []
    for i in range(args.repeat):
        response = engine.search(query, retrieve_top_k=k, rerank_top_k=n, use_reranker=rerank)
        validate_response(response, k, n, rerank)
        records.append(response.timings_ms)
        payload = response.to_dict()
        if not args.json:
            for hit in payload["results"]:
                hit["text"] = hit["text"][:args.text_limit]
        emit({"type": "search", "iteration": i + 1, **payload}, args.json)
    return records


def main(argv=None):
    args = parse_args(argv)
    from app.config import load_settings
    settings = load_settings()
    k = args.retrieve_top_k if args.retrieve_top_k is not None else settings.retrieve_top_k
    n = args.top_k if args.top_k is not None else min(settings.rerank_top_k, k)
    if n > k:
        raise SystemExit("top-k must not exceed retrieve-top-k")
    from app.retrieval.engine import RetrievalEngine
    print("Loading retrieval engine...", file=sys.stderr, flush=True)
    start = perf_counter()
    engine = RetrievalEngine(settings)
    emit({"type": "startup", "seconds": perf_counter() - start,
          "model_info": engine.model_info()}, args.json)
    records = []
    if args.interactive:
        print("Enter a query; :quit or EOF exits.", file=sys.stderr)
        while True:
            try:
                print("query> ", end="", file=sys.stderr, flush=True)
                query = input().strip()
            except (EOFError, KeyboardInterrupt):
                break
            if query == ":quit":
                break
            if query:
                records.extend(run_query(engine, query, k, n, not args.no_rerank, args))
    else:
        for query in args.query:
            records.extend(run_query(engine, query, k, n, not args.no_rerank, args))
    emit({"type": "summary", **summarize(records)}, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
