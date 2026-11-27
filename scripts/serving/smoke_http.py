"""Validate a running HTTP service and save serial latency measurements."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.client import RetrievalClient
from app.schemas import SearchResponse, RerankResponse, HealthResponse, ModelInfoResponse, ErrorResponse
from scripts.serving.smoke_search import summarize, validate_response

QUERIES = ["What causes diabetes?", "How does solar energy work?",
           "What is machine learning?", "How do vaccines work?",
           "What causes global warming?"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(client, warmup=3, repeat=20):
    health, _ = client.request("GET", "/health")
    require(HealthResponse.model_validate(health).ready, "Engine not ready")
    models, _ = client.request("GET", "/models")
    info = ModelInfoResponse.model_validate(models)
    dense, _ = client.request("POST", "/retrieve", json={"query":QUERIES[0],"top_k":5})
    result = SearchResponse.model_validate(dense)
    validate_response(result, 5, 5, False)
    require(result.candidate_count == len(result.results) == 5, "Retrieve count mismatch")
    docs = [{"passage_id":1,"text":"Machine learning learns patterns from data."},
            {"passage_id":2,"text":"Paris is the capital of France."}]
    ranked, _ = client.request("POST", "/rerank", json={"query":QUERIES[2],"documents":docs,"top_k":2})
    result = RerankResponse.model_validate(ranked)
    require(result.candidate_count == len(result.results) == 2, "Rerank count mismatch")
    require({h.passage_id for h in result.results} == {1,2}, "Rerank identity mismatch")
    require(result.results[0].rerank_score >= result.results[1].rerank_score, "Rerank order mismatch")
    for method,path,status,kwargs in [
        ("POST","/search",422,{"json":{"query":"q","retrieve_top_k":5,"rerank_top_k":10}}),
        ("POST","/retrieve",422,{"json":{"query":"q","top_k":"5"}}),
        ("GET","/unknown",404,{}),("POST","/health",405,{})]:
        body, timing = client.request(method,path,expected=status,**kwargs)
        err = ErrorResponse.model_validate(body)
        require(err.request_id == timing["request_id"], "Error request ID mismatch")
    summaries = {}
    for mode in (True, False):
        records = []
        for query in QUERIES:
            for i in range(warmup + repeat):
                data, timing = client.search(query,100,10,mode)
                response = SearchResponse.model_validate(data)
                validate_response(response,100,10,mode)
                require(response.candidate_count == 100 and len(response.results)==10, "Search count mismatch")
                require(response.use_reranker == mode, "Search mode mismatch")
                if i >= warmup:
                    row = dict(client=timing["client_ms"], http_server=timing["server_ms"],
                               **response.timings_ms)
                    require(all(math.isfinite(v) and v >= 0 for v in row.values()), "Invalid timing")
                    records.append(row)
        summaries["two_stage" if mode else "dense_only"] = summarize(records)
    return {"status":"passed","model_info":info.model_dump(), "queries":QUERIES,
            "warmup_per_query":warmup,"repeat_per_query":repeat,"summaries":summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--warmup",type=int,default=3)
    parser.add_argument("--repeat",type=int,default=20)
    parser.add_argument("--output",type=Path,default=Path("reports/serving/m2_http_baseline.json"))
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("invalid warmup/repeat")
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    with RetrievalClient(args.base_url) as client:
        report = run(client,args.warmup,args.repeat)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as out:
        json.dump(report,out,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))


if __name__ == "__main__":
    main()
