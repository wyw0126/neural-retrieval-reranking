"""Run RAG ablations through one live HTTP service; save every request outcome."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.rag import RAGResponse

GROUPS = {False: ("R1","R2"), True: ("R3","R4")}


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1,max_length=2048)
    gold_passage_ids: list[int]
    expected_answerable: bool
    label_source: str
    reference_answer: str | None = None


def load_cases(path):
    raw = Path(path).read_bytes()
    cases = [Case.model_validate_json(line) for line in raw.splitlines() if line.strip()]
    if not cases or len({c.case_id for c in cases}) != len(cases):
        raise ValueError("Empty dataset or duplicate case IDs")
    if any(not c.query.strip() or any(i<0 for i in c.gold_passage_ids)
           or len(c.gold_passage_ids)!=len(set(c.gold_passage_ids)) for c in cases):
        raise ValueError("Invalid case content")
    return cases, hashlib.sha256(raw).hexdigest()


def get_json(client,path):
    response=client.get(path)
    response.raise_for_status()
    return response.json()


def run_one(client,case,group,k,n,repeat):
    started=perf_counter()
    row={"case":case.model_dump(), "group":group, "repeat":repeat}
    try:
        response=client.post("/rag",json={"query":case.query,"retrieve_top_k":k,
                             "rerank_top_k":n,"use_reranker":group in ("R2","R4")})
        row["http_status"]=response.status_code
        row["request_id"]=response.headers.get("x-request-id")
        if response.status_code != 200:
            try:
                body=response.json()
                row["error_code"]=body.get("code","HTTP_ERROR") if isinstance(body,dict) else "HTTP_ERROR"
            except ValueError:
                row["error_code"]="HTTP_ERROR"
            row["status"]="error"
        else:
            result=RAGResponse.model_validate(response.json())
            if result.query != case.query.strip():
                raise ValueError("Query mismatch")
            sem=result.citation_report.semantic_status
            if (group in ("R1","R2") and sem!="not_run") or (
                    group in ("R3","R4") and not result.answer.insufficient_evidence and sem!="supported"):
                raise ValueError("Judge mode mismatch")
            row.update(status="ok",result=result.model_dump(mode="json"))
    except (httpx.HTTPError, ValueError, ValidationError) as exc:
        row.update(status="error",error_code=type(exc).__name__)
    row["client_ms"]=(perf_counter()-started)*1000
    return row


def run(client,cases,sha,out,judge,k,n,repeat,seed):
    health=get_json(client,"/rag/health")
    if health.get("ready") is not True or health.get("semantic_validation") is not judge:
        raise ValueError("Live RAG Judge setting does not match --judge; restart server")
    models=get_json(client,"/models")
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    manifest={"dataset_sha256":sha,"case_ids":[c.case_id for c in cases],"judge":judge,
              "groups":list(GROUPS[judge]),"retrieve_top_k":k,"top_k":n,"repeat":repeat,
              "seed":seed,"models":models,"started_utc":datetime.now(timezone.utc).isoformat(),
              "expected_requests":len(cases)*2*repeat}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    rng=random.Random(seed)
    with (out/"rows.jsonl").open("x",encoding="utf-8") as f:
        for rep in range(repeat):
            ordered=list(cases)
            rng.shuffle(ordered)
            for case in ordered:
                groups=list(GROUPS[judge])
                rng.shuffle(groups)
                for group in groups:
                    row=run_one(client,case,group,k,n,rep)
                    f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
                    f.flush()
                    print(group,case.case_id,row["status"],f'{row["client_ms"]:.1f} ms',flush=True)
    (out/"complete.json").write_text(json.dumps({"requests":manifest["expected_requests"]}),encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--judge",choices=["off","on"],required=True)
    p.add_argument("--base-url",default="http://127.0.0.1:8000")
    p.add_argument("--retrieve-top-k",type=int,default=100)
    p.add_argument("--top-k",type=int,default=10)
    p.add_argument("--repeat",type=int,default=1)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--timeout",type=float,default=150.)
    a=p.parse_args()
    if not 1<=a.top_k<=a.retrieve_top_k<=200 or a.repeat<1 or a.timeout<=0:
        p.error("Invalid top-k, repeat or timeout")
    cases,sha=load_cases(a.dataset)
    with httpx.Client(base_url=a.base_url,timeout=a.timeout,trust_env=False) as client:
        run(client,cases,sha,a.output,a.judge=="on",a.retrieve_top_k,a.top_k,a.repeat,a.seed)


if __name__=="__main__":
    main()
