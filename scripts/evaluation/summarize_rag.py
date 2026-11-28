"""Summarize operational and qrels-proxy metrics; export blind independent review."""
import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean


def ratio(a,b):
    return a/b if b else None


def percentile(values,p):
    return sorted(values)[max(0,math.ceil(p*len(values))-1)] if values else None


def row_id(row):
    return hashlib.sha256(json.dumps([row["group"],row["case"]["case_id"],row["repeat"]]).encode()).hexdigest()[:24]


def summarize(rows):
    ok=[r for r in rows if r["status"]=="ok"]
    answered=[r for r in ok if not r["result"]["answer"]["insufficient_evidence"]]
    positives=[r for r in ok if r["case"]["gold_passage_ids"]]
    hit=[]; recall=[]; cited_hit=[]
    for r in positives:
        gold=set(r["case"]["gold_passage_ids"])
        evidence={e["passage_id"] for e in r["result"]["evidence"]}
        cited={e["passage_id"] for e in r["result"]["cited_evidence"]}
        hit.append(bool(gold & evidence))
        recall.append(len(gold & evidence)/len(gold))
        cited_hit.append(bool(gold & cited))
    expected_positive=[r for r in rows if r["case"]["expected_answerable"]]
    expected_negative=[r for r in rows if not r["case"]["expected_answerable"]]
    false_refusals=sum(r["status"]=="ok" and r["result"]["answer"]["insufficient_evidence"] for r in expected_positive)
    correct_refusals=sum(r["status"]=="ok" and r["result"]["answer"]["insufficient_evidence"] for r in expected_negative)
    correct_proxy=sum(r["case"]["expected_answerable"] != r["result"]["answer"]["insufficient_evidence"] for r in ok)
    token_totals=[]
    for r in ok:
        d=r["result"]
        parts=[]
        for usage,attempts in ((d["usage"],d["attempts"]),(d["judge_usage"],d["judge_attempts"])):
            if attempts==0:
                parts.append(0)
            elif usage is not None:
                parts.append(usage["total_tokens"])
        if len(parts)==2:
            token_totals.append(sum(parts))
    times=[r["client_ms"] for r in ok]
    stages={}
    for stage in sorted({k for r in ok for k in r["result"]["timings_ms"]}):
        values=[r["result"]["timings_ms"][stage] for r in ok if stage in r["result"]["timings_ms"]]
        stages[stage]={"mean":mean(values),"p95":percentile(values,.95)}
    return {"requests":len(rows),"successes":len(ok),"success_rate":ratio(len(ok),len(rows)),
            "errors":dict(Counter(r.get("error_code","UNKNOWN") for r in rows if r["status"]!="ok")),
            "answer_rate_all":ratio(len(answered),len(rows)),
            "qrels_positive_successes":len(positives),
            "evidence_hit_qrels":mean(hit) if hit else None,
            "evidence_recall_qrels":mean(recall) if recall else None,
            "cited_qrels_hit":mean(cited_hit) if cited_hit else None,
            "expected_answerable_count":len(expected_positive),
            "expected_unanswerable_count":len(expected_negative),
            "proxy_false_refusal_rate_all":ratio(false_refusals,len(expected_positive)),
            "synthetic_refusal_rate_all":ratio(correct_refusals,len(expected_negative)),
            "proxy_abstention_accuracy_all":ratio(correct_proxy,len(rows)),
            "client_ms":{"mean":mean(times) if times else None,"p50":percentile(times,.5),"p95":percentile(times,.95)},
            "stages_ms":stages,"known_usage_requests":len(token_totals),
            "mean_known_total_tokens":mean(token_totals) if token_totals else None,
            "known_total_tokens":sum(token_totals),
            "semantic_status":dict(Counter(r["result"]["citation_report"]["semantic_status"] for r in ok))}


def load_runs(paths):
    manifests=[]; rows=[]
    for path in paths:
        path=Path(path)
        m=json.loads((path/"manifest.json").read_text(encoding="utf-8"))
        if not (path/"complete.json").exists():
            raise ValueError("Incomplete run")
        data=[json.loads(line) for line in (path/"rows.jsonl").read_text(encoding="utf-8").splitlines()]
        expected={(group,cid,rep) for group in m["groups"] for cid in m["case_ids"] for rep in range(m["repeat"])}
        keys=[(r["group"],r["case"]["case_id"],r["repeat"]) for r in data]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:
            raise ValueError("Missing or duplicate requests")
        manifests.append(m); rows.extend(data)
    fields=("dataset_sha256","case_ids","retrieve_top_k","top_k","repeat","seed","models")
    if len(manifests)!=2 or {m["judge"] for m in manifests}!={True,False}:
        raise ValueError("Need one Judge-off and one Judge-on run")
    if any(manifests[0][k]!=manifests[1][k] for k in fields):
        raise ValueError("Run settings differ")
    observed={r["result"]["model"] for r in rows if r["status"]=="ok" and r["result"]["model"] is not None}
    if len(observed)>1:
        raise ValueError("Observed generation models differ")
    return manifests,rows


def review_items(rows):
    result=[]
    for r in rows:
        if r["status"]!="ok":
            continue
        d=r["result"]
        result.append({"review_id":row_id(r),"query":r["case"]["query"],
                       "reference_answer":r["case"].get("reference_answer"),
                       "answer":d["answer"],"evidence":d["evidence"],
                       "cited_evidence":d["cited_evidence"],
                       "answer_correct":None,"faithful":None,"citation_precision":None,
                       "citation_recall":None,"abstention_correct":None})
    random.Random(42).shuffle(result)
    return result


def human_metrics(rows, reviews):
    by_id={r["review_id"]:r for r in reviews}
    if len(by_id)!=len(reviews):
        raise ValueError("Duplicate review ID")
    allowed={row_id(r) for r in rows if r["status"]=="ok"}
    if set(by_id)-allowed:
        raise ValueError("Unknown review ID")
    output={}
    for group in ("R1","R2","R3","R4"):
        group_rows=[r for r in rows if r["group"]==group and r["status"]=="ok"]
        result={}
        for field in ("answer_correct","faithful","citation_precision","citation_recall","abstention_correct"):
            values=[]
            eligible=[r for r in group_rows if field=="abstention_correct" or
                      not r["result"]["answer"]["insufficient_evidence"]]
            for r in eligible:
                value=by_id.get(row_id(r),{}).get(field)
                if value is None:
                    continue
                if type(value) not in (int,float,bool) or not 0<=value<=1:
                    raise ValueError("Review scores must be in [0,1]")
                values.append(float(value))
            result[field]={"mean":mean(values) if values else None,"reviewed":len(values),"eligible":len(eligible)}
        output[group]=result
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",nargs=2,required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--reviews",type=Path)
    a=p.parse_args()
    manifests,rows=load_runs(a.runs)
    result={"manifests":manifests,"groups":{g:summarize([r for r in rows if r["group"]==g]) for g in ("R1","R2","R3","R4")},
            "limitations":["qrels are incomplete relevance labels, not answer or entailment labels",
                           "synthetic expected abstention is a proxy, not verified corpus absence",
                           "latency and token averages use successful requests; failed request usage is unknown",
                           "Judge-on/off runs generate independently and may differ stochastically"]}
    if a.reviews:
        result["independent_review"]=human_metrics(rows,[json.loads(l) for l in a.reviews.read_text(encoding="utf-8").splitlines() if l.strip()])
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    with (a.output/"review.jsonl").open("x",encoding="utf-8") as f:
        for item in review_items(rows):
            f.write(json.dumps(item,ensure_ascii=False)+"\n")
    print(json.dumps(result["groups"],indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
