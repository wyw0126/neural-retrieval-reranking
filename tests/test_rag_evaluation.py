import json
from pathlib import Path
import pytest
from scripts.evaluation.prepare_rag_eval import build_cases
from scripts.evaluation.evaluate_rag import load_cases
from scripts.evaluation.summarize_rag import summarize, percentile, review_items, human_metrics, row_id


def test_dataset(tmp_path):
    q=tmp_path/"queries"; r=tmp_path/"qrels"
    q.write_text("1\\tFirst?\\n2\\tSecond?\\n".replace("\\t","\t").replace("\\n","\n"))
    r.write_text("1 0 42 1\n2 0 43 0\n")
    a=build_cases(q,r,1,42,1)
    assert a==build_cases(q,r,1,42,1)
    assert len(a)==2 and next(c for c in a if c["expected_answerable"])["gold_passage_ids"]==[42]
    p=tmp_path/"cases.jsonl"
    p.write_text("\n".join(json.dumps(c) for c in a))
    cases,sha=load_cases(p)
    assert len(cases)==2 and len(sha)==64


def row(group="R1",status="ok",refusal=False,gold=(42,),answerable=True):
    case={"case_id":"a" if answerable else "b","query":"Question?","expected_answerable":answerable,
          "gold_passage_ids":list(gold),"reference_answer":None}
    r={"group":group,"repeat":0,"status":status,"case":case,"client_ms":10.}
    if status=="ok":
        r["result"]={"answer":{"answer":"answer","citations":[] if refusal else [1],"insufficient_evidence":refusal},
            "evidence":[{"passage_id":42}],"cited_evidence":[] if refusal else [{"passage_id":42}],
            "usage":{"total_tokens":20},"attempts":1,"judge_usage":None,"judge_attempts":0,
            "timings_ms":{"total":9.},"citation_report":{"semantic_status":"not_run"}}
    return r


def test_metrics():
    rows=[row(),row(status="error"),row(refusal=True,gold=(),answerable=False)]
    m=summarize(rows)
    assert m["success_rate"]==pytest.approx(2/3)
    assert m["proxy_abstention_accuracy_all"]==pytest.approx(2/3)
    assert m["evidence_recall_qrels"]==1
    assert m["mean_known_total_tokens"]==20
    assert m["synthetic_refusal_rate_all"]==1


def test_missing_usage_and_empty():
    r=row();r["result"]["judge_attempts"]=1
    assert summarize([r])["mean_known_total_tokens"] is None
    assert summarize([])["success_rate"] is None
    assert percentile([1,2,3,4],.95)==4


def test_independent_review():
    rows=[row(),row(refusal=True,gold=(),answerable=False)]
    items=review_items(rows)
    assert all("group" not in i and i["faithful"] is None for i in items)
    review={"review_id":row_id(rows[0]),"faithful":1,"citation_precision":.5}
    m=human_metrics(rows,[review])
    assert m["R1"]["faithful"]=={"mean":1.,"reviewed":1,"eligible":1}
    assert m["R1"]["citation_precision"]["mean"]==.5
    assert m["R1"]["abstention_correct"]["eligible"]==2
