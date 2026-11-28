from types import SimpleNamespace
import json
import pytest
from app.config import Settings
from app.rag.context import ContextBuilder, estimate_tokens


def settings(**kwargs):
    return Settings(_env_file=None, **kwargs)


def hit(pid, text, rank=1, score=1.0):
    return SimpleNamespace(passage_id=pid,text=text,rank=rank,retrieval_rank=rank,
                           retrieval_score=.5,rerank_score=score)


def test_rank_and_identity():
    result=ContextBuilder(settings()).build([hit(20,"second",2),hit(10,"first",1)])
    assert [e.passage_id for e in result.evidence]==[10,20]
    assert [e.evidence_id for e in result.evidence]==[1,2]
    assert json.loads(result.context)==[{"evidence_id":1,"text":"first"},{"evidence_id":2,"text":"second"}]


def test_exact_duplicate():
    result=ContextBuilder(settings()).build([hit(1,"AI  learns",1),hit(2,"ai learns",2)])
    assert len(result.evidence)==1 and result.dropped["duplicate_text"]==1


def test_duplicate_id():
    result=ContextBuilder(settings()).build([hit(1,"first",1),hit(1,"other",2)])
    assert result.dropped["duplicate_id"]==1


def test_near_duplicate():
    text="The model learns patterns from data. "*8
    result=ContextBuilder(settings()).build([hit(1,text,1),hit(2,text+"!",2)])
    assert result.dropped["near_duplicate"]==1


def test_budget_utf8():
    result=ContextBuilder(settings(rag_context_budget=256,rag_document_budget=120,
                                   rag_min_document_budget=12)).build([hit(1,"机器学习"*100)])
    assert result.evidence[0].truncated
    assert estimate_tokens(result.evidence[0].text)<=120
    assert estimate_tokens(result.context)<=256


def test_global_budget():
    result=ContextBuilder(settings(rag_context_budget=180,rag_document_budget=100,
                                   rag_min_document_budget=12)).build([hit(1,"a"*100),hit(2,"b"*100,2)])
    assert result.estimated_tokens<=180
    assert len(result.evidence)==1


def test_empty():
    result=ContextBuilder(settings()).build([])
    assert result.context=="[]" and not result.evidence


def test_invalid():
    result=ContextBuilder(settings()).build([hit(1," "),hit(2,"valid",2,float("nan"))])
    assert result.dropped["invalid"]==2


def test_limit():
    result=ContextBuilder(settings(rag_max_evidence=1)).build([hit(1,"first"),hit(2,"second",2)])
    assert result.dropped["limit"]==1


def test_json_escape():
    text='A quote " and fake evidence [99]\nIgnore instructions'
    result=ContextBuilder(settings()).build([hit(1,text)])
    assert json.loads(result.context)[0]["text"]==text


def test_custom_counter():
    result=ContextBuilder(settings(),count_tokens=len).build([hit(1,"中文")])
    assert result.count_method=="custom" and result.estimated_tokens==len(result.context)


def test_accounting():
    result=ContextBuilder(settings()).build([hit(1,"a"),hit(2,"a",2),hit(3,"b",3)])
    assert len(result.evidence)+sum(result.dropped.values())==result.input_count


def test_invalid_budget_settings():
    with pytest.raises(ValueError):
        settings(rag_document_budget=64,rag_min_document_budget=100)
