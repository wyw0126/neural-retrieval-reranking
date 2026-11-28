import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.llm.exceptions import LLMRequestError
from app.llm.schemas import ChatResult, TokenUsage
from app.rag.chain import RAGCitationError, RAGEngine
from app.rag.prompts import PromptBudgetError
from app.rag.schemas import RAGAnswer


def settings(**kwargs):
    return Settings(_env_file=None, **kwargs)


def hit(pid=42, text="Machine learning learns patterns from data.", rank=1):
    return SimpleNamespace(passage_id=pid, text=text, rank=rank, retrieval_rank=rank,
                           retrieval_score=.5, rerank_score=2.)


class FakeSearch:
    def __init__(self, hits=None, error=None):
        self.hits = [hit()] if hits is None else hits
        self.calls = []
        self.error = error
        self.thread = None

    def search(self, query, **kwargs):
        self.thread = threading.get_ident()
        self.calls.append((query, kwargs))
        if self.error:
            raise self.error
        return SimpleNamespace(results=self.hits, candidate_count=len(self.hits),
                               timings_ms={"retrieval": 2., "total": 3.})


class FakeLLM:
    def __init__(self, answer=None, error=None, usage=True):
        self.answer = answer or RAGAnswer(answer="ML learns patterns [1].",citations=[1],insufficient_evidence=False)
        self.error = error
        self.calls = []
        self.usage = usage

    async def chat_json(self, messages, schema, *, max_message_bytes=None):
        self.calls.append((messages,schema,max_message_bytes))
        if self.error:
            raise self.error
        return self.answer, ChatResult(content=self.answer.model_dump_json(),model="fake",
            elapsed_ms=5.,attempts=1,finish_reason="stop",
            usage=TokenUsage(prompt_tokens=10,completion_tokens=5,total_tokens=15) if self.usage else None)


def test_success():
    s=settings()
    search, llm = FakeSearch(), FakeLLM()
    result=asyncio.run(RAGEngine(s,search,llm).answer(" ML? "))
    assert result.query=="ML?"
    assert result.answer.citations==[1] and result.cited_evidence[0].passage_id==42
    assert result.evidence==result.cited_evidence
    assert result.usage.total_tokens==15 and result.attempts==1 and result.model=="fake"
    assert result.refusal_reason is None and result.retrieved_count==1
    assert result.candidate_count==1 and result.prompt_bytes>0
    assert search.calls==[("ML?",{"retrieve_top_k":100,"rerank_top_k":10,"use_reranker":True})]
    assert llm.calls[0][1] is RAGAnswer and llm.calls[0][2]==s.rag_prompt_budget_bytes
    assert result.retrieval_timings_ms["total"]==3.
    assert all(v>=0 for v in result.timings_ms.values())
    assert json.loads(json.dumps(result.to_dict()))["answer"]["citations"]==[1]


@pytest.mark.parametrize("hits", [[], [hit(text=" ")]])
def test_no_evidence_skips_llm(hits):
    llm=FakeLLM()
    result=asyncio.run(RAGEngine(settings(),FakeSearch(hits),llm).answer("ML?"))
    assert result.answer.insufficient_evidence and not result.answer.citations
    assert not llm.calls and result.attempts==0 and result.model is None and result.usage is None
    assert result.refusal_reason=="no_evidence" and result.prompt_bytes==0
    assert result.timings_ms["generation"]==0 and result.cited_evidence==()


def test_model_refusal():
    llm=FakeLLM(RAGAnswer(answer="Insufficient evidence.",citations=[],insufficient_evidence=True))
    result=asyncio.run(RAGEngine(settings(),FakeSearch(),llm).answer("ML?"))
    assert result.refusal_reason=="model_insufficient_evidence" and result.attempts==1
    assert len(result.evidence)==1 and not result.cited_evidence


def test_unknown_citation_fails_closed():
    llm=FakeLLM(RAGAnswer(answer="Unknown [9].",citations=[9],insufficient_evidence=False))
    with pytest.raises(RAGCitationError):
        asyncio.run(RAGEngine(settings(),FakeSearch(),llm).answer("ML?"))


def test_citation_order():
    search=FakeSearch([hit(41,rank=1),hit(42,text="A separate fact about algorithms.",rank=2)])
    llm=FakeLLM(RAGAnswer(answer="Facts [2] [1].",citations=[2,1],insufficient_evidence=False))
    result=asyncio.run(RAGEngine(settings(),search,llm).answer("ML?"))
    assert [e.passage_id for e in result.cited_evidence]==[42,41]


def test_offloads_sync_search():
    search=FakeSearch()
    async def run():
        main_thread=threading.get_ident()
        await RAGEngine(settings(),search,FakeLLM()).answer("ML?")
        assert search.thread!=main_thread
    asyncio.run(run())


def test_top_k_defaults_clamp_and_dense_only():
    search=FakeSearch()
    asyncio.run(RAGEngine(settings(),search,FakeLLM()).answer("ML?",retrieve_top_k=3,use_reranker=False))
    assert search.calls[0][1]=={"retrieve_top_k":3,"rerank_top_k":3,"use_reranker":False}


@pytest.mark.parametrize("query,kwargs",[(" ",{}),("x"*2049,{}),("Q",{"retrieve_top_k":True}),
    ("Q",{"retrieve_top_k":0}),("Q",{"retrieve_top_k":3,"rerank_top_k":4}),
    ("Q",{"rerank_top_k":False}),("Q",{"use_reranker":1})])
def test_invalid_inputs(query,kwargs):
    search,llm=FakeSearch(),FakeLLM()
    with pytest.raises(ValueError):
        asyncio.run(RAGEngine(settings(),search,llm).answer(query,**kwargs))
    assert not search.calls and not llm.calls


def test_retrieval_failure_propagates():
    llm=FakeLLM()
    with pytest.raises(RuntimeError,match="retrieval failed"):
        asyncio.run(RAGEngine(settings(),FakeSearch(error=RuntimeError("retrieval failed")),llm).answer("ML?"))
    assert not llm.calls


def test_llm_failure_propagates():
    with pytest.raises(LLMRequestError):
        asyncio.run(RAGEngine(settings(),FakeSearch(),FakeLLM(error=LLMRequestError(503))).answer("ML?"))


def test_prompt_budget_failure_skips_llm():
    llm=FakeLLM()
    with pytest.raises(PromptBudgetError):
        asyncio.run(RAGEngine(settings(rag_prompt_budget_bytes=1024),FakeSearch(),llm).answer("ML?"))
    assert not llm.calls


def test_missing_usage():
    result=asyncio.run(RAGEngine(settings(),FakeSearch(),FakeLLM(usage=False)).answer("ML?"))
    assert result.usage is None and result.to_dict()["usage"] is None
