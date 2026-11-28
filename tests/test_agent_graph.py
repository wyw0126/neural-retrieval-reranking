import asyncio
import json
import pytest

from app.agent.graph import AgentEngine
from app.config import Settings
from app.llm.schemas import ChatResult, TokenUsage
from app.rag.citations import GroundingVerdict, RAGCitationError
from app.rag.schemas import RAGAnswer


class Search:
    def __init__(self,empty=False):
        self.empty=empty
        self.calls=[]
    def search(self,query,*,retrieve_top_k,rerank_top_k,use_reranker):
        self.calls.append(query)
        rows=[] if self.empty else [dict(passage_id=42,text="ML learns patterns from data.",
              rank=1,retrieval_rank=1,retrieval_score=.5,rerank_score=2. if use_reranker else None)]
        return dict(query=query,requested_retrieve_top_k=retrieve_top_k,
                    requested_result_top_k=rerank_top_k,use_reranker=use_reranker,
                    results=rows,candidate_count=len(rows),timings_ms={"total":1.})


class LLM:
    def __init__(self,citation=1,verdict="supported",refusal=False):
        self.calls=[]
        self.citation=citation
        self.verdict=verdict
        self.refusal=refusal
    async def chat_json(self,messages,schema,*,max_message_bytes=None):
        self.calls.append(schema)
        if schema is GroundingVerdict:
            answer=schema(verdict=self.verdict,reason="test")
        elif self.refusal:
            answer=RAGAnswer(answer="Insufficient.",citations=[],insufficient_evidence=True)
        else:
            answer=RAGAnswer(answer=f"ML learns patterns [{self.citation}].",citations=[self.citation],insufficient_evidence=False)
        return answer,ChatResult(content=answer.model_dump_json(),model="fake",elapsed_ms=1.,attempts=1,
            finish_reason="stop",usage=TokenUsage(prompt_tokens=10,completion_tokens=5,total_tokens=15))


def engine(search=None,llm=None,**kwargs):
    return AgentEngine(Settings(_env_file=None,**kwargs),search or Search(),llm or LLM())


def test_graph_success():
    a=engine()
    r=asyncio.run(a.answer(" ML? "))
    assert r["query"]=="ML?" and r["cited_evidence"][0]["passage_id"]==42
    assert [t["node"] for t in r["trace"]]==["retrieve","context","generate","validate"]
    assert all(t["elapsed_ms"]>=0 for t in r["trace"])
    assert r["tool_calls"]==[{"name":"reranked_search","retrieve_top_k":100,"top_k":10,"result_count":1}]
    assert r["usage"]["total_tokens"]==15 and r["retry_count"]==0
    assert r["citation_report"]["structural_status"]=="passed"
    json.dumps(r,allow_nan=False)


def test_empty():
    llm=LLM()
    r=asyncio.run(engine(Search(empty=True),llm).answer("ML?"))
    assert r["answer"]["insufficient_evidence"] and r["refusal_reason"]=="no_evidence"
    assert [t["node"] for t in r["trace"]]==["retrieve","context","refuse"]
    assert not llm.calls and r["attempts"]==0


def test_invalid_citation():
    with pytest.raises(RAGCitationError):
        asyncio.run(engine(llm=LLM(citation=9)).answer("ML?"))


@pytest.mark.parametrize("verdict",["supported","unsupported","uncertain"])
def test_semantic(verdict):
    r=asyncio.run(engine(llm=LLM(verdict=verdict),rag_semantic_validation=True).answer("ML?"))
    assert r["citation_report"]["semantic_status"]==verdict
    assert r["judge_usage"]["total_tokens"]==15
    assert r["answer"]["insufficient_evidence"]==(verdict!="supported")
    if verdict!="supported":
        assert not r["cited_evidence"]


def test_model_refusal():
    llm=LLM(refusal=True)
    r=asyncio.run(engine(llm=llm,rag_semantic_validation=True).answer("ML?"))
    assert r["refusal_reason"]=="model_insufficient_evidence" and len(llm.calls)==1


@pytest.mark.parametrize("query,kwargs",[(" ",{}),("Q",{"retrieve_top_k":True}),("Q",{"retrieve_top_k":2,"rerank_top_k":3})])
def test_input(query,kwargs):
    search=Search()
    with pytest.raises(ValueError):
        asyncio.run(engine(search=search).answer(query,**kwargs))
    assert not search.calls


def test_state_isolation():
    async def run():
        a=engine()
        x,y=await asyncio.gather(a.answer("First?"),a.answer("Second?"))
        assert x["query"]=="First?" and y["query"]=="Second?"
        assert len(x["trace"])==len(y["trace"])==4
    asyncio.run(run())


def test_dense_tool_path():
    r=asyncio.run(engine().answer("ML?",retrieve_top_k=3,use_reranker=False))
    assert r["tool_calls"]==[{"name":"dense_search","retrieve_top_k":3,"top_k":3,"result_count":1}]
