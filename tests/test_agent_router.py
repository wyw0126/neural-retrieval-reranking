import asyncio
import json
import pytest

from app.agent.graph import AgentEngine
from app.agent.router import RouteDecision, social_reply
from app.config import Settings
from app.llm.schemas import ChatResult, TokenUsage
from app.rag.schemas import RAGAnswer


class Search:
    def __init__(self):
        self.calls=[]
    def search(self,query,*,retrieve_top_k,rerank_top_k,use_reranker):
        self.calls.append(use_reranker)
        return dict(query=query,requested_retrieve_top_k=retrieve_top_k,
            requested_result_top_k=rerank_top_k,use_reranker=use_reranker,
            candidate_count=1,results=[dict(passage_id=42,text="ML learns patterns.",
            rank=1,retrieval_rank=1,retrieval_score=.5,rerank_score=2. if use_reranker else None)],
            timings_ms={"total":1.})


class LLM:
    def __init__(self,route):
        self.route=route
        self.calls=[]
    async def chat_json(self,messages,schema,*,max_message_bytes=None):
        self.calls.append(schema)
        if schema is RouteDecision:
            data=json.loads(messages[-1].content)
            assert len(data["available_tools"])==2
            parsed=RouteDecision(route=self.route)
        else:
            parsed=RAGAnswer(answer="ML learns patterns [1].",citations=[1],insufficient_evidence=False)
        return parsed,ChatResult(content=parsed.model_dump_json(),model="fake",elapsed_ms=1.,
            attempts=1,usage=TokenUsage(prompt_tokens=10,completion_tokens=5,total_tokens=15))


def run(route,query="ML?",auto=True,**kwargs):
    s,l=Search(),LLM(route)
    a=AgentEngine(Settings(_env_file=None,**kwargs),s,l)
    result=asyncio.run(a.answer(query,auto_route=auto))
    return result,s,l


@pytest.mark.parametrize("route,rerank",[("dense_search",False),("reranked_search",True)])
def test_search_routes(route,rerank):
    r,s,l=run(route)
    assert s.calls==[rerank]
    assert r["route"]==route and r["tool_calls"][0]["name"]==route
    assert r["router_usage"]["total_tokens"]==15 and r["usage"]["total_tokens"]==15
    assert r["trace"][0]["node"]=="route"
    assert r["answer_mode"]=="grounded_rag"


def test_direct():
    r,s,l=run("direct","Hello!")
    assert not s.calls and l.calls==[RouteDecision]
    assert r["answer_mode"]=="social" and not r["answer"]["insufficient_evidence"]
    assert r["citation_report"]["structural_status"]=="not_applicable"
    assert not r["tool_calls"] and r["attempts"]==0


def test_direct_guard():
    r,s,l=run("direct","Hello! Explain ML.")
    assert r["route_source"]=="direct_guard_fallback"
    assert s.calls==[True] and r["route"]=="reranked_search"


def test_unsupported():
    r,s,l=run("unsupported","Send an email")
    assert not s.calls and l.calls==[RouteDecision]
    assert r["refusal_reason"]=="unsupported_task" and r["answer"]["insufficient_evidence"]


def test_fixed():
    r,s,l=run("unsupported",auto=False)
    assert s.calls==[True] and l.calls==[RAGAnswer]
    assert r["route_source"]=="fixed" and r["router_attempts"]==0


def test_config_default():
    r,s,l=run("dense_search",auto=None,agent_router_enabled=True)
    assert r["route"]=="dense_search" and s.calls==[False]


def test_invalid_route():
    with pytest.raises(ValueError):
        run("shell")


def test_social_exact():
    assert social_reply("你好！")
    assert social_reply("你好，请解释机器学习") is None
