import asyncio
import json
import pytest
from app.agent.graph import AgentEngine
from app.agent.rewrite import RewriteDecision
from app.agent.router import RouteDecision
from app.agent.fusion import fuse_results
from app.config import Settings
from app.llm.schemas import ChatResult,TokenUsage
from app.rag.schemas import RAGAnswer
from app.schemas import SearchResponse

def response(query,k,n,rerank,ids):
    return SearchResponse(query=query,requested_retrieve_top_k=k,requested_result_top_k=n,
        use_reranker=rerank,candidate_count=len(ids),timings_ms={"total":1.},
        results=[dict(passage_id=pid,text=f"Evidence passage {pid} contains relevant information.",
                      retrieval_score=.5,rerank_score=2. if rerank else None,
                      rank=i,retrieval_rank=i) for i,pid in enumerate(ids,1)])

class Search:
    def __init__(self): self.calls=[]
    def search(self,query,*,retrieve_top_k,rerank_top_k,use_reranker):
        self.calls.append(query)
        ids=[42,43] if len(self.calls)==1 else [44,43]
        return response(query,retrieve_top_k,rerank_top_k,use_reranker,ids[:rerank_top_k])

class LLM:
    def __init__(self,queries,route="reranked_search"):
        self.queries=queries
        self.route=route
        self.calls=[]
        self.questions=[]
    async def chat_json(self,messages,schema,*,max_message_bytes=None):
        self.calls.append(schema)
        if schema is RewriteDecision:
            parsed=schema(queries=self.queries)
        elif schema is RouteDecision:
            parsed=schema(route=self.route)
        else:
            self.questions.append(json.loads(messages[-1].content)["question"])
            parsed=RAGAnswer(answer="Evidence [1].",citations=[1],insufficient_evidence=False)
        return parsed,ChatResult(content=parsed.model_dump_json(),model="fake",
            elapsed_ms=1.,attempts=1,usage=TokenUsage(prompt_tokens=10,completion_tokens=5,total_tokens=15))

def run(queries,question="什么是机器学习？",rewrite=True,auto=False,route="reranked_search"):
    s,l=Search(),LLM(queries,route)
    a=AgentEngine(Settings(_env_file=None,agent_router_enabled=False,agent_rewrite_enabled=False,
                           rag_semantic_validation=False),s,l)
    r=asyncio.run(a.answer(question,rewrite=rewrite,auto_route=auto))
    return r,s,l

def test_translation_original_preserved():
    r,s,l=run(["what is machine learning"])
    assert s.calls==["what is machine learning"]
    assert l.questions==["什么是机器学习？"]
    assert r["search_queries"]==s.calls
    assert r["rewrite_usage"]["total_tokens"]==15
    assert r["trace"][0]["node"]=="rewrite"

def test_decomposition():
    r,s,l=run(["query one","query two"])
    assert s.calls==["query one","query two"]
    assert len(r["tool_calls"])==2 and len(r["retrieval_runs"])==2
    assert r["candidate_count"]==4
    assert r["retrieval_timings_ms"]["total"]==2
    assert r["evidence"][0]["passage_id"]==43
    assert r["fusion_scores"]["43"]>r["fusion_scores"]["42"]

def test_duplicate_queries():
    r,s,l=run([" Machine learning ","machine  learning","ML"])
    assert s.calls==["Machine learning","ML"]

def test_disabled():
    r,s,l=run(["not used"],rewrite=False)
    assert s.calls==["什么是机器学习？"] and l.calls==[RAGAnswer]
    assert r["rewrite_attempts"]==0

@pytest.mark.parametrize("route,query",[("direct","Hello"),("unsupported","Send email")])
def test_nonsearch_skips_rewrite(route,query):
    r,s,l=run(["not used"],question=query,auto=True,route=route)
    assert not s.calls and l.calls==[RouteDecision] and r["rewrite_attempts"]==0

@pytest.mark.parametrize("queries",[[],[" "],["a","b","c","d"],[1],["x"*2049]])
def test_invalid_rewrite(queries):
    with pytest.raises(ValueError):
        run(queries)

def test_fusion_deterministic():
    a=response("a",10,2,True,[10,20])
    b=response("b",10,2,True,[30,20])
    hits,scores=fuse_results([a,b],2)
    assert [h.passage_id for h in hits]==[20,10]
    assert [h.rank for h in hits]==[1,2]
    assert hits[0].retrieval_rank==2
    assert scores[20]==pytest.approx(2/62)
