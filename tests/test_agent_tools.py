import asyncio
import threading
import pytest
from app.agent.tools import ToolRegistry
from app.config import Settings


class Backend:
    def __init__(self):
        self.calls=[]
        self.thread=None
    def search(self,query,*,retrieve_top_k,rerank_top_k,use_reranker):
        self.thread=threading.get_ident()
        self.calls.append((query,retrieve_top_k,rerank_top_k,use_reranker))
        return dict(query=query,requested_retrieve_top_k=retrieve_top_k,
                    requested_result_top_k=rerank_top_k,use_reranker=use_reranker,
                    candidate_count=1,results=[dict(passage_id=42,text="Evidence",rank=1,
                    retrieval_rank=1,retrieval_score=.5,rerank_score=2. if use_reranker else None)],
                    timings_ms={"total":1.})


def registry(backend=None):
    return ToolRegistry(Settings(_env_file=None),backend or Backend())


def test_definitions():
    specs=registry().definitions()
    assert [x["function"]["name"] for x in specs]==["dense_search","reranked_search"]
    for spec in specs:
        params=spec["function"]["parameters"]
        assert params["additionalProperties"] is False
        assert "query" in params["required"]
        assert spec["function"]["description"]
    specs[0]["function"]["name"]="mutated"
    assert registry().definitions()[0]["function"]["name"]=="dense_search"


@pytest.mark.parametrize("name,rerank",[("dense_search",False),("reranked_search",True)])
def test_dispatch(name,rerank):
    b=Backend()
    result=asyncio.run(registry(b).invoke(name,{"query":" ML? ","retrieve_top_k":20,"top_k":5}))
    assert b.calls==[("ML?",20,5,rerank)]
    assert result.results[0].passage_id==42 and result.use_reranker is rerank


def test_defaults():
    b=Backend()
    asyncio.run(registry(b).invoke("dense_search",{"query":"Q","retrieve_top_k":3}))
    assert b.calls==[("Q",3,3,False)]


@pytest.mark.parametrize("name,args",[
    ("shell",{"query":"Q"}),("dense_search",{"query":" "}),
    ("dense_search",{"query":"Q","top_k":True}),
    ("dense_search",{"query":"Q","top_k":"5"}),
    ("dense_search",{"query":"Q","top_k":0}),
    ("dense_search",{"query":"Q","retrieve_top_k":201}),
    ("reranked_search",{"query":"Q","top_k":5,"retrieve_top_k":3}),
    ("reranked_search",{"query":"Q","top_k":101}),
    ("dense_search",{"query":"Q","use_reranker":True}),
    ("dense_search",{"query":"Q","unexpected":1}),
])
def test_reject_before_execution(name,args):
    b=Backend()
    with pytest.raises(ValueError):
        asyncio.run(registry(b).invoke(name,args))
    assert not b.calls


def test_thread_offload():
    b=Backend()
    async def run():
        await registry(b).invoke("dense_search",{"query":"Q"})
        assert b.thread!=threading.get_ident()
    asyncio.run(run())


def test_inconsistent_backend():
    class Bad(Backend):
        def search(self,*args,**kwargs):
            d=super().search(*args,**kwargs)
            d["query"]="different"
            return d
    with pytest.raises(ValueError,match="inconsistent"):
        asyncio.run(registry(Bad()).invoke("dense_search",{"query":"Q"}))


def test_backend_error():
    class Bad(Backend):
        def search(self,*args,**kwargs):
            raise RuntimeError("backend unavailable")
    with pytest.raises(RuntimeError,match="backend unavailable"):
        asyncio.run(registry(Bad()).invoke("dense_search",{"query":"Q"}))
