from types import SimpleNamespace
import httpx
from fastapi.testclient import TestClient
from app.main import create_app
from app.config import Settings
from app.api.client import RetrievalClient
from scripts.serving.smoke_http import run


class FakeEngine:
    def model_info(self):
        return dict(biencoder="fake",biencoder_hash="fake",reranker="fake",
                    model_device="cpu",faiss_device="cpu",embedding_dim=384,
                    document_count=8841823,collection_sha256="a"*64,reranker_max_length=256)

    def search(self,query,*,retrieve_top_k,rerank_top_k,use_reranker):
        return dict(query=query,use_reranker=use_reranker,
            requested_retrieve_top_k=retrieve_top_k,requested_result_top_k=rerank_top_k,
            candidate_count=retrieve_top_k,results=[
                dict(passage_id=i,text="document",retrieval_score=1/i,
                     rerank_score=1/i if use_reranker else None,retrieval_rank=i,rank=i)
                for i in range(1,rerank_top_k+1)],timings_ms={"total":1.0,"reranking":1.0 if use_reranker else 0.0})

    def retrieve(self,query,top_k=None):
        return self.search(query,retrieve_top_k=top_k,rerank_top_k=top_k,use_reranker=False)

    def rerank(self,query,documents,top_k=None):
        return [SimpleNamespace(passage_id=d.passage_id,text=d.text,rerank_score=1/i,
                                rank=i,original_rank=i) for i,d in enumerate(documents,1)][:top_k]


def test_full_http_workflow():
    app=create_app(Settings(_env_file=None),engine_factory=lambda _:FakeEngine())
    with TestClient(app) as transport:
        with RetrievalClient() as client:
            client.http.close()
            client.http=transport
            report=run(client,warmup=0,repeat=1)
        assert report["status"]=="passed"
        assert report["summaries"]["two_stage"]["requests"]==5
        assert report["summaries"]["dense_only"]["requests"]==5


def test_error_openapi():
    app=create_app(engine_factory=lambda _:FakeEngine())
    schema=app.openapi()
    for path,method in [("/health","get"),("/models","get"),("/search","post"),("/retrieve","post"),("/rerank","post")]:
        responses=schema["paths"][path][method]["responses"]
        for status in ("500","503"):
            assert responses[status]["content"]["application/json"]["schema"]["$ref"].endswith("/ErrorResponse")
    assert schema["paths"]["/search"]["post"]["responses"]["422"]["content"]["application/json"]["schema"]["$ref"].endswith("/ErrorResponse")
