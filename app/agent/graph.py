"""M4.1: compile a minimal, bounded LangGraph workflow."""
from dataclasses import asdict
from time import perf_counter

from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState
from app.agent.nodes.base import AgentNodes
from app.config import Settings
from app.rag.chain import SearchBackend, GenerationBackend
from app.schemas import SearchRequest


def after_context(state: AgentState):
    return "generate" if state["context"].evidence else "refuse"


class AgentEngine:
    def __init__(self,settings: Settings,retrieval: SearchBackend,llm: GenerationBackend):
        self.settings=settings
        nodes=AgentNodes(settings,retrieval,llm)
        builder=StateGraph(AgentState)
        for name in ("retrieve","context","generate","validate","refuse"):
            builder.add_node(name,getattr(nodes,name))
        builder.add_edge(START,"retrieve")
        builder.add_edge("retrieve","context")
        builder.add_conditional_edges("context",after_context,{"generate":"generate","refuse":"refuse"})
        builder.add_edge("generate","validate")
        builder.add_edge("validate",END)
        builder.add_edge("refuse",END)
        self.graph=builder.compile()

    async def answer(self,query: str,*,retrieve_top_k: int | None=None,
                     rerank_top_k: int | None=None,use_reranker: bool=True):
        body=SearchRequest(query=query,retrieve_top_k=retrieve_top_k,
                           rerank_top_k=rerank_top_k,use_reranker=use_reranker)
        k=self.settings.retrieve_top_k if body.retrieve_top_k is None else body.retrieve_top_k
        n=min(self.settings.rerank_top_k,k) if body.rerank_top_k is None else body.rerank_top_k
        if n>k:
            raise ValueError("rerank_top_k must not exceed retrieve_top_k")
        initial: AgentState={"query":body.query,"search_query":body.query,"retrieve_top_k":k,
            "rerank_top_k":n,"use_reranker":body.use_reranker,"retry_count":0,"trace":[]}
        started=perf_counter()
        final=await self.graph.ainvoke(initial,config={"recursion_limit":16})
        # Publish only final validated output and bounded operational trace.
        # Intermediate generation is not exposed through a public stream.
        context=final["context"]
        return {"query":final["query"],"search_query":final["search_query"],
                "answer":final["answer"].model_dump(mode="json"),
                "evidence":[asdict(e) for e in context.evidence],
                "cited_evidence":[asdict(e) for e in final["cited_evidence"]],
                "citation_report":asdict(final["citation_report"]),
                "refusal_reason":final["refusal_reason"],"retry_count":final["retry_count"],
                "candidate_count":final["search"].candidate_count,"retrieved_count":context.input_count,
                "dropped":context.dropped,"context_budget_units":context.estimated_tokens,
                "context_count_method":context.count_method,
                "prompt_bytes":final["prompt_bytes"],"model":final["model"],
                "usage":final["usage"].model_dump(mode="json") if final["usage"] else None,
                "attempts":final["attempts"],"judge_model":final["judge_model"],
                "judge_usage":final["judge_usage"].model_dump(mode="json") if final["judge_usage"] else None,
                "judge_attempts":final["judge_attempts"],"trace":final["trace"],
                "retrieval_timings_ms":final["search"].timings_ms,
                "total_ms":(perf_counter()-started)*1000}
