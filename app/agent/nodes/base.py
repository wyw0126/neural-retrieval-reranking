"""Thin graph nodes reusing the existing RAG components."""
import asyncio
from time import perf_counter

from app.agent.state import AgentState
from app.config import Settings
from app.rag.chain import SearchBackend, GenerationBackend
from app.rag.context import ContextBuilder
from app.rag.prompts import PromptBuilder
from app.rag.citations import CitationValidator, CitationReport
from app.rag.schemas import RAGAnswer
from app.agent.tools import ToolRegistry


def event(node, started, status="ok"):
    return [{"node":node,"elapsed_ms":(perf_counter()-started)*1000,"status":status}]


class AgentNodes:
    def __init__(self, settings: Settings, retrieval: SearchBackend, llm: GenerationBackend):
        self.settings=settings
        self.retrieval=retrieval
        self.tools=ToolRegistry(settings,retrieval)
        self.llm=llm
        self.context_builder=ContextBuilder(settings)
        self.prompt_builder=PromptBuilder(settings)
        self.validator=CitationValidator(settings,llm)

    async def retrieve(self,state: AgentState):
        t=perf_counter()
        name="reranked_search" if state["use_reranker"] else "dense_search"
        search=await self.tools.invoke(name,{"query":state["search_query"],
            "retrieve_top_k":state["retrieve_top_k"],"top_k":state["rerank_top_k"]})
        return {"search":search,"tool_calls":[{"name":name,
            "retrieve_top_k":search.requested_retrieve_top_k,
            "top_k":search.requested_result_top_k,
            "result_count":len(search.results)}],"trace":event("retrieve",t)}

    async def context(self,state: AgentState):
        t=perf_counter()
        context=await asyncio.to_thread(self.context_builder.build,state["search"].results)
        return {"context":context,"trace":event("context",t)}

    async def generate(self,state: AgentState):
        t=perf_counter()
        prompt=self.prompt_builder.build(state["query"],state["context"])
        answer,result=await self.llm.chat_json(list(prompt.messages),RAGAnswer,
                                               max_message_bytes=prompt.budget)
        answer=RAGAnswer.model_validate(answer.model_dump() if isinstance(answer,RAGAnswer) else answer,strict=True)
        return {"answer":answer,"model":result.model,"usage":result.usage,
                "attempts":result.attempts,"prompt_bytes":prompt.message_bytes,
                "trace":event("generate",t)}

    async def validate(self,state: AgentState):
        t=perf_counter()
        answer=state["answer"]
        cited,report=self.validator.validate(answer,state["context"].evidence)
        reason="model_insufficient_evidence" if answer.insufficient_evidence else None
        judge=None
        if self.settings.rag_semantic_validation and not answer.insufficient_evidence:
            report,judge=await self.validator.judge(state["query"],answer,cited,report)
            if report.semantic_status!="supported":
                answer=RAGAnswer(answer="当前证据未能充分支持回答。 / The answer could not be verified against the evidence.",
                                 citations=[],insufficient_evidence=True)
                cited=()
                reason="semantic_"+report.semantic_status
        return {"answer":answer,"cited_evidence":cited,"citation_report":report,
                "refusal_reason":reason,"judge_model":judge.model if judge else None,
                "judge_usage":judge.usage if judge else None,"judge_attempts":judge.attempts if judge else 0,
                "trace":event("validate",t,"refused" if reason else "ok")}

    async def refuse(self,state: AgentState):
        t=perf_counter()
        return {"answer":RAGAnswer(answer="当前检索证据不足，无法回答。 / The retrieved evidence is insufficient.",
                                   citations=[],insufficient_evidence=True),
                "cited_evidence":(),"citation_report":CitationReport("not_applicable","not_run",()),
                "refusal_reason":"no_evidence","model":None,"usage":None,"attempts":0,
                "judge_model":None,"judge_usage":None,"judge_attempts":0,"prompt_bytes":0,
                "trace":event("refuse",t,"refused")}
