import asyncio
import json
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.llm.exceptions import LLMRequestError
from app.llm.schemas import ChatResult, TokenUsage
from app.rag.chain import RAGEngine
from app.rag.citations import CitationValidator, GroundingVerdict, RAGCitationError
from app.rag.context import Evidence
from app.rag.schemas import RAGAnswer


def settings(**kwargs):
    return Settings(_env_file=None, **kwargs)


def evidence():
    return (Evidence(1,42,"ML learns patterns from data.",.5,2.,1,1,False),
            Evidence(2,99,"ML is a branch of AI.",.4,1.,2,2,False))


def answer(text="ML learns patterns [1].", ids=None, insufficient=False):
    return RAGAnswer(answer=text,citations=[1] if ids is None else ids,
                     insufficient_evidence=insufficient)


def test_mapping_and_repeat():
    cited, report = CitationValidator(settings()).validate(
        answer("ML is a branch of AI [2]. It learns patterns [1]. Data matters [1].",[1,2]), evidence())
    assert [e.passage_id for e in cited]==[42,99]
    assert report.inline_ids==(2,1)
    assert report.structural_status=="passed" and report.semantic_status=="not_run"


@pytest.mark.parametrize("text,ids", [
    ("Fact [9].",[9]), ("Fact [9].",[1]), ("Fact [1].",[9]),
    ("Fact.",[1]), ("Fact [1].",[1,2]), ("Fact [1] [2].",[1]),
    ("Fact [1,2].",[1,2]), ("Fact [1-2].",[1,2]),
    ("Fact [01].",[1]), ("Fact [0].",[1]), ("Fact [[1]].",[1]),
])
def test_invalid_inline(text, ids):
    with pytest.raises(RAGCitationError):
        CitationValidator(settings()).validate(answer(text,ids),evidence())


def test_refusal_and_mapping():
    validator=CitationValidator(settings())
    cited,report=validator.validate(answer("Insufficient.",[],True),evidence())
    assert not cited and report.structural_status=="not_applicable"
    with pytest.raises(RAGCitationError):
        validator.validate(answer("Insufficient [1].",[],True),evidence())
    with pytest.raises(RAGCitationError):
        validator.validate(answer(),(evidence()[0],evidence()[0]))


class FakeSearch:
    def search(self, query, **kwargs):
        hits=[SimpleNamespace(passage_id=42,text="ML learns patterns from data.",rank=1,
                              retrieval_rank=1,retrieval_score=.5,rerank_score=2.)]
        return SimpleNamespace(results=hits,candidate_count=1,timings_ms={"total":1.})


class FakeLLM:
    def __init__(self, verdict="supported", refusal=False, error=False):
        self.verdict=verdict
        self.refusal=refusal
        self.error=error
        self.calls=[]

    async def chat_json(self,messages,schema,*,max_message_bytes=None):
        self.calls.append((messages,schema,max_message_bytes))
        if schema is GroundingVerdict:
            if self.error:
                raise LLMRequestError(503)
            data=json.loads(messages[-1].content)
            assert [e["evidence_id"] for e in data["cited_evidence"]]==[1]
            parsed=GroundingVerdict(verdict=self.verdict,reason="test")
        else:
            parsed=answer("Insufficient.",[],True) if self.refusal else answer()
        return parsed,ChatResult(content=parsed.model_dump_json(),model="fake",
            elapsed_ms=1.,attempts=1,finish_reason="stop",
            usage=TokenUsage(prompt_tokens=10,completion_tokens=5,total_tokens=15))


@pytest.mark.parametrize("verdict",["supported","unsupported","uncertain"])
def test_chain_judge(verdict):
    llm=FakeLLM(verdict)
    result=asyncio.run(RAGEngine(settings(rag_semantic_validation=True),FakeSearch(),llm).answer("ML?"))
    assert len(llm.calls)==2 and result.citation_report.semantic_status==verdict
    assert result.judge_attempts==1 and result.judge_usage.total_tokens==15
    assert result.usage.total_tokens==15
    assert result.to_dict()["judge_usage"]["total_tokens"]==15
    if verdict=="supported":
        assert not result.answer.insufficient_evidence and result.cited_evidence
    else:
        assert result.answer.insufficient_evidence and not result.cited_evidence
        assert result.refusal_reason=="semantic_"+verdict
        assert result.answer.citations==[]


@pytest.mark.parametrize("enabled,refusal",[(False,False),(True,True)])
def test_skip_judge(enabled,refusal):
    llm=FakeLLM(refusal=refusal)
    result=asyncio.run(RAGEngine(settings(rag_semantic_validation=enabled),FakeSearch(),llm).answer("ML?"))
    assert len(llm.calls)==1 and result.judge_attempts==0
    assert result.judge_usage is None and result.citation_report.semantic_status=="not_run"


def test_judge_failure_propagates():
    with pytest.raises(LLMRequestError):
        asyncio.run(RAGEngine(settings(rag_semantic_validation=True),FakeSearch(),
                              FakeLLM(error=True)).answer("ML?"))
