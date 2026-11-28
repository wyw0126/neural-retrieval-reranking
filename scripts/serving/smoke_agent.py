"""Run Agent graph against existing /search service."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import httpx
from app.agent.graph import AgentEngine
from app.config import load_settings
from app.llm.client import LLMClient
from scripts.serving.smoke_rag import HTTPBackend


async def run(args):
    settings=load_settings()
    async with LLMClient(settings) as llm:
        agent=AgentEngine(settings,HTTPBackend(args.base_url),llm)
        result=await agent.answer(args.query,retrieve_top_k=args.retrieve_top_k,
                                  rerank_top_k=args.top_k,use_reranker=not args.no_rerank,
                                  auto_route=True if args.auto_route else False if args.no_rerank else None,
                                  rewrite=args.rewrite)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    if args.expect_answer and result["answer"]["insufficient_evidence"]:
        raise SystemExit("Agent abstained; inspect evidence")
    print("Agent Graph Smoke PASSED",file=sys.stderr)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query",required=True)
    p.add_argument("--base-url",default="http://127.0.0.1:8000")
    p.add_argument("--retrieve-top-k",type=int)
    p.add_argument("--top-k",type=int)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--no-rerank",action="store_true")
    mode.add_argument("--auto-route",action="store_true")
    rewrite=p.add_mutually_exclusive_group()
    rewrite.add_argument("--rewrite",dest="rewrite",action="store_true",default=None)
    rewrite.add_argument("--no-rewrite",dest="rewrite",action="store_false")
    p.add_argument("--expect-answer",action="store_true")
    args=p.parse_args()
    try:
        asyncio.run(run(args))
    except httpx.ConnectError:
        raise SystemExit("Cannot connect to retrieval service; check /health") from None


if __name__=="__main__":
    main()
