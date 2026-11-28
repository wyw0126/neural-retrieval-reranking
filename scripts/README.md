# Script layout

Run from repository root. Update local shell commands to the new paths; data/model/report paths are unchanged.

## data
- `scripts/data/inspect_msmarco.py`
- `scripts/data/prepare_e1_data.py`
- `scripts/data/prepare_reranker_validation.py`

## mining
- `scripts/mining/mine_hard_negatives.py`
- `scripts/mining/mine_e2_multi_negatives.py`
- `scripts/mining/analyze_negative_quality.py`
- `scripts/mining/prepare_filtered_negatives.py`

## training/biencoder
- `scripts/training/biencoder/train_biencoder.py`

## training/reranker
- `scripts/training/reranker/train_cross_encoder_BCE.py`
- `scripts/training/reranker/train_cross_encoder_bce_multi.py`
- `scripts/training/reranker/train_cross_encoder_pairwise.py`
- `scripts/training/reranker/train_cross_encoder_listwise.py`

## indexing
- `scripts/indexing/encode_corpus.py`
- `scripts/indexing/build_faiss_index.py`

## evaluation
- `scripts/evaluation/evaluate_bm25.py`
- `scripts/evaluation/evaluate_biencoder.py`
- `scripts/evaluation/evaluate_reranker.py`
- `scripts/evaluation/select_e6_checkpoints.py`

## serving
- `scripts/serving/build_document_store.py`: build the SQLite passage database.
- `scripts/serving/smoke_search.py`: online search CLI; run with `--query "What is machine learning?"` or `--interactive`. Use `--json` for JSON Lines records, `--warmup 2 --repeat 10` for sequential timing, and `--no-rerank` for dense-only mode.

- `scripts/serving/smoke_http.py`: validate running HTTP service; use `--warmup 3 --repeat 20 --output reports/serving/m2_http_baseline.json`.

## RAG evaluation
- `scripts/evaluation/prepare_rag_eval.py`: freeze a seeded MS MARCO qrels pilot plus synthetic refusal cases.
- `scripts/evaluation/evaluate_rag.py`: run two retrieval variants against a live `/rag` service, matching the server Judge setting; save manifest and per-request JSONL.
- `scripts/evaluation/summarize_rag.py`: compare complete Judge-off/on runs, compute operational/qrels-proxy metrics and export blinded independent review records. Qrels are not gold QA answers. Run Judge settings sequentially to avoid duplicate FAISS loading.

## Agent
- `scripts/serving/smoke_agent.py`: run the LangGraph agent against the existing retrieval service. M4.2 routes retrieval through an allowlisted tool registry; use `--no-rerank` to select dense_search, otherwise reranked_search. Tool metadata is recorded separately from node traces.

M4.3: pass `--auto-route` to smoke_agent.py to enable model task routing; `--no-rerank` forces fixed dense mode. Default AGENT_ROUTER_ENABLED=false. Social responses are explicitly non-grounded; factual questions cannot use the guarded direct branch.

M4.4: --rewrite enables up to three English search queries, sequential retrieval and RRF fusion; --no-rewrite forces baseline. Original question is always used for generation. candidate_count sums per-query candidates (not unique corpus IDs); retrieval_runs contains per-query metadata; final evidence scores are representative query scores, not fused scores.
