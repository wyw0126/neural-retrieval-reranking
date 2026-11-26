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
