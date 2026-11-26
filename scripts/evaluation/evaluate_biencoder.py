#B1：评估MiniLM模型

import argparse
import json
import random
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.msmarco_loader import MSMARCOLoader
from scripts.evaluate_bm25 import evaluate_one


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/all-MiniLM-L6-v2",
    )

    parser.add_argument(
        "--experiment-id",
        type=str,
        default="B1",
        help="实验编号：B1、E1、E2-HN 等",
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/raw/msmarco",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=ROOT / "data/indexes/faiss/minilm_flatip.faiss",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "dev.small"],
        default="dev.small",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--top-k", type=int, default=100
    )
    parser.add_argument(
        "--batch-size", type=int, default=128
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--device", default="cuda"
    )
    parser.add_argument(
        "--threads", type=int, default=16
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/experiments/B1_biencoder.json",
    )
    parser.add_argument(
        "--run-output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.top_k < 100:
        parser.error("--top-k 必须 >= 100")
    if args.batch_size <= 0 or args.threads <= 0:
        parser.error("batch-size 和 threads 必须大于 0")
    if args.max_queries is not None and args.max_queries <= 0:
        parser.error("max-queries 必须大于 0")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    faiss.omp_set_num_threads(args.threads)

    index_path = args.index_path.resolve()
    meta_path = index_path.with_suffix(".json")

    if not index_path.is_file():
        raise FileNotFoundError(index_path)

    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)

    metadata = load_json(meta_path)

    print("加载 FAISS 索引...", flush=True)
    started_load = time.perf_counter()
    index = faiss.read_index(str(index_path))

    if index.ntotal != metadata["document_count"]:
        raise RuntimeError("FAISS 索引文档数量不匹配")

    print(
        f"已加载 {index.ntotal:,} 篇 Passage，"
        f"耗时 {time.perf_counter()-started_load:.1f}s"
    )

    print("加载 MiniLM...", flush=True)

    model = SentenceTransformer(
        str(args.model.resolve()),
        device=args.device,
        local_files_only=True,
    )
    model.eval()

    dim = model.get_sentence_embedding_dimension()

    if dim != index.d:
        raise RuntimeError(
            "模型 Embedding 维度与索引不一致"
        )

    # 确保索引与当前模型使用相同的权重版本。
    from scripts.encode_corpus import model_identity

    current_hash = model_identity(args.model.resolve())

    if current_hash != metadata["embedding_model_hash"]:
        raise RuntimeError(
            "当前 MiniLM 与构建索引时的模型不一致"
        )

    loader = MSMARCOLoader(args.data_dir)
    qrels = loader.load_qrels(args.split)

    # 与之前 BM25 的采样算法保持一致。
    valid_qids = {
        qid
        for qid, rels in qrels.items()
        if any(grade > 0 for grade in rels.values())
    }

    queries = loader.load_queries(
        args.split,
        query_ids=valid_qids,
    )

    missing = valid_qids - queries.keys()
    if missing:
        raise RuntimeError(
            f"{len(missing)} 条 Qrels Query 找不到文本"
        )

    qids = sorted(valid_qids)

    if args.max_queries is not None:
        rng = random.Random(args.seed)
        qids = sorted(
            rng.sample(
                qids,
                min(args.max_queries, len(qids)),
            )
        )

    query_texts = [queries[qid] for qid in qids]

    print(
        f"开始编码 {len(qids):,} 条 Query...",
        flush=True,
    )
    t0 = time.perf_counter()

    query_vectors = model.encode(
        query_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    query_vectors = np.ascontiguousarray(
        query_vectors, dtype=np.float32
    )

    encoding_seconds = time.perf_counter() - t0

    print("开始 FAISS 检索...", flush=True)
    t1 = time.perf_counter()

    # 可以批量检索，避免逐条 Python 调用 FAISS。
    scores, doc_ids = index.search(
        query_vectors,
        args.top_k,
    )

    search_seconds = time.perf_counter() - t1

    totals = {
        "MRR@10": 0.0,
        "NDCG@10": 0.0,
        "Recall@10": 0.0,
        "Recall@100": 0.0,
    }

    if args.run_output is not None:
        args.run_output.parent.mkdir(
            parents=True, exist_ok=True
        )

    run_file = (
        args.run_output.open("w", encoding="utf-8")
        if args.run_output is not None
        else None
    )

    try:
        for i, qid in enumerate(qids):
            retrieved = [
                {
                    "pid": int(pid),
                    "score": float(score),
                }
                for pid, score in zip(
                    doc_ids[i], scores[i]
                )
                if pid >= 0
            ]

            metrics = evaluate_one(
                retrieved,
                qrels[qid],
            )

            for name, value in metrics.items():
                totals[name] += value

            if run_file is not None:
                row = {
                    "qid": qid,
                    "metrics": metrics,
                    "retrieved": retrieved,
                }
                run_file.write(
                    json.dumps(row, ensure_ascii=False)
                    + "\n"
                )
    finally:
        if run_file is not None:
            run_file.close()

    average = {
        name: total / len(qids)
        for name, total in totals.items()
    }

    report = {
        "experiment_id": args.experiment_id,
        "model": str(args.model.resolve()),
        "model_hash": current_hash,
        "index_type": metadata["index_type"],
        "embedding_dim": dim,
        "normalize_embeddings": True,
        "split": f"MS MARCO Passage v1 {args.split}",
        "seed": args.seed,
        "query_count": len(qids),
        "corpus_count": index.ntotal,
        "top_k": args.top_k,
        "query_batch_size": args.batch_size,
        "faiss_threads": args.threads,
        "query_encoding_seconds": encoding_seconds,
        "faiss_search_seconds": search_seconds,
        "mean_faiss_search_ms": (
            search_seconds / len(qids) * 1000
        ),
        **average,
    }

    args.output.parent.mkdir(
        parents=True, exist_ok=True
    )

    args.output.write_text(
        json.dumps(
            report, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print("\n========== B1 Bi-Encoder ==========")
    print(json.dumps(
        report, indent=2, ensure_ascii=False
    ))
    print("\n结果保存至:", args.output)


if __name__ == "__main__":
    main()
