
#!/usr/bin/env python3
# scripts/evaluate_reranker.py

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


ROOT = Path(__file__).resolve().parents[1]


def read_queries(path):
    queries = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            qid, text = line.rstrip("\n").split("\t", 1)
            qid = int(qid)

            if qid in queries and queries[qid] != text:
                raise ValueError(f"Query 冲突: {qid}")

            queries[qid] = text

    return queries


def read_qrels(path):
    qrels = defaultdict(dict)

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 4:
                raise ValueError(f"Qrels 格式错误: {line[:100]}")

            qid = int(parts[0])
            pid = int(parts[2])
            rel = int(parts[3])

            if pid in qrels[qid] and qrels[qid][pid] != rel:
                raise ValueError(
                    f"Qrels 标签冲突: qid={qid}, pid={pid}"
                )

            qrels[qid][pid] = rel

    return dict(qrels)


def model_metadata(index_path):
    path = Path(index_path)

    candidates = [
        Path(str(path) + ".meta.json"),
        path.with_suffix(".meta.json"),
        path.with_suffix(".json"),
    ]

    for candidate in candidates:
        if candidate.is_file():
            return json.loads(
                candidate.read_text(encoding="utf-8")
            )

    return {}


def load_faiss(index_path, device, gpu_id):
    cpu_index = faiss.read_index(str(index_path))

    if not isinstance(cpu_index, faiss.IndexIDMap2):
        raise TypeError(
            "索引必须为 IndexIDMap2(IndexFlatIP)"
        )

    if device == "cpu":
        return cpu_index, None, None

    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError("FAISS 当前不支持 GPU")

    if gpu_id >= faiss.get_num_gpus():
        raise RuntimeError("指定的 GPU 不可用")

    flat = faiss.downcast_index(cpu_index.index)

    if not isinstance(flat, faiss.IndexFlatIP):
        raise TypeError("底层索引必须是 IndexFlatIP")

    ids = faiss.vector_to_array(
        cpu_index.id_map
    ).astype(np.int64)

    resources = faiss.StandardGpuResources()
    options = faiss.GpuClonerOptions()
    options.useFloat16 = False

    gpu_index = faiss.index_cpu_to_gpu(
        resources, gpu_id, flat, options
    )

    if gpu_index.ntotal != len(ids):
        raise RuntimeError("GPU 索引数量不一致")

    print(
        "FAISS GPU 类型:",
        type(gpu_index).__name__,
        flush=True,
    )

    return gpu_index, ids, resources


def search(index, vectors, top_k, id_map=None):
    vectors = np.ascontiguousarray(
        vectors, dtype=np.float32
    )

    scores, ids = index.search(vectors, top_k)

    if id_map is not None:
        positions = ids
        ids = np.full(
            positions.shape, -1, dtype=np.int64
        )
        valid = positions >= 0

        if np.any(positions[valid] >= len(id_map)):
            raise RuntimeError("FAISS 内部位置越界")

        ids[valid] = id_map[positions[valid]]

    return scores, ids


def extract_passages(path, required):
    texts = {}

    print(
        f"提取 {len(required):,} 篇候选 Passage...",
        flush=True,
    )

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            pid, sep, text = line.partition("\t")

            if not sep:
                raise ValueError(
                    f"collection 第 {line_no} 行格式错误"
                )

            pid = int(pid)

            if pid in required:
                texts[pid] = text.rstrip("\r\n")

            if len(texts) == len(required):
                break

    missing = required - texts.keys()

    if missing:
        raise ValueError(
            f"缺少 {len(missing)} 篇候选正文"
        )

    return texts


def score_query(ranked_ids, relevance):
    positives = {
        pid: grade
        for pid, grade in relevance.items()
        if grade > 0
    }

    total = len(positives)

    if total == 0:
        raise ValueError("评估 Query 没有正例")

    top10 = ranked_ids[:10]
    top100 = ranked_ids[:100]

    rr = 0.0

    for rank, pid in enumerate(top10, 1):
        if pid in positives:
            rr = 1.0 / rank
            break

    gains = [
        (2 ** positives.get(pid, 0) - 1)
        / math.log2(rank + 1)
        for rank, pid in enumerate(top10, 1)
    ]

    ideal_grades = sorted(
        positives.values(), reverse=True
    )[:10]

    ideal = sum(
        (2 ** grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, 1)
    )

    return {
        "MRR@10": rr,
        "NDCG@10": sum(gains) / ideal if ideal else 0.0,
        "Recall@10": sum(
            pid in positives for pid in top10
        ) / total,
        "Recall@100": sum(
            pid in positives for pid in top100
        ) / total,
    }


def average_metrics(rows):
    names = [
        "MRR@10",
        "NDCG@10",
        "Recall@10",
        "Recall@100",
    ]

    return {
        name: sum(r[name] for r in rows) / len(rows)
        for name in names
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--biencoder",
        default="models/minilm-msmarco-e2-hn",
    )

    p.add_argument(
        "--index",
        default="data/indexes/faiss/minilm_e2_flatip.faiss",
    )

    p.add_argument(
        "--reranker",
        default="models/ms-marco-MiniLM-L6-v2-cross-encoder",
    )

    p.add_argument(
        "--data-dir",
        default="data/raw/msmarco",
    )

    p.add_argument(
        "--output",
        default="reports/experiments/E3_reranker_full.json",
    )

    p.add_argument(
        "--rankings-output",
        default="data/processed/msmarco/E3_top10.jsonl",
    )

    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--query-batch-size", type=int, default=128)
    p.add_argument("--rerank-batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--faiss-device",
        choices=["cpu", "gpu"],
        default="gpu",
    )
    p.add_argument("--gpu-id", type=int, default=0)

    args = p.parse_args()

    if (
        args.top_k < 100
        or args.query_batch_size <= 0
        or args.rerank_batch_size <= 0
    ):
        p.error(
            "top-k 至少为 100，所有 batch size 必须为正数"
        )

    if args.max_queries is not None and args.max_queries <= 0:
        p.error("max-queries 必须大于 0")

    output = Path(args.output)
    rankings_output = Path(args.rankings_output)

    if output.exists() or rankings_output.exists():
        raise FileExistsError(
            "输出文件已存在，请换用新的实验路径"
        )

    data_dir = Path(args.data_dir)

    queries = read_queries(
        data_dir / "queries.dev.small.tsv"
    )

    qrels = read_qrels(
        data_dir / "qrels.dev.small.tsv"
    )

    qids = sorted(
        qid for qid in qrels
        if qid in queries
        and any(grade > 0 for grade in qrels[qid].values())
    )

    if args.max_queries is not None:
        rng = random.Random(args.seed)
        qids = sorted(
            rng.sample(
                qids,
                min(args.max_queries, len(qids)),
            )
        )

    print("Query 数量:", len(qids), flush=True)

    if not Path(args.reranker).is_dir():
        raise FileNotFoundError(
            "缺少本地 Cross-Encoder 模型: "
            + args.reranker
        )

    print("加载 E2 Bi-Encoder...", flush=True)

    biencoder = SentenceTransformer(
        args.biencoder,
        device=args.device,
        local_files_only=True,
    )

    print("加载 FAISS...", flush=True)

    index, id_map, resources = load_faiss(
        args.index,
        args.faiss_device,
        args.gpu_id,
    )

    if index.d != biencoder.get_embedding_dimension():
        raise ValueError("Bi-Encoder 与 FAISS 维度不同")

    metadata = model_metadata(args.index)

    print("索引元数据:", metadata, flush=True)

    # 第一阶段：从全库召回 Top 100
    retrieval = {}
    required_pids = set()

    query_encode_seconds = 0.0
    faiss_search_seconds = 0.0

    print("开始 E2 召回...", flush=True)

    for start in range(
        0, len(qids), args.query_batch_size
    ):
        batch_qids = qids[
            start:start + args.query_batch_size
        ]

        t0 = time.perf_counter()

        vectors = biencoder.encode(
            [queries[qid] for qid in batch_qids],
            batch_size=args.query_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        query_encode_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()

        scores, doc_ids = search(
            index, vectors, args.top_k, id_map
        )

        faiss_search_seconds += time.perf_counter() - t0

        for qid, ids in zip(batch_qids, doc_ids):
            valid = [int(pid) for pid in ids if pid >= 0]

            if len(valid) != args.top_k:
                raise RuntimeError(
                    f"Query {qid} 召回数量不足"
                )

            if len(set(valid)) != len(valid):
                raise RuntimeError(
                    f"Query {qid} 存在重复 Passage ID"
                )

            retrieval[qid] = valid
            required_pids.update(valid)

        print(
            f"\r召回 {start + len(batch_qids):,}/{len(qids):,}",
            end="",
            flush=True,
        )

    print("\n召回完成", flush=True)

    texts = extract_passages(
        data_dir / "collection.tsv",
        required_pids,
    )

    # 第二阶段：Cross-Encoder 重排序
    print("加载 Cross-Encoder...", flush=True)

    reranker = CrossEncoder(
        args.reranker,
        device=args.device,
        local_files_only=True,
    )

    before = []
    after = []
    rerank_seconds = 0.0

    output.parent.mkdir(parents=True, exist_ok=True)
    rankings_output.parent.mkdir(
        parents=True, exist_ok=True
    )

    # 按 Query 逐条保存结果，防止大量排名积压在内存。
    # 正式输出将在全部评估成功后生成。
    partial_rankings = Path(
        str(rankings_output) + ".partial"
    )

    if partial_rankings.exists():
        raise FileExistsError(partial_rankings)

    try:
        with partial_rankings.open(
            "x", encoding="utf-8"
        ) as f:
            for i, qid in enumerate(qids, 1):
                doc_ids = retrieval[qid]

                pairs = [
                    [queries[qid], texts[pid]]
                    for pid in doc_ids
                ]

                t0 = time.perf_counter()

                scores = reranker.predict(
                    pairs,
                    batch_size=args.rerank_batch_size,
                    show_progress_bar=False,
                )

                rerank_seconds += time.perf_counter() - t0

                scores = np.asarray(scores).reshape(-1)

                if len(scores) != len(doc_ids):
                    raise RuntimeError("Cross-Encoder 分数数量错误")

                if not np.isfinite(scores).all():
                    raise RuntimeError("Cross-Encoder 出现非法分数")

                order = np.argsort(
                    -scores, kind="stable"
                )

                reranked = [
                    doc_ids[int(j)]
                    for j in order
                ]

                before.append(
                    score_query(doc_ids, qrels[qid])
                )

                after.append(
                    score_query(reranked, qrels[qid])
                )

                f.write(json.dumps({
                    "query_id": qid,
                    "retrieved_top10": doc_ids[:10],
                    "reranked_top10": reranked[:10],
                    "reranked_scores_top10": [
                        float(scores[int(j)])
                        for j in order[:10]
                    ],
                }, ensure_ascii=False) + "\n")

                if i % 100 == 0 or i == len(qids):
                    print(
                        f"已重排 {i:,}/{len(qids):,}",
                        flush=True,
                    )

        baseline_metrics = average_metrics(before)
        reranked_metrics = average_metrics(after)

        results = {
            "experiment_id": "E3",
            "stage": "E2 Bi-Encoder + pretrained Cross-Encoder",
            "biencoder": str(Path(args.biencoder).resolve()),
            "reranker": str(Path(args.reranker).resolve()),
            "index": str(Path(args.index).resolve()),
            "faiss_device": args.faiss_device,
            "index_metadata": metadata,
            "split": "MS MARCO Passage v1 dev.small",
            "seed": args.seed,
            "query_count": len(qids),
            "corpus_count": int(index.ntotal),
            "top_k": args.top_k,
            "query_batch_size": args.query_batch_size,
            "rerank_batch_size": args.rerank_batch_size,
            "query_encoding_seconds": query_encode_seconds,
            "faiss_search_seconds": faiss_search_seconds,
            "rerank_seconds": rerank_seconds,
            "mean_rerank_ms_per_query": (
                rerank_seconds / len(qids) * 1000
            ),
            "first_stage_metrics": baseline_metrics,
            "reranked_metrics": reranked_metrics,
            "improvement": {
                k: reranked_metrics[k] - baseline_metrics[k]
                for k in baseline_metrics
            },
            "rankings_output": str(rankings_output),
        }

        output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        partial_rankings.rename(rankings_output)

    except Exception:
        print(
            "评估未完成，临时排名保留在:",
            partial_rankings,
            flush=True,
        )
        raise

    print("\n========== E3 结果 ==========")
    print("E2 原始召回指标:", baseline_metrics)
    print("E3 重排序指标:", reranked_metrics)
    print("重排序耗时:", rerank_seconds)
    print("结果文件:", output)


if __name__ == "__main__":
    main()
