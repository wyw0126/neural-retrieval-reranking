#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
E2: GPU FAISS Hard Negative Mining

输入:
  B1 MiniLM 模型
  B1 IndexIDMap2(IndexFlatIP) 索引
  Query-Positive JSONL
  Train Qrels
  collection.tsv

输出:
  E2 Query-Positive-Negative JSONL
  E2 挖掘元数据 JSON

注意:
  1. 使用 GPU FAISS 精确内积搜索
  2. GPU 底层索引返回内部位置，需要恢复 Passage ID
  3. 排除所有已标注正例
  4. 默认从 Top 10-100 随机选择一个候选负例
  5. 未标注不等于确认不相关
  6. 本版本不支持中断续跑
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]

B1_MODEL_HASH = (
    "e502cd4c833c097d450923a325c5c95750952c8a78f80818d593f9e4556220f2"
)


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_qrels(path):
    """
    Qrels 格式:
      qid  0  pid  relevance

    允许完全相同的重复关系。
    不允许同一关系出现不同 relevance。
    """
    labels = {}
    positives = {}
    duplicate_count = 0

    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 4:
                raise ValueError(
                    f"Qrels 第 {line_no} 行格式错误"
                )

            qid = int(parts[0])
            pid = int(parts[2])
            relevance = int(parts[3])

            key = (qid, pid)

            if key in labels:
                if labels[key] != relevance:
                    raise ValueError(
                        f"Qrels 标注冲突: "
                        f"qid={qid}, pid={pid}"
                    )

                duplicate_count += 1
                continue

            labels[key] = relevance

            if relevance > 0:
                positives.setdefault(qid, set()).add(pid)

    print(
        f"Qrels 加载完成: "
        f"{len(positives):,} 个 Query, "
        f"{duplicate_count:,} 条重复关系",
        flush=True,
    )

    return positives, duplicate_count


def load_pairs(path, max_queries=None):
    """读取 Query-Positive 配对。"""
    records = []

    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            r = json.loads(line)

            required = (
                "query_id",
                "positive_id",
                "query",
                "positive",
            )

            for key in required:
                if key not in r:
                    raise ValueError(
                        f"第 {line_no} 行缺少字段: {key}"
                    )

            records.append({
                "query_id": int(r["query_id"]),
                "positive_id": int(r["positive_id"]),
                "query": str(r["query"]),
                "positive": str(r["positive"]),
            })

            if (
                max_queries is not None
                and len(records) >= max_queries
            ):
                break

    return records


def read_index_metadata(index_path):
    """
    查找 B1 索引元数据。
    """
    index_path = Path(index_path)

    candidates = [
        Path(str(index_path) + ".meta.json"),
        index_path.with_suffix(".meta.json"),
        Path(str(index_path) + ".json"),
        index_path.with_suffix(".json"),
    ]

    for candidate in candidates:
        if candidate.exists():
            metadata = json.loads(
                candidate.read_text(encoding="utf-8")
            )
            return candidate, metadata

    raise FileNotFoundError(
        f"找不到索引元数据: {index_path}"
    )


def verify_model_and_index(
    model_path,
    index_path,
    expected_hash,
):
    """
    核对索引元数据对应的模型身份。

    附加检查元数据中的模型路径（如果存在）。
    真正的模型文件内容哈希仍应由项目已有
    model_identity() 实现核验。
    """
    meta_path, metadata = read_index_metadata(
        index_path
    )

    index_hash = metadata.get("embedding_model_hash")

    if index_hash != expected_hash:
        raise ValueError(
            "索引模型哈希不匹配:\n"
            f"预期: {expected_hash}\n"
            f"实际: {index_hash}"
        )

    model_path = Path(model_path).resolve()

    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    recorded_model = metadata.get("model")

    if recorded_model:
        recorded_path = Path(recorded_model)

        if recorded_path.is_absolute():
            if recorded_path.resolve() != model_path:
                raise ValueError(
                    "索引记录的模型路径与当前模型不同"
                )

    print("索引元数据验证通过", flush=True)
    print("模型哈希:", index_hash, flush=True)
    print("元数据文件:", meta_path, flush=True)

    return metadata


def load_gpu_index(
    index_path,
    model_dimension,
    gpu_id=0,
):
    """
    读取 IndexIDMap2(IndexFlatIP)，
    将底层 FlatIP 向量复制到 GPU。

    返回:
      gpu_index
      passage_ids
      gpu_resources
      cpu_index

    cpu_index 保留用于小规模正确性检查。
    """
    print("加载 CPU FAISS 索引...", flush=True)

    cpu_index = faiss.read_index(
        str(index_path)
    )

    if not isinstance(
        cpu_index,
        faiss.IndexIDMap2,
    ):
        raise TypeError(
            "预期 IndexIDMap2 索引，"
            f"实际: {type(cpu_index).__name__}"
        )

    passage_ids = faiss.vector_to_array(
        cpu_index.id_map
    ).astype(np.int64)

    flat_index = faiss.downcast_index(
        cpu_index.index
    )

    if not isinstance(
        flat_index,
        faiss.IndexFlatIP,
    ):
        raise TypeError(
            "底层不是 IndexFlatIP: "
            f"{type(flat_index).__name__}"
        )

    if flat_index.d != model_dimension:
        raise ValueError(
            "模型与索引维度不一致"
        )

    if flat_index.ntotal != len(passage_ids):
        raise ValueError(
            "Passage ID 数量与向量数量不一致"
        )

    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError(
            "FAISS 缺少 GPU 支持"
        )

    gpu_count = faiss.get_num_gpus()

    if gpu_id < 0 or gpu_id >= gpu_count:
        raise RuntimeError(
            f"GPU {gpu_id} 不可用, "
            f"可见 GPU 数量: {gpu_count}"
        )

    print("复制 FlatIP 索引到 GPU...", flush=True)

    gpu_resources = faiss.StandardGpuResources()

    options = faiss.GpuClonerOptions()
    options.useFloat16 = False

    t0 = time.perf_counter()

    gpu_index = faiss.index_cpu_to_gpu(
        gpu_resources,
        gpu_id,
        flat_index,
        options,
    )

    elapsed = time.perf_counter() - t0

    print(
        "转换返回类型:",
        type(gpu_index).__name__,
        flush=True,
    )

    # FAISS 可能返回 GpuIndexFlat，
    # 而不是 GpuIndexFlatIP。
    # 不使用过于严格的具体类型判断。
    gpu_base = getattr(faiss, "GpuIndex", None)

    if gpu_base is not None:
        if not isinstance(gpu_index, gpu_base):
            raise RuntimeError(
                "返回对象不是 GPU 索引类型: "
                f"{type(gpu_index).__name__}"
            )

    if not hasattr(gpu_index, "search"):
        raise RuntimeError(
            "GPU 索引不支持 search"
        )

    if gpu_index.ntotal != len(passage_ids):
        raise RuntimeError(
            "GPU 索引向量数量不一致"
        )

    if gpu_index.d != model_dimension:
        raise RuntimeError(
            "GPU 索引维度不一致"
        )

    print("\nGPU 索引加载完成", flush=True)
    print("设备:", gpu_id, flush=True)
    print(
        "类型:",
        type(gpu_index).__name__,
        flush=True,
    )
    print(
        "向量数量:",
        f"{gpu_index.ntotal:,}",
        flush=True,
    )
    print("维度:", gpu_index.d, flush=True)
    print(
        "复制耗时:",
        f"{elapsed:.2f}s",
        flush=True,
    )

    return (
        gpu_index,
        passage_ids,
        gpu_resources,
        cpu_index,
    )


def gpu_search(
    index,
    vectors,
    passage_ids,
    top_k,
):
    """
    GPU FlatIP 返回的是内部位置，
    不是原始 MS MARCO Passage ID。
    """
    vectors = np.ascontiguousarray(
        vectors,
        dtype=np.float32,
    )

    scores, positions = index.search(
        vectors,
        top_k,
    )

    ids = np.full(
        positions.shape,
        -1,
        dtype=np.int64,
    )

    valid = positions >= 0

    if np.any(
        positions[valid] >= len(passage_ids)
    ):
        raise RuntimeError(
            "GPU 返回了越界位置"
        )

    ids[valid] = passage_ids[
        positions[valid]
    ]

    return scores, ids


def validate_search(
    cpu_index,
    gpu_index,
    passage_ids,
    model,
    queries,
    top_k=10,
):
    """
    使用相同 Query 验证 CPU/GPU 检索结果。

    不强制要求分数逐位一致，
    但要求 Top-K 结果高度一致。
    """
    print(
        "\n验证 CPU/GPU 检索一致性...",
        flush=True,
    )

    vectors = model.encode(
        queries[:2],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    cpu_scores, cpu_ids = cpu_index.search(
        vectors,
        top_k,
    )

    gpu_scores, gpu_ids = gpu_search(
        gpu_index,
        vectors,
        passage_ids,
        top_k,
    )

    for row in range(len(vectors)):
        cpu_set = set(
            int(x) for x in cpu_ids[row]
            if x >= 0
        )

        gpu_set = set(
            int(x) for x in gpu_ids[row]
            if x >= 0
        )

        overlap = len(
            cpu_set & gpu_set
        ) / max(len(cpu_set), 1)

        print(
            f"Query {row}: "
            f"Top-{top_k} 重合率 "
            f"{overlap:.1%}",
            flush=True,
        )

        if overlap < 0.9:
            raise RuntimeError(
                "CPU/GPU 检索结果差异较大，"
                "请检查 GPU 索引或 Passage ID 映射"
            )

        if not np.allclose(
            cpu_scores[row],
            gpu_scores[row],
            rtol=1e-3,
            atol=1e-4,
        ):
            raise RuntimeError(
                "CPU/GPU 相似度分数差异过大"
            )

    print(
        "CPU/GPU 检索一致性检查通过",
        flush=True,
    )


def extract_passages(
    collection_path,
    required_pids,
):
    """
    流式读取 collection.tsv，
    只提取需要的负例 Passage。
    """
    texts = {}

    print(
        f"\n提取 {len(required_pids):,} "
        "个负例 Passage...",
        flush=True,
    )

    with Path(collection_path).open(
        encoding="utf-8"
    ) as f:
        for line_no, line in enumerate(f, 1):
            pid_str, sep, text = line.partition(
                "\t"
            )

            if not sep:
                raise ValueError(
                    f"collection 第 {line_no} 行格式错误"
                )

            pid = int(pid_str)

            if pid in required_pids:
                texts[pid] = text.rstrip("\r\n")

            if len(texts) == len(required_pids):
                break

            if line_no % 2_000_000 == 0:
                print(
                    f"已扫描 {line_no:,} 篇",
                    flush=True,
                )

    missing = required_pids - set(texts)

    if missing:
        raise RuntimeError(
            f"有 {len(missing)} 个 Passage "
            "未找到正文"
        )

    return texts


def main():
    parser = argparse.ArgumentParser(
        description="E2 GPU Hard Negative Mining"
    )

    parser.add_argument(
        "--model",
        default=str(
            ROOT / "models/all-MiniLM-L6-v2"
        ),
    )

    parser.add_argument(
        "--index",
        default=str(
            ROOT / "data/indexes/faiss/minilm_flatip.faiss"
        ),
    )

    parser.add_argument(
        "--pairs",
        default=str(
            ROOT / "data/processed/msmarco/e1_train.jsonl"
        ),
    )

    parser.add_argument(
        "--qrels",
        default=str(
            ROOT / "data/raw/msmarco/qrels.train.tsv"
        ),
    )

    parser.add_argument(
        "--collection",
        default=str(
            ROOT / "data/raw/msmarco/collection.tsv"
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            ROOT / "data/processed/msmarco/e2_triplets_gpu.jsonl"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--rank-min",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--expected-model-hash",
        default=B1_MODEL_HASH,
    )

    args = parser.parse_args()

    if not (
        1 <= args.rank_min <= args.top_k
    ):
        parser.error(
            "必须满足 1 <= rank-min <= top-k"
        )

    if args.batch_size <= 0:
        parser.error(
            "batch-size 必须大于 0"
        )

    if (
        args.max_queries is not None
        and args.max_queries <= 0
    ):
        parser.error(
            "max-queries 必须大于 0"
        )

    output = Path(args.output)
    meta_path = output.with_suffix(
        ".meta.json"
    )
    partial_path = Path(
        str(output) + ".partial"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in (
        output,
        meta_path,
        partial_path,
    ):
        if path.exists():
            raise FileExistsError(
                f"拒绝覆盖已有文件: {path}"
            )

    rng = random.Random(args.seed)

    total_start = time.perf_counter()

    print(
        "===== E2 GPU Hard Negative Mining =====",
        flush=True,
    )

    print(
        "加载已知正例...",
        flush=True,
    )

    positives, duplicate_count = read_qrels(
        args.qrels
    )

    print(
        "加载训练 Query...",
        flush=True,
    )

    pairs = load_pairs(
        args.pairs,
        args.max_queries,
    )

    if not pairs:
        raise RuntimeError(
            "训练 Query 为空"
        )

    print(
        "训练 Query:",
        f"{len(pairs):,}",
        flush=True,
    )

    verify_model_and_index(
        args.model,
        args.index,
        args.expected_model_hash,
    )

    print(
        "加载 B1 MiniLM...",
        flush=True,
    )

    model = SentenceTransformer(
        args.model,
        device=args.device,
        local_files_only=True,
    )

    dimension = (
        model.get_embedding_dimension()
    )

    (
        gpu_index,
        passage_ids,
        gpu_resources,
        cpu_index,
    ) = load_gpu_index(
        args.index,
        model_dimension=dimension,
        gpu_id=args.gpu_id,
    )

    # 用两条固定 Query 验证 CPU/GPU 的结果。
    validate_search(
        cpu_index=cpu_index,
        gpu_index=gpu_index,
        passage_ids=passage_ids,
        model=model,
        queries=[
            r["query"] for r in pairs[:2]
        ],
        top_k=min(args.top_k, 10),
    )

    # 一致性验证结束后释放 CPU 索引。
    del cpu_index

    selected = []
    required_pids = set()

    encode_seconds = 0.0
    search_seconds = 0.0
    filter_seconds = 0.0

    missing_candidates = 0

    print(
        "\n开始 GPU 检索...",
        flush=True,
    )

    for start in range(
        0,
        len(pairs),
        args.batch_size,
    ):
        batch = pairs[
            start:start + args.batch_size
        ]

        # Query 编码
        t0 = time.perf_counter()

        vectors = model.encode(
            [
                record["query"]
                for record in batch
            ],
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        encode_seconds += (
            time.perf_counter() - t0
        )

        # GPU FAISS 检索
        t0 = time.perf_counter()

        scores, ids = gpu_search(
            gpu_index,
            vectors,
            passage_ids,
            args.top_k,
        )

        search_seconds += (
            time.perf_counter() - t0
        )

        # 负例筛选
        t0 = time.perf_counter()

        for row, (
            record,
            doc_ids,
        ) in enumerate(
            zip(batch, ids)
        ):
            qid = record["query_id"]

            forbidden = positives.get(
                qid,
                set(),
            )

            candidates = []

            for rank, pid in enumerate(
                doc_ids,
                1,
            ):
                pid = int(pid)

                if rank < args.rank_min:
                    continue

                if pid < 0:
                    continue

                if pid == record["positive_id"]:
                    continue

                if pid in forbidden:
                    continue

                candidates.append((
                    rank,
                    pid,
                    float(
                        scores[row, rank - 1]
                    ),
                ))

            if not candidates:
                missing_candidates += 1
                continue

            (
                rank,
                negative_id,
                score,
            ) = rng.choice(candidates)

            selected.append({
                **record,
                "negative_id": negative_id,
                "negative_rank": rank,
                "negative_score": score,
            })

            required_pids.add(
                negative_id
            )

        filter_seconds += (
            time.perf_counter() - t0
        )

        processed = min(
            start + len(batch),
            len(pairs),
        )

        if (
            processed % 2048 < args.batch_size
            or processed == len(pairs)
        ):
            print(
                f"已检索 {processed:,}/"
                f"{len(pairs):,} "
                f"({processed / len(pairs) * 100:.2f}%)",
                flush=True,
            )

    print(
        "\nGPU 检索完成",
        flush=True,
    )

    print(
        "Query 编码时间:",
        f"{encode_seconds:.2f}s",
    )

    print(
        "GPU FAISS 搜索时间:",
        f"{search_seconds:.2f}s",
    )

    print(
        "负例筛选时间:",
        f"{filter_seconds:.2f}s",
    )

    print(
        "候选数量:",
        f"{len(selected):,}",
    )

    # 提取负例正文
    negative_texts = extract_passages(
        args.collection,
        required_pids,
    )

    written = 0
    skipped_empty = 0
    skipped_duplicate = 0

    # 写入临时文件，完成后再更名。
    with partial_path.open(
        "x",
        encoding="utf-8",
    ) as f:
        for record in selected:
            negative = negative_texts[
                record["negative_id"]
            ]

            query = record["query"]
            positive = record["positive"]

            if not negative.strip():
                skipped_empty += 1
                continue

            norm_negative = (
                negative.strip().casefold()
            )

            if norm_negative in (
                query.strip().casefold(),
                positive.strip().casefold(),
            ):
                skipped_duplicate += 1
                continue

            triplet = {
                "query_id": record["query_id"],
                "positive_id": record["positive_id"],
                "negative_id": record["negative_id"],
                "query": query,
                "positive": positive,
                "negative": negative,
                "negative_rank": record["negative_rank"],
                "negative_score": record["negative_score"],
            }

            f.write(
                json.dumps(
                    triplet,
                    ensure_ascii=False,
                ) + "\n"
            )

            written += 1

    if output.exists():
        raise FileExistsError(
            f"正式输出已存在: {output}"
        )

    partial_path.rename(output)

    elapsed = (
        time.perf_counter() - total_start
    )

    metadata = {
        "experiment_id": "E2-HN",
        "status": "mining_complete",
        "model": str(
            Path(args.model).resolve()
        ),
        "model_hash": args.expected_model_hash,
        "index": str(
            Path(args.index).resolve()
        ),
        "index_type": type(
            gpu_index
        ).__name__,
        "embedding_dim": int(dimension),
        "corpus_count": int(
            gpu_index.ntotal
        ),
        "query_count": len(pairs),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "top_k": args.top_k,
        "rank_min": args.rank_min,
        "negative_strategy": (
            "random rank-window sampling "
            "after excluding known positives"
        ),
        "candidate_count": len(selected),
        "triplet_count": written,
        "no_candidate_count": missing_candidates,
        "duplicate_qrels": duplicate_count,
        "skipped_empty": skipped_empty,
        "skipped_duplicate": skipped_duplicate,
        "query_encoding_seconds": encode_seconds,
        "gpu_search_seconds": search_seconds,
        "filter_seconds": filter_seconds,
        "total_seconds": elapsed,
        "mean_gpu_search_ms": (
            search_seconds / len(pairs) * 1000
        ),
        "output_sha256": sha256_file(
            output
        ),
        "warning": (
            "Unjudged retrieved passages "
            "may still be relevant."
        ),
    }

    meta_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n===== E2 挖掘完成 =====",
        flush=True,
    )

    print(
        "训练 Triplets:",
        f"{written:,}",
    )

    print(
        "总耗时:",
        f"{elapsed:.2f}s",
    )

    print(
        "输出文件:",
        output,
    )

    print(
        "元数据:",
        meta_path,
    )


if __name__ == "__main__":
    main()
