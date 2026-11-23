#用微调后 MiniLM 挖掘4条困难负样本
# Top 1–10 极困难负例
# Top 11–30 困难负例
# Top 31–60 中等困难负例
# Top 61–100 相对容易负例
import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


E2_HASH = (
    "33bc4d529d3f8c58fed5f7eddd6e94"
    "f598d39bf259fd1eb8ad34210d6fe34b21"
)

BUCKETS = [
    (1, 10),
    (11, 30),
    (31, 60),
    (61, 100),
]


def arguments():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        default="models/minilm-msmarco-e2-hn",
    )
    p.add_argument(
        "--index",
        default="data/indexes/faiss/minilm_e2_flatip.faiss",
    )
    p.add_argument(
        "--source",
        default="data/processed/msmarco/e2_triplets_gpu.jsonl",
    )
    p.add_argument(
        "--qrels",
        default="data/raw/msmarco/qrels.train.tsv",
    )
    p.add_argument(
        "--collection",
        default="data/raw/msmarco/collection.tsv",
    )
    p.add_argument(
        "--output",
        default="data/processed/msmarco/e4_e2_multi_negatives.jsonl",
    )
    p.add_argument("--max-queries", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--encode-batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--faiss-device",
        choices=["gpu", "cpu"],
        default="gpu",
    )

    return p.parse_args()


def load_source(path, max_queries, seed):
    """和此前 BCE 实验保持相同的蓄水池采样方式。"""
    rng = random.Random(seed)
    selected = []
    eligible = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            q = row["query"]
            pos = row["positive"]
            neg = row["negative"]

            if not q.strip() or not pos.strip() or not neg.strip():
                continue
            if pos.strip() == neg.strip():
                continue

            item = {
                "query_id": str(row["query_id"]),
                "query": q,
                "positive": pos,
            }

            eligible += 1

            if max_queries is None:
                selected.append(item)
            elif len(selected) < max_queries:
                selected.append(item)
            else:
                j = rng.randrange(eligible)
                if j < max_queries:
                    selected[j] = item

    if not selected:
        raise RuntimeError("没有有效训练样本")

    seen = set()
    for row in selected:
        qid = row["query_id"]
        if qid in seen:
            raise ValueError(
                f"存在重复 Query ID: {qid}，请先按 Query 去重"
            )
        seen.add(qid)

    return selected


def load_qrels(path, wanted_qids):
    """支持 MS MARCO 常见的三列或四列 Qrels。"""
    known = defaultdict(dict)

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) == 3:
                qid, pid, rel = parts
            elif len(parts) == 4:
                qid, _, pid, rel = parts
            else:
                raise ValueError(f"Qrels 格式错误: {line[:120]}")

            if qid not in wanted_qids:
                continue

            pid = int(pid)
            rel = int(rel)

            if pid in known[qid] and known[qid][pid] != rel:
                raise ValueError(
                    f"Qrels 标注冲突: qid={qid}, pid={pid}"
                )

            known[qid][pid] = rel

    # 仅排除 rel > 0 的已知正例。
    result = {
        qid: {
            pid for pid, rel in labels.items()
            if rel > 0
        }
        for qid, labels in known.items()
    }

    for qid in wanted_qids:
        if not result.get(qid):
            raise ValueError(
                f"Query {qid} 没有已知正例，拒绝挖掘"
            )

    return result


def read_index_metadata(path):
    index_path = Path(path)

    candidates = [
        Path(str(index_path) + ".meta.json"),
        index_path.with_suffix(".meta.json"),
        index_path.with_suffix(".json"),
    ]

    for candidate in candidates:
        if candidate.exists():
            metadata = json.loads(
                candidate.read_text(encoding="utf-8")
            )

            actual_hash = metadata.get(
                "embedding_model_hash"
            )

            if actual_hash != E2_HASH:
                raise ValueError(
                    "E2 索引哈希不匹配："
                    f"expected={E2_HASH}, actual={actual_hash}"
                )

            if not metadata.get("normalize_embeddings"):
                raise ValueError("索引不是归一化向量索引")

            return metadata

    raise FileNotFoundError("没有找到 FAISS 索引元数据")


def load_index(index_path, faiss_device, gpu_id):
    cpu = faiss.read_index(str(index_path))

    if not isinstance(cpu, faiss.IndexIDMap2):
        raise TypeError("索引必须是 IndexIDMap2")

    flat = faiss.downcast_index(cpu.index)

    if not isinstance(flat, faiss.IndexFlatIP):
        raise TypeError("底层索引必须是 IndexFlatIP")

    if faiss_device == "cpu":
        return cpu, None, None

    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError("当前 FAISS 不支持 GPU")

    if gpu_id >= faiss.get_num_gpus():
        raise RuntimeError("指定 GPU 不可用")

    passage_ids = faiss.vector_to_array(
        cpu.id_map
    ).astype(np.int64)

    resources = faiss.StandardGpuResources()
    options = faiss.GpuClonerOptions()
    options.useFloat16 = False

    gpu = faiss.index_cpu_to_gpu(
        resources,
        gpu_id,
        flat,
        options,
    )

    if gpu.ntotal != len(passage_ids):
        raise RuntimeError("GPU 向量数量与 ID 数量不一致")

    return gpu, passage_ids, resources


def faiss_search(index, vectors, id_map):
    scores, positions = index.search(
        np.ascontiguousarray(vectors, dtype=np.float32),
        100,
    )

    if id_map is None:
        return scores, positions

    valid = positions >= 0

    if np.any(positions[valid] >= len(id_map)):
        raise RuntimeError("FAISS 内部位置越界")

    pids = np.full(
        positions.shape, -1, dtype=np.int64
    )
    pids[valid] = id_map[positions[valid]]

    return scores, pids


def select_negatives(ids, scores, positives, seed):
    """每个排名区间优先随机选择一个未标注为正例的候选。"""
    rng = random.Random(seed)

    valid = []

    for rank, (pid, score) in enumerate(
        zip(ids, scores), 1
    ):
        pid = int(pid)

        if pid < 0 or pid in positives:
            continue

        if not np.isfinite(score):
            continue

        valid.append({
            "passage_id": pid,
            "rank": rank,
            "score": float(score),
        })

    chosen = []
    used = set()

    for lo, hi in BUCKETS:
        pool = [
            x for x in valid
            if lo <= x["rank"] <= hi
            and x["passage_id"] not in used
        ]

        if not pool:
            continue

        item = rng.choice(pool)
        chosen.append(item)
        used.add(item["passage_id"])

    # 某个区间全是已知正例时，从剩余排名中补足。
    leftovers = [
        x for x in valid
        if x["passage_id"] not in used
    ]
    rng.shuffle(leftovers)

    for item in leftovers:
        if len(chosen) >= 4:
            break

        if item["passage_id"] not in used:
            chosen.append(item)
            used.add(item["passage_id"])

    return sorted(chosen, key=lambda x: x["rank"])


def get_texts(collection, needed):
    """只提取选中的 Passage，不加载整个 Collection。"""
    texts = {}

    print(
        f"需要提取 {len(needed):,} 个不同 Passage",
        flush=True,
    )

    with open(collection, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            pid, sep, text = line.partition("\t")

            if not sep:
                raise ValueError(
                    f"collection 第 {i} 行无制表符"
                )

            pid = int(pid)

            if pid in needed:
                texts[pid] = text.rstrip("\r\n")

            if len(texts) == len(needed):
                break

    missing = needed - texts.keys()

    if missing:
        raise RuntimeError(
            f"有 {len(missing)} 个 Passage ID 缺少正文"
        )

    return texts


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(8 * 1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = arguments()

    if args.max_queries is not None and args.max_queries <= 0:
        raise ValueError("max-queries 必须大于 0")

    if args.batch_size <= 0 or args.encode_batch_size <= 0:
        raise ValueError("batch size 必须为正数")

    output = Path(args.output)
    metadata_output = Path(
        str(output).removesuffix(".jsonl") + ".meta.json"
    )

    if output.exists() or metadata_output.exists():
        raise FileExistsError("输出已经存在，拒绝覆盖")

    start_time = time.perf_counter()

    print("读取训练 Query...", flush=True)

    rows = load_source(
        args.source,
        args.max_queries,
        args.seed,
    )

    qrels = load_qrels(
        args.qrels,
        {r["query_id"] for r in rows},
    )

    index_metadata = read_index_metadata(args.index)

    model = SentenceTransformer(
        args.model,
        device=args.device,
        local_files_only=True,
    )

    index, id_map, resources = load_index(
        args.index,
        args.faiss_device,
        args.gpu_id,
    )

    if index.d != model.get_embedding_dimension():
        raise ValueError("模型与索引向量维度不一致")

    if index.ntotal != index_metadata["document_count"]:
        raise ValueError("索引文档数量与元数据不一致")

    required_pids = set()
    counts = Counter()

    print("使用 E2 检索 Top 100...", flush=True)
    search_start = time.perf_counter()

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]

        query_vectors = model.encode(
            [r["query"] for r in batch],
            batch_size=args.encode_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        scores, ids = faiss_search(
            index,
            query_vectors,
            id_map,
        )

        for row, query_scores, query_ids in zip(
            batch, scores, ids
        ):
            qid = row["query_id"]

            # 每个 Query 使用稳定的独立随机数种子。
            query_seed = int.from_bytes(
                hashlib.sha256(
                    f"{args.seed}:{qid}".encode()
                ).digest()[:8],
                "big",
            )

            negatives = select_negatives(
                query_ids,
                query_scores,
                qrels[qid],
                query_seed,
            )

            row["negatives"] = negatives
            counts[len(negatives)] += 1

            required_pids.update(
                x["passage_id"] for x in negatives
            )

        print(
            f"\r检索进度: "
            f"{start + len(batch):,}/{len(rows):,}",
            end="",
            flush=True,
        )

    search_seconds = time.perf_counter() - search_start
    print("\n开始提取 Passage...", flush=True)

    texts = get_texts(args.collection, required_pids)

    output.parent.mkdir(parents=True, exist_ok=True)

    # 先写临时文件，完整写入后才正式发布。
    temporary = Path(str(output) + ".partial")

    if temporary.exists():
        raise FileExistsError(temporary)

    valid_rows = 0

    with temporary.open("x", encoding="utf-8") as f:
        for row in rows:
            if not row["negatives"]:
                continue

            for neg in row["negatives"]:
                neg["text"] = texts[neg["passage_id"]]

            f.write(
                json.dumps(
                    row, ensure_ascii=False
                ) + "\n"
            )

            valid_rows += 1

    output_hash = sha256(temporary)

    temporary.rename(output)

    metadata = {
        "experiment_id": "E4-E2-MultiNeg",
        "source": args.source,
        "model": args.model,
        "expected_e2_model_hash": E2_HASH,
        "index": args.index,
        "index_metadata": index_metadata,
        "faiss_device": args.faiss_device,
        "seed": args.seed,
        "rank_buckets": BUCKETS,
        "requested_queries": len(rows),
        "output_queries": valid_rows,
        "negative_count_distribution": dict(counts),
        "unique_negative_passages": len(required_pids),
        "search_and_encoding_seconds": search_seconds,
        "total_seconds": time.perf_counter() - start_time,
        "output_sha256": output_hash,
    }

    metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== E2 多负例挖掘完成 =====")
    print(json.dumps({
        "output_queries": valid_rows,
        "negative_count_distribution": dict(counts),
        "unique_negative_passages": len(required_pids),
        "output_sha256": output_hash,
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
