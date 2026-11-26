
import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)

# 复用仓库已有的 Bi-Encoder 身份计算方法。
# 必须保持与已有 FAISS 索引的哈希算法一致。
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.indexing.encode_corpus import model_identity

PROTOCOL_VERSION = "v2"
CACHE_VERSION = "v1"

METRIC_NAMES = (
    "MRR@10",
    "NDCG@10",
    "Recall@10",
    "Recall@100",
)


# ============================================================
# 1. 通用工具
# ============================================================

def sha256_file(path):
    """分块计算文件 SHA256。"""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git_commit():
    """获取当前项目的 Git Commit。"""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return None


def git_dirty():
    """记录当前项目是否包含未提交修改。"""

    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        return bool(result.strip())

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return None


def save_json_atomic(path, data):
    """先写入临时文件，再提交正式 JSON 文件。"""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        raise FileExistsError(path)

    temporary = Path(str(path) + ".partial")

    with temporary.open(
        "x",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    if path.exists():
        raise FileExistsError(path)

    temporary.rename(path)


def directory_model_hash(model_dir):
    """
    计算模型目录身份。

    覆盖模型权重、模型配置、Tokenizer、
    SentenceTransformers 配置等。

    不包括训练日志和优化器状态。
    """

    model_dir = Path(model_dir).resolve()

    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    files = []

    ignored_names = {
        "train_result.json",
        "trainer_state.json",
        "training_args.bin",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "rng_state.pth",
        "README.md",
    }

    for path in model_dir.rglob("*"):

        if not path.is_file():
            continue

        if path.name in ignored_names:
            continue

        relative = path.relative_to(model_dir)

        if any(
            part.startswith("checkpoint-")
            for part in relative.parts
        ):
            continue

        is_weight = (
            path.suffix == ".safetensors"
            or path.name.startswith("pytorch_model")
            or path.name.startswith("model-")
        )

        is_config = path.suffix in {
            ".json",
            ".txt",
            ".model",
            ".vocab",
        }

        if is_weight or is_config:
            files.append(path)

    if not files:
        raise RuntimeError(
            f"模型目录缺少可校验文件: {model_dir}"
        )

    has_weights = any(
        path.suffix == ".safetensors"
        or path.name.startswith("pytorch_model")
        for path in files
    )

    if not has_weights:
        raise RuntimeError(
            f"模型目录缺少权重文件: {model_dir}"
        )

    digest = hashlib.sha256()

    for path in sorted(files):

        relative = path.relative_to(
            model_dir
        ).as_posix()

        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")

        digest.update(
            sha256_file(path).encode("ascii")
        )

        digest.update(b"\0")

    return digest.hexdigest()


# ============================================================
# 2. 数据集读取
# ============================================================

def read_queries(path):

    queries = {}

    with open(path, encoding="utf-8") as f:

        for line_no, line in enumerate(f, 1):

            if not line.strip():
                continue

            parts = line.rstrip(
                "\r\n"
            ).split("\t", 1)

            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{line_no} Query 格式错误"
                )

            qid = int(parts[0])
            text = parts[1]

            if not text.strip():
                raise ValueError(
                    f"Query {qid} 文本为空"
                )

            if qid in queries:
                raise ValueError(
                    f"重复 Query ID: {qid}"
                )

            queries[qid] = text

    if not queries:
        raise RuntimeError(
            f"Query 文件为空: {path}"
        )

    return queries


def read_qrels(path):

    qrels = defaultdict(dict)

    with open(path, encoding="utf-8") as f:

        for line_no, line in enumerate(f, 1):

            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 4:
                raise ValueError(
                    f"{path}:{line_no} Qrels 格式错误"
                )

            qid = int(parts[0])
            pid = int(parts[2])
            rel = int(parts[3])

            if qid < 0 or pid < 0:
                raise ValueError(
                    f"非法 Qrels ID: {qid}, {pid}"
                )

            if rel < 0:
                raise ValueError(
                    f"非法相关性标签: {rel}"
                )

            if pid in qrels[qid]:

                if qrels[qid][pid] != rel:
                    raise ValueError(
                        "Qrels 标签冲突: "
                        f"qid={qid}, pid={pid}"
                    )

            qrels[qid][pid] = rel

    if not qrels:
        raise RuntimeError(
            f"Qrels 文件为空: {path}"
        )

    return dict(qrels)


def select_queries(
    queries,
    qrels,
    max_queries,
    seed,
):
    """确定本次实验实际使用的 Query 集合。"""

    valid_qids = {
        qid
        for qid, relevance in qrels.items()
        if any(
            grade > 0
            for grade in relevance.values()
        )
    }

    missing = valid_qids - set(queries)

    if missing:
        raise RuntimeError(
            f"{len(missing)} 条标注 Query 缺少文本"
        )

    qids = sorted(valid_qids)

    if max_queries is not None:

        rng = random.Random(seed)

        qids = sorted(
            rng.sample(
                qids,
                min(max_queries, len(qids)),
            )
        )

    if not qids:
        raise RuntimeError(
            "没有可评估的 Query"
        )

    return qids


def extract_passages(path, required):
    """流式扫描语料库，只加载需要的候选正文。"""

    required = set(required)
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

            if pid not in required:
                continue

            text = text.rstrip("\r\n")

            if not text.strip():
                raise ValueError(
                    f"Passage {pid} 文本为空"
                )

            if pid in texts:
                raise ValueError(
                    f"重复 Passage ID: {pid}"
                )

            texts[pid] = text

            if len(texts) == len(required):
                break

    missing = required - set(texts)

    if missing:
        raise ValueError(
            f"缺少 {len(missing)} 篇候选正文"
        )

    return texts


# ============================================================
# 3. FAISS 索引与模型一致性
# ============================================================

def model_metadata(index_path):
    """读取 FAISS 索引元数据。"""

    path = Path(index_path)

    candidates = [
        Path(str(path) + ".meta.json"),
        path.with_suffix(".meta.json"),
        path.with_suffix(".json"),
    ]

    for candidate in candidates:

        if candidate.is_file():

            return json.loads(
                candidate.read_text(
                    encoding="utf-8"
                )
            )

    raise FileNotFoundError(
        f"未找到 FAISS 索引元数据: {path}"
    )


def verify_model_and_index(
    model_dir,
    index_path,
):
    """
    检查当前 Bi-Encoder 与索引构建模型一致。

    使用仓库已有的 model_identity，
    避免与旧 FAISS 元数据哈希算法不兼容。
    """

    metadata = model_metadata(index_path)

    expected_hash = metadata.get(
        "embedding_model_hash"
    )

    if not expected_hash:
        raise RuntimeError(
            "索引元数据缺少 embedding_model_hash"
        )

    actual_hash = model_identity(
        Path(model_dir).resolve()
    )

    if actual_hash != expected_hash:
        raise RuntimeError(
            "Bi-Encoder 与 FAISS 索引模型不一致\n"
            f"expected={expected_hash}\n"
            f"actual={actual_hash}"
        )

    if metadata.get(
        "normalize_embeddings"
    ) is not True:

        raise RuntimeError(
            "FAISS 索引向量未归一化"
        )

    expected_type = "IndexIDMap2(IndexFlatIP)"

    if metadata.get("index_type") != expected_type:

        raise RuntimeError(
            f"索引类型不是 {expected_type}"
        )

    return actual_hash, metadata


def load_faiss(
    index_path,
    device,
    gpu_id,
):

    cpu_index = faiss.read_index(
        str(index_path)
    )

    if not isinstance(
        cpu_index,
        faiss.IndexIDMap2,
    ):

        raise TypeError(
            "索引必须为 IndexIDMap2"
        )

    flat = faiss.downcast_index(
        cpu_index.index
    )

    if not isinstance(
        flat,
        faiss.IndexFlatIP,
    ):

        raise TypeError(
            "底层索引必须为 IndexFlatIP"
        )

    if device == "cpu":
        return cpu_index, None, None

    if not hasattr(
        faiss,
        "StandardGpuResources",
    ):

        raise RuntimeError(
            "当前 FAISS 不支持 GPU"
        )

    if not 0 <= gpu_id < faiss.get_num_gpus():
        raise RuntimeError(
            "指定 GPU 不可用"
        )

    ids = faiss.vector_to_array(
        cpu_index.id_map
    ).astype(np.int64)

    resources = faiss.StandardGpuResources()

    options = faiss.GpuClonerOptions()

    options.useFloat16 = False

    gpu_index = faiss.index_cpu_to_gpu(
        resources,
        gpu_id,
        flat,
        options,
    )

    if gpu_index.ntotal != len(ids):
        raise RuntimeError(
            "GPU 索引与 ID 映射数量不一致"
        )

    print(
        "FAISS GPU 类型:",
        type(gpu_index).__name__,
        flush=True,
    )

    return gpu_index, ids, resources


def search(
    index,
    vectors,
    top_k,
    id_map=None,
):

    vectors = np.ascontiguousarray(
        vectors,
        dtype=np.float32,
    )

    if not np.isfinite(vectors).all():
        raise RuntimeError(
            "Query Embedding 存在非法值"
        )

    scores, ids = index.search(
        vectors,
        top_k,
    )

    # GPU Flat 索引返回内部位置，
    # 需要映射为真正的 Passage ID。
    if id_map is not None:

        positions = ids

        ids = np.full(
            positions.shape,
            -1,
            dtype=np.int64,
        )

        valid = positions >= 0

        if np.any(
            positions[valid] >= len(id_map)
        ):

            raise RuntimeError(
                "FAISS 内部位置越界"
            )

        ids[valid] = id_map[
            positions[valid]
        ]

    return scores, ids


# ============================================================
# 4. 候选缓存
# ============================================================

def cache_metadata_path(cache_path):

    return Path(
        str(cache_path) + ".meta.json"
    )


def validate_candidate_record(
    row,
    queries,
    top_k,
):
    """验证单条 Query 的候选记录。"""

    qid = row["query_id"]

    ids = row["candidate_ids"]

    scores = row["retrieval_scores"]

    if type(qid) is not int:
        raise ValueError(
            f"非法 Query ID: {qid}"
        )

    if qid not in queries:
        raise ValueError(
            f"未知 Query ID: {qid}"
        )

    if row["query"] != queries[qid]:
        raise ValueError(
            f"Query 文本不一致: {qid}"
        )

    if len(ids) != top_k:
        raise ValueError(
            f"候选数量错误: {qid}"
        )

    if len(scores) != top_k:
        raise ValueError(
            f"召回分数数量错误: {qid}"
        )

    if any(
        type(pid) is not int or pid < 0
        for pid in ids
    ):

        raise ValueError(
            f"非法 Passage ID: {qid}"
        )

    if len(set(ids)) != len(ids):

        raise ValueError(
            f"重复 Passage ID: {qid}"
        )

    if any(
        not math.isfinite(score)
        for score in scores
    ):

        raise ValueError(
            f"非法召回分数: {qid}"
        )

    # 检查原始召回结果是否按降序排列。
    for previous, current in zip(
        scores,
        scores[1:],
    ):

        if current > previous + 1e-6:

            raise ValueError(
                f"召回分数顺序异常: {qid}"
            )

    return qid


def expected_cache_metadata(
    args,
    queries_path,
    qrels_path,
    collection_path,
    qids,
):
    """
    生成当前评估所要求的缓存身份。

    此处只读取文件及计算哈希，
    不加载 Bi-Encoder 或 FAISS 到显存。
    """

    biencoder_hash, index_metadata = (
        verify_model_and_index(
            args.biencoder,
            args.index,
        )
    )

    print(
        "校验模型、索引及数据文件 SHA256...",
        flush=True,
    )

    qid_bytes = json.dumps(
        qids,
        separators=(",", ":"),
    ).encode("utf-8")

    metadata = {
        "cache_version": CACHE_VERSION,

        "split": args.split,

        "query_count": len(qids),

        "query_ids_hash": hashlib.sha256(
            qid_bytes
        ).hexdigest(),

        "top_k": args.top_k,

        "seed": args.seed,

        "normalize_embeddings": True,

        "biencoder_hash": biencoder_hash,

        # 额外覆盖 Tokenizer 等配置。
        # 不替换原索引使用的哈希算法。
        "biencoder_full_hash": (
            directory_model_hash(
                args.biencoder
            )
        ),

        "index_hash": sha256_file(
            args.index
        ),

        "collection_hash": sha256_file(
            collection_path
        ),

        "queries_hash": sha256_file(
            queries_path
        ),

        "qrels_hash": sha256_file(
            qrels_path
        ),
    }

    return metadata, index_metadata


def build_retrieval_candidates(
    args,
    queries,
    qids,
    index_metadata,
):
    """执行 E2 全库召回。"""

    print(
        "加载 E2 Bi-Encoder...",
        flush=True,
    )

    biencoder = SentenceTransformer(
        args.biencoder,
        device=args.device,
        local_files_only=True,
    )

    print(
        "加载 FAISS...",
        flush=True,
    )

    index, id_map, resources = load_faiss(
        args.index,
        args.faiss_device,
        args.gpu_id,
    )

    if index.d != (
        biencoder.get_embedding_dimension()
    ):

        raise RuntimeError(
            "Bi-Encoder 与 FAISS 维度不同"
        )

    if index.d != int(
        index_metadata["embedding_dim"]
    ):

        raise RuntimeError(
            "FAISS 维度与元数据不一致"
        )

    if index.ntotal != int(
        index_metadata["document_count"]
    ):

        raise RuntimeError(
            "FAISS 文档数量与元数据不一致"
        )

    if index.ntotal < args.top_k:
        raise RuntimeError(
            "索引文档数量小于 top-k"
        )

    corpus_count = int(index.ntotal)

    records = []

    query_encode_seconds = 0.0

    faiss_search_seconds = 0.0

    print(
        "开始 E2 全库召回...",
        flush=True,
    )

    for start in range(
        0,
        len(qids),
        args.query_batch_size,
    ):

        batch_qids = qids[
            start:
            start + args.query_batch_size
        ]

        t0 = time.perf_counter()

        vectors = biencoder.encode(
            [
                queries[qid]
                for qid in batch_qids
            ],
            batch_size=args.query_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        query_encode_seconds += (
            time.perf_counter() - t0
        )

        t0 = time.perf_counter()

        scores, doc_ids = search(
            index,
            vectors,
            args.top_k,
            id_map,
        )

        faiss_search_seconds += (
            time.perf_counter() - t0
        )

        for qid, ids, row_scores in zip(
            batch_qids,
            doc_ids,
            scores,
        ):

            valid_pairs = [
                (
                    int(pid),
                    float(score),
                )
                for pid, score in zip(
                    ids,
                    row_scores,
                )
                if pid >= 0
            ]

            record = {
                "query_id": qid,

                "query": queries[qid],

                "candidate_ids": [
                    pid
                    for pid, _ in valid_pairs
                ],

                "retrieval_scores": [
                    score
                    for _, score in valid_pairs
                ],
            }

            validate_candidate_record(
                record,
                queries,
                args.top_k,
            )

            records.append(record)

        print(
            f"\r召回 "
            f"{start + len(batch_qids):,}"
            f"/{len(qids):,}",
            end="",
            flush=True,
        )

    print(
        "\nE2 召回完成",
        flush=True,
    )

    statistics = {
        "query_encoding_seconds": (
            query_encode_seconds
        ),

        "faiss_search_seconds": (
            faiss_search_seconds
        ),
    }

    retrieval_configuration = {
        "query_batch_size": (
            args.query_batch_size
        ),

        "biencoder_max_seq_length": (
            biencoder.max_seq_length
        ),

        "device": args.device,

        "faiss_device": args.faiss_device,

        "faiss_gpu_id": args.gpu_id,

        "faiss_version": faiss.__version__,

        "use_float16_index": False,
    }

    # 释放召回阶段资源，
    # 不让 FAISS 与 Cross-Encoder 同时占据显存。
    del index
    del id_map
    del resources
    del biencoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        records,
        corpus_count,
        statistics,
        retrieval_configuration,
    )


def save_candidate_cache(
    cache_path,
    records,
    metadata,
    queries,
    qids,
):

    cache_path = Path(cache_path)

    meta_path = cache_metadata_path(
        cache_path
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if cache_path.exists() or meta_path.exists():

        raise FileExistsError(
            "候选缓存或元数据已存在，"
            "请使用新路径"
        )

    temporary = Path(
        str(cache_path) + ".partial"
    )

    temporary_meta = Path(
        str(meta_path) + ".partial"
    )

    seen = set()

    with temporary.open(
        "x",
        encoding="utf-8",
    ) as f:

        for row in records:

            qid = validate_candidate_record(
                row,
                queries,
                metadata["top_k"],
            )

            if qid in seen:
                raise ValueError(
                    f"重复缓存 Query: {qid}"
                )

            seen.add(qid)

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

    if seen != set(qids):

        raise RuntimeError(
            "候选缓存 Query 集合不完整"
        )

    metadata = dict(metadata)

    metadata["candidate_cache_hash"] = (
        sha256_file(temporary)
    )

    metadata["query_count"] = len(seen)

    with temporary_meta.open(
        "x",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    if cache_path.exists() or meta_path.exists():

        raise FileExistsError(
            "缓存输出路径已存在"
        )

    temporary.rename(cache_path)

    temporary_meta.rename(meta_path)

    print(
        "候选缓存已保存:",
        cache_path,
        flush=True,
    )

    print(
        "缓存元数据已保存:",
        meta_path,
        flush=True,
    )

    return metadata


def load_candidate_cache(
    cache_path,
    expected_metadata,
    queries,
    qids,
):

    cache_path = Path(cache_path)

    meta_path = cache_metadata_path(
        cache_path
    )

    if not cache_path.is_file():

        raise FileNotFoundError(
            f"候选缓存不存在: {cache_path}\n"
            "请先执行 --build-candidate-cache"
        )

    if not meta_path.is_file():

        raise FileNotFoundError(
            f"缓存元数据不存在: {meta_path}"
        )

    metadata = json.loads(
        meta_path.read_text(
            encoding="utf-8"
        )
    )

    # 校验模型、索引、数据及 Query 集合身份。
    for key, expected in (
        expected_metadata.items()
    ):

        actual = metadata.get(key)

        if actual != expected:

            raise RuntimeError(
                f"候选缓存元数据不一致: {key}\n"
                f"expected={expected}\n"
                f"actual={actual}"
            )

    actual_cache_hash = sha256_file(
        cache_path
    )

    expected_cache_hash = metadata.get(
        "candidate_cache_hash"
    )

    if actual_cache_hash != expected_cache_hash:

        raise RuntimeError(
            "候选缓存 SHA256 校验失败"
        )

    if int(
        metadata.get("corpus_count", 0)
    ) < expected_metadata["top_k"]:

        raise RuntimeError(
            "缓存语料数量非法"
        )

    retrieval = {}

    retrieval_scores = {}

    with cache_path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            row = json.loads(line)

            qid = validate_candidate_record(
                row,
                queries,
                metadata["top_k"],
            )

            if qid in retrieval:

                raise ValueError(
                    f"缓存存在重复 Query: {qid}"
                )

            retrieval[qid] = row[
                "candidate_ids"
            ]

            retrieval_scores[qid] = row[
                "retrieval_scores"
            ]

    if set(retrieval) != set(qids):

        raise RuntimeError(
            "缓存与当前评估 Query 集合不一致"
        )

    if len(retrieval) != int(
        metadata["query_count"]
    ):

        raise RuntimeError(
            "缓存 Query 数量不一致"
        )

    print(
        "候选缓存校验通过",
        flush=True,
    )

    print(
        "跳过 E2 模型及 FAISS 索引加载",
        flush=True,
    )

    return (
        retrieval,
        retrieval_scores,
        metadata,
    )


# ============================================================
# 5. 排序指标
# ============================================================

def score_query(
    ranked_ids,
    relevance,
):
    """计算单条 Query 的排序指标。"""

    if len(ranked_ids) != len(
        set(ranked_ids)
    ):

        raise ValueError(
            "排名中存在重复 Passage ID"
        )

    positives = {
        pid: grade
        for pid, grade in relevance.items()
        if grade > 0
    }

    if not positives:

        raise ValueError(
            "评估 Query 没有正例"
        )

    positive_ids = set(positives)

    total = len(positive_ids)

    top10 = ranked_ids[:10]

    top100 = ranked_ids[:100]

    # ------------------------
    # MRR@10
    # ------------------------

    rr = 0.0

    for rank, pid in enumerate(
        top10,
        1,
    ):

        if pid in positive_ids:

            rr = 1.0 / rank

            break

    # ------------------------
    # NDCG@10
    # ------------------------

    dcg = sum(
        (
            2 ** positives.get(pid, 0) - 1
        )
        / math.log2(rank + 1)

        for rank, pid in enumerate(
            top10,
            1,
        )
    )

    ideal_grades = sorted(
        positives.values(),
        reverse=True,
    )[:10]

    idcg = sum(
        (2 ** grade - 1)
        / math.log2(rank + 1)

        for rank, grade in enumerate(
            ideal_grades,
            1,
        )
    )

    ndcg = (
        dcg / idcg
        if idcg > 0
        else 0.0
    )

    # ------------------------
    # Recall
    # ------------------------

    recall10 = (
        len(
            set(top10) & positive_ids
        )
        / total
    )

    recall100 = (
        len(
            set(top100) & positive_ids
        )
        / total
    )

    metrics = {
        "MRR@10": rr,

        "NDCG@10": ndcg,

        "Recall@10": recall10,

        "Recall@100": recall100,
    }

    for name, value in metrics.items():

        if not math.isfinite(value):

            raise RuntimeError(
                f"非法指标: {name}={value}"
            )

        if not -1e-9 <= value <= 1 + 1e-9:

            raise RuntimeError(
                f"指标越界: {name}={value}"
            )

    return metrics


def average_metrics(rows):

    if not rows:

        raise ValueError(
            "没有可计算的 Query 指标"
        )

    return {
        name: sum(
            row[name]
            for row in rows
        ) / len(rows)

        for name in METRIC_NAMES
    }


def first_positive_rank(
    ranked_ids,
    relevance,
):
    """获取第一个已标注正例的排名。"""

    for rank, pid in enumerate(
        ranked_ids,
        1,
    ):

        if relevance.get(pid, 0) > 0:

            return rank

    return None


# ============================================================
# 6. Cross-Encoder 精排
# ============================================================

def evaluate_cross_encoder(
    args,
    queries,
    qrels,
    qids,
    retrieval,
    retrieval_scores,
    texts,
    cache_metadata,
    cache_hit,
    current_statistics,
    output,
    rankings_output,
):

    print(
        "计算 Cross-Encoder 模型身份...",
        flush=True,
    )

    reranker_hash = directory_model_hash(
        args.reranker
    )

    print(
        "加载 Cross-Encoder...",
        flush=True,
    )

    reranker = CrossEncoder(
        args.reranker,
        device=args.device,
        max_length=args.max_length,
        activation_fn=torch.nn.Identity(),
        local_files_only=True,
    )

    if reranker.max_length != args.max_length:

        raise RuntimeError(
            "Cross-Encoder 实际最大长度"
            "与指定长度不一致"
        )

    print(
        "Cross-Encoder max_length:",
        reranker.max_length,
        flush=True,
    )

    print(
        "Cross-Encoder activation:",
        "Identity (raw logits)",
        flush=True,
    )

    before = []

    after = []

    rerank_seconds = 0.0

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rankings_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    partial_rankings = Path(
        str(rankings_output) + ".partial"
    )

    if partial_rankings.exists():

        raise FileExistsError(
            partial_rankings
        )

    try:

        with partial_rankings.open(
            "x",
            encoding="utf-8",
        ) as f:

            for i, qid in enumerate(
                qids,
                1,
            ):

                doc_ids = retrieval[qid]

                original_scores = (
                    retrieval_scores[qid]
                )

                pairs = [
                    [
                        queries[qid],
                        texts[pid],
                    ]

                    for pid in doc_ids
                ]

                t0 = time.perf_counter()

                scores = reranker.predict(
                    pairs,
                    batch_size=(
                        args.rerank_batch_size
                    ),
                    activation_fn=(
                        torch.nn.Identity()
                    ),
                    show_progress_bar=False,
                )

                rerank_seconds += (
                    time.perf_counter() - t0
                )

                scores = np.asarray(scores)

                # 单标签 Cross-Encoder 应当为
                # (N,) 或 (N, 1)。
                if scores.shape == (
                    len(doc_ids),
                    1,
                ):

                    scores = scores[:, 0]

                if scores.shape != (
                    len(doc_ids),
                ):

                    raise RuntimeError(
                        "Cross-Encoder 分数形状错误: "
                        f"{scores.shape}"
                    )

                if not np.isfinite(
                    scores
                ).all():

                    raise RuntimeError(
                        "Cross-Encoder 出现非法分数"
                    )

                # 保持原有稳定降序排序逻辑。
                order = np.argsort(
                    -scores,
                    kind="stable",
                )

                reranked = [
                    doc_ids[int(j)]
                    for j in order
                ]

                reranked_scores = [
                    float(scores[int(j)])
                    for j in order
                ]

                # 精排只能重新排列原始候选，
                # 不允许添加或删除候选。
                if len(reranked) != len(doc_ids):

                    raise RuntimeError(
                        f"Query {qid} 精排数量改变"
                    )

                if set(reranked) != set(doc_ids):

                    raise RuntimeError(
                        f"Query {qid} 精排候选集合改变"
                    )

                retrieval_metrics = (
                    score_query(
                        doc_ids,
                        qrels[qid],
                    )
                )

                reranked_metrics = (
                    score_query(
                        reranked,
                        qrels[qid],
                    )
                )

                # Top 100 精排不应改变 Recall@100。
                if args.top_k == 100:

                    if not math.isclose(
                        retrieval_metrics[
                            "Recall@100"
                        ],
                        reranked_metrics[
                            "Recall@100"
                        ],
                        abs_tol=1e-12,
                        rel_tol=0.0,
                    ):

                        raise RuntimeError(
                            f"Query {qid} "
                            "Recall@100 异常改变"
                        )

                before.append(
                    retrieval_metrics
                )

                after.append(
                    reranked_metrics
                )

                record = {
                    "query_id": qid,

                    "query": queries[qid],

                    "relevant_passage_ids": sorted(
                        pid
                        for pid, grade in (
                            qrels[qid].items()
                        )
                        if grade > 0
                    ),

                    # 完整第一阶段候选
                    "retrieved_ids": doc_ids,

                    "retrieval_scores": (
                        original_scores
                    ),

                    # 完整精排结果
                    "reranked_ids": reranked,

                    "reranked_scores": (
                        reranked_scores
                    ),

                    # 保留旧字段，
                    # 兼容已有错误案例分析脚本。
                    "retrieved_top10": (
                        doc_ids[:10]
                    ),

                    "reranked_top10": (
                        reranked[:10]
                    ),

                    "reranked_scores_top10": (
                        reranked_scores[:10]
                    ),

                    # 逐 Query 指标
                    "retrieval_metrics": (
                        retrieval_metrics
                    ),

                    "reranked_metrics": (
                        reranked_metrics
                    ),

                    "retrieval_first_positive_rank": (
                        first_positive_rank(
                            doc_ids,
                            qrels[qid],
                        )
                    ),

                    "reranked_first_positive_rank": (
                        first_positive_rank(
                            reranked,
                            qrels[qid],
                        )
                    ),

                    "improvement": {
                        key: (
                            reranked_metrics[key]
                            - retrieval_metrics[key]
                        )

                        for key in METRIC_NAMES
                    },
                }

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )

                if i % 100 == 0 or i == len(qids):

                    print(
                        f"已重排 "
                        f"{i:,}/{len(qids):,}",
                        flush=True,
                    )

        baseline_metrics = average_metrics(
            before
        )

        reranked_metrics = average_metrics(
            after
        )

        results = {
            "evaluation_protocol": (
                PROTOCOL_VERSION
            ),

            "experiment_id": (
                args.experiment_id
            ),

            "stage": (
                "E2 Bi-Encoder + "
                "Cross-Encoder Reranking"
            ),

            "git_commit": git_commit(),

            "git_dirty": git_dirty(),

            "biencoder": str(
                Path(args.biencoder).resolve()
            ),

            "biencoder_hash": (
                cache_metadata[
                    "biencoder_hash"
                ]
            ),

            "biencoder_full_hash": (
                cache_metadata[
                    "biencoder_full_hash"
                ]
            ),

            "reranker": str(
                Path(args.reranker).resolve()
            ),

            "reranker_hash": (
                reranker_hash
            ),

            "reranker_max_length": (
                reranker.max_length
            ),

            "reranker_activation": (
                "Identity"
            ),

            "reranker_device": (
                args.device
            ),

            "index": str(
                Path(args.index).resolve()
            ),

            "index_hash": (
                cache_metadata[
                    "index_hash"
                ]
            ),

            "index_metadata": (
                cache_metadata[
                    "index_metadata"
                ]
            ),

            "faiss_device": (
                cache_metadata[
                    "retrieval_configuration"
                ]["faiss_device"]
            ),

            "candidate_cache": str(
                args.candidate_cache.resolve()
            ),

            "candidate_cache_hash": (
                cache_metadata[
                    "candidate_cache_hash"
                ]
            ),

            "candidate_cache_hit": (
                cache_hit
            ),

            "candidate_cache_version": (
                cache_metadata[
                    "cache_version"
                ]
            ),

            "collection_hash": (
                cache_metadata[
                    "collection_hash"
                ]
            ),

            "queries_hash": (
                cache_metadata[
                    "queries_hash"
                ]
            ),

            "qrels_hash": (
                cache_metadata[
                    "qrels_hash"
                ]
            ),

            "query_ids_hash": (
                cache_metadata[
                    "query_ids_hash"
                ]
            ),

            "split": args.split,

            "seed": args.seed,

            "query_count": len(qids),

            "corpus_count": int(
                cache_metadata[
                    "corpus_count"
                ]
            ),

            "top_k": args.top_k,

            "query_batch_size": (
                cache_metadata[
                    "retrieval_configuration"
                ]["query_batch_size"]
            ),

            "rerank_batch_size": (
                args.rerank_batch_size
            ),

            "query_encoding_seconds": (
                current_statistics[
                    "query_encoding_seconds"
                ]
            ),

            "faiss_search_seconds": (
                current_statistics[
                    "faiss_search_seconds"
                ]
            ),

            # 保留缓存首次构建时的实际耗时。
            "candidate_build_statistics": (
                cache_metadata[
                    "retrieval_statistics"
                ]
            ),

            "retrieval_configuration": (
                cache_metadata[
                    "retrieval_configuration"
                ]
            ),

            "rerank_seconds": (
                rerank_seconds
            ),

            "mean_rerank_ms_per_query": (
                rerank_seconds
                / len(qids)
                * 1000
            ),

            "first_stage_metrics": (
                baseline_metrics
            ),

            "reranked_metrics": (
                reranked_metrics
            ),

            "improvement": {
                key: (
                    reranked_metrics[key]
                    - baseline_metrics[key]
                )

                for key in METRIC_NAMES
            },

            "rankings_output": str(
                rankings_output.resolve()
            ),

            "rankings_sha256": (
                sha256_file(
                    partial_rankings
                )
            ),
        }

        # 排名文件写完后再提交实验报告。
        if rankings_output.exists():

            raise FileExistsError(
                rankings_output
            )

        partial_rankings.rename(
            rankings_output
        )

        save_json_atomic(
            output,
            results,
        )

    except Exception:

        print(
            "评估失败，检查临时排名文件 "
            "或已经生成的排名文件:",
            partial_rankings,
            rankings_output,
            flush=True,
        )

        raise

    print(
        f"\n========== "
        f"{args.experiment_id} 结果 =========="
    )

    print(
        "E2 原始召回指标:",
        baseline_metrics,
    )

    print(
        "Cross-Encoder 重排序指标:",
        reranked_metrics,
    )

    print(
        "重排序耗时:",
        rerank_seconds,
    )

    print(
        "候选缓存 SHA256:",
        cache_metadata[
            "candidate_cache_hash"
        ],
    )

    print(
        "实验结果:",
        output,
    )

    print(
        "完整排名:",
        rankings_output,
    )

    return results


# ============================================================
# 7. 命令行参数
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description=(
            "MS MARCO Cross-Encoder "
            "统一精排评估协议 v2"
        )
    )

    p.add_argument(
        "--experiment-id",
        default="E3-v2",
    )

    p.add_argument(
        "--biencoder",
        default=(
            "models/minilm-msmarco-e2-hn"
        ),
    )

    p.add_argument(
        "--reranker",
        default=(
            "models/"
            "ms-marco-MiniLM-L6-v2-cross-encoder"
        ),
    )

    p.add_argument(
        "--index",
        default=(
            "data/indexes/faiss/"
            "minilm_e2_flatip.faiss"
        ),
    )

    p.add_argument(
        "--data-dir",
        default="data/raw/msmarco",
    )

    p.add_argument(
        "--queries-file",
        type=Path,
        default=None,
    )

    p.add_argument(
        "--qrels-file",
        type=Path,
        default=None,
    )

    p.add_argument(
        "--collection-file",
        type=Path,
        default=None,
    )

    p.add_argument(
        "--split",
        default="dev.small",
    )

    p.add_argument(
        "--candidate-cache",
        type=Path,
        default=Path(
            "data/processed/msmarco/"
            "E2_dev_small_top100.jsonl"
        ),
    )

    p.add_argument(
        "--build-candidate-cache",
        action="store_true",
        help="重新执行 E2 召回并构建新缓存",
    )

    p.add_argument(
        "--cache-only",
        action="store_true",
        help="只构建或验证候选缓存，不执行精排",
    )

    p.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    p.add_argument(
        "--rankings-output",
        type=Path,
        default=None,
    )

    p.add_argument(
        "--top-k",
        type=int,
        default=100,
    )

    p.add_argument(
        "--max-queries",
        type=int,
        default=None,
    )

    p.add_argument(
        "--query-batch-size",
        type=int,
        default=128,
    )

    p.add_argument(
        "--rerank-batch-size",
        type=int,
        default=64,
    )

    p.add_argument(
        "--max-length",
        type=int,
        default=256,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--device",
        default="cuda",
    )

    p.add_argument(
        "--faiss-device",
        choices=["cpu", "gpu"],
        default="gpu",
    )

    p.add_argument(
        "--gpu-id",
        type=int,
        default=0,
    )

    args = p.parse_args()

    if args.top_k < 100:

        p.error(
            "--top-k 必须至少为 100"
        )

    if args.query_batch_size <= 0:

        p.error(
            "--query-batch-size 必须为正数"
        )

    if args.rerank_batch_size <= 0:

        p.error(
            "--rerank-batch-size 必须为正数"
        )

    if args.max_length <= 0:

        p.error(
            "--max-length 必须为正数"
        )

    if args.gpu_id < 0:

        p.error(
            "--gpu-id 不能小于 0"
        )

    if (
        args.max_queries is not None
        and args.max_queries <= 0
    ):

        p.error(
            "--max-queries 必须为正数"
        )

    if not args.cache_only:

        if args.output is None:

            args.output = Path(
                "reports/experiments/v2"
            ) / (
                args.experiment_id + ".json"
            )

        if args.rankings_output is None:

            args.rankings_output = Path(
                "data/processed/msmarco/v2"
            ) / (
                args.experiment_id + ".jsonl"
            )

    return args


# ============================================================
# 8. 主流程
# ============================================================

def main():

    args = parse_args()

    random.seed(args.seed)

    torch.manual_seed(args.seed)

    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)

    queries_path = (
        args.queries_file
        if args.queries_file is not None
        else data_dir / "queries.dev.small.tsv"
    )

    qrels_path = (
        args.qrels_file
        if args.qrels_file is not None
        else data_dir / "qrels.dev.small.tsv"
    )

    collection_path = (
        args.collection_file
        if args.collection_file is not None
        else data_dir / "collection.tsv"
    )

    if args.split != "dev.small":

        if (
            args.queries_file is None
            or args.qrels_file is None
        ):

            raise ValueError(
                "自定义 split 必须显式提供 "
                "--queries-file 和 --qrels-file"
            )

    for path in (
        queries_path,
        qrels_path,
        collection_path,
        Path(args.index),
    ):

        if not path.is_file():

            raise FileNotFoundError(path)

    if not args.cache_only:

        if not Path(
            args.reranker
        ).is_dir():

            raise FileNotFoundError(
                args.reranker
            )

        output = args.output

        rankings_output = (
            args.rankings_output
        )

        if (
            output.resolve()
            == rankings_output.resolve()
        ):

            raise ValueError(
                "实验报告和排名输出不能使用相同路径"
            )

        protected_paths = {
            queries_path.resolve(),
            qrels_path.resolve(),
            collection_path.resolve(),
            Path(args.index).resolve(),
            args.candidate_cache.resolve(),
            cache_metadata_path(
                args.candidate_cache
            ).resolve(),
        }

        for path in (
            output,
            rankings_output,
        ):

            if path.resolve() in protected_paths:

                raise ValueError(
                    f"输出路径与输入文件冲突: {path}"
                )

            if path.exists():

                raise FileExistsError(
                    f"输出文件已存在: {path}"
                )

            if Path(
                str(path) + ".partial"
            ).exists():

                raise FileExistsError(
                    f"存在未完成的输出文件: {path}"
                )

    cache_path = args.candidate_cache

    meta_path = cache_metadata_path(
        cache_path
    )

    if args.build_candidate_cache:

        protected_inputs = {
            queries_path.resolve(),
            qrels_path.resolve(),
            collection_path.resolve(),
            Path(args.index).resolve(),
        }

        if (
            cache_path.resolve()
            in protected_inputs

            or meta_path.resolve()
            in protected_inputs
        ):

            raise ValueError(
                "候选缓存路径与输入文件冲突"
            )

        if cache_path.exists() or meta_path.exists():

            raise FileExistsError(
                "候选缓存已存在，"
                "禁止覆盖，请指定新的缓存路径"
            )

        if (
            Path(
                str(cache_path) + ".partial"
            ).exists()

            or Path(
                str(meta_path) + ".partial"
            ).exists()
        ):

            raise FileExistsError(
                "存在未完成的候选缓存"
            )

    # ------------------------
    # 读取评估数据
    # ------------------------

    queries = read_queries(
        queries_path
    )

    qrels = read_qrels(
        qrels_path
    )

    qids = select_queries(
        queries,
        qrels,
        args.max_queries,
        args.seed,
    )

    print(
        "Experiment:",
        args.experiment_id,
        flush=True,
    )

    print(
        "Protocol:",
        PROTOCOL_VERSION,
        flush=True,
    )

    print(
        "Split:",
        args.split,
        flush=True,
    )

    print(
        "Query 数量:",
        len(qids),
        flush=True,
    )

    print(
        "Top K:",
        args.top_k,
        flush=True,
    )

    # ------------------------
    # 构建缓存身份
    # ------------------------

    expected_metadata, index_metadata = (
        expected_cache_metadata(
            args,
            queries_path,
            qrels_path,
            collection_path,
            qids,
        )
    )

    # ------------------------
    # 构建候选缓存
    # ------------------------

    if args.build_candidate_cache:

        (
            records,
            corpus_count,
            current_statistics,
            retrieval_configuration,
        ) = build_retrieval_candidates(
            args,
            queries,
            qids,
            index_metadata,
        )

        cache_metadata = save_candidate_cache(
            cache_path=cache_path,

            records=records,

            metadata={
                **expected_metadata,

                "index_metadata": (
                    index_metadata
                ),

                "corpus_count": (
                    corpus_count
                ),

                "retrieval_statistics": (
                    current_statistics
                ),

                "retrieval_configuration": (
                    retrieval_configuration
                ),

                "build_git_commit": (
                    git_commit()
                ),
            },

            queries=queries,

            qids=qids,
        )

        retrieval = {
            row["query_id"]: (
                row["candidate_ids"]
            )

            for row in records
        }

        retrieval_scores = {
            row["query_id"]: (
                row["retrieval_scores"]
            )

            for row in records
        }

        del records

        cache_hit = False

    # ------------------------
    # 复用已有候选缓存
    # ------------------------

    else:

        print(
            "读取已有候选缓存...",
            flush=True,
        )

        (
            retrieval,
            retrieval_scores,
            cache_metadata,
        ) = load_candidate_cache(
            cache_path=cache_path,

            expected_metadata=(
                expected_metadata
            ),

            queries=queries,

            qids=qids,
        )

        current_statistics = {
            "query_encoding_seconds": 0.0,

            "faiss_search_seconds": 0.0,
        }

        cache_hit = True

    # ------------------------
    # 输出固定召回基线
    # ------------------------

    baseline_metrics = average_metrics([
        score_query(
            retrieval[qid],
            qrels[qid],
        )

        for qid in qids
    ])

    print(
        "固定候选召回指标:",
        baseline_metrics,
        flush=True,
    )

    print(
        "Candidate Cache SHA256:",
        cache_metadata[
            "candidate_cache_hash"
        ],
        flush=True,
    )

    # 仅构建或验证缓存。
    if args.cache_only:

        print(
            "Cache-only 模式完成，"
            "不加载 Cross-Encoder。",
            flush=True,
        )

        return

    # ------------------------
    # 提取候选 Passage 正文
    # ------------------------

    required_pids = {
        pid

        for ids in retrieval.values()

        for pid in ids
    }

    texts = extract_passages(
        collection_path,
        required_pids,
    )

    # ------------------------
    # Cross-Encoder 重排序
    # ------------------------

    evaluate_cross_encoder(
        args=args,

        queries=queries,

        qrels=qrels,

        qids=qids,

        retrieval=retrieval,

        retrieval_scores=(
            retrieval_scores
        ),

        texts=texts,

        cache_metadata=(
            cache_metadata
        ),

        cache_hit=cache_hit,

        current_statistics=(
            current_statistics
        ),

        output=args.output,

        rankings_output=(
            args.rankings_output
        ),
    )


if __name__ == "__main__":
    main()
