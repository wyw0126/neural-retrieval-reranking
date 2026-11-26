#分批编码文章、保存 Embedding 和 ID、断点续跑
"""步骤：
流式读取 collection.tsv。
GPU 批量生成 384 维 Embedding。
每处理一个 Chunk 保存 .npz 文件。
保存 manifest.json，记录进度和模型版本。
再次启动时自动跳过已经完整保存的 Chunk。
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.msmarco_loader import MSMARCOLoader


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(8 * 1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def save_json_atomic(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def model_identity(model_dir):
    """
    使用模型文件的内容哈希标记模型版本。
    B1/E1/E2 模型变化时，避免错误复用旧向量。
    """
    candidates = [
        model_dir / "model.safetensors",
        model_dir / "pytorch_model.bin",
    ]
    weight = next(
        (p for p in candidates if p.is_file()), None
    )
    if weight is None:
        raise FileNotFoundError("未找到模型权重文件")

    files = [
        weight,
        model_dir / "modules.json",
        model_dir / "config.json",
    ]

    pool_config = model_dir / "1_Pooling/config.json"
    if pool_config.exists():
        files.append(pool_config)

    h = hashlib.sha256()
    for path in files:
        if path.exists():
            h.update(path.name.encode())
            h.update(sha256(path).encode())

    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/raw/msmarco",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/embeddings/minilm",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100000
    )
    parser.add_argument(
        "--max-docs", type=int, default=None
    )
    parser.add_argument(
        "--device", default="cuda"
    )

    args = parser.parse_args()

    if args.batch_size <= 0 or args.chunk_size <= 0:
        parser.error("batch-size 和 chunk-size 必须大于 0")
    if args.max_docs is not None and args.max_docs <= 0:
        parser.error("max-docs 必须大于 0")

    model_dir = args.model.resolve()
    corpus_path = (args.data_dir / "collection.tsv").resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not corpus_path.is_file():
        raise FileNotFoundError(corpus_path)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 PyTorch")

    corpus_stat = corpus_path.stat()

    identity = {
        "model_path": str(model_dir),
        "model_hash": model_identity(model_dir),
        "corpus_path": str(corpus_path),
        "corpus_bytes": corpus_stat.st_size,
        "corpus_mtime_ns": corpus_stat.st_mtime_ns,
        "chunk_size": args.chunk_size,
        "max_docs": args.max_docs,
        "normalize_embeddings": True,
        "dtype": "float32",
    }

    manifest_path = out / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        if manifest["identity"] != identity:
            raise RuntimeError(
                "已有编码数据与当前配置不一致。"
                "请使用新的 --output-dir，"
                "不要覆盖旧实验。"
            )

        for item in manifest["chunks"]:
            if not (out / item["file"]).is_file():
                raise RuntimeError(
                    f"缺少已记录的 Chunk: {item['file']}"
                )
    else:
        if list(out.glob("chunk_*.npz")):
            raise RuntimeError(
                "发现旧 Chunk 但没有 manifest.json，"
                "请检查目录，不要直接覆盖。"
            )

        manifest = {
            "identity": identity,
            "embedding_dim": None,
            "processed_docs": 0,
            "complete": False,
            "chunks": [],
        }
        save_json_atomic(manifest_path, manifest)

    print("加载 MiniLM 模型...", flush=True)

    model = SentenceTransformer(
        str(model_dir),
        device=args.device,
        local_files_only=True,
    )
    model.eval()

    dim = model.get_sentence_embedding_dimension()

    if manifest["embedding_dim"] not in (None, dim):
        raise RuntimeError("Embedding 维度与旧数据不一致")

    manifest["embedding_dim"] = dim

    print(f"模型维度: {dim}")
    print(f"设备: {model.device}")
    print(f"已完成文档: {manifest['processed_docs']:,}")

    if manifest["complete"]:
        print("该目录的编码任务已经完成。")
        return

    # 检查已有 Chunk 的编号和累计数量。
    processed = 0
    for i, item in enumerate(manifest["chunks"]):
        if item["index"] != i or item["start"] != processed:
            raise RuntimeError("manifest Chunk 顺序异常")
        processed += item["count"]

    if processed != manifest["processed_docs"]:
        raise RuntimeError("manifest 进度统计不一致")

    # 处理上次意外中断时留下的临时文件。
    for tmp in out.glob("chunk_*.npz.partial"):
        tmp.unlink()

    loader = MSMARCOLoader(args.data_dir)

    # 跳过已经编码的文档；不做重复 GPU 编码。
    records = islice(
        loader.iter_corpus(limit=args.max_docs),
        processed,
        None,
    )

    started = time.perf_counter()
    chunk_index = len(manifest["chunks"])

    while True:
        rows = list(islice(records, args.chunk_size))

        if not rows:
            break

        ids = np.asarray(
            [pid for pid, _ in rows],
            dtype=np.int64,
        )
        texts = [text for _, text in rows]

        vectors = model.encode(
            texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        vectors = np.ascontiguousarray(
            vectors, dtype=np.float32
        )

        if vectors.shape != (len(ids), dim):
            raise RuntimeError("Embedding 形状异常")

        if not np.isfinite(vectors).all():
            raise RuntimeError("Embedding 出现 NaN/Inf")

        chunk_name = f"chunk_{chunk_index:06d}.npz"
        final_path = out / chunk_name
        temp_path = out / (chunk_name + ".partial")

        if final_path.exists():
            raise RuntimeError(
                f"发现未记录的 Chunk: {final_path}。"
                "请先核查，避免覆盖已有结果。"
            )

        # 先写完整文件，再原子重命名。
        with temp_path.open("wb") as f:
            np.savez(f, ids=ids, vectors=vectors)

        os.replace(temp_path, final_path)

        item = {
            "index": chunk_index,
            "file": chunk_name,
            "start": processed,
            "count": len(ids),
            "first_pid": int(ids[0]),
            "last_pid": int(ids[-1]),
        }

        manifest["chunks"].append(item)
        processed += len(ids)
        manifest["processed_docs"] = processed

        save_json_atomic(manifest_path, manifest)

        elapsed = time.perf_counter() - started

        print(
            f"Chunk {chunk_index} 完成 | "
            f"累计 {processed:,} 文档 | "
            f"本次运行 {elapsed:.1f}s",
            flush=True,
        )

        chunk_index += 1

    manifest["complete"] = True
    manifest["finished_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    save_json_atomic(manifest_path, manifest)

    print("\n编码完成")
    print("文档数量:", processed)
    print("Embedding 维度:", dim)
    print("Chunk 数量:", len(manifest["chunks"]))
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
