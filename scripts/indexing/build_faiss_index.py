#构建faiss精确索引

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(8 * 1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def save_json_atomic(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=ROOT / "data/embeddings/minilm",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=ROOT / "data/indexes/faiss/minilm_flatip.faiss",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="明确覆盖已有索引",
    )

    args = parser.parse_args()

    if args.threads <= 0:
        parser.error("--threads 必须大于 0")

    faiss.omp_set_num_threads(args.threads)

    embeddings_dir = args.embeddings_dir.resolve()
    manifest_path = embeddings_dir / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if not manifest.get("complete"):
        raise RuntimeError(
            "Corpus 编码尚未完成，请先继续运行 encode_corpus.py"
        )

    if not manifest["identity"]["normalize_embeddings"]:
        raise RuntimeError("向量未归一化，不能执行当前实验")

    dim = int(manifest["embedding_dim"])
    chunks = manifest["chunks"]

    expected_docs = int(manifest["processed_docs"])

    if not chunks or expected_docs <= 0:
        raise RuntimeError("Manifest 中没有有效向量")

    index_path = args.index_path.resolve()
    meta_path = index_path.with_suffix(".json")

    index_path.parent.mkdir(parents=True, exist_ok=True)

    if (
        index_path.exists() or meta_path.exists()
    ) and not args.force:
        raise FileExistsError(
            "索引或元数据已存在。"
            "请使用其他路径，"
            "或明确指定 --force 覆盖。"
        )

    index = faiss.IndexIDMap2(
        faiss.IndexFlatIP(dim)
    )

    started = time.perf_counter()
    total = 0

    for i, chunk in enumerate(chunks):
        chunk_path = embeddings_dir / chunk["file"]

        if not chunk_path.is_file():
            raise FileNotFoundError(chunk_path)

        with np.load(
            chunk_path,
            allow_pickle=False,
        ) as data:
            ids = np.asarray(
                data["ids"], dtype=np.int64
            )
            vectors = np.ascontiguousarray(
                data["vectors"],
                dtype=np.float32,
            )

        if ids.shape != (chunk["count"],):
            raise ValueError(
                f"Chunk ID 数量不正确: {chunk_path}"
            )

        if vectors.shape != (chunk["count"], dim):
            raise ValueError(
                f"Chunk 向量维度不正确: {chunk_path}"
            )

        if (
            int(ids[0]) != chunk["first_pid"]
            or int(ids[-1]) != chunk["last_pid"]
        ):
            raise ValueError(
                f"Chunk ID 范围不匹配: {chunk_path}"
            )

        if len(np.unique(ids)) != len(ids):
            raise ValueError(
                f"Chunk 内存在重复 ID: {chunk_path}"
            )

        if not np.isfinite(vectors).all():
            raise ValueError(
                f"Chunk 包含无效向量: {chunk_path}"
            )

        index.add_with_ids(vectors, ids)
        total += len(ids)

        print(
            f"导入 {i+1}/{len(chunks)} | "
            f"索引文档 {total:,}",
            flush=True,
        )

    if total != expected_docs:
        raise RuntimeError(
            f"数量不一致：索引 {total}，"
            f"Manifest {expected_docs}"
        )

    if index.ntotal != expected_docs:
        raise RuntimeError("FAISS ntotal 不一致")

    # 写入临时索引；完成后替换为正式索引。
    tmp_path = index_path.with_name(
        index_path.name + ".tmp"
    )

    if tmp_path.exists():
        tmp_path.unlink()

    print("开始保存 FAISS 索引...", flush=True)
    faiss.write_index(index, str(tmp_path))

    # 索引文件成功保存后才创建元数据。
    os.replace(tmp_path, index_path)

    elapsed = time.perf_counter() - started

    metadata = {
        "index_type": "IndexIDMap2(IndexFlatIP)",
        "embedding_dim": dim,
        "document_count": total,
        "normalize_embeddings": True,
        "embedding_model_hash": (
            manifest["identity"]["model_hash"]
        ),
        "embedding_manifest_sha256": sha256(manifest_path),
        "embedding_manifest_path": str(manifest_path),
        "build_seconds": elapsed,
        "faiss_version": faiss.__version__,
        "threads": args.threads,
    }

    save_json_atomic(meta_path, metadata)

    print("\nFAISS 索引构建完成")
    print("索引:", index_path)
    print("元数据:", meta_path)
    print("文档数量:", total)
    print("构建耗时:", round(elapsed, 2), "秒")


if __name__ == "__main__":
    main()
