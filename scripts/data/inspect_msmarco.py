# 检查msmarco的数据

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DATA_DIR = Path("data/raw/msmarco")
REPORT_PATH = Path("reports/data/msmarco_inspection.json")


def file_sha256(path):
    """分块计算文件 SHA256，不一次性加载整个文件。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_text_file(path):
    """检查 Passage / Query ID 的唯一性和文本完整性。"""
    ids = set()
    total = 0
    duplicates = 0
    malformed = 0
    empty_text = 0

    print(f"\n检查 {path.name} ...", flush=True)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1

            parts = line.rstrip("\r\n").split("\t", 1)

            if len(parts) != 2 or not parts[0]:
                malformed += 1
                continue

            item_id, text = parts

            if item_id in ids:
                duplicates += 1

            ids.add(item_id)

            if not text.strip():
                empty_text += 1

            if total % 1_000_000 == 0:
                print(f"  已检查 {total:,} 行", flush=True)

    stats = {
        "rows": total,
        "unique_ids": len(ids),
        "duplicate_ids": duplicates,
        "malformed_rows": malformed,
        "empty_text_rows": empty_text,
    }

    return ids, stats


def inspect_qrels(path, query_ids, passage_ids):
    """检查相关性标注是否引用了有效的 Query / Passage。"""
    total = 0
    malformed = 0
    missing_query = 0
    missing_passage = 0
    labeled_queries = set()
    relevance_counts = Counter()

    print(f"\n检查 {path.name} ...", flush=True)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            parts = line.split()

            if len(parts) != 4:
                malformed += 1
                continue

            qid, iteration, pid, relevance = parts

            try:
                relevance = int(relevance)
            except ValueError:
                malformed += 1
                continue

            labeled_queries.add(qid)
            relevance_counts[relevance] += 1

            if qid not in query_ids:
                missing_query += 1

            if pid not in passage_ids:
                missing_passage += 1

    return {
        "rows": total,
        "labeled_queries": len(labeled_queries),
        "malformed_rows": malformed,
        "missing_query_references": missing_query,
        "missing_passage_references": missing_passage,
        "relevance_distribution": dict(relevance_counts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hash",
        action="store_true",
        help="同时计算数据文件的 SHA256"
    )
    args = parser.parse_args()

    paths = {
        "corpus": DATA_DIR / "collection.tsv",
        "train_queries": DATA_DIR / "queries.train.tsv",
        "dev_queries": DATA_DIR / "queries.dev.tsv",
        "train_qrels": DATA_DIR / "qrels.train.tsv",
        "dev_qrels": DATA_DIR / "qrels.dev.tsv",
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"缺少 {name}: {path}\n"
                "请先完成下载及解压。"
            )

    corpus_ids, corpus_stats = inspect_text_file(
        paths["corpus"]
    )

    train_ids, train_stats = inspect_text_file(
        paths["train_queries"]
    )

    dev_ids, dev_stats = inspect_text_file(
        paths["dev_queries"]
    )

    train_qrels_stats = inspect_qrels(
        paths["train_qrels"], train_ids, corpus_ids
    )

    dev_qrels_stats = inspect_qrels(
        paths["dev_qrels"], dev_ids, corpus_ids
    )

    overlap = len(train_ids & dev_ids)

    report = {
        "dataset": "MS MARCO Passage",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus_stats,
        "train_queries": train_stats,
        "dev_queries": dev_stats,
        "train_qrels": train_qrels_stats,
        "dev_qrels": dev_qrels_stats,
        "train_dev_query_id_overlap": overlap,
    }

    if args.hash:
        print("\n计算 SHA256 ...", flush=True)
        report["sha256"] = {
            name: file_sha256(path)
            for name, path in paths.items()
        }

    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n========== MS MARCO 检查结果 ==========")
    print("Corpus:", corpus_stats["rows"])
    print("Train Query:", train_stats["rows"])
    print("Dev Query:", dev_stats["rows"])
    print("Train Qrels:", train_qrels_stats["rows"])
    print("Dev Qrels:", dev_qrels_stats["rows"])
    print("Train/Dev ID 重叠:", overlap)

    errors = []

    for name, stats in [
        ("corpus", corpus_stats),
        ("train_queries", train_stats),
        ("dev_queries", dev_stats),
    ]:
        for field in [
            "duplicate_ids", "malformed_rows", "empty_text_rows"
        ]:
            if stats[field] != 0:
                errors.append(f"{name}.{field}={stats[field]}")

    for name, stats in [
        ("train_qrels", train_qrels_stats),
        ("dev_qrels", dev_qrels_stats),
    ]:
        for field in [
            "malformed_rows",
            "missing_query_references",
            "missing_passage_references",
        ]:
            if stats[field] != 0:
                errors.append(f"{name}.{field}={stats[field]}")

    if overlap:
        errors.append(f"train_dev_overlap={overlap}")

    if errors:
        print("\n检查发现异常:")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)

    print("\n数据完整性检查通过！")
    print("报告位置:", REPORT_PATH)


if __name__ == "__main__":
    main()