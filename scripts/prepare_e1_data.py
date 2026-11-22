
# scripts/prepare_e1_data.py

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.msmarco_loader import MSMARCOLoader


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path,
        default=ROOT / "data/raw/msmarco"
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/processed/msmarco/e1_train.jsonl"
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_queries is not None and args.max_queries <= 0:
        parser.error("--max-queries 必须大于 0")

    loader = MSMARCOLoader(args.data_dir)

    print("读取 Train Qrels...", flush=True)
    qrels = loader.load_qrels("train")

    # 每个 Query 只选择一个已标注正例。
    # 多正例时选择最小 Passage ID，保证确定性。
    selected = {}
    for qid, docs in qrels.items():
        positives = [pid for pid, grade in docs.items() if grade > 0]
        if positives:
            selected[qid] = min(positives)

    all_qids = sorted(selected)
    print("有效训练 Query:", len(all_qids))

    if args.max_queries is not None:
        rng = random.Random(args.seed)
        all_qids = sorted(
            rng.sample(
                all_qids,
                min(args.max_queries, len(all_qids))
            )
        )

    selected = {qid: selected[qid] for qid in all_qids}

    queries = loader.load_queries(
        "train", query_ids=set(selected)
    )

    missing_queries = set(selected) - set(queries)
    if missing_queries:
        raise ValueError(
            f"{len(missing_queries)} 个 Query 缺少文本"
        )

    # 流式扫描完整 Corpus，只保存需要的正例文本。
    needed_pids = set(selected.values())
    passage_texts = {}

    print("从 Corpus 提取正例 Passage...", flush=True)

    for i, (pid, text) in enumerate(loader.iter_corpus(), 1):
        if pid in needed_pids:
            passage_texts[pid] = text

        if i % 2_000_000 == 0:
            print(f"已扫描 {i:,} Passage", flush=True)

        if len(passage_texts) == len(needed_pids):
            break

    missing_docs = needed_pids - set(passage_texts)
    if missing_docs:
        raise ValueError(
            f"{len(missing_docs)} 个正例 Passage 不存在"
        )

    # 去掉相同文本的重复 Query，避免批内假负例。
    seen_queries = set()
    records = []

    for qid in all_qids:
        query = queries[qid].strip()
        pid = selected[qid]
        positive = passage_texts[pid].strip()

        key = query.casefold()
        if not query or not positive or key in seen_queries:
            continue

        seen_queries.add(key)

        records.append({
            "query_id": qid,
            "positive_id": pid,
            "query": query,
            "positive": positive,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(
                record, ensure_ascii=False
            ) + "\n")

    report = {
        "dataset": "MS MARCO Passage v1 Train",
        "seed": args.seed,
        "max_queries": args.max_queries,
        "qrels_positive_query_count": len(selected),
        "training_pair_count": len(records),
        "unique_positive_passages": len({
            r["positive_id"] for r in records
        }),
        "source_query_file": "queries.train.tsv",
        "source_qrels_file": "qrels.train.tsv",
        "output": str(args.output),
        "sha256": file_hash(args.output),
        "strategy": "one labeled positive per query",
    }

    report_path = args.output.with_suffix(".meta.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n训练数据构建完成")
    print("样本数:", len(records))
    print("文件:", args.output)
    print("SHA256:", report["sha256"])


if __name__ == "__main__":
    main()
