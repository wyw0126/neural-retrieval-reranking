#B0：评估bm25模型
import argparse
import json
import math
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.msmarco_loader import MSMARCOLoader


def open_db(path):
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA cache_size=-65536")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")

    return conn


def build_index(
    loader,
    index_path,
    max_docs=None,
    batch_size=5000,
    rebuild=False,
):
    """
    流式构建 SQLite FTS5 BM25 索引。

    每条 Passage 使用 rowid = pid + 1。
    这样能够原样恢复 MS MARCO 的 Passage ID。
    """
    if index_path.exists():
        if not rebuild:
            conn = open_db(index_path)

            try:
                status_row = conn.execute(
                    "SELECT value FROM index_meta "
                    "WHERE key='status'"
                ).fetchone()

                source_row = conn.execute(
                    "SELECT value FROM index_meta "
                    "WHERE key='source'"
                ).fetchone()

                limit_row = conn.execute(
                    "SELECT value FROM index_meta "
                    "WHERE key='max_docs'"
                ).fetchone()

                expected_limit = (
                    "all" if max_docs is None
                    else str(max_docs)
                )

                if (
                    status_row == ("ready",)
                    and source_row == (
                        str(
                            (loader.data_dir / "collection.tsv")
                            .resolve()
                        ),
                    )
                    and limit_row == (expected_limit,)
                ):
                    print("发现已完成的索引，跳过构建。")
                    return

                raise RuntimeError(
                    "现有索引未完成或配置不匹配。"
                    "请指定其他 --index-path，"
                    "或明确使用 --rebuild-index。"
                )
            except sqlite3.OperationalError as exc:
                raise RuntimeError(
                    "索引文件存在但元数据不完整，"
                    "请检查或使用 --rebuild-index"
                ) from exc
            finally:
                conn.close()

        index_path.unlink()

    conn = open_db(index_path)

    try:
        conn.execute(
            "CREATE VIRTUAL TABLE passages "
            "USING fts5("
            "text, "
            "content='', "
            "tokenize='porter unicode61'"
            ")"
        )

        conn.execute(
            "CREATE TABLE index_meta("
            "key TEXT PRIMARY KEY, "
            "value TEXT NOT NULL)"
        )

        metadata = {
            "status": "building",
            "source": str(
                (loader.data_dir / "collection.tsv")
                .resolve()
            ),
            "max_docs": (
                "all" if max_docs is None
                else str(max_docs)
            ),
            "tokenizer": "porter unicode61",
            "documents": "0",
        }

        conn.executemany(
            "INSERT INTO index_meta(key, value) VALUES (?, ?)",
            list(metadata.items()),
        )
        conn.commit()

        start = time.perf_counter()
        batch = []
        count = 0

        for pid, text in loader.iter_corpus(limit=max_docs):
            if pid < 0:
                raise ValueError(
                    f"Passage ID 不能为负: {pid}"
                )

            batch.append((pid + 1, text))
            count += 1

            if len(batch) >= batch_size:
                conn.executemany(
                    "INSERT INTO passages(rowid, text) "
                    "VALUES (?, ?)",
                    batch,
                )
                conn.commit()
                batch.clear()

            if count % 100_000 == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"索引进度: {count:,} Passage | "
                    f"{elapsed:.1f}s",
                    flush=True,
                )

        if batch:
            conn.executemany(
                "INSERT INTO passages(rowid, text) "
                "VALUES (?, ?)",
                batch,
            )
            conn.commit()

        conn.execute(
            "UPDATE index_meta SET value=? "
            "WHERE key='documents'",
            (str(count),),
        )

        conn.execute(
            "UPDATE index_meta SET value='ready' "
            "WHERE key='status'"
        )
        conn.commit()

        elapsed = time.perf_counter() - start
        print(
            f"索引完成: {count:,} Passage, "
            f"耗时 {elapsed:.1f}s"
        )
    finally:
        conn.close()


def query_to_fts(query):
    """
    将自然语言 Query 转成安全的 FTS5 OR 查询。

    最多使用前 24 个不同词项。
    保留词项顺序，避免重复词。
    """
    terms = re.findall(r"[^\W_]+", query, re.UNICODE)

    unique_terms = list(
        dict.fromkeys(term.lower() for term in terms)
    )[:24]

    if not unique_terms:
        return None

    # 使用双引号包裹单个词项，避免 FTS5 运算符冲突。
    return " OR ".join(
        f'"{term}"' for term in unique_terms
    )


def search(conn, query, top_k):
    expression = query_to_fts(query)

    if expression is None:
        return []

    rows = conn.execute(
        """
        SELECT rowid, bm25(passages) AS score
        FROM passages
        WHERE passages MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (expression, top_k),
    ).fetchall()

    # FTS5 的 bm25 数值越小越相关。
    # 取负值，使输出分数越大越好。
    return [
        {
            "pid": int(rowid) - 1,
            "score": float(-score),
        }
        for rowid, score in rows
    ]


def evaluate_one(retrieved, relevance):
    """
    计算单个 Query 的:
      MRR@10
      NDCG@10
      Recall@10
      Recall@100

    relevance:
      {passage_id: relevance_grade}
    """
    positives = {
        pid: grade
        for pid, grade in relevance.items()
        if grade > 0
    }

    if not positives:
        raise ValueError(
            "无法评估没有正例标注的 Query"
        )

    ids = [item["pid"] for item in retrieved]
    top10 = ids[:10]
    top100 = ids[:100]

    # MRR@10
    mrr = 0.0

    for rank, pid in enumerate(top10, start=1):
        if pid in positives:
            mrr = 1.0 / rank
            break

    # Recall
    positive_ids = set(positives)

    recall10 = (
        len(set(top10) & positive_ids)
        / len(positive_ids)
    )

    recall100 = (
        len(set(top100) & positive_ids)
        / len(positive_ids)
    )

    # NDCG@10
    dcg = 0.0

    for rank, pid in enumerate(top10, start=1):
        grade = positives.get(pid, 0)

        dcg += (
            (2 ** grade - 1)
            / math.log2(rank + 1)
        )

    ideal_grades = sorted(
        positives.values(), reverse=True
    )[:10]

    idcg = sum(
        (2 ** grade - 1)
        / math.log2(rank + 1)
        for rank, grade in enumerate(
            ideal_grades, start=1
        )
    )

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "MRR@10": mrr,
        "NDCG@10": ndcg,
        "Recall@10": recall10,
        "Recall@100": recall100,
    }


def evaluate(
    loader,
    index_path,
    split="dev",
    max_queries=None,
    top_k=100,
    seed=42,
):
    qrels = loader.load_qrels(split)

    valid_qids = {
        qid
        for qid, rels in qrels.items()
        if any(grade > 0 for grade in rels.values())
    }

    queries = loader.load_queries(
        split, query_ids=valid_qids
    )

    missing = valid_qids - set(queries)

    if missing:
        raise ValueError(
            f"Dev Qrels 存在 {len(missing)} "
            "条找不到文本的 Query"
        )

    qids = sorted(valid_qids)

    # 固定种子，抽取可复现的 Smoke Test 子集。
    if max_queries is not None:
        rng = random.Random(seed)
        qids = sorted(
            rng.sample(
                qids,
                min(max_queries, len(qids)),
            )
        )

    if not qids:
        raise ValueError("没有可评估的 Query")

    conn = open_db(index_path)

    try:
        status = conn.execute(
            "SELECT value FROM index_meta "
            "WHERE key='status'"
        ).fetchone()

        if status != ("ready",):
            raise RuntimeError(
                "BM25 索引尚未构建完成"
            )

        documents = int(
            conn.execute(
                "SELECT value FROM index_meta "
                "WHERE key='documents'"
            ).fetchone()[0]
        )

        results = []
        totals = {
            "MRR@10": 0.0,
            "NDCG@10": 0.0,
            "Recall@10": 0.0,
            "Recall@100": 0.0,
        }

        start = time.perf_counter()

        for i, qid in enumerate(qids, start=1):
            retrieved = search(
                conn, queries[qid], top_k
            )

            scores = evaluate_one(
                retrieved, qrels[qid]
            )

            for metric, value in scores.items():
                totals[metric] += value

            results.append({
                "qid": qid,
                "query": queries[qid],
                "metrics": scores,
                "retrieved": retrieved,
            })

            if i % 10 == 0:
                print(
                    f"检索进度: {i}/{len(qids)}",
                    flush=True,
                )

        elapsed = time.perf_counter() - start

    finally:
        conn.close()

    metrics = {
        name: value / len(qids)
        for name, value in totals.items()
    }

    metrics["query_count"] = len(qids)
    metrics["corpus_count"] = documents
    metrics["top_k"] = top_k
    metrics["total_search_seconds"] = elapsed
    metrics["mean_search_ms"] = (
        elapsed / len(qids) * 1000
    )

    return metrics, results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        type=str,
        default="dev.small",
        choices=["dev", "dev.small"],
        help="选择评估集：dev 或 dev.small",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/raw/msmarco",
    )

    parser.add_argument(
        "--index-path",
        type=Path,
        default=ROOT / "data/indexes/bm25/msmarco_fts5.db",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/experiments/B0_bm25.json",
    )

    parser.add_argument(
        "--run-output",
        type=Path,
        default=None,
        help="可选：保存每条 Query 的检索排名 JSONL",
    )

    parser.add_argument(
        "--build-index",
        action="store_true",
    )

    parser.add_argument(
        "--rebuild-index",
        action="store_true",
    )

    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="仅用于小规模索引测试",
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="限制评估 Query 数量",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.top_k < 100:
        parser.error(
            "--top-k 必须 >= 100，才能计算 Recall@100"
        )

    if args.max_queries is not None and args.max_queries <= 0:
        parser.error("--max-queries 必须大于 0")

    if args.max_docs is not None and args.max_docs <= 0:
        parser.error("--max-docs 必须大于 0")

    if args.rebuild_index and not args.build_index:
        parser.error(
            "--rebuild-index 需要同时指定 --build-index"
        )

    loader = MSMARCOLoader(args.data_dir)

    if args.build_index:
        build_index(
            loader=loader,
            index_path=args.index_path,
            max_docs=args.max_docs,
            rebuild=args.rebuild_index,
        )

    if not args.index_path.is_file():
        parser.error(
            "索引不存在，请先添加 --build-index"
        )

    metrics, results = evaluate(
        loader=loader,
        index_path=args.index_path,
        split=args.split,
        max_queries=args.max_queries,
        top_k=args.top_k,
        seed=args.seed,
    )

    metrics["experiment_id"] = "B0"
    metrics["model"] = "SQLite FTS5 BM25"
    metrics["tokenizer"] = "porter unicode61"
    metrics["seed"] = args.seed
    metrics["subset_corpus"] = args.max_docs is not None
    metrics["split"] = f"MS MARCO Passage v1 {args.split}"

    args.output.parent.mkdir(
        parents=True, exist_ok=True
    )

    args.output.write_text(
        json.dumps(
            metrics, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    if args.run_output is not None:
        args.run_output.parent.mkdir(
            parents=True, exist_ok=True
        )

        with args.run_output.open(
            "w", encoding="utf-8"
        ) as f:
            for result in results:
                f.write(
                    json.dumps(
                        result, ensure_ascii=False
                    ) + "\n"
                )

    print("\n========== B0 BM25 ==========")
    print(json.dumps(
        metrics, ensure_ascii=False, indent=2
    ))

    print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
