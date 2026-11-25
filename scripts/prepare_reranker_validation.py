

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def parse_args():

    p = argparse.ArgumentParser(
        description="构建固定 Cross-Encoder 排序验证集"
    )

    p.add_argument(
        "--split-file",
        type=Path,
        default=Path(
            "data/processed/msmarco/e5_filtered/"
            "split_metadata.json"
        ),
    )

    p.add_argument(
        "--validation-file",
        type=Path,
        default=Path(
            "data/processed/msmarco/e5_filtered/"
            "validation_unfiltered.jsonl"
        ),
    )

    p.add_argument(
        "--queries-file",
        type=Path,
        default=Path(
            "data/raw/msmarco/queries.train.tsv"
        ),
    )

    p.add_argument(
        "--qrels-file",
        type=Path,
        default=Path(
            "data/raw/msmarco/qrels.train.tsv"
        ),
    )

    p.add_argument(
        "--dev-queries-file",
        type=Path,
        default=Path(
            "data/raw/msmarco/queries.dev.small.tsv"
        ),
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/processed/msmarco/"
            "reranker_validation"
        ),
    )

    return p.parse_args()


# ============================================================
# 1. 文件身份校验
# ============================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):

            digest.update(chunk)

    return digest.hexdigest()


def normalize_text(text):

    return " ".join(
        text.casefold().split()
    )


# ============================================================
# 2. 读取固定 Query 划分
# ============================================================

def load_split(path):

    metadata = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    train_list = metadata.get(
        "original_train_qids",
        metadata["train_qids"],
    )

    validation_list = metadata.get(
        "ranking_validation_qids",
        metadata["validation_qids"],
    )

    train_qids = {
        int(qid)
        for qid in train_list
    }

    val_qids = {
        int(qid)
        for qid in validation_list
    }

    if len(train_qids) != len(train_list):

        raise RuntimeError(
            "训练划分存在重复 Query ID"
        )

    if len(val_qids) != len(validation_list):

        raise RuntimeError(
            "验证划分存在重复 Query ID"
        )

    if not train_qids or not val_qids:

        raise RuntimeError(
            "训练或验证 Query 集合为空"
        )

    if train_qids & val_qids:

        raise RuntimeError(
            "训练与排序验证 Query 存在交叉"
        )

    expected_count = metadata.get(
        "ranking_validation_query_count",
        len(validation_list),
    )

    if len(val_qids) != expected_count:

        raise RuntimeError(
            "排序验证 Query 数量不一致"
        )

    return (
        train_qids,
        val_qids,
        metadata,
    )


# ============================================================
# 3. 读取原始 Query
# ============================================================

def read_queries(path, required_qids=None):

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

            query = parts[1]

            if (
                required_qids is not None
                and qid not in required_qids
            ):

                continue

            if not query.strip():

                raise ValueError(
                    f"Query {qid} 文本为空"
                )

            if qid in queries:

                raise ValueError(
                    f"重复 Query ID: {qid}"
                )

            queries[qid] = query

    return queries


# ============================================================
# 4. 读取原始 Qrels
# ============================================================

def read_qrels(path, required_qids):

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
            relevance = int(parts[3])

            if qid not in required_qids:

                continue

            if pid < 0 or relevance < 0:

                raise ValueError(
                    f"非法 Qrels: {qid}, {pid}"
                )

            if pid in qrels[qid]:

                if qrels[qid][pid] != relevance:

                    raise ValueError(
                        f"Qrels 标签冲突: "
                        f"{qid}, {pid}"
                    )

            qrels[qid][pid] = relevance

    return dict(qrels)


# ============================================================
# 5. 校验已有 BCE 验证集
# ============================================================

def verify_bce_validation(
    path,
    metadata,
    queries,
    qrels,
    ranking_val_qids,
):

    actual_hash = sha256_file(path)

    expected_hash = metadata[
        "validation_sha256"
    ]

    if actual_hash != expected_hash:

        raise RuntimeError(
            "BCE 验证文件 SHA256 校验失败"
        )

    seen_qids = set()

    positive_id_missing = 0

    qrels_negative_conflicts = []

    with open(path, encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            qid = int(row["query_id"])

            if qid in seen_qids:

                raise RuntimeError(
                    f"BCE 验证 Query 重复: {qid}"
                )

            seen_qids.add(qid)

            if qid not in ranking_val_qids:

                raise RuntimeError(
                    f"非法验证 Query: {qid}"
                )

            if normalize_text(
                row["query"]
            ) != normalize_text(
                queries[qid]
            ):

                raise RuntimeError(
                    f"Query {qid} 正文不一致"
                )

            positive_id = row.get(
                "positive_id"
            )

            if positive_id is None:

                positive_id_missing += 1

            else:

                positive_id = int(
                    positive_id
                )

                if qrels[qid].get(
                    positive_id, 0
                ) <= 0:

                    raise RuntimeError(
                        f"Query {qid} 正例 ID "
                        "不属于 Qrels 已知正例"
                    )

            for neg in row["negatives"]:

                pid = int(
                    neg["passage_id"]
                )

                if qrels[qid].get(pid, 0) > 0:

                    qrels_negative_conflicts.append({
                        "query_id": qid,
                        "passage_id": pid,
                    })

    expected_qids = {
        int(qid)
        for qid in metadata[
            "validation_qids"
        ]
    }

    if seen_qids != expected_qids:

        raise RuntimeError(
            "BCE 验证 Query 集合与划分不一致"
        )

    if qrels_negative_conflicts:

        raise RuntimeError(
            "BCE 验证集中存在 Qrels 已知正例"
            "被标记为负例: "
            f"{qrels_negative_conflicts[:10]}"
        )

    return {
        "validation_file_sha256": actual_hash,

        "bce_validation_query_count": (
            len(seen_qids)
        ),

        "missing_positive_id_count": (
            positive_id_missing
        ),

        "known_positive_as_negative_count": (
            len(qrels_negative_conflicts)
        ),
    }


# ============================================================
# 6. 主流程
# ============================================================

def main():

    args = parse_args()

    for path in (
        args.split_file,
        args.validation_file,
        args.queries_file,
        args.qrels_file,
        args.dev_queries_file,
    ):

        if not path.is_file():

            raise FileNotFoundError(path)

    if (
        args.output_dir.exists()
        and any(args.output_dir.iterdir())
    ):

        raise FileExistsError(
            f"输出目录非空: {args.output_dir}"
        )

    print("读取固定数据划分...")

    (
        train_qids,
        val_qids,
        split_metadata,
    ) = load_split(args.split_file)

    print("训练 Query:", len(train_qids))

    print("排序验证 Query:", len(val_qids))

    # ------------------------------------------------
    # 防止内部训练 / 验证 Query 与 Dev Small 重叠。
    # ------------------------------------------------

    dev_queries = read_queries(
        args.dev_queries_file
    )

    dev_qids = set(dev_queries)

    train_dev_overlap = (
        train_qids & dev_qids
    )

    val_dev_overlap = (
        val_qids & dev_qids
    )

    if train_dev_overlap or val_dev_overlap:

        raise RuntimeError(
            "内部数据与 Dev Small Query ID 重叠: "
            f"train={len(train_dev_overlap)}, "
            f"validation={len(val_dev_overlap)}"
        )

    # ------------------------------------------------
    # 提取固定排序验证 Query。
    # ------------------------------------------------

    print("从 MS MARCO Train 提取验证 Query...")

    queries = read_queries(
        args.queries_file,
        val_qids,
    )

    missing_queries = (
        val_qids - set(queries)
    )

    if missing_queries:

        raise RuntimeError(
            f"缺少 {len(missing_queries)} 条验证 Query"
        )

    # ------------------------------------------------
    # 提取完整原始 Qrels。
    # ------------------------------------------------

    print("提取验证 Query 的原始 Qrels...")

    qrels = read_qrels(
        args.qrels_file,
        val_qids,
    )

    positive_counts = {}

    for qid in val_qids:

        positives = {
            pid
            for pid, rel in qrels.get(
                qid, {}
            ).items()
            if rel > 0
        }

        if not positives:

            raise RuntimeError(
                f"Query {qid} 缺少已知正例"
            )

        positive_counts[qid] = len(
            positives
        )

    # ------------------------------------------------
    # 验证与之前生成的 BCE 验证集一致。
    # ------------------------------------------------

    bce_validation_stats = (
        verify_bce_validation(
            args.validation_file,
            split_metadata,
            queries,
            qrels,
            val_qids,
        )
    )

    # ------------------------------------------------
    # 保存标准 MS MARCO TSV 文件。
    # ------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    query_output = (
        args.output_dir
        / "queries.validation.tsv"
    )

    qrels_output = (
        args.output_dir
        / "qrels.validation.tsv"
    )

    metadata_output = (
        args.output_dir
        / "metadata.json"
    )

    with query_output.open(
        "x",
        encoding="utf-8",
    ) as f:

        for qid in sorted(val_qids):

            f.write(
                f"{qid}\t{queries[qid]}\n"
            )

    qrels_count = 0

    with qrels_output.open(
        "x",
        encoding="utf-8",
    ) as f:

        for qid in sorted(val_qids):

            for pid, rel in sorted(
                qrels[qid].items()
            ):

                f.write(
                    f"{qid}\t0\t{pid}\t{rel}\n"
                )

                qrels_count += 1

    report = {
        "split": "reranker-validation",

        "query_count": len(val_qids),

        "qrels_count": qrels_count,

        "positive_count": sum(
            positive_counts.values()
        ),

        "mean_positives_per_query": (
            sum(positive_counts.values())
            / len(val_qids)
        ),

        "train_validation_overlap": 0,

        "train_dev_overlap": 0,

        "validation_dev_overlap": 0,

        "split_file": str(
            args.split_file.resolve()
        ),

        "split_sha256": sha256_file(
            args.split_file
        ),

        "source_queries_sha256": sha256_file(
            args.queries_file
        ),

        "source_qrels_sha256": sha256_file(
            args.qrels_file
        ),

        "queries_sha256": sha256_file(
            query_output
        ),

        "qrels_sha256": sha256_file(
            qrels_output
        ),

        "bce_validation_statistics": (
            bce_validation_stats
        ),

        "notes": (
            "本验证集独立于本轮 E5 Cross-Encoder "
            "微调数据，但不保证独立于 E2 或 E3 "
            "模型之前的训练数据。"
        ),
    }

    with metadata_output.open(
        "x",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(
        "\n========== 验证集构建完成 =========="
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("Queries:", query_output)

    print("Qrels:", qrels_output)

    print("Metadata:", metadata_output)


if __name__ == "__main__":
    main()
