#用教师模型筛选负例

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="构建教师模型引导的多负例训练集"
    )

    p.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/msmarco/"
            "negative_quality_details.jsonl"
        ),
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/processed/msmarco/e5_filtered"
        ),
    )

    p.add_argument(
        "--report-dir",
        type=Path,
        default=Path(
            "reports/experiments/e5_data"
        ),
    )

    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
    )

    p.add_argument(
        "--min-negatives",
        type=int,
        default=1,
    )

    p.add_argument(
        "--max-negatives",
        type=int,
        default=4,
    )

    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.02,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = p.parse_args()

    if not 0 < args.val_ratio < 1:
        p.error("val-ratio 必须在 0 到 1 之间")

    if not (
        1 <= args.min_negatives
        <= args.max_negatives
        <= 4
    ):
        p.error("负例数量必须在 1–4 之间")

    if any(
        not math.isfinite(t) or t < 0
        for t in args.thresholds
    ):
        p.error("阈值必须是有限非负数")

    args.thresholds = sorted(
        set(args.thresholds)
    )

    return args


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validation_bucket(query_id, seed):
    """
    保持与原 E4 训练脚本完全一致的
    Query ID 哈希划分逻辑。
    """

    key = f"{seed}:{query_id}".encode("utf-8")

    digest = hashlib.sha256(key).digest()

    return int.from_bytes(
        digest[:8], "big"
    ) / (2**64)


def threshold_name(threshold):

    return str(threshold).replace(".", "p")


def write_json(path, data):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("x", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def write_jsonl(path, rows):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("x", encoding="utf-8") as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                ) + "\n"
            )


def rank_bucket(rank):

    if rank <= 10:
        return "1-10"

    if rank <= 30:
        return "11-30"

    if rank <= 60:
        return "31-60"

    if rank <= 100:
        return "61-100"

    return "101+"


def normalize_text(text):

    return " ".join(
        text.casefold().split()
    )



def validate_row(row):

    qid = int(row["query_id"])
    query = row["query"].strip()
    positive = row["positive"].strip()
    positive_score = float(row["positive_score"])

    positive_id = row.get("positive_id")

    if positive_id is not None:
        positive_id = int(positive_id)

    if not query or not positive:
        raise ValueError(
            f"Query {qid} 文本为空"
        )

    if not math.isfinite(positive_score):
        raise ValueError(
            f"Query {qid} 正例分数非法"
        )

    negatives = row["negatives"]

    if len(negatives) != 4:
        raise ValueError(
            f"Query {qid} 原始负例不是四个"
        )

    seen_pids = set()
    seen_texts = set()

    validated = []
    removed_negatives = []

    normalized_positive = normalize_text(positive)

    for neg in negatives:

        pid = int(neg["passage_id"])
        text = neg["text"].strip()
        rank = int(neg["rank"])
        teacher_score = float(neg["teacher_score"])
        margin = float(neg["margin"])

        # ========================================
        # 1. 检查数据完整性
        # ========================================

        if pid < 0 or rank <= 0:
            raise ValueError(
                f"Query {qid} ID 或排名非法"
            )

        if not all(
            math.isfinite(x)
            for x in (teacher_score, margin)
        ):
            raise ValueError(
                f"Query {qid} 分数非法"
            )

        calculated_margin = (
            positive_score - teacher_score
        )

        if not math.isclose(
            calculated_margin,
            margin,
            rel_tol=1e-7,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"Query {qid} Margin 计算不一致"
            )

        # ========================================
        # 2. 基础数据清洗
        # ========================================

        normalized = normalize_text(text)

        reason = None

        if not text:
            reason = "empty_negative_text"

        elif (
            positive_id is not None
            and pid == positive_id
        ):
            reason = "same_positive_id"

        elif normalized == normalized_positive:
            reason = "same_positive_text"

        elif pid in seen_pids:
            reason = "duplicate_negative_id"

        elif normalized in seen_texts:
            reason = "duplicate_negative_text"

        # 发现异常负例时，记录并跳过。
        if reason is not None:

            removed_negatives.append({
                "passage_id": pid,
                "rank": rank,
                "teacher_score": teacher_score,
                "margin": calculated_margin,
                "reason": reason,
            })

            continue

        # ========================================
        # 3. 保留合法负例
        # ========================================

        seen_pids.add(pid)
        seen_texts.add(normalized)

        validated.append({
            **neg,
            "passage_id": pid,
            "text": text,
            "rank": rank,
            "teacher_score": teacher_score,
            "margin": calculated_margin,
        })

    return {
        "query_id": qid,
        "query": query,
        "positive_id": positive_id,
        "positive": positive,
        "positive_score": positive_score,
        "negatives": validated,

        "original_negative_count": len(negatives),

        "cleaned_negative_count": len(validated),

        "removed_invalid_negatives": removed_negatives,
    }


def build_cleaning_report(rows):

    removal_counts = Counter()

    negative_distribution = Counter()

    affected_queries = set()

    removed_cases = []

    original_negative_count = 0

    cleaned_negative_count = 0

    empty_query_count = 0

    for row in rows:

        qid = row["query_id"]

        original_negative_count += row[
            "original_negative_count"
        ]

        cleaned_negative_count += len(
            row["negatives"]
        )

        negative_distribution[
            len(row["negatives"])
        ] += 1

        if not row["negatives"]:
            empty_query_count += 1

        for neg in row["removed_invalid_negatives"]:

            reason = neg["reason"]

            removal_counts[reason] += 1

            affected_queries.add(qid)

            removed_cases.append({
                "query_id": qid,
                **neg,
            })

    removed_count = sum(
        removal_counts.values()
    )

    if (
        original_negative_count
        - cleaned_negative_count
        != removed_count
    ):
        raise RuntimeError(
            "基础清洗数量统计不一致"
        )

    return {
        "original_queries": len(rows),

        "original_negatives": (
            original_negative_count
        ),

        "cleaned_negatives": (
            cleaned_negative_count
        ),

        "removed_negatives": removed_count,

        "removal_ratio": (
            removed_count / original_negative_count
            if original_negative_count
            else 0.0
        ),

        "affected_queries": len(
            affected_queries
        ),

        "affected_query_ratio": (
            len(affected_queries) / len(rows)
            if rows
            else 0.0
        ),

        "empty_queries": empty_query_count,

        "removal_reasons": dict(
            removal_counts
        ),

        "negative_count_distribution": dict(
            sorted(
                negative_distribution.items()
            )
        ),

        "removed_cases": removed_cases,
    }



def load_data(path):

    rows = []
    seen_qids = set()

    with open(path, encoding="utf-8") as f:

        for line_no, line in enumerate(f, 1):

            if not line.strip():
                continue

            try:

                row = validate_row(
                    json.loads(line)
                )

            except Exception as exc:

                raise ValueError(
                    f"第 {line_no} 行数据错误: {exc}"
                ) from exc

            qid = row["query_id"]

            if qid in seen_qids:

                raise ValueError(
                    f"重复 Query ID: {qid}"
                )

            seen_qids.add(qid)

            rows.append(row)

    if not rows:
        raise RuntimeError("输入数据为空")

    return sorted(
        rows,
        key=lambda x: x["query_id"],
    )


def split_data(rows, seed, val_ratio):

    train_rows = []
    val_rows = []

    for row in rows:

        qid = row["query_id"]

        if validation_bucket(
            qid, seed
        ) < val_ratio:

            val_rows.append(row)

        else:

            train_rows.append(row)

    train_qids = {
        row["query_id"]
        for row in train_rows
    }

    val_qids = {
        row["query_id"]
        for row in val_rows
    }

    if train_qids & val_qids:

        raise RuntimeError(
            "训练与验证 Query 存在交叉"
        )

    if not train_rows or not val_rows:

        raise RuntimeError(
            "训练或验证集为空"
        )

    return train_rows, val_rows


def build_filtered_dataset(
    rows,
    threshold,
    min_negatives,
    max_negatives,
):

    retained_rows = []

    dropped_queries = []

    retained_distribution = Counter()

    original_rank_counts = Counter()
    retained_rank_counts = Counter()

    original_negative_count = 0

    removed_by_margin = 0

    removed_by_cap = 0

    removed_due_to_dropped_queries = 0

    for row in rows:

        negatives = row["negatives"]

        original_negative_count += len(
            negatives
        )

        for neg in negatives:

            original_rank_counts[
                rank_bucket(neg["rank"])
            ] += 1

        # 第一步：根据 Margin 筛选。
        eligible = [
            neg for neg in negatives
            if neg["margin"] > threshold
        ]

        removed_by_margin += (
            len(negatives) - len(eligible)
        )

        # 第二步：剩余负例不足时，
        # 跳过整个 Query。
        if len(eligible) < min_negatives:

            removed_due_to_dropped_queries += (
                len(eligible)
            )

            dropped_queries.append({
                "query_id": row["query_id"],
                "positive_score": (
                    row["positive_score"]
                ),
                "original_margins": [
                    neg["margin"]
                    for neg in negatives
                ],
                "remaining_negatives": len(
                    eligible
                ),
                "reason": (
                    "insufficient_negatives"
                ),
            })

            continue

        # 第三步：如果设置了更少的负例上限，
        # 优先保留通过筛选后更困难的负例。
        eligible = sorted(
            eligible,
            key=lambda neg: (
                neg["margin"],
                neg["rank"],
            ),
        )

        retained = eligible[:max_negatives]

        removed_by_cap += (
            len(eligible) - len(retained)
        )

        # 保持原始 E2 召回排名顺序。
        retained.sort(
            key=lambda neg: neg["rank"]
        )

        for neg in retained:

            retained_rank_counts[
                rank_bucket(neg["rank"])
            ] += 1

        retained_distribution[
            len(retained)
        ] += 1

        retained_rows.append({
            **row,

            "negatives": retained,

            "negative_count": len(retained),

            "filter_metadata": {
                "margin_threshold": threshold,
                "original_negative_count": len(
                    negatives
                ),
                "retained_negative_count": len(
                    retained
                ),
            },
        })

    retained_negative_count = sum(
        len(row["negatives"])
        for row in retained_rows
    )

    removed_negative_count = (
        original_negative_count
        - retained_negative_count
    )

    expected_removed_count = (
        removed_by_margin
        + removed_by_cap
        + removed_due_to_dropped_queries
    )

    if removed_negative_count != expected_removed_count:

        raise RuntimeError(
            "负例过滤数量统计不一致"
        )

    if not retained_rows:

        raise RuntimeError(
            f"阈值 {threshold} 下训练数据为空"
        )

    stats = {
        "margin_threshold": threshold,

        "original_queries": len(rows),

        "retained_queries": len(
            retained_rows
        ),

        "dropped_queries": len(
            dropped_queries
        ),

        "original_negatives": (
            original_negative_count
        ),

        "retained_negatives": (
            retained_negative_count
        ),

        "removed_negatives": (
            removed_negative_count
        ),

        "removed_by_margin": (
            removed_by_margin
        ),

        "removed_by_max_negatives": (
            removed_by_cap
        ),

        "removed_due_to_dropped_queries": (
            removed_due_to_dropped_queries
        ),

        "negative_removal_ratio": (
            removed_negative_count
            / original_negative_count
        ),

        "query_removal_ratio": (
            len(dropped_queries)
            / len(rows)
        ),

        "retained_negative_distribution": dict(
            sorted(
                retained_distribution.items()
            )
        ),

        "mean_negatives_per_query": (
            retained_negative_count
            / len(retained_rows)
        ),

        "training_pairs": (
            len(retained_rows)
            + retained_negative_count
        ),

        "rank_bucket_statistics": {
            bucket: {
                "original_count": count,

                "retained_count": (
                    retained_rank_counts[bucket]
                ),

                "retained_ratio": (
                    retained_rank_counts[bucket]
                    / count
                ),
            }

            for bucket, count in sorted(
                original_rank_counts.items()
            )
        },
    }

    return (
        retained_rows,
        dropped_queries,
        stats,
    )


def main():

    args = parse_args()

    if not args.input.is_file():

        raise FileNotFoundError(args.input)

    # 新实验目录禁止覆盖已有内容。
    for directory in (
        args.output_dir,
        args.report_dir,
    ):

        if directory.exists() and any(
            directory.iterdir()
        ):

            raise FileExistsError(
                f"输出目录非空: {directory}"
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    
    print("读取教师模型评分数据...")

    rows = load_data(args.input)

    print("原始 Query 数量:", len(rows))

    # ========================================
    # 1. 生成基础数据清洗报告
    # ========================================

    cleaning_report = build_cleaning_report(rows)

    cleaning_report_path = (
        args.report_dir / "data_cleaning_report.json"
    )

    write_json(
        cleaning_report_path,
        cleaning_report,
    )

    print("\n========== 基础清洗结果 ==========")

    print(
        json.dumps(
            {
                k: v
                for k, v in cleaning_report.items()
                if k != "removed_cases"
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    # ========================================
    # 2. 固定 Query 级训练与验证划分
    # ========================================

    train_rows, val_rows = split_data(
        rows,
        args.seed,
        args.val_ratio,
    )

    # 保留完整划分，避免清洗改变排序验证集。
    original_train_qids = [
        row["query_id"]
        for row in train_rows
    ]

    ranking_validation_qids = [
        row["query_id"]
        for row in val_rows
    ]

    # ========================================
    # 3. 处理清洗后没有负例的 Query
    # ========================================

    empty_train_qids = [
        row["query_id"]
        for row in train_rows
        if not row["negatives"]
    ]

    empty_val_qids = [
        row["query_id"]
        for row in val_rows
        if not row["negatives"]
    ]

    train_rows = [
        row
        for row in train_rows
        if row["negatives"]
    ]

    val_rows = [
        row
        for row in val_rows
        if row["negatives"]
    ]

    if not train_rows or not val_rows:
        raise RuntimeError(
            "基础清洗后训练集或 BCE 验证集为空"
        )

    write_json(
        args.report_dir / "empty_negative_queries.json",
        {
            "train_qids": empty_train_qids,
            "validation_qids": empty_val_qids,
        },
    )

    print("\n========== 数据划分结果 ==========")

    print("原始训练 Query:", len(original_train_qids))

    print("清洗后训练 Query:", len(train_rows))

    print("清洗后 BCE 验证 Query:", len(val_rows))

    print(
        "完整排序验证 Query:",
        len(ranking_validation_qids)
    )

    print(
        "被删除的训练 Query:",
        len(empty_train_qids)
    )

    print(
        "被删除的 BCE 验证 Query:",
        len(empty_val_qids)
    )

    

    # ------------------------------------------------
    # 保存固定训练和验证划分。
    # ------------------------------------------------

    train_path = (
        args.output_dir
        / "train_unfiltered.jsonl"
    )

    val_path = (
        args.output_dir
        / "validation_unfiltered.jsonl"
    )

    write_jsonl(train_path, train_rows)

    write_jsonl(val_path, val_rows)

    split_path = (
        args.output_dir
        / "split_metadata.json"
    )

    split_metadata = {
        "source": str(args.input.resolve()),

        "source_sha256": sha256_file(
            args.input
        ),

        "seed": args.seed,

        "val_ratio": args.val_ratio,

        "train_query_count": len(train_rows),

        "validation_query_count": len(val_rows),

        "train_qids": [
            row["query_id"]
            for row in train_rows
        ],

        "validation_qids": [
            row["query_id"]
            for row in val_rows
        ],

        "train_sha256": sha256_file(
            train_path
        ),

        "validation_sha256": sha256_file(
            val_path
        ),

        "original_train_qids": original_train_qids,

        "ranking_validation_qids": ranking_validation_qids,

        "ranking_validation_query_count": len(
            ranking_validation_qids
        ),

        "empty_train_qids": empty_train_qids,

        "empty_validation_qids": empty_val_qids,

        "cleaning_report_sha256": sha256_file(
            cleaning_report_path
        ),
    }

    write_json(
        split_path,
        split_metadata,
    )

    all_results = []

    # ------------------------------------------------
    # 不同阈值只作用于训练数据。
    # ------------------------------------------------

    for threshold in args.thresholds:

        print(
            f"\n========== Margin > "
            f"{threshold} =========="
        )

        (
            filtered_rows,
            dropped_queries,
            stats,
        ) = build_filtered_dataset(
            train_rows,
            threshold,
            args.min_negatives,
            args.max_negatives,
        )

        name = threshold_name(threshold)

        output_path = (
            args.output_dir
            / f"train_margin_{name}.jsonl"
        )

        dropped_path = (
            args.report_dir
            / f"dropped_queries_{name}.json"
        )

        report_path = (
            args.report_dir
            / f"margin_{name}_summary.json"
        )

        write_jsonl(
            output_path,
            filtered_rows,
        )

        write_json(
            dropped_path,
            dropped_queries,
        )

        stats.update({
            "experiment_id": (
                f"E5-Margin-{threshold}"
            ),

            "input_sha256": (
                split_metadata["source_sha256"]
            ),

            "split_sha256": sha256_file(
                split_path
            ),

            "validation_query_count": len(
                val_rows
            ),

            "validation_sha256": (
                split_metadata["validation_sha256"]
            ),

            "output": str(
                output_path.resolve()
            ),

            "output_sha256": sha256_file(
                output_path
            ),
        })

        write_json(
            report_path,
            stats,
        )

        all_results.append(stats)

        print(
            "保留 Query:",
            stats["retained_queries"]
        )

        print(
            "保留负例:",
            stats["retained_negatives"]
        )

        print(
            "过滤比例:",
            f"{stats['negative_removal_ratio']:.2%}"
        )

        print(
            "每组平均负例数:",
            stats["mean_negatives_per_query"]
        )

        print(
            "负例数量分布:",
            stats["retained_negative_distribution"]
        )

    write_json(
        args.report_dir / "overview.json",
        {
            "experiment_id": (
                "E5-Filtered-Negative-Datasets"
            ),

            "split_metadata": str(
                split_path.resolve()
            ),

            "original_train_queries": len(
                train_rows
            ),

            "validation_queries": len(
                val_rows
            ),

            "threshold_results": all_results,
        },
    )

    print("\n全部筛选完成")

    print("训练数据:", args.output_dir)

    print("统计报告:", args.report_dir)


if __name__ == "__main__":
    main()
