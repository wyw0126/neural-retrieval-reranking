#用原始cross encoder模型对负例进行评估

import argparse
import hashlib
import json
import math
import random
import subprocess
import time

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from sentence_transformers import CrossEncoder


ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 1. 参数设置
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description="E2 困难负例质量诊断"
    )

    p.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/msmarco/"
            "e4_e2_multi_50k.jsonl"
        ),
    )

    p.add_argument(
        "--model",
        type=Path,
        default=Path(
            "models/"
            "ms-marco-MiniLM-L6-v2-cross-encoder"
        ),
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/experiments/"
            "negative_quality_summary.json"
        ),
    )

    p.add_argument(
        "--details-output",
        type=Path,
        default=Path(
            "data/processed/msmarco/"
            "negative_quality_details.jsonl"
        ),
    )

    p.add_argument(
        "--cases-output",
        type=Path,
        default=Path(
            "reports/experiments/"
            "negative_quality_cases.json"
        ),
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    p.add_argument(
        "--query-batch-size",
        type=int,
        default=128,
    )

    p.add_argument(
        "--max-length",
        type=int,
        default=256,
    )

    p.add_argument(
        "--max-queries",
        type=int,
        default=None,
    )

    p.add_argument(
        "--expected-negatives",
        type=int,
        default=4,
    )

    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.0, 3.0],
    )

    p.add_argument(
        "--case-count",
        type=int,
        default=100,
    )

    p.add_argument(
        "--device",
        default="cuda",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = p.parse_args()

    for name in (
        "batch_size",
        "query_batch_size",
        "max_length",
        "expected_negatives",
        "case_count",
    ):

        if getattr(args, name) <= 0:

            p.error(
                f"{name} 必须为正数"
            )

    if (
        args.max_queries is not None
        and args.max_queries <= 0
    ):

        p.error(
            "--max-queries 必须为正数"
        )

    if any(
        not math.isfinite(t) or t < 0
        for t in args.thresholds
    ):

        p.error(
            "Margin 阈值必须为有限非负数"
        )

    args.thresholds = sorted(
        set(args.thresholds)
    )

    return args


# ============================================================
# 2. 实验身份与文件校验
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


def model_hash(model_dir):

    model_dir = Path(model_dir).resolve()

    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    files = []

    for path in model_dir.rglob("*"):

        if not path.is_file():
            continue

        relative = path.relative_to(model_dir)

        if any(
            part.startswith("checkpoint-")
            for part in relative.parts
        ):

            continue

        if path.name in {
            "train_result.json",
            "trainer_state.json",
            "README.md",
        }:

            continue

        if (
            path.suffix == ".safetensors"
            or path.name.startswith("pytorch_model")
            or path.suffix in {
                ".json",
                ".txt",
                ".model",
                ".vocab",
            }
        ):

            files.append(path)

    if not any(
        p.suffix == ".safetensors"
        or p.name.startswith("pytorch_model")
        for p in files
    ):

        raise RuntimeError(
            "模型目录缺少权重文件"
        )

    digest = hashlib.sha256()

    for path in sorted(files):

        relative = path.relative_to(
            model_dir
        ).as_posix()

        digest.update(
            relative.encode("utf-8")
        )

        digest.update(b"\0")

        digest.update(
            sha256_file(path).encode("ascii")
        )

        digest.update(b"\0")

    return digest.hexdigest()


def git_commit():

    try:

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):

        return None


def prepare_outputs(args):

    outputs = [
        args.output,
        args.details_output,
        args.cases_output,
    ]

    resolved = [
        p.resolve()
        for p in outputs
    ]

    if len(set(resolved)) != len(resolved):

        raise ValueError(
            "输出文件路径不能相同"
        )

    protected = {args.input.resolve()}

    # 同时检查正式输出和临时输出路径。
    all_paths = resolved + [
        Path(str(p) + ".partial").resolve()
        for p in outputs
    ]

    if len(set(all_paths)) != len(all_paths):

        raise ValueError(
            "正式输出与临时输出路径冲突"
        )

    for path in all_paths:

        if path in protected:

            raise ValueError(
                "禁止覆盖原始训练数据"
            )

        if path.is_relative_to(
            args.model.resolve()
        ):

            raise ValueError(
                "禁止在模型目录中写入分析结果"
            )

        if path.exists():

            raise FileExistsError(
                f"输出文件已存在: {path}"
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


def save_json(path, content):

    path = Path(path)

    temporary = Path(
        str(path) + ".partial"
    )

    with temporary.open(
        "x",
        encoding="utf-8",
    ) as f:

        json.dump(
            content,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    if path.exists():
        raise FileExistsError(path)

    temporary.rename(path)


# ============================================================
# 3. 数据读取与校验
# ============================================================

def validate_row(row, expected_negatives):

    qid = int(row["query_id"])

    query = row["query"].strip()

    positive = row["positive"].strip()

    positive_id = row.get("positive_id")

    if positive_id is not None:
        positive_id = int(positive_id)

    if qid < 0:
        raise ValueError("非法 Query ID")

    if positive_id is not None and positive_id < 0:
        raise ValueError("非法 Positive ID")

    negatives = row["negatives"]

    if not query or not positive:

        raise ValueError(
            f"Query {qid} 正例或 Query 为空"
        )

    if len(negatives) != expected_negatives:

        raise ValueError(
            f"Query {qid} 实际负例数量为 "
            f"{len(negatives)}，"
            f"预期 {expected_negatives}"
        )

    seen_pids = set()

    seen_texts = set()

    normalized_negatives = []

    duplicate_text_count = 0

    for neg in negatives:

        pid = int(neg["passage_id"])

        text = neg["text"].strip()

        rank = int(neg["rank"])

        # 兼容常见召回分数字段。
        retrieval_score = neg.get(
            "retrieval_score",
            neg.get(
                "score",
                neg.get("negative_score"),
            ),
        )

        if pid < 0 or rank <= 0:

            raise ValueError(
                f"Query {qid} 负例 ID 或排名非法"
            )

        if not text:

            raise ValueError(
                f"Query {qid} 存在空负例"
            )

        if pid in seen_pids:

            raise ValueError(
                f"Query {qid} 负例 ID 重复"
            )

        if positive_id is not None and pid == positive_id:

            raise ValueError(
                f"Query {qid} 正例被当作负例"
            )

        if text == positive:

            raise ValueError(
                f"Query {qid} 正负例文本相同"
            )

        normalized_text = " ".join(
            text.casefold().split()
        )

        if normalized_text in seen_texts:

            duplicate_text_count += 1

        seen_pids.add(pid)

        seen_texts.add(normalized_text)

        if retrieval_score is not None:

            retrieval_score = float(
                retrieval_score
            )

            if not math.isfinite(
                retrieval_score
            ):

                raise ValueError(
                    f"Query {qid} 召回分数非法"
                )

        normalized_negatives.append({
            "passage_id": pid,
            "text": text,
            "rank": rank,
            "retrieval_score": retrieval_score,
        })

    return {
        "query_id": qid,
        "query": query,
        "positive_id": positive_id,
        "positive": positive,
        "negatives": normalized_negatives,
        "duplicate_negative_text_count": (
            duplicate_text_count
        ),
    }


def load_data(args):

    rng = random.Random(args.seed)

    rows = []

    seen_qids = set()

    total = 0

    duplicate_text_count = 0

    negative_distribution = Counter()

    with args.input.open(
        encoding="utf-8"
    ) as f:

        for line_no, line in enumerate(f, 1):

            if not line.strip():
                continue

            try:

                row = validate_row(
                    json.loads(line),
                    args.expected_negatives,
                )

            except (
                ValueError,
                TypeError,
                KeyError,
            ) as exc:

                raise ValueError(
                    f"第 {line_no} 行数据不合法: {exc}"
                ) from exc

            qid = row["query_id"]

            if qid in seen_qids:

                raise ValueError(
                    f"重复 Query ID: {qid}"
                )

            seen_qids.add(qid)

            total += 1

            negative_distribution[
                len(row["negatives"])
            ] += 1

            duplicate_text_count += row[
                "duplicate_negative_text_count"
            ]

            # 不限制 Query 数量时使用全部数据。
            if args.max_queries is None:

                rows.append(row)

            # 否则使用蓄水池采样，
            # 避免总是选取文件前 N 条。
            elif len(rows) < args.max_queries:

                rows.append(row)

            else:

                j = rng.randrange(total)

                if j < args.max_queries:

                    rows[j] = row

    if not rows:

        raise RuntimeError(
            "没有可分析的数据"
        )

    rows.sort(
        key=lambda row: row["query_id"]
    )

    stats = {
        "source_query_count": total,

        "selected_query_count": len(rows),

        "selected_negative_count": sum(
            len(row["negatives"])
            for row in rows
        ),

        "source_negative_count_distribution": dict(
            negative_distribution
        ),

        "source_duplicate_negative_text_count": (
            duplicate_text_count
        ),

        "selected_duplicate_negative_text_count": sum(
            row["duplicate_negative_text_count"]
            for row in rows
        ),
    }

    return rows, stats


# ============================================================
# 4. 批量 Cross-Encoder 评分
# ============================================================

def predict_logits(
    model,
    pairs,
    batch_size,
):

    with torch.inference_mode():

        scores = model.predict(
            pairs,
            batch_size=batch_size,
            activation_fn=torch.nn.Identity(),
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    scores = np.asarray(scores)

    if scores.shape == (len(pairs), 1):

        scores = scores[:, 0]

    if scores.shape != (len(pairs),):

        raise RuntimeError(
            "Cross-Encoder 分数维度错误: "
            f"{scores.shape}"
        )

    if not np.isfinite(scores).all():

        raise RuntimeError(
            "Cross-Encoder 出现 NaN 或 Inf"
        )

    return scores


def score_batch(
    model,
    rows,
    batch_size,
):

    pairs = []

    # 每组依次放入一个正例和全部负例。
    for row in rows:

        query = row["query"]

        pairs.append([
            query,
            row["positive"],
        ])

        for neg in row["negatives"]:

            pairs.append([
                query,
                neg["text"],
            ])

    scores = predict_logits(
        model,
        pairs,
        batch_size,
    )

    cursor = 0

    results = []

    for row in rows:

        positive_score = float(
            scores[cursor]
        )

        cursor += 1

        negatives = []

        for neg in row["negatives"]:

            negative_score = float(
                scores[cursor]
            )

            cursor += 1

            margin = (
                positive_score - negative_score
            )

            negatives.append({
                **neg,

                "teacher_score": negative_score,

                "margin": margin,

                "negative_ge_positive": (
                    negative_score
                    >= positive_score
                ),

                "negative_gt_positive": (
                    negative_score
                    > positive_score
                ),
            })

        margins = [
            neg["margin"]
            for neg in negatives
        ]

        results.append({
            "query_id": row["query_id"],

            "query": row["query"],

            "positive_id": row["positive_id"],

            "positive": row["positive"],

            "positive_score": positive_score,

            "negatives": negatives,

            "min_margin": min(margins),

            "mean_margin": float(
                np.mean(margins)
            ),

            "negative_ge_positive_count": sum(
                neg["negative_ge_positive"]
                for neg in negatives
            ),

            "negative_gt_positive_count": sum(
                neg["negative_gt_positive"]
                for neg in negatives
            ),
        })

    if cursor != len(scores):

        raise RuntimeError(
            "模型评分与输入样本数量不一致"
        )

    return results


# ============================================================
# 5. 统计分析
# ============================================================

def distribution(values):

    if not values:

        return {
            "count": 0,
        }

    arr = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "count": int(arr.size),

        "mean": float(arr.mean()),

        "std": float(arr.std()),

        "min": float(arr.min()),

        "p05": float(
            np.percentile(arr, 5)
        ),

        "p10": float(
            np.percentile(arr, 10)
        ),

        "p25": float(
            np.percentile(arr, 25)
        ),

        "median": float(
            np.percentile(arr, 50)
        ),

        "p75": float(
            np.percentile(arr, 75)
        ),

        "p90": float(
            np.percentile(arr, 90)
        ),

        "p95": float(
            np.percentile(arr, 95)
        ),

        "max": float(arr.max()),
    }


def ratio(count, total):

    if total == 0:
        return 0.0

    return count / total


def rank_bucket(rank):

    # 固定诊断区间，不改变原始采样规则。
    if rank <= 10:

        return "1-10"

    if rank <= 30:

        return "11-30"

    if rank <= 60:

        return "31-60"

    if rank <= 100:

        return "61-100"

    return "101+"


def margin_bucket(margin):

    if margin <= 0:

        return "nonpositive"

    if margin <= 0.5:

        return "(0, 0.5]"

    if margin <= 1.0:

        return "(0.5, 1]"

    if margin <= 2.0:

        return "(1, 2]"

    if margin <= 3.0:

        return "(2, 3]"

    return "(3, +inf)"


class QualityStatistics:

    def __init__(self, thresholds):

        self.thresholds = thresholds

        self.query_count = 0

        self.negative_count = 0

        self.positive_scores = []

        self.negative_scores = []

        self.margins = []

        self.query_min_margins = []

        self.rank_groups = defaultdict(
            lambda: {
                "margins": [],
                "scores": [],
                "retrieval_scores": [],
            }
        )

        self.margin_counts = Counter()

        self.query_ge_counts = Counter()

        self.query_gt_counts = Counter()

        self.threshold_counts = Counter()

        self.query_threshold_counts = Counter()

    def update(self, result):

        self.query_count += 1

        positive_score = result[
            "positive_score"
        ]

        self.positive_scores.append(
            positive_score
        )

        self.query_min_margins.append(
            result["min_margin"]
        )

        self.query_ge_counts[
            result["negative_ge_positive_count"]
        ] += 1

        self.query_gt_counts[
            result["negative_gt_positive_count"]
        ] += 1

        for threshold in self.thresholds:

            if result["min_margin"] <= threshold:

                self.query_threshold_counts[
                    threshold
                ] += 1

        for neg in result["negatives"]:

            self.negative_count += 1

            negative_score = neg[
                "teacher_score"
            ]

            margin = neg["margin"]

            self.negative_scores.append(
                negative_score
            )

            self.margins.append(margin)

            self.margin_counts[
                margin_bucket(margin)
            ] += 1

            for threshold in self.thresholds:

                if margin <= threshold:

                    self.threshold_counts[
                        threshold
                    ] += 1

            bucket = rank_bucket(
                neg["rank"]
            )

            group = self.rank_groups[bucket]

            group["margins"].append(
                margin
            )

            group["scores"].append(
                negative_score
            )

            if neg["retrieval_score"] is not None:

                group["retrieval_scores"].append(
                    neg["retrieval_score"]
                )

    def summary(self):

        ge_count = sum(
            m <= 0
            for m in self.margins
        )

        gt_count = sum(
            m < 0
            for m in self.margins
        )

        equal_count = sum(
            m == 0
            for m in self.margins
        )

        query_ge_count = sum(
            count
            for k, count in self.query_ge_counts.items()
            if k > 0
        )

        query_gt_count = sum(
            count
            for k, count in self.query_gt_counts.items()
            if k > 0
        )

        rank_results = {}

        for bucket in [
            "1-10",
            "11-30",
            "31-60",
            "61-100",
            "101+",
        ]:

            if bucket not in self.rank_groups:

                continue

            group = self.rank_groups[bucket]

            margins = group["margins"]

            count = len(margins)

            rank_results[bucket] = {
                "negative_count": count,

                "teacher_score_distribution": (
                    distribution(group["scores"])
                ),

                "margin_distribution": (
                    distribution(margins)
                ),

                "retrieval_score_distribution": (
                    distribution(
                        group["retrieval_scores"]
                    )
                ),

                "negative_ge_positive_ratio": ratio(
                    sum(m <= 0 for m in margins),
                    count,
                ),

                "negative_gt_positive_ratio": ratio(
                    sum(m < 0 for m in margins),
                    count,
                ),

                "margin_thresholds": {
                    str(t): {
                        "count": sum(
                            m <= t
                            for m in margins
                        ),

                        "ratio": ratio(
                            sum(
                                m <= t
                                for m in margins
                            ),
                            count,
                        ),
                    }

                    for t in self.thresholds
                },
            }

        return {
            "query_count": self.query_count,

            "negative_count": self.negative_count,

            "positive_score_distribution": (
                distribution(self.positive_scores)
            ),

            "negative_score_distribution": (
                distribution(self.negative_scores)
            ),

            "margin_distribution": (
                distribution(self.margins)
            ),

            "query_min_margin_distribution": (
                distribution(self.query_min_margins)
            ),

            # 负例级别的教师排序冲突。
            "negative_ge_positive_count": ge_count,

            "negative_ge_positive_ratio": ratio(
                ge_count,
                self.negative_count,
            ),

            "negative_gt_positive_count": gt_count,

            "negative_gt_positive_ratio": ratio(
                gt_count,
                self.negative_count,
            ),

            "equal_score_count": equal_count,

            # Query 级别的教师排序冲突。
            "query_with_conflict_count": (
                query_ge_count
            ),

            "query_with_conflict_ratio": ratio(
                query_ge_count,
                self.query_count,
            ),

            "query_with_strict_conflict_count": (
                query_gt_count
            ),

            "query_with_strict_conflict_ratio": ratio(
                query_gt_count,
                self.query_count,
            ),

            # 每组存在几个冲突负例。
            "query_conflict_count_distribution": {
                str(k): {
                    "query_count": count,

                    "ratio": ratio(
                        count,
                        self.query_count,
                    ),
                }

                for k, count in sorted(
                    self.query_ge_counts.items()
                )
            },

            # Margin 区间分布。
            "margin_bucket_distribution": {
                bucket: {
                    "count": count,

                    "ratio": ratio(
                        count,
                        self.negative_count,
                    ),
                }

                for bucket, count in (
                    self.margin_counts.items()
                )
            },

            # 不同阈值下的累计覆盖率。
            "margin_threshold_analysis": {
                str(t): {
                    "negative_count": (
                        self.threshold_counts[t]
                    ),

                    "negative_ratio": ratio(
                        self.threshold_counts[t],
                        self.negative_count,
                    ),

                    "affected_query_count": (
                        self.query_threshold_counts[t]
                    ),

                    "affected_query_ratio": ratio(
                        self.query_threshold_counts[t],
                        self.query_count,
                    ),
                }

                for t in self.thresholds
            },

            # 不同原始召回排名区间。
            "rank_bucket_analysis": rank_results,
        }


# ============================================================
# 6. 典型案例保存
# ============================================================

def collect_cases(
    result,
    worst_cases,
    case_count,
):

    for neg in result["negatives"]:

        case = {
            "query_id": result["query_id"],

            "query": result["query"],

            "positive_id": result["positive_id"],

            "positive": result["positive"],

            "positive_score": result[
                "positive_score"
            ],

            "negative_id": neg[
                "passage_id"
            ],

            "negative": neg["text"],

            "negative_score": neg[
                "teacher_score"
            ],

            "retrieval_rank": neg["rank"],

            "retrieval_score": neg[
                "retrieval_score"
            ],

            "margin": neg["margin"],

            "negative_ge_positive": neg[
                "negative_ge_positive"
            ],
        }

        worst_cases.append(case)

    # 只保留 Margin 最低的若干案例，
    # 避免保存全部样本正文到内存。
    worst_cases.sort(
        key=lambda x: (
            x["margin"],
            x["query_id"],
            x["negative_id"],
        )
    )

    del worst_cases[case_count:]


# ============================================================
# 7. 控制台结果
# ============================================================

def print_summary(summary):

    print(
        "\n========== 困难负例质量诊断 =========="
    )

    print(
        "分析 Query 数量:",
        summary["query_count"],
    )

    print(
        "分析负例数量:",
        summary["negative_count"],
    )

    print(
        "\n正例平均分数:",
        summary[
            "positive_score_distribution"
        ]["mean"],
    )

    print(
        "负例平均分数:",
        summary[
            "negative_score_distribution"
        ]["mean"],
    )

    print(
        "平均 Margin:",
        summary[
            "margin_distribution"
        ]["mean"],
    )

    print(
        "中位数 Margin:",
        summary[
            "margin_distribution"
        ]["median"],
    )

    print(
        "\n负例得分 >= 正例:",
        f"{summary['negative_ge_positive_ratio']:.2%}",
    )

    print(
        "负例得分 > 正例:",
        f"{summary['negative_gt_positive_ratio']:.2%}",
    )

    print(
        "存在教师排序冲突的 Query:",
        f"{summary['query_with_conflict_ratio']:.2%}",
    )

    print(
        "\n========== Margin 阈值分析 =========="
    )

    for threshold, item in summary[
        "margin_threshold_analysis"
    ].items():

        print(
            f"Margin <= {threshold}: "
            f"负例覆盖率 "
            f"{item['negative_ratio']:.2%} | "
            f"Query 覆盖率 "
            f"{item['affected_query_ratio']:.2%}"
        )

    print(
        "\n========== 召回排名区间分析 =========="
    )

    for bucket, item in summary[
        "rank_bucket_analysis"
    ].items():

        print(
            f"Rank {bucket}: "
            f"数量={item['negative_count']:,} | "
            f"平均 Margin="
            f"{item['margin_distribution']['mean']:.4f} | "
            f"冲突率="
            f"{item['negative_ge_positive_ratio']:.2%}"
        )


# ============================================================
# 8. 主流程
# ============================================================

def main():

    args = parse_args()

    started = time.perf_counter()

    random.seed(args.seed)

    np.random.seed(args.seed)

    torch.manual_seed(args.seed)

    if not args.input.is_file():

        raise FileNotFoundError(args.input)

    if not args.model.is_dir():

        raise FileNotFoundError(args.model)

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "CUDA 不可用，请检查 GPU 环境"
        )

    prepare_outputs(args)

    print(
        "计算训练文件 SHA256...",
        flush=True,
    )

    input_hash = sha256_file(args.input)

    print(
        "计算参考模型身份...",
        flush=True,
    )

    teacher_hash = model_hash(args.model)

    print(
        "读取并检查困难负例数据...",
        flush=True,
    )

    rows, data_stats = load_data(args)

    print(
        json.dumps(
            data_stats,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    print(
        "加载 E3 Cross-Encoder...",
        flush=True,
    )

    model = CrossEncoder(
        str(args.model),
        device=args.device,
        max_length=args.max_length,
        activation_fn=torch.nn.Identity(),
        local_files_only=True,
    )

    if model.max_length != args.max_length:

        raise RuntimeError(
            "参考模型实际 max_length 不一致"
        )

    statistics = QualityStatistics(
        args.thresholds
    )

    worst_cases = []

    inference_seconds = 0.0

    processed_pairs = 0

    details_temp = Path(
        str(args.details_output) + ".partial"
    )

    print(
        "开始困难负例评分...",
        flush=True,
    )

    with details_temp.open(
        "x",
        encoding="utf-8",
    ) as f:

        for start in range(
            0,
            len(rows),
            args.query_batch_size,
        ):

            batch = rows[
                start:
                start + args.query_batch_size
            ]

            t0 = time.perf_counter()

            results = score_batch(
                model,
                batch,
                args.batch_size,
            )

            inference_seconds += (
                time.perf_counter() - t0
            )

            for result in results:

                statistics.update(result)

                collect_cases(
                    result,
                    worst_cases,
                    args.case_count,
                )

                processed_pairs += (
                    1 + len(result["negatives"])
                )

                # 保存完整的逐 Query 评分结果。
                f.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        allow_nan=False,
                    ) + "\n"
                )

            processed_queries = (
                start + len(batch)
            )

            speed = (
                processed_pairs
                / inference_seconds
                if inference_seconds > 0
                else 0.0
            )

            print(
                f"\r已分析 "
                f"{processed_queries:,}/{len(rows):,} "
                f"Query | "
                f"{speed:.1f} pairs/s",
                end="",
                flush=True,
            )

    print(
        "\n全部评分完成",
        flush=True,
    )

    if statistics.query_count != len(rows):

        raise RuntimeError(
            "实际评分 Query 数量不一致"
        )

    if statistics.negative_count != data_stats[
        "selected_negative_count"
    ]:

        raise RuntimeError(
            "实际评分负例数量不一致"
        )

    quality_summary = statistics.summary()

    details_hash = sha256_file(
        details_temp
    )

    # 再次检查，保证未覆盖历史结果。
    if args.details_output.exists():

        raise FileExistsError(
            args.details_output
        )

    details_temp.rename(
        args.details_output
    )

    cases_report = {
        "description": (
            "按照正负例 Margin 从低到高排序的"
            "教师排序冲突或高困难度案例。"
            "并非已经证实的假负例。"
        ),

        "case_count": len(worst_cases),

        "cases": worst_cases,
    }

    save_json(
        args.cases_output,
        cases_report,
    )

    summary = {
        "experiment_id": (
            "E4-Negative-Quality-Diagnosis"
        ),

        "git_commit": git_commit(),

        "input": str(args.input.resolve()),

        "input_sha256": input_hash,

        "teacher_model": str(
            args.model.resolve()
        ),

        "teacher_model_hash": teacher_hash,

        "max_length": model.max_length,

        "activation": "Identity",

        "device": args.device,

        "seed": args.seed,

        "batch_size": args.batch_size,

        "query_batch_size": (
            args.query_batch_size
        ),

        "expected_negatives": (
            args.expected_negatives
        ),

        "thresholds": args.thresholds,

        "data_statistics": data_stats,

        "quality_statistics": (
            quality_summary
        ),

        "total_scored_pairs": (
            processed_pairs
        ),

        "inference_seconds": (
            inference_seconds
        ),

        "mean_pairs_per_second": ratio(
            processed_pairs,
            inference_seconds,
        ),

        "total_seconds": (
            time.perf_counter() - started
        ),

        "details_output": str(
            args.details_output.resolve()
        ),

        "details_sha256": details_hash,

        "cases_output": str(
            args.cases_output.resolve()
        ),

        "cases_sha256": sha256_file(
            args.cases_output
        ),

        "interpretation": (
            "教师模型评分仅用于困难负例质量诊断。"
            "高分负例不等同于真实假负例，"
            "不得将教师排序冲突率直接报告为假负例率。"
        ),
    }

    save_json(
        args.output,
        summary,
    )

    print_summary(
        quality_summary
    )

    print(
        "\n汇总报告:",
        args.output,
    )

    print(
        "逐 Query 详细评分:",
        args.details_output,
    )

    print(
        "典型案例:",
        args.cases_output,
    )


if __name__ == "__main__":
    main()
