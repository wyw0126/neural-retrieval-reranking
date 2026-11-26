
#!/usr/bin/env python3
# scripts/train_cross_encoder_bce_multi.py

import argparse
import hashlib
import json
import math
import random
import subprocess
import time

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from datasets import Dataset

from sentence_transformers import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)

from sentence_transformers.cross_encoder.losses import (
    BinaryCrossEntropyLoss,
)


ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# 1. 命令行参数
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description="变长多负例 Cross-Encoder BCE 训练"
    )

    p.add_argument(
        "--experiment-id",
        required=True,
    )

    p.add_argument(
        "--train-file",
        type=Path,
        required=True,
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
        "--split-file",
        type=Path,
        default=Path(
            "data/processed/msmarco/e5_filtered/"
            "split_metadata.json"
        ),
    )

    # 对筛选版本进一步校验训练文件身份。
    p.add_argument(
        "--data-report",
        type=Path,
        default=None,
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
        "--output-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--final-model-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--report-dir",
        type=Path,
        default=Path(
            "reports/training"
        ),
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
    )

    p.add_argument(
        "--epochs",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--max-steps",
        type=int,
        default=-1,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    p.add_argument(
        "--max-length",
        type=int,
        default=256,
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
        "--eval-steps",
        type=int,
        default=250,
    )

    p.add_argument(
        "--save-steps",
        type=int,
        default=250,
    )

    p.add_argument(
        "--logging-steps",
        type=int,
        default=100,
    )

    p.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
    )

    p.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )

    p.add_argument(
        "--precision",
        choices=["auto", "fp32", "bf16"],
        default="auto",
    )

    # 默认仅保存模型权重，减少磁盘占用。
    # 如果需要完整断点恢复，可以启用该参数。
    p.add_argument(
        "--save-training-state",
        action="store_true",
    )

    # 只检查数据，不启动训练。
    p.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = p.parse_args()

    if not (
        1 <= args.min_negatives
        <= args.max_negatives
        <= 4
    ):

        p.error(
            "负例数量必须在 1–4 之间"
        )

    positive_args = (
        "batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "eval_steps",
        "save_steps",
        "logging_steps",
    )

    for name in positive_args:

        if getattr(args, name) <= 0:

            p.error(
                f"{name} 必须为正数"
            )

    if args.epochs <= 0:

        p.error(
            "epochs 必须大于 0"
        )

    if args.max_steps != -1 and args.max_steps <= 0:

        p.error(
            "max-steps 必须为 -1 或正整数"
        )

    if not math.isfinite(args.lr) or args.lr <= 0:

        p.error(
            "lr 必须为有限正数"
        )

    if not 0 <= args.warmup_ratio <= 1:

        p.error(
            "warmup-ratio 必须在 0–1 之间"
        )

    if args.num_workers < 0:

        p.error(
            "num-workers 不能小于 0"
        )

    if (
        args.save_total_limit is not None
        and args.save_total_limit <= 0
    ):

        p.error(
            "save-total-limit 必须大于 0"
        )

    return args


# ============================================================
# 2. 通用工具
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

    """
    记录原始模型权重及配置身份。

    不包含训练日志和优化器状态。
    """

    model_dir = Path(model_dir).resolve()

    if not model_dir.is_dir():

        raise FileNotFoundError(model_dir)

    files = []

    ignored = {
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

        if path.name in ignored:

            continue

        relative = path.relative_to(
            model_dir
        )

        if any(
            part.startswith("checkpoint-")
            for part in relative.parts
        ):

            continue

        if (
            path.suffix == ".safetensors"
            or path.name.startswith("pytorch_model")
            or path.name.startswith("model-")
            or path.suffix in {
                ".json",
                ".txt",
                ".model",
                ".vocab",
            }
        ):

            files.append(path)

    if not any(
        path.suffix == ".safetensors"
        or path.name.startswith("pytorch_model")
        for path in files
    ):

        raise RuntimeError(
            f"模型目录缺少权重: {model_dir}"
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


def git_state():

    try:

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )

        return {
            "git_commit": commit,
            "git_dirty": dirty,
        }

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):

        return {
            "git_commit": None,
            "git_dirty": None,
        }


def normalize_text(text):

    return " ".join(
        text.casefold().split()
    )


def write_json(path, data):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
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


# ============================================================
# 3. 输出路径保护
# ============================================================

def verify_paths(args):

    for path in (
        args.train_file,
        args.validation_file,
        args.split_file,
    ):

        if not path.is_file():

            raise FileNotFoundError(path)

    if (
        args.data_report is not None
        and not args.data_report.is_file()
    ):

        raise FileNotFoundError(
            args.data_report
        )

    if not args.model.is_dir():

        raise FileNotFoundError(args.model)

    if (
        args.train_file.resolve()
        == args.validation_file.resolve()
    ):

        raise ValueError(
            "训练集和验证集不能是同一个文件"
        )

    model_dir = args.model.resolve()

    output_dir = args.output_dir.resolve()

    final_dir = args.final_model_dir.resolve()

    for directory in (
        output_dir,
        final_dir,
    ):

        if (
            directory == model_dir
            or directory.is_relative_to(model_dir)
            or model_dir.is_relative_to(directory)
        ):

            raise ValueError(
                "输出目录不能与初始模型目录重叠"
            )

    if (
        output_dir == final_dir
        or output_dir.is_relative_to(final_dir)
        or final_dir.is_relative_to(output_dir)
    ):

        raise ValueError(
            "Checkpoint 与最终模型目录不能重叠"
        )

    if args.dry_run:

        return

    for directory in (
        output_dir,
        final_dir,
    ):

        if directory.exists():

            if not directory.is_dir():

                raise FileExistsError(directory)

            if any(directory.iterdir()):

                raise FileExistsError(
                    f"实验输出目录非空: {directory}"
                )

    report_path = (
        args.report_dir
        / f"{args.experiment_id}.json"
    )

    if report_path.exists():

        raise FileExistsError(report_path)


# ============================================================
# 4. 固定数据划分与哈希校验
# ============================================================

def verify_data_identity(args):

    split = json.loads(
        args.split_file.read_text(
            encoding="utf-8"
        )
    )

    split_hash = sha256_file(
        args.split_file
    )

    train_hash = sha256_file(
        args.train_file
    )

    validation_hash = sha256_file(
        args.validation_file
    )

    if validation_hash != split[
        "validation_sha256"
    ]:

        raise RuntimeError(
            "验证文件与固定划分的哈希不一致"
        )

    expected_train_qids = {
        int(qid)
        for qid in split["train_qids"]
    }

    expected_val_qids = {
        int(qid)
        for qid in split["validation_qids"]
    }

    ranking_val_qids = {
        int(qid)

        for qid in split.get(
            "ranking_validation_qids",
            split["validation_qids"],
        )
    }

    if expected_train_qids & ranking_val_qids:

        raise RuntimeError(
            "固定训练与排序验证 Query 重叠"
        )

    data_report = None

    if args.data_report is not None:

        data_report = json.loads(
            args.data_report.read_text(
                encoding="utf-8"
            )
        )

        if train_hash != data_report[
            "output_sha256"
        ]:

            raise RuntimeError(
                "筛选训练文件哈希校验失败"
            )

        if split_hash != data_report[
            "split_sha256"
        ]:

            raise RuntimeError(
                "筛选报告与数据划分不一致"
            )

        if validation_hash != data_report[
            "validation_sha256"
        ]:

            raise RuntimeError(
                "筛选报告与验证集不一致"
            )

        dataset_type = "margin_filtered"

    elif train_hash == split["train_sha256"]:

        dataset_type = "unfiltered"

    else:

        raise RuntimeError(
            "训练文件无法通过身份校验。"
            "筛选版本必须提供 --data-report"
        )

    identity = {
        "dataset_type": dataset_type,

        "train_sha256": train_hash,

        "validation_sha256": validation_hash,

        "split_sha256": split_hash,

        "data_report_sha256": (
            sha256_file(args.data_report)
            if args.data_report is not None
            else None
        ),

        "margin_threshold": (
            data_report["margin_threshold"]
            if data_report is not None
            else None
        ),
    }

    return (
        identity,
        expected_train_qids,
        expected_val_qids,
        ranking_val_qids,
        data_report,
    )


# ============================================================
# 5. 读取变长多负例数据
# ============================================================

def load_query_rows(
    path,
    min_negatives,
    max_negatives,
    margin_threshold=None,
):

    rows = []

    seen_qids = set()

    with open(path, encoding="utf-8") as f:

        for line_no, line in enumerate(f, 1):

            if not line.strip():
                continue

            row = json.loads(line)

            qid = int(row["query_id"])

            query = row["query"].strip()

            positive = row["positive"].strip()

            negatives = row["negatives"]

            positive_id = row.get(
                "positive_id"
            )

            if qid in seen_qids:

                raise ValueError(
                    f"重复 Query ID: {qid}"
                )

            seen_qids.add(qid)

            if not query or not positive:

                raise ValueError(
                    f"第 {line_no} 行文本为空"
                )

            if not (
                min_negatives
                <= len(negatives)
                <= max_negatives
            ):

                raise ValueError(
                    f"Query {qid} 负例数量非法: "
                    f"{len(negatives)}"
                )

            normalized_positive = (
                normalize_text(positive)
            )

            seen_pids = set()

            seen_texts = set()

            negative_texts = []

            for neg in negatives:

                pid = int(
                    neg["passage_id"]
                )

                text = neg["text"].strip()

                normalized = normalize_text(
                    text
                )

                if pid < 0 or not text:

                    raise ValueError(
                        f"Query {qid} 负例非法"
                    )

                if pid in seen_pids:

                    raise ValueError(
                        f"Query {qid} 负例 ID 重复"
                    )

                if (
                    positive_id is not None
                    and pid == int(positive_id)
                ):

                    raise ValueError(
                        f"Query {qid} 正负例 ID 相同"
                    )

                if normalized == normalized_positive:

                    raise ValueError(
                        f"Query {qid} 正负例文本相同"
                    )

                if normalized in seen_texts:

                    raise ValueError(
                        f"Query {qid} 负例文本重复"
                    )

                # 对 Margin 筛选数据进行二次检查。
                if margin_threshold is not None:

                    margin = float(
                        neg["margin"]
                    )

                    if not math.isfinite(margin):

                        raise ValueError(
                            f"Query {qid} Margin 非法"
                        )

                    if margin <= margin_threshold:

                        raise ValueError(
                            f"Query {qid} 存在不满足 "
                            f"Margin > {margin_threshold} "
                            "的负例"
                        )

                seen_pids.add(pid)

                seen_texts.add(normalized)

                negative_texts.append(text)

            rows.append({
                "query_id": qid,

                "query": query,

                "positive": positive,

                "negatives": negative_texts,
            })

    if not rows:

        raise RuntimeError(
            f"数据文件为空: {path}"
        )

    return rows


# ============================================================
# 6. 将 Query 分组转换为 BCE 样本对
# ============================================================

def flatten_groups(groups):

    pairs = []

    negative_distribution = Counter()

    negative_count = 0

    for row in groups:

        query = row["query"]

        # 每条 Query 的正例只添加一次。
        pairs.append({
            "sentence1": query,

            "sentence2": row["positive"],

            "label": 1.0,
        })

        # 添加全部剩余负例，不随机丢弃。
        for text in row["negatives"]:

            pairs.append({
                "sentence1": query,

                "sentence2": text,

                "label": 0.0,
            })

        count = len(
            row["negatives"]
        )

        negative_count += count

        negative_distribution[count] += 1

    query_count = len(groups)

    stats = {
        "queries": query_count,

        "positive_count": query_count,

        "negative_count": negative_count,

        "pair_count": len(pairs),

        "negative_distribution": dict(
            sorted(
                negative_distribution.items()
            )
        ),

        "mean_negatives_per_query": (
            negative_count / query_count
        ),

        "positive_fraction": (
            query_count / len(pairs)
        ),

        "negative_fraction": (
            negative_count / len(pairs)
        ),
    }

    if len(pairs) != (
        query_count + negative_count
    ):

        raise RuntimeError(
            "BCE 样本数量统计不一致"
        )

    return pairs, stats


def load_data(args):

    (
        identity,
        expected_train_qids,
        expected_val_qids,
        ranking_val_qids,
        data_report,
    ) = verify_data_identity(args)

    train_groups = load_query_rows(
        args.train_file,

        min_negatives=args.min_negatives,

        max_negatives=args.max_negatives,

        margin_threshold=identity[
            "margin_threshold"
        ],
    )

    val_groups = load_query_rows(
        args.validation_file,

        min_negatives=1,

        max_negatives=4,

        margin_threshold=None,
    )

    train_qids = {
        row["query_id"]
        for row in train_groups
    }

    val_qids = {
        row["query_id"]
        for row in val_groups
    }

    if train_qids & ranking_val_qids:

        raise RuntimeError(
            "训练与排序验证集 Query 泄漏"
        )

    if val_qids != expected_val_qids:

        raise RuntimeError(
            "验证 Query 集合与固定划分不一致"
        )

    if not train_qids.issubset(
        expected_train_qids
    ):

        raise RuntimeError(
            "训练集中出现划分之外的 Query"
        )

    if identity["dataset_type"] == "unfiltered":

        if train_qids != expected_train_qids:

            raise RuntimeError(
                "未筛选训练 Query 集合不完整"
            )

    train_rows, train_stats = (
        flatten_groups(train_groups)
    )

    val_rows, val_stats = (
        flatten_groups(val_groups)
    )

    # 检查实际训练数据与筛选报告一致。
    if data_report is not None:

        checks = {
            "retained_queries": (
                train_stats["queries"]
            ),

            "retained_negatives": (
                train_stats["negative_count"]
            ),

            "training_pairs": (
                train_stats["pair_count"]
            ),
        }

        for key, actual in checks.items():

            expected = data_report[key]

            if actual != expected:

                raise RuntimeError(
                    f"数据报告不一致: {key}; "
                    f"expected={expected}, "
                    f"actual={actual}"
                )

    random.Random(
        args.seed
    ).shuffle(train_rows)

    stats = {
        "train": train_stats,

        "validation": val_stats,

        "train_validation_overlap": 0,

        "ranking_validation_query_count": (
            len(ranking_val_qids)
        ),

        "dropped_train_queries": (
            len(expected_train_qids - train_qids)
        ),

        "positive_negative_ratio": (
            f"1:"
            f"{train_stats['mean_negatives_per_query']:.6f}"
        ),
    }

    return (
        Dataset.from_list(train_rows),

        Dataset.from_list(val_rows),

        stats,

        identity,
    )


# ============================================================
# 7. 训练精度配置
# ============================================================

def resolve_precision(args):

    if args.device == "cpu":

        if args.precision == "bf16":

            raise ValueError(
                "当前脚本不启用 CPU BF16"
            )

        return False, False

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA 不可用"
        )

    bf16_supported = (
        torch.cuda.is_bf16_supported()
    )

    if args.precision == "bf16":

        if not bf16_supported:

            raise RuntimeError(
                "当前 GPU 不支持 BF16"
            )

        return True, False

    if args.precision == "fp32":

        return False, False

    # 保持之前训练脚本的精度策略。
    return bf16_supported, False


# ============================================================
# 8. 主训练流程
# ============================================================

def main():

    args = parse_args()

    verify_paths(args)

    started = time.perf_counter()

    started_at = datetime.now(
        timezone.utc
    ).isoformat()

    source_git_state = git_state()

    random.seed(args.seed)

    torch.manual_seed(args.seed)

    print(
        "校验初始模型和数据文件...",
        flush=True,
    )

    initial_model_hash = model_hash(
        args.model
    )

    (
        train_data,
        val_data,
        stats,
        data_identity,
    ) = load_data(args)

    print(
        "\n========== 训练数据统计 =========="
    )

    print(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    print(
        "\n========== 数据身份 =========="
    )

    print(
        json.dumps(
            data_identity,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    effective_batch_size = (
        args.batch_size
        * args.gradient_accumulation_steps
    )

    # 单 GPU 情况下的估算。
    estimated_steps_per_epoch = math.ceil(
        len(train_data)
        / effective_batch_size
    )

    estimated_total_steps = (
        args.max_steps
        if args.max_steps > 0
        else math.ceil(
            estimated_steps_per_epoch
            * args.epochs
        )
    )

    print(
        "\n预计每 Epoch 更新步数（单 GPU）:",
        estimated_steps_per_epoch,
    )

    print(
        "预计总更新步数:",
        estimated_total_steps,
    )

    if args.dry_run:

        print(
            "\nDry-run 检查完成，不启动训练。"
        )

        return

    use_bf16, use_fp16 = (
        resolve_precision(args)
    )

    print(
        "\n========== 加载初始模型 =========="
    )

    model = CrossEncoder(
        str(args.model),

        num_labels=1,

        max_length=args.max_length,

        device=args.device,

        activation_fn=torch.nn.Identity(),

        local_files_only=True,
    )

    if model.max_length != args.max_length:

        raise RuntimeError(
            "Cross-Encoder 实际最大长度不一致"
        )

    training_args = CrossEncoderTrainingArguments(
        output_dir=str(args.output_dir),

        num_train_epochs=args.epochs,

        max_steps=args.max_steps,

        per_device_train_batch_size=(
            args.batch_size
        ),

        per_device_eval_batch_size=(
            args.batch_size
        ),

        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),

        learning_rate=args.lr,

        warmup_ratio=args.warmup_ratio,

        weight_decay=args.weight_decay,

        max_grad_norm=args.max_grad_norm,

        bf16=use_bf16,

        fp16=use_fp16,

        use_cpu=(
            args.device == "cpu"
        ),

        eval_strategy="steps",

        eval_steps=args.eval_steps,

        save_strategy="steps",

        save_steps=args.save_steps,

        save_total_limit=args.save_total_limit,

        # 不使用 BCE Eval Loss 自动选择模型。
        load_best_model_at_end=False,

        # 默认不保存优化器等训练状态。
        save_only_model=(
            not args.save_training_state
        ),

        logging_steps=args.logging_steps,

        report_to="none",

        seed=args.seed,

        data_seed=args.seed,

        dataloader_num_workers=args.num_workers,

        dataloader_drop_last=False,
    )

    # 保持原 E4 BCE 损失函数。
    loss = BinaryCrossEntropyLoss(model)

    trainer = CrossEncoderTrainer(
        model=model,

        args=training_args,

        train_dataset=train_data,

        eval_dataset=val_data,

        loss=loss,
    )

    print(
        "\n========== 开始训练 ==========",
        flush=True,
    )

    result = trainer.train()

    # 保存最后一步模型，不宣称它是最佳模型。
    final_dir = args.final_model_dir

    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(
        str(final_dir)
    )

    checkpoints = []

    for path in args.output_dir.glob(
        "checkpoint-*"
    ):

        if not path.is_dir():

            continue

        try:

            step = int(
                path.name.split("-")[-1]
            )

        except ValueError:

            continue

        checkpoints.append({
            "step": step,

            "path": str(path.resolve()),
        })

    checkpoints.sort(
        key=lambda item: item["step"]
    )

    final_model_hash = model_hash(
        final_dir
    )

    summary = {
        "experiment_id": args.experiment_id,

        "training_protocol": "E5-v1",

        **source_git_state,

        "initial_model": str(
            args.model.resolve()
        ),

        "initial_model_hash": (
            initial_model_hash
        ),

        "final_model": str(
            final_dir.resolve()
        ),

        "final_model_hash": final_model_hash,

        "checkpoint_dir": str(
            args.output_dir.resolve()
        ),

        "train_file": str(
            args.train_file.resolve()
        ),

        "validation_file": str(
            args.validation_file.resolve()
        ),

        "split_file": str(
            args.split_file.resolve()
        ),

        "data_report": (
            str(args.data_report.resolve())
            if args.data_report is not None
            else None
        ),

        "data_identity": data_identity,

        "loss": "BinaryCrossEntropyLoss",

        "learning_rate": args.lr,

        "batch_size": args.batch_size,

        "gradient_accumulation_steps": (
            args.gradient_accumulation_steps
        ),

        "epochs": args.epochs,

        "max_steps": args.max_steps,

        "global_step": int(
            result.global_step
        ),

        "train_loss": float(
            result.training_loss
        ),

        "max_length": args.max_length,

        "activation": "Identity",

        "bf16": use_bf16,

        "fp16": use_fp16,

        "device": args.device,

        "warmup_ratio": args.warmup_ratio,

        "weight_decay": args.weight_decay,

        "max_grad_norm": args.max_grad_norm,

        "seed": args.seed,

        "min_negatives": args.min_negatives,

        "max_negatives": args.max_negatives,

        "eval_steps": args.eval_steps,

        "save_steps": args.save_steps,

        "save_only_model": (
            not args.save_training_state
        ),

        "data_stats": stats,

        "checkpoints": checkpoints,

        "checkpoint_count": len(checkpoints),

        "model_selection": {
            "status": "pending_external_evaluation",

            "metric": "NDCG@10",

            "split": "reranker-validation",

            "allow_step_zero": True,
        },

        "log_history": trainer.state.log_history,

        "started_at": started_at,

        "finished_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "total_seconds": (
            time.perf_counter() - started
        ),
    }

    # 模型目录报告。
    write_json(
        final_dir / "train_result.json",
        summary,
    )

    # 方便同步 GitHub 的训练报告。
    report_path = (
        args.report_dir
        / f"{args.experiment_id}.json"
    )

    write_json(
        report_path,
        summary,
    )

    print(
        "\n========== 训练完成 =========="
    )

    print(
        "实际训练步数:",
        result.global_step,
    )

    print(
        "保存 Checkpoint:",
        len(checkpoints),
    )

    print(
        "最后一步模型:",
        final_dir,
    )

    print(
        "训练报告:",
        report_path,
    )

    print(
        "注意：最终模型不一定是最佳模型。"
        "请使用固定排序验证集选择 Checkpoint。"
    )


if __name__ == "__main__":
    main()
