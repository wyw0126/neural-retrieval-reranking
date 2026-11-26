
#!/usr/bin/env python3
# scripts/training/reranker/train_cross_encoder_listwise.py

import argparse
import hashlib
import json
import math
import random
import subprocess
import time

from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from transformers import get_linear_schedule_with_warmup
from sentence_transformers import CrossEncoder


ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# 1. 命令行参数
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description="Listwise 多负例 Cross-Encoder 训练"
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

    p.add_argument(
        "--data-report",
        type=Path,
        default=None,
    )

    p.add_argument(
        "--model",
        type=Path,
        default=Path(
            "models/ms-marco-MiniLM-L6-v2-cross-encoder"
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
        default=Path("reports/training"),
    )

    # 这里的 Batch Size 指 Query 分组数量，
    # 而不是独立 Query-Passage 对数量。
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
    )

    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
    )

    p.add_argument(
        "--max-steps",
        type=int,
        default=1000,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=2e-6,
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

    # 教师 KL 约束权重。
    # 默认关闭，执行纯 Listwise 训练。
    p.add_argument(
        "--teacher-kl-weight",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--temperature",
        type=float,
        default=2.0,
    )

    p.add_argument(
        "--eval-steps",
        type=int,
        default=50,
    )

    p.add_argument(
        "--save-steps",
        type=int,
        default=50,
    )

    p.add_argument(
        "--logging-steps",
        type=int,
        default=10,
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
        p.error("负例数量必须在 1–4 之间")

    for name in (
        "batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "max_length",
        "eval_steps",
        "save_steps",
        "logging_steps",
    ):

        if getattr(args, name) <= 0:
            p.error(f"{name} 必须大于 0")

    if not math.isfinite(args.lr) or args.lr <= 0:
        p.error("lr 必须为有限正数")

    if not 0 <= args.warmup_ratio <= 1:
        p.error("warmup-ratio 必须在 0–1 之间")

    if not (
        math.isfinite(args.teacher_kl_weight)
        and args.teacher_kl_weight >= 0
    ):
        p.error("teacher-kl-weight 非法")

    if not (
        math.isfinite(args.temperature)
        and args.temperature > 0
    ):
        p.error("temperature 必须大于 0")

    if not (
        math.isfinite(args.weight_decay)
        and args.weight_decay >= 0
    ):
        p.error("weight-decay 非法")

    if not (
        math.isfinite(args.max_grad_norm)
        and args.max_grad_norm > 0
    ):
        p.error("max-grad-norm 必须大于 0")

    if Path(args.experiment_id).name != args.experiment_id:
        p.error("experiment-id 不能包含路径")

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

    model_dir = Path(model_dir).resolve()

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

    files = []

    for path in model_dir.rglob("*"):

        if not path.is_file():
            continue

        if path.name in ignored:
            continue

        relative = path.relative_to(model_dir)

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

        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")

        digest.update(
            sha256_file(path).encode("ascii")
        )

        digest.update(b"\0")

    return digest.hexdigest()


def normalize_text(text):

    return " ".join(
        text.casefold().split()
    )


def git_state():

    try:

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
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


def set_seed(seed):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. 路径检查
# ============================================================

def verify_paths(args):

    required_files = [
        args.train_file,
        args.validation_file,
        args.split_file,
    ]

    if args.data_report is not None:
        required_files.append(args.data_report)

    for path in required_files:

        if not path.is_file():
            raise FileNotFoundError(path)

    if not args.model.is_dir():
        raise FileNotFoundError(args.model)

    if (
        args.train_file.resolve()
        == args.validation_file.resolve()
    ):
        raise ValueError("训练文件与验证文件相同")

    model_dir = args.model.resolve()

    output_dir = args.output_dir.resolve()

    final_dir = args.final_model_dir.resolve()

    for directory in (output_dir, final_dir):

        if (
            directory == model_dir
            or directory.is_relative_to(model_dir)
            or model_dir.is_relative_to(directory)
        ):
            raise ValueError("输出目录与初始模型重叠")

    if (
        output_dir == final_dir
        or output_dir.is_relative_to(final_dir)
        or final_dir.is_relative_to(output_dir)
    ):
        raise ValueError("两个输出目录不能重叠")

    if args.dry_run:
        return

    for directory in (output_dir, final_dir):

        if directory.exists():

            if not directory.is_dir():
                raise FileExistsError(directory)

            if any(directory.iterdir()):
                raise FileExistsError(
                    f"输出目录非空: {directory}"
                )

    report_path = (
        args.report_dir
        / f"{args.experiment_id}.json"
    )

    if report_path.exists():
        raise FileExistsError(report_path)


# ============================================================
# 4. 数据身份校验
# ============================================================

def verify_data_identity(args):

    split = json.loads(
        args.split_file.read_text(
            encoding="utf-8"
        )
    )

    split_hash = sha256_file(args.split_file)

    train_hash = sha256_file(args.train_file)

    val_hash = sha256_file(args.validation_file)

    if val_hash != split["validation_sha256"]:

        raise RuntimeError(
            "验证集 SHA256 校验失败"
        )

    train_qids = {
        int(qid)
        for qid in split["train_qids"]
    }

    val_qids = {
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

    if train_qids & ranking_val_qids:

        raise RuntimeError(
            "训练与排序验证 Query 重叠"
        )

    report = None

    threshold = None

    if args.data_report is not None:

        report = json.loads(
            args.data_report.read_text(
                encoding="utf-8"
            )
        )

        if train_hash != report["output_sha256"]:
            raise RuntimeError("训练数据哈希不一致")

        if split_hash != report["split_sha256"]:
            raise RuntimeError("划分哈希不一致")

        if val_hash != report["validation_sha256"]:
            raise RuntimeError("验证数据哈希不一致")

        threshold = float(
            report["margin_threshold"]
        )

        if not math.isfinite(threshold):
            raise RuntimeError("非法 Margin 阈值")

        dataset_type = "margin_filtered"

    else:

        if train_hash != split["train_sha256"]:

            raise RuntimeError(
                "训练文件身份校验失败；"
                "筛选数据必须提供 --data-report"
            )

        dataset_type = "unfiltered"

    identity = {
        "dataset_type": dataset_type,
        "train_sha256": train_hash,
        "validation_sha256": val_hash,
        "split_sha256": split_hash,
        "margin_threshold": threshold,

        "data_report_sha256": (
            sha256_file(args.data_report)
            if args.data_report is not None
            else None
        ),
    }

    return (
        identity,
        train_qids,
        val_qids,
        ranking_val_qids,
        report,
    )


# ============================================================
# 5. 加载 Query 分组数据
# ============================================================

def load_groups(
    path,
    min_negatives=1,
    max_negatives=4,
    margin_threshold=None,
):

    groups = []

    seen_qids = set()

    with open(path, encoding="utf-8") as f:

        for line_no, line in enumerate(f, 1):

            if not line.strip():
                continue

            row = json.loads(line)

            qid = int(row["query_id"])

            query = row["query"].strip()

            positive = row["positive"].strip()

            positive_score = float(
                row["positive_score"]
            )

            positive_id = row.get("positive_id")

            negatives = row["negatives"]

            if qid in seen_qids:

                raise ValueError(
                    f"重复 Query ID: {qid}"
                )

            seen_qids.add(qid)

            if not query or not positive:

                raise ValueError(
                    f"第 {line_no} 行正文为空"
                )

            if not math.isfinite(positive_score):

                raise ValueError(
                    f"Query {qid} 正例分数非法"
                )

            if not (
                min_negatives
                <= len(negatives)
                <= max_negatives
            ):

                raise ValueError(
                    f"Query {qid} 负例数量非法"
                )

            # 正例始终放在候选列表的第一个位置。
            passages = [positive]

            teacher_scores = [positive_score]

            seen_pids = set()

            seen_texts = {
                normalize_text(positive)
            }

            for neg in negatives:

                pid = int(neg["passage_id"])

                text = neg["text"].strip()

                score = float(
                    neg["teacher_score"]
                )

                margin = float(neg["margin"])

                if pid < 0 or not text:

                    raise ValueError(
                        f"Query {qid} 负例非法"
                    )

                if not all(
                    math.isfinite(x)
                    for x in (score, margin)
                ):

                    raise ValueError(
                        f"Query {qid} 分数非法"
                    )

                if not math.isclose(
                    positive_score - score,
                    margin,
                    rel_tol=1e-7,
                    abs_tol=1e-6,
                ):

                    raise ValueError(
                        f"Query {qid} Margin 不一致"
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

                normalized = normalize_text(text)

                if normalized in seen_texts:

                    raise ValueError(
                        f"Query {qid} 存在重复文本"
                    )

                if (
                    margin_threshold is not None
                    and margin <= margin_threshold
                ):

                    raise ValueError(
                        f"Query {qid} 未通过 Margin 筛选"
                    )

                seen_pids.add(pid)

                seen_texts.add(normalized)

                passages.append(text)

                teacher_scores.append(score)

            groups.append({
                "query_id": qid,
                "query": query,
                "passages": passages,
                "teacher_scores": teacher_scores,
            })

    if not groups:

        raise RuntimeError(
            f"数据文件为空: {path}"
        )

    return groups


def summarize_groups(groups):

    distribution = Counter()

    negative_count = 0

    for row in groups:

        count = len(row["passages"]) - 1

        distribution[count] += 1

        negative_count += count

    return {
        "queries": len(groups),

        "positive_count": len(groups),

        "negative_count": negative_count,

        "pair_count": (
            len(groups) + negative_count
        ),

        "negative_distribution": dict(
            sorted(distribution.items())
        ),

        "mean_negatives_per_query": (
            negative_count / len(groups)
        ),
    }


def load_data(args):

    (
        identity,
        expected_train_qids,
        expected_val_qids,
        ranking_val_qids,
        report,
    ) = verify_data_identity(args)

    train_groups = load_groups(
        args.train_file,
        min_negatives=args.min_negatives,
        max_negatives=args.max_negatives,
        margin_threshold=identity[
            "margin_threshold"
        ],
    )

    val_groups = load_groups(
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
        raise RuntimeError("训练与排序验证数据泄漏")

    if val_qids != expected_val_qids:
        raise RuntimeError("验证 Query 集合不一致")

    if not train_qids.issubset(
        expected_train_qids
    ):
        raise RuntimeError("存在非法训练 Query")

    if identity["dataset_type"] == "unfiltered":

        if train_qids != expected_train_qids:
            raise RuntimeError("原始训练划分不完整")

    train_stats = summarize_groups(train_groups)

    val_stats = summarize_groups(val_groups)

    if report is not None:

        checks = {
            "retained_queries":
                train_stats["queries"],

            "retained_negatives":
                train_stats["negative_count"],

            "training_pairs":
                train_stats["pair_count"],
        }

        for key, actual in checks.items():

            if actual != report[key]:

                raise RuntimeError(
                    f"数据统计不一致: {key}"
                )

    stats = {
        "train": train_stats,
        "validation": val_stats,

        "ranking_validation_query_count": (
            len(ranking_val_qids)
        ),

        "query_overlap_count": 0,

        "dropped_train_queries": (
            len(expected_train_qids - train_qids)
        ),
    }

    return (
        train_groups,
        val_groups,
        identity,
        stats,
    )


# ============================================================
# 6. Dataset 与变长分组 Collator
# ============================================================

class QueryGroupDataset(Dataset):

    def __init__(self, groups):

        self.groups = groups

    def __len__(self):

        return len(self.groups)

    def __getitem__(self, index):

        return self.groups[index]


class GroupCollator:

    def __init__(self, tokenizer, max_length):

        self.tokenizer = tokenizer

        self.max_length = max_length

    def __call__(self, groups):

        queries = []

        passages = []

        teacher_scores = []

        group_sizes = []

        for row in groups:

            group_size = len(row["passages"])

            group_sizes.append(group_size)

            queries.extend(
                [row["query"]] * group_size
            )

            passages.extend(
                row["passages"]
            )

            teacher_scores.extend(
                row["teacher_scores"]
            )

        # 所有 Query-Passage 对独立 Tokenize。
        # Padding 只针对 Token 序列长度，
        # 不针对候选文档数量。
        features = self.tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "features": features,

            "group_sizes": group_sizes,

            "teacher_scores": torch.tensor(
                teacher_scores,
                dtype=torch.float32,
            ),
        }


# ============================================================
# 7. Listwise Loss
# ============================================================

def compute_listwise_loss(
    student_scores,
    teacher_scores,
    group_sizes,
    teacher_kl_weight=0.0,
    temperature=2.0,
):
    # Loss 使用 FP32，提升数值稳定性。
    student_scores = (
        student_scores.reshape(-1).float()
    )
    teacher_scores = (
        teacher_scores.reshape(-1).float()
    )
    if sum(group_sizes) != student_scores.numel():
        raise RuntimeError(
            "学生评分与候选分组数量不一致"
        )

    if teacher_scores.numel() != student_scores.numel():
        raise RuntimeError(
            "教师评分与学生评分数量不一致"
        )

    student_groups = torch.split(
        student_scores,
        group_sizes,
    )

    teacher_groups = torch.split(
        teacher_scores,
        group_sizes,
    )

    listwise_losses = []

    kl_losses = []

    for student, teacher in zip(
        student_groups,
        teacher_groups,
    ):
        if student.numel() < 2:
            raise RuntimeError(
                "每个 Query 至少需要两个候选"
            )

        # 正例始终位于 index=0。
        # -log Softmax(student)[0]
        # 等价于：
        # logsumexp(student) - student[0]

        hard_loss = (
            torch.logsumexp(student, dim=0)
            - student[0]
        )
        listwise_losses.append(hard_loss)
        # 可选教师分布约束。
        if teacher_kl_weight > 0:
            teacher_probs = F.softmax(
                teacher.detach() / temperature,
                dim=0,
            )
            student_log_probs = F.log_softmax(
                student / temperature,
                dim=0,
            )
            kl_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="sum",
            )
            kl_loss = (
                kl_loss * temperature ** 2
            )
            kl_losses.append(kl_loss)

    # 每个 Query 等权重，而不是每个 Passage 等权重。
    listwise_loss = torch.stack(
        listwise_losses
    ).mean()
    if kl_losses:
        teacher_kl_loss = torch.stack(
            kl_losses
        ).mean()
    else:
        teacher_kl_loss = torch.zeros(
            (),
            device=student_scores.device,
            dtype=torch.float32,
        )

    total_loss = (
        listwise_loss
        + teacher_kl_weight * teacher_kl_loss
    )

    return (
        total_loss,
        listwise_loss,
        teacher_kl_loss,
    )


# ============================================================
# 8. 精度与模型前向传播
# ============================================================

def resolve_bf16(args):
    if args.device == "cpu":
        if args.precision == "bf16":
            raise ValueError(
                "当前脚本不启用 CPU BF16"
            )
        return False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    supported = torch.cuda.is_bf16_supported()

    if args.precision == "bf16":
        if not supported:
            raise RuntimeError("GPU 不支持 BF16")
        return True

    if args.precision == "fp32":
        return False
    return supported


def autocast_context(device, use_bf16):

    if use_bf16:
        return torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
        )
    return nullcontext()


def forward_batch(
    encoder,
    batch,
    device,
    use_bf16,
):
    features = {
        key: value.to(device)
        for key, value in batch["features"].items()
    }
    teacher_scores = (
        batch["teacher_scores"].to(device)
    )
    with autocast_context(
        device,
        use_bf16,
    ):
        # 直接使用底层模型原始 logits，
        # 不使用 Sigmoid，不使用 predict()。
        outputs = encoder.model(**features)
        student_scores = (
            outputs.logits.reshape(-1)
        )

    return (
        student_scores.float(),
        teacher_scores,
        batch["group_sizes"],
    )


# ============================================================
# 9. 分组验证
# ============================================================

@torch.no_grad()
def evaluate_groups(
    encoder,
    loader,
    device,
    use_bf16,
    args,
):

    previous_mode = encoder.model.training
    encoder.model.eval()
    total_queries = 0
    total_loss = 0.0
    total_listwise = 0.0
    total_kl = 0.0
    total_bce = 0.0
    total_pairs = 0
    top1_correct = 0
    reciprocal_rank = 0.0
    ndcg_sum = 0.0
    positive_score_sum = 0.0
    negative_score_sum = 0.0
    negative_count = 0
    margin_sum = 0.0
    conflicts = 0

    for batch in loader:
        (
            student_scores,
            teacher_scores,
            sizes,
        ) = forward_batch(
            encoder,
            batch,
            device,
            use_bf16,
        )
        (
            loss,
            listwise,
            kl,
        ) = compute_listwise_loss(
            student_scores,
            teacher_scores,
            sizes,
            teacher_kl_weight=(
                args.teacher_kl_weight
            ),
            temperature=args.temperature,
        )

        batch_queries = len(sizes)
        total_queries += batch_queries
        total_loss += loss.item() * batch_queries
        total_listwise += (
            listwise.item() * batch_queries
        )
        total_kl += kl.item() * batch_queries
        groups = torch.split(
            student_scores,
            sizes,
        )
        labels = torch.zeros_like(
            student_scores
        )
        offset = 0
        for scores in groups:
            labels[offset] = 1.0
            offset += scores.numel()
            positive = scores[0]
            negatives = scores[1:]
            # 同分负例按排在正例之前处理，
            # 避免人为给正例同分优势。
            rank = 1 + int(
                (negatives >= positive)
                .sum()
                .item()
            )
            if rank == 1:
                top1_correct += 1
            reciprocal_rank += 1.0 / rank
            ndcg_sum += (
                1.0 / math.log2(rank + 1)
            )
            positive_score_sum += (
                positive.item()
            )
            negative_score_sum += (
                negatives.sum().item()
            )
            negative_count += negatives.numel()
            margins = positive - negatives
            margin_sum += margins.sum().item()
            conflicts += int(
                (margins <= 0)
                .sum()
                .item()
            )

        # 同时计算 BCE 验证损失，
        bce = F.binary_cross_entropy_with_logits(
            student_scores,
            labels,
            reduction="sum",
        )
        total_bce += bce.item()
        total_pairs += student_scores.numel()
    encoder.model.train(previous_mode)

    if total_queries == 0:
        raise RuntimeError("验证集为空")

    return {
        "eval_loss": total_loss / total_queries,
        "eval_listwise_loss": (
            total_listwise / total_queries
        ),
        "eval_teacher_kl_loss": (
            total_kl / total_queries
        ),
        "eval_bce_loss": (
            total_bce / total_pairs
        ),
        "group_top1_accuracy": (
            top1_correct / total_queries
        ),
        "group_mrr": (
            reciprocal_rank / total_queries
        ),
        "group_ndcg": (
            ndcg_sum / total_queries
        ),
        "mean_positive_score": (
            positive_score_sum / total_queries
        ),
        "mean_negative_score": (
            negative_score_sum / negative_count
        ),
        "mean_margin": (
            margin_sum / negative_count
        ),
        "negative_conflict_ratio": (
            conflicts / negative_count
        ),
        "query_count": total_queries,
        "pair_count": total_pairs,
        "notes": (
            "这些排序指标来自固定的 1–4 "
            "负例分组，不等价于 Top100 "
            "真实检索排序指标。"
        ),
    }


# ============================================================
# 10. 优化器
# ============================================================

def build_optimizer(encoder, args):

    decay = []
    no_decay = []

    for name, param in (
        encoder.model.named_parameters()
    ):
        if not param.requires_grad:
            continue

        lower_name = name.lower()
        if (
            lower_name.endswith("bias")
            or "layernorm" in lower_name
            or "layer_norm" in lower_name
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    parameter_groups = [
        {
            "params": decay,
            "weight_decay": args.weight_decay,
        },
        {
            "params": no_decay,
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    warmup_steps = math.ceil(
        args.max_steps * args.warmup_ratio
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_steps,
    )

    return (
        optimizer,
        scheduler,
        warmup_steps,
    )


# ============================================================
# 11. 保存模型
# ============================================================

def save_model(
    encoder,
    output_dir,
    step,
    evaluation=None,
):

    output_dir = Path(output_dir)

    if output_dir.exists():

        if any(output_dir.iterdir()):

            raise FileExistsError(
                f"模型目录非空: {output_dir}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 保存完整 CrossEncoder 格式，
    # 包含底层模型、Tokenizer 以及相关配置。
    encoder.save(str(output_dir))

    write_json(
        output_dir / "training_step.json",
        {
            "global_step": step,
            "evaluation": evaluation,
        },
    )

    print(
        f"Saved checkpoint: {output_dir}",
        flush=True,
    )


# ============================================================
# 12. 主训练流程
# ============================================================

def main():

    args = parse_args()

    verify_paths(args)

    set_seed(args.seed)

    start_time = time.perf_counter()

    started_at = datetime.now(
        timezone.utc
    ).isoformat()

    source_git_state = git_state()

    print(
        "========== 加载训练数据 ==========",
        flush=True,
    )

    (
        train_groups,
        val_groups,
        identity,
        data_stats,
    ) = load_data(args)

    initial_model_hash = model_hash(
        args.model
    )

    print(
        json.dumps(
            data_stats,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    print(
        json.dumps(
            identity,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    effective_query_batch = (
        args.batch_size
        * args.gradient_accumulation_steps
    )

    estimated_queries = (
        args.max_steps * effective_query_batch
    )

    estimated_pairs = (
        estimated_queries
        * (
            1 + data_stats["train"][
                "mean_negatives_per_query"
            ]
        )
    )

    print(
        "每次更新 Query 数量:",
        effective_query_batch,
    )

    print(
        "预计处理 Query 数量:",
        estimated_queries,
    )

    print(
        "预计处理样本对:",
        round(estimated_pairs),
    )

    if args.dry_run:

        print(
            "Dry-run 完成，不启动训练。",
            flush=True,
        )

        return

    device = torch.device(args.device)

    use_bf16 = resolve_bf16(args)

    print(
        "\n========== 加载初始模型 ==========",
        flush=True,
    )

    encoder = CrossEncoder(
        str(args.model),

        num_labels=1,

        max_length=args.max_length,

        device=str(device),

        activation_fn=torch.nn.Identity(),

        local_files_only=True,
    )

    encoder.model.to(device)

    encoder.model.train()

    if encoder.max_length != args.max_length:

        raise RuntimeError(
            "CrossEncoder 最大输入长度不一致"
        )

    collator = GroupCollator(
        encoder.tokenizer,
        args.max_length,
    )

    train_dataset = QueryGroupDataset(
        train_groups
    )

    val_dataset = QueryGroupDataset(
        val_groups
    )

    generator = torch.Generator()

    generator.manual_seed(args.seed)

    if len(train_dataset) < args.batch_size:

        raise RuntimeError(
            "训练组数量小于 Batch Size"
        )

    train_loader = DataLoader(
        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

        generator=generator,

        collate_fn=collator,

        num_workers=0,

        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=args.eval_batch_size,

        shuffle=False,

        collate_fn=collator,

        num_workers=0,

        drop_last=False,
    )

    (
        optimizer,
        scheduler,
        warmup_steps,
    ) = build_optimizer(
        encoder,
        args,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        args.output_dir / "training_log.jsonl"
    )

    if log_path.exists():
        raise FileExistsError(log_path)

    history = []

    checkpoints = []

    def log_event(event):

        history.append(event)

        with log_path.open(
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    allow_nan=False,
                ) + "\n"
            )

        print(
            json.dumps(
                event,
                ensure_ascii=False,
            ),
            flush=True,
        )

    print(
        "\n========== Step 0 验证 ==========",
        flush=True,
    )

    # 补充原始模型在固定分组验证集上的指标。
    # 同时获得 Step 0 的 BCE Loss。
    step0_metrics = evaluate_groups(
        encoder,
        val_loader,
        device,
        use_bf16,
        args,
    )

    log_event({
        "event": "evaluation",
        "step": 0,
        **step0_metrics,
    })

    encoder.model.train()

    iterator = iter(train_loader)

    global_step = 0

    data_passes = 1

    processed_queries = 0

    processed_pairs = 0

    cumulative_loss = 0.0

    window_loss = 0.0

    window_listwise = 0.0

    window_kl = 0.0

    window_steps = 0

    print(
        "\n========== 开始 Listwise 训练 ==========",
        flush=True,
    )

    while global_step < args.max_steps:

        optimizer.zero_grad(
            set_to_none=True
        )

        step_total_loss = 0.0

        step_listwise_loss = 0.0

        step_kl_loss = 0.0

        for _ in range(
            args.gradient_accumulation_steps
        ):

            try:

                batch = next(iterator)

            except StopIteration:

                data_passes += 1

                iterator = iter(train_loader)

                batch = next(iterator)

            (
                student_scores,
                teacher_scores,
                sizes,
            ) = forward_batch(
                encoder,
                batch,
                device,
                use_bf16,
            )

            (
                loss,
                hard_loss,
                kl_loss,
            ) = compute_listwise_loss(
                student_scores,
                teacher_scores,
                sizes,

                teacher_kl_weight=(
                    args.teacher_kl_weight
                ),

                temperature=args.temperature,
            )

            if not torch.isfinite(loss):

                raise RuntimeError(
                    f"Step {global_step} Loss 非法"
                )

            scaled_loss = (
                loss
                / args.gradient_accumulation_steps
            )

            scaled_loss.backward()

            processed_queries += len(sizes)

            processed_pairs += sum(sizes)

            step_total_loss += loss.item()

            step_listwise_loss += hard_loss.item()

            step_kl_loss += kl_loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            encoder.model.parameters(),

            args.max_grad_norm,

            error_if_nonfinite=True,
        )

        used_lr = optimizer.param_groups[0]["lr"]

        optimizer.step()

        scheduler.step()

        global_step += 1

        step_total_loss /= (
            args.gradient_accumulation_steps
        )

        step_listwise_loss /= (
            args.gradient_accumulation_steps
        )

        step_kl_loss /= (
            args.gradient_accumulation_steps
        )

        cumulative_loss += step_total_loss

        window_loss += step_total_loss

        window_listwise += step_listwise_loss

        window_kl += step_kl_loss

        window_steps += 1

        is_last = (
            global_step == args.max_steps
        )

        if (
            global_step % args.logging_steps == 0
            or is_last
        ):

            log_event({
                "event": "training",

                "step": global_step,

                "loss": (
                    window_loss / window_steps
                ),

                "listwise_loss": (
                    window_listwise / window_steps
                ),

                "teacher_kl_loss": (
                    window_kl / window_steps
                ),

                "learning_rate": used_lr,

                "grad_norm": float(grad_norm),

                "processed_queries": (
                    processed_queries
                ),

                "processed_pairs": (
                    processed_pairs
                ),

                "data_passes": data_passes,
            })

            window_loss = 0.0

            window_listwise = 0.0

            window_kl = 0.0

            window_steps = 0

        evaluation = None

        if (
            global_step % args.eval_steps == 0
            or is_last
        ):

            evaluation = evaluate_groups(
                encoder,
                val_loader,
                device,
                use_bf16,
                args,
            )

            log_event({
                "event": "evaluation",

                "step": global_step,

                **evaluation,
            })

        if (
            global_step % args.save_steps == 0
            or is_last
        ):

            checkpoint_dir = (
                args.output_dir
                / f"checkpoint-{global_step}"
            )

            save_model(
                encoder,

                checkpoint_dir,

                global_step,

                evaluation,
            )

            checkpoints.append({
                "step": global_step,

                "path": str(
                    checkpoint_dir.resolve()
                ),
            })

    # ------------------------------------------------
    # 保存最后一步模型。
    # 不使用分组验证指标选出所谓最佳模型。
    # ------------------------------------------------

    final_metrics = evaluate_groups(
        encoder,
        val_loader,
        device,
        use_bf16,
        args,
    )

    save_model(
        encoder,

        args.final_model_dir,

        global_step,

        final_metrics,
    )

    final_hash = model_hash(
        args.final_model_dir
    )

    summary = {
        "experiment_id": args.experiment_id,

        "training_protocol": "E6-Listwise-v1",

        **source_git_state,

        "initial_model": str(
            args.model.resolve()
        ),

        "initial_model_hash": (
            initial_model_hash
        ),

        "final_model": str(
            args.final_model_dir.resolve()
        ),

        "final_model_hash": final_hash,

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

        "data_identity": identity,

        "data_stats": data_stats,

        "loss": (
            "ListwiseSoftmax"
            if args.teacher_kl_weight == 0
            else "ListwiseSoftmax+TeacherKL"
        ),

        "teacher_kl_weight": (
            args.teacher_kl_weight
        ),

        "temperature": args.temperature,

        "batch_size_unit": "query_group",

        "batch_size": args.batch_size,

        "gradient_accumulation_steps": (
            args.gradient_accumulation_steps
        ),

        "effective_query_batch_size": (
            effective_query_batch
        ),

        "learning_rate": args.lr,

        "warmup_steps": warmup_steps,

        "warmup_ratio": args.warmup_ratio,

        "weight_decay": args.weight_decay,

        "max_grad_norm": args.max_grad_norm,

        "max_steps": args.max_steps,

        "global_step": global_step,

        "max_length": args.max_length,

        "activation": "Identity",

        "device": args.device,

        "bf16": use_bf16,

        "seed": args.seed,

        "processed_queries": processed_queries,

        "processed_pairs": processed_pairs,

        "mean_training_loss": (
            cumulative_loss / global_step
        ),

        "eval_steps": args.eval_steps,

        "save_steps": args.save_steps,

        "checkpoint_count": len(checkpoints),

        "checkpoints": checkpoints,

        "step0_metrics": step0_metrics,

        "final_group_metrics": final_metrics,

        "model_selection": {
            "status": "pending_external_evaluation",

            "split": "reranker-validation",

            "metric": "NDCG@10",

            "allow_step_zero": True,
        },

        "log_history": history,

        "started_at": started_at,

        "finished_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "total_seconds": (
            time.perf_counter() - start_time
        ),
    }

    write_json(
        args.final_model_dir / "train_result.json",
        summary,
    )

    report_path = (
        args.report_dir
        / f"{args.experiment_id}.json"
    )

    write_json(
        report_path,
        summary,
    )

    print(
        "\n========== 训练完成 ==========",
        flush=True,
    )

    print("训练步数:", global_step)

    print("处理 Query 数量:", processed_queries)

    print("处理样本对:", processed_pairs)

    print("Checkpoint 数量:", len(checkpoints))

    print("最后一步模型:", args.final_model_dir)

    print("训练报告:", report_path)

    print(
        "请使用固定 Top100 排序验证集"
        "进行最终模型选择。"
    )


if __name__ == "__main__":

    main()
