#E1：不加入负样本的微调 MiniLM

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers
from transformers.trainer_utils import get_last_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(8 * 1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/all-MiniLM-L6-v2",
    )

    parser.add_argument(
        "--train-file",
        type=Path,
        default=ROOT / "data/processed/msmarco/e1_train.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "checkpoints/e1_mnrl",
    )

    parser.add_argument(
        "--final-model-dir",
        type=Path,
        default=ROOT / "models/minilm-msmarco-e1",
    )

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    if (
        args.batch_size <= 0
        or args.epochs <= 0
        or args.lr <= 0
        or args.logging_steps <= 0
        or args.save_steps <= 0
        or args.max_length <= 0
        or args.max_steps == 0
        or args.max_steps < -1
    ):
        parser.error("训练参数无效")

    if not 0 <= args.warmup_ratio <= 1:
        parser.error("warmup-ratio 必须在 0 到 1 之间")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    output_dir = args.output_dir.resolve()
    final_model_dir = args.final_model_dir.resolve()
    train_file = args.train_file.resolve()

    if not train_file.is_file():
        raise FileNotFoundError(train_file)

    if not args.model.is_dir():
        raise FileNotFoundError(args.model)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 禁止在已有 Checkpoint 目录中意外重新开始训练。
    last_checkpoint = get_last_checkpoint(str(output_dir))

    if args.resume:
        if last_checkpoint is None:
            raise RuntimeError(
                "指定了 --resume，但没有发现 Checkpoint"
            )
    elif last_checkpoint is not None:
        raise RuntimeError(
            "已存在 Checkpoint。恢复训练请使用 --resume；"
            "新实验请更换 --output-dir。"
        )

    if final_model_dir.exists() and not args.resume:
        raise FileExistsError(
            f"最终模型目录已存在: {final_model_dir}"
        )

    print("读取训练数据...", flush=True)

    dataset = load_dataset(
        "json",
        data_files=str(train_file),
        split="train",
    )

    required_columns = {
        "query_id", "positive_id", "query", "positive"
    }

    if not required_columns.issubset(dataset.column_names):
        raise ValueError(
            "训练文件缺少必要字段: "
            + str(required_columns - set(dataset.column_names))
        )

    # 非文本 ID 只用于实验追踪，不传给 MNRL。
    # MNRL 接收两列：anchor Query、positive Passage。
    dataset = dataset.select_columns([
        "query", "positive"
    ])

    print("训练样本:", len(dataset), flush=True)

    print("加载 MiniLM...", flush=True)

    model = SentenceTransformer(
        str(args.model.resolve()),
        device=args.device,
        local_files_only=True,
    )

    model.max_seq_length = args.max_length

    loss = losses.MultipleNegativesRankingLoss(model)

    use_bf16 = (
        args.device.startswith("cuda")
        and torch.cuda.is_bf16_supported()
    )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        bf16=use_bf16,
        fp16=False,
        seed=args.seed,
        data_seed=args.seed,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_num_workers=2,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        eval_strategy="no",
        load_best_model_at_end=False,
        remove_unused_columns=False,
    )

    config = {
        "experiment_id": "E1",
        "initial_model": str(args.model.resolve()),
        "train_file": str(train_file),
        "train_sha256": sha256(train_file),
        "num_train_samples": len(dataset),
        "loss": "MultipleNegativesRankingLoss",
        "batch_sampler": "NO_DUPLICATES",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "max_seq_length": args.max_length,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "bf16": use_bf16,
        "checkpoint_every_steps": args.save_steps,
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "device": args.device,
        "gpu": (
            torch.cuda.get_device_name(0)
            if args.device.startswith("cuda")
            else "CPU"
        ),
        "started_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    config_path = output_dir / "experiment_config.json"

    if args.resume:
        previous_config = json.loads(
            config_path.read_text(encoding="utf-8")
        )

        immutable_keys = [
            "initial_model", "train_sha256",
            "num_train_samples", "loss",
            "batch_sampler", "batch_size",
            "epochs", "learning_rate",
            "max_seq_length", "warmup_ratio",
            "weight_decay", "seed", "max_steps",
        ]

        for key in immutable_keys:
            if previous_config[key] != config[key]:
                raise RuntimeError(
                    f"恢复训练参数不一致: {key}"
                )
    else:
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
    )

    print("\n========== E1 训练 ==========")
    print("GPU:", config["gpu"])
    print("训练数据:", len(dataset))
    print("Batch Size:", args.batch_size)
    print("Epoch:", args.epochs)
    print("BF16:", use_bf16)
    print("Checkpoint:", output_dir)
    print("Resume:", last_checkpoint if args.resume else "否")

    t0 = time.perf_counter()

    train_result = trainer.train(
        resume_from_checkpoint=(
            last_checkpoint if args.resume else None
        )
    )

    elapsed = time.perf_counter() - t0

    if final_model_dir.exists() and any(final_model_dir.iterdir()):
        raise FileExistsError(
            "最终模型目录非空，拒绝覆盖"
        )

    final_model_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_model_dir))
    trainer.save_state()

    result = {
        **config,
        "train_loss": float(train_result.training_loss),
        "global_step": int(trainer.state.global_step),
        "training_seconds": elapsed,
        "finished_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "final_model": str(final_model_dir),
        "validation_metrics": None,
        "note": (
            "当前仅完成训练；Dev Small 全库验证"
            "需要重新编码 Corpus 并构建 FAISS 索引"
        ),
    }

    result_path = output_dir / "train_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nE1 训练完成")
    print("训练 Loss:", result["train_loss"])
    print("Global Step:", result["global_step"])
    print("训练耗时(秒):", elapsed)
    print("微调模型:", final_model_dir)
    print("训练报告:", result_path)


if __name__ == "__main__":
    main()
