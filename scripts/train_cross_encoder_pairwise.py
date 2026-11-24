
#!/usr/bin/env python3
# scripts/train_cross_encoder_pairwise.py

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import RankNetLoss


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--train-file",
        default="data/processed/msmarco/e2_triplets_gpu.jsonl",
    )
    p.add_argument(
        "--model",
        default="models/ms-marco-MiniLM-L6-v2-cross-encoder",
    )
    p.add_argument(
        "--output-dir",
        default="checkpoints/e4_pairwise_50k",
    )
    p.add_argument(
        "--final-model-dir",
        default="models/ms-marco-MiniLM-L6-v2-e4-pairwise",
    )

    p.add_argument("--max-queries", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--eval-steps", type=int, default=500)

    return p.parse_args()


def validation_bucket(query_id, seed):
    key = f"{seed}:{query_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / (2**64)


def load_triplets(path, max_queries, seed, val_ratio):
    if not 0 < val_ratio < 1:
        raise ValueError("val-ratio 必须在 0 到 1 之间")

    if max_queries is not None and max_queries <= 0:
        raise ValueError("max-queries 必须为正整数")

    rng = random.Random(seed)
    selected = []
    eligible = 0

    # 保持和 E4-BCE 相同的蓄水池采样逻辑
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            q = row["query"]
            pos = row["positive"]
            neg = row["negative"]
            qid = str(row["query_id"])

            if not q.strip() or not pos.strip() or not neg.strip():
                continue

            if pos.strip() == neg.strip():
                continue

            item = {
                "query_id": qid,
                "query": q,
                "positive": pos,
                "negative": neg,
            }

            eligible += 1

            if max_queries is None:
                selected.append(item)
            elif len(selected) < max_queries:
                selected.append(item)
            else:
                j = rng.randrange(eligible)
                if j < max_queries:
                    selected[j] = item

    if not selected:
        raise RuntimeError("没有有效训练数据")

    train_rows = []
    val_rows = []
    train_ids = set()
    val_ids = set()

    for row in selected:
        qid = row["query"]

        # 注意：划分依据必须使用 Query ID
        query_id = row["query_id"]

        is_val = (
            validation_bucket(query_id, seed) < val_ratio
        )

        target = val_rows if is_val else train_rows

        if is_val:
            val_ids.add(query_id)
        else:
            train_ids.add(query_id)

        target.append({
            "query": qid,
            "docs": [
                row["positive"],
                row["negative"],
            ],
            "labels": [1.0, 0.0],
        })

    if train_ids & val_ids:
        raise RuntimeError("训练/验证 Query 存在交叉")

    if not train_rows or not val_rows:
        raise RuntimeError("训练或验证数据为空")

    rng.shuffle(train_rows)

    stats = {
        "selected_triplets": len(selected),
        "train_triplets": len(train_rows),
        "validation_triplets": len(val_rows),
        "train_scored_pairs": len(train_rows) * 2,
        "validation_scored_pairs": len(val_rows) * 2,
        "unique_train_queries": len(train_ids),
        "unique_validation_queries": len(val_ids),
    }

    return (
        Dataset.from_list(train_rows),
        Dataset.from_list(val_rows),
        stats,
    )


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.output_dir == args.final_model_dir:
        raise ValueError(
            "Checkpoint 和最终模型目录不能相同"
        )

    print("加载 Pairwise 训练数据...", flush=True)

    train_dataset, val_dataset, stats = load_triplets(
        args.train_file,
        args.max_queries,
        args.seed,
        args.val_ratio,
    )

    print(
        json.dumps(stats, ensure_ascii=False, indent=2),
        flush=True,
    )

    # Pairwise 必须使用原始分数。
    # Identity 避免模型在排序 Loss 之前应用 Sigmoid。
    print("加载原始 Cross-Encoder...", flush=True)

    model = CrossEncoder(
        args.model,
        num_labels=1,
        max_length=args.max_length,
        activation_fn=torch.nn.Identity(),
        local_files_only=True,
    )

    bf16_enabled = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    training_args = CrossEncoderTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=bf16_enabled,
        fp16=False,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=4,
    )

    loss = RankNetLoss(
        model,
        sigma=1.0,
        reduction_log="natural",
        activation_fn=torch.nn.Identity(),
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
    )

    print("开始 E4-Pairwise 训练...", flush=True)

    train_result = trainer.train()

    final_dir = Path(args.final_model_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_dir))

    result = {
        "experiment_id": "E4-Pairwise",
        "loss": "RankNetLoss",
        "initial_model": args.model,
        "final_model": str(final_dir),
        "train_file": args.train_file,
        "seed": args.seed,
        "learning_rate": args.lr,
        "batch_size_triplets": args.batch_size,
        "batch_size_scored_pairs": args.batch_size * 2,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "bf16": bf16_enabled,
        "train_loss": float(train_result.training_loss),
        "global_step": int(train_result.global_step),
        "data_stats": stats,
        "log_history": trainer.state.log_history,
    }

    result_path = final_dir / "train_result.json"

    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n训练完成")
    print("模型目录:", final_dir)
    print("训练日志:", result_path)


if __name__ == "__main__":
    main()
