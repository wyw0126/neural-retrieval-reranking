
#!/usr/bin/env python3
# scripts/train_cross_encoder_multi.py

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
from sentence_transformers.cross_encoder.losses import (
    BinaryCrossEntropyLoss,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--train-file",
        default="data/processed/msmarco/e4_e2_multi_50k.jsonl",
    )
    p.add_argument(
        "--model",
        default="models/ms-marco-MiniLM-L6-v2-cross-encoder",
    )
    p.add_argument(
        "--output-dir",
        default="checkpoints/e4_multi_bce",
    )
    p.add_argument(
        "--final-model-dir",
        default="models/ms-marco-MiniLM-L6-v2-e4-multi-bce",
    )

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--max-steps", type=int, default=-1)

    return p.parse_args()


def validation_bucket(query_id, seed):
    key = f"{seed}:{query_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / (2**64)


def load_data(path, seed, val_ratio):
    if not 0 < val_ratio < 1:
        raise ValueError("val-ratio 必须在 0 到 1 之间")

    train_rows = []
    val_rows = []

    train_qids = set()
    val_qids = set()
    seen_qids = set()

    query_count = 0
    negative_count = 0
    negative_distribution = {}

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            row = json.loads(line)

            qid = str(row["query_id"])
            query = row["query"].strip()
            positive = row["positive"].strip()
            negatives = row["negatives"]

            if qid in seen_qids:
                raise ValueError(f"重复 Query ID: {qid}")

            seen_qids.add(qid)

            if not query or not positive:
                raise ValueError(
                    f"第 {line_no} 行 Query 或 Positive 为空"
                )

            if len(negatives) != 4:
                raise ValueError(
                    f"Query {qid} 只有 {len(negatives)} 个负例，"
                    "本实验要求每条 Query 恰好 4 个"
                )

            neg_ids = [
                int(neg["passage_id"])
                for neg in negatives
            ]

            if len(set(neg_ids)) != len(neg_ids):
                raise ValueError(f"Query {qid} 负例 ID 重复")

            negative_texts = [
                neg["text"].strip()
                for neg in negatives
            ]

            if any(not text for text in negative_texts):
                raise ValueError(f"Query {qid} 存在空负例")

            if positive in negative_texts:
                raise ValueError(
                    f"Query {qid} 存在与正例文本相同的负例"
                )

            is_val = (
                validation_bucket(qid, seed) < val_ratio
            )

            target = val_rows if is_val else train_rows

            if is_val:
                val_qids.add(qid)
            else:
                train_qids.add(qid)

            # 1 个正例
            target.append({
                "sentence1": query,
                "sentence2": positive,
                "label": 1.0,
            })

            # 全部 4 个困难负例
            for text in negative_texts:
                target.append({
                    "sentence1": query,
                    "sentence2": text,
                    "label": 0.0,
                })

            query_count += 1
            negative_count += len(negatives)

            key = str(len(negatives))
            negative_distribution[key] = (
                negative_distribution.get(key, 0) + 1
            )

    if not train_rows or not val_rows:
        raise RuntimeError("训练集或验证集为空")

    if train_qids & val_qids:
        raise RuntimeError("Query 级数据泄漏")

    random.Random(seed).shuffle(train_rows)

    stats = {
        "total_queries": query_count,
        "total_negatives": negative_count,
        "negative_distribution": negative_distribution,
        "train_queries": len(train_qids),
        "val_queries": len(val_qids),
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "positive_negative_ratio": "1:4",
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

    train_data, val_data, stats = load_data(
        args.train_file,
        args.seed,
        args.val_ratio,
    )

    print("训练数据统计:")
    print(
        json.dumps(stats, ensure_ascii=False, indent=2),
        flush=True,
    )

    model = CrossEncoder(
        args.model,
        num_labels=1,
        max_length=args.max_length,
        local_files_only=True,
    )

    use_bf16 = (
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
        bf16=use_bf16,
        fp16=False,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=4,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        loss=BinaryCrossEntropyLoss(model),
    )

    result = trainer.train()

    final_dir = Path(args.final_model_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_dir))

    summary = {
        "experiment_id": "E4-Multi-BCE",
        "initial_model": args.model,
        "final_model": str(final_dir),
        "train_file": args.train_file,
        "loss": "BinaryCrossEntropyLoss",
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "global_step": result.global_step,
        "train_loss": result.training_loss,
        "bf16": use_bf16,
        "seed": args.seed,
        "data_stats": stats,
        "log_history": trainer.state.log_history,
    }

    result_path = final_dir / "train_result.json"

    result_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("训练完成:", final_dir)
    print("训练报告:", result_path)


if __name__ == "__main__":
    main()
