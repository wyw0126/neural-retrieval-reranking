#训练cross encoder模型
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
        default="data/processed/msmarco/e2_triplets_gpu.jsonl",
    )
    p.add_argument(
        "--model",
        default="models/ms-marco-MiniLM-L6-v2-cross-encoder",
    )
    p.add_argument(
        "--output-dir",
        default="checkpoints/e4_cross_encoder",
    )
    p.add_argument(
        "--final-model-dir",
        default="models/ms-marco-MiniLM-L6-v2-e4",
    )
    p.add_argument("--max-queries", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--max-steps", type=int, default=-1)

    return p.parse_args()


def validation_bucket(query_id, seed):
    # 按 Query ID 做稳定划分。
    # 同一个 Query 的正负例不会跨训练/验证集。
    key = f"{seed}:{query_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / (2**64)


def load_triplets(path, max_queries, seed, val_ratio):
    if not 0 < val_ratio < 1:
        raise ValueError("val-ratio 必须在 0 到 1 之间")

    if max_queries is not None and max_queries <= 0:
        raise ValueError("max-queries 必须大于 0")

    rng = random.Random(seed)

    # 固定种子蓄水池采样，避免总是取文件的前 N 条。
    selected = []
    eligible = 0

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
        raise RuntimeError("未读取到有效三元组")

    train_rows = []
    val_rows = []

    train_ids = set()
    val_ids = set()

    for row in selected:
        qid = row["query_id"]

        is_val = (
            validation_bucket(qid, seed) < val_ratio
        )

        target = val_rows if is_val else train_rows

        if is_val:
            val_ids.add(qid)
        else:
            train_ids.add(qid)

        target.append({
            "sentence1": row["query"],
            "sentence2": row["positive"],
            "label": 1.0,
        })

        target.append({
            "sentence1": row["query"],
            "sentence2": row["negative"],
            "label": 0.0,
        })

    if not train_rows or not val_rows:
        raise RuntimeError(
            "训练集或验证集为空，请增大样本量"
        )

    if train_ids & val_ids:
        raise RuntimeError(
            "发生 Query 级别的数据泄漏"
        )

    rng.shuffle(train_rows)

    return (
        Dataset.from_list(train_rows),
        Dataset.from_list(val_rows),
        {
            "selected_triplets": len(selected),
            "train_pairs": len(train_rows),
            "validation_pairs": len(val_rows),
            "unique_train_queries": len(train_ids),
            "unique_validation_queries": len(val_ids),
        },
    )


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("加载训练三元组...", flush=True)

    train_dataset, val_dataset, stats = load_triplets(
        args.train_file,
        args.max_queries,
        args.seed,
        args.val_ratio,
    )

    print(
        json.dumps(stats, indent=2, ensure_ascii=False),
        flush=True,
    )

    print("加载 Cross-Encoder...", flush=True)

    model = CrossEncoder(
        args.model,
        num_labels=1,
        max_length=args.max_length,
        local_files_only=True,
    )

    bf16_enabled = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    print("BF16:", bf16_enabled, flush=True)

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
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=4,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=BinaryCrossEntropyLoss(model),
    )

    print("开始 E4 训练...", flush=True)

    train_result = trainer.train()

    final_dir = Path(args.final_model_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_dir))

    result = {
        "experiment_id": "E4",
        "initial_model": args.model,
        "final_model": str(final_dir),
        "train_file": args.train_file,
        "seed": args.seed,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "bf16": bf16_enabled,
        "loss": "BinaryCrossEntropyLoss",
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

    print("E4 训练完成", flush=True)
    print("模型目录:", final_dir, flush=True)
    print("结果文件:", result_path, flush=True)


if __name__ == "__main__":
    main()
