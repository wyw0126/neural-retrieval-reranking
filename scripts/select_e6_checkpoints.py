
import json

from pathlib import Path


ROOT = Path(
    "reports/experiments/e5_validation"
)


def read_json(path):

    return json.loads(
        path.read_text(encoding="utf-8")
    )


baseline = read_json(
    ROOT / "step0.json"
)


def get_record(path, step):

    data = read_json(path)

    check_keys = [
        "candidate_cache_hash",
        "queries_hash",
        "query_ids_hash",
        "qrels_hash",
        "index_hash",
        "collection_hash",
        "biencoder_full_hash",
        "first_stage_metrics",
        "split",
        "query_count",
        "top_k",
        "reranker_max_length",
        "reranker_activation",
    ]

    for key in check_keys:

        if key not in baseline:

            raise RuntimeError(
                f"基线报告缺少字段: {key}"
            )

        if data.get(key) != baseline[key]:

            raise RuntimeError(
                f"{path} 评估协议不一致: {key}"
            )

    metrics = data["reranked_metrics"]

    return {
        "step": step,

        "model": data["reranker"],

        "ndcg": metrics["NDCG@10"],

        "mrr": metrics["MRR@10"],

        "recall": metrics["Recall@10"],

        "report": str(path),
    }


base_record = get_record(
    ROOT / "step0.json",
    0,
)


selection = {}


for experiment in ["E6-A", "E6-B"]:

    folder = ROOT / experiment

    if not folder.is_dir():

        raise FileNotFoundError(folder)

    train_report = read_json(
        Path("reports/training")
        / f"{experiment}.json"
    )

    rows = [base_record]

    expected_steps = {
        int(item["step"])
        for item in train_report["checkpoints"]
    }

    evaluated_steps = set()

    for path in folder.glob("step-*.json"):

        step = int(
            path.stem.split("-")[-1]
        )

        evaluated_steps.add(step)

        rows.append(
            get_record(path, step)
        )

    if evaluated_steps != expected_steps:

        raise RuntimeError(
            f"{experiment} Checkpoint 评估不完整: "
            f"missing={sorted(expected_steps-evaluated_steps)}"
        )

    final_report = folder / "final.json"

    if not final_report.is_file():

        raise FileNotFoundError(final_report)

    rows.append(
        get_record(final_report, "final")
    )

    # 保留 Step 0 作为合法选择。
    rows.sort(
        key=lambda x: (
            x["ndcg"],
            x["step"] == 0,
            x["mrr"],
        ),
        reverse=True,
    )

    best = rows[0]

    best_trained = max(
        (
            row for row in rows
            if row["step"] != 0
        ),
        key=lambda row: (
            row["ndcg"],
            row["mrr"],
        ),
    )

    selection[experiment] = {
        "best_model": best,

        "best_trained_model": best_trained,

        "baseline": base_record,

        "ndcg_improvement": (
            best["ndcg"]
            - base_record["ndcg"]
        ),

        "best_trained_ndcg_improvement": (
            best_trained["ndcg"]
            - base_record["ndcg"]
        ),

        "evaluated_model_count": len(rows),

        "all_results": rows,
    }

    print(
        f"\n========== {experiment} =========="
    )

    print("基线 NDCG:", base_record["ndcg"])

    print("最佳模型:", best)

    print("最佳微调模型:", best_trained)

    print(
        "最佳微调 NDCG 变化:",
        best_trained["ndcg"]
        - base_record["ndcg"],
    )


output = ROOT / "model_selection_e6.json"

with output.open(
    "x",
    encoding="utf-8",
) as f:

    json.dump(
        selection,
        f,
        ensure_ascii=False,
        indent=2,
    )


print("\n结果保存至:", output)
