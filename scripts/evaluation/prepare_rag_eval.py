"""Freeze a reproducible pilot from existing MS MARCO queries and qrels."""
import argparse
import json
import random
from pathlib import Path


def build_cases(queries_file, qrels_file, count=50, seed=42, synthetic_count=10):
    queries = {}
    with Path(queries_file).open(encoding="utf-8") as f:
        for line in f:
            qid, query = line.rstrip("\r\n").split("\t", 1)
            if qid in queries:
                raise ValueError("Duplicate query ID")
            queries[qid] = query.strip()
    positives = {}
    with Path(qrels_file).open(encoding="utf-8") as f:
        for line in f:
            qid, _, pid, rel = line.split()
            if int(rel) > 0:
                positives.setdefault(qid, set()).add(int(pid))
    eligible = sorted(qid for qid in positives if qid in queries
                      and 1 <= len(queries[qid]) <= 2048)
    if not 1 <= count <= len(eligible) or not 0 <= synthetic_count <= 100:
        raise ValueError("Invalid sample count")
    rng = random.Random(seed)
    cases = [{"case_id": "msmarco:"+qid, "query": queries[qid],
              "gold_passage_ids": sorted(positives[qid]), "expected_answerable": True,
              "label_source": "msmarco_positive_qrels_proxy", "reference_answer": None}
             for qid in rng.sample(eligible, count)]
    for i in range(synthetic_count):
        name = f"Zorvex-{seed:08d}-{i:04d}-fictional"
        cases.append({"case_id": f"synthetic:{i:04d}",
                      "query": f"What is the exact factory serial number of the imaginary {name} machine on Mars?",
                      "gold_passage_ids": [], "expected_answerable": False,
                      "label_source": "synthetic_expected_abstention", "reference_answer": None})
    rng.shuffle(cases)
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", type=Path, default=Path("data/raw/msmarco/queries.dev.small.tsv"))
    p.add_argument("--qrels", type=Path, default=Path("data/raw/msmarco/qrels.dev.small.tsv"))
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--synthetic-count", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cases = build_cases(a.queries, a.qrels, a.count, a.seed, a.synthetic_count)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False)+"\n")
    print(f"Frozen {len(cases)} cases at {a.output}")


if __name__ == "__main__":
    main()
