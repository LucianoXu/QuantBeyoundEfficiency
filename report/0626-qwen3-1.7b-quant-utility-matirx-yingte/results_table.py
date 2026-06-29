"""
Result analysis script. 

AI generated.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

# benchmark display name, lm-eval task id, random-chance accuracy (%)
BENCHMARKS = [
    ("GSM8K",      "gsm8k",          0.0),
    ("MMLU",       "mmlu",           25.0),
    ("MMLU-Pro",   "mmlu_pro",       10.0),
    ("HellaSwag",  "hellaswag",      25.0),
    ("ARC",        "arc_challenge",  25.0),
    ("TruthfulQA", "truthfulqa_mc2", 25.0),
    ("BBH",        "bbh",            25.0),
    ("MATH",       "minerva_math",   0.0),
]

QUANTS = ["bf16", "int8", "nf4"]

# metric to report per task, first one present wins
METRICS = [
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match,none",
    "exact_match,custom-extract",
    "exact_match,get-answer",
    "acc_norm,none",
    "acc,none",
]


def read_result(path, task_id):
    """Return (metric, score, stderr) as fractions for one result.json."""
    results = json.loads(path.read_text())["results"][task_id]
    for key in METRICS:
        if key in results:
            metric, _, suffix = key.partition(",")
            stderr_key = f"{metric}_stderr,{suffix}" if suffix else f"{metric}_stderr"
            stderr = results.get(stderr_key)
            return metric, results[key], stderr if isinstance(stderr, float) else None
    raise KeyError(f"no known metric in {path}")


def collect(root):
    """(bench_name, quant_type) -> (metric, score, stderr) over every leaf run."""
    task_of = {name: task_id for name, task_id, _ in BENCHMARKS}
    scores = {}
    for config_path in root.glob("**/config.yaml"):
        result_path = config_path.with_name("result.json")
        if not result_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text())
        bench = config["bench"]["bench_name"]
        quant = config["model"]["quant_type"]
        scores[bench, quant] = read_result(result_path, task_of[bench])
    return scores


def build_rows(scores):
    rows = []
    for name, _, random in BENCHMARKS:
        cells = {q: scores.get((name, q)) for q in QUANTS}
        if cells["bf16"] is None:
            continue
        metric, ref, stderr = cells["bf16"]
        pct = lambda q: f"{cells[q][1] * 100:.2f}" if cells[q] else ""
        delta = lambda q: f"{(cells[q][1] - ref) * 100:+.2f}" if cells[q] else ""
        # std. err. only for bf16; it is ~constant across quants (fixed n)
        rows.append([
            name, metric, f"{random:.1f}",
            pct("bf16"), pct("int8"), pct("nf4"),
            delta("int8"), delta("nf4"),
            f"{stderr * 100:.2f}" if stderr is not None else "",
        ])
    return rows


HEADER = ["Benchmark", "Metric", "Random", "bf16", "int8", "nf4",
          "Δint8", "Δnf4", "Std. err."]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="model-bench matrix output dir")
    parser.add_argument("-o", "--out", type=Path, help="output CSV (default: stdout)")
    args = parser.parse_args()

    rows = build_rows(collect(args.results_dir))
    out = open(args.out, "w", newline="") if args.out else sys.stdout
    writer = csv.writer(out)
    writer.writerow(HEADER)
    writer.writerows(rows)
    if args.out:
        out.close()


if __name__ == "__main__":
    main()
