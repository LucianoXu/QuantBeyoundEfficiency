"""
Result analysis script for the SORRY-Bench quantization matrix.

AI generated.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import yaml

QUANTS = ["bf16", "int8", "nf4"]
BENCH = "SORRY-Bench"


def collect(root):
    """quant_type -> (overall_rate, n, {category: rate}) over every leaf run."""
    out = {}
    for config_path in root.glob("**/config.yaml"):
        result_path = config_path.with_name("result.json")
        if not result_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text())
        if config["bench"]["bench_name"] != BENCH:
            continue
        quant = config["model"]["quant_type"]
        res = json.loads(result_path.read_text())["results"][BENCH]
        out[quant] = (res["fulfillment_rate"], res["n_samples"],
                      res["per_category_fulfillment_rate"])
    return out


def build_rows(scores):
    rate = {q: scores[q][0] for q in QUANTS if q in scores}
    pc = {q: scores[q][2] for q in QUANTS if q in scores}
    cats = sorted(pc["bf16"], key=int)

    pct = lambda v: f"{v * 100:.2f}" if v is not None else ""
    delta = lambda v, ref: f"{(v - ref) * 100:+.2f}" if v is not None else ""

    rows = []
    for c in cats:
        ref = pc["bf16"][c]
        rows.append([
            c, pct(ref), pct(pc["int8"].get(c)), pct(pc["nf4"].get(c)),
            delta(pc["int8"].get(c), ref), delta(pc["nf4"].get(c), ref),
        ])

    # OVERALL row, with binomial std. err. for the bf16 baseline (fixed n)
    n = scores["bf16"][1]
    se = math.sqrt(rate["bf16"] * (1 - rate["bf16"]) / n)
    rows.append([
        "OVERALL", pct(rate["bf16"]), pct(rate["int8"]), pct(rate["nf4"]),
        delta(rate["int8"], rate["bf16"]), delta(rate["nf4"], rate["bf16"]),
        f"{se * 100:.2f}",
    ])
    return rows


HEADER = ["Category", "bf16", "int8", "nf4", "Δint8", "Δnf4", "Std. err."]


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
