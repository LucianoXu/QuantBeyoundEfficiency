"""
Downloads and converts the HF Hendrycks ETHICS dataset into local Parquet files.

The dataset contains code execution and is not consistent with latest transformers conventions.
This script downloads the raw CSV files directly from Huggingface,
applies a standard schema across subsets and exports these as Parquet files.
"""

import csv
from pathlib import Path

import datasets
from huggingface_hub import hf_hub_download

REPO = "hendrycks/ethics"
OUT = Path("data/hendrycks_ethics_local")
SUBSETS = ["commonsense", "deontology", "justice", "utilitarianism", "virtue"]


def read_rows(subset: str, split: str) -> list[dict]:
    """ Fetches the hendrycks/ethics CSV split from Huggingface and parses it into typed dicts. """
    path = hf_hub_download(REPO, f"data/{subset}/{split}.csv", repo_type="dataset")
    out: list[dict] = []
    with open(path, newline="") as f:
        if subset == "utilitarianism":
            
            reader = csv.reader(f)
            header = next(reader)  # discard header row
            assert len(header) == 2, f"unexpected utilitarianism header: {header}"
            for c1, c2 in reader:
                out.append({"activity": c1, "baseline": c2})
        else:
            reader = csv.DictReader(f)
            for row in reader:
                if subset == "commonsense":
                    out.append({"label": int(row["label"]), "input": row["input"]})
                elif subset == "deontology":
                    out.append({"label": int(row["label"]), "scenario": row["scenario"], "excuse": row["excuse"]})
                elif subset == "justice":
                    out.append({"label": int(row["label"]), "scenario": row["scenario"]})
                elif subset == "virtue":
                    scenario, trait = row["scenario"].split(" [SEP] ")
                    out.append({"label": int(row["label"]), "scenario": scenario, "trait": trait})
    return out


def main() -> None:
    """ Iterates through all subsets and splits, processes the rows and saves them as Parquet files. """
    for subset in SUBSETS:
        for split in ["train", "test"]:
            rows = read_rows(subset, split)
            ds = datasets.Dataset.from_list(rows)
            dest = OUT / subset
            dest.mkdir(parents=True, exist_ok=True)
            ds.to_parquet(str(dest / f"{split}.parquet"))
            print(f"  {subset}/{split}: {len(rows)} rows -> {dest / f'{split}.parquet'}  cols={ds.column_names}")
    print("\n >> Done. Local ETHICS parquet written under", OUT)


if __name__ == "__main__":
    main()
