# Scripts

This folder contains several utility scripts.

- `prefetch_datasets.py` downloads all benchmark datasets into the Huggingface cache to avoid race conditions during parallel evaluation.

- `build_hendrycks_ethics_loca.py` downloads and converts the HF Hendrycks ETHICS dataset into local Parquet files.