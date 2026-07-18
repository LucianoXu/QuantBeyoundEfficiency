# Scripts

This folder contains several utility scripts.

- `prefetch_datasets.py` downloads all benchmark datasets into the Huggingface cache to avoid race conditions during parallel evaluation.
    - Run this script before submitting several evaluation jobs in parallel.

- `build_hendrycks_ethics_local.py` downloads and converts the HF Hendrycks ETHICS dataset into local Parquet files.
  - The dataset does not match modern transformers conventions. This script fixes these issues and saves a modern Parquet file under `data\hendrycks_ethics_local`
- `/sbatches` contains the job files that were used to run the experiments on a cluster.