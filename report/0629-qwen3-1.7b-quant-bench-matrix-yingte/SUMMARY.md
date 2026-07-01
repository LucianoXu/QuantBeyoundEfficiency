## Meta Data

A model-quantization vs. benchmark matrix experiment, utility + safety, adding a
4-bit AWQ and run zero-shot.

- date: 0629
- operator: Yingte Xu
- machine: A100 * 4
- models: Qwen3-1.7B, bf16 / int8 / nf4 / AWQ-W4A16-asym, no CoT reasoning
- benchmarks: utility benchmarks + SORRY-Bench
- decoding: zero-shot, greedy
- time consumption (wall clock):
    - bf16 1.5h
    - AWQ  1.8h
    - nf4  2.7h
    - int8 4.3h — CANCELLED in the middle

## Results — utility

| Benchmark | Metric | Random | bf16 | int8 | nf4 | AWQ | Δint8 | Δnf4 | ΔAWQ | Std. err. |
|---|---|---|---|---|---|---|---|---|---|---|
| GSM8K | exact_match (strict) | 0.0 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | +0.00 | +0.00 | 0.00 |
| MMLU | acc | 25.0 | 57.75 | 57.75 | 55.72 | 56.42 | +0.00 | -2.04 | -1.33 | ± 1.24 |
| MMLU-Pro | exact_match | 10.0 | 43.43 | 43.43 | 37.71 | 35.71 | +0.00 | -5.71 | -7.71 | ± 2.53 |
| HellaSwag | acc_norm | 25.0 | 46.00 | 46.00 | 52.00 | 43.00 | +0.00 | +6.00 | -3.00 | ± 5.01 |
| ARC | acc_norm | 25.0 | 44.00 | 47.00 | 42.00 | 46.00 | +3.00 | -2.00 | +2.00 | ± 4.99 |
| TruthfulQA | acc | 25.0 | 47.53 | 48.00 | 47.69 | 47.14 | +0.48 | +0.16 | -0.39 | ± 4.26 |
| BBH | exact_match | 25.0 | 0.56 | – | 1.48 | 0.74 | – | +0.93 | +0.19 | ± 0.32 |
| MATH | exact_match | 0.0 | 0.00 | – | 0.00 | 0.00 | – | +0.00 | +0.00 | 0.00 |

- The generative tasks collapsed, caused by the zero-shot + greedy decoding protocol on a 1.7B model.
- The model emits free-form CoT that falls into repetition loops.

| GSM8K | bf16 | int8 | nf4 | AWQ |
|---|---|---|---|---|
| flexible-extract | 0.52 | 0.45 | 0.40 | 0.30 |

## Results — SORRY-Bench

| variant | fulfillment | Δ vs bf16 |
|---|---|---|
| bf16 | 0.567 | — |
| nf4 | 0.569 | +0.002 |
| int8 | 0.598 | **+0.031** |
| AWQ | 0.602 | **+0.035** |