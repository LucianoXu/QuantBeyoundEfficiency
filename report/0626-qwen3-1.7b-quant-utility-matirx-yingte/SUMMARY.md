## Meta Data

A model-quantization vs. utility benchmark matrix experiment.

- date: 0626
- operator: Yingte Xu
- machine: A100 * 3, parallel
- models: Qwen3-1.7B, bf16/int8/nf4, no CoT reasoning
- benchmarks: utility benchmarks, sampled (20 ~ 50 per benchmark track)
- time consumption:
    - bf16 1.0h
    - int8 3.8h
    - nf4 2.6h

## Results

| Benchmark | Metric | Random | bf16 | int8 | nf4 | Δint8 | Δnf4 | Std. err. |
|---|---|---|---|---|---|---|---|---|
| GSM8K | exact_match | 0.0 | 62.00 | 68.00 | 64.00 | +6.00 | +2.00 | ± 4.88 |
| MMLU | acc | 25.0 | 63.58 | 63.02 | 59.23 | -0.56 | -4.35 | ± 1.23 |
| MMLU-Pro | exact_match | 10.0 | 41.43 | 38.57 | 33.71 | -2.86 | -7.71 | ± 2.59 |
| HellaSwag | acc_norm | 25.0 | 53.00 | 53.00 | 54.00 | +0.00 | +1.00 | ± 5.02 |
| ARC | acc_norm | 25.0 | 53.00 | 54.00 | 54.00 | +1.00 | +1.00 | ± 5.02 |
| TruthfulQA | acc | 25.0 | 47.53 | 48.00 | 47.69 | +0.48 | +0.16 | ± 4.26 |
| BBH | exact_match | 25.0 | 49.44 | 50.74 | 39.44 | +1.30 | -10.00 | ± 1.68 |
| MATH | exact_match | 0.0 | 28.00 | 28.57 | 20.29 | +0.57 | -7.71 | ± 2.25 |

acc / acc_norm: used in evaluating multiple selections. Calculuate the log likelyhood of generating different selections given the prompt.

## Conclusion

- int8 quantization almost has the same performance as bf16. Most difference smaller than standard error.
- nf4 quantization degrades significantly on tasks that requires reasoning, e.g. MMLU, MATH, BBH.