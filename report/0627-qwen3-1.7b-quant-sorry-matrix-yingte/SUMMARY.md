## Meta Data

A model-quantization vs. jailbreak refusal benchmark matrix experiment.

- date: 0627
- operator: Yingte Xu
- machine: A100 * 1
- models: Qwen3-1.7B, bf16/int8/nf4, no CoT reasoning
- benchmark: SORRY-Bench
- time consumption:
    - bf16 10 min
    - int8 30 min
    - nf4 15 min

## Results

| Benchmark | Metric | bf16 | int8 | nf4 | Δint8 | Δnf4 | Std. err. |
|---|---|---|---|---|---|---|---|
| SORRY-Bench | fulfillment_rate | 56.67 | 59.78 | 56.89 | +3.11 | +0.22 | ± 2.34 |