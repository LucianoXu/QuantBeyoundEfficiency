## July 7

Yingte:

Added quantization-aware adversarial suffix attack code and experiment configuration.


## July 2

Yingte:

We agreed on doing experiment1. The infrastructure for hendrycks_ethics_local is manually built. Experiments finished.

## June 27

Yingte:

SORRY-Bench implemented and tested on Qwen3-1.7B bf16/int8/nf4. Minor degrade in jailbreak refusal observaed for int8 quantization.

Raw data of two experiments synced in git.


## June 26

Yingte: 

Utility/capability evaluation:
- GSM8K: Grade School Math 8K (answering the result directly)
- MMLU: Massive Multitask Language Understanding (multiple selection)
- MMLU-Pro: More advanced MMLU. Harder problems, more selections.
- HellaSwag: An enhanced dataset to test the common knowledge of the model. It describes a daily scenario, with selections of the follow-up events, and ask for the most reasonable one.
- ARC: AI2 Reasoning Challenge. It tests the basic science knowledge and reasoning ability of the model. Multiple selection.
- TruthfulQA: Test whether the model will stick to true results under misleading prompts. Format is free: free generation / multiple selection
- BBH: BIG-Bench Hard. A selection of tasks to test the general reasoning ability of the model.
- HumanEval: Evaluate the model's ability to generate python code according to description.
- MATH: high-school competition level math.

Jailbreak:
- SORRY-Bench (https://arxiv.org/abs/2406.14598)
- JailbreakBench (https://arxiv.org/abs/2404.01318)

## June 24
Ramneek: Inference was tested on the RTX 4060 8GB. The 1.7B model performs decently. 30 Minutes for GSM8K Test Dataset for BF16 and an estimated 2 hours for INT8. 


## June 22

Yingte: 
started to work on implementing and evaluation of quantization. I am building the shared project repo. Full observability, extendability and reproducibility are the main concerns.

Vanilla quantization has really low inference speed, at least on MacBook MPS. It takes 2 minites to benchmark a single GSM8K example. Impratical for scaling up.