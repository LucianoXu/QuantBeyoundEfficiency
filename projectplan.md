# Research Questions

**RQ1:** In what way do quantization methods impact models beyond standard benchmarks, such as bias, trustworthiness, and counterfactual generation?
Standard benchmarks such as accuracy in GSM8K have been well studied. The impact of inference acceleration on bias has been evaluated by [Kirsten et al.](https://arxiv.org/pdf/2410.22118), but since we are studying different models and methods, this would still be interesting research. Furthermore, we could test benchmarks such as [JailbreakBench](https://jailbreakbench.github.io/) or [Emergent Misalignment through ICL](https://arxiv.org/abs/2510.11288). Or compare the models capability to generate valid counterfactuals. 

**RQ2:** Does the gap between commonly known pure performance benchmarks and benchmarks that test more edge cases/subtle behaviour change while comparing the different quantization methods? 
If such a gap exists, this would support the current belief that many model providers are specifically training models to perform better on commonly used datasets. 


# Initial tests

* **Model:** [Qwen/Qwen3](https://huggingface.co/collections/Qwen/qwen3)
* **Model size:** [1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
* **Comparision:** bfloat16 with standard bits and bytes int8

# What benchmarks/eval methods do we choose?

**Has been explored previously:** 
(Interesting if we choose a novel quantization method)
* [Bias](https://arxiv.org/pdf/2410.22118)
* [Jailbreaking,Sycophancy](https://arxiv.org/pdf/2602.09130) 

**Novel (Confident), Hard:** 
* Emergent Misalignment 
	* [Great introduction](https://jeakwon.github.io/blog/2026/emergent-misalignment/)
	* [ICL Emergent Misalignment](https://arxiv.org/pdf/2510.11288) is the only valid choice in time frame 
* Interpretability methods 

**Novel (Confident), Easy:** 
* Counterfactual generation

**Novel (Likely) Easy:** 
* Safety refusal (for example SORRY benchmark) 
* Very specific benchmarks, that test a narrow subset of a bias, or misaligned behaviour. Such as medical data, Geographic bias, political bias. 

**Open Questions:** 
 
# Plan: 
**First Week:** 
Larger goal: Build up a base of benchmarks largely avaliable in the eval harness and create our Minimal Viable Product in the first week.
Perform a broad range of benchmarks to avoid risk of accidental cherrypicking. First a benchmark for standard performance, then multiple benchmarks for different types of largely unexplored or underexplored topics => It could be a good idea to benchmark an equal amount in both types. To answer the research question 2 we can also perform benchmarks with int4 to observe if the gap between the standard inference and unexplored benchmark increases. The idea was that more common benchmarks are more likely to be finetuned on common performance benchmarks. If that was the case, then we could notice a gap increase between int4 and int8 compared to int8 and bf16. Maybe the "memorized" performance benchmark knowledge would be more stable to the higher degradation than other benchmarks. 

Each picks one eval detail
	* Research details
	* how do models perform
	* existing research on this method?
	* anything related to quantization
* Write the code for the each eval based on existing pipeline => Same seed, same model, config, 
* Perform smoke test on own hardware
* Atleast perform one small run of the benchmark with a small model 
* Write benchmark for standard inference benchmark (For example GSM8K) and one bias benchmark to compare to Prof Zafars study. 


**Second Week:**
* First Round of Iteration 
* Implementation 
* And Analysing the Results

**Third Week:** 
* Further iterate on the finding on the second week 
* Refine the code base 
* Finalize results 

**Final Week:**
* Write up the report 
* Finish presentation

**Literature research:** 

# AI Usage
AI generated Code will be marked as such. Only usage for small boilerplate. Core logic and large majority written by hand. 

# Compute Usage
* Ramneek: RTX 4060 8GB VRAM
* Yingte: Macbook, Compute cluster
* Hans: Macbook M4 16GB Ram, Cloud Compute 
