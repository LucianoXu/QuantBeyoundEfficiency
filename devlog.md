## June 22

Yingte: 
started to work on implementing and evaluation of quantization. I am building the shared project repo. Full observability, extendability and reproducibility are the main concerns.

Vanilla quantization has really low inference speed, at least on MacBook MPS. It takes 2 minites to benchmark a single GSM8K example. Impratical for scaling up.

## June 24
Ramneek: Inference was tested on the RTX 4060 8GB. The 1.7B model performs decently. 30 Minutes for GSM8K Test Dataset for BF16 and an estimated 2 hours for INT8. 