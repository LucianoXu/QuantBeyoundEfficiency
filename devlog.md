## June 22

Yingte: 
started to work on implementing and evaluation of quantization. I am building the shared project repo. Full observability, extendability and reproducibility are the main concerns.

Vanilla quantization has really low inference speed, at least on MacBook MPS. It takes 2 minites to benchmark a single GSM8K example. Impratical for scaling up.