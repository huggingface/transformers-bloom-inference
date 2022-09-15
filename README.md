# Fast Inference Solutions for BLOOM

This repo provides demos and packages to perform fast inference solutions for BLOOM. Some of the solutions have their own repos in which case a link to the corresponding repos is provided instead.

Some of the solutions provide both half-precision and int8-quantized solution.

## Client-side solutions

Solutions developed to perform large batch inference locally:

Pytorch:

* [Accelerate, DeepSpeed-Inference and DeepSpeed-ZeRO](./bloom-inference-scripts)

* [Custom Fused Kernel approach](https://github.com/huggingface/transformers_bloom_parallel)

JAX:

* [BLOOM Inference in JAX](https://github.com/huggingface/bloom-jax-inference)



## Server solutions

Solutions developed to be used in a server mode (i.e. varied batch size, varied request rate):

Pytorch:

* [Accelerate and DeepSpeed-Inference based solutions](./bloom-inference-server)

Rust:

* [Bloom-server](https://github.com/Narsil/bloomserver)
