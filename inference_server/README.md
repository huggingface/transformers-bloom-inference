## Inference solutions for BLOOM 176B

We support HuggingFace accelerate and DeepSpeed Inference for generation.

Install required packages:

```shell
pip install flask flask_api gunicorn pydantic accelerate huggingface_hub>=0.9.0 deepspeed>=0.7.3 deepspeed-mii==0.0.2
```

alternatively you can also install deepspeed from source:
```shell
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
CFLAGS="-I$CONDA_PREFIX/include/" LDFLAGS="-L$CONDA_PREFIX/lib/" TORCH_CUDA_ARCH_LIST="7.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check
```

All the provided scripts are tested on 8 A100 80GB GPUs for BLOOM 176B (fp16/bf16) and 4 A100 80GB GPUs for BLOOM 176B (int8). These scripts might not work for other models or a different number of GPUs.

DS inference is deployed using the DeepSpeed MII library which requires the resharded checkpoints for 8 x Tensor Parallel.

Note: Sometimes GPU memory is not freed when DS inference deployment crashes. You can free this memory by running `killall python` in terminal.

For using BLOOM quantized, use dtype = int8. Also, change the model_name to microsoft/bloom-deepspeed-inference-int8 for DeepSpeed-Inference. For HF accelerate, no change is needed for model_name.

HF accelerate uses [LLM.int8()](https://arxiv.org/abs/2208.07339) and DS-inference uses [ZeroQuant](https://arxiv.org/abs/2206.01861) for post-training quantization.

#### BLOOM inference via command-line

This asks for generate_kwargs everytime.
Example: generate_kwargs =
```json
{"min_length": 100, "max_new_tokens": 100, "do_sample": false}
```

1. using HF accelerate
```shell
python cli.py --model_name bigscience/bloom --dtype bf16 --deployment_framework hf_accelerate --generate_kwargs '{"min_length": 100, "max_new_tokens": 100, "do_sample": false}'
```

2. using DS inference
```shell
python cli.py --model_name microsoft/bloom-deepspeed-inference-fp16 --dtype fp16 --deployment_framework ds_inference --generate_kwargs '{"min_length": 100, "max_new_tokens": 100, "do_sample": false}'
```

#### BLOOM server deployment

[server.sh](server.sh) can be used to launch a generation server. Please note that the serving method is synchronous and users have to wait in queue until the preceding requests have been processed.

#### Benchmark system for BLOOM inference

1. using HF accelerate
```shell
python benchmark.py --model_name bigscience/bloom --dtype bf16 --deployment_framework hf_accelerate --benchmark_cycles 5
```

2. using DS inference
```shell
deepspeed --num_gpus 8 benchmark.py --model_name bigscience/bloom --dtype fp16 --deployment_framework ds_inference --benchmark_cycles 5
```
alternatively, to load model faster:
```shell
deepspeed --num_gpus 8 benchmark.py --model_name microsoft/bloom-deepspeed-inference-fp16 --dtype fp16 --deployment_framework ds_inference --benchmark_cycles 5
```

3. using DS ZeRO
```shell
deepspeed --num_gpus 8 benchmark.py --model_name bigscience/bloom --dtype bf16 --deployment_framework ds_zero --benchmark_cycles 5
```

#### Internal benchmarks:

```shell
for bs in {1,2,4,8,16,32,64,128}
do
    python scripts/bloom-inference-server/benchmark.py --model_name bigscience/bloom --dtype bf16 --deployment_framework hf_accelerate --benchmark_cycles 5 --batch_size $bs 2>&1 | tee logs/hf-$bs.log

    deepspeed --num_gpus 8 scripts/bloom-inference-server/benchmark.py --model_name bigscience/bloom --dtype fp16 --deployment_framework ds_inference --save_mp_checkpoint_path <PATH TO DS CACHED MODEL> --benchmark_cycles 5 --batch_size $bs 2>&1 | tee logs/ds-$bs.log

    deepspeed --num_gpus 8 scripts/bloom-inference-server/benchmark.py --model_name bigscience/bloom --dtype bf16 --deployment_framework ds_zero --benchmark_cycles 5 --batch_size $bs 2>&1 | tee logs/ds-zero-$bs.log
done
```

The following will benchmark sequence length for batch size = 1 on DS inference.
```shell
for sq in {1,10,50,100,200,300,400,500,600,700,800,900,1000,1500,2000,2500,3000,3500,4000,4500,5000}
do
    deepspeed --num_gpus 8 scripts/bloom-inference-server/benchmark.py --model_name bigscience/bloom --dtype fp16 --batch_size 1 --benchmark_cycles 5 --deployment_framework ds_inference --generate_kwargs '{"do_sample": false, "min_length": '$sq', "max_new_tokens": '$sq'}' 2>&1 | tee logs/ds_$sq.log
done
```

## Support


If you run into things not working or have other questions please open an Issue in the corresponding backend:

- [Accelerate](https://github.com/huggingface/accelerate/issues)
- [Deepspeed-Inference](https://github.com/microsoft/DeepSpeed/issues)
- [Deepspeed-ZeRO](https://github.com/microsoft/DeepSpeed/issues)

If there a specific issue with one of the scripts and not the backend only then please open an Issue here and tag [@mayank31398](https://github.com/mayank31398).
