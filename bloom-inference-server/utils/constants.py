# launch script
BENCHMARK = "benchmark"
CLI = "cli"
SERVER = "server"

# inference method (args.deployment_framework)
HF_ACCELERATE = "hf_accelerate"
DS_INFERENCE = "ds_inference"
DS_ZERO = "ds_zero"

# model weights
BIGSCIENCE_BLOOM = "bigscience/bloom"
DS_INFERENCE_BLOOM_FP16 = "microsoft/bloom-deepspeed-inference-fp16"
DS_INFERENCE_BLOOM_INT8 = "microsoft/bloom-deepspeed-inference-int8"

# dtype
BF16 = "bf16"
FP16 = "fp16"
INT8 = "int8"

# this dictionary contains the following structure: launch script -> inference
# method -> dtype. If the path launch script -> inference method -> dtype is
# present in the dictionary, it is allowed otherwise an error is thrown
SCRIPT_FRAMEWORK_MODEL_DTYPE_ALLOWED = {
    BENCHMARK: {
        HF_ACCELERATE: {BIGSCIENCE_BLOOM: {BF16, FP16, INT8}},
        DS_INFERENCE: {BIGSCIENCE_BLOOM: {FP16}, DS_INFERENCE_BLOOM_FP16: {FP16}, DS_INFERENCE_BLOOM_INT8: {INT8}},
        DS_ZERO: {BIGSCIENCE_BLOOM: {BF16, FP16}},
    },
    CLI: {
        HF_ACCELERATE: {
            BIGSCIENCE_BLOOM: {
                BF16,
                FP16,
                # INT8
            }
        },
        DS_INFERENCE: {DS_INFERENCE_BLOOM_FP16: {FP16}, DS_INFERENCE_BLOOM_INT8: {INT8}},
    },
    SERVER: {
        HF_ACCELERATE: {
            BIGSCIENCE_BLOOM: {
                BF16,
                FP16,
                # INT8
            }
        },
        DS_INFERENCE: {DS_INFERENCE_BLOOM_FP16: {FP16}, DS_INFERENCE_BLOOM_INT8: {INT8}},
    },
}
