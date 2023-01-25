from ..constants import DS_INFERENCE, DS_ZERO, HF_ACCELERATE, HF_CPU
from .model import Model, get_hf_model_class, load_tokenizer


def get_model_class(deployment_framework: str):
    if deployment_framework == HF_ACCELERATE:
        from .hf_accelerate import HFAccelerateModel

        return HFAccelerateModel
    elif deployment_framework == HF_CPU:
        from .hf_cpu import HFCPUModel

        return HFCPUModel
    elif deployment_framework == DS_INFERENCE:
        from .ds_inference import DSInferenceModel

        return DSInferenceModel
    elif deployment_framework == DS_ZERO:
        from .ds_zero import DSZeROModel

        return DSZeROModel
    else:
        raise ValueError(f"Unknown deployment framework {deployment_framework}")


def start_inference_engine(deployment_framework: str) -> None:
    if deployment_framework in [DS_INFERENCE, DS_ZERO]:
        import deepspeed

        deepspeed.init_distributed("nccl")
