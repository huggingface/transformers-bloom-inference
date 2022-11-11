from ..constants import DS_INFERENCE, DS_ZERO, HF_ACCELERATE
from .model import Model, get_downloaded_model_path, load_tokenizer


def get_model_class(deployment_framework: str):
    if deployment_framework == HF_ACCELERATE:
        from .hf_accelerate import HFAccelerateModel

        return HFAccelerateModel
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
