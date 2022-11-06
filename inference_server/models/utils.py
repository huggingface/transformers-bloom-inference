import deepspeed

from ..constants import DS_INFERENCE, DS_ZERO, HF_ACCELERATE
from .ds_inference import DSInferenceModel
from .ds_zero import DSZeROModel
from .hf_accelerate import HFAccelerateModel


def get_model_class(deployment_framework: str):
    if deployment_framework == HF_ACCELERATE:
        return HFAccelerateModel
    elif deployment_framework == DS_INFERENCE:
        return DSInferenceModel
    elif deployment_framework == DS_ZERO:
        return DSZeROModel
    else:
        raise ValueError(f"Unknown deployment framework {deployment_framework}")


def start_inference_engine(deployment_framework: str) -> None:
    if deployment_framework in [DS_INFERENCE, DS_ZERO]:
        deepspeed.init_distributed("nccl")
