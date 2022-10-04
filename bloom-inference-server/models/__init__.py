from utils import DS_INFERENCE, DS_ZERO, HF_ACCELERATE

from .ds_inference import DSInferenceGRPCServer, DSInferenceModel
from .ds_zero import DSZeROModel
from .hf_accelerate import HFAccelerateModel
from .model import Model, get_stopping_criteria


def get_model_class(deployment_framework: str, basic: bool = False):
    if deployment_framework == HF_ACCELERATE:
        return HFAccelerateModel
    elif deployment_framework == DS_INFERENCE:
        if basic:
            return DSInferenceModel
        else:
            return DSInferenceGRPCServer
    elif deployment_framework == DS_ZERO:
        return DSZeROModel
    else:
        raise ValueError(f"Unknown deployment framework {deployment_framework}")
