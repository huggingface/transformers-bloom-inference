from utils import DS_INFERENCE
from utils import DS_ZERO
from utils import HF_ACCELERATE

from .ds_inference import DSInferenceGRPCServer
from .ds_inference import DSInferenceModel
from .ds_zero import DSZeROModel
from .hf_accelerate import HFAccelerateModel
from .model import Model


def get_model_class(deployment_framework: str, basic: bool = False):
    if (deployment_framework == HF_ACCELERATE):
        return HFAccelerateModel
    elif (deployment_framework == DS_INFERENCE):
        if (basic):
            return DSInferenceModel
        else:
            return DSInferenceGRPCServer
    elif (deployment_framework == DS_ZERO):
        return DSZeROModel
    else:
        raise ValueError(
            f"Unknown deployment framework {deployment_framework}")
