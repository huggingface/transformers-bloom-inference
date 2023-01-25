from argparse import Namespace

from .hf_accelerate import HFAccelerateModel


class HFCPUModel(HFAccelerateModel):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.input_device = "cpu"
