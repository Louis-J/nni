from typing import Union

import torch
from torch import nn

class Slot:
    def __init__(self) -> None:
        self.value_0 = None
        self.value_1 = None
        self.value_2 = None
        self.mask_1 = None
        self.mask_2 = None
        self.status = {
            'value_0': 0,
            'value_1': 0,
            'value_2': 0,
            'mask_1': 0,
            'mask_2': 0,
        }

class NodeInfo:
    def __init__(self, slots_in: list[str], slots_out: Union[list[str], str], flatten_caller, param_masks: dict[str, torch.Tensor]) -> None:
        self.slots_in = slots_in
        self.slots_out = slots_out
        self.flatten_caller = flatten_caller
        self.param_masks_0 = param_masks
        self.param_masks_1 = {}
        self.status = {
            'param_0': 0,
            'param_1': 0,
        }
