import torch
from torch import nn

class Slot:
    def __init__(self) -> None:
        self.value_orig = None
        self.value_orig_inplace = None
        self.value_clone_1 = None
        self.value_clone_2 = None
        self.value_clone_3 = None
        self.mask_1 = None
        self.mask_2 = None
        self.mask_3 = None
        self.status = {
            'value_0': 0,
            'value_1': 0,
            'value_2': 0,
            'value_3': 0,
            'value_4': 0,
            'mask_1': 0,
            'mask_2': 0,
            'mask_3': 0,
        }

class NodeInfo:
    def __init__(self, slots_in: list[dir], slots_out: list[dir], flatten_caller, param_masks: dict[str, torch.Tensor]) -> None:
        self.slots_in = slots_in
        self.slots_out = slots_out
        self.flatten_caller = flatten_caller
        self.param_masks = param_masks
        self.in_mask_clone_0 = None
        self.in_mask_clone_1 = None
        self.in_mask_clone_2 = None
        self.out_mask_clone_0 = None
        self.out_mask_clone_1 = None
        self.out_mask_clone_2 = None
        self.param_masks_clone_0 = None
        self.param_masks_clone_1 = None
        self.param_masks_clone_2 = None
        self.status = {
            'in_0': 0,
            'in_1': 0,
            'in_2': 0,
            'out_0': 0,
            'out_1': 0,
            'out_2': 0,
            'param_0': 0,
            'param_1': 0,
            'param_2': 0,
        }
