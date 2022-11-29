# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy

import logging
from pathlib import Path
from collections import deque
# import queue

import torch
import torch.nn as nn

from nni.common.graph_utils import build_module_graph
from nni.compression.pytorch.utils.mask_conflict import fix_mask_conflict
from nni.compression.pytorch.utils.utils import get_module_by_name, rand_like_with_shape, randomize_tensor, torch_float_dtype, torch_integer_dtype
from nni.compression.pytorch.speedup.compress_modules import replace_module
from nni.compression.pytorch.speedup.v1.container import Slot, NodeInfo
from nni.compression.pytorch.speedup.v1.jit_translate import r_jit_to_python_function
from nni.compression.pytorch.speedup.v1.utils import run_onlyif_instance, map_recursive, map_recursive_zip

from typing import TYPE_CHECKING
if TYPE_CHECKING:  # Only imports the below statements during type checking
    from nni.common.graph_utils import NodePyGroup

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class ModelSpeedup:
    """
    This class is to speedup the model with provided weight mask.

    Parameters
    ----------
    model : pytorch model
        The model user wants to speedup
    dummy_input : pytorch tensor, tuple of tensor, list of tensor
        Note: The first dimension of the dummy_input should be the batchsize.
        The dummy input for ```jit.trace```, users should put it on the right
        device.
    masks_file : str/dict
        The path of user provided mask file, or the mask object
    map_location : str
        the device on which masks are placed, same to map_location in ```torch.load```
    batch_dim : int
        the index of batch dimension in the dummy_input
    confidence: the confidence coefficient of the sparsity inference. This value is
        actually used as the batchsize of the dummy_input.
    customized_replace_func: None/Dict
        If `customized_replace_func` is not None, then we will use the given function to replace the
        corresponding modules. The `key` of the dict is the opertor types and the `value`
        is the replace function of corresponding opertor. The replace function should take
        two input parameters, one is the original module, the second input parameter is tuple
        of the input mask, output mask and weight mask. This replace function should prune the module
        accordingly. Here is an example of the replace function(more examples can refer to compress_modules.py)::

            def example_replace(ori_module, masks):
                in_mask, out_mask, weight_mask = masks
                # prune the ori_module to a new smaller module according to the mask
                return new_small_module

    """

    def __init__(self, model, dummy_input, masks_file, map_location=None,
                 batch_dim=0, confidence=8, customized_replace_func=None):
        assert confidence > 1
        # The auto inference will change the values of the parameters in the model
        # so we need make a copy before the mask inference
        self.ori_state_dict = copy.deepcopy(model.state_dict())
        self.bound_model = model
        self.inferred_masks = dict()  # key: module_name, value: ModuleMasks
        self.batch_dim = batch_dim
        self.confidence = confidence
        self.dummy_input, self.device = self.r_random_model_input(dummy_input, batch_dim, confidence)
        self.torch_graph = build_module_graph(model, self.dummy_input)
        # load the mask tensor to the same device with the dummy_input
        # self.masks save the mask tensors pruned by the user and the infered
        # masks of the others modules
        if isinstance(masks_file, (str, Path)) and Path(masks_file).exists():
            self.masks = torch.load(masks_file, map_location if map_location is not None else str(self.device))
        elif isinstance(masks_file, dict):
            self.masks = masks_file
        else:
            raise Exception('Please provide the mask or the path of the mask file')
        self.customized_replace_func = customized_replace_func if customized_replace_func is not None else {}

    def direct_order(self):
        in_degree = {}
        visit_queue = deque()
        for node in self.torch_graph.nodes_py.nodes_op:
            predecessors = self.torch_graph.find_predecessors(node.unique_name)
            predecessors = sorted(set(predecessors), key = predecessors.index)
            if len(predecessors) == 0:
                visit_queue.append(node)
            else:
                in_degree[node.unique_name] = len(predecessors)
        while len(visit_queue) != 0:
            node: NodePyGroup = visit_queue.popleft()
            successors = self.torch_graph.find_successors(node.unique_name)
            successors = sorted(set(successors), key = successors.index)
            for successor in successors:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    visit_queue.append(self.torch_graph.name_to_node[successor])
                    in_degree.pop(successor)
            yield node
        assert len(in_degree) == 0

    def indirect_order(self):
        out_degree = {}
        visit_queue = deque()
        for node in self.torch_graph.nodes_py.nodes_op:
            successors = self.torch_graph.find_successors(node.unique_name)
            successors = sorted(set(successors), key = successors.index)
            if len(successors) == 0:
                visit_queue.append(node)
            else:
                out_degree[node.unique_name] = len(successors)
        while len(visit_queue) != 0:
            node: NodePyGroup = visit_queue.popleft()
            predecessors = self.torch_graph.find_predecessors(node.unique_name)
            predecessors = sorted(set(predecessors), key = predecessors.index)
            for predecessor in predecessors:
                out_degree[predecessor] -= 1
                if out_degree[predecessor] == 0:
                    visit_queue.append(self.torch_graph.name_to_node[predecessor])
                    out_degree.pop(predecessor)
            yield node
        assert len(out_degree) == 0

    def debug_hash_value_one(self, obj):
        if isinstance(obj, torch.Tensor) and obj.numel() > 1:
            torch.manual_seed(100)
            out = torch.sum(torch.rand_like(obj.to(torch.float)) * obj).item()
            if obj.grad is not None:
                torch.manual_seed(100)
                grad = torch.sum(torch.rand_like(obj.grad.to(torch.float)) * obj.grad).item()
                return [out, 'grad: %s' % grad]
            else:
                return [out, 'no grad']
        else:
            return obj

    def debug_hash_value(self, obj):
        return map_recursive(self.debug_hash_value_one, obj)

    def tensor_propagate_check(self, obj):
        # return isinstance(obj, torch.Tensor) and obj.dim() > self.batch_dim and obj.size(self.batch_dim) == self.confidence
        return isinstance(obj, torch.Tensor) and obj.numel() > self.confidence and obj.numel() % self.confidence == 0

    def tensor_detacher(self, obj: torch.Tensor):
        if self.tensor_propagate_check(obj):
            return obj.detach()
        return obj

    def tensor_cloner(self, obj: torch.Tensor):
        if self.tensor_propagate_check(obj):
            return obj.clone()
        return obj

    def tensor_clone_detacher(self, obj: torch.Tensor):
        if self.tensor_propagate_check(obj):
            return obj.clone().detach()
        return obj

    def slot_getter_value_0(self, input_name: str):
        assert isinstance(input_name, str)
        assert input_name in self.slots, 'slot error: lack of slot(%s)' % input_name
        assert self.slots[input_name].status['value_0'] == 1, 'slot error: bad value_0(%d)' % self.slots[input_name].status['value_0']

        return self.slots[input_name].value_0

    def slot_getter_value_1(self, input_name: str):
        assert isinstance(input_name, str)
        assert input_name in self.slots, 'slot error: lack of slot(%s)' % input_name
        assert self.slots[input_name].status['value_1'] == 1, 'slot error: bad value_1(%d)' % self.slots[input_name].status['value_1']

        return self.slots[input_name].value_1

    def slot_getter_value_2(self, input_name: str):
        assert isinstance(input_name, str)
        assert input_name in self.slots, 'slot error: lack of slot(%s)' % input_name
        assert self.slots[input_name].status['value_2'] >= 1, 'slot error: bad value_2(%d)' % self.slots[input_name].status['value_2']

        return self.slots[input_name].value_2

    def clone_randomizer(self, obj):
        import copy
        # if self.tensor_propagate_check(obj):
        if isinstance(obj, torch.Tensor) and obj.numel() != 1 and len(obj.size()) > self.batch_dim\
            and obj.size(self.batch_dim) == self.confidence:
            new_obj = obj.clone().detach()
            if not new_obj.is_contiguous():
                new_obj = new_obj.contiguous()
            randomize_tensor(new_obj, start=0.1, end=8.0)
            return new_obj
        else:
            try:
                return copy.deepcopy(obj)
            except copy.Error:
                return obj

    def slot_getter_mask_1(self, input_name: str):
        assert isinstance(input_name, str)
        assert input_name in self.slots, 'slot error: lack of slot(%s)' % input_name
        assert self.slots[input_name].status['mask_1'] == 1, 'slot error: bad mask_1(%d)' % self.slots[input_name].status['mask_1']

        return self.slots[input_name].mask_1

    def slot_getter_mask_2(self, input_name: str):
        assert isinstance(input_name, str)
        assert input_name in self.slots, 'slot error: lack of slot(%s)' % input_name
        if self.slots[input_name].mask_2 is None:
            return None
        else:
            assert self.slots[input_name].status['mask_2'] >= 1, 'slot error: bad mask_2(%d)' % self.slots[input_name].status['mask_2']

            return self.slots[input_name].mask_2

    def slot_getter_mask_2_or_1(self, input_name: str):
        assert isinstance(input_name, str)
        assert input_name in self.slots, 'slot error: lack of slot(%s)' % input_name
        if self.slots[input_name].mask_2 is None:
            assert self.slots[input_name].status['mask_1'] == 1, 'slot error: bad mask_1(%d)' % self.slots[input_name].status['mask_1']

            return self.slots[input_name].mask_1
        else:
            assert self.slots[input_name].status['mask_2'] >= 1, 'slot error: bad mask_2(%d)' % self.slots[input_name].status['mask_2']

            return self.slots[input_name].mask_2

    def mask_applier(self, value, mask):
        if self.tensor_propagate_check(value):
            assert isinstance(mask, torch.Tensor) and value.shape == mask.shape
            return value * mask
        else:
            assert mask is None
            return value

    def calc_one_mask(self, obj):
        if self.tensor_propagate_check(obj):
            obj: torch.Tensor
            STD_DELTA = 1e-6

            out_mask = torch.ones_like(obj)
            if obj.dtype in torch_integer_dtype:
                same = obj[:] == obj[0]
                reduced = torch.sum(same, dim=0)
                is_constant = reduced == obj.size(0)
                out_mask[:, is_constant] = 0
            else:
                std = torch.std(obj, dim=0)
                mask_pos = std < STD_DELTA
                out_mask[:, mask_pos] = 0
            return out_mask
        else:
            return None

    def tensor_requires_grad(self, obj):
        if self.tensor_propagate_check(obj) and obj.dtype in torch_float_dtype:
            # only float type can require the gradient
            # enable the auto gradient
            obj.requires_grad_(True)

    def _random_model_input(self, dummy_input, batch_dim, confidence):
        """
        Get the new random dummy input accordint to the original dummy_input
        and confidence, batch_dim.

        Parameters
        ----------
        dummy_input: Tensor or list/dict of Tensors
            The dummy_input given by the user.
        confidence: int
            The new batch size of the generated dummy_input.
        batch_dim: int
            The index of the batch dimension.

        Returns
        ------
        new_dummy_input: Tensor or list/dict of Tensors
            The generated dummy_input for mask inference.
        device: torch.device
            The device of the generated dummy_inputs
        """
        input_errmsg = 'Only support the tensor, list/tuple/dict of tensors as input'
        # Some model may use list of tensors as input, for example transformers
        new_dummy_input, device = None, None
        if isinstance(dummy_input, torch.Tensor):
            input_shape = list(dummy_input.size())
            # set the batchsize to the confidence ratio
            input_shape[batch_dim] = confidence
            new_dummy_input = rand_like_with_shape(input_shape, dummy_input)
            device = dummy_input.device
        elif isinstance(dummy_input, (tuple, list)):
            # else if the dummy input is list/tuple
            new_dummy_input = []
            old_batchsize = dummy_input[0].size(0)
            device = dummy_input[0].device
            for _, t_input in enumerate(dummy_input):
                assert isinstance(t_input, torch.Tensor), input_errmsg
                assert t_input.size(0) == old_batchsize, 'The first dimension should be batchsize\
                    and the batchsize of all inputs should be the same!'
                input_shape = list(t_input.size())
                input_shape[batch_dim] = confidence
                # rand_func = torch.randint if t_input.dtype
                new_dummy_input.append(
                    rand_like_with_shape(input_shape, t_input))
        elif isinstance(dummy_input, dict):
            new_dummy_input = {}
            tmp_key = list(dummy_input.keys())[0]
            old_batchsize = dummy_input[tmp_key].size(0)
            device = dummy_input[tmp_key].device
            for in_name, t_input in dummy_input.items():
                assert isinstance(t_input, torch.Tensor), input_errmsg
                assert old_batchsize == t_input.size(0), 'The first dimension should be batchsize\
                and the batchsize of all inputs should be the same!'
                input_shape = list(t_input.size())
                input_shape[batch_dim] = confidence
                new_dummy_input[in_name] = rand_like_with_shape(
                    input_shape, t_input)
        else:
            raise TypeError(input_errmsg)
        return new_dummy_input, device

    def r_random_model_input(self, dummy_input, batch_dim, confidence):
        devices = set()
        def model_tensor_randomizer(obj):
            if isinstance(obj, torch.Tensor) and obj.dim() > batch_dim:
                devices.add(dummy_input.device)
                input_shape = list(obj.size())
                # set the batchsize to the confidence ratio
                input_shape[batch_dim] = confidence
                return rand_like_with_shape(input_shape, dummy_input)
            else:
                return obj
        new_dummy_input = map_recursive(model_tensor_randomizer, dummy_input)
        assert len(devices) == 1
        return new_dummy_input, list(devices)[0]

    def r_propagate_orig(self):
        self.node_infos: dict[NodePyIO, NodeInfo]
        self.slots: dict[str, Slot]

        _logger.info("propagate original variables")
        for node in self.direct_order():
            module_name = node.name
            unique_name = node.unique_name
            node_info = self.node_infos[node]

            _logger.info('r_Propagate variables for %s', unique_name)

            input_names = node_info.slots_in
            input_values = map_recursive(self.slot_getter_value_1, input_names)
            input_value_detachs = map_recursive(self.tensor_detacher, input_values)

            output_values = node_info.flatten_caller(*input_value_detachs)

            if isinstance(node_info.slots_out, str):
                slot_out = node_info.slots_out
                self.slots[slot_out].value_0 = output_values
                self.slots[slot_out].status['value_0'] += 1
                self.slots[slot_out].value_1 = map_recursive(self.tensor_clone_detacher, output_values)
                self.slots[slot_out].status['value_1'] += 1
            else:
                assert len(node_info.slots_out) == len(output_values)
                for slot_out, output_value in zip(node_info.slots_out, output_values):
                    self.slots[slot_out].value_0 = output_value
                    self.slots[slot_out].status['value_0'] += 1
                    self.slots[slot_out].value_1 = map_recursive(self.tensor_clone_detacher, output_value)
                    self.slots[slot_out].status['value_1'] += 1

    def r_update_direct_sparsity(self):
        self.node_infos: dict[NodePyIO, NodeInfo]
        self.slots: dict[str, Slot]

        _logger.info("update direct sparsity")

        for slot in self.slots.values():
            slot.value_2 = map_recursive(self.clone_randomizer, slot.value_0)
            slot.status['value_2'] += 1

        for node in self.direct_order():
            module_name = node.name
            unique_name = node.unique_name
            node_info = self.node_infos[node]

            _logger.info('r_Update mask for %s', unique_name)

            with torch.no_grad():
                if isinstance(node_info.flatten_caller, nn.Module):
                    sub_module = node_info.flatten_caller
                    for _k, v in sub_module.named_parameters():
                        randomize_tensor(v.data, start=0.1, end=8.0)

                    for k, v in sub_module.named_parameters():
                        sub_module.register_parameter(
                            k,
                            torch.nn.Parameter(v * node_info.param_masks_0[k])
                        )

                input_values = map_recursive(self.slot_getter_value_2, node_info.slots_in)

                input_masks = map_recursive(self.slot_getter_mask_1, node_info.slots_in)
                input_values = map_recursive_zip(self.mask_applier, input_values, input_masks)

                output_values = node_info.flatten_caller(*input_values)

                output_masks = map_recursive(self.calc_one_mask, output_values)

                if isinstance(node_info.slots_out, str):
                    output_name = node_info.slots_out
                    self.slots[output_name].mask_1 = output_masks
                    self.slots[output_name].status['mask_1'] += 1
                else:
                    assert len(node_info.slots_out) == len(output_masks)
                    for output_name, output_mask in zip(node_info.slots_out, output_masks):
                        self.slots[output_name].mask_1 = output_mask
                        self.slots[output_name].status['mask_1'] += 1

    def r_update_indirect_sparsity(self):
        """
        This function will update the indirect sparsity. To explain what's
        indirect sparsity, for example, there is two tensors TA and TB, and
        we perform the calculation: TC = TA x TB in which TC is also a tensor.
        Once some values in TA are masked to zeros, then the corresponding
        positions in TB are also potential sparsities, because these have no
        effect of the final output(the gradient of these positions in TB equal
        to 0 all the time). This function it to fine the potential sparsity caused
        by other sparsity(we call it indirect sparsity here). Basically we can find
        these potential sparsity through gradient.

        Parameters
        ---------
        node: the NodePy
            The target node to update the indirect sparsity
        """
        # update indirect out mask
        def calc_indirect_mask(mask, obj):
            if self.tensor_propagate_check(obj):
                assert isinstance(mask, torch.Tensor) and obj.shape == mask.shape
                if obj.grad is not None:
                    gradient_sum = torch.sum(torch.abs(obj.grad), dim=0)
                    _grad_zero = gradient_sum == 0
                    new_mask = mask.clone()
                    for batchid in range(obj.size(0)):
                        # set the same mask value for the whole batche
                        new_mask[batchid][_grad_zero] = 0
                    return new_mask
            return mask

        def update_indirect_weight_mask_helper(output, mask):
            # Note: output maybe tensor or list/tuple of tensors
            if self.tensor_propagate_check(output):
                assert isinstance(mask, torch.Tensor)
                if output.grad_fn is not None:
                    output.backward(mask)
            else:
                assert not isinstance(mask, torch.Tensor)

        # pass the gradient to the predecessor nodes
        def pass_grad(slot_val, out):
            if self.tensor_propagate_check(slot_val):
                assert isinstance(out, torch.Tensor)
                if slot_val.grad is not None and out.grad is not None:
                    # slot_val.grad.data += out.grad.data
                    slot_val.grad += out.grad
                elif slot_val.grad is None:
                    slot_val.grad = out.grad
                elif slot_val.grad is not None and out.grad is None:
                    # for example, tin.view(batch, tin.size(1)/2, tin.view(2)*2)
                    # the size operation of tin will have no gradient
                    pass

        self.node_infos: dict[NodePyIO, NodeInfo]
        self.slots: dict[str, Slot]

        _logger.info("update indirect sparsity")
        for node in self.indirect_order():
            module_name = node.name
            unique_name = node.unique_name
            node_info = self.node_infos[node]

            _logger.info('r_Update the indirect sparsity for the %s', unique_name)

            if isinstance(node_info.flatten_caller, nn.Module):
                sub_module = node_info.flatten_caller
                for k, v in sub_module.named_parameters():
                    if v.dtype in torch_float_dtype:
                        self.tensor_requires_grad(v)

            output_values = map_recursive(self.slot_getter_value_2, node_info.slots_out)
            output_masks_1 = map_recursive(self.slot_getter_mask_1, node_info.slots_out)
            output_masks_2 = map_recursive_zip(calc_indirect_mask, output_masks_1, output_values)

            if isinstance(node_info.slots_out, str):
                output_name = node_info.slots_out
                self.slots[output_name].mask_2 = output_masks_2
                self.slots[output_name].status['mask_2'] += 1
            else:
                assert len(node_info.slots_out) == len(output_masks_2)
                for output_name, output_mask in zip(node_info.slots_out, output_masks_2):
                    self.slots[output_name].mask_2 = output_mask
                    self.slots[output_name].status['mask_2'] += 1

            # init apply input
            # randomized, so it's same to use slot_getter_value_orig or slot_getter_value_orig_inplace
            input_values = map_recursive(self.slot_getter_value_0, node_info.slots_in)
            input_values = map_recursive(self.clone_randomizer, input_values)

            input_masks = map_recursive(self.slot_getter_mask_1, node_info.slots_in)
            input_values = map_recursive_zip(self.mask_applier, input_values, input_masks)
            map_recursive(self.tensor_requires_grad, input_values)

            output_values = node_info.flatten_caller(*input_values)

            # out_masks = map_recursive(self.slot_getter_mask_2, node_info.slots_out)

            map_recursive_zip(update_indirect_weight_mask_helper, output_values, output_masks_2)
            if isinstance(node_info.flatten_caller, nn.Module):
                sub_module = node_info.flatten_caller
                for k, v in sub_module.named_parameters():
                    grad_zero = v.grad.data == 0
                    node_info.param_masks_1[k] = node_info.param_masks_0[k].clone()
                    node_info.param_masks_1[k][grad_zero] = 0

            input_values_2 = map_recursive(self.slot_getter_value_2, node_info.slots_in)
            map_recursive_zip(pass_grad, input_values_2, input_values)

    def _vnode_to_value(self, c_node):
        """
        translate the C Value node into the values/tensors.
        """
        errmsg = "Only support the torch._C.Value type"
        assert isinstance(c_node, torch._C.Value), errmsg
        if isinstance(c_node.type(), torch._C.TensorType):
            shape = tuple(c_node.type().sizes())
            dtype = c_node.type().scalarType()
            # TODO should use a more general way to get the input
            if dtype.startswith('Float') or dtype.startswith('Double'):
                return torch.rand(shape).to(self.device)
            else:
                # This small range is due to the ·ReLU6·, we will add
                # the manual specific mask inference rule for several
                # ops in the future, so that we can remove the constraint.
                return torch.randint(0, 10, shape, device=self.device)
        else:
            value = c_node.toIValue()
            # TODO support more kinds of value node
            errmsg = "Doesn't support convert %s to values", str(c_node.type())
            # currently only support the tensors and constant values
            assert value is not None, errmsg
            return value


    def r_replace_compressed_modules(self):
        # load the original stat dict before replace the model
        self.bound_model.load_state_dict(self.ori_state_dict)
        _logger.info("replace compressed modules...")

        with torch.no_grad():
            for node in self.direct_order():
                self.r_replace_submodule(node)

    def r_replace_submodule(self, node):
        node: NodePyGroup
        _logger.debug("replace %s, in %s type, with op_type %s", node.unique_name, node.type, node.op_type)
        node_info = self.node_infos[node]
        def tensors_flattener(masks):
            flattened = []
            def helper(obj):
                if isinstance(obj, torch.Tensor) and obj.numel() > 1:
                    flattened.append(obj)
            map_recursive(helper, masks)
            return flattened

        if node.type == 'module':
            super_module, leaf_module = get_module_by_name(self.bound_model, node.name)
            if (not node.op_type in replace_module) and (node.op_type not in self.customized_replace_func):
                err_msg = f"Has not supported replacing module with type: {node.op_type}, "
                err_msg += f"you could report an issue at https://github.com/microsoft/nni. "
                err_msg += f"If you know how to replace {node.op_type}, "
                err_msg += f"you could implement module replacement by passing in"
                err_msg += f"`customized_replace_func` to `{self.__class__.__name__}`. "
                err_msg += f"You are welcome to contribute back to nni as native support if you have implemented the replacement function, "
                err_msg += f"so that more users can benefit from your contributions."
                raise RuntimeError(err_msg)
            _logger.info("replace module (name: %s, op_type: %s)", node.name, node.op_type)
            replace_function = self.customized_replace_func.get(node.op_type, replace_module.get(node.op_type, None))

            in_masks = tensors_flattener(map_recursive(self.slot_getter_mask_2_or_1, node_info.slots_in))
            out_masks = map_recursive(self.slot_getter_mask_2_or_1, node_info.slots_out)
            param_masks = node_info.param_masks_1

            compressed_module = replace_function(leaf_module, (in_masks, out_masks, param_masks))
            
            # print('\ninter var in replacement:')
            # print('replace module:', node.name)
            # print('replace in_masks:')
            # print(self.debug_hash_value(in_masks))
            # print('replace out_masks:')
            # print(self.debug_hash_value(out_masks))
            # print('replace weight_mask:')
            # print(self.debug_hash_value(param_masks))
            
            new_submodule = compressed_module
            setattr(super_module, node.name.split('.')[-1], compressed_module)
            return new_submodule
        else:
            return None

    def r_initialize_propagate(self):
        self.node_infos: dict[NodePyIO, NodeInfo] = {}
        self.slots: dict[str, Slot] = {}

        _logger.info("infer module masks...")
        for node in self.torch_graph.nodes_py.nodes_op:
            node: NodePyGroup

            if node.type == 'func':
                c_node:torch._C.Node = node.key_node
                slots_in = list(i.debugName() for i in c_node.inputs())
                slots_out = list(i.debugName() for i in c_node.outputs())
                flatten_caller = r_jit_to_python_function(node)
                param_masks = None
            else:
                slots_in: list[str] = node.inputs
                slots_out: list[str] = node.outputs
                flatten_caller = get_module_by_name(self.bound_model, node.name)[1]
                param_masks = self.masks.get(node.name, {})
                for k, v in flatten_caller.named_parameters():
                    if k not in param_masks:
                        param_masks[k] = torch.ones_like(v)

            if len(slots_out) == 1:
                slots_out = slots_out[0]
            else:
                # assert prim::output/listunpack/tupleunpack
                pass

            self.node_infos[node] = NodeInfo(
                slots_in,
                slots_out,
                flatten_caller,
                param_masks
            )

        @run_onlyif_instance(torch.Tensor)
        def value_rand_init(obj: torch.Tensor):
            return torch.rand_like(obj)

        @run_onlyif_instance(torch.Tensor, False)
        def mask_ones_init(obj: torch.Tensor):
            return torch.ones_like(obj)

        for c_node in self.torch_graph.trace.graph.nodes():
            c_node: torch._C.Node

            slots_in = (i.debugName() for i in c_node.inputs())
            slots_out = (i.debugName() for i in c_node.outputs())
            for slot_name in (*slots_in, *slots_out):
                if slot_name not in self.slots:
                    self.slots[slot_name] = Slot()

            if c_node.kind() == 'prim::Constant':
                c_value: torch._C.Value = c_node.output()
                value = c_value.toIValue()
                self.slots[c_value.debugName()].value_0 = value
                self.slots[c_value.debugName()].status['value_0'] += 1
                self.slots[c_value.debugName()].value_1 = map_recursive(self.tensor_cloner, value)
                self.slots[c_value.debugName()].status['value_1'] += 1

                self.slots[c_value.debugName()].mask_1 = map_recursive(mask_ones_init, value)
                self.slots[c_value.debugName()].status['mask_1'] += 1

        for input_v_node in self.torch_graph.trace.graph.inputs():
            input_v_node: torch._C.Value
            assert input_v_node.node().inputsSize() == 0
            slot_name = input_v_node.debugName()
            # will not be accessed if slot_name is not in self.slots
            if slot_name in self.slots:
                if input_v_node.type().kind() == 'ClassType':
                    self.slots[slot_name].value_0 = self.bound_model
                    self.slots[slot_name].status['value_0'] += 1
                    self.slots[slot_name].value_1 = self.bound_model
                    self.slots[slot_name].status['value_1'] += 1

                    self.slots[slot_name].mask_1 = map_recursive(mask_ones_init, self.bound_model)
                    self.slots[slot_name].status['mask_1'] += 1
                else:
                    value = self._vnode_to_value(input_v_node)
                    self.slots[slot_name].value_0 = value
                    self.slots[slot_name].status['value_0'] += 1
                    self.slots[slot_name].value_1 = map_recursive(value_rand_init, value)
                    self.slots[slot_name].status['value_1'] += 1

                    self.slots[slot_name].mask_1 = map_recursive(mask_ones_init, value)
                    self.slots[slot_name].status['mask_1'] += 1

    def speedup_model(self):
        """
        There are basically two steps: first, do mask/shape inference,
        second, replace modules.
        """

        _logger.info("start to speedup the model")
        self.training = self.bound_model.training
        # set to the evaluation mode
        self.bound_model.train(False)
        # TODO suppose to fix the conflict after the sparsity propagation
        # which is more elegent
        fix_mask_conflict(self.masks, self.bound_model, self.dummy_input)

        self.r_initialize_propagate()

        self.r_propagate_orig()

        self.r_update_direct_sparsity()

        self.r_update_indirect_sparsity()

        _logger.info('resolve the mask conflict')
        # the mask conflict should be already resolved

        self.r_replace_compressed_modules()

        self.bound_model.train(self.training)

        _logger.info("speedup done")
