# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Type, Union
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # Only imports the below statements during type checking
    from nni.compression.pytorch.speedup import ModelSpeedup
    from nni.common.graph_utils import NodePyGroup

import re
import string
import logging
from functools import partial, lru_cache
import copy
import torch

from nni.compression.pytorch.speedup.jit_translate import parse_aten_schema_version_1_8_x, parse_aten_schema


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
jitid_2_dtype = {4: torch.long, 6:torch.float32}

# to exclude partial

__all__ = [
    'getattr_python', 'r_jit_to_python_function', 'num2tensor_python', 'parse_constant', 'tupleunpack_python',
]

def parse_constant(cvalue: torch._C.Value, speedup: ModelSpeedup) -> Any:
    """
    Parse the constant values from this Node

    Parameters
    ----------
    cvalue
        The cpp node of the target constant value.
    speedup
        The Model speedup module.

    Returns
    -------
    value
        The constant values parsed from the node.
    """
    logger.debug('Try to parse the constant value: %s', cvalue.debugName())
    if cvalue.toIValue() is not None:
        return cvalue.toIValue()
    if cvalue.debugName() in speedup.internal_result:
        return speedup.internal_result[cvalue.debugName()]
    # Get the operator node of the this value
    op_node = cvalue.node()

    inputs = op_node.inputs()
    input_values = [parse_constant(_i, speedup) for _i in inputs]
    if op_node.kind() not in trans_func_dict:
        raise RuntimeError('Unsupported function op node type: {}'.format(op_node.kind()))

    func = trans_func_dict[op_node.kind()](op_node, speedup)
    return func(*input_values)

def tupleunpack_python(_node: NodePyGroup, _speedup: ModelSpeedup) -> Optional[Callable]:
    # Note: tuple unpack should only exists at the
    # the end of the model, and is no need to replace/propagate mask
    return None

def num2tensor_python(_node: NodePyGroup, _speedup: ModelSpeedup):
    return torch.nn.Identity()

def getattr_python(node: NodePyGroup, _speedup: ModelSpeedup):
    """
    Note: Ops started with Prim:: is not taken as the key node,
    so we directly pass the Cpp node into this funciton.

    Parameters
    ----------
    node
        The cpp node of prim::Getattr
    speedup
        The corresponding speedup object.
    """
    class GetModule(torch.nn.Module):
        def __init__(self, key):
            super(GetModule, self).__init__()
            self.key = key

        def forward(self, obj):
            logger.info('Get attribute: %s', self.key)
            return getattr(obj, self.key)
    # get the name of the attribute, for example
    # prim::GetAttr[name="module_list"](%self.1)
    assert node.kind() == 'prim::GetAttr'
    pattern = '\[name=\"(.*?)\"\]'
    key_words = re.findall(pattern, str(node))
    assert len(key_words) == 1
    return GetModule(key_words[0])

class r_FuncAdapter:
    """
    A function adapter which can reorder arguments.
    It can be initialate with constant argument, and positions of each non-constant
    argument. When called, it can put arguments into correct position, then call the
    function.

    Attributes
    ----------
    func
        The function or method to be called.
    positional
        Positional arguments values. The placeholder is None if it's non-constant.
    keyword
        Keyword arguments values. The placeholder is None if it's non-constant.
    undetermined
        A list of the right positions of arguments.
        Position is an int in positional or a str in keyword.
    special_treat
        A Dict of the positions and methods.
        The values of these positions should be treat by those methods.

    """

    def __init__(self, func: Callable, positional: int, keyword: List[str], special_treat: Dict[Union[int, str], Callable]):
        if not callable(func):
            raise TypeError('the "func" argument must be callable')

        self.func = func
        self.positional = positional
        self.keyword = keyword
        self.special_treat = special_treat

    def __call__(self, *args):
        assert len(args) == self.positional + len(self.keyword)

        adapted_args = args[:self.positional]
        adapted_kwargs = {k: v for k, v in zip(self.keyword, args[self.positional:])}

        result = self.func(*adapted_args, **adapted_kwargs)
        return result

def list_construct(*args) -> list:
    return list(args)

def tuple_construct(*args) -> tuple:
    return tuple(args)

def list_tuple_unpack(arg) -> Union[list, tuple]:
    return arg

trans_func_dict = {
    'prim::ListConstruct': list_construct,
    'prim::TupleConstruct': tuple_construct,

    'prim::ListUnpack': list_tuple_unpack,
    'prim::TupleUnpack': list_tuple_unpack,

    'prim::GetAttr': getattr_python,
}

def r_jit_to_python_function(node: NodePyGroup) -> r_FuncAdapter:
    """
    Return a callable object to inference the mask according to the node.op_type.

    Parameters
    ---------
    node
        The target node to inference the mask
    speedup
        The speedup object of the target model.

    Returns
    ------
    func
        Return the translated function that used to inference the mask
        , if current op_type is not supported, then we return None.
    """
    logger.debug('Translate C function %s into its python version', node.op_type)
    schema = node.key_node.schema()
    op_namespace_overload = schema[:schema.find('(')]

    if node.key_node.kind() in trans_func_dict:
        return trans_func_dict[node.key_node.kind()]
    elif op_namespace_overload in trans_func_dict:
        return trans_func_dict[op_namespace_overload]
    else:
        op_namespace = op_namespace_overload[:4]
        op_split_dot = schema.find('.')
        if op_split_dot == -1:
            op_name = op_namespace_overload[6:]
            op_overload = ''
        else:
            op_name = op_namespace_overload[6:op_split_dot]
            op_overload = op_namespace_overload[op_split_dot+1:]
        try:
            from torch._ops import OpOverload
            op_call: OpOverload = getattr(getattr(getattr(torch.ops, op_namespace), op_name), op_overload)
        except AttributeError:
            logger.error('%s is not Supported! Please report an issue at https://github.com/microsoft/nni. Thanks~', op_namespace_overload)
            return None

        if torch.__version__ < '1.9.0':
            positional_num, keyword_list, special_treat = parse_aten_schema_version_1_8_x(schema)
        else:
            positional_num, keyword_list, special_treat = parse_aten_schema(schema)
        return r_FuncAdapter(op_call, positional_num, keyword_list, special_treat)
