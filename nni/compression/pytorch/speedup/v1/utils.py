import builtins
import operator
from typing import Any, Callable, Type
import functools

import torch


def run_onlyif_instance(cond_type: Type[Any], return_orig: bool = True, return_const: Any = None):
    def helper(fn):
        if return_orig:
            @functools.wraps(fn)
            def wrapper(*args):
                if isinstance(args[-1], cond_type):
                    return fn(*args)
                return args[-1]
            return wrapper
        else:
            @functools.wraps(fn)
            def wrapper(*args):
                if isinstance(args[-1], cond_type):
                    return fn(*args)
                return return_const
            return wrapper
    return helper

def map_recursive(fn: Callable, arg) -> Any:
    """
    Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys.
    """
    if (not isinstance(arg, torch.Size)) and isinstance(arg, tuple):
        t = tuple(map_recursive(fn, elem) for elem in arg)
        # Support NamedTuple (if it has `_fields`) by repacking into original type.
        return t if not hasattr(arg, '_fields') else type(arg)(*t)
    elif isinstance(arg, list):
        return list(map_recursive(fn, elem) for elem in arg)
    elif isinstance(arg, dict):
        return {k: map_recursive(fn, v) for k, v in arg.items()}
    else:
        return fn(arg)

def map_recursive_zip(fn: Callable, arg0, *args) -> Any:
    """
    Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys.
    """
    if (not isinstance(arg0, torch.Size)) and isinstance(arg0, tuple):
        for arg in args:
            assert (not isinstance(arg, torch.Size)) and isinstance(arg, tuple)
            assert len(arg0) == len(arg)
        return tuple(map_recursive_zip(fn, *sub_args) for sub_args in zip(arg0, *args))
    elif isinstance(arg0, list):
        for arg in args:
            assert isinstance(arg, list)
            assert len(arg0) == len(arg)
        return list(map_recursive_zip(fn, *sub_args) for sub_args in zip(arg0, *args))
    elif isinstance(arg0, dict):
        keys = set(arg0.keys())
        keys_len = len(keys)
        for arg in args:
            assert isinstance(arg, dict)
            keys.update(arg.keys())
            assert keys_len == len(keys)
        return {k: map_recursive_zip(fn, arg0[k], *(arg[k] for arg in args)) for k in keys}
    else:
        # assert not isinstance(arg0, slice)
        return fn(arg0, *args)
