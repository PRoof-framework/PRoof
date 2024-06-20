from __future__ import annotations
from functools import partial
from typing import Callable, Dict, List, Union

import numpy as np

from datatype import ModelBenchBatchLayerData

# use getter functions for each backend to avoid unnecessary imports
__backend_getters: Dict[str, Callable[[], _BaseBackend]] = {}
__register_backend = lambda name: partial(__backend_getters.__setitem__, name)

@__register_backend('nart_trt')
def _():
    from .nart import Nart_TRT
    return Nart_TRT

@__register_backend('trtexec')
def _():
    from .trtexec import Trtexec
    return Trtexec

@__register_backend('onnxruntime')
def _():
    from .onnxruntime import ONNXRuntimeBackend
    return ONNXRuntimeBackend

@__register_backend('openvino')
def _():
    from .openvino import OpenVINOBackend
    return OpenVINOBackend


def get_available_backends() -> List[str]:
    return list(__backend_getters.keys())

def get_backend(name: str) -> _BaseBackend:
    return __backend_getters[name]()


class _BaseBackend():
    supported = ['e2e_prof', 'layer_prof']

    def __init__(self, ctx, onnx_model: str, batch_size_list: list, backend_options: str) -> None:
        raise NotImplementedError

    def version_info(self) -> str:
        raise NotImplementedError

    def prepare(self) -> None:
        raise NotImplementedError

    def pre_batch_run(self, batch_size: int) -> None:
        raise NotImplementedError

    def e2e_prof(self, batch_size: int, repeat: int = 10, warm_up: int = 3) -> Union[np.ndarray, dict]:
        "return a numpy array of latences, or a dict like {'avg': .., 'min': ..., 'std': ...}, in second"
        raise NotImplementedError

    def layer_prof(self, batch_size: int) -> List[ModelBenchBatchLayerData]:
        raise NotImplementedError
