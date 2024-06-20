from __future__ import annotations
from copy import copy
from math import prod

from typing import Any, List, Dict, Union, Optional
from dataclasses import dataclass
import logging

log = logging.getLogger(__name__)


# provide obj['xxx'] as obj.xxx for dataclassed
class _MixedAccess():
    def __contains__(self, v):
        return hasattr(self, v)
    def __getitem__(self, *a, **k):
        log.debug("DEPRECATED: _MixedAccess %s is getting attr by dict like access (*%s, **%s)", self, a, k)
        return self.__getattribute__(*a, **k)
    def __setitem__(self, *a, **k):
        log.debug("DEPRECATED: _MixedAccess %s is setting attr by dict like access (*%s, **%s)", self, a, k)
        return self.__setattr__(*a, **k)


@dataclass
class CollectedData(_MixedAccess):
    subjects: List[str] = None
    roofline: Optional[RooflineData] = None
    model: Optional[ModelData] = None


@dataclass
class RooflineData(_MixedAccess):
    model_type: str = None
    backend: str = None
    backend_options: Optional[str] = None
    backend_version_info: Optional[str] = None
    data_width_onnx: Optional[int] = None
    data_width_backend: Optional[int] = None

    flops: float = None
    memory_bandwidth: float = None


@dataclass
class ModelData(_MixedAccess):
    name: str = None
    path: str = None
    backend: Optional[str] = None
    backend_options: Optional[str] = None
    backend_version_info: Optional[str] = None
    data_width_onnx: Optional[int] = None
    data_width_backend: Optional[int] = None
    llc_reuse_size: Optional[float] = None
    inputs_shape_override: Optional[Dict[str, Union[int, None]]] = None

    analyze: Optional[AnalyzeData] = None
    bench: Optional[ModelBenchData] = None


### model.analyze ###
TensorsShape = Dict[str, List[int]]

@dataclass
class AnalyzeData(_MixedAccess):
    inputs: TensorsShape = None
    nodes: Dict[str, NodeData] = None
    total_flops: int = None
    total_memory: int = None
    total_memory_effort_fused: int = None
    total_params: int = None


@dataclass
class NodeData(_MixedAccess):
    name: str = None
    type: str = None
    flops: int = None
    memory: int = None
    params: int = None
    inputs: List[TensorShape] = None
    outputs: List[TensorShape] = None


@dataclass
class TensorShape(_MixedAccess):
    name: str
    shape: List[int]

    def size(self):
        return prod(self.shape)

    # def batched(self, batch: int):
    #     # assert self.shape[0] == 1, f"{self.shape}[0] is not 1"
    #     cloned = copy(self)
    #     cloned.shape = cloned.shape[:]
    #     # cloned.shape[0] = batch * cloned.shape[0]
    #     if len(cloned.shape) >= 1 and cloned.shape[0] == 1:
    #         cloned.shape[0] = batch
    #     else:
    #         # normally not used
    #         cloned.shape = [batch] + cloned.shape
    #     return cloned

### model.banch ###
@dataclass
class ModelBenchData(_MixedAccess):
    batch_size_list: list = None
    results: Dict[str, ModelBenchBatchData] = None  # key: batch_size


@dataclass
class ModelBenchBatchData(_MixedAccess):
    batch_size: int = None
    times: List[float] = None
    time_avg: float = None
    time_min: float = None
    time_std: float = None
    flops_avg: float = None
    flops_max: float = None
    flops_std: float = None
    memory_avg: float = None
    memory_max: float = None
    memory_std: float = None
    layer_prof: Optional[List[ModelBenchBatchLayerData]] = None
    better_total_flops: Optional[int] = None
    better_total_memory: Optional[int] = None


@dataclass
class ModelBenchBatchLayerData(_MixedAccess):
    name: str = None
    median_time: float = None
    time_percentage: float = None
    flops: int = None
    memory: int = None
    extra: Dict[str, Any] = None
