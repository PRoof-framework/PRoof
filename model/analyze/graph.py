from typing import Dict, List
from dataclasses import dataclass, field
import logging

from datatype import TensorShape
from .op import _BaseOp

log = logging.getLogger(__name__)


@dataclass
class TensorInfo(TensorShape):
    is_param: bool
    produced_node: _BaseOp = None
    required_nodes: List[_BaseOp] = field(default_factory=list)


def get_tensor_info_from_ops(op_list: List[_BaseOp]) -> Dict[str, TensorInfo]:
    tensors: Dict[str, TensorInfo] = {}
    for op in op_list:
        for input in op.inputs:
            if input.name in tensors:
                tensors[input.name].required_nodes.append(op)
            else:
                tensors[input.name] = TensorInfo(
                    name = input.name,
                    shape = input.shape,
                    is_param = input.name in op.params,
                    required_nodes = [op],
                )
        for output in op.outputs:
            if output.name in tensors:
                if tensors[output.name].produced_node:  # TODO: tmp
                    log.error("multi produced_node, is this correct?")
                tensors[output.name].produced_node = op
            else:
                tensors[output.name] = TensorInfo(
                    name = output.name,
                    shape = output.shape,
                    is_param = False,
                    produced_node = op,
                )
    return tensors
