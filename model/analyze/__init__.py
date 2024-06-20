from typing import Dict, List, Literal
from itertools import chain
import logging

import onnx

from datatype import AnalyzeData
from .model import onnx_model_layer_profile
from .graph import TensorInfo, get_tensor_info_from_ops
from .op import _BaseOp, _PointwiseOp, _ReductionOp, ReshapeOp, ShapeOp

log = logging.getLogger(__name__)


class Analyze():
    def __init__(self, onnx_model: str) -> None:
        log.info("analyze %s", onnx_model)
        self.model = onnx.load(onnx_model)

        # it will set batch_size to 1
        self.data, self.ops = onnx_model_layer_profile(self.model)

        self.tensors: Dict[str, TensorInfo] = self._get_tensors_info()

    def _get_tensors_info(self) -> Dict[str, TensorInfo]:
        return get_tensor_info_from_ops(self.ops.values())

    def export_data(self) -> AnalyzeData:
        return self.data

    def op_inputs(self, op: _BaseOp) -> List[TensorInfo]:
        assert op.name in self.ops
        return [self.tensors[x.name] for x in op.inputs]

    def op_outputs(self, op: _BaseOp) -> List[TensorInfo]:
        assert op.name in self.ops
        return [self.tensors[x.name] for x in op.outputs]

    def get_flops(self) -> int:
        return sum(op.get_flops() for op in self.ops.values())

    def get_memory(self) -> int:
        return sum(op.get_memory() for op in self.ops.values())

    def get_prev_op(self, op: _BaseOp) -> List[_BaseOp]:
        """get the list of 'prev' (upstream) op, which output is current op's input"""
        assert op.name in self.ops
        prev_op_list: List[_BaseOp] = []
        for i in op.inputs:
            t = self.tensors[i.name]
            if t.produced_node:
                prev_op_list.append(t.produced_node)
        return prev_op_list

    def get_next_op(self, op: _BaseOp) -> List[_BaseOp]:
        """get the list of 'next' (downstream) op, which use current op's output as input"""
        assert op.name in self.ops
        next_op_list: List[_BaseOp] = []
        for o in op.outputs:
            t = self.tensors[o.name]
            next_op_list += t.required_nodes
        return next_op_list

    def get_op_may_fused_with(self, op: _BaseOp) -> List[_BaseOp]:
        """simple method to get the list of the downstream op which may fused with current op (current op not included), only consider dependence and op type (like _ReductionOp + _PointwiseOp)"""
        assert op.name in self.ops
        op_list: List[_BaseOp] = []
        while True:
            next_ops = self.get_next_op(op)
            if len(next_ops) != 1:
                break
            op = next_ops[0]
            if isinstance(op, _PointwiseOp):
                op_list.append(op)
            else:
                break
        return op_list

    def get_subgraph_ops_by_io(self, inputs: List[str], outputs: List[str]) -> List[_BaseOp]:
        """get subgraph (a set of op) by the boundary tensors"""
        # reversed lookup to prevent unrelated node when input tensors is required by more than one node
        inputs = set(inputs)

        op_list: List[_BaseOp] = []
        visited = set()
        for t in outputs:
            if t not in inputs:
                t = self.tensors[t]
                n = t.produced_node
                if n and n.name not in visited:
                    op_list.append(n)
                    visited.add(n.name)

        p = 0
        while p < len(op_list):
            for t in op_list[p].inputs:
                # print('at', t.name)
                if t.name not in inputs:    # endpoint not reached
                    # print('reach', t.name)
                    t = self.tensors[t.name]
                    n = t.produced_node
                    if n and n.name not in visited:
                        op_list.append(n)
                        # print('  append node', n.name)
                        visited.add(n.name)
            p += 1

        op_list.reverse()   # reverse it for better readability when debug
        return op_list

    def remove_op(self, op: _BaseOp):
        if op.name in self.ops:
            for t in op.inputs:
                t = self.tensors[t.name]
                t.required_nodes.remove(op)
                if not t.required_nodes and not t.produced_node:
                    # print(f"remove_op: remove tensor {t}")
                    del self.tensors[t.name]
            for t in op.outputs:
                t = self.tensors[t.name]
                t.produced_node = None
                if not t.required_nodes and not t.produced_node:
                    # print(f"remove_op: remove tensor {t}")
                    del self.tensors[t.name]
            # print(f"remove_op: remove op {op}")
            del self.ops[op.name]
        # else:
        #     print(f"remove_op: already removed {op}")

    def purge_shape_op(self):
        def _is_shape_calculation_op(op: _BaseOp):
            return all(len(t.shape) <= 1 and t.size() <= 10 for t in chain(op.inputs, op.outputs))

        def _dfs_down_remove(op: _BaseOp):
            for t in op.outputs:
                for downstream_op in self.tensors[t.name].required_nodes[:]:
                    if downstream_op.name in self.ops and _is_shape_calculation_op(downstream_op): #and not isinstance(downstream_op, ReshapeOp):
                        _dfs_down_remove(downstream_op)
            self.remove_op(op)
            # print("_dfs_down_remove: removed", op)

        for op in list(self.ops.values()):
            if op.name in self.ops and isinstance(op, ShapeOp):
                _dfs_down_remove(op)

    # def get_memory_with_cache_reduced(self, op: _BaseOp, batch_size:int, cache_size: int, reduce_type: Literal['input', 'output', 'both'] = 'output', prev_op: _BaseOp = None, next_op: _BaseOp = None) -> int:
    #     """cache_size: num of vars cache can hold, e.g. 512 for 1KiB cache under fp16"""
    #     assert op.name in self.ops
    #     assert not prev_op or prev_op.name in self.ops
    #     assert not next_op or next_op.name in self.ops

    #     prev_out = self.op_outputs(prev_op) if prev_op else []
    #     next_in = self.op_inputs(next_op) if next_op else []

    #     cache_remains = cache_size
    #     total_mem = 0
    #     for t in self.op_inputs(op):
    #         if not t.is_param:
    #             mem = t.size() * batch_size
    #         else:
    #             mem = t.size()
    #         if reduce_type in ('input', 'both') and t in prev_out:
    #             cached = min(cache_remains, mem)
    #             mem -= cached
    #             cache_remains -= cached
    #         total_mem += mem

    #     for t in self.op_outputs(op):
    #         if not t.is_param:
    #             mem = t.size() * batch_size
    #         else:
    #             mem = t.size()
    #         if reduce_type in ('output', 'both') and t in next_in \
    #             and len(t.required_nodes) <= 1:
    #             cached = min(cache_remains, mem)
    #             mem -= cached
    #             cache_remains -= cached
    #         total_mem += mem

    #     cache_used = cache_size - cache_remains
    #     log.debug(f"assume l2 cache used {cache_used / 1e6:.3f} / {cache_size / 1e6:.3f} at {op.name}")
    #     log.debug(f"remain memory access {total_mem / 1e6:.3f}")
    #     for i in op.inputs:
    #         log.debug(f"i {i}")
    #     for o in op.outputs:
    #         log.debug(f"o {o}")

    #     log.debug(f"    + prev {prev_op}")
    #     if prev_op:
    #         for i in prev_op.inputs:
    #             log.debug(f"      i {i}")
    #         for o in prev_op.outputs:
    #             log.debug(f"      o {o}")

    #     log.debug(f"    + next {next_op}")
    #     if next_op:
    #         for i in next_op.inputs:
    #             log.debug(f"      i {i}")
    #         for o in next_op.outputs:
    #             log.debug(f"      o {o}")
    #     log.debug(f"")

    #     return total_mem, cache_used
