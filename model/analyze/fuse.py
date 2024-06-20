from typing import List, Dict, Tuple, Union
import sys
import itertools
import logging
from collections import defaultdict
from copy import deepcopy

from datatype import TensorShape
from . import Analyze, TensorInfo
from .op import _BaseOp, _DummyOp
from .graph import get_tensor_info_from_ops

log = logging.getLogger(__name__)

# subgraph
class _FusedOp(_BaseOp):
    "'visited' Op should been converted to _FusedOp(), even only contain 1 Op inside"
    def __init__(self, name: str, fused_ops: List[_BaseOp], analyze_info: Analyze):
        self._fused_ops = fused_ops
        self._analyze_info = analyze_info
        self._local_tensors = get_tensor_info_from_ops(fused_ops)
        self.name: str = name
        self.type: str = '_FusedOp'
        self.inputs: List[TensorInfo] = self._get_inputs()     # local TensorInfo, used as TensorShape (like normal Op), not the same one in Analyze.tensors
        self.outputs: List[TensorInfo] = self._get_outputs()
        self._io_tensor_origin_op = self._get_io_tensor_origin_op() # edge tensor name to original Op and position, {name: [(Op, pos), ...], ...}
        self.params: set = set(itertools.chain.from_iterable(x.params for x in self._fused_ops))
        self._check()

    def _check(self):
        ops_dict = {x.name: x for x in self._fused_ops}
        not_visited = set(x.name for x in self._fused_ops)
        subgraph = 0

        def dfs(op: Union[_BaseOp, None]):
            # print("_check dfs", op, not_visited)
            if op is not None and op.name in not_visited:
                # print("_check dfs visit", op)
                not_visited.remove(op.name)
                for t in op.inputs:
                    t = self._local_tensors[t.name]
                    dfs(t.produced_node)
                for t in op.outputs:
                    t = self._local_tensors[t.name]
                    for n in t.required_nodes:
                        dfs(n)

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10**6)

        while not_visited:
            subgraph += 1
            old_set = not_visited.copy()
            n = ops_dict[next(iter(not_visited))]
            dfs(n)
            delta = old_set - not_visited
            # print(f"    subgraph {subgraph}:", delta)

        sys.setrecursionlimit(old_limit)

        if subgraph != 1:
            log.error(f"_FusedOp.fused_ops should be in single connected subgraph, bus has {subgraph} subgraph")

    def _get_inputs(self) -> List[TensorInfo]:
        tensor_list = []
        for t in self._local_tensors.values():
            if not t.produced_node:
                tensor_list.append(t)
        return tensor_list

    def _get_outputs(self) -> List[TensorInfo]:
        tensor_list = []
        for t in self._local_tensors.values():
            if not t.required_nodes:
                # only used by node out of subgraph
                tensor_list.append(t)
            else:
                # has used by node in subgraph
                if t.produced_node:
                    # and is in the middle of subgraph (not a input of the subgraph)
                    global_t = self._analyze_info.tensors[t.name]
                    if len(global_t.required_nodes) > len(t.required_nodes):
                        # but also required by another node out the subgraph
                        log.debug(" --- required by another node out the subgraph, this is uncommon")     # TODO: for debug, remove this
                        log.debug("global_t: %s", global_t)
                        log.debug("t: %s", t)
                        # log.debug("global_t.required_nodes: %s", global_t.required_nodes)
                        # log.debug("t.required_nodes: %s", t.required_nodes)
                        # so it still is a output
                        tensor_list.append(t)
        return tensor_list

    def _get_io_tensor_origin_op(self) -> Dict[str, List[Tuple[_BaseOp, int]]]:
        origin_op_pos: Dict[str, List[Tuple[_BaseOp, int]]] = defaultdict(list)
        for t in self.inputs:
            for origin_input_op in t.required_nodes:
                for i, origin_t in enumerate(origin_input_op.inputs):
                    if origin_t.name == t.name:
                        origin_op_pos[t.name].append((origin_input_op, i))
                        break
                else:
                    assert False

        for t in self.outputs:
            origin_output_op = t.produced_node
            for i, origin_t in enumerate(origin_output_op.outputs):
                if origin_t.name == t.name:
                    origin_op_pos[t.name].append((origin_output_op, i))
                    break
            else:
                assert False
        return dict(origin_op_pos)

    def get_input_size(self, input_idx: int) -> int:
        # print(f"fused get_input_size: {self}:{input_idx}")
        if all(isinstance(x, _DummyOp) for x in self._fused_ops):
            # print("all dummy, return 0")
            return 0
        # print(*(
        #     f"    I {(origin_op, pos, origin_op.get_input_size(pos))}"
        #     for origin_op, pos in self._io_tensor_origin_op[self.inputs[input_idx].name]
        # ), sep='\n')
        return max(
            origin_op.get_input_size(pos)
            for origin_op, pos in self._io_tensor_origin_op[self.inputs[input_idx].name]
        )

    def get_output_size(self, output_idx: int) -> int:
        # print(f"fused get_output_size: {self}:{output_idx}")
        if all(isinstance(x, _DummyOp) for x in self._fused_ops):
            # print("all dummy, return 0")
            return 0
        # print(*(
        #     f"    O {(origin_op, pos, origin_op.get_output_size(pos))}"
        #     for origin_op, pos in self._io_tensor_origin_op[self.outputs[output_idx].name]
        # ), sep='\n')
        return max(
            origin_op.get_output_size(pos)
            for origin_op, pos in self._io_tensor_origin_op[self.outputs[output_idx].name]
        )

    def get_flops(self) -> int:
        return sum(op.get_flops() for op in self._fused_ops)


class FusedAnalyze(Analyze):
    def __init__(self, analyze: Analyze) -> None:
        self._analyze = analyze
        self.model = analyze.model
        self.data = analyze.data
        self.ops = deepcopy(analyze.ops)
        # self.tensors = deepcopy(analyze.tensors)
        self.tensors: Dict[str, TensorInfo] = self._get_tensors_info()  # re-create them to pointing to new copys in self.ops
        # self.tensor_alias: Dict[str, TensorInfo] = {}

    def set_tensor_alias(self, new_name: str, origin: str):
        if origin not in self.tensors:
            raise ValueError("origin tensor name '%s' not found" % origin)
        if (new_name in self.tensors
            and self.tensors[new_name] is not self.tensors[origin]):
            raise ValueError("alias tensor name '%s' already exist and is not the same tensor" % new_name)
        self.tensors[new_name] = self.tensors[origin]

    def _get_minimal_subgraph_from_op_list(self, op_list: List[_BaseOp]):
        # print = lambda *x: None # FIXME
        # print('_get_minimal_subgraph_from_op_list start', op_list)
        # assert all(x in self.ops.values() for x in op_list)   # FIXME
        _td = lambda t: self.tensors[t.name]
        op_set = set(x.name for x in op_list)

        dead_paths = set()
        def dfs_down(op: _BaseOp, path: List[_BaseOp]) -> bool:
            # print("visit", op, path if len(path) < 10 else len(path))
            found = False
            if op.name not in dead_paths:
                for op_out in op.outputs:
                    for n in _td(op_out).required_nodes:
                        # print("  next:", n)
                        # assert n.name in self.ops, f"{n.name}"   # FIXME
                        # assert n in self.ops.values(), f"{n}"   # FIXME
                        if n.name in op_set:
                            # print(" ", path)
                            found = True
                            for path_node in path:
                                log.debug("_get_minimal_subgraph_from_op_list: add node %s", path_node)
                                assert not isinstance(path_node, _FusedOp)   # FIXME
                                op_set.add(path_node.name)
                        else:
                            if not isinstance(n, _FusedOp):     # TODO: workaround
                                dead_path = not dfs_down(n, path + [n])
                                if dead_path:
                                    # print("  dead_path", n)
                                    dead_paths.add(n.name)
                                else:
                                    found = True
                            # else:
                                # log.info(f"_get_minimal_subgraph_from_op_list meet _FusedOp, next: {n}, op_list: {op_list}")
                                # assert False, f"next: {n}, op_list: {op_list}"
            return found

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10**6)

        last_iter_added = -1
        while last_iter_added:
            dead_paths.clear()
            old_len = len(op_set)
            for op_name in tuple(op_set):
                dfs_down(self.ops[op_name], [])
            last_iter_added = len(op_set) - old_len

        sys.setrecursionlimit(old_limit)

        return list(self.ops[x] for x in op_set)

    def set_fused_op(self, new_name: str, op_list: Union[List[str], List[_FusedOp]], _do_one_subgraph_fix=False, _add_previous_not_matched_node=False) -> _FusedOp:
        assert len(op_list) >= 1
        if type(op_list[0]) is str:
            op_list = [self.ops[name] for name in op_list]
        # assert all(x.name in self.ops for x in op_list), op_list
        op_list = [x for x in op_list if x.name in self.ops]    # FIXME FIXME: TMP

        if _do_one_subgraph_fix:
            # NOTE: may wrong if a backend REALLY (most unlikely) fused two subgraph into one Op
            op_list = self._get_minimal_subgraph_from_op_list(op_list)
            # assert all(x.name in self.ops for x in op_list), f"{op_list}" # FIXME
            # assert all(x in self.ops.values() for x in op_list), f"{op_list}" # FIXME

        if _add_previous_not_matched_node:
            # NOTE:
            op_list += self.get_previous_not_fused_op_by_list(op_list)

        for old_op in op_list:
            del self.ops[old_op.name]

        new_op = _FusedOp(new_name, op_list, self)
        self.ops[new_name] = new_op

        # remove old Ops in required_nodes and add the new one
        for t in new_op.inputs:
            t = self.tensors[t.name]
            t.required_nodes = [n for n in t.required_nodes if n not in op_list]
            t.required_nodes.append(new_op)

        # replace old Op on produced_node with new one
        for t in new_op.outputs:
            t = self.tensors[t.name]
            t.required_nodes = [n for n in t.required_nodes if n not in op_list]    # may both required by outer Op and inner Op
            t.produced_node = new_op

        return new_op

    def get_subgraph_ops_by_io(self, inputs: List[str], outputs: List[str]) -> List[_BaseOp]:
        inputs = [self.tensors[x].name if x in self.tensors else x for x in inputs]
        outputs = [self.tensors[x].name if x in self.tensors else x for x in outputs]
        return super().get_subgraph_ops_by_io(inputs, outputs)

    def get_origin_ops(self, fused_name: str) -> List[_BaseOp]:
        op = self.ops[fused_name]
        if isinstance(op, _FusedOp):
            return op._fused_ops
        return [op]

    def get_previous_not_fused_op(self, current_op: _BaseOp) -> List[_BaseOp]:
        assert current_op.name in self.ops

        not_fused_op = set()
        def dfs_up(op: _BaseOp):
            for t in op.inputs:
                n = self.tensors[t.name].produced_node
                if n and not isinstance(n, _FusedOp):
                    not_fused_op.add(n)
                    dfs_up(n)

        dfs_up(current_op)
        # assert all(x in self.ops.values() for x in not_fused_op)  # FIXME: remove this
        return list(not_fused_op)


    def get_previous_not_fused_op_by_list(self, op_list: List[_BaseOp]) -> List[_BaseOp]:
        # print("get_previous_not_fused_op_by_list", op_list)
        assert all(x.name in self.ops for x in op_list)
        op_set = set(op_list)

        not_fused_op = set()
        def dfs_up(op: _BaseOp):
            for t in op.inputs:
                n = self.tensors[t.name].produced_node
                if n and not n in op_set and not isinstance(n, _FusedOp):
                    not_fused_op.add(n)
                    dfs_up(n)

        for op in op_set:
            dfs_up(op)
        # assert all(x in self.ops.values() for x in not_fused_op)  # FIXME: remove this
        results = list(not_fused_op - op_set)
        # print("get_previous_not_fused_op_by_list got:", results)
        return results


@DeprecationWarning
def batched_fused_op_get_memory(op_list: List[_BaseOp], batch_size: int = 1) -> int:
    # TODO: deprecated function, use _FusedOp and _FusedOp.get_memory() instead
    # FIXME: a rare case: in fused op subgraph, if a node output tensor is used by
    # both a node in subgraph and a node NOT in subgraph, it will not been added, but has actualy DRAM writeback
    tensor_sizes = defaultdict(int)
    tensor_used_input = defaultdict(int)
    tensor_used_output = defaultdict(int)
    params = set()
    for op in op_list:
        for i, s in enumerate(op.inputs):
            tensor_sizes[s.name] = max(tensor_sizes[s.name], op.get_input_size(i))
            tensor_used_input[s.name] += 1
        for i, s in enumerate(op.outputs):
            tensor_sizes[s.name] = max(tensor_sizes[s.name], op.get_output_size(i))
            tensor_used_output[s.name] += 1
        params.update(op.params)

    memory = 0
    tensor_to_be_add = set(itertools.chain(tensor_used_input.keys(), tensor_used_output.keys()))
    for op in op_list:
        for s in itertools.chain(op.inputs, op.outputs):
            if s.name in tensor_to_be_add:
                # don't added same tensor more than once
                tensor_to_be_add.remove(s.name)

                if s.name in params:
                    # is parameter
                    memory += tensor_sizes[s.name]
                else:
                    if tensor_used_input[s.name] >= 1 and tensor_used_output[s.name] >= 1:
                        # is inner tensor
                        pass  # we assume it's passed in SRAM
                    else:
                        # is edge tensor
                        memory += tensor_sizes[s.name] * batch_size

    return memory


def get_effort_fused_model(origin_analyze: Analyze) -> FusedAnalyze:
    fused_analyze = FusedAnalyze(origin_analyze)
    input_ops = []
    for n in fused_analyze.data.inputs.keys():
        input_ops += fused_analyze.tensors[n].required_nodes

    visited = set()

    def visit(op: _BaseOp) -> None:
        if op.name in visited:
            # print("! already visted:", op)
            return
        visited.add(op.name)
        # print('\n + visit', op)
        fused = [op] + fused_analyze.get_op_may_fused_with(op)
        if len(fused) > 1:
            # print("=== get_op_may_fused_with:", *fused, "fused at", op, sep='\n    ')
            fused_name = op.name + ('_%d_fused' % len(fused))
            fused_analyze.set_fused_op(fused_name, fused)
            op = fused_analyze.ops[fused_name]
            visited.add(op.name)
            # print("+fused_op:", op, op._fused_ops)
            # print("+fused_op inputs:", fused_analyze.op_inputs(op))
            # print("+fused_op outputs:", fused_analyze.op_outputs(op))

        next_ops = fused_analyze.get_next_op(op)
        # print(' + next', *next_ops, sep='\n    ')
        for next_op in next_ops:
            if next_op.name in fused_analyze.ops:    # Op still in graph (or already removed by another op fuse)
                visit(next_op)

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10**6)

    for op in input_ops:
        visit(op)

    sys.setrecursionlimit(old_limit)

    # print(len(origin_analyze.ops))
    # print(sum(len(x._fused_ops) if isinstance(x, _FusedOp) else 1 for x in fused_analyze.ops.values()))
    return fused_analyze
