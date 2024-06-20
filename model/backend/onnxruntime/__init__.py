from typing import List, Dict
from pathlib import Path
import sys
import logging
import json

import numpy as np

from context import PerfContext
from datatype import ModelBenchBatchLayerData
from util import TMPDIR
from model.analyze.fuse import FusedAnalyze, _FusedOp
from model.analyze.shape import change_inputs_batch_size, get_inputs_shape
from .. import _BaseBackend

import onnx
import onnxruntime

log = logging.getLogger(__name__)


class ONNXRuntimeBackend(_BaseBackend):
    supported = ['e2e_prof', 'layer_prof']

    def __init__(self, ctx: PerfContext, onnx_model: str, batch_size_list: list, backend_options: str) -> None:
        self.ctx = ctx
        self.collected_data = ctx.collected_data
        self.onnx_model = onnx_model
        self.batch_size_list = batch_size_list

        self.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.providers = None
        graph_optimization_level_all = ['ORT_DISABLE_ALL', 'ORT_ENABLE_BASIC', 'ORT_ENABLE_EXTENDED', 'ORT_ENABLE_ALL']
        for option in backend_options.split(','):
            if option:
                key, *value = option.strip().split('=')
                if key == 'graph_optimization_level':
                    try:
                        self.graph_optimization_level = getattr(onnxruntime.GraphOptimizationLevel, value[0])
                    except AttributeError:
                        log.error("Unknown graph_optimization_level '%s', which is not in %s" % (value[0], graph_optimization_level_all))
                        raise
                elif key == 'providers':
                    self.providers = value[0].replace(' ', '').split('+')
                elif key == 'help':
                    print("backend %s options help:" % self.__class__)
                    print("    graph_optimization_level:           set session_options.graph_optimization_level from %s (default)" % ', '.join(graph_optimization_level_all))
                    print("    providers:                          InferenceSession's providers option, format like: CUDAExecutionProvider+CPUxecutionProvider etc. (default: not set, means all available)")
                    print("    help:                               print this help and exit")
                    sys.exit(0)
                else:
                    raise RuntimeError("Unknown backend_options: %s" % key)

    def version_info(self) -> str:
        info = f"Using backend {self.__class__}\n"

        info += f"onnxruntime version: {onnxruntime.__version__}\n"
        info += f"available providers: {onnxruntime.get_available_providers()}\n"
        info += f"graph_optimization_level: {self.graph_optimization_level}\n"

        return info[:-1]

    def prepare(self) -> None:
        self.model = onnx.load(self.onnx_model)

    ### e2e_prof ###

    def pre_batch_run(self, batch_size: int) -> None:
        change_inputs_batch_size(self.model, batch_size)
        log.debug("pre_batch_run() end")

    def _gen_inputs(self):
        inputs_shape = get_inputs_shape(self.model)
        inputs = {}
        for t in self.model.graph.input:
            dtype = onnx.helper.tensor_dtype_to_np_dtype(t.type.tensor_type.elem_type)
            inputs[t.name] = np.random.randn(*inputs_shape[t.name]).astype(dtype)
        return inputs

    def e2e_prof(self, batch_size: int, repeat: int = 10, warm_up: int = 3) -> np.ndarray:
        """using InferenceSession.run"""

        sess_options = onnxruntime.SessionOptions()
        # sess_options.log_severity_level = 0
        # sess_options.log_verbosity_level = 0
        sess_options.enable_profiling = True
        sess_options.graph_optimization_level = self.graph_optimization_level
        sess_options.profile_file_prefix = str(TMPDIR / ('onnxruntime_profile_' + Path(self.onnx_model).stem))

        ort_sess = onnxruntime.InferenceSession(self.model.SerializeToString(), sess_options, providers=self.providers)

        log.debug("ort_sess.run ...")

        for _ in range(repeat + warm_up):
            x = self._gen_inputs()
            ort_sess.run(None, x)

        tmp_profile_file = ort_sess.end_profiling()

        times_ms = []
        with open(tmp_profile_file, 'r') as f:
            results = json.load(f)
            for res in results:
                if res['name'] == "model_run":
                    times_ms.append(res['dur'] / 1000)

        times_ms = times_ms[warm_up:]       # drop warm_up runs
        return np.array(times_ms) / 1000

    ### layer_prof ###

    def _try_set_alias(self, fused_analyze: FusedAnalyze, alias_name: str, origin_name: str):
        if origin_name not in fused_analyze.tensors:
            # origin_name not exist
            log.warning("origin_name '%s' not exist", origin_name)
        elif alias_name in fused_analyze.tensors:
            # alias_name already set
            if fused_analyze.tensors[alias_name] is not fused_analyze.tensors[origin_name]:
                log.warning("alias_name '%s' already exist, but origin_name is %s, not %s to set",
                    alias_name, fused_analyze.tensors[origin_name].name, origin_name)
        else:
            # ok
            fused_analyze.set_tensor_alias(alias_name, origin_name)

    def layer_prof(self, batch_size: int) -> List[ModelBenchBatchLayerData]:

        sess_options = onnxruntime.SessionOptions()
        # sess_options.log_severity_level = 0
        # sess_options.log_verbosity_level = 0
        sess_options.enable_profiling = True
        sess_options.graph_optimization_level = self.graph_optimization_level
        sess_options.profile_file_prefix = str(TMPDIR / ('onnxruntime_profile_' + Path(self.onnx_model).stem))
        sess_options.optimized_model_filepath = str(TMPDIR / ('onnxruntime_optimized_model_' + Path(self.onnx_model).name))

        ort_sess = onnxruntime.InferenceSession(self.model.SerializeToString(), sess_options, providers=self.providers)

        log.debug("ort_sess.run ...")

        repeat = 10
        warm_up = 3
        for _ in range(repeat + warm_up):
            x = self._gen_inputs()
            ort_sess.run(None, x)

        tmp_profile_file = ort_sess.end_profiling()

        log.debug("running done")

        with open(tmp_profile_file, 'r') as f:
            results_all = json.load(f)

        # pass: only keep *_kernel_time and remove the postfix
        results_all = [x for x in results_all if x['name'].endswith('_kernel_time')]
        for x in results_all:
            x['name'] = x['name'][:-len('_kernel_time')]

        # pass: combine all repeats' dur, and drop other repeats
        results: Dict[str, Dict] = {}
        for res in results_all:
            if res['name'] in results:
                if 'dur' in res:
                    results[res['name']]['all_dur'].append(res['dur'])
            else:
                results[res['name']] = res
                results[res['name']]['all_dur'] = [res['dur']]

        # pass: trim warm_up in all_dur
        for res in results.values():
            if 'all_dur' in res:
                res['all_dur'] = res['all_dur'][warm_up:]

        layers_profile = [
            ModelBenchBatchLayerData(
                name = x['name'],
                median_time = np.median(x['all_dur']) / 1e6,
                time_percentage = None,    # medianMs / sum(medianMs) * 100%
                flops = None,      # FLOPs, (float) operation count
                memory = None,     # memory access count
                extra = {"onnx_nodes": []},
            ) for x in results.values()]

        # pass: find the matched nodes in onnx model for optimized model nodes
        optimized_model = onnx.load(sess_options.optimized_model_filepath)
        fused_analyze = FusedAnalyze(self.ctx.analyze)
        if self.graph_optimization_level == onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL:
            # TODO: WIP, very unstable
            log.warning("support for onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL is unstable")
            log.warning("set '-o graph_optimization_level=ORT_ENABLE_EXTENDED' if got error")
            # on x86, almost all node and tensor will be replaced to nchwc format, guass tensor alias name by node name (<output_name>_nchwc), but not correct every time

        # print(*optimized_model.graph.node, sep='\n\n---\n')

        # pass: process tensor alias
        for node in optimized_model.graph.node:
            if node.op_type == 'ReorderInput':
                # print('match ReorderInput, set', node.output[0], 'to', node.input[0])
                self._try_set_alias(fused_analyze, node.output[0], node.input[0])
            if node.op_type == 'ReorderOutput':
                # print('match ReorderOutput, set', node.input[0], 'to', node.output[0])
                self._try_set_alias(fused_analyze, node.input[0], node.output[0])
            if node.name.endswith('_nchwc'):
                # TODO: only support node with single output
                origin_output = node.name[:-len('_nchwc')]
                # print("tensor alias", node.output[0], "=", origin_output)
                self._try_set_alias(fused_analyze, node.output[0], origin_output)

        # pass: fuse node
        for node in optimized_model.graph.node:
            # print("get op from", node.input, "to", node.output)

            # check
            for name in node.input:
                if name not in fused_analyze.tensors:
                    log.debug("input tensor '%s' not found", name)

            for name in node.output:
                if name not in fused_analyze.tensors:
                    log.debug("output tensor '%s' not found", name)
                    name_o0 = (name + '_output_0')
                    if name_o0 in fused_analyze.tensors:
                        log.debug("but tensor '%s' found, assume it's same", name_o0)
                        self._try_set_alias(fused_analyze, name, name_o0)

            try:
                fused_ops = fused_analyze.get_subgraph_ops_by_io(node.input, node.output)
            except Exception as e:
                log.warning("fused_analyze.get_subgraph_ops_by_io error: %s", e)
                continue

            # TODO: workaround, remove all already fused op, this is unnessery if tensor mapping is correct
            fused_ops = [n for n in fused_ops if not isinstance(n, _FusedOp)]
            # print("fused op", node.name, "=", fused_ops)
            if fused_ops:
                fused_analyze.set_fused_op(node.name, fused_ops)
            else:
                log.debug("got empty subgraph when processing node %s", node.name)

        origin_op_count = len(self.ctx.analyze.ops)
        mapped_op_count = sum(len(x._fused_ops) if isinstance(x, _FusedOp) else 1 for x in fused_analyze.ops.values())
        log.debug('origin_op_count: %s', origin_op_count)
        log.debug('mapped_op_count: %s', mapped_op_count)

        # pass: remove dummy layers
        def _filter(x: ModelBenchBatchLayerData) -> bool:
            if x.name not in fused_analyze.ops:
                log.debug("ignore layerdata %s (%s), may dummy", x.name, x.median_time)
                return False
            return True
        layers_profile = [x for x in layers_profile if _filter(x)]

        # pass: add onnx_nodes info for frontend to show
        for l in layers_profile:
            op = fused_analyze.ops[l.name]
            if isinstance(op, _FusedOp):
                l.extra['onnx_nodes'] += [n.name for n in fused_analyze.ops[l.name]._fused_ops]
            else:
                l.extra['onnx_nodes'].append(fused_analyze.ops[l.name].name)

        # pass: calculate time percentage
        time_sum = sum(layer.median_time for layer in layers_profile)
        log.debug("layer_prof time_sum is %s", time_sum)
        log.debug("may differ from e2e_prof, since layer_prof will use medium latency per layer")
        for layer in layers_profile:
            layer.time_percentage = layer.median_time / time_sum * 100

        # pass: calculate FLOPS and memory bandwidth
        for layer in layers_profile:
            layer.flops = fused_analyze.ops[layer.name].get_flops() * batch_size
            layer.memory = fused_analyze.ops[layer.name].get_memory(batch_size) * (self.ctx.data_width_backend / 8)

        # print(*layers_profile, sep='\n')
        return layers_profile

