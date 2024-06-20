from typing import List, Union
import os
import sys
import subprocess
import logging
import json

import numpy as np
import onnx

from util import TMPDIR
from context import ModelContext
from datatype import ModelBenchBatchLayerData
from model.analyze.op import _BaseOp
from model.analyze.fuse import FusedAnalyze, batched_fused_op_get_memory, _FusedOp
from model.analyze.shape import change_inputs_batch_size
from .. import _BaseBackend
from ..cache import InterOpCacheSimulator

log = logging.getLogger(__name__)


class Trtexec(_BaseBackend):
    supported = ['e2e_prof', 'layer_prof']

    def __init__(self, ctx: ModelContext, onnx_model: str, batch_size_list: list, backend_options: str) -> None:
        self.ctx = ctx
        self.collected_data = ctx.collected_data
        self.onnx_model = onnx_model
        self.batch_size_list = batch_size_list

        self.layer_prof_ncu_mode = False
        self.ncu_bin = 'ncu'
        self.layer_prof_use_nsys_for_myelin = False
        self.nsys_bin = 'nsys'
        self.trtexec_convert_fp16 = False
        self.trtexec_convert_int8 = False
        self.trtexec_bin = '/usr/src/tensorrt/bin/trtexec'
        self.trtexec_convert_arg = None
        self.trtexec_e2e_run_arg = None
        self.trtexec_layer_prof_run_arg = None
        for option in backend_options.split(','):
            if option:
                key, *value = option.strip().split('=', maxsplit=1)     # value is optional
                if key == 'use_ncu':
                    self.layer_prof_ncu_mode = True
                elif key == 'ncu_bin':
                    self.ncu_bin = value[0]
                elif key == 'use_nsys_for_myelin':
                    self.layer_prof_use_nsys_for_myelin = True
                elif key == 'nsys_bin':
                    self.nsys_bin = value[0]
                elif key == 'fp16':
                    self.trtexec_convert_fp16 = True
                elif key == 'int8':
                    self.trtexec_convert_int8 = True
                elif key == 'trtexec_bin':
                    self.trtexec_bin = value[0]
                elif key == 'trtexec_convert_arg':
                    self.trtexec_convert_arg = value[0]
                elif key == 'trtexec_e2e_run_arg':
                    self.trtexec_e2e_run_arg = value[0]
                elif key == 'trtexec_layer_prof_run_arg':
                    self.trtexec_layer_prof_run_arg = value[0]
                elif key == 'help':
                    print("backend %s options help:" % self.__class__)
                    print("    use_ncu:                            use dlprof + ncu(Nsight Compute) stack in layer_porf to get real tested layer performance "
                                                                  "by doing kernel profiling, instead of approximately value by model analyze, takes more time, default: no")
                    print("    ncu_bin=PATH:                       path to ncu binary (PATH), only necessary when enable use_ncu and ncu is not in PATH, default: ncu")
                    print("    use_nsys_for_myelin:                use nsys(Nsight Systems) when seen TensorRT Myelin layer for better profiling, default: no")
                    print("    nsys_bin=PATH:                      path to nsys binary (PATH), only necessary when enable use_nsys_for_myelin and nsys is not in PATH, default: nsys")
                    print("    int8:                               add --int8 flag to trtexec during convert (build)")
                    print("    fp16:                               add --fp16 flag to trtexec during convert (build)")
                    print("    trtexec_bin=PATH:                   path to trtexec binary (PATH), default: /usr/src/tensorrt/bin/trtexec")
                    print("    trtexec_convert_arg=ARGS:           addition arguments (ARGS) passed to trtexec during convert (build)")
                    print("    trtexec_e2e_run_arg=ARGS:           addition arguments (ARGS) passed to trtexec during end-to-end timing model inference")
                    print("    trtexec_layer_prof_run_arg=ARGS:    addition arguments (ARGS) passed to trtexec during layer profile model runs inference")
                    print("    help:                               print this help and exit")
                    sys.exit(0)
                else:
                    raise RuntimeError("Unknown backend_options: %s" % key)

        if self.layer_prof_ncu_mode and self.layer_prof_use_nsys_for_myelin:
            raise RuntimeError("when using ncu mode (use_ncu), use_nsys_for_myelin is not supported yet")

    def version_info(self) -> str:
        info = f"Using backend {self.__class__}\n"

        trtexec_check = [self.trtexec_bin, '--verbose', '--loadEngine=dummy']   # TODO: maybe better way?
        try:
            trtexec_check_run = subprocess.run(trtexec_check,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            import re
            trt_version_match = re.search(r"TensorRT version: (.*)\n", trtexec_check_run.stdout.decode())
            trt_version = trt_version_match.group(1) if trt_version_match else '<failed to get>'
            info += f"tensorrt (trtexec) version: {trt_version}\n"

        except FileNotFoundError:
            log.error("trtexec binary path %s not exist, you may set it with backend options 'trtexec_bin'", self.trtexec_bin)
            log.error("    e.g. -o trtexec_bin=/path/to/trtexec'")
            sys.exit(1)

        return info[:-1]

    def prepare(self) -> None:
        pass

    ### e2e_prof ###

    def pre_batch_run(self, batch_size: int) -> None:
        inputs = self.ctx.analyze.data.inputs
        onnx_model = self.onnx_model
        if any(type(x[0]) is int for x in inputs.values()):
            log.warning("model has fixed batch_size, will make a dynamic one")
            m = onnx.load(onnx_model)
            change_inputs_batch_size(m, 'batch_size')
            onnx_model = str(TMPDIR / "dyn.onnx")
            onnx.save(m, onnx_model)

        input_shapes_batched = []
        for input, shape in inputs.items():
            shape = shape[:]
            shape[0] = batch_size
            input_shapes_batched.append(input + ':' + 'x'.join(map(str, shape)))
        input_shapes_batched_arg = ','.join(input_shapes_batched)

        self.target_model = str(TMPDIR / "model.trt")
        trtexec_convert_cmd = [self.trtexec_bin,
            "--onnx="+onnx_model,
            "--saveEngine="+self.target_model,
            "--minShapes="+input_shapes_batched_arg,
            "--optShapes="+input_shapes_batched_arg,
            "--maxShapes="+input_shapes_batched_arg,
            "--buildOnly"]

        if not os.getenv('PROOF_BACKEND_TRTEXEC_NO_PROFILING_VERBOSITY'):
            # to get more info from '--exportLayerInfo', but not supported in old version of tensorrt (e.g. 8.0.3)
            trtexec_convert_cmd.append("--profilingVerbosity=detailed")

        if self.trtexec_convert_fp16:
            trtexec_convert_cmd.append('--fp16')

        if self.trtexec_convert_int8:
            trtexec_convert_cmd.append('--int8')

        if self.trtexec_convert_arg:
            trtexec_convert_cmd += self.trtexec_convert_arg.split()

        log.debug("running trtexec (convert) ...")
        log.debug(' '.join(trtexec_convert_cmd))

        # CONVERT #
        if not os.getenv('PROOF_BACKEND_TRTEXEC_SKIP_CONVERT'):     # TODO: debug flag, remove in release
            trtexec_convert_run = subprocess.run(trtexec_convert_cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if trtexec_convert_run.returncode != 0:
                log.error("model convert failed")
                if "Unknown option: --profilingVerbosity" in trtexec_convert_run.stderr.decode():
                    log.error("if you are using old version of TensorRT, try to set 'PROOF_BACKEND_TRTEXEC_NO_PROFILING_VERBOSITY'")
                log.error("command: %s", ' '.join(trtexec_convert_run.args))
                log.error("output:\n%s\n%s", trtexec_convert_run.stdout.decode(), trtexec_convert_run.stderr.decode())
                raise RuntimeError

        log.debug("pre_batch_run() end")

    def e2e_prof(self, batch_size: int, repeat: int = 10, warm_up: int = 3) -> np.ndarray:
        """note: 'repeat' will set to trtexec's minimum iterations (--iterations), 'warm_up' will be ignore (will be fixed 1000ms warmup in trtexec)"""
        tmp_time_file = str(TMPDIR / "trtexec_times.txt")
        trtexec_inference_cmd = [self.trtexec_bin,
            "--warmUp=1000",
            "--loadEngine="+self.target_model,
            "--iterations="+str(repeat),
            "--exportTimes="+tmp_time_file]

        if self.trtexec_e2e_run_arg:
            trtexec_inference_cmd += self.trtexec_e2e_run_arg.split()

        log.debug("running trtexec (inference) ...")
        log.debug(' '.join(trtexec_inference_cmd))

        trtexec_inference_run = subprocess.run(trtexec_inference_cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if trtexec_inference_run.returncode != 0:
            log.error("model inference failed")
            log.error("command: %s", ' '.join(trtexec_inference_run.args))
            log.error("output:\n[stdout]\n%s\n[stderr]\n%s", trtexec_inference_run.stdout.decode(), trtexec_inference_run.stderr.decode())
            raise RuntimeError

        log.debug("running done")

        with open(tmp_time_file, 'r') as f:
            results = json.load(f)
            times_ms = [item['computeMs'] for item in results][-repeat:]

        return np.array(times_ms) / 1000

    ### layer_prof ###

    def _trt_layer_filter(self, layer: ModelBenchBatchLayerData) -> bool:
        return not (
            # layer.name.startswith("Reformatting CopyNode for") or   # TODO: remove all "Reformatting CopyNode for xxx" is not accurate
            layer.median_time == 0)

    def _trt_layer_name_tokenizer(self, name: str) -> list:
        if not hasattr(self, '_trt_layer_name_tokenizer_re'):
            import re
            # self._trt_layer_name_tokenizer_re = re.compile(r'[0-9A-Za-z_]+')
            self._trt_layer_name_tokenizer_re = re.compile(r'[0-9A-Za-z_/.]+')
        return self._trt_layer_name_tokenizer_re.findall(name)

    def _trt_layer_total_flops(self, onnx_nodes: list, batch_size: int) -> int:
        return sum(
            self.ctx.analyze.data.nodes[x].flops
            for x in onnx_nodes) * batch_size

    def _trt_layer_total_memory(self, onnx_nodes: list, batch_size: int) -> int:
        # TODO: not accurate, lower boundary
        if not onnx_nodes:
            return 0
        op_list = [self.ctx.analyze.ops[x] for x in onnx_nodes]
        memory = batched_fused_op_get_memory(op_list, batch_size)
        data_width = self.ctx.data_width_backend
        return memory * data_width / 8

    def layer_prof(self, batch_size: int) -> List[ModelBenchBatchLayerData]:
        if self.layer_prof_ncu_mode:
            return self.layer_prof_ncu(batch_size)

        tmp_profile_file = str(TMPDIR / "trtexec_profile.json")
        tmp_layer_info_file = str(TMPDIR / "trtexec_layer_info.json")
        trtexec_inference_cmd = [self.trtexec_bin,
            "--warmUp=1000",
            "--loadEngine="+self.target_model,
            "--exportProfile="+tmp_profile_file,
            "--exportLayerInfo="+tmp_layer_info_file]

        if self.trtexec_layer_prof_run_arg:
            trtexec_inference_cmd += self.trtexec_layer_prof_run_arg.split()

        log.debug("running trtexec (inference) ...")
        log.debug(' '.join(trtexec_inference_cmd))

        trtexec_inference_run = subprocess.run(trtexec_inference_cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if trtexec_inference_run.returncode != 0:
            log.error("model inference failed")
            log.error("command: %s", ' '.join(trtexec_inference_run.args))
            log.error("output:\n[stdout]\n%s\n[stderr]\n%s", trtexec_inference_run.stdout.decode(), trtexec_inference_run.stderr.decode())
            raise RuntimeError

        log.debug("running done")

        if not os.getenv('PROOF_BACKEND_TRTEXEC_NO_PROFILING_VERBOSITY'):
            # new method using analyze FusedAnalyze object
            with open(tmp_profile_file, 'r') as f:
                results = json.load(f)

            with open(tmp_layer_info_file, 'r') as f:
                layer_info: list[dict] = json.load(f)['Layers']

            if results[0].get('name') is None:
                results = results[1:]      # remove "count"

            layers_profile = [
                ModelBenchBatchLayerData(
                    name = x['name'],
                    median_time = (x['medianMs'] if 'medianMs' in x else x['averageMs']) / 1000,
                    time_percentage = None,    # medianMs / sum(medianMs) * 100%
                    flops = 0,      # FLOPs, (float) operation count
                    memory = 0,     # memory access count
                    extra = {"onnx_nodes": []},
                ) for x in results]

            fused_analyze = FusedAnalyze(self.ctx.analyze)
            fused_analyze.purge_shape_op()  # TODO: more testing

            # pass: check tensor as output in more than one layer (e.g. Concat)
            from collections import defaultdict
            produced_layer = defaultdict(list)
            for layer in layer_info:
                for output in layer["Outputs"]:
                    produced_layer[output['Name']].append(layer['Name'])

            layer_info_dict = {layer['Name']:layer for layer in layer_info}
            for output, layers in produced_layer.items():
                if len(layers) > 1:
                    log.debug("output '%s' is produced by multi layer: %s", output, layers)
                    if output in fused_analyze.tensors:
                        t = fused_analyze.tensors[output]
                        log.debug("origin produced node: '%s", t.produced_node)
                        # TODO: another method (just by layer name) exist
                        layers_set = set(layers)
                        for upstream_t in t.produced_node.inputs:
                            upstream_t = fused_analyze.tensors[upstream_t.name]
                            upstream_node = upstream_t.produced_node
                            log.debug("upstream_node: %s", upstream_node)
                            for layer in layers_set.copy():
                                log.debug("process layer %s", layer)
                                if upstream_node.name in layer:
                                    log.debug("%s match %s", upstream_node.name, layer)
                                    for o in layer_info_dict[layer]['Outputs']:
                                        if o['Name'] == output:
                                            if len(upstream_node.outputs) > 1:
                                                log.warning("len(%s.outputs) > 1", upstream_node.name)
                                            log.debug("set layer '%s' output '%s' to '%s'", layer, o['Name'], upstream_node.outputs[0].name)
                                            o['Name'] = upstream_node.outputs[0].name
                                            layers_set.remove(layer)
                        if len(layers_set) > 0:
                            log.warning("some layer still not corrected: %s", list(layers_set))
                    else:
                        log.warning("can't correct it")

            # pass: set tensor alias
            pending_layers = {layer['Name']:layer for layer in layer_info}
            dummy_layers = set()

            has_alias_set = True
            while pending_layers:
                if not has_alias_set:
                    log.warning("flowing layers may should to set a tensor alias, but failed %s", list(pending_layers.values()))
                    break

                has_alias_set = False
                for name, layer in pending_layers.copy().items():
                    if (name.startswith('Reformatting CopyNode for Output Tensor ')
                        or name.startswith('Reformatting CopyNode for Input Tensor ')
                        or name.startswith('reshape_before_')
                        or name.startswith('reshape_after_')
                        or name.endswith(' copy')
                        or layer['LayerType'] == 'Reformat' or layer['LayerType'] == 'NoOp'):    # FIXME: is this correct

                        # may be an alias node
                        if len(layer['Inputs']) != 1 or len(layer['Outputs']) != 1:
                            log.warning("alias: look like alias layer, but len(layer['Inputs']) != 1 or len(layer['Outputs']) != 1")
                            log.warning(layer)

                        if layer['Inputs'][0]['Name'] in fused_analyze.tensors and layer['Outputs'][0]['Name'] in fused_analyze.tensors:
                            # not a dummy (alias) node
                            log.debug("all tensor exist, may not a dummy (alias) node, skip: %s (%s -> %s)", \
                                      name, layer['Inputs'][0]['Name'], layer['Outputs'][0]['Name'])
                            del pending_layers[name]
                            continue

                        if layer['Outputs'][0]['Name'] in fused_analyze.tensors:
                            fused_analyze.set_tensor_alias(layer['Inputs'][0]['Name'], layer['Outputs'][0]['Name'])
                            log.debug("set alias: '%s' (new name) <- '%s'", layer['Inputs'][0]['Name'], layer['Outputs'][0]['Name'])
                            has_alias_set = True
                            dummy_layers.add(name)
                            del pending_layers[name]

                        elif layer['Inputs'][0]['Name'] in fused_analyze.tensors:
                            fused_analyze.set_tensor_alias(layer['Outputs'][0]['Name'], layer['Inputs'][0]['Name'])
                            log.debug("set alias: '%s' (new name) <- '%s'", layer['Outputs'][0]['Name'], layer['Inputs'][0]['Name'])
                            has_alias_set = True
                            dummy_layers.add(name)
                            del pending_layers[name]

                        else:
                            log.debug("both I/O tensor name not exist yet, skip: '%s' and '%s'", \
                                      layer['Inputs'][0]['Name'], layer['Outputs'][0]['Name'])


                    # elif layer['LayerType'] == 'Reformat' or layer['LayerType'] == 'NoOp':
                    #     dummy_layers.add(name)
                    #     log.warning("LayerType is '%s' but not supported", layer['LayerType'])
                    #     log.warning(f"input exist: {layer['Inputs'][0]['Name'] in fused_analyze.tensors}")
                    #     log.warning(f"output exist: {layer['Outputs'][0]['Name'] in fused_analyze.tensors}")
                    #     log.warning(layer)
                    #     del pending_layers[name]

                    else:
                        del pending_layers[name]

            # pass: filter layers in layers_profile
            for layer in layers_profile:
                if layer.median_time == 0:
                    log.debug("will remove zero time layer '%s'", layer.name)
            layer_times = {layer.name: layer.median_time for layer in layers_profile}
            layer_info = [layer for layer in layer_info if layer_times[layer['Name']] != 0]
            layers_profile = [layer for layer in layers_profile if layer.median_time != 0]

            nsys_report = None

            previous_not_matched_layer = None
            def __match_previous_not_matched_layer(previous: Union[ModelBenchBatchLayerData, None], current_op: _BaseOp):
                "retrun `None` if success, else retrun `previous` again for later try"
                if previous:
                    log.debug("__match_previous_not_matched_layer %s", previous.name)
                    op_list = fused_analyze.get_previous_not_fused_op(current_op)
                    # print("get_previous_not_fused_op", op_list)
                    if op_list:
                        fused_op = fused_analyze.set_fused_op(previous.name, op_list, _do_one_subgraph_fix=True)
                        log.debug("    fused %s", fused_op)
                    else:
                        log.debug("    no un-fused op found by data dependency")
                        return previous
                # else:
                #     # FIXME: clean up un matched
                #     op_list = fused_analyze.get_previous_not_fused_op(current_op)
                #     if op_list:
                #         log.debug("__match_previous_not_matched_layer CLEAN_UP")
                #         print("get_previous_not_fused_op (CLEAN_UP)", op_list)
                #         fused_op = fused_analyze.set_fused_op(current_op.name + "[CLEAN_UP_PRE]", op_list, _do_one_subgraph_fix=False)
                #         log.debug("    CLEAN_UP_PRE fused dummy %s", fused_op)

            # pass: set fused op
            myelin_idx = 0
            for layer in layer_info:
                if layer['Name'] not in dummy_layers:  # not alias layer in previous pass
                    name = layer['Name']
                    inputs = [t['Name'] for t in layer['Inputs']]
                    outputs = [t['Name'] for t in layer['Outputs']]

                    if layer['LayerType'] == 'Myelin':
                        log.debug("Myelin layer %s: %s", myelin_idx, name)
                        name_prefix = '(Myelin %d) ' % myelin_idx
                        myelin_idx += 1

                        # Myelin layer process via Nsight System
                        if self.layer_prof_use_nsys_for_myelin:
                            from .nsys_myelin import nsys_run, nsys_myelin_layer_get_names
                            if not nsys_report:
                                nsys_report = nsys_run(self)
                            sub_names = nsys_myelin_layer_get_names(nsys_report, name)

                            for i, layer in enumerate(layers_profile):
                                if layer.name == name:
                                    insert_index = i
                                    break

                            del layers_profile[insert_index]
                            total_layer_time = 0
                            not_matched_time = 0
                            for sub_layer, time, kernel_name in sub_names:
                                total_layer_time += time
                                sub_layer_origin = sub_layer
                                log.debug('\n (' + sub_layer + ')')
                                # TODO: use .endswith() to match these postfix is better
                                sub_layer = sub_layer.replace('_matrix_multiply', '')
                                sub_layer = sub_layer.replace('_reshape', '')
                                sub_layer = sub_layer.replace('_first_transpose', '')
                                sub_layer = sub_layer.replace('_slice', '')
                                sub_layer = sub_layer.replace('_decomptile', '')

                                sub_layer = sub_layer.replace(' _ ', '+')
                                sub_layers = sub_layer.split('+')
                                log.debug(sub_layers)

                                converted_name_op_dict = {k.replace('.', '_'): v for k, v in fused_analyze.ops.items()}
                                op_name_set = set()
                                for sub_name in sub_layers:
                                    if sub_name in converted_name_op_dict:
                                        op_name_set.add(converted_name_op_dict[sub_name].name)
                                log.debug(op_name_set)
                                if op_name_set:
                                    op_name_list = list(op_name_set)
                                    fused_op_name = name_prefix + sub_layer_origin
                                    fused_op = fused_analyze.set_fused_op(
                                        fused_op_name,
                                        op_name_list,
                                        _do_one_subgraph_fix=True,
                                        _add_previous_not_matched_node=not previous_not_matched_layer
                                    )
                                    layers_profile.insert(insert_index, ModelBenchBatchLayerData(
                                        name = fused_op_name,
                                        median_time = time,
                                        flops = 0,
                                        memory = 0,
                                        extra = {"onnx_nodes": [], 'myelin': True, 'kernel_name': kernel_name},
                                    ))
                                    insert_index += 1

                                    previous_not_matched_layer = __match_previous_not_matched_layer(previous_not_matched_layer, fused_op)
                                else:
                                    if previous_not_matched_layer:
                                        # will be overwrite
                                        not_matched_time += previous_not_matched_layer.median_time
                                    previous_not_matched_layer = ModelBenchBatchLayerData(
                                        name = name_prefix + sub_layer_origin,
                                        median_time = time,
                                        flops = 0,
                                        memory = 0,
                                        extra = {"onnx_nodes": [], 'myelin': True},
                                    )
                                    layers_profile.insert(insert_index, previous_not_matched_layer)
                                    insert_index += 1
                                    log.info("pending not matched sub_name %s (%s ms)", sub_layer_origin, time * 1e3)
                                    # not_matched_time += time
                                    # log.info("skip not matched sub_name %s (%s ms)", sub_layer_origin, time * 1e3)

                            log.debug("unmatched layer total time (approximate) %s / %s ms", not_matched_time * 1e3, total_layer_time * 1e3)
                            continue
                        else:
                            log.warning("found Myelin layer: %s, consider to set use_nsys_for_myelin option, or the result may inaccurate", name)

                    try:
                        # print(name, inputs, outputs)
                        ops = fused_analyze.get_subgraph_ops_by_io(inputs, outputs)
                        # print("get_subgraph_ops_by_io", ops)
                        for op in ops:
                            if isinstance(op, _FusedOp):
                                log.warning("get_subgraph_ops_by_io got _FusedOp, will exclude it: %s", op)
                        ops = [op for op in ops if not isinstance(op, _FusedOp)]

                        log.debug("set fused op: '%s' <- %s", name, ops)

                        fused_op = fused_analyze.set_fused_op(name, ops, _do_one_subgraph_fix=True)
                        previous_not_matched_layer = __match_previous_not_matched_layer(previous_not_matched_layer, fused_op)

                    except Exception as e:
                        dummy_layers.add(name)
                        log.warning("set fused op failed for non-zero time layer, set to dummy: \
                                    '%s' ('%s' -> '%s'), error: %s %s", name, inputs, outputs, e.__class__.__name__, e)

            # pass: calculate and record to layers_profile
            cache = InterOpCacheSimulator(capacity=self.ctx.collected_data.model.llc_reuse_size * 1e6 / (self.ctx.data_width_backend / 8))
            data_width = self.ctx.data_width_backend
            for i, layer in enumerate(layers_profile):
                if layer.name not in dummy_layers:  # not alias layer in previous pass
                    try:
                        layer.extra['onnx_nodes'] = [op.name for op in fused_analyze.get_origin_ops(layer.name)]
                    except KeyError:
                        log.warning("layer '%s' not found, this should not happened", layer.name)
                        continue
                    op = fused_analyze.ops[layer.name]
                    layer.flops = op.get_flops() * batch_size

                    # layer.memory = op.get_memory(batch_size) * data_width / 8

                    # consider the inter-op cache
                    # prev_op = None
                    # prev_index = i - 1
                    # while prev_index >= 0:
                    #     name = layers_profile[prev_index].name
                    #     log.debug(f"visit prev {name}")
                    #     if name not in dummy_layers:
                    #         prev_op = fused_analyze.ops[name]
                    #         break
                    #     prev_index -= 1

                    # next_op = None
                    # next_index = i + 1
                    # while next_index < len(layers_profile):
                    #     name = layers_profile[next_index].name
                    #     log.debug(f"visit next {name}")
                    #     if name not in dummy_layers:
                    #         next_op = fused_analyze.ops[name]
                    #         break
                    #     next_index += 1
                    # memory, cached = fused_analyze.get_memory_with_cache_reduced(op, batch_size, int(0 * 1e6) // (data_width / 8), 'both', prev_op, next_op)

                    layer.extra['cached'], layer.extra['evicted'], layer.extra['write_back'] = 0, 0, 0
                    # _print = log.debug if cache.capacity else lambda *x: None
                    for i, t in enumerate(op.inputs):
                        # _print()
                        # _print(f"{cache.used}, {cache.cached.values()}")
                        factor = batch_size if t.name not in op.params else 1
                        memory, cached, evicted, write_back = cache.read(t.name, op.get_input_size(i) * factor)
                        # _print(f"read tensor {t}")
                        # _print(f"    size: {t.size()/1e6:.3f} M, memory: {memory/1e6:.3f} M, cached: {cached/1e6:.3f} M, evicted: {evicted/1e6:.3f} M")
                        layer.memory += memory * data_width / 8
                        layer.extra['cached'] += cached * data_width / 8
                        layer.extra['evicted'] += evicted * data_width / 8
                        layer.extra['write_back'] += write_back * data_width / 8
                    for i, t in enumerate(op.outputs):
                        # _print()
                        # _print(cache.used, cache.cached.values())
                        memory, evicted, write_back = cache.write(t.name, op.get_output_size(i) * batch_size)
                        # _print(f"write tensor {t}")
                        # _print(f"    size: {t.size()/1e6:.3f} M, memory: {memory/1e6:.3f} M, evicted: {cached/1e6:.3f} M, write_back: {evicted/1e6:.3f} M")
                        layer.memory += memory * data_width / 8
                        layer.extra['evicted'] += evicted * data_width / 8
                        layer.extra['write_back'] += write_back * data_width / 8

                    # TODO: debug
                    # if layer.name == 'Reformatting CopyNode for Output Tensor 0 to /stage2/stage2.3/Reshape_1':
                    #     log.error("%s", op)
                    #     log.error("%s", op.inputs)
                    #     log.error("%s", op.outputs)
                    #     log.error("%s", op.get_input_size(0))
                    #     log.error("%s", op.get_output_size(0))
                else:
                    layer.extra['dummy'] = True

            # pass: calculate time percentage
            time_sum = sum(layer.median_time for layer in layers_profile)
            for layer in layers_profile:
                layer.time_percentage = layer.median_time / time_sum * 100

            # debug: not mapped check
            total_flops, total_memory = 0, 0
            mapped_layers = set(n for l in layers_profile for n in l.extra['onnx_nodes'])
            for op in fused_analyze.ops.values():
                nodes = []
                if isinstance(op, _FusedOp):
                    nodes = op._fused_ops
                else:
                    nodes = [op]
                for n in nodes:
                    if n.name not in mapped_layers:
                        # print("not mapped layers:", n.name, n.inputs)
                        total_flops += batch_size * n.get_flops()
                        total_memory += n.get_memory(batch_size) * data_width / 8
            # print(f"total not mapped {total_flops/1e6:.3f} GFLOPs, {total_memory/1e6:.3f} MB")

            return layers_profile

        ### FALLBACK METHOD
        # old method based on layer name, if new method is disabled or failed, this will run
        with open(tmp_profile_file, 'r') as f:
            results = json.load(f)

        # TODO: just work, need more testing
        results = results[1:]      # remove "count"

        layers_profile = [
            ModelBenchBatchLayerData(
                name = x['name'],
                median_time = (x['medianMs'] if 'medianMs' in x else x['averageMs']) / 1000,
                time_percentage = None,    # medianMs / sum(medianMs) * 100%
                flops = None,      # FLOPs, (float) operation count
                memory = None,     # memory access count
                extra = {"onnx_nodes": []},
            ) for x in results]

        # pass: remove dummy layers
        layers_profile = [x for x in layers_profile if self._trt_layer_filter(x)]

        # pass: find the matched nodes in onnx model for each tensorrt layer
        onnx_nodes = self.ctx.analyze.data.nodes
        for layer in layers_profile:
            for token in self._trt_layer_name_tokenizer(layer.name):
                if token in onnx_nodes:
                    layer.extra['onnx_nodes'].append(token)

        # pass: calculate time percentage
        time_sum = sum(layer.median_time for layer in layers_profile)
        for layer in layers_profile:
            layer.time_percentage = layer.median_time / time_sum * 100

        # pass: calculate FLOPS and memory bandwidth
        for layer in layers_profile:
            layer.flops = self._trt_layer_total_flops(layer.extra['onnx_nodes'], batch_size)
            layer.memory = self._trt_layer_total_memory(layer.extra['onnx_nodes'], batch_size)

        return layers_profile

    def layer_prof_ncu(self, batch_size: int) -> List[ModelBenchBatchLayerData]:
        from .ncu import layer_prof_ncu
        return layer_prof_ncu(self, batch_size)
