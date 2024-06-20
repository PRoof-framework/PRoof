from __future__ import annotations

from typing import Tuple, List, Dict, Union
import os
import sys
from pathlib import Path
import logging
import numpy as np

from util import TMPDIR
from datatype import CollectedData, RooflineData, ModelData, ModelBenchData, ModelBenchBatchData
import model.analyze
from model.analyze.fuse import get_effort_fused_model
from model.backend import _BaseBackend

log = logging.getLogger(__name__)


# Used by main.py for CLI or as top level API of the package for other python program
class PerfContext():
    """Context of PRoof, """

    # defines all test subjects and their topology
    _ = None
    _all_subjects = {
        'roofline': _,
        'model': {
            'analyze': _,
            'bench': {
                'layer_prof': _,
                'e2e_prof': _
            }
        }
    }

    # test subjects and their depends
    _all_subjects_depends = {
        'model.bench': ['model.analyze'],
        'model.bench.layer_prof': ['model.bench.e2e_prof']
    }

    @classmethod
    def list_subjects(cls, _root: dict = _all_subjects) -> List[str]:
        l = []
        if _root:
            for k, v in _root.items():
                l.append(k)
                if type(v) is dict:
                    l += [k + '.' + x for x in cls.list_subjects(v)]
        return l

    def _process_subjects(self, subjects_list) -> None:
        "['a', 'b.1', 'b.2'] to {'a': ..., 'b': {'1': ..., '2': ...}}"

        self.subjects = set()
        subjects_list = subjects_list[:]

        for subject in subjects_list:
            all_pos = self._all_subjects
            subject_l = subject.split('.')
            for i in range(len(subject_l)):
                mid_name = subject_l[i]
                prefix = '.'.join(subject_l[:i+1])
                self.subjects.add(prefix)
                if mid_name not in all_pos:
                    log.error("no such subject '%s', available: %s", prefix, self.list_subjects())
                    sys.exit(1)

                all_pos = all_pos[mid_name]
                if prefix in self._all_subjects_depends:
                    subjects_list += self._all_subjects_depends[prefix]
            child_subjects = set(subject + '.' + x for x in self.list_subjects(_root=all_pos))
            self.subjects.update(child_subjects)

    def __init__(self,
            subjects: list,
            model_backend: _BaseBackend,

            # optional, keep the defaults same as in main.py
            onnx_model: str = '',
            batch_size_list: list = [1],
            repeat_count: int = 10,
            backend_options: str = '',
            data_width: Tuple[int, int] = (32, 32),
            *,

            # addition
            llc_reuse_size: float = .0,
            roofline_small: bool = False,
            inputs_shape_override: Dict[str, List[Union[int, None]]] = {}) -> None:

        self._process_subjects(subjects)
        self.collected_data = CollectedData()
        self.collected_data.subjects = list(self.subjects)

        self.model_backend = model_backend
        self.backend_options = backend_options
        self.data_width = data_width

        if 'roofline' in self.subjects:
            self.roofline_ctx = RooflineContext(self.subjects, self.collected_data, model_backend, backend_options, data_width, roofline_small)

        if 'model' in self.subjects:
            self.model_ctx = ModelContext(self.subjects, self.collected_data, model_backend, onnx_model, batch_size_list, repeat_count, backend_options, data_width, llc_reuse_size, inputs_shape_override)

    def run(self) -> None:
        if 'roofline' in self.subjects:
            self.roofline_ctx.run()

        if 'model' in self.subjects:
            self.model_ctx.run()


class RooflineContext():
    def __init__(self,
            subjects: list, collected_data: CollectedData, model_backend: _BaseBackend, backend_options: str, data_width: Tuple[int, int], small_model: bool) -> None:
        self.subjects = subjects
        self.collected_data = collected_data
        self.data_width_onnx = data_width[0]
        self.data_width_backend = data_width[1]

        self.collected_data.roofline = RooflineData()
        self.collected_data.roofline.backend = model_backend.__name__
        self.collected_data.roofline.backend_options = backend_options
        self.collected_data.roofline.data_width_onnx = data_width[0]
        self.collected_data.roofline.data_width_backend = data_width[1]

        from model.roofline import generate_roofline_test_model
        if not small_model:
            self.collected_data.roofline.model_type = 'default'
            model = generate_roofline_test_model(128, 8)    # default size, for GPU like NVIDIA A100
        else:
            self.collected_data.roofline.model_type = 'small'
            model = generate_roofline_test_model(32, 7)     # small size, for edge or cpu

        import onnx
        self.onnx_model = str(TMPDIR / 'roofline_test_model.onnx')
        onnx.save(model, self.onnx_model)

        self.model_backend_env: _BaseBackend = model_backend(self, self.onnx_model, [1], backend_options)
        self.collected_data.roofline.backend_version_info = self.model_backend_env.version_info()
        print(self.collected_data.roofline.backend_version_info)
        self.model_backend_env.prepare()

    def run(self) -> None:
        if 'layer_prof' not in self.model_backend_env.supported:
            log.error("can not run roofline test, backend %s not support [layer_prof]", self.collected_data.roofline.backend)
            return

        self.analyze = model.analyze.Analyze(self.onnx_model)
        self.model_backend_env.pre_batch_run(1)
        layer_prof = self.model_backend_env.layer_prof(1)

        flops: Dict[str, float] = {}    # name: FLOPS
        memory_bandwidth: Dict[str, float] = {}  # name: Byte/s
        for layer in layer_prof:
            onnx_nodes = layer.extra['onnx_nodes']
            for name in onnx_nodes:
                if name.startswith('MatMul'):
                    if len(onnx_nodes) > 1:
                        log.warning("in roofline_test_model, original onnx node %s is fused with %s, the results may inaccurate", name, onnx_nodes)
                    size = int(name[len('MatMul_'):])
                    flops[name] = layer.flops / layer.median_time
                    log.debug("MatMulOp size {}x{} reached {:.4f} GFLOPS".format(size, size, flops[name] / 1e9))
                if name.startswith('Relu_'):
                    if len(onnx_nodes) > 1:
                        log.warning("in roofline_test_model, original onnx node %s is fused with %s, the results may inaccurate", name, onnx_nodes)
                    size = int(name[len('Relu_'):])
                    memory_bandwidth[name] = layer.memory / layer.median_time
                    log.debug("ReluOp size {}x{} reached {:.4f} GB/s".format(size, size, memory_bandwidth[name] / 1e9))
                if name.startswith('Transpose_'):
                    if len(onnx_nodes) > 1:
                        log.warning("in roofline_test_model, original onnx node %s is fused with %s, the results may inaccurate", name, onnx_nodes)
                    size = int(name[len('Transpose_'):])
                    memory_bandwidth[name] = layer.memory / layer.median_time
                    log.debug("TransposeOp size {}x{} reached {:.4f} GB/s".format(size, size, memory_bandwidth[name] / 1e9))
                if name.startswith('Concat_'):
                    # if len(onnx_nodes) > 1:
                    #     log.warning("in roofline_test_model, original onnx node %s is fused with %s, the results may inaccurate", name, onnx_nodes)
                    memory_bandwidth[name] = layer.memory / layer.median_time
                    log.debug("{} reached {:.4f} GB/s".format(name, memory_bandwidth[name] / 1e9))

        self.collected_data.roofline.flops = max(flops.values())
        self.collected_data.roofline.memory_bandwidth = max(memory_bandwidth.values())
        print("reached roofline (large matmul): {:.4f} GFLOPS, {:.4f} GB/s".format(self.collected_data.roofline.flops / 1e9, self.collected_data.roofline.memory_bandwidth / 1e9))
        print("NOTE: Also run a ResNet-34 model to test Conv roofline, this may necessary for some device. ")

class ModelContext():
    def __init__(self, subjects: list, collected_data: CollectedData, model_backend: _BaseBackend, onnx_model: str, batch_size_list: list, repeat_count: int, backend_options: str, data_width: Tuple[int, int], llc_reuse_size: float, inputs_shape_override: Dict[str, List[Union[int, None]]]) -> None:
        self.subjects = subjects
        self.collected_data = collected_data
        self.onnx_model = onnx_model
        self.batch_size_list = batch_size_list
        self.repeat_count = repeat_count
        self.data_width_onnx = data_width[0]
        self.data_width_backend = data_width[1]
        log.info("data_width in onnx and backend is %s bit, change it if not correct", data_width)

        self.collected_data.model = ModelData()
        self.collected_data.model.name = Path(onnx_model).name
        self.collected_data.model.path = onnx_model
        self.collected_data.model.backend = model_backend.__name__
        self.collected_data.model.backend_options = backend_options
        self.collected_data.model.data_width_onnx = data_width[0]
        self.collected_data.model.data_width_backend = data_width[1]
        self.collected_data.model.llc_reuse_size = llc_reuse_size
        self.collected_data.model.inputs_shape_override = inputs_shape_override

        if inputs_shape_override:
            log.info("inputs_shape_override is set, will save the modified model to tmpdir as a copy")
            import onnx
            m = onnx.load(self.onnx_model)
            for t in m.graph.input:
                if t.name in inputs_shape_override:
                    for i, dim in enumerate(inputs_shape_override[t.name]):
                        if dim:
                            t.type.tensor_type.shape.dim[i].dim_value = dim
            self.onnx_model = str(TMPDIR / 'inputs_shape_override_model.onnx')
            if os.path.isfile(str(TMPDIR / 'model_external_data.pb')):
                os.unlink(str(TMPDIR / 'model_external_data.pb'))
            onnx.save(m, self.onnx_model, save_as_external_data=True, location="model_external_data.pb")

        if 'model.bench' in self.subjects:
            self.model_backend_env: _BaseBackend = model_backend(self, self.onnx_model, self.batch_size_list, backend_options)
            self.collected_data.model.backend_version_info = self.model_backend_env.version_info()
            print(self.collected_data.model.backend_version_info)
            self.model_backend_env.prepare()

    def run(self) -> None:

        if 'model.analyze' in self.subjects:
            self.analyze = model.analyze.Analyze(self.onnx_model)
            self.collected_data.model.analyze = self.analyze.export_data()

            # max fused memory
            effort_fused = get_effort_fused_model(self.analyze)
            self.collected_data.model.analyze.total_memory_effort_fused = effort_fused.get_memory()
            log.info("total memory %.3f M (vars) (approximate, effort fused)", effort_fused.get_memory() / 1e6)


        if 'model.bench' in self.subjects:
            self.collected_data.model.bench = ModelBenchData()
            self.collected_data.model.bench.batch_size_list = self.batch_size_list
            self.collected_data.model.bench.results = {}
            log.debug("batch_size_list: %s", self.batch_size_list)
            for batch_size in self.batch_size_list:
                batch_data = ModelBenchBatchData()
                batch_data.batch_size = batch_size

                print("="*60)
                print("batch_size: %s" % batch_size)

                self.model_backend_env.pre_batch_run(batch_size)
                if 'model.bench.layer_prof' in self.subjects:
                    if 'layer_prof' not in self.model_backend_env.supported:
                        log.error("layer_prof is not support in %s, skip", self.model_backend_env.__class__)
                    else:
                        batch_data.layer_prof = self.model_backend_env.layer_prof(batch_size)
                        batch_data.better_total_flops = sum(l.flops for l in batch_data.layer_prof)
                        batch_data.better_total_memory = sum(l.memory for l in batch_data.layer_prof)

                        DUMP_TO_DEBUG = False
                        DUMP_TO_DEBUG = True   # TODO_tmp: dev only
                        if DUMP_TO_DEBUG:
                            for layer in batch_data.layer_prof:
                                log.debug("[layer_prof dump] avg: %s ms, %s GFLOPS, %s GB/s, name: %s - %s",
                                    '{:8.4f}'.format(layer.median_time * 1000),
                                    '{:12.4f}'.format(layer.flops / layer.median_time / 1e9),
                                    '{:12.4f}'.format(layer.memory / layer.median_time / 1e9),
                                    '{:<64}'.format(layer.name),
                                    layer.extra)
                            log.debug("[layer_prof dump] GB/s is approximate memory bandwidth")
                            log.debug("[layer_prof dump] total_flops %.3f MFLOPs (%.3f * %d)",
                                batch_data.better_total_flops / 1e6,
                                batch_data.better_total_flops / 1e6 / batch_size,
                                batch_size)
                            log.debug("[layer_prof dump] total_memory %.3f MB (%.3f * %d)",
                                batch_data.better_total_memory / 1e6,
                                batch_data.better_total_memory / 1e6 / batch_size,
                                batch_size)
                            max_flops_layer = max(batch_data.layer_prof, key=lambda x: x.flops / x.median_time)
                            log.info(f"max flops node: {max_flops_layer.flops / max_flops_layer.median_time / 1e12:.3f} TFLOPS, ({max_flops_layer.median_time*1e3:.3f} ms)")


                if 'model.bench.e2e_prof' in self.subjects:
                    if 'e2e_prof' not in self.model_backend_env.supported:
                        log.error("e2e_prof is not support in %s, skip", self.model_backend_env.__class__)
                    else:
                        times = self.model_backend_env.e2e_prof(batch_size, self.repeat_count)

                        if isinstance(times, np.ndarray):
                            batch_data.times = list(times)
                            log.debug(times)

                            batch_data.time_avg = np.average(times)
                            batch_data.time_min = np.min(times)
                            batch_data.time_std = np.std(times)
                        elif isinstance(times, dict):
                            batch_data.time_avg = times['avg']
                            batch_data.time_min = times['min']
                            batch_data.time_std = times['std']

                        print("TIME:   average: {:12.4f} ms,        min: {:12.4f} ms,        std: {:8.4f} ms".format(
                            batch_data.time_avg * 1000, batch_data.time_min * 1000, batch_data.time_std * 1000))

                        if batch_data.better_total_flops and not os.getenv('PROOF_E2E_NOT_USE_LAYER_DATA'):
                            model_flops = batch_data.better_total_flops
                        else:
                            model_flops = float(self.collected_data.model.analyze.total_flops) * batch_size
                        batch_data.flops_avg = model_flops / batch_data.time_avg
                        batch_data.flops_max = model_flops / batch_data.time_min
                        batch_data.flops_std = np.std(model_flops / times) if isinstance(times, np.ndarray) else -1
                        print("GFLOPS: average: {:12.4f} GFLOPS,    max: {:12.4f} GFLOPS,    std: {:8.4f} GFLOPS".format(
                            batch_data.flops_avg / 1e9, batch_data.flops_max / 1e9, batch_data.flops_std / 1e9))

                        if batch_data.better_total_memory and not os.getenv('PROOF_E2E_NOT_USE_LAYER_DATA'):
                            accurate = "approximate"
                            model_memory_access = batch_data.better_total_memory
                        else:
                            accurate = "inaccurate"
                            model_memory_access = float(self.collected_data.model.analyze.total_memory) * batch_size
                        batch_data.memory_avg = model_memory_access / batch_data.time_avg
                        batch_data.memory_max = model_memory_access / batch_data.time_min
                        batch_data.memory_std = np.std(model_memory_access / times) if isinstance(times, np.ndarray) else -1
                        print("Memory: average: {:12.4f} GB/s,      max: {:12.4f} GB/s,      std: {:8.4f} GB/s \t({})".format(
                            batch_data.memory_avg / 1e9, batch_data.memory_max / 1e9, batch_data.memory_std / 1e9, accurate))


                self.collected_data.model.bench.results[str(batch_size)] = batch_data  # JSON format need a str for the key (batch_size)
                log.debug("run() model batch_size=%s done", batch_size)

            log.debug("run() model all batch_size done")

        log.debug("run() all done")
