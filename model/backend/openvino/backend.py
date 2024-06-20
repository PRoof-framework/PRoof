
import json
import logging
from pathlib import Path
import re
from typing import List
import subprocess
import sys
from typing import List
import numpy as np

from context import ModelContext
from datatype import ModelBenchBatchLayerData
from model.analyze.fuse import _FusedOp, FusedAnalyze
from model.analyze.op import _DummyOp
from model.backend import _BaseBackend
from util import TMPDIR

log = logging.getLogger(__name__)


class OpenVINOBackend(_BaseBackend):
    supported = ['e2e_prof', 'layer_prof']

    def __init__(self, ctx: ModelContext, onnx_model: str, batch_size_list: list, backend_options: str) -> None:
        self.ctx = ctx
        self.collected_data = ctx.collected_data
        self.onnx_model = onnx_model
        self.batch_size_list = batch_size_list

        self.ov_device = None
        self.benchmark_app_bin = 'benchmark_app'
        self.benchmark_app_arg = None
        self.int8 = False
        for option in backend_options.split(','):
            if option:
                key, *value = option.strip().split('=', maxsplit=1)     # value is optional
                if key == 'device':
                    self.ov_device = value[0]
                elif key == 'benchmark_app_bin':
                    self.benchmark_app_bin = value[0]
                elif key == 'benchmark_app_arg':
                    self.benchmark_app_arg = value[0]
                elif key == 'int8':
                    self.int8 = True
                elif key == 'help':
                    print("backend %s options help:" % self.__class__)
                    print("    device=DEV:                         device to use (-d for benchmark_app), default: CPU")
                    print("    benchmark_app_bin=PATH:             path to benchmark_app binary (PATH), default: benchmark_app")
                    print("    benchmark_app_arg=ARGS:             addition arguments (ARGS) passed to benchmark_app")
                    print("    help:                               print this help and exit")
                    sys.exit(0)
                else:
                    raise RuntimeError("Unknown backend_options: %s" % key)

        if not self.ov_device:
            self.ov_device = 'CPU'
            log.warning("target device default to CPU, use \"-o device=DEV\" to set")

    def version_info(self) -> str:
        info = f"Using backend {self.__class__}\n"

        ov_check = [self.benchmark_app_bin, '--help']
        try:
            ov_check_run = subprocess.run(ov_check,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            import re
            ov_devices_match = re.search(r"Available target devices: (.*)\n", ov_check_run.stdout.decode())
            ov_devices = ov_devices_match.group(1) if ov_devices_match else '<failed to get>'
            info += f"OpenVINO Available target devices: {ov_devices}\n"

        except FileNotFoundError:
            log.error("benchmark_app binary (part of OpenVINO) path %s not exist, you may set it with backend options 'benchmark_app_bin'", self.benchmark_app_bin)
            log.error(r"    e.g. -o benchmark_app_bin='c:\venv\Scripts\benchmark_app.exe'")
            sys.exit(1)

        return info[:-1]

    def prepare(self) -> None:
        if self.int8:
            log.info("quantize model to int8 (require 'onnxruntime' package)")
            from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
            quantized_model = str(TMPDIR / (Path(self.onnx_model).stem + '-int8.onnx'))
            quantize_static(
                self.onnx_model,
                quantized_model,
                type('', (), {'_val': iter(({'input': np.single(np.random.rand(1, 3, 224, 224))}, )), 'get_next': lambda self: next(self._val, None)})(),
                # quant_format=args.quant_format,
                # per_channel=args.per_channel,
                weight_type=QuantType.QInt8,
            )

            self.onnx_model = quantized_model
            log.debug(f"model saved to {self.onnx_model}")

        elif self.ov_device == 'NPU':
            log.info("device is NPU, will convert model to fp16 (require 'onnxconverter_common' package)")
            from onnxconverter_common import float16
            import onnx
            model = onnx.load(self.onnx_model)

            model_fp16 = float16.convert_float_to_float16(model, disable_shape_infer=True)
            self.onnx_model = str(TMPDIR / (Path(self.onnx_model).stem + '-fp16.onnx'))
            onnx.save(model_fp16, self.onnx_model)
            log.debug(f"model saved to {self.onnx_model}")


    def pre_batch_run(self, batch_size: int) -> None:
        pass

    def e2e_prof(self, batch_size: int, repeat: int = 10, warm_up: int = 3) -> dict:
        report_folder = str(TMPDIR)
        ov_inference_cmd = [self.benchmark_app_bin,
            "--target_device="+self.ov_device,
            "--path_to_model="+self.onnx_model,
            "--inference_only",     # TODO: is this always correct?
            # "--number_iterations="+str(repeat),
            "--time="+str(repeat),  # FIXME
            "--report_type=detailed_counters",
            "--json_stats",
            "--number_infer_requests=1",    # FIXME
            "--perf_counts_sort=no_sort",
            "--report_folder="+report_folder]

        if batch_size:
            ov_inference_cmd.append("--batch_size="+str(batch_size))

        if self.benchmark_app_arg:
            ov_inference_cmd += self.benchmark_app_arg.split()

        log.debug("running openvino's benchmark_app (inference) ...")
        log.debug(' '.join(ov_inference_cmd))

        ov_inference_run = subprocess.run(ov_inference_cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if ov_inference_run.returncode != 0:
            log.error("model inference failed")
            log.error("command: %s", ' '.join(ov_inference_run.args))
            log.error("output:\n[stdout]\n%s\n[stderr]\n%s", ov_inference_run.stdout.decode(), ov_inference_run.stderr.decode())
            raise RuntimeError

        command_output = ov_inference_run.stdout.decode()
        time_avg = float(re.search(r"Average: (.*) ms", command_output).group(1).strip()) / 1000
        time_min = float(re.search(r"Min: (.*) ms", command_output).group(1).strip()) / 1000
        time_std = 0

        return dict(avg=time_avg, min=time_min, std=time_std)

    def layer_prof(self, batch_size: int) -> List[ModelBenchBatchLayerData]:
        report_folder = TMPDIR
        ov_inference_cmd = [self.benchmark_app_bin,
            "--target_device="+self.ov_device,
            "--path_to_model="+self.onnx_model,
            "--inference_only",     # TODO: is this always correct?
            # "--number_iterations="+str(repeat),
            "--time="+str(10),  # FIXME
            "--report_type=detailed_counters",
            "--json_stats",
            "--number_infer_requests=1",
            "--perf_counts_sort=no_sort",
            "--report_folder="+str(report_folder)]

        if batch_size:
            # ov_inference_cmd.append("-shape=[1,32,32]")
            ov_inference_cmd.append("--batch_size="+str(batch_size))

        if self.benchmark_app_arg:
            ov_inference_cmd += self.benchmark_app_arg.split()


        log.debug("running openvino's benchmark_app (inference) ...")
        log.debug(' '.join(ov_inference_cmd))

        ov_inference_run = subprocess.run(ov_inference_cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if ov_inference_run.returncode != 0:
            log.error("model inference failed")
            log.error("command: %s", ' '.join(ov_inference_run.args))
            log.error("output:\n[stdout]\n%s\n[stderr]\n%s", ov_inference_run.stdout.decode(), ov_inference_run.stderr.decode())
            raise RuntimeError

        log.debug(ov_inference_run.stdout.decode())
        # command_output = ov_inference_run.stdout.decode()
        # time_avg = float(re.search(r"Average: (.*) ms", command_output).group(1).strip()) / 1000
        # time_min = float(re.search(r"Min: (.*) ms", command_output).group(1).strip()) / 1000
        # time_std = 0

        with open(report_folder / "benchmark_detailed_counters_report.json") as f:
            results = json.load(f)

        layers = results['detailed_performance'][0]['nodes']

        fused_analyze = FusedAnalyze(self.ctx.analyze)

        layers_profile: List[ModelBenchBatchLayerData] = []
        for layer in layers:
            lp = ModelBenchBatchLayerData(
                name = layer['name'],
                flops = 0,
                memory = 0,
                median_time = float(layer['real_time']) / 1000,
                extra = {"onnx_nodes": [], "ov_exec_type": layer['exec_type']},
            )
            layers_profile.append(lp)
            for x in fused_analyze.ops.values():
                if x.name == layer['name'][:len(x.name)]:
                    lp.extra['onnx_nodes'].append(x.name)
                    lp.flops = x.get_flops() * batch_size
                    lp.memory = x.get_memory() * self.ctx.data_width_backend / 8
                    fused_analyze.set_fused_op(layer['name'], [x.name])
                    break


        # pass: show unfused op
        layers_profile_dict = {x.name: x for x in layers_profile}
        for op in list(fused_analyze.ops.values()):
            if not isinstance(op, _FusedOp):
                if isinstance(op, _DummyOp):
                    log.debug("ignore not fused dummy op %s", op)

                log.warning("not fused op %s", op)
                parent_ops = fused_analyze.get_prev_op(op)
                if len(parent_ops) == 1:
                    log.warning("  fuse it to upstream op %s", parent_ops[0])
                    layers_profile_dict[parent_ops[0].name].extra['onnx_nodes'].append(op.name)
                    fused_analyze.set_fused_op(parent_ops[0].name, [parent_ops[0], op])
                else:
                    log.error("  can not fuse it (parent_ops = %s)", parent_ops)


        # pass: filter layers in layers_profile
        for layer in layers_profile:
            if layer.median_time == 0:
                log.debug("will remove zero time layer '%s'", layer.name)
        layers_profile = [layer for layer in layers_profile if layer.median_time != 0]

        # pass: calculate time percentage
        time_sum = sum(layer.median_time for layer in layers_profile)
        for layer in layers_profile:
            layer.time_percentage = layer.median_time / time_sum * 100

        return layers_profile

