import os
import sys
import subprocess
import logging

import numpy as np

from nart.art import load_parade, create_context_from_json, get_empty_io

from context import PerfContext
from util import TMPDIR
from .. import _BaseBackend

log = logging.getLogger(__name__)

class Nart_TRT(_BaseBackend):
    supported = ['e2e_prof']

    def __init__(self, ctx: PerfContext, onnx_model: str, batch_size_list: list, backend_options: str) -> None:
        self.collected_data = ctx.collected_data
        self.onnx_model = onnx_model
        self.batch_size_list = batch_size_list

        self.convert_config_file = None
        for option in backend_options.split(','):
            if option:
                key, *value = option.strip().split('=')
                if key == 'config':
                    self.convert_config_file = value[0]
                elif key == 'help':
                    print("backend %s options help:" % self.__class__)
                    print("    config=<nart.switch config.json>:   specify config file during module convert")
                    print("    help:                               print this help and exit")
                    sys.exit(0)
                else:
                    raise RuntimeError("Unknown backend_options: %s" % key)

    def version_info(self) -> str:
        info = f"Using backend {self.__class__}\n"
        import nart
        info += f"nart version: {nart.__version__}\n"
        try:
            import tensorrt
            info += f"tensorrt version: {tensorrt.__version__}\n"
        except ModuleNotFoundError:
            log.warning("python module tensorrt not found")

        return info[:-1]

    def prepare(self) -> None:
        pass

    def pre_batch_run(self, batch_size: int) -> None:
        self.target_model = str(TMPDIR / "engine.bin")
        nart_switch_cmd = ["python",
            "-m", "nart.switch",
            "-t", "tensorrt",
            "--onnx", self.onnx_model,
            "--output", self.target_model,
            "-b", str(batch_size)]
        if self.convert_config_file:
            nart_switch_cmd += ["-c", self.convert_config_file]

        log.debug("running nart.switch ...")
        log.debug(' '.join(nart_switch_cmd))

        if not os.getenv('PROOF_BACKEND_NART_SKIP_CONVERT'):  # TODO: debug flag, remove in release
            convert_tool_run = subprocess.run(nart_switch_cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if convert_tool_run.returncode != 0:
                log.error("model convert failed")
                log.error("command: %s", ' '.join(convert_tool_run.args))
                log.error("output:\n[stdout]\n%s\n[stderr]\n%s", convert_tool_run.stdout.decode(), convert_tool_run.stderr.decode())
                raise RuntimeError

        log.debug("pre_batch_run() end")

    def e2e_prof(self, batch_size: int, repeat: int = 10, warm_up: int = 3) -> np.ndarray:
        log.debug("load target model")
        ctx = create_context_from_json(self.target_model + '.json')
        parade = load_parade(self.target_model, ctx)
        log.debug("model I/O shape: %s %s", parade.input_shapes(), parade.output_shapes())
        inputs, outputs = get_empty_io(parade)
        assert len(inputs) == 1 and len(outputs) == 1

        input_array = inputs[next(iter(inputs.keys()))]
        output_array = outputs[next(iter(outputs.keys()))]

        input_array[:] = np.random.standard_normal(input_array.shape)

        log.debug("using _nart_run.perf_parade_run()")
        from . import _nart_run
        # times = _nart_run.perf_parade_run(parade, inputs, outputs, 1, 0) # TODO_TMP: tmp for ncu
        times_ms = _nart_run.perf_parade_run(parade, inputs, outputs, repeat, warm_up)

        log.debug("running done")

        return np.array(times_ms) / 1000
