from typing import List
from pathlib import Path
import itertools
import os
import subprocess
import logging
import json
import csv

from util import TMPDIR
from datatype import ModelBenchBatchLayerData
from . import Trtexec

log = logging.getLogger(__name__)
_notice = log.warning

def _ncu_get_flops_fallback(kernel_data: dict, data_width: int):
    _notice("kernel [%s] %s is using a fallback FLOPs formula", kernel_data['ID'], kernel_data['Kernel Name'])

    if data_width == 64:
        flops = float(kernel_data['derived__sm__sass_thread_inst_executed_op_dfma_pred_on_x2']) \
            * float(kernel_data['sm__throughput.avg.pct_of_peak_sustained_elapsed']) / 100 \
            * float(kernel_data['sm__cycles_elapsed.avg.per_second']) \
            * float(kernel_data['gpu__time_duration.sum']) / 1e9
    elif data_width == 32:
        flops = float(kernel_data['derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2']) \
            * float(kernel_data['sm__throughput.avg.pct_of_peak_sustained_elapsed']) / 100 \
            * float(kernel_data['sm__cycles_elapsed.avg.per_second']) \
            * float(kernel_data['gpu__time_duration.sum']) / 1e9
    elif data_width == 16:
        flops = float(kernel_data['derived__sm__sass_thread_inst_executed_op_hfma_pred_on_x2']) \
            * float(kernel_data['sm__throughput.avg.pct_of_peak_sustained_elapsed']) / 100 \
            * float(kernel_data['sm__cycles_elapsed.avg.per_second']) \
            * float(kernel_data['gpu__time_duration.sum']) / 1e9
    else:
        flops = 0
        _notice("data_width %s is not support", data_width)

    if os.getenv("PROOF_BACKEND_TRTEXEC_NCU_IGNORE_FALLBACK"):
        log.info("_ncu_get_flops_fallback_ignore:", flops)
        return -1
    return flops


def _ncu_get_flops_double(kernel_data: dict) -> float:
    flops = (float(kernel_data['smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed']) \
                + float(kernel_data['smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed']) \
                + float(kernel_data['derived__smsp__sass_thread_inst_executed_op_dfma_pred_on_x2'])) \
            * float(kernel_data['smsp__cycles_elapsed.avg.per_second']) \
            * float(kernel_data['gpu__time_duration.sum']) / 1e9

    return flops


def _ncu_get_flops_single(kernel_data: dict) -> float:
    flops = (float(kernel_data['smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed']) \
                + float(kernel_data['smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed']) \
                + float(kernel_data['derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2'])) \
            * float(kernel_data['smsp__cycles_elapsed.avg.per_second']) \
            * float(kernel_data['gpu__time_duration.sum']) / 1e9

    return flops


def _ncu_get_flops_half(kernel_data: dict) -> float:
    flops = (float(kernel_data['smsp__sass_thread_inst_executed_op_hadd_pred_on.sum.per_cycle_elapsed']) \
                + float(kernel_data['smsp__sass_thread_inst_executed_op_hmul_pred_on.sum.per_cycle_elapsed']) \
                + float(kernel_data['derived__smsp__sass_thread_inst_executed_op_hfma_pred_on_x2'])) \
            * float(kernel_data['smsp__cycles_elapsed.avg.per_second']) \
            * float(kernel_data['gpu__time_duration.sum']) / 1e9

    return flops


def _have_strings(name: str, *strings):
    return any(s in name for s in strings)


def _ncu_get_flops_tensor(kernel_data: dict) -> float:
    kernel_name = kernel_data['Kernel Name']

    factor = 512    # default (volta), 8x8x4 x 2 OP/FMA

    # ampere (A100 etc) fp16
    if _have_strings(kernel_name, '16816', 'tensor16x8x16'):
        factor = 4096
    elif _have_strings(kernel_name, '1688', 'tensor16x8x8'):
        factor = 2048

    # ampere (A100 etc) int8
    elif _have_strings(kernel_name, 'i8i8_i8i32_f32') \
        and _have_strings(kernel_name, 'tensor16x8x32'):
        factor = 8192
    elif _have_strings(kernel_name, 'i8i8_i32_f32'):
        factor = 8192
    elif _have_strings(kernel_name, 'i8i8_i8i32_f32') \
        and _have_strings(kernel_name, 'tensor8x8x16'):
        factor = 2048
    elif _have_strings(kernel_name, 'imma') and _have_strings(kernel_name, 'ampere'):    # ampere_first_layer_filter3x3_imma_fwd_swish_execute_filter3x3_swish_kernel_trt
        factor = 2048

    # TODO: need to verify
    # volta (V100 etc), HMMA.884.F16.F16 fix
    elif (
            (_have_strings(kernel_name, 'h884') or
                (_have_strings(kernel_name, 'f16f16_f16f16_f16') and _have_strings(kernel_name, 'tensor8x8x4'))
            ) and not _have_strings(kernel_name, 's884')
        ):
        factor = 1024

    log.debug("kernel_name %s", kernel_name)
    log.debug("tensorcore HMMA factor %s", factor)
    flops = float(kernel_data['smsp__inst_executed_pipe_tensor.sum.per_cycle_elapsed']) \
            * factor \
            * float(kernel_data['smsp__cycles_elapsed.avg.per_second']) \
            * float(kernel_data['gpu__time_duration.sum']) / 1e9

    return flops


def ncu_get_flops(kernel_data: dict, data_width: int) -> float:
    """return all double/single/half/tensor FLOPs (count of FLoat OP)"""
    all_flops = (
        _ncu_get_flops_double(kernel_data),
        _ncu_get_flops_single(kernel_data),
        _ncu_get_flops_half(kernel_data),
        _ncu_get_flops_tensor(kernel_data)
    )
    log.debug("flops: d/s/h/t %s M", list(map("{:.3f}".format, map(1e6.__rtruediv__, all_flops))))
    flops = sum(all_flops)
    if not flops:
        flops = _ncu_get_flops_fallback(kernel_data, data_width)    # not good!
        log.debug("flops: {:.3f} M".format(flops / 1e6))
    return flops


def ncu_get_memory_io(kernel_data: dict) -> float:
    memory = (float(kernel_data['dram__bytes.sum.per_second'])
                * float(kernel_data['gpu__time_duration.sum']) / 1e9)
    # log.debug("dram__bytes.sum: %.3f M", memory)
    return memory


def _remove_double_quotes(s: str) -> str:
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    return s


def _kernel_name_simpler(obj, name: str) -> str:
    if not hasattr(obj, '_kernel_name_simpler_re'):
        import re
        # obj._kernel_name_simpler_re = re.compile(r'^(?:[a-zA-Z0-9_]+ )?(?:[a-zA-Z0-9_]+::)?([a-zA-Z0-9_]+)(?:<.*>)?(?:\(.*\))?$')
        obj._kernel_name_simpler_re = re.compile(r'^(?:[a-zA-Z0-9_]+ )?([a-zA-Z0-9_:]+)(?:<.*>)?(?:\(.*\))?$')
    return obj._kernel_name_simpler_re.match(name).group(1)


def layer_prof_ncu(obj: Trtexec, batch_size: int) -> List[ModelBenchBatchLayerData]:
    # use times from ncu instead of trtexec, times from ncu is a little bit slow than trtexec
    USE_TIMES_FROM_NCU = os.getenv('PROOF_BACKEND_TRTEXEC_NCU_USE_TIMES_FROM_NCU')

    # stage1: trtexec to get layer names
    tmp_profile_file = str(TMPDIR / "trtexec_profile.txt")
    trtexec_inference_cmd = [obj.trtexec_bin,
        "--warmUp=1000",
        "--loadEngine="+obj.target_model,
        "--exportProfile="+tmp_profile_file]

    if obj.trtexec_layer_prof_run_arg:
        trtexec_inference_cmd += obj.trtexec_layer_prof_run_arg.split()

    log.debug("stage1: running trtexec (inference) ...")
    log.debug(' '.join(trtexec_inference_cmd))

    trtexec_inference_run = subprocess.run(trtexec_inference_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if trtexec_inference_run.returncode != 0:
        log.error("model inference failed")
        log.error("command: %s", ' '.join(trtexec_inference_run.args))
        log.error("output:\n[stdout]\n%s\n[stderr]\n%s", trtexec_inference_run.stdout.decode(), trtexec_inference_run.stderr.decode())
        raise RuntimeError

    log.debug("trtexec running done")

    with open(tmp_profile_file, 'r') as f:
        results = json.load(f)

    results = results[1:]      # remove "count"

    layers_profile = [
        ModelBenchBatchLayerData(
            name = x['name'],
            median_time = (x['medianMs'] if 'medianMs' in x else x['averageMs']) / 1000,
            time_percentage = None, # medianMs / sum(medianMs) * 100%
            flops = 0,              # FLOPs, (float) operation count
            memory = 0,             # memory access count
            extra = {
                "tensorcore_used": None,
                "kernels": {}        # cuda kernel (name: [ID_in_ncu, ...])
            }
        ) for x in results]

    # TODO: may not necessary
    # pass: remove dummy layers
    layers_profile = [x for x in layers_profile if obj._trt_layer_filter(x)]

    del results

    # stage2: use dlprof to get (layer name, kernel name) mapping
    dlprof_output_dir = str(TMPDIR)
    dlprof_cmd = ["dlprof",
        "--mode=tensorrt",
        "--iter_start=8",
        "--iter_stop=8",
        "--reports=iteration",
        "--formats=json",
        "--output_path="+dlprof_output_dir,
        "--force=true",
        obj.trtexec_bin,
        "--loadEngine="+obj.target_model,
        "--iterations=10", "--duration=0", "--warmUp=0"]

    if obj.trtexec_layer_prof_run_arg:
        dlprof_cmd += obj.trtexec_layer_prof_run_arg.split()

    log.debug("stage2: running dlprof ...")
    log.debug(' '.join(dlprof_cmd))

    dlprof_run = subprocess.run(dlprof_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if dlprof_run.returncode != 0:
        log.error("dlprof failed")
        log.error("command: %s", ' '.join(dlprof_run.args))
        log.error("output:\n[stdout]\n%s\n[stderr]\n%s", dlprof_run.stdout.decode(), dlprof_run.stderr.decode())
        # raise RuntimeError
        log.warning("fallback with very limited info")

    else:
        log.debug("dlprof running done")

        with open(Path(dlprof_output_dir) / "dlprof_iteration.json", 'r') as f:
            results = json.load(f)["Iteration Report"]

        dict_layer_name = {layer.name: layer for layer in layers_profile}
        for it in results:
            name = _remove_double_quotes(it['Op Name'])

            kernels = _remove_double_quotes(it['Long Kernel Name'])
            tc_used = (it['Uses TC'] == "yes")

            if name in dict_layer_name:
                if kernels not in dict_layer_name[name].extra['kernels']:
                    dict_layer_name[name].extra['kernels'][kernels] = []
                dict_layer_name[name].extra['kernels'][kernels].append(-1)     # name: ID (to be found)
                dict_layer_name[name].extra['tensorcore_used'] = dict_layer_name[name].extra['tensorcore_used'] or tc_used

            else:
                log.debug("layer name (Op Name) '%s' not found in layers_profile (maybe dummy), ignore", name)

        del results

    # print(*layers_profile, sep='\n')

    # stage3: ncu
    # output_name = "%s_%d" % (Path(obj.onnx_model).stem, batch_size)
    output_name = "ncu"
    ncu_output_file = str(TMPDIR / ("%s.csv" % output_name))
    ncu_rep = "-o " + output_name if os.getenv('PROOF_BACKEND_TRTEXEC_NCU_SAVE_NCU_REP') else ""
    ncu_set = "--set=roofline"
    if os.getenv('PROOF_BACKEND_TRTEXEC_NCU_FULL_SET'):
        ncu_set = "--set=full \
            --section=SpeedOfLight_HierarchicalDoubleRooflineChart \
            --section=SpeedOfLight_HierarchicalHalfRooflineChart \
            --section=SpeedOfLight_HierarchicalSingleRooflineChart \
            --section=SpeedOfLight_HierarchicalTensorRooflineChart"
    ncu_cmd = [obj.ncu_bin,
        "--page=raw",
        "--csv",
        *ncu_set.split(),
        "--log-file="+ncu_output_file,
        *ncu_rep.split(),
        "-f",
        "--cache-control", "none",
        "--replay-mode", "application",
        obj.trtexec_bin,
        "--loadEngine="+obj.target_model,
        "--duration=0", "--warmUp=0", "--iterations=1"]

    if obj.trtexec_layer_prof_run_arg:
        ncu_cmd += obj.trtexec_layer_prof_run_arg.split()

    log.debug("stage3: running ncu ...")
    log.debug(' '.join(ncu_cmd))

    # NCU #
    if not os.getenv('PROOF_BACKEND_TRTEXEC_SKIP_NCU_RUN'):     # TODO: debug flag, remove in release
        ncu_run = subprocess.run(ncu_cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ncu_run.returncode != 0:
            log.error("model inference failed")
            log.error("command: %s", ' '.join(ncu_run.args))
            log.error("output:\n[stdout]\n%s\n[stderr]\n%s", ncu_run.stdout.decode(), ncu_run.stderr.decode())
            raise RuntimeError

    log.debug("ncu running done")

    if USE_TIMES_FROM_NCU:
        for layer in layers_profile:
            layer.median_time = 0

    with open(ncu_output_file, 'r') as f:
        while True:     # find and seek to the begin of csv table
            last_pos = f.tell()
            line = f.readline()
            if not line:
                log.error("failed to read NCU output file (%s), can't found CSV header", ncu_output_file)
                raise RuntimeError('failed to read ncu_output_file')
            if line.startswith('"ID"'):
                f.seek(last_pos)
                break


        # Fallback if dlprof bugged
        if dlprof_run.returncode != 0:
            log.error("DLProf failed, unable to get layer/kernel mapping, only total FLOP and MEM.IO correct")
            reader = csv.DictReader(f)
            next(reader)    # skip units line
            all_flops, all_memory = 0, 0
            for kernel in reader:
                all_flops += ncu_get_flops(kernel, obj.ctx.data_width_backend)
                all_memory += ncu_get_memory_io(kernel)
            log.info(f"all_flops: {all_flops / 1e9:.3f} GFLOPs")
            log.info(f"all_memory: {all_memory / 1e6:.3f} MB")
            log.error("give total FLOP and MEM.IO to last layer as fallback workaround for e2e profiling")
            layers_profile[-1].flops = all_flops
            layers_profile[-1].memory = all_memory
            return layers_profile


        reader = csv.DictReader(f)
        next(reader)    # skip units line

        data_width = obj.ctx.data_width_backend
        layers_kernels = [
            list(itertools.chain(
                *([_kernel_name_simpler(obj, k)] * len(v)
                for k, v in l.extra['kernels'].items()))) \
            for l in layers_profile]
        layer_idx = 0
        for kernel_idx, kernel in enumerate(reader):
            if layer_idx == len(layers_profile):
                break

            kernel_current = _kernel_name_simpler(obj, kernel['Kernel Name'])
            kernel_wanted = layers_kernels[layer_idx]
            while not kernel_wanted:
                layer_idx += 1
                kernel_wanted = layers_kernels[layer_idx]

            if kernel_current in kernel_wanted:   # FIXME: layer to multi kernel not tested
                log.debug("matched kernel [%s] %s at layer [%s] %s", kernel_idx, kernel_current, layer_idx, layers_profile[layer_idx].name)
                kernel_id_list: list = layers_profile[layer_idx].extra['kernels'][kernel['Kernel Name']]
                kernel_id_list[kernel_id_list.index(-1)] = kernel_idx

                if USE_TIMES_FROM_NCU:
                    layers_profile[layer_idx].median_time += float(kernel['gpu__time_duration.sum']) / 1e9

                flops = ncu_get_flops(kernel, data_width)
                if flops == -1:
                    # do something
                    flops = 0
                layers_profile[layer_idx].flops += int(flops)

                memory = ncu_get_memory_io(kernel)
                layers_profile[layer_idx].memory += int(memory)

                kernel_wanted.remove(kernel_current)
                if not kernel_wanted:
                    layer_idx += 1
            else:
                log.debug("skip not matched kernel [%s] %s (wants %s at layer %s)", kernel_idx, kernel_current, kernel_wanted, layer_idx)

        if layer_idx < len(layers_profile):
            log.warning("kernel match failed, only %s of %s layer matched", layer_idx, len(layers_profile))

    # pass: calculate time percentage
    time_sum = sum(layer.median_time for layer in layers_profile)
    print(time_sum)
    for layer in layers_profile:
        layer.time_percentage = layer.median_time / time_sum * 100

    # pass: if layer FLOPs or memory I/O is 0, show warning and set to -1 prevent error in context
    for layer in layers_profile:
        if layer.flops == 0:
            log.warning("layer %s 's FLOPs is 0, set it to -1 for unknown", layer.name)
            layer.flops = -1
        if layer.memory == 0:
            log.warning("layer %s 's memory I/O is 0, set it to -1 for unknown", layer.name)
            layer.memory = -1

    return layers_profile

