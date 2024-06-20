from typing import List
from pathlib import Path
from dataclasses import asdict
import os
import sys
import tempfile
import json
import argparse
import logging

import coloredlogs

import model.backend
from context import PerfContext
from util import TMPDIR

log = logging.getLogger(__name__)


def _get_batch_size_list(batch_size_arg: str) -> List[int]:
    if '-' in batch_size_arg:   # range
        min_b, max_b = map(int, batch_size_arg.split('-'))
        assert min_b > 0
        assert max_b >= min_b
        assert min_b & (min_b - 1) == 0
        assert max_b & (max_b - 1) == 0
        li = [min_b]
        while li[-1] < max_b:
            li.append(li[-1] * 2)
        assert li[-1] == max_b
        return li
    elif ',' in batch_size_arg:
        li = list(map(int, batch_size_arg.split(',')))
        assert all(b > 0 for b in li)
        return li
    else:   # number
        batch_size = int(batch_size_arg)
        assert batch_size > 0
        return [batch_size]


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    # add empty line if help ends with \n
    def _split_lines(self, text, width):
        lines = super()._split_lines(text, width)
        if '\n (default:' in text:
            lines.append('')
        return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = "PRoof",
        description = "PRoof is designed to be a universal, end-to-end, fine-grained AI model inference performance analysis and hardware AI benchmark tool",
        epilog = "for each model backend, it may contains it's own additional options (via -o), use '-B <backend> -o help' to view it's help",
        formatter_class=_HelpFormatter)

    parser.add_argument('-v', '--verbose', action='store_true', help="will set log_level to logging.DEBUG\n")
    parser.add_argument('-s', '--subjects', default='', help="test subjects to run, prefix means all sub subjects, empty value means all, available: %s" % PerfContext.list_subjects())
    parser.add_argument('-f', '--output', default="report.json", help="output file contain all test result\n")

    parser.add_argument('-B', '--model-backend', default='trtexec', help="backend to run model", choices=model.backend.get_available_backends())
    parser.add_argument('-D', '--data-width', default='32,32', help="specify data width for model analyze, format 'onnx,backend', \
        fp32 = 32, fp16 = 16, should same as data in onnx model and actual used in backend (e.g. in converted model)", type=str)
    parser.add_argument('-o', '--backend-options', default='', help="options passed to backend\n", type=str)

    parser.add_argument('-m', '--onnx-model', default='', help="[model] onnx format model to test")
    parser.add_argument('-b', '--batch-size', default='1', help="[model] specific or range of batch_size to test, like '4' or '1-16'")
    parser.add_argument('-r', '--repeat', default=10, help="[model] repeat count", type=int)
    parser.add_argument('--inputs-shape-override', default='', help="[model] override the shape of one or more input (dim0 will still been overwrite later as batch_size), use json format like: {\"input1\": [null, 128], ...}, null value will been skiped\n")


    parser.add_argument('--roofline-use-small-model', action='store_true', help="[roofline] use small model in roofline test, for edge or cpu")
    parser.add_argument('--llc-reuse-size', default=0.0, help="in MiB, size of activation DRAM access (R+W) reduced by Last-Level-Cache", type=float)
    parser.add_argument('--no-color', action='store_true', help="force disable coloredlogs, may useful for log record on Windows")
    args = parser.parse_args()

    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG

    logging.addLevelName(logging.INFO, "INFO")
    logging.addLevelName(logging.DEBUG, "DEBUG")
    logging.addLevelName(logging.WARNING, "WARNING")
    logging.addLevelName(logging.CRITICAL, "CRITICAL")
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s", level=log_level)
    if not args.no_color:
        level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.copy()
        level_styles['debug']['color'] = 'white'
        coloredlogs.install(fmt="%(asctime)s %(levelname)s %(name)s %(message)s", level=log_level, level_styles=level_styles)
    log.debug("NOTICE: debug logging is ON")

    log.info("Temporary directory (PROOF_TMPDIR): %s", TMPDIR)

    subjects = [x.strip() for x in (args.subjects.split(','))] if args.subjects else []

    if args.model_backend not in model.backend.get_available_backends():
        log.error("unsupported backend %s", args.model_backend)
        sys.exit(1)

    model_backend = model.backend.get_backend(args.model_backend)
    onnx_model = args.onnx_model
    batch_size_list = _get_batch_size_list(args.batch_size)
    repeat_count = args.repeat
    backend_options = args.backend_options
    data_width = tuple(map(int, args.data_width.split(",")))
    llc_reuse_size = args.llc_reuse_size
    roofline_small = args.roofline_use_small_model
    inputs_shape_override = json.loads(args.inputs_shape_override) if args.inputs_shape_override else {}

    if not subjects:
        subjects = PerfContext.list_subjects()
    log.info("subjects: %s", subjects)
    if any(x.startswith('model') for x in subjects):
        if not onnx_model:
            log.error("for subject 'model', the ONNX model to test is required (-m ./xxx.onnx)")
            sys.exit(1)

    ctx = PerfContext(subjects, model_backend,
        onnx_model, batch_size_list, repeat_count, backend_options, data_width,
        llc_reuse_size=llc_reuse_size,
        roofline_small=roofline_small,
        inputs_shape_override=inputs_shape_override)

    try:
        ctx.run()
    except BaseException as e:
        import traceback
        traceback.print_exc()
        log.error("got exception %s when running test" % e.__class__)
        log.error("report file (if any) maybe incomplete")
    finally:
        # TODO: cleanup
        pass

    if args.output:
        dat = asdict(ctx.collected_data)
        # DEBUG: to debug with dump error:
        # from test.util import dump_dict_with_type
        # dump_dict_with_type(dat)

        with open(args.output, 'w') as f:
            f.write(json.dumps(dat))
            log.info("collected data saved to %s", args.output)
