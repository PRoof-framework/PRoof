from typing import Dict, List, Tuple
from collections import defaultdict
import subprocess
import logging
import json

import numpy as np

from util import TMPDIR
from . import Trtexec

log = logging.getLogger(__name__)


def nsys_run(obj: Trtexec) -> str:
    tmp_profile_file = str(TMPDIR / 'nsys_profile.nsys-rep')
    tmp_json_file = str(TMPDIR / 'nsys_profile.json')

    nsys_trtexec_inference_cmd = [obj.nsys_bin,
        "profile", "--force-overwrite", "true",
        "--export", "json",
        "-o", tmp_profile_file,
        obj.trtexec_bin,
        "--loadEngine="+obj.target_model,
        "--duration=0", "--warmUp=0", "--iterations=20"]

    log.debug("running nsys ...")
    log.debug(' '.join(nsys_trtexec_inference_cmd))

    nsys_trtexec_inference_run = subprocess.run(nsys_trtexec_inference_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if nsys_trtexec_inference_run.returncode != 0:
        log.error("nsys profile failed")
        log.error("command: %s", ' '.join(nsys_trtexec_inference_run.args))
        log.error("output:\n[stdout]\n%s\n[stderr]\n%s", nsys_trtexec_inference_run.stdout.decode(), nsys_trtexec_inference_run.stderr.decode())
        raise RuntimeError

    return tmp_json_file    # return json record file


# return [(sub_name, median_time, kernel_name), ...]
def nsys_myelin_layer_get_names(json_file: str, myelin_layer_name: str, skip: int = 10) -> List[Tuple[str, float, str]]:
    with open(json_file) as f:
        texts: List[str] = json.loads(f.readline())['data']
        events: List[dict] = []
        for line in f.readlines():
            events.append(json.loads(line))

    myelin_layer_name_id = texts.index(myelin_layer_name)
    # print('myelin_layer_name_id', myelin_layer_name_id)
    results: List[Tuple[str, float]] = []
    results_name: List[str] = []
    results_dict: Dict[str: List[float]] = defaultdict(list)

    correlation_id_cuda_event = {}
    for e in events:
        try:
            correlation_id_cuda_event[e['CudaEvent']['correlationId']] = e
        except KeyError:
            pass

    skip_count = skip
    name_added = False
    time_sum = 0
    for top_i, top_e in enumerate(events):
        if 'NvtxEvent' in top_e and 'TextId' in top_e['NvtxEvent'] \
            and top_e['NvtxEvent']['TextId'] == myelin_layer_name_id:

            # print(" + match")
            # print(top_e)

            if skip_count:
                skip_count -= 1
                continue

            start = float(top_e['NvtxEvent']['Timestamp'])
            end = float(top_e['NvtxEvent']['EndTimestamp'])
            # print(" - time range", start, end)

            i = top_i
            for sub_e in events[top_i+1:]:
                i += 1
                if 'NvtxEvent' in sub_e and 'TextId' in sub_e['NvtxEvent']:
                    # print(i, sub_e)
                    sub_start = float(sub_e['NvtxEvent']['Timestamp'])
                    sub_end = float(sub_e['NvtxEvent']['EndTimestamp'])

                    if end <= sub_start:
                        break

                    # assert start <= sub_start and sub_end <= end
                    if start <= sub_start and sub_end <= end:
                        # print("    - in range", json.dumps(sub_e))

                        # if 'Text' in sub_e['NvtxEvent']:
                        #     print("    - Text", sub_e['NvtxEvent']['Text'])
                        #     continue

                        # is NvtxEvent
                        name = texts[sub_e['NvtxEvent']['TextId']]
                        if not name_added:
                            results_name.append(name)
                        # print("    - TextId", name)

                        # search for GPU side event
                        correlation_ids = []
                        for e in events[i+1:]:
                            if 'TraceProcessEvent' in e and sub_end <= float(e['TraceProcessEvent']['startNs']):
                                break
                            elif 'CudaEvent' in e and sub_end <= float(e['CudaEvent']['startNs']):
                                break
                            elif 'NvtxEvent' in e and sub_end <= float(e['NvtxEvent']['Timestamp']):
                                break

                            if 'TraceProcessEvent' in e:
                                e_start = float(e['TraceProcessEvent']['startNs'])
                                e_end = float(e['TraceProcessEvent']['endNs'])
                                if sub_start <= e_start and e_end <= sub_end:
                                    correlation_id = e['TraceProcessEvent']['correlationId']
                                    correlation_ids.append(correlation_id)

                        for correlation_id in correlation_ids:
                            try:
                                e = correlation_id_cuda_event[correlation_id]
                            except KeyError as e:
                                log.warning(e)
                                continue
                            kernel_name = texts[int(e['CudaEvent']['kernel']['demangledName'])]
                            # print(f"{name = }, {kernel_name = }", e)
                            time_kernel = (float(e['CudaEvent']['endNs']) - float(e['CudaEvent']['startNs'])) / 1e9
                            results_dict[name].append((kernel_name, time_kernel))
                            time_sum += time_kernel
                            # print('time %s ms' % (time_kernel * 1000))

            name_added = True
            # print('time sum %s ms' % (time_sum * 1000))
            time_sum = 0

    for name in results_name:
        kernel_names = [x for x, _ in results_dict[name]]
        assert all(kernel_names[0] == x for x in kernel_names[1:]), "kernel name not same"
        times = [x for _, x in results_dict[name]]
        # print(f"{ name = }")
        # print(f"{ kernel_names = }")
        # print(f"{ times = }\n")
        results.append((name, np.median(times), kernel_names[0]))

    return results
