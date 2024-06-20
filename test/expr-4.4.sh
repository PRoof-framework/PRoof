#!/bin/bash

# meta type: tensorrt + ncu

filename=$(basename "$0")
export EXPR_NAME="${filename%.*}"
export BASE_DIR="$(pwd)/.."
export RESULT_DIR="$(pwd)/result_${EXPR_NAME}"
export WORK_DIR="$(pwd)/result_${EXPR_NAME}/workdir"

mkdir -p $WORK_DIR

: ${MODEL_DIR:="$(pwd)/model"}

echo "EXPR_NAME: ${EXPR_NAME}"
echo "BASE_DIR: ${MODEL_DIR}"
echo "RESULT_DIR: ${RESULT_DIR}"
echo "WORK_DIR: ${WORK_DIR}"
echo "MODEL_DIR: ${MODEL_DIR}"


function trt_cache_pre() {
    if [[ -n ${DO_NOT_USE_CACHE} ]]; then
        echo "DO_NOT_USE_CACHE set, skip cache check"
        return
    fi

    cached_trt="${WORK_DIR}/${m}.trt"
    if [[ -e ${cached_trt} ]]; then
        echo "using cached tensorrt engine ${cached_trt}"
        export PROOF_BACKEND_TRTEXEC_SKIP_CONVERT=1
        cp ${cached_trt} /tmp/proof/model.trt
    else
        unset PROOF_BACKEND_TRTEXEC_SKIP_CONVERT
    fi

    cached_ncu="${WORK_DIR}/${m}_${bs}_ncu.csv"
    if [[ -e ${cached_ncu} ]]; then
        echo "using cached ncu result ${cached_ncu}"
        export PROOF_BACKEND_TRTEXEC_SKIP_NCU_RUN=1
        cp ${cached_ncu} /tmp/proof/ncu.csv
    else
        unset PROOF_BACKEND_TRTEXEC_SKIP_NCU_RUN
    fi
}

function trt_cache_post() {
    if [[ -n ${DO_NOT_USE_CACHE} ]]; then
        echo "DO_NOT_USE_CACHE set, skip cache check"
        return
    fi

    cached_trt="${WORK_DIR}/${m}.trt"
    if [[ -n ${PROOF_BACKEND_TRTEXEC_SKIP_CONVERT} ]]; then
        unset PROOF_BACKEND_TRTEXEC_SKIP_CONVERT
    else
        cp /tmp/proof/model.trt ${cached_trt}
        echo "cached tensorrt engine ${cached_trt}"
    fi

    cached_ncu="${WORK_DIR}/${m}_${bs}_ncu.csv"
    if [[ -n ${PROOF_BACKEND_TRTEXEC_SKIP_NCU_RUN} ]]; then
        unset PROOF_BACKEND_TRTEXEC_SKIP_NCU_RUN
    else
        cp /tmp/proof/ncu.csv ${cached_ncu}
        echo "cached ncu result ${cached_ncu}"
    fi
}


### normal models
MODELS='resnet50.onnx vit_tiny_patch16_224.onnx efficientnet_b4.onnx efficientnetv2_rw_t.onnx'

set -x
bs=128
for m in $MODELS; do
    echo $m

    # with or without Nsight Compute
    trt_cache_pre
    python -u ${BASE_DIR}/main.py -m ${MODEL_DIR}/${m} -s model -b ${bs} -o fp16,use_ncu -D 32,16 -v -f ${WORK_DIR}/${m}-ncu-fp16.json 2>&1 | tee ${WORK_DIR}/${m}-ncu-fp16.log
    PROOF_BACKEND_TRTEXEC_SKIP_NCU_RUN=1 python -u ${BASE_DIR}/main.py -m ${MODEL_DIR}/${m} -s model -b ${bs} -o fp16,use_nsys_for_myelin -D 32,16 -v -f ${WORK_DIR}/${m}-nsys-fp16.json 2>&1 | tee ${WORK_DIR}/${m}-nsys-fp16.log
    trt_cache_post

done


echo "=== PRoof done, generate graph ==="

python ${BASE_DIR}/dataviewer/read_value.py 'model.bench.results.*.layer_prof.*.time_percentage; model.bench.results.*.layer_prof.*.flops; model.bench.results.*.layer_prof.*.memory; model.bench.results.*.layer_prof.*.median_time; model.bench.results.*.layer_prof.*.name' ${WORK_DIR}/resnet50.onnx-ncu-fp16.json -e > ${WORK_DIR}/fig3_1.csv
python ${BASE_DIR}/dataviewer/read_value.py 'model.bench.results.*.layer_prof.*.time_percentage; model.bench.results.*.layer_prof.*.flops; model.bench.results.*.layer_prof.*.memory; model.bench.results.*.layer_prof.*.median_time; model.bench.results.*.layer_prof.*.name' ${WORK_DIR}/vit_tiny_patch16_224.onnx-nsys-fp16.json -e > ${WORK_DIR}/fig3_2.csv
python ${BASE_DIR}/dataviewer/read_value.py 'model.bench.results.*.layer_prof.*.time_percentage; model.bench.results.*.layer_prof.*.flops; model.bench.results.*.layer_prof.*.memory; model.bench.results.*.layer_prof.*.median_time; model.bench.results.*.layer_prof.*.name' ${WORK_DIR}/efficientnet_b4.onnx-ncu-fp16.json -e > ${WORK_DIR}/fig3_3.csv
python ${BASE_DIR}/dataviewer/read_value.py 'model.bench.results.*.layer_prof.*.time_percentage; model.bench.results.*.layer_prof.*.flops; model.bench.results.*.layer_prof.*.memory; model.bench.results.*.layer_prof.*.median_time; model.bench.results.*.layer_prof.*.name' ${WORK_DIR}/efficientnetv2_rw_t.onnx-ncu-fp16.json -e > ${WORK_DIR}/fig3_4.csv

python ${BASE_DIR}/test/helper_figure3.py ${WORK_DIR} ${RESULT_DIR}/figure3

echo "=== figure/table saved to ${RESULT_DIR} ==="
