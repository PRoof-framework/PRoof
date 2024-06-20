#!/bin/bash

# meta type: tensorrt
# sudo nvpmodel -m 2

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

}


### normal models
MODELS='efficientnet_b0.onnx efficientnet_b4.onnx efficientnetv2_rw_s.onnx efficientnetv2_rw_t.onnx mixer_b16_224.onnx mobilenetv2_050.onnx mobilenetv2_100.onnx resnet34.onnx resnet50.onnx shufflenet_v2_x0_5.onnx shufflenet_v2_x1_0.onnx shufflenet_v2_x1_0-mod.onnx'

set -x
bs=32
for m in $MODELS; do
    echo $m

    trt_cache_pre
    python -u ${BASE_DIR}/main.py -m ${MODEL_DIR}/${m} -s model -b ${bs} -o fp16 -D 32,16 -v -f ${WORK_DIR}/${m}-fp16.json 2>&1 | tee ${WORK_DIR}/${m}-fp16.log
    trt_cache_post

done




echo "=== PRoof done, generate graph ==="
python ${BASE_DIR}/dataviewer/read_value.py 'model.name; model.bench.results.*.time_avg; model.bench.results.*.flops_avg; model.bench.results.*.memory_avg' ${WORK_DIR}/*.json > ${WORK_DIR}/e2e.csv
python ${BASE_DIR}/test/helper_fig2_single.py 7 ${WORK_DIR}/e2e.csv ${RESULT_DIR}/figure2_subplot

echo "=== figure/table saved to ${RESULT_DIR} ==="
