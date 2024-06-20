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
MODELS='efficientnet_b0.onnx efficientnet_b4.onnx efficientnetv2_rw_s.onnx efficientnetv2_rw_t.onnx mixer_b16_224.onnx mobilenetv2_050.onnx mobilenetv2_100.onnx resnet34.onnx resnet50.onnx shufflenet_v2_x0_5.onnx shufflenet_v2_x1_0.onnx shufflenet_v2_x1_0-mod.onnx swin_base_patch4_window7_224.onnx swin_small_patch4_window7_224.onnx swin_tiny_patch4_window7_224.onnx vit_base_patch16_224.onnx vit_small_patch16_224.onnx vit_tiny_patch16_224.onnx'

set -x
bs=128
for m in $MODELS; do
    echo $m

    # Using Nsight Compute
    trt_cache_pre
    python -u ${BASE_DIR}/main.py -m ${MODEL_DIR}/${m} -s model -b ${bs} -o int8,use_ncu -D 32,8 -v -f ${WORK_DIR}/${m}-int8.json 2>&1 | tee ${WORK_DIR}/${m}-int8.log
    trt_cache_post

done


# models with dynamic input shape, need addition config

m="distilbert-base-uncased-finetuned-sst-2-english.onnx"
bs=128
trt_cache_pre
python -u ${BASE_DIR}/main.py -m ${MODEL_DIR}/${m} -s model -b ${bs} -o use_ncu,int8 -D 32,8 -v --inputs-shape-override='{"input_ids": [null, 512], "attention_mask": [null, 512]}' -f ${WORK_DIR}/${m}-int8.json 2>&1 | tee ${WORK_DIR}/${m}-int8.log
trt_cache_post



echo "=== PRoof done, generate graph ==="
python ${BASE_DIR}/dataviewer/read_value.py 'model.name; model.bench.results.*.time_avg; model.bench.results.*.flops_avg; model.bench.results.*.memory_avg' ${WORK_DIR}/*.json > ${WORK_DIR}/e2e.csv
python ${BASE_DIR}/test/helper_fig2_single.py 4 ${WORK_DIR}/e2e.csv ${RESULT_DIR}/figure2_subplot

echo "=== figure/table saved to ${RESULT_DIR} ==="
