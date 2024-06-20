#!/bin/bash

# meta type: CPU x86

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


### normal models
MODELS='efficientnet_b0.onnx efficientnet_b4.onnx efficientnetv2_rw_s.onnx efficientnetv2_rw_t.onnx mixer_b16_224.onnx mobilenetv2_050.onnx mobilenetv2_100.onnx resnet34.onnx resnet50.onnx shufflenet_v2_x0_5.onnx shufflenet_v2_x1_0.onnx shufflenet_v2_x1_0-mod.onnx'

set -x
bs=4
for m in $MODELS; do
    echo $m
    python -u ${BASE_DIR}/main.py -B onnxruntime -m ${MODEL_DIR}/${m} -s model -b ${bs} -o providers=CPUExecutionProvider -D 32,32 -v -f ${WORK_DIR}/${m}.json 2>&1 | tee ${WORK_DIR}/${m}.log

done



echo "=== PRoof done, generate graph ==="
python ${BASE_DIR}/dataviewer/read_value.py 'model.name; model.bench.results.*.time_avg; model.bench.results.*.flops_avg; model.bench.results.*.memory_avg' ${WORK_DIR}/*.json > ${WORK_DIR}/e2e.csv
python ${BASE_DIR}/test/helper_fig2_single.py 10 ${WORK_DIR}/e2e.csv ${RESULT_DIR}/figure2_subplot

echo "=== figure/table saved to ${RESULT_DIR} ==="
