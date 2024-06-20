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
MODELS='efficientnet_b0.onnx efficientnet_b4.onnx efficientnetv2_rw_s.onnx efficientnetv2_rw_t.onnx mixer_b16_224.onnx mobilenetv2_050.onnx mobilenetv2_100.onnx resnet34.onnx resnet50.onnx shufflenet_v2_x0_5.onnx shufflenet_v2_x1_0.onnx shufflenet_v2_x1_0-mod.onnx swin_base_patch4_window7_224.onnx swin_small_patch4_window7_224.onnx swin_tiny_patch4_window7_224.onnx vit_base_patch16_224.onnx vit_small_patch16_224.onnx vit_tiny_patch16_224.onnx'

set -x
bs=16
for m in $MODELS; do
    echo $m

    python -u ${BASE_DIR}/main.py -B onnxruntime -m ${MODEL_DIR}/${m} -s model -b ${bs} -o providers=CPUExecutionProvider -D 32,32 -v -f ${WORK_DIR}/${m}.json 2>&1 | tee ${WORK_DIR}/${m}.log

done


# models with dynamic input shape, need addition config

m="distilbert-base-uncased-finetuned-sst-2-english.onnx"
bs=16
python -u ${BASE_DIR}/main.py -B onnxruntime -m ${MODEL_DIR}/${m} -s model -b ${bs} -o providers=CPUExecutionProvider -D 32,32 -v --inputs-shape-override='{"input_ids": [null, 512], "attention_mask": [null, 512]}' -f ${WORK_DIR}/${m}.json 2>&1 | tee ${WORK_DIR}/${m}.log



echo "=== PRoof done, generate graph ==="
python ${BASE_DIR}/dataviewer/read_value.py 'model.name; model.bench.results.*.time_avg; model.bench.results.*.flops_avg; model.bench.results.*.memory_avg' ${WORK_DIR}/*.json > ${WORK_DIR}/e2e.csv
python ${BASE_DIR}/test/helper_fig2_single.py 9 ${WORK_DIR}/e2e.csv ${RESULT_DIR}/figure2_subplot

echo "=== figure/table saved to ${RESULT_DIR} ==="
