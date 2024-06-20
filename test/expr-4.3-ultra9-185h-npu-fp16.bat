@echo off

REM meta type: NPU

setlocal enabledelayedexpansion

set filename=%~nx0
set EXPR_NAME=%~n0
set BASE_DIR=%cd%\..
set RESULT_DIR=%cd%\result_%EXPR_NAME%
set WORK_DIR=%RESULT_DIR%\workdir

if not exist %WORK_DIR% (
    mkdir %WORK_DIR%
)

if "%MODEL_DIR%"=="" (
    set MODEL_DIR=%cd%\model
)

echo EXPR_NAME: %EXPR_NAME%
echo BASE_DIR: %MODEL_DIR%
echo RESULT_DIR: %RESULT_DIR%
echo WORK_DIR: %WORK_DIR%
echo MODEL_DIR: %MODEL_DIR%

REM normal models
set MODELS=efficientnet_b0.onnx efficientnet_b4.onnx efficientnetv2_rw_s.onnx efficientnetv2_rw_t.onnx mobilenetv2_050.onnx mobilenetv2_100.onnx resnet34.onnx resnet50.onnx shufflenet_v2_x0_5.onnx shufflenet_v2_x1_0.onnx

set bs=1
for %%m in (%MODELS%) do (
    echo %%m
    python -u %BASE_DIR%\main.py -B openvino -o device=NPU -m %MODEL_DIR%\%%m -s model -b %bs% -D 32,16 -v -f %WORK_DIR%\%%m.json 2>&1 > %WORK_DIR%\%%m.log
)

echo === PRoof done, generate graph ===
set json_files=
for %%f in (%WORK_DIR%\*.json) do (
    set json_files=!json_files! %%f
)
python %BASE_DIR%\dataviewer\read_value.py "model.name; model.bench.results.*.time_avg; model.bench.results.*.flops_avg; model.bench.results.*.memory_avg" !json_files! > %WORK_DIR%\e2e.csv
python %BASE_DIR%\test\helper_fig2_single.py 11 %WORK_DIR%\e2e.csv %RESULT_DIR%\figure2_subplot

echo === figure/table saved to %RESULT_DIR% ===
