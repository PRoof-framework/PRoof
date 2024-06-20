# PRoof
PRoof is designed to be a universal, end-to-end, fine-grained AI model inference performance analysis and hardware AI benchmark tool.

It will take a `.onnx` model file as input, and produce a detailed report about the end-to-end profile and layer profile, include the Effective FLOPS and Memory Bandwidth, including a roofline chart for each layer in the model.


# build
```sh
pip install -r requirements.txt
```

### Optional: NART (deprecated)
Put the `NART` repo (should contain libart.so) aside this repo (or change path in `model/backend/nart/Makefile`), then:
```sh
cd model/backend/nart/Makefile
make
```

# usage
Example:
```sh
python main.py -h
```

## trtexec backend (default)
This backend is using nvidia tensorrt, you should install **tensorrt** by yourself, which will make trtexec command available.

Support model end-to-end profile and layer profile, in layer profile also support using nsight compute(ncu) by `-o use_ncu,ncu_bin=/path/to/ncu`

For now, when using 'trtexec' backend, the model (xxx.onnx) should has dynamic batch size.

e.g. `{'input': ['batch_size', 3, 224, 224]}`, which you should set `dynamic_axes` when using `torch.onnx.export()`

```sh
python main.py -B trtexec -m resnet50.onnx -b 1-1024 -f report.json -v
python main.py -B trtexec -m resnet50.onnx -b 1-1024 -f report.json -D 32,16 -o fp16 -v
```

The `report.json` will store all data from the analyze and benchmark, to view it, use dataviewer tool to generate HTML report
```
python dataviewer/main.py report.json <output_dir>
```

### NCU mode for layer profile
By default, in layer profile, the tool will get layer lantency from trtexec and use the FLOPs and Memory Access Amount info from ONNX model analyze (model.analyze), even it could correctly consider the memory access reduced by layer fusion, it's still a approximate value, but will work on all platforms.

On nvidia platform, the Nsight Compute (ncu) tool is able to do the kernel level profiling to get the hardware counted FLOPs and Memory Access Amount info, and this backend support using ncu instead of info from model analyze. However, the ncu will be 10x or more slower than without it.

You should install **nvidia-dlprof and NCU** to use this mode. You may install dlprof with pip and `dlprof` command should findable in the PATH. Then you can use `use_ncu` flag in backend option to active this mode, and use `ncu_bin` option to set the ncu program binary path if it's not at /usr/local/NVIDIA-Nsight-Compute/ncu.
```
python -u main.py -B trtexec -m test/model/resnet50.onnx -D 32,16 -o fp16,use_ncu,ncu_bin=/opt/nvidia/nsight-compute/2023.1.0/ncu -b 128 -v -f resnet50-128-fp16-ncu.json
```

## onnxruntime or openvino backend
Install onnxruntime or openvino via pip is recommended. Use `-B onnxruntime` or `-B openvino` to use them with PRoof, use `-o help` to get more details for each backend.

## NART backend (deprecated)
WIP, Only support end-to-end profile for now.
The 'nart_trt' backend support both dynamic batch size and fixed 1 batch size (like `[1, 3, 224, 224]`)
```sh
python main.py -B nart_trt -m resnet50.onnx -b 1-1024 -v
python main.py -B nart_trt -m resnet50.onnx -b 1-1024 -v -D 32,16 -o config=test/model/narttrt-config-fp16.json
```

## env flags
Some addition options for PRoof, which should not been used in most situation. env flags could been used like this:
```sh
PROOF_TMPDIR=/some/otherwhere/tmp PROOF_BACKEND_TRTEXEC_NCU_SAVE_NCU_REP=1 python main.py ...
```

tweaks:
- `PROOF_TMPDIR`:
default is [system tmpdir]/proof (e.g. /tmp/proof)
- `PROOF_E2E_NOT_USE_LAYER_DATA`:
When model layer profiling is enable, the layer's total FLOPs and memory access amount info will also used for end-to-end profling, to achieve more accurate info for end-to-end considering optimization like layer fuse used in backend. But if layer profiling is not reliable, turn on this flag for a fallback total FLOPs and memory access amount.
- `PROOF_BACKEND_TRTEXEC_NO_PROFILING_VERBOSITY`:
Use the old version layer_prof() based on layer name lookup, for the old version of TensorRT which not support `--profilingVerbosity=detailed` option in `trtexec`. This mode not support 'Myelin' layer in TensorRT.
- `PROOF_BACKEND_TRTEXEC_NCU_USE_TIMES_FROM_NCU`:
Use kernel times from ncu report to calculate layer latency, instead of trtexec's layer-profile,
it will be longer and we think it is unaccurate, (however, FLOPs (calculate by ins. count) and memory access amount is accurate), so is off by default
- `PROOF_BACKEND_TRTEXEC_NCU_SAVE_NCU_REP`:
Save the Nsight Compute's .ncu-rep repot file to current directory for other propose
- `PROOF_BACKEND_TRTEXEC_NCU_FULL_SET`:
Collect full set as well as all roofline section in NCU for other propose, slower (it will use `--set=full --section=SpeedOfLight_HierarchicalDoubleRooflineChart --section=SpeedOfLight_HierarchicalHalfRooflineChart --section=SpeedOfLight_HierarchicalSingleRooflineChart --section=SpeedOfLight_HierarchicalTensorRooflineChart` instead of `--set=roofline` in ncu args))


for dev only, do not use:
- `PROOF_BACKEND_NART_SKIP_CONVERT`
- `PROOF_BACKEND_TRTEXEC_SKIP_CONVERT`
- `PROOF_BACKEND_TRTEXEC_SKIP_NCU_RUN`
- `PROOF_BACKEND_TRTEXEC_NCU_IGNORE_FALLBACK`
